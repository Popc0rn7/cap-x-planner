from __future__ import annotations

from typing import Any

import pytest

from capx.envs.trial_fine_grained import (
    FineGrainedTrialComponents,
    FineGrainedTrialConfig,
    run_fine_grained_trial,
)
from capx.memory import HierarchicalTrialMemory
from capx.planning.agentic_builder import AgenticStagePlanBuilder
from capx.planning.primitives import ToolCall, ToolResult, libero_p0_primitive_catalog
from capx.planning.recovery import RuleBasedRecoveryPolicy
from capx.planning.stage_planner import StagePlanner
from capx.planning.stage_reward import RewardStatus, StageReward


def _stage(stage_id: str, prompt: str, target: str) -> dict[str, Any]:
    return {
        "id": stage_id,
        "objective": prompt,
        "primary_call": {
            "action": "VLA",
            "target": target,
            "params": {"prompt": prompt},
            "reward_rules": [
                {
                    "predicate": f"{stage_id}_complete",
                    "description": f"{stage_id} is complete",
                }
            ],
        },
        "retry_budget": 1,
        "restage_budget": 1,
    }


class FakeStageAgent:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if request.failed_stage is not None:
            return {"stages": []}
        return {
            "stages": [
                _stage("pick", "pick up the alphabet soup", "alphabet soup"),
                _stage("place", "place the alphabet soup in the basket", "basket"),
            ]
        }


class FakeLiberoEnv:
    def __init__(self) -> None:
        self.completed = False
        self.holding = None

    def reset(self, *, options, seed):  # noqa: ARG002
        self.completed = False
        self.holding = None
        return self.get_observation(), {
            "task_prompt": "pick up the alphabet soup and place it in the basket"
        }

    def get_observation(self):
        return {
            "objects": {"alphabet soup": {"visible": True}, "basket": {"visible": True}},
            "holding": self.holding,
            "gripper_state": "closed" if self.holding else "open",
        }

    def task_completed(self):
        return self.completed

    def compute_reward(self):
        return float(self.completed)


class ScenarioExecutor:
    def __init__(self, env: FakeLiberoEnv, pick_failures: list[str] | None = None) -> None:
        self.env = env
        self.pick_failures = list(pick_failures or [])
        self.actions = []

    def execute(self, call: ToolCall, state) -> ToolResult:  # noqa: ARG002
        self.actions.append(call.action)
        if call.action == "SET_GRIPPER":
            self.env.holding = None
            return ToolResult(True, "gripper_open")
        if call.target == "alphabet soup":
            if self.pick_failures:
                return ToolResult(False, self.pick_failures.pop(0))
            self.env.holding = "alphabet soup"
            return ToolResult(True, "picked", updated_state={"holding": "alphabet soup"})
        self.env.holding = None
        self.env.completed = True
        return ToolResult(True, "placed", updated_state={"holding": None})


class ResultRewardVerifier:
    def compute_reward(self, context):
        if context.result.code == "collision":
            status = RewardStatus.UNSAFE
        else:
            status = RewardStatus.SUCCESS if context.result.ok else RewardStatus.FAILURE
        return StageReward(status, context.result.code)


def _run(pick_failures: list[str] | None = None):
    env = FakeLiberoEnv()
    agent = FakeStageAgent()
    builder = AgenticStagePlanBuilder(
        agent,
        libero_p0_primitive_catalog(),
        reward_tools=("read_gripper_state",),
    )
    executor = ScenarioExecutor(env, pick_failures)
    summary = run_fine_grained_trial(
        env,
        0,
        FineGrainedTrialComponents(
            StagePlanner(builder),
            executor,
            ResultRewardVerifier(),
            RuleBasedRecoveryPolicy(),
            HierarchicalTrialMemory(),
        ),
        config=FineGrainedTrialConfig(max_turns=8, max_replans=2),
    )
    return summary, executor, agent


@pytest.mark.parametrize(
    "pick_failures,expected_actions",
    [
        (None, ["VLA", "VLA"]),
        (["empty_grasp"], ["VLA", "SET_GRIPPER", "VLA", "VLA"]),
    ],
)
def test_agentic_pick_and_place_success_and_empty_grasp_recovery(
    pick_failures, expected_actions
) -> None:
    summary, executor, _ = _run(pick_failures)
    assert summary.task_completed
    assert executor.actions == expected_actions


def test_agentic_retry_exhaustion_replans_to_stop() -> None:
    summary, executor, agent = _run(["controller_failed", "controller_failed"])
    assert not summary.task_completed
    assert executor.actions == ["VLA", "VLA"]
    assert agent.requests[-1].failed_stage.id == "pick"
    assert summary.stage_records[0]["status"] == "failed"


def test_agentic_unsafe_abort() -> None:
    summary, executor, _ = _run(["collision"])
    assert not summary.task_completed
    assert executor.actions == ["VLA"]
    assert summary.stage_records[0]["status"] == "aborted"
