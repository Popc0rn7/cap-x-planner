from __future__ import annotations

from typing import Any

import pytest

from capx.planning.agentic_builder import (
    AgenticStagePlanBuilder,
    ModelStagePlanAgent,
    StagePlanBuildError,
)
from capx.planning.primitives import libero_p0_primitive_catalog
from capx.planning.stage_planner import Stage
from capx.planning.stage_reward import RewardStatus, StageReward


def _stage(stage_id: str = "pick") -> dict[str, Any]:
    return {
        "id": stage_id,
        "objective": "pick up the alphabet soup",
        "primary_call": {
            "action": "VLA",
            "target": "alphabet soup",
            "params": {"prompt": "pick up the alphabet soup"},
            "reward_rules": [
                {
                    "predicate": "object_held",
                    "description": "alphabet soup is held",
                    "required_evidence": ["holding"],
                }
            ],
        },
        "retry_budget": 1,
        "restage_budget": 1,
    }


class SequenceAgent:
    def __init__(self, *outputs: Any) -> None:
        self.outputs = list(outputs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.outputs.pop(0)


def _builder(agent: SequenceAgent) -> AgenticStagePlanBuilder:
    return AgenticStagePlanBuilder(
        agent,
        libero_p0_primitive_catalog(),
        reward_tools=("read_gripper_state",),
    )


def test_builds_valid_agent_json_and_repairs_invalid_json_once() -> None:
    agent = SequenceAgent("not json", {"stages": [_stage()]})

    plan = _builder(agent).build("pick soup", {"objects": {"soup": {"visible": True}}})

    assert [stage.id for stage in plan.stages] == ["pick"]
    assert agent.requests[1].validation_errors


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda stage: stage["primary_call"].update(action="FLY"), "unknown primitive"),
        (lambda stage: stage["primary_call"].update(reward_rules=[]), "reward rule"),
        (
            lambda stage: stage["primary_call"]["reward_rules"][0].update(
                allowed_tools=["oracle"]
            ),
            "unregistered reward tools",
        ),
        (lambda stage: stage["primary_call"].update(target=[0.1, 0.2, 0.3]), "coordinate"),
        (lambda stage: stage.update(retry_budget=2), "retry_budget"),
    ],
)
def test_rejects_invalid_agent_capabilities(mutate, match) -> None:
    stage = _stage()
    mutate(stage)
    agent = SequenceAgent({"stages": [stage]}, {"stages": [stage]})

    with pytest.raises(StagePlanBuildError, match=match):
        _builder(agent).build("pick soup", {})


def test_rejects_duplicate_ids_and_more_than_eight_stages() -> None:
    duplicate = {"stages": [_stage(), _stage()]}
    oversized = {"stages": [_stage(str(index)) for index in range(9)]}
    for payload, match in ((duplicate, "unique"), (oversized, "maximum")):
        with pytest.raises(StagePlanBuildError, match=match):
            _builder(SequenceAgent(payload, payload)).build("task", {})


def test_prompt_contains_symbolic_memory_but_excludes_pixels_coordinates_and_privileged() -> None:
    request_agent = SequenceAgent({"stages": [_stage()]})
    builder = _builder(request_agent)
    state = {
        "objects": {"soup": {"visible": True, "position": [1, 2, 3]}},
        "holding": None,
        "task_memory": {"strategy": "pick then place"},
        "global_memory": {"tip": "verify grasp"},
        "observation": {"rgb": [[1]], "privileged_state": {"pose": [1, 2, 3]}},
    }
    builder.build("pick soup", state)
    prompt = ModelStagePlanAgent.build_prompt(request_agent.requests[0])

    assert "pick soup" in prompt
    assert "pick then place" in prompt
    assert "verify grasp" in prompt
    assert "read_gripper_state" in prompt
    assert "rgb" not in prompt
    assert "privileged_state" not in prompt
    assert "position" not in prompt


def test_rejects_unknown_schema_fields() -> None:
    stage = _stage()
    stage["primary_call"]["python"] = "open_gripper()"
    payload = {"stages": [stage]}

    with pytest.raises(StagePlanBuildError, match="unknown fields"):
        _builder(SequenceAgent(payload, payload)).build("task", {})


def test_replan_keeps_completed_prefix_out_of_replacement_and_allows_stop() -> None:
    completed_payload = _stage("completed")
    completed = _builder(SequenceAgent({"stages": [completed_payload]})).build("task", {}).stages[0]
    failed = Stage("failed", "failed", completed.primary_call)
    reward = StageReward(RewardStatus.FAILURE, "empty_grasp")
    agent = SequenceAgent({"stages": []})

    replacement = _builder(agent).replan("task", {}, (completed,), failed, reward)

    assert replacement == ()
    assert agent.requests[0].completed_stages == (completed,)


def test_invalid_replan_does_not_mutate_stage_planner_plan() -> None:
    completed_payload = _stage("completed")
    agent = SequenceAgent(
        {"stages": [completed_payload]},
        {"stages": [completed_payload]},
        {"stages": [completed_payload]},
    )
    builder = _builder(agent)
    completed = builder.build("task", {}).stages[0]
    failed = Stage("failed", "failed", completed.primary_call)

    with pytest.raises(StagePlanBuildError, match="repeats completed"):
        builder.replan(
            "task", {}, (completed,), failed, StageReward(RewardStatus.FAILURE, "failed")
        )
