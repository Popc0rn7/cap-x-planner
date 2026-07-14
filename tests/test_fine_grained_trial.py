from __future__ import annotations

import json
from typing import Any

from capx.envs.trial_fine_grained import (
    FineGrainedTrialComponents,
    FineGrainedTrialConfig,
    RecoveryAction,
    RecoveryDecision,
    ToolCall,
    ToolResult,
    run_fine_grained_trial,
)
from capx.planning.stage_planner import Stage, TurnContext, TurnRole
from capx.planning.stage_reward import RewardStatus, StageReward


class FakeEnv:
    def __init__(self) -> None:
        self.completed = False
        self.observation_count = 0

    def reset(self, *, seed: int, options: dict[str, Any]):
        self.completed = False
        self.observation_count = 0
        return self.get_observation(), {"task_prompt": "grasp the cup"}

    def get_observation(self) -> dict[str, Any]:
        self.observation_count += 1
        return {
            "observation_count": self.observation_count,
            "completed": self.completed,
        }

    def task_completed(self) -> bool:
        return self.completed

    def compute_reward(self) -> float:
        return float(self.completed)


class FakePlanner:
    def __init__(self, *stages: Stage) -> None:
        self.stages = list(stages)
        self.index = 0
        self.initialized_with: tuple[str, dict[str, Any]] | None = None
        self.results: list[tuple[str, TurnRole, RewardStatus]] = []
        self.replans = 0

    def initialize(self, task: str, state: dict[str, Any]) -> None:
        self.initialized_with = (task, state)

    def next_stage(self, task: str, state: dict[str, Any]) -> Stage | None:
        return self.stages[self.index] if self.index < len(self.stages) else None

    def observe_turn(self, stage, role, result, verdict) -> None:
        self.results.append((stage.id, role, verdict.status))
        if role is not TurnRole.RESTAGE and verdict.status is RewardStatus.SUCCESS:
            self.index += 1

    def replan(self, task, state, failed_stage, verdict) -> None:
        self.replans += 1
        self.index += 1


class FakeMemory:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.observed_after: list[str] = []
        self.rewards: list[RewardStatus] = []
        self.contexts: list[TurnContext] = []
        self.recoveries: list[RecoveryAction] = []
        self.finalized: bool | None = None

    def bootstrap(self, task: str, observation: dict[str, Any]) -> dict[str, Any]:
        self.state = {"task": task, "observation": observation}
        return self.state

    def update_after_execution(self, context, call, result, observation) -> dict[str, Any]:
        self.contexts.append(context)
        self.observed_after.append(call.action)
        self.state = {
            **self.state,
            **result.updated_state,
            "observation": observation,
        }
        return self.state

    def record_stage_reward(self, context, call, result, reward, state) -> None:
        self.rewards.append(reward.status)

    def record_recovery(self, context, decision, state) -> None:
        self.recoveries.append(decision.action)

    def finalize(self, success: bool) -> None:
        self.finalized = success


class ResultVerifier:
    def compute_reward(self, context) -> StageReward:
        return StageReward(
            status=RewardStatus.SUCCESS if context.result.ok else RewardStatus.FAILURE,
            code="reward_met" if context.result.ok else context.result.code,
        )


class AbortRecovery:
    def decide(self, call, state, verdict, retry_count) -> RecoveryDecision:
        return RecoveryDecision(action=RecoveryAction.ABORT)


class CompletingExecutor:
    def __init__(self, env: FakeEnv) -> None:
        self.env = env
        self.actions: list[str] = []

    def execute(self, call: ToolCall, state: dict[str, Any]) -> ToolResult:
        self.actions.append(call.action)
        self.env.completed = True
        return ToolResult(ok=True, code="done", updated_state={"holding": "cup"})


def _events(summary) -> list[dict[str, Any]]:
    return [json.loads(line) for line in summary.log.splitlines()]


def test_fine_grained_trial_observes_and_verifies_after_one_tool_call() -> None:
    env = FakeEnv()
    call = ToolCall(action="VLA", target="cup")
    planner = FakePlanner(Stage("grasp", "grasp cup", call))
    memory = FakeMemory()
    executor = CompletingExecutor(env)

    summary = run_fine_grained_trial(
        env,
        trial=3,
        components=FineGrainedTrialComponents(
            planner=planner,
            tool_executor=executor,
            verifier=ResultVerifier(),
            recovery_policy=AbortRecovery(),
            memory=memory,
        ),
    )

    assert summary.success is True
    assert summary.task_completed is True
    assert summary.num_code_blocks == 1
    assert executor.actions == ["VLA"]
    assert memory.observed_after == ["VLA"]
    assert memory.rewards == [RewardStatus.SUCCESS]
    assert memory.finalized is True
    assert memory.contexts == [TurnContext(0, "grasp", TurnRole.PRIMARY)]
    assert summary.num_stages == 1
    assert summary.num_stages_completed == 1
    assert summary.stage_records[0]["status"] == "success"
    assert [event["event"] for event in _events(summary)] == [
        "stage_started",
        "tool_call",
        "stage_reward",
        "stage_completed",
        "trial_finished",
    ]


class RestagingExecutor:
    def __init__(self, env: FakeEnv) -> None:
        self.env = env
        self.actions: list[str] = []
        self.grasp_attempts = 0

    def execute(self, call: ToolCall, state: dict[str, Any]) -> ToolResult:
        self.actions.append(call.action)
        if call.action == "MOVE_EEF":
            return ToolResult(ok=True, code="staged")
        self.grasp_attempts += 1
        if self.grasp_attempts == 1:
            return ToolResult(ok=False, code="empty_grasp")
        self.env.completed = True
        return ToolResult(ok=True, code="grasped", updated_state={"holding": "cup"})


class RestageEmptyGrasp:
    def decide(self, call, state, verdict, retry_count) -> RecoveryDecision:
        assert verdict.code == "empty_grasp"
        return RecoveryDecision(
            action=RecoveryAction.RESTAGE,
            tool_calls=(
                ToolCall(
                    action="MOVE_EEF",
                    target="alternate_pregrasp(cup)",
                ),
            ),
            retry_failed_call=True,
        )


def test_fine_grained_trial_runs_restage_calls_through_same_closed_loop() -> None:
    env = FakeEnv()
    call = ToolCall(action="VLA", target="cup")
    planner = FakePlanner(
        Stage("grasp", "grasp cup", call, restage_budget=1)
    )
    memory = FakeMemory()
    executor = RestagingExecutor(env)

    summary = run_fine_grained_trial(
        env,
        trial=1,
        components=FineGrainedTrialComponents(
            planner=planner,
            tool_executor=executor,
            verifier=ResultVerifier(),
            recovery_policy=RestageEmptyGrasp(),
            memory=memory,
        ),
        config=FineGrainedTrialConfig(max_turns=5),
    )

    assert summary.task_completed is True
    assert executor.actions == ["VLA", "MOVE_EEF", "VLA"]
    assert memory.observed_after == ["VLA", "MOVE_EEF", "VLA"]
    assert memory.rewards == [
        RewardStatus.FAILURE,
        RewardStatus.SUCCESS,
        RewardStatus.SUCCESS,
    ]
    assert any(event["event"] == "recovery" for event in _events(summary))
    assert [context.role for context in memory.contexts] == [
        TurnRole.PRIMARY,
        TurnRole.RESTAGE,
        TurnRole.RESTAGE_RETRY,
    ]
    assert {context.stage_id for context in memory.contexts} == {"grasp"}
    assert summary.stage_records[0]["restages_used"] == 1
    assert memory.recoveries == [RecoveryAction.RESTAGE]


class SequenceExecutor:
    def __init__(self, env: FakeEnv, outcomes: list[tuple[bool, str]]) -> None:
        self.env = env
        self.outcomes = outcomes
        self.actions: list[str] = []

    def execute(self, call, state) -> ToolResult:
        self.actions.append(call.action)
        ok, code = self.outcomes.pop(0)
        if ok and not self.outcomes:
            self.env.completed = True
        return ToolResult(ok=ok, code=code)


class RetryRecovery:
    def decide(self, call, state, verdict, retry_count) -> RecoveryDecision:
        return RecoveryDecision(action=RecoveryAction.RETRY)


def test_multiple_stages_emit_lifecycle_events_and_summary_records() -> None:
    env = FakeEnv()
    planner = FakePlanner(
        Stage("approach", "approach cup", ToolCall(action="MOVE_EEF")),
        Stage("grasp", "grasp cup", ToolCall(action="VLA")),
    )
    executor = SequenceExecutor(env, [(True, "approached"), (True, "grasped")])

    summary = run_fine_grained_trial(
        env,
        1,
        FineGrainedTrialComponents(
            planner, executor, ResultVerifier(), AbortRecovery(), FakeMemory()
        ),
    )

    assert executor.actions == ["MOVE_EEF", "VLA"]
    assert summary.num_stages == 2
    assert summary.num_stages_completed == 2
    assert [record["stage_id"] for record in summary.stage_records] == [
        "approach",
        "grasp",
    ]
    assert [event["event"] for event in _events(summary)].count("stage_started") == 2
    assert [event["event"] for event in _events(summary)].count("stage_completed") == 2


def test_retry_stays_in_stage_and_consumes_only_retry_budget() -> None:
    env = FakeEnv()
    stage = Stage(
        "grasp",
        "grasp cup",
        ToolCall(action="VLA"),
        retry_budget=1,
        restage_budget=2,
    )
    memory = FakeMemory()
    executor = SequenceExecutor(env, [(False, "miss"), (True, "grasped")])

    summary = run_fine_grained_trial(
        env,
        1,
        FineGrainedTrialComponents(
            FakePlanner(stage), executor, ResultVerifier(), RetryRecovery(), memory
        ),
    )

    assert [context.role for context in memory.contexts] == [
        TurnRole.PRIMARY,
        TurnRole.RETRY,
    ]
    assert summary.stage_records[0]["retries_used"] == 1
    assert summary.stage_records[0]["restages_used"] == 0


def test_retry_budget_exhaustion_fails_stage_and_replans() -> None:
    env = FakeEnv()
    planner = FakePlanner(
        Stage("grasp", "grasp cup", ToolCall(action="VLA"), retry_budget=1)
    )
    executor = SequenceExecutor(env, [(False, "miss"), (False, "miss")])

    summary = run_fine_grained_trial(
        env,
        1,
        FineGrainedTrialComponents(
            planner, executor, ResultVerifier(), RetryRecovery(), FakeMemory()
        ),
        config=FineGrainedTrialConfig(max_turns=5),
    )

    assert planner.replans == 1
    assert summary.stage_records[0]["status"] == "failed"
    assert any(event["event"] == "retry_budget_exhausted" for event in _events(summary))


class TwoRepairRestage:
    def decide(self, call, state, verdict, retry_count) -> RecoveryDecision:
        return RecoveryDecision(
            RecoveryAction.RESTAGE,
            (ToolCall(action="REPAIR_ONE"), ToolCall(action="REPAIR_TWO")),
        )


def test_failed_repair_cancels_remaining_repairs_and_replans() -> None:
    env = FakeEnv()
    planner = FakePlanner(
        Stage(
            "grasp",
            "grasp cup",
            ToolCall(action="VLA"),
            restage_budget=1,
        )
    )
    executor = SequenceExecutor(env, [(False, "miss"), (False, "blocked")])

    summary = run_fine_grained_trial(
        env,
        1,
        FineGrainedTrialComponents(
            planner, executor, ResultVerifier(), TwoRepairRestage(), FakeMemory()
        ),
        config=FineGrainedTrialConfig(max_turns=5),
    )

    assert executor.actions == ["VLA", "REPAIR_ONE"]
    assert planner.replans == 1
    assert summary.stage_records[0]["status"] == "failed"


class UnsafeVerifier:
    def compute_reward(self, context) -> StageReward:
        status = (
            RewardStatus.UNSAFE
            if context.result.code == "collision"
            else RewardStatus.FAILURE
        )
        return StageReward(status=status, code=context.result.code)


def test_unsafe_repair_aborts_stage_and_trial_immediately() -> None:
    env = FakeEnv()
    planner = FakePlanner(
        Stage(
            "grasp",
            "grasp cup",
            ToolCall(action="VLA"),
            restage_budget=1,
        )
    )
    executor = SequenceExecutor(env, [(False, "miss"), (False, "collision")])

    summary = run_fine_grained_trial(
        env,
        1,
        FineGrainedTrialComponents(
            planner, executor, UnsafeVerifier(), TwoRepairRestage(), FakeMemory()
        ),
        config=FineGrainedTrialConfig(max_turns=5),
    )

    assert executor.actions == ["VLA", "REPAIR_ONE"]
    assert planner.replans == 0
    assert summary.stage_records[0]["status"] == "aborted"
    assert any(event["event"] == "stage_aborted" for event in _events(summary))
