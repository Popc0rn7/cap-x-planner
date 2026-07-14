from __future__ import annotations

import pytest

from capx.envs.trial_fine_grained import ToolCall, ToolResult
from capx.planning.stage_planner import (
    Stage,
    StagePlan,
    StagePlanner,
    StageStatus,
    TurnRole,
)
from capx.planning.stage_reward import RewardStatus, StageReward


def _stage(stage_id: str, **budgets: int) -> Stage:
    return Stage(
        id=stage_id,
        objective=f"run {stage_id}",
        primary_call=ToolCall(action=stage_id.upper()),
        **budgets,
    )


def _reward(status: RewardStatus) -> StageReward:
    return StageReward(status=status, code=status.value)


class StaticBuilder:
    def __init__(self, stages: tuple[Stage, ...], replacement: tuple[Stage, ...] = ()) -> None:
        self.stages = stages
        self.replacement = replacement
        self.replan_args = None

    def build(self, task, state) -> StagePlan:
        return StagePlan(task=task, stages=self.stages)

    def replan(self, task, state, completed_stages, failed_stage, verdict):
        self.replan_args = (completed_stages, failed_stage, verdict)
        return self.replacement


@pytest.mark.parametrize(
    "stages,match",
    [
        ((_stage("same"), _stage("same")), "Duplicate Stage.id"),
        ((_stage("bad", retry_budget=-1),), "retry_budget"),
        ((_stage("bad", restage_budget=-1),), "restage_budget"),
    ],
)
def test_initialize_rejects_invalid_stage_plans(stages, match) -> None:
    planner = StagePlanner(StaticBuilder(stages))

    with pytest.raises(ValueError, match=match):
        planner.initialize("task", {})


def test_planner_returns_stages_in_order_and_only_advances_on_primary_success() -> None:
    first, second = _stage("first"), _stage("second")
    planner = StagePlanner(StaticBuilder((first, second)))
    planner.initialize("task", {})

    assert planner.next_stage("task", {}) is first
    planner.observe_turn(
        first,
        TurnRole.PRIMARY,
        ToolResult(ok=False, code="failed"),
        _reward(RewardStatus.FAILURE),
    )
    assert planner.next_stage("task", {}) is first
    assert planner.progress is not None
    assert planner.progress.status is StageStatus.RUNNING

    planner.observe_turn(
        first,
        TurnRole.PRIMARY,
        ToolResult(ok=True, code="done"),
        _reward(RewardStatus.SUCCESS),
    )
    assert planner.next_stage("task", {}) is second


def test_successful_repair_does_not_complete_stage() -> None:
    stage = _stage("grasp")
    planner = StagePlanner(StaticBuilder((stage,)))
    planner.initialize("task", {})

    planner.observe_turn(
        stage,
        TurnRole.RESTAGE,
        ToolResult(ok=True, code="staged"),
        _reward(RewardStatus.SUCCESS),
    )

    assert planner.next_stage("task", {}) is stage
    assert planner.progress is not None
    assert planner.progress.status is StageStatus.RUNNING


def test_replan_preserves_completed_prefix_and_replaces_current_and_tail() -> None:
    first, failed, old_tail = _stage("first"), _stage("failed"), _stage("old_tail")
    replacement = (_stage("replacement"), _stage("new_tail"))
    builder = StaticBuilder((first, failed, old_tail), replacement)
    planner = StagePlanner(builder)
    planner.initialize("task", {})
    planner.observe_turn(
        first,
        TurnRole.PRIMARY,
        ToolResult(ok=True, code="done"),
        _reward(RewardStatus.SUCCESS),
    )
    verdict = _reward(RewardStatus.FAILURE)

    planner.replan("task", {"changed": True}, failed, verdict)

    assert planner.plan.stages == (first, *replacement)
    assert planner.next_stage("task", {}) is replacement[0]
    assert builder.replan_args == ((first,), failed, verdict)
