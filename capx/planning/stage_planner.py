"""Stage-level planning state for primitive-based trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from capx.envs.trial_fine_grained import (
        ToolCall,
        ToolResult,
        WorldState,
    )
    from capx.planning.stage_reward import StageReward


class StageStatus(StrEnum):
    """Lifecycle state of one planned primary primitive."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"


class TurnRole(StrEnum):
    """Relationship between an executed call and its containing stage."""

    PRIMARY = "primary"
    RETRY = "retry"
    RESTAGE = "restage"
    RESTAGE_RETRY = "restage_retry"


@dataclass(frozen=True)
class Stage:
    """One primary primitive together with its recovery budgets."""

    id: str
    objective: str
    primary_call: ToolCall
    retry_budget: int = 0
    restage_budget: int = 0


@dataclass(frozen=True)
class StagePlan:
    """Ordered stages for one task attempt."""

    task: str
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class TurnContext:
    """Stable metadata attached to every physical execution turn."""

    turn_id: int
    stage_id: str
    role: TurnRole


@dataclass(frozen=True)
class StageProgress:
    """Read-only snapshot of the currently active stage."""

    stage: Stage
    status: StageStatus
    turns: int = 0
    retries_used: int = 0
    restages_used: int = 0


class StagePlanBuilder(Protocol):
    """Build and revise a symbolic stage plan."""

    def build(self, task: str, state: WorldState) -> StagePlan: ...

    def replan(
        self,
        task: str,
        state: WorldState,
        completed_stages: tuple[Stage, ...],
        failed_stage: Stage,
        reward: StageReward,
    ) -> tuple[Stage, ...]: ...


class StagePlanner:
    """Deterministic stage state machine backed by an injected plan builder."""

    def __init__(self, builder: StagePlanBuilder) -> None:
        self._builder = builder
        self._plan: StagePlan | None = None
        self._index = 0
        self._progress: StageProgress | None = None

    @property
    def plan(self) -> StagePlan:
        """Return the validated active plan."""

        if self._plan is None:
            raise RuntimeError("StagePlanner has not been initialized")
        return self._plan

    @property
    def progress(self) -> StageProgress | None:
        """Return a snapshot for the active stage, if one remains."""

        return self._progress

    def initialize(self, task: str, state: WorldState) -> None:
        plan = self._builder.build(task, state)
        self._validate_plan(plan)
        self._plan = plan
        self._index = 0
        self._progress = self._new_progress() if plan.stages else None

    def next_stage(self, task: str, state: WorldState) -> Stage | None:
        del task, state
        if self._plan is None:
            raise RuntimeError("StagePlanner has not been initialized")
        return self._progress.stage if self._progress is not None else None

    def observe_turn(
        self,
        stage: Stage,
        role: TurnRole,
        result: ToolResult,
        reward: StageReward,
    ) -> None:
        del result
        progress = self._require_current(stage)
        retries_used = progress.retries_used + int(role is TurnRole.RETRY)
        restages_used = progress.restages_used + int(role is TurnRole.RESTAGE_RETRY)
        status = StageStatus.ABORTED if reward.status.value == "unsafe" else StageStatus.RUNNING
        is_primary_attempt = role in {
            TurnRole.PRIMARY,
            TurnRole.RETRY,
            TurnRole.RESTAGE_RETRY,
        }
        if is_primary_attempt and reward.status.value == "success":
            status = StageStatus.SUCCESS
        self._progress = StageProgress(
            stage=stage,
            status=status,
            turns=progress.turns + 1,
            retries_used=retries_used,
            restages_used=restages_used,
        )
        if status is StageStatus.SUCCESS:
            self._index += 1
            self._progress = self._new_progress()

    def replan(
        self,
        task: str,
        state: WorldState,
        failed_stage: Stage,
        reward: StageReward,
    ) -> None:
        self._require_current(failed_stage)
        assert self._plan is not None
        completed = self._plan.stages[: self._index]
        replacement = tuple(self._builder.replan(task, state, completed, failed_stage, reward))
        new_plan = StagePlan(task=task, stages=completed + replacement)
        self._validate_plan(new_plan)
        self._plan = new_plan
        self._progress = self._new_progress()

    def _new_progress(self) -> StageProgress | None:
        assert self._plan is not None
        if self._index >= len(self._plan.stages):
            return None
        return StageProgress(self._plan.stages[self._index], StageStatus.PENDING)

    def _require_current(self, stage: Stage) -> StageProgress:
        if self._progress is None or self._progress.stage.id != stage.id:
            raise ValueError(f"Stage {stage.id!r} is not the current stage")
        return self._progress

    @staticmethod
    def _validate_plan(plan: StagePlan) -> None:
        if not isinstance(plan, StagePlan):
            raise TypeError("StagePlanBuilder.build must return a StagePlan")
        seen: set[str] = set()
        for stage in plan.stages:
            if not stage.id.strip():
                raise ValueError("Stage.id must be non-empty")
            if stage.id in seen:
                raise ValueError(f"Duplicate Stage.id: {stage.id}")
            seen.add(stage.id)
            if not stage.objective.strip():
                raise ValueError(f"Stage {stage.id!r} objective must be non-empty")
            if stage.retry_budget < 0:
                raise ValueError(f"Stage {stage.id!r} retry_budget must be non-negative")
            if stage.restage_budget < 0:
                raise ValueError(f"Stage {stage.id!r} restage_budget must be non-negative")


__all__ = [
    "Stage",
    "StagePlan",
    "StagePlanBuilder",
    "StagePlanner",
    "StageProgress",
    "StageStatus",
    "TurnContext",
    "TurnRole",
]
