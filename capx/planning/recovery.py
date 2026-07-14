"""Deterministic, budget-aware recovery for failed physical primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from capx.planning.primitives import ToolCall, ToolResult, WorldState
from capx.planning.stage_planner import Stage, TurnContext
from capx.planning.stage_reward import StageReward, StageRewardRule


class RecoveryAction(StrEnum):
    RETRY = "retry"
    RESTAGE = "restage"
    REPLAN = "replan"
    ABORT = "abort"


class FailureCode(StrEnum):
    EMPTY_GRASP = "empty_grasp"
    OBJECT_DROPPED = "object_dropped"
    IK_UNREACHABLE = "ik_unreachable"
    UNRESOLVED_TARGET = "unresolved_target"
    PLACEMENT_SHORT = "placement_short"
    VLA_NO_EFFECT = "vla_no_effect"
    WRONG_TARGET = "wrong_target"
    CONTROLLER_FAILED = "controller_failed"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class RecoveryDecision:
    """Recovery action and optional analytic calls used to re-stage the robot."""

    action: RecoveryAction
    tool_calls: tuple[ToolCall, ...] = ()
    retry_failed_call: bool = True
    message: str = ""
    failure_code: FailureCode = FailureCode.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not isinstance(self.action, RecoveryAction):
            object.__setattr__(self, "action", RecoveryAction(self.action))
        if not isinstance(self.failure_code, FailureCode):
            object.__setattr__(self, "failure_code", FailureCode(self.failure_code))
        if self.action is RecoveryAction.RESTAGE and not self.tool_calls:
            raise ValueError("RESTAGE requires at least one recovery ToolCall")


@dataclass(frozen=True)
class RecoveryContext:
    turn: TurnContext
    stage: Stage
    call: ToolCall
    result: ToolResult
    state: WorldState
    reward: StageReward
    retries_used: int
    restages_used: int


class RecoveryPolicy(Protocol):
    def decide(self, context: RecoveryContext) -> RecoveryDecision: ...


class RuleBasedRecoveryPolicy:
    """Classify evidence and apply the fixed P0 recovery table without a model call."""

    _RETRYABLE = {
        FailureCode.VLA_NO_EFFECT,
        FailureCode.CONTROLLER_FAILED,
        FailureCode.TOOL_ERROR,
    }
    _REPLAN = {
        FailureCode.OBJECT_DROPPED,
        FailureCode.PLACEMENT_SHORT,
        FailureCode.IK_UNREACHABLE,
        FailureCode.UNRESOLVED_TARGET,
        FailureCode.WRONG_TARGET,
        FailureCode.UNKNOWN,
    }

    def decide(self, context: RecoveryContext) -> RecoveryDecision:
        failure = classify_failure(context.result, context.reward)
        if failure is FailureCode.UNSAFE:
            return RecoveryDecision(
                RecoveryAction.ABORT,
                message="unsafe evidence requires immediate abort",
                failure_code=failure,
            )
        if failure is FailureCode.EMPTY_GRASP:
            if context.restages_used < context.stage.restage_budget:
                repair = ToolCall(
                    action="SET_GRIPPER",
                    target="open",
                    params={"state": "open"},
                    reward_rules=(
                        StageRewardRule(
                            predicate="gripper_open",
                            description="the gripper is confirmed open",
                            required_evidence=("gripper_state",),
                            allowed_tools=("read_gripper_state",),
                        ),
                    ),
                )
                return RecoveryDecision(
                    RecoveryAction.RESTAGE,
                    (repair,),
                    message="open gripper before retrying the failed grasp stage",
                    failure_code=failure,
                )
            return RecoveryDecision(
                RecoveryAction.REPLAN,
                message="empty-grasp restage budget exhausted",
                failure_code=failure,
            )
        if failure in self._RETRYABLE and context.retries_used < context.stage.retry_budget:
            return RecoveryDecision(
                RecoveryAction.RETRY,
                message=f"retry transient failure: {failure.value}",
                failure_code=failure,
            )
        return RecoveryDecision(
            RecoveryAction.REPLAN,
            message=f"replan after {failure.value}",
            failure_code=failure,
        )


def classify_failure(result: ToolResult, reward: StageReward) -> FailureCode:
    """Normalize heterogeneous executor/verifier codes into the recovery taxonomy."""

    if reward.status.value == "unsafe":
        return FailureCode.UNSAFE
    combined = " ".join((reward.code, reward.message, result.code, result.message)).lower()
    aliases: tuple[tuple[FailureCode, tuple[str, ...]], ...] = (
        (FailureCode.UNSAFE, ("unsafe", "collision", "speed_limit")),
        (FailureCode.EMPTY_GRASP, ("empty_grasp", "grasp_empty", "nothing_grasped")),
        (FailureCode.OBJECT_DROPPED, ("object_dropped", "dropped")),
        (FailureCode.IK_UNREACHABLE, ("ik_unreachable", "unreachable", "no_ik")),
        (
            FailureCode.UNRESOLVED_TARGET,
            ("unresolved_target", "unresolved_spatial_target", "target_not_found"),
        ),
        (FailureCode.PLACEMENT_SHORT, ("placement_short", "not_in_receptacle")),
        (FailureCode.VLA_NO_EFFECT, ("vla_no_effect", "no_effect", "no_progress")),
        (FailureCode.WRONG_TARGET, ("wrong_target", "incorrect_object")),
        (FailureCode.CONTROLLER_FAILED, ("controller_failed", "control_failed")),
        (
            FailureCode.TOOL_ERROR,
            ("tool_error", "tool_exception", "timeout", "service_unavailable"),
        ),
    )
    for failure, tokens in aliases:
        if any(token in combined for token in tokens):
            return failure
    return FailureCode.UNKNOWN


__all__ = [
    "FailureCode",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryDecision",
    "RecoveryPolicy",
    "RuleBasedRecoveryPolicy",
    "classify_failure",
]
