from __future__ import annotations

import pytest

from capx.planning.primitives import ToolCall, ToolResult
from capx.planning.recovery import (
    FailureCode,
    RecoveryAction,
    RecoveryContext,
    RuleBasedRecoveryPolicy,
)
from capx.planning.stage_planner import Stage, TurnContext, TurnRole
from capx.planning.stage_reward import RewardStatus, StageReward


def _context(
    code: str,
    *,
    retry_budget: int = 1,
    restage_budget: int = 1,
    retries_used: int = 0,
    restages_used: int = 0,
    unsafe: bool = False,
) -> RecoveryContext:
    call = ToolCall("VLA", target="soup")
    stage = Stage("pick", "pick soup", call, retry_budget, restage_budget)
    return RecoveryContext(
        TurnContext(0, "pick", TurnRole.PRIMARY),
        stage,
        call,
        ToolResult(False, code),
        {},
        StageReward(RewardStatus.UNSAFE if unsafe else RewardStatus.FAILURE, code),
        retries_used,
        restages_used,
    )


@pytest.mark.parametrize(
    "code,failure",
    [
        ("vla_no_effect", FailureCode.VLA_NO_EFFECT),
        ("controller_failed", FailureCode.CONTROLLER_FAILED),
        ("tool_exception", FailureCode.TOOL_ERROR),
    ],
)
def test_transient_failures_retry_with_budget_then_replan(code, failure) -> None:
    policy = RuleBasedRecoveryPolicy()
    decision = policy.decide(_context(code))
    exhausted = policy.decide(_context(code, retries_used=1))

    assert (decision.action, decision.failure_code) == (RecoveryAction.RETRY, failure)
    assert exhausted.action is RecoveryAction.REPLAN


def test_empty_grasp_opens_gripper_then_retries_stage() -> None:
    decision = RuleBasedRecoveryPolicy().decide(_context("empty_grasp"))

    assert decision.action is RecoveryAction.RESTAGE
    assert decision.failure_code is FailureCode.EMPTY_GRASP
    assert decision.tool_calls[0].action == "SET_GRIPPER"
    assert decision.tool_calls[0].params == {"state": "open"}
    assert decision.tool_calls[0].reward_rules[0].allowed_tools == ("read_gripper_state",)
    assert RuleBasedRecoveryPolicy().decide(
        _context("empty_grasp", restages_used=1)
    ).action is RecoveryAction.REPLAN


@pytest.mark.parametrize(
    "code,failure",
    [
        ("object_dropped", FailureCode.OBJECT_DROPPED),
        ("placement_short", FailureCode.PLACEMENT_SHORT),
        ("ik_unreachable", FailureCode.IK_UNREACHABLE),
        ("unresolved_target", FailureCode.UNRESOLVED_TARGET),
        ("wrong_target", FailureCode.WRONG_TARGET),
        ("unrecognized_failure", FailureCode.UNKNOWN),
    ],
)
def test_nonlocal_or_unknown_failures_replan(code, failure) -> None:
    decision = RuleBasedRecoveryPolicy().decide(_context(code))
    assert (decision.action, decision.failure_code) == (RecoveryAction.REPLAN, failure)


def test_unsafe_aborts() -> None:
    decision = RuleBasedRecoveryPolicy().decide(_context("collision", unsafe=True))
    assert (decision.action, decision.failure_code) == (
        RecoveryAction.ABORT,
        FailureCode.UNSAFE,
    )
