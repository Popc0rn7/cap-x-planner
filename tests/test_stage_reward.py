from __future__ import annotations

import pytest

from capx.envs.trial_fine_grained import ToolCall, ToolResult
from capx.planning.stage_planner import TurnContext, TurnRole
from capx.planning.stage_reward import (
    AgenticStageRewardVerifier,
    ModelStageRewardCodeAgent,
    ReadOnlyStageRewardSandbox,
    ReadOnlyVerificationTool,
    RewardStatus,
    StageReward,
    StageRewardContext,
    StageRewardRule,
    StaticStageRewardCodeAgent,
    VerificationToolRegistry,
)


def _context(
    code_rule: StageRewardRule | None = None,
    *,
    before: dict | None = None,
    after: dict | None = None,
) -> StageRewardContext:
    rule = code_rule or StageRewardRule(
        predicate="object_stably_grasped",
        description="the requested object is held by the gripper",
        target="cup",
        required_evidence=("holding",),
        allowed_tools=("read_holding",),
    )
    call = ToolCall(action="VLA", target="cup", reward_rules=(rule,))
    return StageRewardContext(
        turn=TurnContext(0, "grasp", TurnRole.PRIMARY),
        call=call,
        result=ToolResult(ok=True, code="controller_stopped"),
        state_before=before or {"holding": None},
        state_after=after or {"holding": "cup"},
        rules=call.reward_rules,
    )


def _sandbox() -> ReadOnlyStageRewardSandbox:
    return ReadOnlyStageRewardSandbox(
        VerificationToolRegistry(
            (
                ReadOnlyVerificationTool(
                    "read_holding",
                    "Return the symbolic object currently held by the gripper.",
                    lambda state: state.get("holding"),
                ),
            )
        )
    )


def test_agent_code_computes_binary_reward_with_read_only_tool_evidence() -> None:
    code = """
def compute_reward(state_before, state_after, tool_result, tools):
    holding = tools["read_holding"](state_after)
    return {
        "reward": 1.0 if holding == "cup" else 0.0,
        "code": "stable_grasp" if holding == "cup" else "empty_grasp",
        "evidence": {"holding": holding},
    }
"""
    verifier = AgenticStageRewardVerifier(
        StaticStageRewardCodeAgent(code),
        _sandbox(),
    )

    reward = verifier.compute_reward(_context())

    assert reward.status is RewardStatus.SUCCESS
    assert reward.reward == 1.0
    assert reward.reward_valid is True
    assert reward.evidence == {"holding": "cup"}
    assert reward.tool_trace[0]["tool"] == "read_holding"
    assert reward.generated_code == code


def test_model_code_agent_extracts_code_from_repository_client_response(monkeypatch) -> None:
    generated = (
        "def compute_reward(state_before, state_after, tool_result, tools):\n"
        "    return {'reward': 1.0, 'evidence': {'holding': 'cup'}}"
    )

    def fake_query_model(args, prompt):
        assert prompt[0]["role"] == "user"
        return {"content": f"```python\n{generated}\n```", "reasoning": None}

    monkeypatch.setattr("capx.llm.client.query_model", fake_query_model)

    code = ModelStageRewardCodeAgent(model_args=object()).generate(
        _context(), _sandbox().tool_descriptions(_context())
    )

    assert code == generated


def test_missing_required_evidence_returns_unknown_zero_reward() -> None:
    code = """
def compute_reward(state_before, state_after, tool_result, tools):
    return {"reward": 0.0, "code": "not_sure", "evidence": {}}
"""
    verifier = AgenticStageRewardVerifier(StaticStageRewardCodeAgent(code), _sandbox())

    reward = verifier.compute_reward(_context())

    assert reward.status is RewardStatus.UNKNOWN
    assert reward.reward == 0.0
    assert reward.reward_valid is False
    assert "holding" in reward.message


@pytest.mark.parametrize(
    "code",
    [
        "import os\n\ndef compute_reward(state_before, state_after, tool_result, tools):\n    "
        "return {'reward': 1.0, 'evidence': {'holding': 'cup'}}",
        "def compute_reward(state_before, state_after, tool_result, tools):\n    "
        "state_after['holding'] = 'cup'\n    return {'reward': 1.0, 'evidence': {}}",
        "def compute_reward(state_before, state_after, tool_result, tools):\n    "
        "while True:\n        pass",
    ],
)
def test_sandbox_rejects_import_mutation_and_unbounded_control_flow(code: str) -> None:
    verifier = AgenticStageRewardVerifier(StaticStageRewardCodeAgent(code), _sandbox())

    reward = verifier.compute_reward(_context())

    assert reward.status is RewardStatus.UNKNOWN
    assert reward.code == "reward_code_error"
    assert reward.reward_valid is False


def test_unknown_tool_in_rule_is_rejected_before_agent_code_runs() -> None:
    rule = StageRewardRule(
        predicate="visible",
        description="the target remains visible",
        allowed_tools=("move_robot",),
    )
    verifier = AgenticStageRewardVerifier(
        StaticStageRewardCodeAgent(
            "def compute_reward(state_before, state_after, tool_result, tools):\n"
            "    return {'reward': 1.0, 'evidence': {}}"
        ),
        _sandbox(),
    )

    reward = verifier.compute_reward(_context(rule))

    assert reward.status is RewardStatus.UNKNOWN
    assert reward.code == "reward_code_error"
    assert "Unknown verification tool" in reward.message


def test_agent_can_report_evidence_backed_unsafe_state() -> None:
    rule = StageRewardRule(
        predicate="collision_free",
        description="the primitive must not leave the robot in collision",
        required_evidence=("collision",),
    )
    code = """
def compute_reward(state_before, state_after, tool_result, tools):
    collision = state_after.get("collision", False)
    return {
        "reward": 0.0,
        "unsafe": collision,
        "code": "collision" if collision else "clear",
        "evidence": {"collision": collision},
    }
"""
    verifier = AgenticStageRewardVerifier(StaticStageRewardCodeAgent(code), _sandbox())

    reward = verifier.compute_reward(_context(rule, after={"collision": True}))

    assert reward.status is RewardStatus.UNSAFE
    assert reward.reward == 0.0
    assert reward.reward_valid is True


def test_multiple_rules_use_all_semantics() -> None:
    rules = (
        StageRewardRule("grasped", "object is held", required_evidence=("holding",)),
        StageRewardRule("lifted", "object is lifted", required_evidence=("height",)),
    )
    call = ToolCall(action="VLA", reward_rules=rules)
    context = StageRewardContext(
        TurnContext(0, "grasp", TurnRole.PRIMARY),
        call,
        ToolResult(ok=True, code="done"),
        {},
        {"holding": "cup", "height": 0.0},
        rules,
    )
    code = """
def compute_reward(state_before, state_after, tool_result, tools):
    return {
        "reward": 0.0,
        "rule_rewards": {"grasped": 1.0, "lifted": 0.0},
        "code": "not_lifted",
        "evidence": {"holding": "cup", "height": 0.0},
    }
"""

    reward = AgenticStageRewardVerifier(
        StaticStageRewardCodeAgent(code), _sandbox()
    ).compute_reward(context)

    assert reward.status is RewardStatus.FAILURE
    assert reward.reward == 0.0


def test_missing_rule_is_unknown_instead_of_using_tool_result_ok() -> None:
    context = _context()
    context = StageRewardContext(
        turn=context.turn,
        call=ToolCall(action="VLA"),
        result=ToolResult(ok=True, code="done"),
        state_before={},
        state_after={},
        rules=(),
    )
    verifier = AgenticStageRewardVerifier(StaticStageRewardCodeAgent(""), _sandbox())

    reward = verifier.compute_reward(context)

    assert reward.status is RewardStatus.UNKNOWN
    assert reward.code == "missing_reward_rule"


def test_legacy_postcondition_is_converted_to_canonical_reward_rule() -> None:
    call = ToolCall(
        action="MOVE_EEF",
        postconditions=({"predicate": "at_target", "pose": [1, 2, 3]},),
    )

    assert call.reward_rules == (
        StageRewardRule(
            predicate="at_target",
            description="at target",
            target={"pose": [1, 2, 3]},
        ),
    )

    with pytest.raises(ValueError, match="both reward_rules and postconditions"):
        ToolCall(
            action="MOVE_EEF",
            reward_rules=call.reward_rules,
            postconditions=({"predicate": "at_target"},),
        )


def test_stage_reward_rejects_inconsistent_status_and_binary_value() -> None:
    with pytest.raises(ValueError, match="requires reward=1.0"):
        StageReward(RewardStatus.SUCCESS, "done", reward=0.0)

    with pytest.raises(ValueError, match="reward_valid=False"):
        StageReward(RewardStatus.UNKNOWN, "missing", reward_valid=True)
