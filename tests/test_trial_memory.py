from __future__ import annotations

import json

from capx.envs.trial_fine_grained import (
    RecoveryAction,
    RecoveryDecision,
    ToolCall,
    ToolResult,
)
from capx.memory.global_memory import StaticGlobalMemory
from capx.memory.task_memory import FileTaskMemory, InMemoryTaskMemory, TaskMemoryRecord
from capx.memory.trial_memory import HierarchicalTrialMemory
from capx.planning.stage_planner import TurnContext, TurnRole
from capx.planning.stage_reward import RewardStatus, StageReward, StageRewardRule


def test_bootstrap_reads_global_and_task_memory_at_their_correct_scopes() -> None:
    task_record = TaskMemoryRecord(
        task="grasp cup",
        audit={"strategy": "approach then grasp", "pose": [1, 2, 3]},
        commands=(
            {"action": "VLA", "target": "cup", "params": {"xyz": [1, 2, 3]}},
        ),
    )
    memory = HierarchicalTrialMemory(
        task_memory=InMemoryTaskMemory((task_record,)),
        global_memory=StaticGlobalMemory(
            rules=({"code": "verify_grasp", "text": "require object co-motion"},),
            failure_models=({"code": "empty_grasp"},),
        ),
        budgets={"turns": 5},
    )

    state = memory.bootstrap(
        "grasp cup",
        {
            "robot": {"gripper": "open"},
            "objects": {"cup": {"visible": True}},
            "privileged_state": {"cup_pose": [1, 2, 3]},
        },
    )

    assert state["task_memory"]["task"] == "grasp cup"
    assert state["global_memory"]["rules"][0]["code"] == "verify_grasp"
    assert state["observation_id"] == 0
    assert state["budgets"] == {
        "turns": 5,
        "turns_used": 0,
        "retries_used": 0,
        "restages_used": 0,
        "repair_turns_used": 0,
    }
    assert "pose" not in state["task_memory"]["audit"]
    assert state["task_memory"]["commands"][0]["params"] == {}
    assert "privileged_state" not in state["observation"]


def test_failed_execution_updates_state_before_recording_verdict_and_recovery() -> None:
    memory = HierarchicalTrialMemory()
    memory.bootstrap("grasp cup", {"holding": None})
    context = TurnContext(0, "grasp", TurnRole.PRIMARY)
    call = ToolCall(action="VLA", target="cup")
    result = ToolResult(
        ok=False,
        code="empty_grasp",
        updated_state={"holding": None},
        telemetry={"chunks": 2},
    )

    state = memory.update_after_execution(
        context,
        call,
        result,
        {"holding": None, "robot": {"gripper": "closed"}},
    )
    verdict = StageReward(
        RewardStatus.FAILURE,
        "empty_grasp",
        evidence={"object_motion": False},
    )
    memory.record_stage_reward(context, call, result, verdict, state)
    memory.record_recovery(
        context,
        RecoveryDecision(RecoveryAction.RESTAGE, (ToolCall(action="MOVE_EEF"),)),
        state,
    )

    attempt = memory.attempts[0]
    assert state["observation_id"] == 1
    assert state["robot"]["gripper"] == "closed"
    assert attempt["state_before"]["observation_id"] == 0
    assert attempt["state_after"]["observation_id"] == 1
    assert attempt["stage_reward"]["code"] == "empty_grasp"
    assert attempt["stage_reward"]["reward"] == 0.0
    assert attempt["recovery"]["action"] == "restage"
    assert memory.state["failure_history"][0]["turn_role"] == "primary"


def test_successful_trial_promotes_only_symbolic_calls_to_its_task_memory() -> None:
    task_memory = InMemoryTaskMemory()
    memory = HierarchicalTrialMemory(task_memory=task_memory)
    memory.bootstrap("place cup", {})
    context = TurnContext(0, "place", TurnRole.PRIMARY)
    call = ToolCall(
        action="MOVE_EEF",
        target="sink_region",
        params={
            "xyz": [0.1, 0.2, 0.3],
            "quaternion": [0, 0, 0, 1],
            "speed": 0.2,
        },
        reward_rules=(
            StageRewardRule(
                predicate="at_target",
                description="end effector reached the target",
                target={"pose": [1, 2, 3]},
            ),
        ),
    )
    result = ToolResult(ok=True, code="done")
    state = memory.update_after_execution(context, call, result, {})
    memory.record_stage_reward(
        context,
        call,
        result,
        StageReward(RewardStatus.SUCCESS, "at_target"),
        state,
    )

    artifact = memory.finalize(True)
    record = task_memory.get("place cup")

    assert artifact.success is True
    assert record is not None
    assert record.commands == (
        {
            "action": "MOVE_EEF",
            "target": "sink_region",
            "params": {"speed": 0.2},
            "reward_rules": [
                {
                    "predicate": "at_target",
                    "description": "end effector reached the target",
                    "target": {},
                    "required_evidence": [],
                    "allowed_tools": [],
                }
            ],
        },
    )


def test_failed_trial_is_not_promoted_to_task_memory() -> None:
    task_memory = InMemoryTaskMemory()
    memory = HierarchicalTrialMemory(task_memory=task_memory)
    memory.bootstrap("grasp cup", {})

    memory.finalize(False)

    assert task_memory.get("grasp cup") is None


def test_file_task_memory_and_trial_jsonl_round_trip(tmp_path) -> None:
    task_memory = FileTaskMemory(tmp_path / "tasks")
    record = TaskMemoryRecord(
        task="grasp cup",
        audit={"success": True},
        commands=({"action": "VLA", "target": "cup"},),
    )
    task_memory.put(record)

    loaded = FileTaskMemory(tmp_path / "tasks").get("grasp cup")
    assert loaded == record

    memory = HierarchicalTrialMemory()
    memory.bootstrap("observe cup", {})
    context = TurnContext(0, "observe", TurnRole.PRIMARY)
    call = ToolCall(action="OBSERVE")
    result = ToolResult(ok=True, code="observed")
    state = memory.update_after_execution(context, call, result, {})
    memory.record_stage_reward(
        context,
        call,
        result,
        StageReward(RewardStatus.SUCCESS, "visible"),
        state,
    )
    trace_path = tmp_path / "trial.jsonl"
    memory.write_trace_jsonl(trace_path)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["stage_id"] == "observe"
