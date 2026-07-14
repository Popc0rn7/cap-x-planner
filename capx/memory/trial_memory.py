"""Trial Memory state and trace for one fine-grained task attempt."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from capx.memory._common import (
    json_value as _json_value,
    without_privileged_fields as _without_privileged_fields,
    without_spatial_bindings as _without_spatial_bindings,
)
from capx.memory.global_memory import GlobalMemory, GlobalMemorySnapshot, StaticGlobalMemory
from capx.memory.task_memory import (
    FileTaskMemory,
    InMemoryTaskMemory,
    TaskMemory,
    TaskMemoryRecord,
)

if TYPE_CHECKING:
    from capx.envs.trial_fine_grained import (
        Observation,
        RecoveryDecision,
        ToolCall,
        ToolResult,
        WorldState,
    )
    from capx.planning.stage_planner import TurnContext
    from capx.planning.stage_reward import StageReward


@dataclass(frozen=True)
class TrialMemoryArtifact:
    """Serializable terminal snapshot of one trial's lowest memory layer."""

    task: str
    success: bool
    final_state: dict[str, Any]
    attempts: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class HierarchicalTrialMemory:
    """Trial-scoped state backed by Task Memory and read-only Global Memory."""

    def __init__(
        self,
        *,
        task_memory: TaskMemory | None = None,
        global_memory: GlobalMemory | None = None,
        budgets: Mapping[str, int | float | None] | None = None,
    ) -> None:
        self.task_memory = task_memory or InMemoryTaskMemory()
        self.global_memory = global_memory or StaticGlobalMemory()
        self._initial_budgets = dict(budgets or {})
        self._state: dict[str, Any] | None = None
        self._attempts: list[dict[str, Any]] = []
        self._pending_by_turn: dict[int, dict[str, Any]] = {}
        self._finalized: TrialMemoryArtifact | None = None

    @property
    def state(self) -> dict[str, Any]:
        return copy.deepcopy(self._require_state())

    @property
    def attempts(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._attempts))

    def bootstrap(self, task: str, observation: Observation) -> WorldState:
        if not task.strip():
            raise ValueError("Trial Memory requires a non-empty task")
        clean_observation = _without_privileged_fields(observation)
        task_record = self.task_memory.get(task)
        global_snapshot = self.global_memory.read()
        task_payload = task_record.to_dict() if task_record is not None else None
        if task_payload is not None:
            task_payload["audit"] = _without_spatial_bindings(task_payload["audit"])
            task_payload["commands"] = [
                self._symbolic_call(command) for command in task_payload["commands"]
            ]
        self._attempts = []
        self._pending_by_turn = {}
        self._finalized = None
        self._state = {
            "task": task,
            "observation_id": 0,
            "observation": clean_observation,
            "robot": self._robot_state(clean_observation),
            "objects": copy.deepcopy(clean_observation.get("objects", {})),
            "holding": clean_observation.get("holding"),
            "current_stage": None,
            "last_outcome": None,
            "failure_history": [],
            "budgets": {
                **self._initial_budgets,
                "turns_used": 0,
                "retries_used": 0,
                "restages_used": 0,
                "repair_turns_used": 0,
            },
            "task_memory": task_payload,
            "global_memory": global_snapshot.to_dict(),
        }
        return self.state

    def update_after_execution(
        self,
        context: TurnContext,
        call: ToolCall,
        result: ToolResult,
        observation: Observation,
    ) -> WorldState:
        state = self._require_state()
        if context.turn_id in self._pending_by_turn:
            raise ValueError(f"Turn {context.turn_id} was already recorded")

        before = self._state_snapshot(state)
        clean_observation = _without_privileged_fields(observation)
        state.update(_without_privileged_fields(result.updated_state))
        state["observation_id"] += 1
        state["observation"] = clean_observation
        state["robot"] = self._robot_state(clean_observation, state.get("robot"))
        if "objects" in clean_observation:
            state["objects"] = copy.deepcopy(clean_observation["objects"])
        if "holding" in clean_observation:
            state["holding"] = copy.deepcopy(clean_observation["holding"])
        state["current_stage"] = context.stage_id
        state["budgets"]["turns_used"] = len(self._attempts) + 1
        if context.role.value == "retry":
            state["budgets"]["retries_used"] += 1
        elif context.role.value == "restage_retry":
            state["budgets"]["restages_used"] += 1
        elif context.role.value == "restage":
            state["budgets"]["repair_turns_used"] += 1
        state["last_outcome"] = {
            "turn_id": context.turn_id,
            "tool_ok": result.ok,
            "tool_code": result.code,
            "stage_reward": None,
        }

        attempt = {
            "turn_id": context.turn_id,
            "stage_id": context.stage_id,
            "turn_role": context.role.value,
            "call": self._call_dict(call),
            "result": self._result_dict(result),
            "observation_id": state["observation_id"],
            "observation": _json_value(clean_observation),
            "state_before": before,
            "state_after": self._state_snapshot(state),
            "stage_reward": None,
            "recovery": None,
        }
        self._attempts.append(attempt)
        self._pending_by_turn[context.turn_id] = attempt
        return self.state

    def record_stage_reward(
        self,
        context: TurnContext,
        call: ToolCall,
        result: ToolResult,
        reward: StageReward,
        state: WorldState,
    ) -> None:
        del call, result, state
        attempt = self._pending_attempt(context)
        if attempt["stage_reward"] is not None:
            raise ValueError(f"Turn {context.turn_id} already has a Stage reward")
        reward_record = {
            "reward": reward.reward,
            "reward_valid": reward.reward_valid,
            "status": reward.status.value,
            "code": reward.code,
            "message": reward.message,
            "evidence": _json_value(reward.evidence),
            "generated_code": reward.generated_code,
            "tool_trace": _json_value(reward.tool_trace),
        }
        attempt["stage_reward"] = reward_record
        current = self._require_state()
        current["last_outcome"]["stage_reward"] = copy.deepcopy(reward_record)
        if reward.status.value != "success":
            current["failure_history"].append(
                {
                    "turn_id": context.turn_id,
                    "stage_id": context.stage_id,
                    "turn_role": context.role.value,
                    **copy.deepcopy(reward_record),
                }
            )

    def record_verdict(
        self,
        context: TurnContext,
        call: ToolCall,
        result: ToolResult,
        verdict: StageReward,
        state: WorldState,
    ) -> None:
        """Deprecated alias for integrations using the P0 Memory protocol."""

        self.record_stage_reward(context, call, result, verdict, state)

    def record_recovery(
        self,
        context: TurnContext,
        decision: RecoveryDecision,
        state: WorldState,
    ) -> None:
        del state
        attempt = self._pending_attempt(context)
        if attempt["stage_reward"] is None:
            raise ValueError("Recovery cannot be recorded before Stage reward")
        if attempt["recovery"] is not None:
            raise ValueError(f"Turn {context.turn_id} already has a recovery decision")
        attempt["recovery"] = {
            "action": decision.action.value,
            "message": decision.message,
            "retry_failed_call": decision.retry_failed_call,
            "tool_calls": [self._call_dict(call) for call in decision.tool_calls],
        }

    def finalize(
        self,
        success: bool,
        *,
        audit: Mapping[str, Any] | None = None,
    ) -> TrialMemoryArtifact:
        if self._finalized is not None:
            return copy.deepcopy(self._finalized)
        state = self._require_state()
        artifact = TrialMemoryArtifact(
            task=state["task"],
            success=success,
            final_state=_json_value(state),
            attempts=tuple(_json_value(self._attempts)),
        )
        self._finalized = artifact
        if success:
            commands = tuple(
                self._symbolic_call(attempt["call"])
                for attempt in self._attempts
                if attempt["stage_reward"] is not None
                and attempt["stage_reward"]["status"] == "success"
            )
            failures = copy.deepcopy(state["failure_history"])
            task_audit = {
                "success": True,
                "strategy": [command["action"] for command in commands],
                "failure_observations": failures,
                **_json_value(dict(audit or {})),
            }
            self.task_memory.put(
                TaskMemoryRecord(task=state["task"], audit=task_audit, commands=commands)
            )
        return copy.deepcopy(artifact)

    def write_trace_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(_json_value(attempt), sort_keys=True) for attempt in self._attempts]
        destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _require_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Trial Memory has not been bootstrapped")
        return self._state

    def _pending_attempt(self, context: TurnContext) -> dict[str, Any]:
        attempt = self._pending_by_turn.get(context.turn_id)
        if attempt is None:
            raise ValueError(f"Turn {context.turn_id} has no execution record")
        if attempt["stage_id"] != context.stage_id or attempt["turn_role"] != context.role.value:
            raise ValueError(f"TurnContext does not match turn {context.turn_id}")
        return attempt

    @staticmethod
    def _robot_state(
        observation: Mapping[str, Any],
        previous: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        robot = copy.deepcopy(dict(previous or {}))
        if isinstance(observation.get("robot"), Mapping):
            robot.update(copy.deepcopy(observation["robot"]))
        for key in (
            "base_pose",
            "gripper",
            "robot_cartesian_pos",
            "robot_joint_pos",
        ):
            if key in observation:
                robot[key] = copy.deepcopy(observation[key])
        return robot

    @staticmethod
    def _state_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "observation_id",
            "robot",
            "objects",
            "holding",
            "current_stage",
            "last_outcome",
            "budgets",
        )
        return _json_value({key: state.get(key) for key in keys})

    @staticmethod
    def _call_dict(call: ToolCall) -> dict[str, Any]:
        return {
            "id": call.id,
            "action": call.action,
            "target": _json_value(call.target),
            "params": _json_value(call.params),
            "reward_rules": _json_value(call.reward_rules),
            "timeout_s": call.timeout_s,
        }

    @staticmethod
    def _result_dict(result: ToolResult) -> dict[str, Any]:
        return {
            "ok": result.ok,
            "code": result.code,
            "message": result.message,
            "updated_state": _json_value(result.updated_state),
            "telemetry": _json_value(result.telemetry),
        }

    @staticmethod
    def _symbolic_call(call: Mapping[str, Any]) -> dict[str, Any]:
        target = call.get("target")
        return {
            "action": call["action"],
            "target": target if isinstance(target, str) else None,
            "params": _without_spatial_bindings(call.get("params", {})),
            "reward_rules": _without_spatial_bindings(
                call.get("reward_rules", call.get("postconditions", []))
            ),
        }


__all__ = [
    "GlobalMemory",
    "GlobalMemorySnapshot",
    "HierarchicalTrialMemory",
    "InMemoryTaskMemory",
    "FileTaskMemory",
    "StaticGlobalMemory",
    "TaskMemory",
    "TaskMemoryRecord",
    "TrialMemoryArtifact",
]
