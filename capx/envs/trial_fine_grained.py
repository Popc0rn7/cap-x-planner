"""Primitive-level trial loop for closed-loop agentic planning.

Unlike :mod:`capx.envs.trial`, this module never executes a generated Python program. Each
turn executes exactly one structured tool call, refreshes the observation, verifies the
post-condition, and then asks a recovery policy whether to retry, re-stage, replan, or stop.

The planner and execution services are injected through protocols so this loop can be tested
without a simulator or model server and connected to either a coding-agent or model-backed
planner later.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from capx.envs.configs.instantiate import instantiate, locate
from capx.envs.stage_planner import Stage, TurnContext, TurnRole
from capx.envs.trial_base import TrialContext, TrialExecutorBase
from capx.utils.launch_utils import TrialSummary

WorldState = dict[str, Any]
Observation = dict[str, Any]


class VerdictStatus(StrEnum):
    """Result of checking one primitive's post-condition."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    UNSAFE = "unsafe"


class RecoveryAction(StrEnum):
    """Control-flow decision after an unsuccessful primitive."""

    RETRY = "retry"
    RESTAGE = "restage"
    REPLAN = "replan"
    ABORT = "abort"


@dataclass(frozen=True)
class ToolCall:
    """One planner-selected primitive invocation."""

    action: str
    target: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)
    postconditions: tuple[dict[str, Any], ...] = ()
    timeout_s: float = 10.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not self.action.strip():
            raise ValueError("ToolCall.action must be non-empty")
        if self.timeout_s <= 0:
            raise ValueError("ToolCall.timeout_s must be positive")


@dataclass(frozen=True)
class ToolResult:
    """Normalized result from a primitive executor."""

    ok: bool
    code: str
    message: str = ""
    updated_state: WorldState = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationVerdict:
    """Evidence-backed interpretation of a tool result and refreshed state."""

    status: VerdictStatus
    code: str
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryDecision:
    """Recovery action and optional analytic calls used to re-stage the robot."""

    action: RecoveryAction
    tool_calls: tuple[ToolCall, ...] = ()
    retry_failed_call: bool = True
    message: str = ""

    def __post_init__(self) -> None:
        if self.action is RecoveryAction.RESTAGE and not self.tool_calls:
            raise ValueError("RESTAGE requires at least one recovery ToolCall")


@dataclass(frozen=True)
class FineGrainedTrialConfig:
    """Budgets owned by the primitive-level trial loop."""

    max_turns: int = 50
    max_wall_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_wall_time_s is not None and self.max_wall_time_s <= 0:
            raise ValueError("max_wall_time_s must be positive when set")


class FineGrainedPlanner(Protocol):
    """Planner interface consumed by the fine-grained trial loop."""

    def initialize(self, task: str, state: WorldState) -> None:
        """Create or load the stage plan for a new trial."""

    def next_stage(self, task: str, state: WorldState) -> Stage | None:
        """Return the current stage, or ``None`` when the plan is exhausted."""

    def observe_turn(
        self,
        stage: Stage,
        role: TurnRole,
        result: ToolResult,
        verdict: VerificationVerdict,
    ) -> None:
        """Update stage state after any primary or repair attempt."""

    def replan(
        self,
        task: str,
        state: WorldState,
        failed_stage: Stage,
        verdict: VerificationVerdict,
    ) -> None:
        """Replace or revise the remaining stage plan."""


class ToolExecutor(Protocol):
    """Execute one primitive against the current physical state."""

    def execute(self, call: ToolCall, state: WorldState) -> ToolResult:
        """Run a call until its internal stop condition or timeout."""


class Verifier(Protocol):
    """Check a primitive after the observation has been refreshed."""

    def check(
        self,
        call: ToolCall,
        state: WorldState,
        result: ToolResult,
    ) -> VerificationVerdict:
        """Return an evidence-backed post-condition verdict."""


class RecoveryPolicy(Protocol):
    """Select retry, re-stage, replan, or abort after a failed verification."""

    def decide(
        self,
        call: ToolCall,
        state: WorldState,
        verdict: VerificationVerdict,
        retry_count: int,
    ) -> RecoveryDecision:
        """Choose the next control-flow action."""


class TrialMemory(Protocol):
    """Own explicit world state and the auditable execution trace."""

    def bootstrap(self, task: str, observation: Observation) -> WorldState:
        """Build the initial state from the first observation and retrieved memory."""

    def update_after_execution(
        self,
        context: TurnContext,
        call: ToolCall,
        result: ToolResult,
        observation: Observation,
    ) -> WorldState:
        """Refresh state after every physical attempt, including failed attempts."""

    def record_verdict(
        self,
        context: TurnContext,
        call: ToolCall,
        result: ToolResult,
        verdict: VerificationVerdict,
        state: WorldState,
    ) -> None:
        """Append the verified attempt to episode memory."""


@dataclass(frozen=True)
class FineGrainedTrialComponents:
    """Services required to run one fine-grained trial."""

    planner: FineGrainedPlanner
    tool_executor: ToolExecutor
    verifier: Verifier
    recovery_policy: RecoveryPolicy
    memory: TrialMemory


@dataclass(frozen=True)
class _ScheduledCall:
    stage: Stage
    call: ToolCall
    role: TurnRole


def _low_level_env(env: Any) -> Any:
    return getattr(env, "low_level_env", env)


def _observe(env: Any) -> Observation:
    observation = _low_level_env(env).get_observation()
    if not isinstance(observation, dict):
        raise TypeError("Fine-grained observations must be dictionaries")
    return observation


def _task_completed(env: Any) -> bool:
    checker = getattr(_low_level_env(env), "task_completed", None)
    return bool(checker()) if callable(checker) else False


def _reward(env: Any) -> float:
    reward_fn = getattr(env, "compute_reward", None)
    if not callable(reward_fn):
        reward_fn = getattr(_low_level_env(env), "compute_reward", None)
    return float(reward_fn()) if callable(reward_fn) else 0.0


def _trace_line(turn: int, event: str, **fields: Any) -> str:
    payload = {"turn": turn, "event": event, **fields}
    return json.dumps(payload, default=repr, sort_keys=True)


def run_fine_grained_trial(
    env: Any,
    trial: int,
    components: FineGrainedTrialComponents,
    *,
    task: str | None = None,
    config: FineGrainedTrialConfig | None = None,
) -> TrialSummary:
    """Run one task attempt with exactly one primitive per execution turn.

    This function intentionally has the same environment/trial inputs and returns the same
    :class:`TrialSummary` type as the existing code-agent trial. The configured trial
    executor constructs ``FineGrainedTrialComponents`` before dispatching here.
    """

    trial_config = config or FineGrainedTrialConfig()
    started_at = time.monotonic()
    trace: list[str] = []
    pending_calls: deque[_ScheduledCall] = deque()
    current_stage: Stage | None = None
    current_record: dict[str, Any] | None = None
    stage_records: list[dict[str, Any]] = []
    retry_used = 0
    restage_used = 0
    turns_executed = 0
    stop_reason = "budget_exhausted"

    observation, reset_info = env.reset(options={"trial": trial}, seed=trial)
    if not isinstance(observation, dict):
        raise TypeError("Fine-grained reset observations must be dictionaries")
    reset_info = reset_info if isinstance(reset_info, dict) else {}
    task_text = task or reset_info.get("task_prompt") or getattr(env, "_task_prompt", None)
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("A non-empty task must be provided to run_fine_grained_trial")

    state = components.memory.bootstrap(task_text, observation)
    components.planner.initialize(task_text, state)

    for turn in range(trial_config.max_turns):
        if _task_completed(env):
            stop_reason = "task_completed"
            break
        if (
            trial_config.max_wall_time_s is not None
            and time.monotonic() - started_at >= trial_config.max_wall_time_s
        ):
            stop_reason = "wall_time_exhausted"
            break

        if pending_calls:
            scheduled = pending_calls.popleft()
        else:
            next_stage = components.planner.next_stage(task_text, state)
            if next_stage is None:
                stop_reason = "planner_stopped"
                break
            if current_stage is None or next_stage.id != current_stage.id:
                current_stage = next_stage
                retry_used = 0
                restage_used = 0
                current_record = {
                    "stage_id": next_stage.id,
                    "objective": next_stage.objective,
                    "primary_call_id": next_stage.primary_call.id,
                    "primary_action": next_stage.primary_call.action,
                    "status": "running",
                    "retry_budget": next_stage.retry_budget,
                    "restage_budget": next_stage.restage_budget,
                    "retries_used": 0,
                    "restages_used": 0,
                    "turns": 0,
                }
                stage_records.append(current_record)
                trace.append(
                    _trace_line(
                        turn,
                        "stage_started",
                        stage_id=next_stage.id,
                        objective=next_stage.objective,
                        retry_budget=next_stage.retry_budget,
                        restage_budget=next_stage.restage_budget,
                    )
                )
            scheduled = _ScheduledCall(
                stage=next_stage,
                call=next_stage.primary_call,
                role=TurnRole.PRIMARY,
            )

        stage = scheduled.stage
        call = scheduled.call
        context = TurnContext(turn_id=turn, stage_id=stage.id, role=scheduled.role)
        turns_executed += 1
        assert current_record is not None
        current_record["turns"] += 1
        trace.append(
            _trace_line(
                turn,
                "tool_call",
                stage_id=stage.id,
                turn_role=scheduled.role.value,
                call_id=call.id,
                action=call.action,
                target=call.target,
                params=call.params,
                retries_used=retry_used,
                retries_remaining=stage.retry_budget - retry_used,
                restages_used=restage_used,
                restages_remaining=stage.restage_budget - restage_used,
            )
        )

        try:
            result = components.tool_executor.execute(call, state)
        except Exception as exc:  # Normalize tool boundary failures for recovery.
            result = ToolResult(
                ok=False,
                code="tool_exception",
                message=repr(exc),
                telemetry={"exception_type": type(exc).__name__},
            )

        # A failed primitive can still move the robot or scene. Always observe and update first.
        observation = _observe(env)
        state = components.memory.update_after_execution(context, call, result, observation)
        verdict = components.verifier.check(call, state, result)
        components.memory.record_verdict(context, call, result, verdict, state)
        components.planner.observe_turn(stage, scheduled.role, result, verdict)
        trace.append(
            _trace_line(
                turn,
                "verdict",
                stage_id=stage.id,
                turn_role=scheduled.role.value,
                call_id=call.id,
                tool_ok=result.ok,
                tool_code=result.code,
                status=verdict.status.value,
                verdict_code=verdict.code,
            )
        )

        is_primary_attempt = scheduled.role in {
            TurnRole.PRIMARY,
            TurnRole.RETRY,
            TurnRole.RESTAGE_RETRY,
        }
        if verdict.status is VerdictStatus.SUCCESS and is_primary_attempt:
            current_record["status"] = "success"
            current_record["retries_used"] = retry_used
            current_record["restages_used"] = restage_used
            trace.append(
                _trace_line(
                    turn,
                    "stage_completed",
                    stage_id=stage.id,
                    turn_role=scheduled.role.value,
                    retries_used=retry_used,
                    restages_used=restage_used,
                )
            )
            current_stage = None
            current_record = None
            pending_calls.clear()
        if _task_completed(env):
            stop_reason = "task_completed"
            break
        if verdict.status is VerdictStatus.SUCCESS:
            continue
        if verdict.status is VerdictStatus.UNSAFE:
            current_record["status"] = "aborted"
            current_record["retries_used"] = retry_used
            current_record["restages_used"] = restage_used
            trace.append(
                _trace_line(
                    turn,
                    "stage_aborted",
                    stage_id=stage.id,
                    turn_role=scheduled.role.value,
                    reason=f"unsafe:{verdict.code}",
                )
            )
            stop_reason = f"unsafe:{verdict.code}"
            break

        # A failed repair invalidates the recovery sequence. It is never recursively recovered.
        if scheduled.role is TurnRole.RESTAGE:
            pending_calls.clear()
            trace.append(
                _trace_line(
                    turn,
                    "recovery",
                    stage_id=stage.id,
                    turn_role=scheduled.role.value,
                    call_id=call.id,
                    action=RecoveryAction.REPLAN.value,
                    message="restage repair failed",
                    retries_used=retry_used,
                    restages_used=restage_used,
                )
            )
            current_record["status"] = "failed"
            current_record["retries_used"] = retry_used
            current_record["restages_used"] = restage_used
            trace.append(
                _trace_line(
                    turn,
                    "stage_failed",
                    stage_id=stage.id,
                    turn_role=scheduled.role.value,
                    reason=verdict.code,
                )
            )
            components.planner.replan(task_text, state, stage, verdict)
            current_stage = None
            current_record = None
            continue

        decision = components.recovery_policy.decide(call, state, verdict, retry_used)
        trace.append(
            _trace_line(
                turn,
                "recovery",
                stage_id=stage.id,
                turn_role=scheduled.role.value,
                call_id=call.id,
                action=decision.action.value,
                message=decision.message,
                retries_used=retry_used,
                retries_remaining=stage.retry_budget - retry_used,
                restages_used=restage_used,
                restages_remaining=stage.restage_budget - restage_used,
            )
        )

        if decision.action is RecoveryAction.RETRY:
            if retry_used >= stage.retry_budget:
                pending_calls.clear()
                current_record["status"] = "failed"
                current_record["retries_used"] = retry_used
                current_record["restages_used"] = restage_used
                trace.append(
                    _trace_line(
                        turn,
                        "retry_budget_exhausted",
                        stage_id=stage.id,
                        turn_role=scheduled.role.value,
                        call_id=call.id,
                        retries_used=retry_used,
                        retries_remaining=0,
                    )
                )
                trace.append(
                    _trace_line(
                        turn,
                        "stage_failed",
                        stage_id=stage.id,
                        turn_role=scheduled.role.value,
                        reason="retry_budget_exhausted",
                    )
                )
                components.planner.replan(task_text, state, stage, verdict)
                current_stage = None
                current_record = None
                continue
            retry_used += 1
            current_record["retries_used"] = retry_used
            pending_calls.appendleft(
                _ScheduledCall(stage=stage, call=stage.primary_call, role=TurnRole.RETRY)
            )
        elif decision.action is RecoveryAction.RESTAGE:
            pending_calls.clear()
            if restage_used >= stage.restage_budget:
                current_record["status"] = "failed"
                current_record["retries_used"] = retry_used
                current_record["restages_used"] = restage_used
                trace.append(
                    _trace_line(
                        turn,
                        "restage_budget_exhausted",
                        stage_id=stage.id,
                        turn_role=scheduled.role.value,
                        restages_used=restage_used,
                        restages_remaining=0,
                    )
                )
                trace.append(
                    _trace_line(
                        turn,
                        "stage_failed",
                        stage_id=stage.id,
                        turn_role=scheduled.role.value,
                        reason="restage_budget_exhausted",
                    )
                )
                components.planner.replan(task_text, state, stage, verdict)
                current_stage = None
                current_record = None
                continue
            restage_used += 1
            current_record["restages_used"] = restage_used
            pending_calls.append(
                _ScheduledCall(
                    stage=stage,
                    call=stage.primary_call,
                    role=TurnRole.RESTAGE_RETRY,
                )
            )
            for repair_call in reversed(decision.tool_calls):
                pending_calls.appendleft(
                    _ScheduledCall(stage=stage, call=repair_call, role=TurnRole.RESTAGE)
                )
        elif decision.action is RecoveryAction.REPLAN:
            pending_calls.clear()
            current_record["status"] = "failed"
            current_record["retries_used"] = retry_used
            current_record["restages_used"] = restage_used
            trace.append(
                _trace_line(
                    turn,
                    "stage_failed",
                    stage_id=stage.id,
                    turn_role=scheduled.role.value,
                    reason=verdict.code,
                )
            )
            components.planner.replan(task_text, state, stage, verdict)
            current_stage = None
            current_record = None
        else:
            pending_calls.clear()
            current_record["status"] = "aborted"
            current_record["retries_used"] = retry_used
            current_record["restages_used"] = restage_used
            trace.append(
                _trace_line(
                    turn,
                    "stage_aborted",
                    stage_id=stage.id,
                    turn_role=scheduled.role.value,
                    reason=verdict.code,
                )
            )
            stop_reason = f"aborted:{verdict.code}"
            break

    task_completed = _task_completed(env)
    reward = _reward(env)
    if task_completed:
        stop_reason = "task_completed"
    trace.append(
        _trace_line(
            turns_executed,
            "trial_finished",
            stop_reason=stop_reason,
            task_completed=task_completed,
            reward=reward,
        )
    )

    return TrialSummary(
        trial=trial,
        success=task_completed,
        reward=reward,
        terminated=task_completed,
        truncated=not task_completed,
        sandbox_rc=0,
        log="\n".join(trace),
        task_completed=task_completed,
        code_path=None,
        num_regenerations=0,
        num_finishes=int(stop_reason == "planner_stopped"),
        num_code_blocks=turns_executed,
        stage_records=stage_records,
        num_stages=len(stage_records),
        num_stages_completed=sum(record["status"] == "success" for record in stage_records),
    )


class FineGrainedTrialExecutor(TrialExecutorBase):
    """Configure and run the primitive-level trial algorithm."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def run(self, context: TrialContext) -> TrialSummary:
        components = self._build_components(context.env)
        trial_config = FineGrainedTrialConfig(
            max_turns=int(self.config.get("max_turns", 50)),
            max_wall_time_s=self.config.get("max_wall_time_s"),
        )
        return run_fine_grained_trial(
            context.env,
            context.trial,
            components,
            task=self.config.get("task"),
            config=trial_config,
        )

    def _build_components(self, env: Any) -> FineGrainedTrialComponents:
        factory_spec = self.config.get("component_factory")
        if factory_spec is None:
            raise ValueError(
                "fine_grained trial executor requires 'component_factory'; it must build "
                "planner, tool_executor, verifier, recovery_policy, and memory"
            )

        factory_kwargs: dict[str, Any] = {}
        if isinstance(factory_spec, str):
            factory = locate(factory_spec)
        elif isinstance(factory_spec, Mapping):
            factory_config = dict(factory_spec)
            target = factory_config.pop("_target_", None)
            if not isinstance(target, str) or not target:
                raise ValueError("component_factory mapping requires a string '_target_'")
            factory = locate(target)
            factory_kwargs = {key: instantiate(value) for key, value in factory_config.items()}
        elif callable(factory_spec):
            # Callable support is useful for tests and programmatic integrations.
            factory = factory_spec
        else:
            raise TypeError("component_factory must be a dotted name, mapping, or callable")

        if not callable(factory):
            raise TypeError("component_factory did not resolve to a callable")
        components = factory(env=env, **factory_kwargs)
        if not isinstance(components, FineGrainedTrialComponents):
            raise TypeError("component_factory must return FineGrainedTrialComponents")
        return components


__all__ = [
    "FineGrainedPlanner",
    "FineGrainedTrialComponents",
    "FineGrainedTrialConfig",
    "FineGrainedTrialExecutor",
    "Observation",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryPolicy",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "TrialMemory",
    "VerificationVerdict",
    "Verifier",
    "VerdictStatus",
    "WorldState",
    "run_fine_grained_trial",
]
