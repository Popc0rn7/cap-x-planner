"""Model-backed, schema-constrained symbolic stage-plan construction."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from capx.memory._common import json_value, without_privileged_fields, without_spatial_bindings
from capx.planning.primitives import PrimitiveCatalog, ToolCall, ToolResult, WorldState
from capx.planning.stage_planner import Stage, StagePlan
from capx.planning.stage_reward import StageReward, StageRewardRule

_SPATIAL_KEYS = {
    "base_pose",
    "coordinates",
    "eef_pose",
    "object_pose",
    "pixel",
    "pose",
    "position",
    "quat",
    "quaternion",
    "xy",
    "xyz",
    "x",
    "y",
    "z",
}
_RAW_SENSOR_KEYS = {"depth", "image", "images", "mask", "masks", "raw_rgb", "rgb"}


@dataclass(frozen=True)
class PlannerStateView:
    """Small symbolic view of Memory that is safe to send to a planning model."""

    task: str
    objects: Any = field(default_factory=dict)
    holding: Any = None
    robot: Any = field(default_factory=dict)
    budgets: Mapping[str, Any] = field(default_factory=dict)
    task_memory: Any = None
    global_memory: Any = None
    failure_history: tuple[Any, ...] = ()

    @classmethod
    def from_world_state(cls, task: str, state: WorldState) -> PlannerStateView:
        clean = without_privileged_fields(state)
        robot = _symbolic_value(clean.get("robot", {}))
        objects = _symbolic_value(clean.get("objects", {}))
        return cls(
            task=task,
            objects=json_value(objects),
            holding=json_value(clean.get("holding")),
            robot=json_value(robot),
            budgets=json_value(clean.get("budgets", {})),
            task_memory=_symbolic_value(clean.get("task_memory")),
            global_memory=_symbolic_value(clean.get("global_memory")),
            failure_history=tuple(_symbolic_value(clean.get("failure_history", ()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


@dataclass(frozen=True)
class StagePlanRequest:
    task: str
    state: PlannerStateView
    primitive_catalog: PrimitiveCatalog
    completed_stages: tuple[Stage, ...] = ()
    failed_stage: Stage | None = None
    failed_reward: StageReward | None = None
    failed_result: ToolResult | None = None
    validation_errors: tuple[str, ...] = ()
    reward_tools: tuple[str, ...] = ()


class StagePlanAgent(Protocol):
    def generate(self, request: StagePlanRequest) -> Mapping[str, Any]: ...


class ModelStagePlanAgent:
    """Production builder agent using the repository's configured model client."""

    def __init__(self, model_args: Any) -> None:
        self.model_args = model_args

    def generate(self, request: StagePlanRequest) -> Mapping[str, Any]:
        from capx.llm.client import query_model

        prompt = self.build_prompt(request)
        response = query_model(
            self.model_args,
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        content = response.get("content", "") if isinstance(response, Mapping) else response
        return _parse_json_object(content)

    @staticmethod
    def build_prompt(request: StagePlanRequest) -> str:
        payload = {
            "task": request.task,
            "symbolic_state": request.state.to_dict(),
            "primitive_catalog": request.primitive_catalog.to_dict(),
            "registered_reward_tools": list(request.reward_tools),
            "completed_stages": [_stage_dict(stage) for stage in request.completed_stages],
            "failed_stage": _stage_dict(request.failed_stage) if request.failed_stage else None,
            "failed_reward": json_value(request.failed_reward),
            "failed_tool_result": json_value(request.failed_result),
            "validation_errors_from_previous_attempt": list(request.validation_errors),
        }
        mode = "replacement current-and-tail stages" if request.failed_stage else "initial stages"
        return (
            "Return only one JSON object with a 'stages' array containing "
            f"{mode}. Do not write or execute Python. Each stage must contain id, objective, "
            "primary_call (action, symbolic target, params, reward_rules), retry_budget, and "
            "restage_budget. Use only catalog actions and registered reward tools. Never emit "
            "numeric poses, coordinates, pixels, or privileged state. A replan must not repeat "
            "or modify completed stages; an empty replacement means stop.\n\n"
            + json.dumps(payload, sort_keys=True)
        )


class StagePlanBuildError(ValueError):
    """The builder agent failed schema or capability validation twice."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid agent stage plan: " + "; ".join(self.errors))


class AgenticStagePlanBuilder:
    """Turn agent JSON into validated executable stages, with one repair attempt."""

    def __init__(
        self,
        agent: StagePlanAgent,
        primitive_catalog: PrimitiveCatalog,
        *,
        reward_tools: tuple[str, ...] = (),
        max_stages: int = 8,
        max_generation_attempts: int = 2,
    ) -> None:
        if not 1 <= max_stages <= 8:
            raise ValueError("max_stages must be between 1 and 8")
        if max_generation_attempts != 2:
            raise ValueError("P0 builder supports exactly two generation attempts")
        self.agent = agent
        self.primitive_catalog = primitive_catalog
        self.reward_tools = frozenset(reward_tools)
        self.max_stages = max_stages
        self.max_generation_attempts = max_generation_attempts

    def build(self, task: str, state: WorldState) -> StagePlan:
        request = StagePlanRequest(
            task=task,
            state=PlannerStateView.from_world_state(task, state),
            primitive_catalog=self.primitive_catalog,
            reward_tools=tuple(sorted(self.reward_tools)),
        )
        stages = self._generate(request, allow_empty=False)
        return StagePlan(task=task, stages=stages)

    def replan(
        self,
        task: str,
        state: WorldState,
        completed_stages: tuple[Stage, ...],
        failed_stage: Stage,
        reward: StageReward,
        result: ToolResult | None = None,
    ) -> tuple[Stage, ...]:
        request = StagePlanRequest(
            task=task,
            state=PlannerStateView.from_world_state(task, state),
            primitive_catalog=self.primitive_catalog,
            reward_tools=tuple(sorted(self.reward_tools)),
            completed_stages=completed_stages,
            failed_stage=failed_stage,
            failed_reward=reward,
            failed_result=result,
        )
        return self._generate(request, allow_empty=True)

    def _generate(self, request: StagePlanRequest, *, allow_empty: bool) -> tuple[Stage, ...]:
        errors: list[str] = []
        for _ in range(self.max_generation_attempts):
            attempt = request if not errors else dataclass_replace_errors(request, errors)
            try:
                raw = self.agent.generate(attempt)
                stages = self._decode(raw, allow_empty=allow_empty)
                completed_ids = {stage.id for stage in request.completed_stages}
                overlap = completed_ids.intersection(stage.id for stage in stages)
                if overlap:
                    raise ValueError(
                        f"replacement repeats completed Stage.id: {sorted(overlap)}"
                    )
                if len(request.completed_stages) + len(stages) > self.max_stages:
                    raise ValueError(
                        f"combined plan exceeds maximum of {self.max_stages} stages"
                    )
                return stages
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors = [str(exc)]
        raise StagePlanBuildError(errors)

    def _decode(self, raw: Any, *, allow_empty: bool) -> tuple[Stage, ...]:
        payload = _parse_json_object(raw)
        _reject_unknown_keys(payload, {"stages"}, "stage plan")
        stages_raw = payload.get("stages")
        if not isinstance(stages_raw, list):
            raise TypeError("'stages' must be an array")
        if not stages_raw and not allow_empty:
            raise ValueError("initial plan must contain at least one Stage")
        if len(stages_raw) > self.max_stages:
            raise ValueError(f"plan exceeds maximum of {self.max_stages} stages")
        stages = tuple(self._decode_stage(item) for item in stages_raw)
        ids = [stage.id for stage in stages]
        if len(ids) != len(set(ids)):
            raise ValueError("Stage IDs must be unique")
        return stages

    def _decode_stage(self, raw: Any) -> Stage:
        if not isinstance(raw, Mapping):
            raise TypeError("each Stage must be an object")
        _reject_unknown_keys(
            raw,
            {"id", "objective", "primary_call", "retry_budget", "restage_budget"},
            "Stage",
        )
        stage_id = _required_string(raw, "id")
        objective = _required_string(raw, "objective")
        call_raw = raw.get("primary_call")
        if not isinstance(call_raw, Mapping):
            raise TypeError(f"Stage {stage_id!r} requires one primary_call object")
        _reject_unknown_keys(
            call_raw,
            {"action", "target", "params", "reward_rules", "timeout_s"},
            f"Stage {stage_id!r} primary_call",
        )
        action = _required_string(call_raw, "action").upper()
        if action not in self.primitive_catalog:
            raise ValueError(f"Stage {stage_id!r} uses unknown primitive {action!r}")
        target = copy.deepcopy(call_raw.get("target"))
        params = copy.deepcopy(call_raw.get("params", {}))
        if not isinstance(params, Mapping):
            raise TypeError(f"Stage {stage_id!r} primary_call.params must be an object")
        _reject_numeric_spatial_data(target, path="target")
        _reject_numeric_spatial_data(params, path="params")
        spec = self.primitive_catalog.get(action)
        unknown_params = set(params) - set(spec.parameters)
        if unknown_params:
            raise ValueError(
                f"Stage {stage_id!r} uses unsupported {action} params: {sorted(unknown_params)}"
            )
        if target is not None and spec.symbolic_target and not isinstance(target, str):
            raise ValueError(f"Stage {stage_id!r} target must be symbolic text")
        if target is not None and not spec.symbolic_target:
            raise ValueError(f"Stage {stage_id!r} primitive {action} does not accept a target")
        rules_raw = call_raw.get("reward_rules")
        if not isinstance(rules_raw, list) or not rules_raw:
            raise ValueError(f"Stage {stage_id!r} primary_call requires a reward rule")
        rules = tuple(self._decode_rule(item, stage_id) for item in rules_raw)
        retry_budget = _budget(raw, "retry_budget")
        restage_budget = _budget(raw, "restage_budget")
        timeout_s = call_raw.get("timeout_s", 10.0)
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int | float)
            or timeout_s <= 0
        ):
            raise ValueError(f"Stage {stage_id!r} timeout_s must be a positive number")
        call = ToolCall(
            action=action,
            target=target,
            params=dict(params),
            reward_rules=rules,
            timeout_s=float(timeout_s),
        )
        return Stage(stage_id, objective, call, retry_budget, restage_budget)

    def _decode_rule(self, raw: Any, stage_id: str) -> StageRewardRule:
        if not isinstance(raw, Mapping):
            raise TypeError(f"Stage {stage_id!r} reward rules must be objects")
        _reject_unknown_keys(
            raw,
            {
                "predicate",
                "description",
                "target",
                "required_evidence",
                "allowed_tools",
            },
            f"Stage {stage_id!r} reward rule",
        )
        required_evidence = raw.get("required_evidence", ())
        allowed_tools = raw.get("allowed_tools", ())
        if not isinstance(required_evidence, list | tuple):
            raise TypeError(
                f"Stage {stage_id!r} reward rule required_evidence must be an array"
            )
        if not isinstance(allowed_tools, list | tuple):
            raise TypeError(f"Stage {stage_id!r} reward rule allowed_tools must be an array")
        rule = StageRewardRule(
            predicate=_required_string(raw, "predicate"),
            description=_required_string(raw, "description"),
            target=copy.deepcopy(raw.get("target")),
            required_evidence=tuple(required_evidence),
            allowed_tools=tuple(allowed_tools),
        )
        unknown = set(rule.allowed_tools) - self.reward_tools
        if unknown:
            raise ValueError(
                f"Stage {stage_id!r} uses unregistered reward tools: {sorted(unknown)}"
            )
        _reject_numeric_spatial_data(rule.target, path="reward_rule.target")
        return rule


def dataclass_replace_errors(request: StagePlanRequest, errors: list[str]) -> StagePlanRequest:
    return StagePlanRequest(
        task=request.task,
        state=request.state,
        primitive_catalog=request.primitive_catalog,
        reward_tools=request.reward_tools,
        completed_stages=request.completed_stages,
        failed_stage=request.failed_stage,
        failed_reward=request.failed_reward,
        failed_result=request.failed_result,
        validation_errors=tuple(errors),
    )


def _parse_json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise TypeError("StagePlanAgent.generate must return a JSON object")
    text = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1)
    decoded = json.loads(text)
    if not isinstance(decoded, Mapping):
        raise TypeError("agent JSON root must be an object")
    return decoded


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key!r} must be a non-empty string")
    return item.strip()


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{location} contains unknown fields: {sorted(str(key) for key in unknown)}"
        )


def _budget(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key, 0)
    if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1:
        raise ValueError(f"{key} must be integer 0 or 1")
    return item


def _reject_numeric_spatial_data(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in _SPATIAL_KEYS and _contains_number(item):
                raise ValueError(f"numeric pose/coordinate data is forbidden at {child}")
            _reject_numeric_spatial_data(item, path=child)
    elif isinstance(value, (list, tuple)):
        if value and all(
            isinstance(item, int | float) and not isinstance(item, bool) for item in value
        ):
            raise ValueError(f"numeric pose/coordinate arrays are forbidden at {path}")
        for index, item in enumerate(value):
            _reject_numeric_spatial_data(item, path=f"{path}[{index}]")


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(item) for item in value)
    return False


def _symbolic_value(value: Any) -> Any:
    spatially_clean = without_spatial_bindings(value)
    if isinstance(spatially_clean, Mapping):
        return {
            str(key): _symbolic_value(item)
            for key, item in spatially_clean.items()
            if str(key).lower() not in _RAW_SENSOR_KEYS
        }
    if isinstance(spatially_clean, list | tuple):
        return [_symbolic_value(item) for item in spatially_clean]
    return json_value(spatially_clean)


def _stage_dict(stage: Stage) -> dict[str, Any]:
    return {
        "id": stage.id,
        "objective": stage.objective,
        "primary_call": json_value(stage.primary_call),
        "retry_budget": stage.retry_budget,
        "restage_budget": stage.restage_budget,
    }


__all__ = [
    "AgenticStagePlanBuilder",
    "ModelStagePlanAgent",
    "PlannerStateView",
    "StagePlanAgent",
    "StagePlanBuildError",
    "StagePlanRequest",
]
