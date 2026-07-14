"""Execute structured primitives through existing CaP-X tools.

This bridge lets the primitive-level trial reuse the same perception, motion, navigation, and
VLA tools that code-agent trials can call directly. It normalizes their heterogeneous return
values into ``ToolResult`` but does not claim that a returned tool satisfied the requested
physical post-condition. The verifier remains responsible for that decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from capx.planning.primitives import ToolCall, ToolResult, WorldState

ToolFunction = Callable[..., Any]


class ToolBackedPrimitiveExecutor:
    """Translate canonical primitives into existing ``ApiBase.functions()`` calls.

    The adapter currently supports the minimum bring-up vocabulary:

    - ``MOVE_EEF`` / ``MOVE_TO`` / ``MOVE_POSE`` -> ``goto_pose``
    - ``SET_GRIPPER`` -> ``open_gripper`` or ``close_gripper``
    - ``RELEASE`` -> ``open_gripper``
    - ``VLA`` -> ``vla_act``
    - ``NAVIGATE_TO`` -> ``navigate_to_pose``

    Additional canonical actions fall back to a same-name lowercase API function when one is
    registered. Missing functions produce a failed ``ToolResult`` rather than raising.
    """

    def __init__(self, functions: Mapping[str, ToolFunction]) -> None:
        self._functions = dict(functions)

    @classmethod
    def from_environment(cls, env: Any) -> ToolBackedPrimitiveExecutor:
        """Collect the existing tool functions exposed by an environment.

        TODO: Add a public immutable tool-function view to ``CodeExecutionEnvBase`` and stop
        reading its private ``_apis`` attribute here.
        """

        apis = getattr(env, "_apis", None)
        if not isinstance(apis, Mapping):
            raise TypeError("Expected an environment with an '_apis' mapping")

        functions: dict[str, ToolFunction] = {}
        for api_name, api in apis.items():
            api_functions = api.functions()
            for function_name, function in api_functions.items():
                if function_name in functions:
                    raise ValueError(
                        f"Duplicate tool function '{function_name}' exposed by API '{api_name}'"
                    )
                functions[function_name] = function
        return cls(functions)

    def execute(self, call: ToolCall, state: WorldState) -> ToolResult:
        """Execute one canonical call and normalize its return value."""

        del state  # Reserved for future grounding, frame checks, and safety validation.
        action = call.action.strip().upper()

        try:
            if action in {"MOVE_EEF", "MOVE_TO", "MOVE_POSE"}:
                return self._execute_move(call)
            if action == "SET_GRIPPER":
                return self._execute_set_gripper(call)
            if action == "RELEASE":
                return self._invoke("open_gripper", {}, default_code="released")
            if action == "VLA":
                return self._execute_vla(call)
            if action == "NAVIGATE_TO":
                return self._execute_navigation(call)

            # Naive escape hatch for new primitives whose canonical name already matches an
            # existing API function, for example ROTATE_WRIST -> rotate_wrist.
            # TODO: Replace this with explicit schemas and embodiment capability descriptors.
            function_name = action.lower()
            return self._invoke(
                function_name,
                dict(call.params),
                default_code=f"{function_name}_completed",
            )
        except (TypeError, ValueError) as exc:
            return ToolResult(
                ok=False,
                code="invalid_tool_arguments",
                message=str(exc),
                telemetry={"action": action},
            )

    def _execute_move(self, call: ToolCall) -> ToolResult:
        params = dict(call.params)
        position = params.pop("position", None)
        quaternion = params.pop("quaternion_wxyz", params.pop("orientation", None))

        pose = params.pop("pose", None)
        if pose is None and isinstance(call.target, (list, tuple)):
            pose = call.target
        if isinstance(call.target, Mapping):
            if position is None:
                position = call.target.get("position")
            if quaternion is None:
                quaternion = call.target.get("quaternion_wxyz")

        if pose is not None:
            if not isinstance(pose, (list, tuple)) or len(pose) != 7:
                raise ValueError("MOVE_EEF pose must be [x, y, z, qw, qx, qy, qz]")
            position = pose[:3]
            quaternion = pose[3:]

        if position is None or quaternion is None:
            return ToolResult(
                ok=False,
                code="unresolved_spatial_target",
                message=(
                    "MOVE_EEF requires position and quaternion_wxyz; symbolic targets must "
                    "be grounded before execution"
                ),
                telemetry={"target": call.target},
            )

        kwargs = {
            "position": position,
            "quaternion_wxyz": quaternion,
        }
        if "z_approach" in params:
            kwargs["z_approach"] = params.pop("z_approach")
        if params:
            raise ValueError(f"Unsupported MOVE_EEF parameters: {sorted(params)}")

        # TODO: Validate coordinate frames, workspace bounds, collision constraints, and pose
        # tolerance before invoking the current blocking controller.
        return self._invoke("goto_pose", kwargs, default_code="move_completed")

    def _execute_set_gripper(self, call: ToolCall) -> ToolResult:
        params = dict(call.params)
        gripper_state = params.pop("state", params.pop("gripper", call.target))
        if params:
            raise ValueError(f"Unsupported SET_GRIPPER parameters: {sorted(params)}")
        if not isinstance(gripper_state, str):
            raise ValueError("SET_GRIPPER requires state='open' or state='close'")

        normalized = gripper_state.strip().lower()
        if normalized == "open":
            return self._invoke("open_gripper", {}, default_code="gripper_opened")
        if normalized in {"close", "closed"}:
            return self._invoke("close_gripper", {}, default_code="gripper_closed")
        raise ValueError("SET_GRIPPER state must be 'open' or 'close'")

    def _execute_vla(self, call: ToolCall) -> ToolResult:
        params = dict(call.params)
        prompt = params.pop("prompt", None)
        if prompt is None and isinstance(call.target, str):
            prompt = call.target
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("VLA requires a non-empty prompt")

        kwargs: dict[str, Any] = {"prompt": prompt}
        if "max_chunks" in params:
            kwargs["max_chunks"] = params.pop("max_chunks")

        # Current vla_act accepts a Python callable as ``stop``. The structured primitive
        # path should use registered predicate names instead.
        # TODO: Resolve PredicateSpec through a predicate registry and pass a safe callable.
        params.pop("stop", None)
        if params:
            raise ValueError(f"Unsupported VLA parameters: {sorted(params)}")
        return self._invoke("vla_act", kwargs, default_code="vla_returned")

    def _execute_navigation(self, call: ToolCall) -> ToolResult:
        params = dict(call.params)
        pose_2d = params.pop("pose_2d", params.pop("pose", call.target))
        if params:
            raise ValueError(f"Unsupported NAVIGATE_TO parameters: {sorted(params)}")
        if not isinstance(pose_2d, (list, tuple)) or len(pose_2d) != 3:
            raise ValueError("NAVIGATE_TO requires [x, y, yaw]")

        # TODO: Introduce a common navigation schema with frame, tolerance, and timeout.
        return self._invoke(
            "navigate_to_pose",
            {"pose_2d": pose_2d},
            default_code="navigation_completed",
        )

    def _invoke(
        self,
        function_name: str,
        kwargs: dict[str, Any],
        *,
        default_code: str,
    ) -> ToolResult:
        function = self._functions.get(function_name)
        if function is None:
            return ToolResult(
                ok=False,
                code="unsupported_tool",
                message=f"No API function is registered for '{function_name}'",
                telemetry={"function": function_name},
            )

        # TODO: Enforce ToolCall.timeout_s around blocking API functions. The outer trial wall
        # clock budget currently provides only coarse protection.
        raw_result = function(**kwargs)
        return self._normalize_result(raw_result, default_code=default_code)

    @staticmethod
    def _normalize_result(raw_result: Any, *, default_code: str) -> ToolResult:
        if isinstance(raw_result, ToolResult):
            return raw_result
        if isinstance(raw_result, bool):
            return ToolResult(
                ok=raw_result,
                code=default_code if raw_result else "controller_failed",
                telemetry={"raw_result": raw_result},
            )
        if isinstance(raw_result, Mapping):
            payload = dict(raw_result)
            explicit_ok = payload.get("ok")
            ok = bool(explicit_ok) if explicit_ok is not None else True
            code = str(payload.get("code") or payload.get("status") or default_code)
            message = str(payload.get("message", ""))
            updated_state = payload.get("updated_state")
            return ToolResult(
                ok=ok,
                code=code,
                message=message,
                updated_state=dict(updated_state) if isinstance(updated_state, Mapping) else {},
                telemetry={"raw_result": payload},
            )

        # Existing blocking motion and gripper APIs generally return None on completion. This
        # only means the function returned without raising; the verifier must check the scene.
        return ToolResult(
            ok=True,
            code=default_code,
            telemetry={"raw_result": raw_result},
        )


__all__ = ["ToolBackedPrimitiveExecutor", "ToolFunction"]
