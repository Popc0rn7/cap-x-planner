"""Agent-facing primitive that exposes a frozen VLA as a retryable tool."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase

from .action_adapters import get_vla_action_adapter
from .backend import AdaptedVlaBackend, HttpVlaBackend, VlaBackend
from .model import MockVlaModel, RemoteVlaModel, VlaModel
from .model_adapters import get_vla_model_adapter
from .transport import HttpVlaTransport

StopCondition = Callable[[dict[str, Any]], bool]


class VlaPrimitiveApi(ApiBase):
    """Run a frozen VLA in short, planner-selected bursts."""

    def __init__(
        self,
        env: BaseEnv,
        backend: VlaBackend | None = None,
        default_max_chunks: int = 1,
    ) -> None:
        super().__init__(env)
        if default_max_chunks < 1:
            raise ValueError("default_max_chunks must be at least 1")
        self.default_max_chunks = default_max_chunks
        self._backend = backend or HttpVlaBackend(
            action_adapter=get_vla_action_adapter("identity", env)
        )

    def functions(self) -> dict[str, Any]:
        return {"vla_act": self.vla_act}

    def vla_act(
        self,
        prompt: str,
        max_chunks: int | None = None,
        stop: StopCondition | None = None,
    ) -> dict[str, Any]:
        """Invoke a frozen VLA for a local contact-rich operation.

        Use this for grasping, insertion, articulated-object interaction, or
        other local contact phases. Use analytic APIs for staging, transport,
        navigation, release, and recovery.

        Args:
            prompt: Local instruction for the VLA, such as ``"grasp the black bowl"``.
            max_chunks: Maximum action chunks to execute before returning control.
            stop: Optional callable receiving the latest observation. Return True
                to stop early after an action.

        Returns:
            Diagnostic dictionary with status, chunk/action counts, task success,
            and backend metadata. A returned status is not a grasp-success oracle;
            inspect the next observation before transport or retry.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        chunk_budget = self.default_max_chunks if max_chunks is None else max_chunks
        if chunk_budget < 1:
            raise ValueError("max_chunks must be at least 1")
        execute_action = getattr(self._env, "execute_policy_action", None)
        if not callable(execute_action):
            raise RuntimeError(
                f"{type(self._env).__name__} does not support native VLA actions; "
                "implement execute_policy_action(action) first"
            )

        chunks_executed = 0
        actions_executed = 0
        stopped_by = "chunk_budget"
        metadata: dict[str, Any] = {}
        self._log_step("vla_act", f"Running VLA primitive: {prompt}", highlight=True)

        for _ in range(chunk_budget):
            observation = self._env.get_observation()
            chunk = self._backend.predict(observation, prompt.strip())
            chunks_executed += 1
            metadata = chunk.metadata

            for action in chunk.actions:
                transition = execute_action(action)
                actions_executed += 1
                observation = self._env.get_observation()
                if bool(transition.get("done", False)):
                    stopped_by = "environment_done"
                    break
                if stop is not None and stop(observation):
                    stopped_by = "stop_condition"
                    break
            else:
                if chunk.done:
                    stopped_by = "backend_done"
                    break
                continue
            break

        task_completed = bool(self._env.task_completed())
        status = "task_completed" if task_completed else stopped_by
        result = {
            "status": status,
            "chunks_executed": chunks_executed,
            "actions_executed": actions_executed,
            "task_completed": task_completed,
            "metadata": metadata,
        }
        self._log_step_update(text=f"VLA primitive returned: {status}")
        return result


def _adapter_spec(spec: str | dict[str, Any] | None, default: str) -> tuple[str, dict[str, Any]]:
    if spec is None:
        return default, {}
    if isinstance(spec, str):
        return spec, {}
    if not isinstance(spec, dict):
        raise TypeError("adapter config must be a name or mapping")
    config = dict(spec)
    name = config.pop("name", default)
    if not isinstance(name, str):
        raise TypeError("adapter name must be a string")
    return name, config


def _build_model(config: dict[str, Any]) -> VlaModel:
    model_config = dict(config)
    model_type = model_config.pop("type", "http")
    if model_type == "mock":
        return MockVlaModel(**model_config)
    if model_type != "http":
        raise ValueError(f"Unsupported VLA model type: {model_type}")

    adapter_name, adapter_config = _adapter_spec(model_config.pop("adapter", None), "native")
    transport_config = dict(model_config.pop("transport", {}))
    if model_config:
        raise ValueError(f"Unknown HTTP VLA model config keys: {sorted(model_config)}")
    return RemoteVlaModel(
        adapter=get_vla_model_adapter(adapter_name, **adapter_config),
        transport=HttpVlaTransport(**transport_config),
    )


def build_vla_primitive_api(
    env: BaseEnv,
    config: dict[str, Any] | None = None,
) -> VlaPrimitiveApi:
    """Build a VLA primitive after the low-level environment exists."""
    config = dict(config or {})
    action_name, action_config = _adapter_spec(config.get("action_adapter"), "identity")

    primitive_config = dict(config.get("primitive", {}))
    unknown = set(config) - {"model", "action_adapter", "primitive"}
    if unknown:
        raise ValueError(f"Unknown VLA primitive config keys: {sorted(unknown)}")

    backend = AdaptedVlaBackend(
        model=_build_model(dict(config.get("model", {}))),
        action_adapter=get_vla_action_adapter(action_name, env, **action_config),
    )
    return VlaPrimitiveApi(env=env, backend=backend, **primitive_config)
