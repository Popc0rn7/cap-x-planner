"""Construct trial executors from top-level experiment configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from capx.envs.trial_base import TrialExecutor

TrialExecutorFactory = Callable[[dict[str, Any]], TrialExecutor]


def _build_code_agent(config: dict[str, Any]) -> TrialExecutor:
    # Keep code-agent model and image dependencies out of fine-grained-only processes.
    from capx.envs.trial import CodeAgentTrialExecutor

    return CodeAgentTrialExecutor(config)


def _build_fine_grained(config: dict[str, Any]) -> TrialExecutor:
    from capx.envs.trial_fine_grained import FineGrainedTrialExecutor

    return FineGrainedTrialExecutor(config)


_TRIAL_EXECUTOR_FACTORIES: dict[str, TrialExecutorFactory] = {
    "code_agent": _build_code_agent,
    "fine_grained": _build_fine_grained,
}


def register_trial_executor(
    name: str,
    factory: TrialExecutorFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an executor without adding implementation branches to the runner."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("trial executor name must be non-empty")
    if normalized in _TRIAL_EXECUTOR_FACTORIES and not replace:
        raise ValueError(f"trial executor '{normalized}' is already registered")
    _TRIAL_EXECUTOR_FACTORIES[normalized] = factory


def build_trial_executor(spec: str | Mapping[str, Any] | None) -> TrialExecutor:
    """Build a configured executor, defaulting existing configs to ``code_agent``."""

    if spec is None:
        name = "code_agent"
        executor_config: dict[str, Any] = {}
    elif isinstance(spec, str):
        name = spec.strip().lower()
        executor_config = {}
    elif isinstance(spec, Mapping):
        executor_config = dict(spec)
        name = str(executor_config.pop("type", "code_agent")).strip().lower()
    else:
        raise TypeError("trial_executor must be a name or mapping")

    try:
        factory = _TRIAL_EXECUTOR_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRIAL_EXECUTOR_FACTORIES))
        raise ValueError(
            f"Unknown trial executor '{name}'. Available: {available}"
        ) from exc
    return factory(executor_config)


__all__ = ["build_trial_executor", "register_trial_executor"]
