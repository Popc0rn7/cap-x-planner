"""Environment action adapter interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from capx.envs.base import BaseEnv

from ..types import ModelActionChunk, NativeActionChunk


class VlaActionAdapter(ABC):
    """Convert decoded model actions into one environment's native action space."""

    def __init__(self, env: BaseEnv) -> None:
        self.env = env

    @abstractmethod
    def convert(
        self,
        chunk: ModelActionChunk,
        observation: dict[str, Any],
    ) -> NativeActionChunk:
        """Convert one model action chunk into native controller actions."""


_ACTION_ADAPTER_FACTORIES: dict[str, Callable[..., VlaActionAdapter]] = {}


def register_vla_action_adapter(
    name: str, factory: Callable[..., VlaActionAdapter], *, replace: bool = False
) -> None:
    """Register an environment action adapter factory."""
    if not name or name != name.lower():
        raise ValueError("VLA action adapter names must be non-empty lowercase strings")
    if name in _ACTION_ADAPTER_FACTORIES and not replace:
        raise ValueError(f"VLA action adapter '{name}' is already registered")
    _ACTION_ADAPTER_FACTORIES[name] = factory


def get_vla_action_adapter(
    name: str, env: BaseEnv, **config: Any
) -> VlaActionAdapter:
    """Construct a registered environment action adapter."""
    try:
        return _ACTION_ADAPTER_FACTORIES[name](env=env, **config)
    except KeyError as exc:
        available = ", ".join(sorted(_ACTION_ADAPTER_FACTORIES)) or "none"
        raise KeyError(f"Unknown VLA action adapter '{name}'. Available: {available}") from exc


def list_vla_action_adapters() -> list[str]:
    return sorted(_ACTION_ADAPTER_FACTORIES)
