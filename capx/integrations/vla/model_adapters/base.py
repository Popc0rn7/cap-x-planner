"""Model-side adapter interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..types import ModelActionChunk


class VlaModelAdapter(ABC):
    """Translate between CaP-X observations and one VLA model contract."""

    @abstractmethod
    def build_request(
        self, observation: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """Convert a CaP-X observation into the model's inference request."""

    @abstractmethod
    def decode_response(self, response: dict[str, Any]) -> ModelActionChunk:
        """Decode a model response without applying environment control semantics."""


_MODEL_ADAPTER_FACTORIES: dict[str, Callable[..., VlaModelAdapter]] = {}


def register_vla_model_adapter(
    name: str, factory: Callable[..., VlaModelAdapter], *, replace: bool = False
) -> None:
    """Register a model adapter factory."""
    if not name or name != name.lower():
        raise ValueError("VLA model adapter names must be non-empty lowercase strings")
    if name in _MODEL_ADAPTER_FACTORIES and not replace:
        raise ValueError(f"VLA model adapter '{name}' is already registered")
    _MODEL_ADAPTER_FACTORIES[name] = factory


def get_vla_model_adapter(name: str, **config: Any) -> VlaModelAdapter:
    """Construct a registered model adapter."""
    try:
        return _MODEL_ADAPTER_FACTORIES[name](**config)
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_ADAPTER_FACTORIES)) or "none"
        raise KeyError(f"Unknown VLA model adapter '{name}'. Available: {available}") from exc


def list_vla_model_adapters() -> list[str]:
    return sorted(_MODEL_ADAPTER_FACTORIES)
