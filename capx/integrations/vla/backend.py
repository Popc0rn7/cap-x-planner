"""Composition layer for model adapters and inference transports."""

from __future__ import annotations

from typing import Any, Protocol

from .action_adapters import VlaActionAdapter
from .model import RemoteVlaModel, VlaModel
from .model_adapters import NativeModelAdapter, VlaModelAdapter
from .transport import HttpVlaTransport
from .types import NativeActionChunk, VlaActionChunk


class VlaBackend(Protocol):
    """Model-independent interface consumed by ``VlaPrimitiveApi``."""

    def predict(self, observation: dict[str, Any], prompt: str) -> VlaActionChunk:
        """Predict an environment-action chunk from an observation and prompt."""


class AdaptedVlaBackend:
    """Compose a chunk-producing model with an environment action adapter."""

    def __init__(
        self,
        model: VlaModel,
        action_adapter: VlaActionAdapter,
    ) -> None:
        self.model = model
        self.action_adapter = action_adapter

    def predict(self, observation: dict[str, Any], prompt: str) -> VlaActionChunk:
        model_chunk = self.model.predict(observation, prompt)
        return self.action_adapter.convert(model_chunk, observation)


class HttpVlaBackend(AdaptedVlaBackend):
    """Convenience backend using a registered model adapter over HTTP."""

    def __init__(
        self,
        action_adapter: VlaActionAdapter,
        base_url: str | None = None,
        timeout: float = 120.0,
        model_adapter: VlaModelAdapter | None = None,
    ) -> None:
        super().__init__(
            model=RemoteVlaModel(
                adapter=model_adapter or NativeModelAdapter(),
                transport=HttpVlaTransport(base_url=base_url, timeout=timeout),
            ),
            action_adapter=action_adapter,
        )


__all__ = [
    "AdaptedVlaBackend",
    "HttpVlaBackend",
    "NativeActionChunk",
    "VlaActionChunk",
    "VlaBackend",
]
