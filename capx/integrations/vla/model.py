"""VLA model invocation boundary used by real and mock inference."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .model_adapters import VlaModelAdapter
from .transport import VlaInferenceTransport
from .types import ModelActionChunk


class VlaModel(Protocol):
    """Return decoded model-action chunks, regardless of inference runtime."""

    def predict(self, observation: dict[str, Any], prompt: str) -> ModelActionChunk:
        """Generate one model-action chunk."""


class RemoteVlaModel:
    """Invoke a remote model and decode its response with a model adapter."""

    def __init__(
        self,
        adapter: VlaModelAdapter,
        transport: VlaInferenceTransport,
    ) -> None:
        self.adapter = adapter
        self.transport = transport

    def predict(self, observation: dict[str, Any], prompt: str) -> ModelActionChunk:
        request = self.adapter.build_request(observation, prompt)
        response = self.transport.infer(request)
        return self.adapter.decode_response(response)


class MockVlaModel:
    """Deterministic model that returns its configured actions as one chunk."""

    def __init__(
        self,
        actions: list[list[float]] | np.ndarray,
        done: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        template = ModelActionChunk(
            actions=actions,
            done=done,
            metadata=dict(metadata or {"model": "mock"}),
        )
        self.actions = template.actions
        self.done = template.done
        self.metadata = template.metadata
        self.last_observation: dict[str, Any] | None = None
        self.last_prompt: str | None = None

    def predict(self, observation: dict[str, Any], prompt: str) -> ModelActionChunk:
        self.last_observation = observation
        self.last_prompt = prompt
        return ModelActionChunk(
            actions=self.actions.copy(),
            done=self.done,
            metadata=dict(self.metadata),
        )
