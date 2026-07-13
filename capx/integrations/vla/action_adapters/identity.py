"""Pass-through adapter for models already emitting environment-native actions."""

from __future__ import annotations

from typing import Any

from ..types import ModelActionChunk, NativeActionChunk
from .base import VlaActionAdapter


class IdentityActionAdapter(VlaActionAdapter):
    """Copy decoded model actions directly into a native action chunk."""

    def convert(
        self,
        chunk: ModelActionChunk,
        observation: dict[str, Any],
    ) -> NativeActionChunk:
        del observation
        return NativeActionChunk(
            actions=chunk.actions,
            done=chunk.done,
            metadata=chunk.metadata,
        )
