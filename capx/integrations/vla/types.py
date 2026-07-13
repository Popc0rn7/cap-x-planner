"""Shared value types for VLA integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _normalize_actions(actions: np.ndarray) -> np.ndarray:
    normalized = np.asarray(actions, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized[None, :]
    if normalized.ndim != 2 or normalized.shape[0] == 0 or normalized.shape[1] == 0:
        raise ValueError("VLA actions must have shape (time, action_dim)")
    if not np.isfinite(normalized).all():
        raise ValueError("VLA actions must contain only finite values")
    return normalized


@dataclass(frozen=True)
class ModelActionChunk:
    """Actions decoded from one model, before environment adaptation."""

    actions: np.ndarray
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", _normalize_actions(self.actions))


@dataclass(frozen=True)
class NativeActionChunk:
    """Actions converted to the target environment's native controller space."""

    actions: np.ndarray
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", _normalize_actions(self.actions))


# Compatibility name used by the first primitive implementation.
VlaActionChunk = NativeActionChunk
