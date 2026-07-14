"""Structured physical primitives shared by planning and environment adapters."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from capx.planning.stage_reward import StageRewardRule

WorldState = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One planner-selected primitive invocation."""

    action: str
    target: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)
    reward_rules: tuple[StageRewardRule, ...] = ()
    # Deprecated constructor compatibility. New code should use reward_rules.
    postconditions: tuple[dict[str, Any], ...] = ()
    timeout_s: float = 10.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "reward_rules", tuple(self.reward_rules))
        object.__setattr__(self, "postconditions", tuple(self.postconditions))
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("ToolCall.action must be non-empty")
        if self.timeout_s <= 0:
            raise ValueError("ToolCall.timeout_s must be positive")
        if self.reward_rules and self.postconditions:
            raise ValueError("ToolCall cannot define both reward_rules and postconditions")
        if self.postconditions:
            object.__setattr__(
                self,
                "reward_rules",
                tuple(StageRewardRule.from_legacy(item) for item in self.postconditions),
            )
            object.__setattr__(self, "postconditions", ())
        if any(not isinstance(rule, StageRewardRule) for rule in self.reward_rules):
            raise TypeError("ToolCall.reward_rules entries must be StageRewardRule instances")
        predicates = [rule.predicate for rule in self.reward_rules]
        if len(predicates) != len(set(predicates)):
            raise ValueError("ToolCall reward rule predicates must be unique")


@dataclass(frozen=True)
class ToolResult:
    """Normalized result from a primitive executor."""

    ok: bool
    code: str
    message: str = ""
    updated_state: WorldState = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrimitiveSpec:
    """Agent-facing description and schema for one allowed primitive."""

    action: str
    description: str
    parameters: Mapping[str, str] = field(default_factory=dict)
    symbolic_target: bool = True

    def __post_init__(self) -> None:
        action = self.action.strip().upper()
        if not action or not self.description.strip():
            raise ValueError("PrimitiveSpec action and description must be non-empty")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "parameters", dict(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "description": self.description,
            "parameters": dict(self.parameters),
            "target": "symbolic" if self.symbolic_target else "none",
        }


class PrimitiveCatalog:
    """Immutable-by-interface catalog of primitives available to a builder agent."""

    def __init__(self, specs: tuple[PrimitiveSpec, ...] | list[PrimitiveSpec]) -> None:
        self._specs: dict[str, PrimitiveSpec] = {}
        for spec in specs:
            if spec.action in self._specs:
                raise ValueError(f"Duplicate primitive action: {spec.action}")
            self._specs[spec.action] = spec
        if not self._specs:
            raise ValueError("PrimitiveCatalog requires at least one primitive")

    def __contains__(self, action: object) -> bool:
        return isinstance(action, str) and action.strip().upper() in self._specs

    def __iter__(self) -> Iterator[PrimitiveSpec]:
        return iter(self._specs.values())

    def get(self, action: str) -> PrimitiveSpec:
        try:
            return self._specs[action.strip().upper()]
        except KeyError as exc:
            raise KeyError(f"Unknown primitive action: {action}") from exc

    def to_dict(self) -> tuple[dict[str, Any], ...]:
        return tuple(spec.to_dict() for spec in self._specs.values())


def libero_p0_primitive_catalog() -> PrimitiveCatalog:
    """Minimal symbolic primitive vocabulary for the LIBERO-PRO P0 workflow."""

    return PrimitiveCatalog(
        [
            PrimitiveSpec(
                "VLA",
                "Execute one local contact-rich manipulation phase from a text prompt.",
                {"prompt": "non-empty instruction", "max_chunks": "optional small integer"},
            ),
            PrimitiveSpec(
                "SET_GRIPPER",
                "Set the gripper to an open or closed state.",
                {"state": "open or close"},
            ),
            PrimitiveSpec("RELEASE", "Open the gripper to release a held object."),
        ]
    )


__all__ = [
    "PrimitiveCatalog",
    "PrimitiveSpec",
    "ToolCall",
    "ToolResult",
    "WorldState",
    "libero_p0_primitive_catalog",
]
