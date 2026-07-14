"""Global Memory shared read-only by all tasks and trials."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from capx.memory._common import json_value


@dataclass(frozen=True)
class GlobalMemorySnapshot:
    """Task-independent knowledge visible to every planner trial."""

    rules: tuple[dict[str, Any], ...] = ()
    failure_models: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


class GlobalMemory(Protocol):
    """Read-only source of knowledge shared by all tasks."""

    def read(self) -> GlobalMemorySnapshot: ...


class StaticGlobalMemory:
    """P0 Global Memory backed by an immutable in-process snapshot."""

    def __init__(
        self,
        *,
        rules: tuple[dict[str, Any], ...] = (),
        failure_models: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self._snapshot = GlobalMemorySnapshot(
            rules=tuple(copy.deepcopy(rules)),
            failure_models=tuple(copy.deepcopy(failure_models)),
        )

    def read(self) -> GlobalMemorySnapshot:
        return copy.deepcopy(self._snapshot)

    @classmethod
    def from_json(cls, path: str | Path) -> StaticGlobalMemory:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            rules=tuple(payload.get("rules", ())),
            failure_models=tuple(payload.get("failure_models", ())),
        )


__all__ = ["GlobalMemory", "GlobalMemorySnapshot", "StaticGlobalMemory"]
