"""Task Memory shared by all trials of one task."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from capx.memory._common import json_value


@dataclass(frozen=True)
class TaskMemoryRecord:
    """Successful symbolic experience for one task across its trials."""

    task: str
    audit: dict[str, Any] = field(default_factory=dict)
    commands: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskMemoryRecord:
        return cls(
            task=str(payload["task"]),
            audit=copy.deepcopy(dict(payload.get("audit", {}))),
            commands=tuple(copy.deepcopy(payload.get("commands", ()))),
        )


class TaskMemory(Protocol):
    """Store scoped to one task identity but shared across that task's trials."""

    def get(self, task: str) -> TaskMemoryRecord | None: ...

    def put(self, record: TaskMemoryRecord) -> None: ...


class InMemoryTaskMemory:
    """Process-local Task Memory useful for tests and single-worker runs."""

    def __init__(self, records: tuple[TaskMemoryRecord, ...] = ()) -> None:
        self._records = {record.task: copy.deepcopy(record) for record in records}

    def get(self, task: str) -> TaskMemoryRecord | None:
        record = self._records.get(task)
        return copy.deepcopy(record) if record is not None else None

    def put(self, record: TaskMemoryRecord) -> None:
        self._records[record.task] = copy.deepcopy(record)


class FileTaskMemory:
    """Task Memory stored as one audit JSON and command JSONL per task."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get(self, task: str) -> TaskMemoryRecord | None:
        task_directory = self._task_directory(task)
        audit_path = task_directory / "audit.json"
        commands_path = task_directory / "commands.jsonl"
        if not audit_path.exists() or not commands_path.exists():
            return None
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        stored_task = str(audit_payload.pop("task"))
        if stored_task != task:
            raise ValueError("Task Memory identity does not match its storage directory")
        commands = tuple(
            json.loads(line)
            for line in commands_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return TaskMemoryRecord(task=stored_task, audit=audit_payload, commands=commands)

    def put(self, record: TaskMemoryRecord) -> None:
        task_directory = self._task_directory(record.task)
        task_directory.mkdir(parents=True, exist_ok=True)
        audit_path = task_directory / "audit.json"
        commands_path = task_directory / "commands.jsonl"
        audit_temporary = task_directory / "audit.json.tmp"
        commands_temporary = task_directory / "commands.jsonl.tmp"
        audit_temporary.write_text(
            json.dumps(
                {"task": record.task, **json_value(record.audit)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        command_lines = [
            json.dumps(json_value(command), sort_keys=True) for command in record.commands
        ]
        commands_temporary.write_text(
            "\n".join(command_lines) + ("\n" if command_lines else ""),
            encoding="utf-8",
        )
        commands_temporary.replace(commands_path)
        audit_temporary.replace(audit_path)

    def _task_directory(self, task: str) -> Path:
        task_key = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
        return self.root / task_key


__all__ = ["FileTaskMemory", "InMemoryTaskMemory", "TaskMemory", "TaskMemoryRecord"]
