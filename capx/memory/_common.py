"""Private serialization and sanitization helpers shared by memory layers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

_PRIVILEGED_KEYS = {
    "ground_truth",
    "oracle_state",
    "privileged",
    "privileged_state",
    "sim_state",
    "simulator_state",
}
_SPATIAL_BINDING_KEYS = {
    "base_pose",
    "coordinates",
    "eef_pose",
    "fixture_pose",
    "object_pose",
    "pixel",
    "pixels",
    "pose",
    "position",
    "quat",
    "quaternion",
    "xy",
    "xyz",
}


def json_value(value: Any) -> Any:
    """Convert common runtime values into JSON-compatible data."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_value(tolist())
    return repr(value)


def without_privileged_fields(value: Any) -> Any:
    """Remove simulator-only fields before state reaches planner-facing Memory."""

    if isinstance(value, Mapping):
        return {
            str(key): without_privileged_fields(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVILEGED_KEYS and not str(key).startswith("_")
        }
    if isinstance(value, tuple | list):
        return type(value)(without_privileged_fields(item) for item in value)
    return copy.deepcopy(value)


def without_spatial_bindings(value: Any) -> Any:
    """Strip scene-specific coordinates before promoting experience to Task Memory."""

    if isinstance(value, Mapping):
        return {
            str(key): without_spatial_bindings(item)
            for key, item in value.items()
            if str(key).lower() not in _SPATIAL_BINDING_KEYS
        }
    if isinstance(value, tuple | list):
        return [without_spatial_bindings(item) for item in value]
    return json_value(value)
