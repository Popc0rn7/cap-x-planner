"""Built-in VLA environment action adapters."""

from .base import (
    VlaActionAdapter,
    get_vla_action_adapter,
    list_vla_action_adapters,
    register_vla_action_adapter,
)
from .identity import IdentityActionAdapter

register_vla_action_adapter("identity", IdentityActionAdapter)

__all__ = [
    "IdentityActionAdapter",
    "VlaActionAdapter",
    "get_vla_action_adapter",
    "list_vla_action_adapters",
    "register_vla_action_adapter",
]
