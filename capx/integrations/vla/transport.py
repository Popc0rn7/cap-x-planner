"""Inference transports without model-specific assumptions."""

from __future__ import annotations

import os
from typing import Any, Protocol

import requests


class VlaInferenceTransport(Protocol):
    """Transport arbitrary adapter-produced inference payloads."""

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send one inference request and return the decoded response."""


class HttpVlaTransport:
    """JSON-over-HTTP transport for a separately hosted VLA model."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (
            base_url or os.environ.get("CAPX_VLA_SERVICE_URL", "http://127.0.0.1:8123")
        ).rstrip("/")
        self.timeout = timeout

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/predict", json=request, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("VLA service response must be a JSON object")
        return result

