"""Reference model adapter for the baseline CaP-X HTTP contract."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from ..types import ModelActionChunk
from .base import VlaModelAdapter


def _encode_rgb(image: np.ndarray) -> str:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_npy(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def serialize_native_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Serialize the common RGB-D and proprioceptive observation contract."""
    payload: dict[str, Any] = {"cameras": {}, "proprioception": {}}
    for name, value in observation.items():
        if not isinstance(value, dict):
            continue
        images = value.get("images")
        if isinstance(images, dict) and images.get("rgb") is not None:
            camera = {"rgb_png_base64": _encode_rgb(images["rgb"])}
            if images.get("depth") is not None:
                camera["depth_npy_base64"] = _encode_npy(images["depth"])
            for key in ("intrinsics", "pose_mat"):
                if value.get(key) is not None:
                    camera[key] = np.asarray(value[key]).tolist()
            payload["cameras"][name] = camera

    for key in ("robot_cartesian_pos", "robot_joint_pos"):
        value = observation.get(key)
        if value is not None:
            payload["proprioception"][key] = np.asarray(value).tolist()
    return payload


class NativeModelAdapter(VlaModelAdapter):
    """Decode services that already return an ``actions`` array."""

    def build_request(
        self, observation: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        return {"prompt": prompt, "observation": serialize_native_observation(observation)}

    def decode_response(self, response: dict[str, Any]) -> ModelActionChunk:
        if "actions" not in response:
            raise ValueError("VLA response is missing required 'actions' field")
        return ModelActionChunk(
            actions=np.asarray(response["actions"], dtype=np.float32),
            done=bool(response.get("done", False)),
            metadata=dict(response.get("metadata", {})),
        )
