from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from capx.integrations.vla.action_adapters import VlaActionAdapter, list_vla_action_adapters
from capx.integrations.vla.backend import AdaptedVlaBackend, VlaActionChunk
from capx.integrations.vla.model import MockVlaModel, RemoteVlaModel
from capx.integrations.vla.model_adapters import VlaModelAdapter, list_vla_model_adapters
from capx.integrations.vla.primitive import VlaPrimitiveApi, build_vla_primitive_api
from capx.integrations.vla.types import ModelActionChunk, NativeActionChunk


class ScriptedBackend:
    def __init__(self, chunks: list[VlaActionChunk]) -> None:
        self.chunks = iter(chunks)
        self.prompts: list[str] = []

    def predict(self, observation: dict[str, Any], prompt: str) -> VlaActionChunk:
        self.prompts.append(prompt)
        return next(self.chunks)


class FakePolicyEnv:
    def __init__(self, *, done_after: int | None = None) -> None:
        self.actions: list[np.ndarray] = []
        self.done_after = done_after

    def get_observation(self) -> dict[str, Any]:
        return {"action_count": len(self.actions)}

    def execute_policy_action(self, action: np.ndarray) -> dict[str, Any]:
        self.actions.append(np.asarray(action))
        done = self.done_after is not None and len(self.actions) >= self.done_after
        return {"reward": float(done), "done": done, "info": {}}

    def task_completed(self) -> bool:
        return self.done_after is not None and len(self.actions) >= self.done_after


class RecordingTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return self.response


class OffsetModelAdapter(VlaModelAdapter):
    def build_request(
        self, observation: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        return {"instruction": prompt, "state": observation["state"]}

    def decode_response(self, response: dict[str, Any]) -> ModelActionChunk:
        return ModelActionChunk(
            actions=np.asarray(response["model_actions"]),
            metadata={"model_adapter": "offset"},
        )


class OffsetActionAdapter(VlaActionAdapter):
    def convert(
        self,
        chunk: ModelActionChunk,
        observation: dict[str, Any],
    ) -> NativeActionChunk:
        return NativeActionChunk(
            actions=chunk.actions + observation["action_offset"],
            done=chunk.done,
            metadata={**chunk.metadata, "action_adapter": "offset"},
        )


def test_action_chunk_normalizes_single_action() -> None:
    chunk = VlaActionChunk(actions=np.array([0.1, -0.2, 1.0]))
    assert chunk.actions.shape == (1, 3)


def test_adapted_backend_delegates_model_specific_contract() -> None:
    transport = RecordingTransport({"model_actions": [[1.0, 2.0]]})
    env = FakePolicyEnv()
    backend = AdaptedVlaBackend(
        model=RemoteVlaModel(adapter=OffsetModelAdapter(), transport=transport),
        action_adapter=OffsetActionAdapter(env),  # type: ignore[arg-type]
    )

    chunk = backend.predict({"state": [3.0], "action_offset": 10.0}, "move")

    assert transport.requests == [{"instruction": "move", "state": [3.0]}]
    np.testing.assert_allclose(chunk.actions, [[11.0, 12.0]])
    assert chunk.metadata == {
        "model_adapter": "offset",
        "action_adapter": "offset",
    }
    assert "native" in list_vla_model_adapters()
    assert "identity" in list_vla_action_adapters()


def test_structured_vla_config_builds_after_env_exists() -> None:
    env = FakePolicyEnv()
    api = build_vla_primitive_api(
        env,  # type: ignore[arg-type]
        config={
            "model": {
                "type": "http",
                "adapter": {"name": "native"},
                "transport": {"base_url": "http://vla.test"},
            },
            "action_adapter": {"name": "identity"},
            "primitive": {"default_max_chunks": 2},
        },
    )

    assert api.default_max_chunks == 2
    assert api._backend.model.transport.base_url == "http://vla.test"  # type: ignore[attr-defined]


def test_mock_model_returns_configured_action_chunk() -> None:
    actions = [[0.1, 0.2, -1.0], [0.3, 0.4, 1.0]]
    model = MockVlaModel(actions=actions)

    chunk = model.predict({"state": "test"}, "mock prompt")

    np.testing.assert_allclose(chunk.actions, actions)
    assert chunk.done is True
    assert chunk.metadata == {"model": "mock"}
    assert model.last_observation == {"state": "test"}
    assert model.last_prompt == "mock prompt"


def test_mock_model_runs_through_primitive() -> None:
    env = FakePolicyEnv()
    actions = [[0.1, 0.2, -1.0], [0.3, 0.4, 1.0]]
    api = build_vla_primitive_api(
        env,  # type: ignore[arg-type]
        config={
            "action_adapter": "identity",
            "model": {"type": "mock", "actions": actions},
        },
    )

    result = api.vla_act("mock action")

    np.testing.assert_allclose(env.actions, actions)
    assert result["status"] == "backend_done"
    assert result["actions_executed"] == 2


def test_vla_act_executes_chunks_until_backend_done() -> None:
    env = FakePolicyEnv()
    backend = ScriptedBackend(
        [
            VlaActionChunk(actions=np.zeros((2, 4))),
            VlaActionChunk(actions=np.ones((1, 4)), done=True, metadata={"attempt": 2}),
        ]
    )
    api = VlaPrimitiveApi(env, backend=backend)  # type: ignore[arg-type]

    result = api.vla_act("grasp the bowl", max_chunks=3)

    assert result == {
        "status": "backend_done",
        "chunks_executed": 2,
        "actions_executed": 3,
        "task_completed": False,
        "metadata": {"attempt": 2},
    }
    assert backend.prompts == ["grasp the bowl", "grasp the bowl"]


def test_vla_act_stops_on_predicate() -> None:
    env = FakePolicyEnv()
    backend = ScriptedBackend([VlaActionChunk(actions=np.zeros((3, 4)))])
    api = VlaPrimitiveApi(env, backend=backend)  # type: ignore[arg-type]

    result = api.vla_act("touch target", stop=lambda obs: obs["action_count"] == 1)

    assert result["status"] == "stop_condition"
    assert result["actions_executed"] == 1


def test_vla_act_reports_environment_completion() -> None:
    env = FakePolicyEnv(done_after=2)
    backend = ScriptedBackend([VlaActionChunk(actions=np.zeros((4, 4)))])
    api = VlaPrimitiveApi(env, backend=backend)  # type: ignore[arg-type]

    result = api.vla_act("complete task")

    assert result["status"] == "task_completed"
    assert result["actions_executed"] == 2


def test_vla_act_rejects_invalid_requests() -> None:
    env = FakePolicyEnv()
    api = VlaPrimitiveApi(env, backend=ScriptedBackend([]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        api.vla_act(" ")
    with pytest.raises(ValueError, match="at least 1"):
        api.vla_act("grasp", max_chunks=0)
