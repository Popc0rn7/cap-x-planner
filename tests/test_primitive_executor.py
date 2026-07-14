from __future__ import annotations

from typing import Any

from capx.envs.primitive_executor import ToolBackedPrimitiveExecutor
from capx.envs.trial_fine_grained import ToolCall


class RecordingFunctions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def goto_pose(self, **kwargs: Any) -> None:
        self.calls.append(("goto_pose", kwargs))

    def open_gripper(self, **kwargs: Any) -> None:
        self.calls.append(("open_gripper", kwargs))

    def vla_act(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("vla_act", kwargs))
        return {"status": "chunk_budget", "actions_executed": 8}

    def mapping(self):
        return {
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "vla_act": self.vla_act,
        }


def test_executor_maps_move_and_release_to_existing_tool_functions() -> None:
    functions = RecordingFunctions()
    executor = ToolBackedPrimitiveExecutor(functions.mapping())

    move_result = executor.execute(
        ToolCall(
            action="MOVE_EEF",
            params={"pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]},
        ),
        state={},
    )
    release_result = executor.execute(ToolCall(action="RELEASE"), state={})

    assert move_result.ok is True
    assert release_result.ok is True
    assert functions.calls == [
        (
            "goto_pose",
            {
                "position": [0.1, 0.2, 0.3],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        ),
        ("open_gripper", {}),
    ]


def test_executor_preserves_vla_stop_reason_as_tool_code_not_physical_success() -> None:
    functions = RecordingFunctions()
    executor = ToolBackedPrimitiveExecutor(functions.mapping())

    result = executor.execute(
        ToolCall(
            action="VLA",
            target="cup",
            params={"prompt": "grasp the cup", "max_chunks": 2},
        ),
        state={},
    )

    assert result.ok is True
    assert result.code == "chunk_budget"
    assert result.telemetry["raw_result"]["actions_executed"] == 8
    assert functions.calls == [
        ("vla_act", {"prompt": "grasp the cup", "max_chunks": 2})
    ]


def test_executor_rejects_symbolic_move_target_until_grounded() -> None:
    executor = ToolBackedPrimitiveExecutor({})

    result = executor.execute(
        ToolCall(action="MOVE_EEF", target="pregrasp(cup)"),
        state={},
    )

    assert result.ok is False
    assert result.code == "unresolved_spatial_target"


def test_executor_reports_missing_capability_without_raising() -> None:
    executor = ToolBackedPrimitiveExecutor({})

    result = executor.execute(
        ToolCall(action="ROTATE_WRIST", params={"target_yaw": 0.5}),
        state={},
    )

    assert result.ok is False
    assert result.code == "unsupported_tool"
