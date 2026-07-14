"""Production assembly for the agentic fine-grained manipulation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from capx.envs.primitive_executor import ToolBackedPrimitiveExecutor
from capx.envs.trial_fine_grained import FineGrainedTrialComponents
from capx.memory import HierarchicalTrialMemory
from capx.memory.global_memory import GlobalMemory
from capx.memory.task_memory import TaskMemory
from capx.planning.agentic_builder import AgenticStagePlanBuilder, ModelStagePlanAgent
from capx.planning.primitives import libero_p0_primitive_catalog
from capx.planning.recovery import RuleBasedRecoveryPolicy
from capx.planning.stage_planner import StagePlanner
from capx.planning.stage_reward import (
    AgenticStageRewardVerifier,
    ModelStageRewardCodeAgent,
    ReadOnlyStageRewardSandbox,
    ReadOnlyVerificationTool,
    VerificationToolRegistry,
)


def build_fine_grained_components(
    *,
    env: Any,
    planner_model_args: Any,
    reward_model_args: Any | None = None,
    task_memory: TaskMemory | None = None,
    global_memory: GlobalMemory | None = None,
    budgets: Mapping[str, int | float | None] | None = None,
) -> FineGrainedTrialComponents:
    """Assemble the P0 LIBERO workflow with independent planner/reward model settings."""

    catalog = libero_p0_primitive_catalog()
    registry = VerificationToolRegistry(
        (
            ReadOnlyVerificationTool(
                name="read_gripper_state",
                description="Read the current symbolic gripper state without changing the scene.",
                handler=lambda: _read_gripper_state(env),
            ),
        )
    )
    builder = AgenticStagePlanBuilder(
        ModelStagePlanAgent(planner_model_args),
        catalog,
        reward_tools=("read_gripper_state",),
    )
    verifier = AgenticStageRewardVerifier(
        ModelStageRewardCodeAgent(reward_model_args or planner_model_args),
        ReadOnlyStageRewardSandbox(registry),
    )
    return FineGrainedTrialComponents(
        planner=StagePlanner(builder),
        tool_executor=ToolBackedPrimitiveExecutor.from_environment(env),
        verifier=verifier,
        recovery_policy=RuleBasedRecoveryPolicy(),
        memory=HierarchicalTrialMemory(
            task_memory=task_memory,
            global_memory=global_memory,
            budgets=budgets,
        ),
    )


def _read_gripper_state(env: Any) -> dict[str, Any]:
    low_level = getattr(env, "low_level_env", env)
    observation = low_level.get_observation()
    if not isinstance(observation, Mapping):
        return {"state": "unknown"}
    explicit = observation.get("gripper_state")
    if explicit is not None:
        return {"state": explicit}
    robot = observation.get("robot")
    if isinstance(robot, Mapping):
        for key in ("gripper_state", "gripper", "gripper_open"):
            if key in robot:
                return {"state": robot[key]}
    return {"state": "unknown"}


__all__ = ["build_fine_grained_components"]
