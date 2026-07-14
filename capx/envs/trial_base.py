"""Common contract for executing one environment trial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from capx.utils.launch_utils import TrialSummary


@dataclass
class TrialContext:
    """Inputs shared by every single-trial executor."""

    env: Any
    trial: int
    args: Any
    runtime_config: dict[str, Any]
    multi_turn_prompt: str | None
    partial_artifacts: dict[str, Any]


class TrialExecutor(Protocol):
    """Execute one trial attempt behind the runner's common infrastructure."""

    def run(self, context: TrialContext) -> TrialSummary:
        """Run one attempt and return a common summary."""

    def build_timeout_summary(
        self,
        context: TrialContext,
        timeout_seconds: int,
        exc: BaseException,
    ) -> TrialSummary:
        """Build an implementation-specific summary for an interrupted attempt."""


class TrialExecutorBase:
    """Base executor with a trial-neutral, best-effort timeout summary."""

    def build_timeout_summary(
        self,
        context: TrialContext,
        timeout_seconds: int,
        exc: BaseException,
    ) -> TrialSummary:
        task_completed = False
        reward = 0.0
        low_level_env = getattr(context.env, "low_level_env", context.env)
        completion_fn = getattr(low_level_env, "task_completed", None)
        reward_fn = getattr(context.env, "compute_reward", None)
        if not callable(reward_fn):
            reward_fn = getattr(low_level_env, "compute_reward", None)

        try:
            if callable(completion_fn):
                task_completed = bool(completion_fn())
            if callable(reward_fn):
                reward = float(reward_fn())
        except Exception:
            # Timeout reporting must still succeed when the simulator is unhealthy.
            pass

        return TrialSummary(
            trial=context.trial,
            success=False,
            reward=reward,
            terminated=task_completed,
            truncated=True,
            sandbox_rc=1,
            log=f"Trial {context.trial} timed out after {timeout_seconds} seconds: {exc}",
            task_completed=task_completed,
            code_path=None,
            num_regenerations=0,
            num_finishes=0,
            num_code_blocks=0,
        )


__all__ = ["TrialContext", "TrialExecutor", "TrialExecutorBase"]
