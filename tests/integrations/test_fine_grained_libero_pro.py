"""Opt-in real LIBERO-PRO smoke test for the agentic stage workflow.

Run with ``CAPX_RUN_LIBERO_FINE_GRAINED=1`` after starting the planner model,
VLA, and perception services described in ``docs/fine_grained_libero.md``.
"""

from __future__ import annotations

import importlib.util
import os
import socket

import pytest


def _service_available(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


@pytest.mark.integration
def test_libero_object_0_agentic_pick_and_place_smoke() -> None:
    if os.environ.get("CAPX_RUN_LIBERO_FINE_GRAINED") != "1":
        pytest.skip("set CAPX_RUN_LIBERO_FINE_GRAINED=1 to run the real LIBERO-PRO smoke test")
    if importlib.util.find_spec("libero") is None:
        pytest.skip("LIBERO is unavailable; install the separate LIBERO environment")
    torch = pytest.importorskip("torch", reason="PyTorch/CUDA is required by LIBERO-PRO")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    missing = [port for port in (8110, 8114, 8123) if not _service_available(port)]
    if missing:
        pytest.skip(f"required planner/perception/VLA services are unavailable: {missing}")

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader
    from capx.envs.trial_base import TrialContext
    from capx.envs.trial_factory import build_trial_executor

    config = DictLoader.load("env_configs/libero/franka_libero_object_0_fine_grained.yaml")
    env = instantiate(config["env"])
    executor = build_trial_executor(config["trial_executor"])
    summary = executor.run(
        TrialContext(
            env=env,
            trial=0,
            args=None,
            runtime_config=config,
            multi_turn_prompt=None,
            partial_artifacts={},
        )
    )

    assert summary.task_completed, summary.log
