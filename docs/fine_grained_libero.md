# Agentic fine-grained LIBERO-PRO smoke test

The P0 configuration is
`env_configs/libero/franka_libero_object_0_fine_grained.yaml`. It runs
`libero_object`, task 0, as one trial on one worker. The stage builder may produce only
`VLA`, `SET_GRIPPER`, and `RELEASE` calls; final task success is always read from LIBERO's
`task_completed()` result.

Use the separate LIBERO environment described in the repository README. The smoke test also
requires CUDA, a planner/reward model endpoint on port 8110, the configured perception service
on port 8114, and a native VLA HTTP endpoint on port 8123. Then run:

```bash
CAPX_RUN_LIBERO_FINE_GRAINED=1 uv run --no-sync --active pytest \
  tests/integrations/test_fine_grained_libero_pro.py -q
```

Or launch the single-trial configuration directly:

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_object_0_fine_grained.yaml
```

The integration test skips with a specific reason if the opt-in flag, LIBERO, CUDA, or any
required endpoint is unavailable.
