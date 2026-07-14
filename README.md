# Harness VLA for CaP-X

Harness VLA adds a configurable vision-language-action (VLA) primitive to
[CaP-X](README-cap-x.md). It lets a coding agent hand short, contact-rich phases—such as
grasping, insertion, and articulated-object interaction—to a frozen VLA, while retaining
analytic CaP-X primitives for perception, staging, transport, release, and recovery.

The integration separates model-specific inference from environment-specific control:

- `VlaPrimitiveApi` exposes the planner-facing `vla_act(...)` tool and limits execution to
  short action chunks.
- `VlaModelAdapter` translates CaP-X observations and prompts to a model contract.
- `VlaActionAdapter` converts decoded model actions to the environment's native action space.
- `VlaInferenceTransport` connects the adapter to an inference runtime.

The initial implementation includes an HTTP `native` model adapter, an `identity` action
adapter for models that already emit native controller actions, and an in-process mock model
for smoke testing.

## Quick start

Follow the [CaP-X setup guide](README-cap-x.md#installation) and install the separate LIBERO
environment before running these examples. The evaluation harness also needs its normal LLM
proxy.

### Mock VLA

The mock configuration exercises the complete primitive and environment execution path without
starting a VLA inference service:

```bash
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_harness_vla_mock.yaml
```

### Remote VLA

Start a compatible service, then run the native configuration:

```bash
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_harness_vla_native.yaml
```

The native adapter sends `POST /predict` with a prompt and serialized observation:

```json
{
  "prompt": "grasp the black bowl",
  "observation": {
    "cameras": {
      "agentview": {
        "rgb_png_base64": "...",
        "depth_npy_base64": "...",
        "intrinsics": [],
        "pose_mat": []
      }
    },
    "proprioception": {
      "robot_cartesian_pos": [],
      "robot_joint_pos": []
    }
  }
}
```

It expects a JSON response whose actions already match the loaded environment controller:

```json
{
  "actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
  "done": false,
  "metadata": {"model": "checkpoint-name"}
}
```

The eight-dimensional action above is an example for the included LIBERO setup, not a universal
VLA action schema. Add a model adapter and, when necessary, an action adapter for models with a
different request, response, or control convention.

## Configuration

Add the primitive next to the analytic API used by a task:

```yaml
apis:
  - FrankaLiberoApiReduced
  - name: VlaPrimitiveApi
    config:
      model:
        type: http
        adapter: native
        transport:
          base_url: http://127.0.0.1:8123
          timeout: 120
      action_adapter: identity
      primitive:
        default_max_chunks: 2
```

Generated code can then delegate one local phase and inspect the result before continuing:

```python
result = vla_act("grasp the black bowl")
print(result)
```

`vla_act` returns its stop reason, executed chunk and action counts, task-completion state, and
backend metadata. A successful call is not itself a grasp-success oracle; inspect the next
observation before transport or retry.

## Documentation

- [Harness VLA primitive guide](docs/harness-vla.md)
- [Full CaP-X README](README-cap-x.md)
- [Adding APIs](docs/adding-apis.md)
- [Configuration reference](docs/configuration.md)
- [Development and testing](docs/development.md)

Run the focused unit tests with:

```bash
uv run pytest tests/test_vla_primitive.py -q
```
