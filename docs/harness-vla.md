# Harness VLA Primitive

CaP-X exposes a frozen VLA through a model adapter, an environment action adapter, and a
transport. Configure the API alongside analytic control primitives:

```yaml
env:
  cfg:
    apis:
      - FrankaLiberoApiReduced
      - name: VlaPrimitiveApi
        config:
          model:
            type: http
            adapter:
              name: native
            transport:
              base_url: http://127.0.0.1:8123
              timeout: 120
          action_adapter:
            name: identity
          primitive:
            default_max_chunks: 2
```

String API entries remain backward compatible. Structured entries are intentionally built
only after the low-level environment exists, allowing action adapters to inspect the target
robot and controller.

Generated code can use the VLA for a local contact-rich phase:

```python
result = vla_act("grasp the black bowl")
print(result)
```

The layers have separate responsibilities:

1. `VlaPrimitiveApi` schedules chunks and evaluates stop conditions.
2. `VlaModelAdapter` encodes observations/prompts and decodes model responses.
3. `VlaActionAdapter` converts decoded actions to the environment's native controller space.
4. `VlaInferenceTransport` moves dictionaries to and from the inference runtime.

The initial `native` + `identity` pair is deliberately minimal. It expects `POST /predict`
to return actions already compatible with the environment:

```json
{
  "actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
  "done": false,
  "metadata": {"model": "checkpoint-name"}
}
```

For LIBERO, native actions are executed by `FrankaLiberoEnv.execute_policy_action`. A future
pi0.5 integration should add `Pi05ModelAdapter` and one matching LIBERO action adapter; GR00T
can add another model adapter without changing the primitive or trial loop.

## Mock model

Use the in-process mock model to replace the remote-model invocation function. Its
`predict()` method directly returns the configured `ModelActionChunk`, just as the HTTP model
path does after response decoding:

```yaml
- name: VlaPrimitiveApi
  config:
    model:
      type: mock
      actions:
        - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
      done: true
    action_adapter: identity
```

This bypasses HTTP while preserving the model-chunk, action-adapter, primitive, and environment
execution boundaries.
