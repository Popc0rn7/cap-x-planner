# CaP-X Trial Execution Note

## Execution hierarchy

```text
Batch evaluation
└── Trial: one complete task attempt
    ├── Reset the environment
    ├── Turn 0: generate and execute Python code
    ├── Turn 1+: observe, then REGENERATE or FINISH
    └── Save reward, code, video, and logs
```

- **Batch**: runs multiple trials, possibly in parallel.
- **Trial**: one task attempt from environment reset until success, finish, timeout, or budget exhaustion.
- **Turn**: one planner generation/decision followed by code execution and observation.
- **Action / primitive**: a concrete robot operation executed inside a turn.

## Current single-trial loop

```text
reset
  → capture initial observation
  → ask the model to generate Python
  → execute the code with env.step(code)
  → collect observation, reward, stdout/stderr, and visual feedback
  → ask the model to REGENERATE or FINISH
  → repeat until completion or the trial budget is exhausted
  → save artifacts and summary
```

The loop is implemented in `capx/envs/trial.py::_run_single_trial`. The outer
`capx/envs/runner.py::_run_trial_with_retries` retries the entire trial on infrastructure
timeouts; it is not a local robot-action retry.

## Configurable trial executors

`trials` is the number of attempts. `trial_executor` selects the algorithm used inside each
attempt and is configured at the top level of an environment YAML. It is intentionally not
part of `env.cfg`, because it controls runner behavior rather than simulator construction.

```yaml
trial_executor:
  type: code_agent  # default when omitted
```

```yaml
trial_executor:
  type: fine_grained
  component_factory:
    _target_: capx.my_components.build_fine_grained_components
  max_turns: 50
  max_wall_time_s: 900
```

`runner.py` owns batches, workers, and hard trial timeouts. A `TrialExecutor` owns the
single-attempt algorithm. Fine-grained planner, verifier, memory, and recovery configuration
stays below `FineGrainedTrialExecutor`.

## Future Agentic Planner trial

A trial remains one complete task attempt. Only its inner loop changes:

```text
observe → choose one ToolCall → execute → verify
        ↑                              │
        └─ retry / re-stage / replan ─────────┘
```

## Fine-grained trial migration TODO

### Goal

Build a primitive-level, closed-loop Agentic Planner in which the planner emits one structured
`ToolCall` per turn. Every physical attempt must be followed by a refreshed observation,
post-condition verification, explicit state update, and a decision to advance, retry, re-stage,
replan, or stop. VLA is one contact-rich primitive rather than the end-to-end task controller.

### Current versus target

| Current code-agent trial | Target fine-grained trial |
| --- | --- |
| One turn executes generated Python | One turn executes one `ToolCall` |
| Recovery regenerates code | Recovery selects retry, re-stage, replan, or abort |
| State mainly lives in Python globals | State lives in explicit `WorldState` and Memory |
| API return is treated as execution completion | Verifier checks physical post-conditions |
| VLA may be called freely from code | VLA is restricted to local contact-rich phases |
| Metrics describe code blocks | Metrics describe stages, tools, failures, and recovery |

### P0: make the headless trial runnable

- [ ] Implement `build_fine_grained_components(env, ...)`.
- [ ] Complete a `FineGrainedPlanner` whose symbolic `StagePlan` defines each Stage as exactly one
      primary primitive; retries and re-stage repairs remain turns inside that Stage.
- [ ] Implement layered `TrialMemory`: episode `WorldState` and attempt trace, a task-specific
      symbolic solution skeleton, and a read-only P0 global-rule interface.
- [ ] Implement an evidence-backed `Verifier` that separates primitive post-conditions from the
      official task predicate and returns `SUCCESS`, `FAILURE`, `UNKNOWN`, or `UNSAFE`.
- [ ] Implement a typed failure taxonomy and `RecoveryPolicy`.
- [x] Add trial-executor dispatch selected by `trial_executor.type` in environment YAML.
- [ ] Add an embodiment-specific YAML that selects `fine_grained` and a real component factory.
- [ ] Run an end-to-end mock trial before connecting a real VLA checkpoint.

Detailed P0.2/P0.3/P0.4 specifications and paper notes are in
[`docs/harness-vla.md`](docs/harness-vla.md).

### P1: complete primitive execution semantics

- [ ] Replace symbolic targets with grounded, frame-aware poses immediately before execution.
- [ ] Add pose frames, quaternion convention, arm binding, tolerances, and observation timestamps.
- [ ] Enforce `ToolCall.timeout_s` for blocking API calls.
- [ ] Add workspace, collision, velocity, and unsafe-state checks.
- [ ] Replace the naive same-name function fallback with explicit primitive schemas and
      embodiment capability descriptors.
- [ ] Add a safe predicate registry for VLA stop conditions and verifier post-conditions.
- [ ] Re-localize after robot, camera, object, base, fixture, or grasp-state changes.

### P2: recovery and memory

- [ ] Distinguish infrastructure retry, primitive retry, re-stage, and replan in code and logs.
- [ ] Implement initial failure codes such as `EMPTY_GRASP`, `OBJECT_DROPPED`,
      `IK_UNREACHABLE`, `PLACEMENT_SHORT`, `VLA_NO_EFFECT`, and `WRONG_TARGET`.
- [ ] Ensure every re-stage repair call uses the same execute-observe-verify path.
- [ ] Add Episode Memory, parameterized Task Specific Memory, and Global Memory.
- [ ] Store successful procedural structure without replaying literal coordinates.
- [ ] Track turn, VLA chunk, retry, re-stage, replan, environment-step, and wall-time budgets.

### P3: observability and integration

- [ ] Add fine-grained summary fields: tool calls, VLA calls, retries, re-stages, replans,
      failure codes, trace path, and audit path.
- [ ] Save fine-grained partial artifacts incrementally so timeout reports include the last
      `ToolCall`, verdict, and `WorldState`.
- [ ] Add Web UI events for stage, tool call, tool result, verification, recovery, and budget.
- [ ] Make headless and Web UI consume the same trial executor instead of duplicating loops.
- [ ] Add regression comparisons for code-agent, VLA-only, retry-only, and re-stage trials.
