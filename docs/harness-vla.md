# Harness VLA reading notes

Paper: [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via
Memory-Guided Agents](https://arxiv.org/abs/2607.08448), arXiv:2607.08448v1, 9 July 2026.

## Main idea

Harness VLA turns a frozen VLA from an end-to-end task policy into one primitive in a small,
fixed action library. The agentic planner handles task-language grounding, scene re-grounding,
long-horizon composition, free-space motion, staging, transport, release, verification, and
recovery. The VLA is reserved for local contact-rich behavior such as grasping, insertion,
articulated interaction, pressing, and constrained placement.

The planner does not produce low-level actions or an open-loop program. At every turn it reads
the current task, RGB-D evidence, robot state, Task Specific Memory, and Global Memory, then
emits exactly one JSON primitive call. The environment executes that primitive until its local
return condition, publishes a new observation and diagnostic record, and returns control to the
planner.

The important reliability mechanism is **re-stage, then retry**. A failed VLA call is treated as
a failed local contact attempt, not a failed episode. The planner can change the viewpoint,
approach pose, wrist/base posture, or other staging conditions before invoking the same frozen
VLA again. Analytic primitives isolate these non-contact corrections from contact-rich control.

The paper reports improvements of 38.6 and 25.4 percentage points over the strongest relevant
baselines on LIBERO-Pro and RoboCasa365, respectively, and 58.4% success on RoboTwin C2R.

## Primitive and execution contract

The shared paper vocabulary consists of `MOVE_TO`, `MOVE_POSE`, `ROTATE_WRIST`,
`ROTATE_PITCH`, `SET_GRIPPER`, `RELEASE`, and `VLA_ACT`. Mobile manipulation additionally
uses `NAVIGATE_TO` and `MOVE_BASE`. `RESET` is available only during exploratory
bootstrapping and is disabled during strict evaluation.

Properties relevant to our implementation:

- The vocabulary is fixed before deployment; the planner cannot invent a new primitive.
- One planner turn contains one structured primitive invocation.
- Every primitive is followed by a refreshed observation and diagnosis.
- Spatial values from a reference rollout are not replayed; targets are re-grounded from the
  current observation.
- VLA calls are sparse, short local attempts with bounded chunk budgets and stop conditions.
- A primitive post-condition only ends that primitive. It is not a replacement for the
  benchmark-provided task completion predicate.
- Planner-visible state must not expose privileged simulator poses or controller internals.

This maps naturally onto our existing loop:

```text
FineGrainedPlanner.next_stage()
  -> Stage.primary_call
  -> ToolBackedPrimitiveExecutor.execute()
  -> refreshed observation
  -> TrialMemory.update_after_execution()
  -> Verifier.check()
  -> RecoveryPolicy.decide()
```

The paper implements diagnosis inside the agentic harness rather than defining separate Python
`Verifier` and `RecoveryPolicy` classes. The split above is our engineering interpretation: it
makes evidence, failure classification, and recovery decisions explicit and testable while
preserving the paper's closed-loop behavior.

## Refined P0.2: minimal FineGrainedPlanner

### Responsibility

The planner owns semantic binding and stage progression. It decides **what primitive should run
next**, but does not execute tools, decide physical success without evidence, or emit raw joint
actions.

### Inputs

Each decision should receive:

- authoritative task language from the current environment state;
- the latest `WorldState`, including observation ID and robot state;
- available primitive schemas/capabilities for the embodiment;
- optional Task Specific Memory solution skeleton;
- optional Global Memory success rules and failure models;
- current stage and remaining turn/VLA/recovery budgets.

### Minimal stage representation

A Stage is exactly one primary primitive. Retry and re-stage repair calls are additional turns
within that same Stage; they are not nested stages or broad semantic subtasks.

```python
@dataclass
class Stage:
    id: str
    objective: str
    primary_call: ToolCall
    retry_budget: int = 0
    restage_budget: int = 0


@dataclass
class StagePlan:
    task: str
    stages: tuple[Stage, ...]
```

For P0, stages may be produced by a deterministic task template or a stub planner. They must use
symbolic targets such as `red_cup` and `sink_region`, not reference-scene coordinates.

### Decision rules

1. Parse the live task language; never infer the target from a task ID, filename, or old trace.
2. Treat Task Specific Memory as a structural prior: reuse primitive order and contact/non-contact
   boundaries, but re-ground every spatial argument.
3. Prefer analytic primitives for staging, free-space transport, posture adjustment, release, and
   recovery.
4. Use `VLA_ACT` only for a local contact-rich stage and supply the current semantic target in its
   prompt.
5. Emit one Stage (and therefore one primary `ToolCall`) per `next_stage()` progression.
6. Advance a stage only after the Verifier returns success for its predicate.
7. After failure, consume the recorded verdict/recovery outcome before choosing another planner
   call. Do not blindly replay the last call.
8. Stop only on official task success, budget exhaustion, unrecoverability, or an explicit planner
   completion after all stages are verified.

### P0 acceptance cases

- A fixed pick-and-place plan represents staging, VLA grasp, transport, and release as separate
  one-primary-primitive stages.
- Redirecting the task from object A to object B changes the symbolic target and VLA prompt.
- A stored trace containing coordinates never causes those coordinates to be replayed.
- A failed grasp does not advance to transport.
- A re-stage call completes before the failed VLA call is scheduled again.

## Refined P0.3: TrialMemory

Harness VLA separates two persistent memories. Task Specific Memory contains a successful
task-level solution skeleton plus a semantic audit; Global Memory contains task-independent
success rules and failure models. Our runtime additionally needs Episode Memory to represent the
currently executing rollout.

### Episode Memory: required in P0

`TrialMemory` should own the current `WorldState` and an append-only attempt trace:

```python
WorldState = {
    "task": str,
    "observation_id": int,
    "robot": {"eef_pose": ..., "gripper": ..., "base_pose": ...},
    "objects": {...},
    "holding": str | None,
    "current_stage": str,
    "last_outcome": {...},
    "failure_history": [...],
    "budgets": {"turns": ..., "vla_calls": ..., "recoveries": ...},
}
```

Each attempt record should contain:

- monotonic turn/observation IDs;
- accepted `ToolCall`, its Stage ID, and whether its turn role is primary, retry, re-stage, or
  re-stage retry;
- normalized `ToolResult` and execution telemetry;
- post-execution observation reference and robot state;
- `VerificationVerdict`, evidence, failure code, and recovery decision;
- stage and budget counters before/after the call.

Memory update ordering is an invariant:

```text
execute -> observe -> update state -> verify -> record verdict -> decide recovery
```

Failed primitives must still update state because they may have moved the robot or scene.

### Task Specific Memory: minimal P0 interface

Represent a solved reference task with two artifacts, following the paper:

- audit JSON: success, strategy, useful primitive choices, recovery notes, failure observations;
- command JSONL: ordered primitive calls forming the procedural skeleton.

P0 only needs load/save and retrieval by task identity. Before reuse, literal `xyz`, quaternion,
pixel, object-pose, fixture, and base-pose bindings must be removed or marked as reference-only.
Only the symbolic stage structure is passed to the planner.

### Global Memory: P0 stub, later learning

Expose a read-only interface returning success rules and failure models. It may start from a small
static file, for example:

- use VLA for irregular contact and analytic motion after a stable grasp;
- closed gripper without object co-motion indicates an empty grasp;
- visual proximity alone is not final success;
- re-localize and re-stage before retrying an unchanged failed contact attempt.

Automatic cross-task distillation and trace replacement belong to P2, not P0.

### P0 acceptance cases

- Observation IDs and trace entries increase once per executed primitive.
- A failed tool call records the resulting changed observation before recovery.
- Memory can distinguish planner, retry, and re-stage calls.
- A successful trace can be serialized to JSONL and loaded as a symbolic skeleton.
- No planner-facing memory field exposes privileged simulator object poses.

## Refined P0.4: basic Verifier

### Two levels of success

The verifier must keep these separate:

1. **Primitive post-condition**: determines whether the local call achieved its intended effect
   and whether the planner may advance the current stage.
2. **Official task predicate**: the environment/benchmark completion signal; this is the only
   authority for reporting task success.

`ToolResult.ok` means that the underlying function returned successfully. It is evidence, not a
physical-success verdict.

### Interface and verdict

```python
VerificationVerdict(
    status=SUCCESS | FAILURE | UNKNOWN | UNSAFE,
    code="object_in_gripper",
    message="...",
    evidence={...},
)
```

`Verifier.check()` consumes the call, normalized result, previous state, refreshed state, and
latest observation evidence. The current protocol may initially retrieve previous state through
Memory, but the interface should eventually make the before/after states explicit.

### Minimum P0 checks

- Motion: controller returned, current EEF/base pose is within a coarse configured tolerance, and
  no unsafe execution status was reported.
- Grasp/VLA grasp: gripper/contact evidence plus object lift or object/end-effector co-motion.
  A closed gripper or plausible-looking image alone is insufficient.
- Release: gripper is open, the object no longer follows the end effector, and target-region
  evidence is present when the stage requires placement.
- VLA non-grasp contact: use the call's named post-condition and before/after evidence; a chunk
  budget or VLA return is not automatically success.
- Official completion: query the benchmark-provided `task_completed()` after every primitive and
  at termination. It overrides visual guesses about final success.

If evidence is missing or contradictory, return `UNKNOWN`, not `SUCCESS`. Recovery can then
request another observation, re-stage, replan, or abort according to budget and safety policy.

### Evidence discipline

Every verdict should record the observation IDs, relevant robot fields, tool status, predicate
evaluated, and measured/observed reason. This makes false success, empty grasps, wrong targets,
short placements, and no-effect VLA calls diagnosable instead of collapsing them into a boolean.

### P0 acceptance cases

- `goto_pose` returning `None` does not pass if the measured pose remains outside tolerance.
- A gripper closing without object co-motion produces an empty-grasp failure.
- VLA returning because `max_chunks` was reached does not imply contact success.
- Visual placement without the official predicate cannot mark the trial complete.
- Missing perception produces `UNKNOWN` and never advances the stage.

## What remains outside P0

P0 may use coarse pose checks, mock perception, static global rules, and a deterministic stage
planner. Precise RGB-D grounding, coordinate-frame schemas, safe predicate registries, controller
timeouts, collision/velocity checks, automatic memory distillation, and production VLA stop
conditions remain P1/P2 work.
