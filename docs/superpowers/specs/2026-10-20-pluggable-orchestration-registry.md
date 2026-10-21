# RFC: Pluggable orchestration patterns (an orchestration registry)

**Status:** Proposed — design only, no engine code until this is reviewed.
**Scope:** `process-bigraph` core (`Composite`), with a downstream registration
point for `vivarium-workbench`.
**Relates to:** `docs/tick_lifecycle.md`, `docs/architecture.md`, the
investigation-as-composite work in vivarium-workbench
(`investigation_execution.py`), and the composite contract/holes machinery.

## 1. Motivation

A `Composite` today advances through **one hardcoded orchestration loop** —
the event-scheduling tick lifecycle in `Composite.run()` (the per-process
`front`, `determine_steps` for zero-interval `step`s, reconcile, apply). What
look like "two orchestration patterns" are really **two regimes of that one
loop**: temporal multi-timestepping, with zero-interval steps forming a DAG
*nested inside* a tick.

This rigidity has a concrete cost one level up. An *investigation* is run three
different ways in vivarium-workbench today — the `run_jobs` manager
(`/api/investigation-run-unblocked`, canonical, async), an inline-subprocess
path (`/api/investigation-run`), and the sidelined `StudyStep` composite
(`run_investigation_composite`) — because "run the members of an investigation
in prerequisite order, dispatching each to the run subsystem" is **not an
orchestration pattern the core knows about**. Each re-implements membership,
prerequisite ordering, and dispatch ad hoc. Earlier analysis confirmed these
are not freely interchangeable (they serve different investigation shapes and
granularities), so neither "delete one" nor "force one canonical" is right.

The generalization that dissolves this: make **orchestration a first-class,
registered, pluggable dimension of a composite** — the same move the framework
already makes for *types* (`register_type`) and *processes* (`register_process`).
A composite then *declares* which orchestration pattern governs it; the three
ad-hoc investigation orchestrators collapse into **one registered pattern**; and
new patterns (ensemble sweeps, optimization loops) register cleanly instead of
being bolted on. Generalization here *reduces* code by making the special cases
instances of one extensible thing.

## 2. Current state (verified)

- `Composite.run()` owns the loop and the state it needs: `self.front`
  (per-process next-event time + pending `Defer` update), `_cached_view`,
  `process_update`/`invoke`, `apply_updates`, reconciliation, and rescheduling.
  `scheduling.determine_steps` orders the zero-interval DAG within a tick.
- There is **no orchestration/scheduler strategy object and no orchestration
  registry.** The registries that exist are the type registry and the link
  (process-address) registry.
- There are already two *partial* seams into the lifecycle, which this RFC
  generalizes rather than replaces:
  - **`flush_pending`** (`_flush_protocol_runtimes`) — batch pending RPCs
    between invoke and apply.
  - **`tick_lifecycle()`** (`docs/tick_lifecycle.md`) — a runtime may take over
    the invoke+apply lifecycle for the processes it manages. This is precedent:
    the engine already lets an external component own part of the tick. The
    orchestration registry lifts that from "a runtime batching *within* the
    temporal loop" to "a strategy that owns the *whole* advance."

## 3. Proposal

### 3.1 An orchestration registry

Add a third registry beside types and links:

```
core.register_orchestration(name: str, strategy: type[OrchestrationStrategy])
core.orchestration_registry: dict[str, type[OrchestrationStrategy]]
```

Registration is discoverable the same way processes/types are, so **downstream
packages register their own patterns at `build_core` time** (critical — see
§3.4: the `investigation` pattern must live in vivarium-workbench, not the
core).

### 3.2 The strategy protocol

An orchestration strategy owns how a composite advances. The protocol is drawn
directly from what `Composite.run()` already does, so the current loop *is* a
conforming strategy:

```python
class OrchestrationStrategy:
    def __init__(self, composite, config): ...
    def initialize_front(self, state) -> None:
        """Seed per-unit scheduling state (today: Composite.front)."""
    def advance(self, interval) -> None:
        """Advance the composite by `interval`, mutating its state through the
        engine's apply path. For `temporal` this is the event loop; for
        `workflow` a single topological DAG pass; for `investigation` a
        prerequisite-ordered dispatch of member studies."""
    def next_event_time(self) -> float | None:
        """For composability: the composite's own next due time when it is
        itself a process in a parent composite (recursive composition)."""
```

`Composite.run()/update()` becomes a thin shell that resolves its strategy from
the registry and delegates. The strategy reuses the engine's existing
primitives (`_cached_view`, `process_update`, `apply_updates`, `determine_steps`)
— it decides *ordering and dispatch*, not how a single update is typed or
applied (those stay in the type engine).

### 3.3 A composite declares its orchestration

```yaml
# default when omitted: "temporal"
orchestration: temporal        # | workflow | investigation | <registered name>
orchestration_config: { ... }  # strategy-specific (e.g. dispatch target)
```

### 3.4 The three initial patterns

| pattern | what it does | ships in |
|---|---|---|
| **`temporal`** | today's event-scheduling loop (front + nested zero-interval DAG), extracted verbatim. The default; behavior-preserving. | process-bigraph (core) |
| **`workflow`** | a pure zero-interval topological DAG (no time advance) — analysis/step pipelines. Already expressible today; this names it. | process-bigraph (core) |
| **`investigation`** | a prerequisite-DAG over member studies, dispatching each to the run subsystem; `orchestration_config` selects sync/async and local/deployment target. **The three ad-hoc workbench orchestrators collapse into this one pattern.** | vivarium-workbench (registers into the core at `build_core`) |

The `investigation` strategy must dispatch to the workbench's run subsystem
(env-worker / `run_jobs` / deployment), which is **not** in the core. So the
core defines the *protocol and registry*; vivarium-workbench *registers* the
`investigation` strategy — exactly as downstream packages register processes
and types today. No workbench dependency leaks into the core.

## 4. What this buys (the reduction)

- The investigation-orchestration triplicate (`run_jobs` enumeration + inline
  subprocess + `StudyStep` composite) becomes **one registered strategy**
  parameterized by dispatch target — genuine consolidation, not a fourth path.
- "Investigation as a composite" becomes principled: an investigation composite
  declares `orchestration: investigation`. The supplement's claim is then
  literal and mechanism-backed, not aspirational.
- The two implicit regimes become **named, documented, testable** strategies.
- Future patterns (ensemble/parameter sweep, optimization loop, trigger /
  pull-or-compute) register without touching `Composite`.

## 5. Migration path (behavior-preserving, delete-gated)

1. **Extract, don't change.** Introduce the registry + protocol; wrap today's
   `Composite.run()` loop unchanged as the built-in `temporal` strategy behind
   it. **Gate:** the full process-bigraph engine suite is green with zero
   behavior change and `orchestration` defaulting to `temporal`. No composite
   anywhere declares a pattern yet.
2. **Name `workflow`.** Register the pure zero-interval DAG regime as its own
   strategy; migrate step-only pipelines to declare it (optional; `temporal`
   already handles them).
3. **Add `investigation`** in vivarium-workbench as a registered strategy, built
   to parity with the canonical `run_jobs` path (async/poll, deployment
   dispatch). **Gate:** a parity harness runs a real investigation through the
   new strategy and the existing `run_jobs` path and asserts equivalent run
   refs / verdicts on both run targets.
4. **Collapse the triplicate.** Repoint the endpoints/UI to the `investigation`
   strategy; retire the redundant ad-hoc code **only** behind green parity
   tests (each deletion gated, per the consolidation discipline).

## 6. Non-goals

- Not changing tick semantics, the delta/apply model, reconciliation, or the
  type engine. A strategy decides ordering/dispatch, not typing.
- Not a scheduler rewrite. `determine_steps` and the front are reused.
- Default behavior is unchanged: a composite with no `orchestration` key runs
  exactly as today.

## 7. Risks

- **Highest-stakes surface in the stack** — `Composite.run()` backs every
  simulation. Mitigation: step 1 is a pure extraction with a zero-behavior-change
  gate; nothing downstream changes until `temporal` is proven identical.
- **Recursive composition** — a composite-as-process must still expose a next
  event time; `next_event_time()` is in the protocol for exactly this, and the
  `temporal` extraction must preserve current nested-composite behavior.
- **The `tick_lifecycle`/`flush_pending` seams** must keep working inside the
  `temporal` strategy (they are a within-strategy optimization, not a competing
  mechanism).

## 8. Open questions

- Exact `advance`/front ownership split between `Composite` and the strategy
  (how much of `self.front` moves into the strategy vs. stays on the composite).
- Whether `workflow` is worth materializing separately or stays a documented
  regime of `temporal` (it earns its keep only if a pure-DAG strategy is simpler
  or faster than the general loop for step-only composites).
- Registration timing for downstream strategies vs. when a composite resolves
  its orchestration (must resolve against the same `build_core` the processes
  came from).
