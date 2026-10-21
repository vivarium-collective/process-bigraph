# Plan: a streamlined Composite primitive interface for pluggable orchestration

**Status:** Proposed — design, informed by the shipped `temporal` seam (#254)
and the new `workflow` strategy (this branch). Follows
`2026-10-20-pluggable-orchestration-registry.md`.
**Goal:** make orchestration strategies *thin* by giving `Composite` a small,
efficient, documented primitive interface that every strategy composes — so
`temporal`, `workflow`, and future patterns are ≤~20-line `advance` methods over
the same primitives, and `Composite` no longer *owns* any one orchestration
algorithm.

## 1. What the second strategy revealed

`WorkflowOrchestration.advance` is the whole strategy:

```python
def advance(self, interval, force_complete, run_start):
    if composite.to_run:
        composite.run_steps(composite.to_run)   # self-cascades the step DAG
        composite.to_run = []
    composite.state['global_time'] += interval
```

It composes **only two Composite primitives** — the ready-step queue
(`to_run`) and `run_steps` (which drains the DAG via its own
`cycle_step_state`/`trigger_steps` cascade). It touches **none** of the temporal
machinery (`front`, `run_process`, `full_step`, `_run_tick_lifecycle`, the
parallel-process layer). That cleaves the engine's primitives cleanly in two:

| primitive group | used by | status |
|---|---|---|
| **step-DAG execution** — `to_run`, `run_steps` (self-cascading), `apply_updates` | workflow *and* temporal | already small + clean |
| **temporal execution** — `front`, `run_process` / `_run_processes_layer_parallel`, `_partition_processes_by_runtime` / `_run_tick_lifecycle`, `full_step`, reschedule | temporal only | the entangled part |

So the "fully-clean target" is not "factor everything" — it's: **keep the
step-DAG primitives shared and small, and move the *temporal-only* machinery
(loop + its `front` state) out of `Composite` into `TemporalOrchestration`.**

## 2. The target interface

A strategy is constructed with its Composite and implements `advance`. The
Composite exposes a **small primitive surface** strategies compose — nothing
more:

```python
class Composite:
    # --- shared primitives (already exist; keep) ---
    def run_step_network(self) -> None:      # drain the ready-step DAG (= to_run + run_steps)
    def apply_updates(self, updates) -> list # type-directed apply (unchanged)
    # --- temporal primitive (extracted from _run_inner) ---
    def advance_tick(self, end_time, force_complete) -> float   # ONE tick; returns full_step
    def next_event_time(self) -> float | None                    # min front time (recursive composition)
    # state access: self.state['global_time']
```

- **`workflow`** composes `run_step_network` + clock bump. (Already true.)
- **`temporal`** owns the driving loop + its scheduling state (`front`), calling
  `advance_tick` until `global_time` reaches the horizon; `next_event_time`
  serves recursive composition when a Composite is itself a process.
- **A future `investigation`** strategy is workflow-shaped (a member-study step
  DAG) — it composes `run_step_network`, no new engine code.

`run_step_network` is the named form of today's `to_run`/`run_steps` idiom
(workflow uses it; temporal already runs steps each tick, so it adopts the same
name). `advance_tick` is the one-tick body carved out of `_run_inner`.

## 3. Migration — behavior-preserving, delete-gated, in risk order

1. **Name the step-DAG primitive.** Add `Composite.run_step_network()` wrapping
   the existing `if to_run: run_steps(to_run); to_run=[]`. Repoint `workflow`
   and the top of the temporal loop to it. *Gate:* full engine suite green,
   zero behavior change. **Low risk.**
2. **Extract `advance_tick`.** Carve the single-tick body out of `_run_inner`
   into `Composite.advance_tick(end_time, force_complete) -> full_step` (the
   partition → invoke → apply → reschedule of one tick). `_run_inner` becomes a
   thin `while` over `advance_tick`; then move *that while loop* into
   `TemporalOrchestration.advance`, leaving `advance_tick` on the Composite.
   *Gate:* full suite green incl. unequal-interval / chunking / parallel /
   `tick_lifecycle` tests. **Medium risk** (the hot path).
3. **Move `front` into `TemporalOrchestration`.** The temporal scheduling state
   stops being Composite state. `Composite.next_event_time()` delegates to the
   strategy (for recursive composition); `clean_front` and the nested-composite
   next-time path move with it. *Gate:* full suite + nested-composite +
   recursive-composition tests green. **Highest risk** — do last, alone, and
   only if the purity is judged worth it; step 2 already removes the loop from
   `Composite`, so this step is optional polish.
4. **Delete** `_run_inner` (now empty) and the `TemporalOrchestration`
   delegator-to-`_run_inner`; document the final primitive interface.

Each step is independently shippable and leaves the default behavior identical.
Steps 1–2 deliver the streamlined interface (Composite no longer owns the loop);
step 3 is the last, invasive bit of purity.

## 4. Efficiency (an explicit constraint)

The interface must add **no per-tick overhead**. `advance_tick` does exactly the
work `_run_inner`'s loop body does today; `run_step_network` is the existing
`run_steps`. The only new cost is **one strategy dispatch per `run()` call**
(already paid since #254), not per tick. The `_partition_processes_by_runtime` /
`_run_tick_lifecycle` batching and the `parallel_processes` layer stay *inside*
`advance_tick`, so the tick-lifecycle optimization in
`docs/tick_lifecycle.md` is unaffected.

## 5. Risks / open questions

- **Recursive composition** is the crux of step 3: a Composite-as-process asks
  its orchestration for its next event time. `next_event_time()` must delegate
  to the strategy once `front` moves, and the `temporal` extraction must
  preserve current nested behavior exactly. This is why step 3 is gated on the
  nested/recursive tests and kept last.
- **Non-temporal strategies have no `front`/next-event-time** — `next_event_time`
  returns `None` (or `global_time`), matching how a step-only composite behaves
  under temporal today.
- **Do not over-fit `advance_tick`'s signature to temporal.** It is temporal's
  primitive; `workflow` never calls it. A third pattern that is neither pure-DAG
  nor pure-temporal (e.g. an ensemble sweep) may reveal one more primitive —
  add it then, not speculatively.
