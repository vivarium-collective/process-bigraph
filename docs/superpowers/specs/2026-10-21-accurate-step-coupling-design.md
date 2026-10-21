# Accurate Step Coupling — Design

**Status:** proposed
**Date:** 2026-10-21
**Repo:** `vivarium-collective/process-bigraph`
**Scope:** Approach A (this spec). Approach B (explicit port directions) is a named follow-on, out of scope here.

## Problem

`build_step_network` derives the step-dependency graph from each step's **wiring**, not its **interface**. Because composites wire shared stores (`bulk`, `unique`, `listeners`, `environment`, `periplasm`) *inout* and at container granularity, a step that only *reads* such a store is registered as a *producer* of it. Across hundreds of steps sharing a handful of big containers this fabricates producer↔consumer loops that don't reflect real data flow — ~**194 spurious cycles** in `ecoli_baseline` alone. Those cycles force the scheduler into priority tie-breaks and, per-model, into hand-assigned priorities (e.g. `reactor_bird_coupled`). The coupling the scheduler sees is far denser than the coupling that actually exists.

The accuracy that *does* exist is **asymmetric**: a step can narrow its *inputs* to real edges via `triggers()`, but its *outputs* come straight from the inflated wiring with no equivalent narrowing. This spec closes that asymmetry.

## Goal

Make the step-dependency graph reflect **genuine** reads and writes, so:
- the ~194 spurious cycles in `ecoli_baseline` collapse to the genuine handful;
- process coupling is inferred from interfaces, not from inflated wiring;
- **zero changes to any composite or process** are required (existing `outputs()` interfaces are already honest; we start trusting them);
- priority ordering is unchanged, and the #257 priority-guard remains as the safety net for the real cycles.

## Non-goals

- Replacing priority ordering with explicit phases/layers (the Class-2 re-architecture) — explicitly deferred.
- Approach B (a declared per-port `in`/`out`/`inout` contract) — named as the follow-on in §Follow-on, not built here.
- Any change to `reactor_bird_coupled`'s genuine tied-priority cycles — those are real feedback and remain the modeller's priority-assignment task.

## How it works today (baseline)

In `process_bigraph/scheduling.py::build_step_network(steps)`, per step:

1. **Input edges.** `input_paths = find_leaves(step['inputs'])`; narrowed to `triggers()` ports if the step defines them. Each input path adds the step to `nodes[path]['after']` (consumer).
2. **Output edges.** `out_wiring = step.get('_dep_outputs', step.get('outputs', {}))`; `output_paths = find_leaves(out_wiring)`. Each output path adds the step to `nodes[path]['before']` (producer), with two mitigations:
   - **self-loop skip:** a path also present in this step's `input_paths` is not registered as its output (`if explode not in input_paths`);
   - **`structural_outputs`:** an output port whose interface type is `node`/map-of-node (e.g. `division`'s `agents`) is recorded as structural and later passes do not treat it as a producer of the data leaves beneath it (#248/#250).
3. **Container propagation** (second/third pass): a step that outputs a *parent* store is linked to steps reading *child* leaves under it, and child outputs propagate to parent inputs.
4. `determine_steps` runs producers before consumers per store; a cycle falls through to priority order; `find_step_cycle` (post-#257) allows distinct-priority cycles and rejects tied-priority ones.

**Failure mode.** `output_paths` is taken from `out_wiring` (the inflated wiring). The self-loop skip only cancels inflation when the *same* step reads and writes the *same* leaf; it does nothing for the common case where step A inout-touches a shared/container store that step B genuinely reads. Those become `before`/`after` edges with no real data flow, and container propagation multiplies them — the 194.

## Design (Approach A)

Two changes to `build_step_network`, both confined to how `output_paths` and producer registration are computed. Nothing else in scheduling, in `determine_steps`, or in any composite changes.

### A1 — Narrow outputs to the interface (symmetric with `triggers()`)

Compute `output_paths` from the step's **`outputs()` interface ports** intersected with the wiring, instead of from the raw `out_wiring`:

- let `out_ports = set(step['instance'].interface().get('outputs', {}).keys())`;
- restrict the wiring used for `find_leaves` to ports in `out_ports` (keeping framework `_layer_*` ports);
- preserve the existing `_dep_outputs` override and `structural_outputs` recording unchanged — they continue to take precedence / annotate.

This is the output-side analogue of the `triggers()` input narrowing already in step 1. For a step whose wiring lists `bulk`/`unique`/`listeners`/`periplasm` as outputs but whose `outputs()` is `['listeners']`, only `listeners` is now a producer edge.

### A2 — Promote the self-loop skip to a general inout (read-modify-write) rule

A port present in **both** `inputs()` and `outputs()` of a step is read-modify-write and must never be a **cross-step** producer edge (it is the step mutating its own accessed store in place, exactly like a shared accumulator). Concretely, when registering producers, exclude any output leaf that is also a leaf of the step's **full `inputs()` interface** — *not* the possibly-trigger-narrowed `input_paths`. This matters: today's self-loop skip tests against `input_paths`, so when `triggers()` narrows `input_paths` to a subset, an inout store dropped from `input_paths` is no longer skipped and leaks back as a false producer. A2 tests inout against the complete input interface so the skip is independent of trigger narrowing. This makes `division`'s inout `bulk`/`unique`/`listeners` (its `outputs()` returns `inputs()` + `agents`) non-producing, leaving only the structural `agents` output — which `structural_outputs`/#250 already exempts from member-leaf production.

### Interaction with existing mechanisms

- **`triggers()`** (inputs) — unchanged; A makes outputs symmetric to it.
- **`_dep_outputs`** — still wins when a step provides it (explicit override path, a mini-precursor of B).
- **`structural_outputs` / #248 / #250** — unchanged; A2 ensures `division`'s data ports stop producing, and the structural `agents` port keeps its existing treatment.
- **Priority-guard (#257)** — unchanged; it now fires only on the genuinely remaining cycles, which is the intent.

## Backward compatibility

- **No composite/process edits.** Every composite keeps its wiring; A changes only which edges the scheduler *infers* from that wiring plus the already-published `outputs()` interface.
- **Behavior-change surface.** A can only **remove** producer edges (narrow outputs, drop inout producers); it never adds edges. Removing a *false* edge cannot change correct results. The one real risk is a process whose `outputs()` **omits a genuine write** (dishonest interface): A would then drop a real producer edge and a consumer could run too early. This is caught by the validation below; interfaces are contractually expected to be honest, and `_dep_outputs` is the escape hatch for any exception.

## Validation & testing

1. **Cycle-count measurement.** On `ecoli_baseline` (and the sulfa/folate composite), record `find_step_cycle`'s allowed-cycle warning count before/after. Target: 194 → single digits; the residue should be only genuine feedback.
2. **Determinism preserved.** Build + run each of {`ecoli_baseline` single-lineage, colony, the folate/sulfa injected composite} for N ticks under `PYTHONHASHSEED` 0 and 1, before and after A; assert the state fingerprint is **bit-identical across hash seeds** and that per-arm results are unchanged vs. baseline (no dropped real edge changed any output).
3. **Existing scheduler tests** (`tests.py` cycle/dependency/accumulator/structural/ascending-wire + `process_bigraph/tests/`) stay green.
4. **New unit tests:** (a) a two-step composite where A inout-touches a store B reads — asserts no cycle and no A→B edge; (b) a step whose `outputs()` is narrower than its wiring — asserts only interface outputs produce; (c) a `division`-shaped structural step — asserts only `agents` is treated structurally and no data-port producers remain.

## Follow-on (Approach B, not in this spec)

Promote the implicit `inputs()`/`outputs()`-∩-wiring inference into a **declared per-port contract**: a process states each port `in` / `out` / `inout` in `describe()`, and `build_step_network` uses it verbatim when present, falling back to A's inference otherwise. It makes coupling self-documenting and exact (the precision escape-hatch for any process A infers wrong) and is adopted gradually. B does not fix today's 194 by itself — A does — so B is sequenced after A lands and the residue is measured.

## Risks

| Risk | Mitigation |
|---|---|
| A process's `outputs()` is dishonest (omits a real write) → dropped real edge | Validation #2 (results unchanged vs baseline) catches it; `_dep_outputs` override; such an interface is a bug to fix regardless. |
| A hidden composite relied on a *false* ordering edge for correctness | That ordering was hash-seed-fragile already; validation #2 surfaces any output change; priority-guard still gates determinism. |
| Container-propagation edge not covered by A1/A2 | New test (c) + the cycle-count measurement expose any residual false cycle; iterate on the propagation pass if the residue isn't genuine. |
