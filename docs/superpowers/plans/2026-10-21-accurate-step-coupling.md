# Accurate Step Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `build_step_network` derive step-dependency edges from each step's genuine interface (reads/writes), eliminating the ~194 spurious cycles in `ecoli_baseline`, with zero composite changes.

**Architecture:** Two confined changes in `process_bigraph/scheduling.py::build_step_network`: (A1) compute `output_paths` from the `outputs()` interface ∩ wiring — the output-side analogue of the existing `triggers()` input narrowing; (A2) a general read-modify-write rule — an output leaf that is also a leaf of the step's full `inputs()` interface is never a cross-step producer. Nothing else in scheduling, `determine_steps`, the priority-guard, or any composite changes.

**Tech Stack:** Python 3.12, pytest, process-bigraph (`process_bigraph/scheduling.py`, `tests.py`, `process_bigraph/tests/`).

**Spec:** `docs/superpowers/specs/2026-10-21-accurate-step-coupling-design.md`

## Global Constraints

- **Zero composite/process edits.** Only `process_bigraph/scheduling.py` (+ tests) change.
- **A may only REMOVE producer edges, never add.** (Backward-compat invariant — removing a false edge can't change correct results.)
- **`_dep_outputs` override keeps precedence**: when a step provides `_dep_outputs`, trust it verbatim (no A1 narrowing).
- **`structural_outputs` / #248 / #250 behavior is preserved** (structural `node`/map-of-node outputs keep their member-leaf exemption).
- **Priority-guard (`find_step_cycle`, #257) is unchanged** — it stays as the safety net for genuinely remaining cycles.
- **The inout test in A2 uses the FULL `inputs()` leaves**, not the possibly-trigger-narrowed `input_paths`.

## Review Focus

- **`_dep_outputs` present:** a step that sets `_dep_outputs` must bypass A1 narrowing entirely (explicit override wins) → Task 1 test `test_dep_outputs_override_bypasses_narrowing`.
- **`outputs()` raises / missing:** a step whose `interface()` has no `outputs` key or whose `.outputs()` throws must fall back to the raw wiring, not crash → Task 1 test `test_missing_output_schema_falls_back_to_wiring`.
- **`triggers()`-narrowed inout store:** a store a step reads (but isn't a trigger) and also writes must still be skipped as a producer → Task 2 test `test_triggered_step_inout_store_is_not_a_producer`.
- **Structural (`division`-shaped) step:** `outputs() == inputs() + {node port}` must leave only the node port structural and register no data-port producers → Task 2 test `test_division_shaped_step_has_no_data_producers`.
- **Genuine distinct output survives:** a step that genuinely writes a store it does NOT read must still register as its producer (A must not over-narrow) → Task 1 test `test_genuine_output_still_produces`.

---

### Task 1: A1 — narrow `output_paths` to the `outputs()` interface

**Files:**
- Modify: `process_bigraph/scheduling.py:379-383` (the `out_wiring` / `output_paths` computation)
- Test: `tests.py` (add tests beside the existing scheduler tests, ~after `test_structural_node_output_is_not_a_producer_of_member_leaves`)

**Interfaces:**
- Consumes: `schema = step['instance'].interface()` (already computed earlier in the loop; `schema['outputs']` is the declared output-port map), `find_leaves`, `step['outputs']`, `step.get('_dep_outputs')`.
- Produces: `ancestors[step_key]['output_paths']` now contains only leaves of ports declared in `outputs()` (or the `_dep_outputs` override), plus `_layer_*` framework ports. `composite.node_dependencies[path]['before']` is the producer set later tasks/tests inspect.

- [ ] **Step 1: Write the failing tests**

```python
# in tests.py, near the other scheduler tests
def test_output_wired_but_not_declared_is_not_a_producer(core):
    """A1: a store wired as a step's output but absent from its outputs()
    interface is inout-inflation, not a real write — the step must not be
    registered as its producer."""
    class ReaderWriter(Step):
        # declares it writes only 'w'; 'r' is read-only even though the
        # composite wires 'r' as an output too (inout inflation).
        def inputs(self):
            return {'r': 'float', 'w': 'float'}
        def outputs(self):
            return {'w': 'float'}
        def update(self, state):
            return {'w': state['r'] + state['w']}

    core.register_link('ReaderWriter', ReaderWriter)
    document = {
        'rw': {'_type': 'step', 'address': 'local:ReaderWriter', 'priority': 1.0,
               'inputs': {'r': ['r'], 'w': ['w']},
               'outputs': {'r': ['r'], 'w': ['w']}},   # 'r' wired out = inflation
        'r': 1.0, 'w': 0.0}
    sim = Composite({'state': document}, core=core)
    assert ('rw',) not in sim.node_dependencies.get(('r',), {}).get('before', set())
    assert ('rw',) in sim.node_dependencies.get(('w',), {}).get('before', set())


def test_genuine_output_still_produces(core):
    """A1 must only REMOVE false edges: a store a step genuinely writes and
    does not read must still register the step as its producer."""
    class Writer(Step):
        def inputs(self):
            return {'src': 'float'}
        def outputs(self):
            return {'dst': 'float'}
        def update(self, state):
            return {'dst': state['src']}

    core.register_link('Writer', Writer)
    document = {
        'wr': {'_type': 'step', 'address': 'local:Writer', 'priority': 1.0,
               'inputs': {'src': ['src']}, 'outputs': {'dst': ['dst']}},
        'src': 2.0, 'dst': 0.0}
    sim = Composite({'state': document}, core=core)
    assert ('wr',) in sim.node_dependencies.get(('dst',), {}).get('before', set())


def test_dep_outputs_override_bypasses_narrowing(core):
    """`_dep_outputs`, when present, is the explicit dependency-output
    override and must be used verbatim — A1 must not re-narrow it."""
    class OnlyW(Step):
        def inputs(self):
            return {'w': 'float'}
        def outputs(self):
            return {'w': 'float'}
        def update(self, state):
            return {'w': state['w'] + 1.0}

    core.register_link('OnlyW', OnlyW)
    document = {
        'ow': {'_type': 'step', 'address': 'local:OnlyW', 'priority': 1.0,
               'inputs': {'w': ['w']}, 'outputs': {'w': ['w']},
               '_dep_outputs': {'extra': ['extra']}},  # override names a different store
        'w': 0.0, 'extra': 0.0}
    sim = Composite({'state': document}, core=core)
    # override is honored: 'extra' is produced, declared 'w' is not (override replaces)
    assert ('ow',) in sim.node_dependencies.get(('extra',), {}).get('before', set())


def test_missing_output_schema_falls_back_to_wiring(core):
    """A step whose interface() exposes no 'outputs' key must fall back to the
    raw wiring rather than crash (defensive; mirrors the existing input path)."""
    class NoOutSchema(Step):
        def interface(self):
            return {'inputs': {'a': 'float'}}   # no 'outputs' key
        def inputs(self):
            return {'a': 'float'}
        def outputs(self):
            return {}
        def update(self, state):
            return {}
    core.register_link('NoOutSchema', NoOutSchema)
    document = {
        'ns': {'_type': 'step', 'address': 'local:NoOutSchema', 'priority': 1.0,
               'inputs': {'a': ['a']}, 'outputs': {'a': ['a']}},
        'a': 0.0}
    # must not raise during build
    Composite({'state': document}, core=core)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests.py -k "output_wired_but_not_declared or genuine_output_still_produces or dep_outputs_override or missing_output_schema" -v`
Expected: `test_output_wired_but_not_declared_is_not_a_producer` FAILS (step is currently a producer of `r`). The other three may pass already — keep them (they pin invariants A1 must preserve).

- [ ] **Step 3: Implement A1**

Replace `process_bigraph/scheduling.py:379-383`:

```python
        # Compute output paths once per step
        out_wiring = step.get('_dep_outputs', step.get('outputs', {}))
        if ancestors[step_key]['output_paths'] is None:
            ancestors[step_key]['output_paths'] = find_leaves(
                out_wiring, path=step_key[:-1])
```

with:

```python
        # Compute output paths once per step. A `_dep_outputs` override is
        # trusted verbatim; otherwise narrow the wired outputs to the step's
        # declared outputs() ports (the output-side analogue of the triggers()
        # input narrowing above) — a port wired as an output but not declared
        # in outputs() is inout-inflation, not a real write. `_layer_*`
        # framework ports are always kept.
        dep_outputs = step.get('_dep_outputs')
        if dep_outputs is not None:
            out_wiring = dep_outputs
        else:
            wiring = step.get('outputs', {})
            # Distinguish an ABSENT outputs schema (key missing -> `None`) from
            # an explicitly EMPTY one (`{}`). Absent = can't narrow safely, fall
            # back to the raw wiring; empty = the step declares no outputs, so
            # narrow to nothing (only `_layer_*` survive).
            out_schema_raw = (schema.get('outputs')
                              if isinstance(schema, dict) else None)
            out_ports = (set(out_schema_raw.keys())
                         if out_schema_raw is not None else None)
            if isinstance(wiring, dict) and out_ports is not None:
                out_wiring = {k: v for k, v in wiring.items()
                              if k in out_ports or str(k).startswith('_layer')}
            else:
                out_wiring = wiring
        if ancestors[step_key]['output_paths'] is None:
            ancestors[step_key]['output_paths'] = find_leaves(
                out_wiring, path=step_key[:-1])
```

(The later `structural_outputs` block at lines 388-393 already reads `out_wiring` and `schema.get('outputs')`; declared structural ports survive the narrowing, so it is unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests.py -k "output_wired_but_not_declared or genuine_output_still_produces or dep_outputs_override or missing_output_schema" -v`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/scheduling.py tests.py
git commit -m "fix(scheduling): narrow output_paths to the outputs() interface (A1)"
```

---

### Task 2: A2 — read-modify-write rule against the full `inputs()`

**Files:**
- Modify: `process_bigraph/scheduling.py:399-418` (the producer/consumer registration loop)
- Test: `tests.py` (beside Task 1's tests)

**Interfaces:**
- Consumes: `step.get('inputs', {})` (the FULL input wiring), `find_leaves`, `ancestors[step_key]['input_paths']`, `output_paths`, `explode_path`.
- Produces: producer registration (`nodes[path]['before']`) now also skips any output leaf that is a leaf of the full `inputs()` wiring, independent of `triggers()` narrowing.

- [ ] **Step 1: Write the failing tests**

```python
def test_triggered_step_inout_store_is_not_a_producer(core):
    """A2: a store a step both reads and writes must not be a cross-step
    producer even when triggers() narrowed its edge-creating inputs so that
    the store is absent from input_paths."""
    class Triggered(Step):
        def inputs(self):
            return {'trig': 'float', 'acc': 'list'}   # reads trig AND acc
        def triggers(self):
            return {'trig': 'float'}                   # only trig makes edges
        def outputs(self):
            return {'acc': 'list'}                     # read-modify-writes acc
        def update(self, state):
            return {'acc': state['acc'] + [state['trig']]}

    core.register_link('Triggered', Triggered)
    document = {
        't': {'_type': 'step', 'address': 'local:Triggered', 'priority': 1.0,
              'inputs': {'trig': ['trig'], 'acc': ['acc']},
              'outputs': {'acc': ['acc']}},
        'trig': 0.0, 'acc': []}
    sim = Composite({'state': document}, core=core)
    # acc is inout: NOT produced (even though acc is not in the trigger-narrowed
    # input_paths), so a peer reading acc won't falsely depend on this step.
    assert ('t',) not in sim.node_dependencies.get(('acc',), {}).get('before', set())


def test_two_steps_sharing_an_inout_store_do_not_form_a_false_cycle(core):
    """End-to-end: step A reads shared 's' + writes 'p'; step B reads 'p' +
    writes shared 's' (inout). Under the old wiring both looked like producers
    of 's', forming A<->B. With A1+A2 neither produces 's', so no cycle."""
    class A(Step):
        def inputs(self):
            return {'s': 'float', 'p': 'float'}
        def outputs(self):
            return {'p': 'float', 's': 'float'}   # s is inout (reads s)
        def update(self, state):
            return {'p': state['s'], 's': state['s']}
    class B(Step):
        def inputs(self):
            return {'p': 'float', 's': 'float'}
        def outputs(self):
            return {'s': 'float', 'p': 'float'}   # p is inout (reads p)
        def update(self, state):
            return {'s': state['p'], 'p': state['p']}
    core.register_link('A', A)
    core.register_link('B', B)
    document = {
        'a': {'_type': 'step', 'address': 'local:A', 'priority': 1.0,
              'inputs': {'s': ['s'], 'p': ['p']}, 'outputs': {'p': ['p'], 's': ['s']}},
        'b': {'_type': 'step', 'address': 'local:B', 'priority': 2.0,
              'inputs': {'p': ['p'], 's': ['s']}, 'outputs': {'s': ['s'], 'p': ['p']}},
        's': 0.0, 'p': 0.0}
    # must build without the cycle check rejecting it
    Composite({'state': document}, core=core)


def test_division_shaped_step_has_no_data_producers(core):
    """A2 + structural: a division-shaped step (outputs() == inputs() + a
    node/map-of-node port) registers no data-port producers; only the node
    port is structural."""
    class Divider(Step):
        def inputs(self):
            return {'bulk': 'float', 'agents': {'_type': 'map', '_value': 'node'}}
        def outputs(self):
            return {'bulk': 'float', 'agents': {'_type': 'map', '_value': 'node'}}
        def update(self, state):
            return {}
    core.register_link('Divider', Divider)
    document = {
        'd': {'_type': 'step', 'address': 'local:Divider', 'priority': 1.0,
              'inputs': {'bulk': ['bulk'], 'agents': ['agents']},
              'outputs': {'bulk': ['bulk'], 'agents': ['agents']}},
        'bulk': 0.0, 'agents': {}}
    sim = Composite({'state': document}, core=core)
    # bulk is inout -> not produced
    assert ('d',) not in sim.node_dependencies.get(('bulk',), {}).get('before', set())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests.py -k "triggered_step_inout_store or two_steps_sharing_an_inout_store or division_shaped_step_has_no_data" -v`
Expected: `test_triggered_step_inout_store_is_not_a_producer` FAILS (acc is produced because it's not in the trigger-narrowed `input_paths`); the inout cycle test may already pass via A1 but keep it as the end-to-end guard.

- [ ] **Step 3: Implement A2**

Replace the producer loop at `process_bigraph/scheduling.py:399-418`:

```python
        input_paths = ancestors[step_key]['input_paths'] or []
        output_paths = ancestors[step_key]['output_paths'] or []

        # Track which steps consume/produce each path
        for input_path in input_paths:
            path = tuple(input_path)
            nodes.setdefault(path, {'before': set(), 'after': set()})
            nodes[path]['after'].add(step_key)

        for output_path in output_paths:
            exploded_path = explode_path(output_path)[1:]
            for explode in exploded_path:
                # Skip self-loops: don't register this step as a
                # producer of paths it also consumes. Self-loops
                # can't trigger (a step runs at most once per cycle)
                # and they clutter the dependency graph.
                if explode not in input_paths:
                    path = tuple(explode)
                    nodes.setdefault(path, {'before': set(), 'after': set()})
                    nodes[path]['before'].add(step_key)
```

with:

```python
        input_paths = ancestors[step_key]['input_paths'] or []
        output_paths = ancestors[step_key]['output_paths'] or []

        # Full input leaves (the complete inputs() wiring, independent of any
        # triggers() narrowing of input_paths) — the read side of the
        # read-modify-write test below. input_paths may be a trigger-narrowed
        # SUBSET of these, so testing inout against the full set is what makes
        # the skip hold for triggered steps.
        full_input_leaves = {
            tuple(p) for p in find_leaves(step.get('inputs', {}), path=step_key[:-1])}

        # Track which steps consume/produce each path
        for input_path in input_paths:
            path = tuple(input_path)
            nodes.setdefault(path, {'before': set(), 'after': set()})
            nodes[path]['after'].add(step_key)

        for output_path in output_paths:
            exploded_path = explode_path(output_path)[1:]
            for explode in exploded_path:
                # Read-modify-write: a store this step both reads and writes is
                # never a cross-step producer (it mutates its own accessed store
                # in place, like a shared accumulator). Test against the FULL
                # inputs() leaves, not the trigger-narrowed input_paths.
                if tuple(explode) in full_input_leaves or explode in input_paths:
                    continue
                path = tuple(explode)
                nodes.setdefault(path, {'before': set(), 'after': set()})
                nodes[path]['before'].add(step_key)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests.py -k "triggered_step_inout_store or two_steps_sharing_an_inout_store or division_shaped_step_has_no_data" -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/scheduling.py tests.py
git commit -m "fix(scheduling): read-modify-write rule against full inputs() (A2)"
```

---

### Task 3: Regression + real-composite validation gate

**Files:**
- Modify: none (validation only)
- Test: full `tests.py` + `process_bigraph/tests/`; plus the out-of-repo integration gate below.

**Interfaces:**
- Consumes: the A1+A2 changes from Tasks 1-2.
- Produces: a go/no-go merge gate (regression-green + cycle-count drop + determinism preserved).

- [ ] **Step 1: Full in-repo suite stays green**

Run: `.venv/bin/python -m pytest tests.py process_bigraph/tests/ -q`
Expected: no new failures vs. `origin/main` (same pass/skip counts; the existing cycle/accumulator/structural/ascending-wire tests still pass — A1/A2 must not regress them).

- [ ] **Step 2: Cycle-count drop on `ecoli_baseline` (integration, sms-ecoli venv)**

This needs v2ecoli + a ParCa cache, which live outside this repo. Run in the sms-ecoli venv against the editable process-bigraph from this branch:

```bash
# in ~/code/sms-ecoli--<worktree>, PYTHONPATH set, with process_bigraph editable = this branch
PYTHONHASHSEED=0 python - <<'PY'
import warnings
from scripts.run_comparison_ensemble import _build_v2ecoli
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    _build_v2ecoli(seed=0, condition='basal', cache_dir='out/cache_folate_met_minimal', overrides={})
    cyc = [str(x.message) for x in w if 'distinct-priority' in str(x.message)]
print('allowed-cycle warnings:', cyc[0][:80] if cyc else '(none)')
PY
```
Expected: the allowed-cycle count drops from ~194 to single digits (record the exact number). If it is not single digits, the residue is either genuine feedback (acceptable, document it) or an uncovered false-cycle source (open a follow-up per the spec's risk row on container propagation).

- [ ] **Step 3: Determinism + results unchanged (integration, sms-ecoli venv)**

For each composite in {`ecoli_baseline` single-lineage, `ecoli_baseline` colony/keep-all-daughters, the folate/sulfa injected `#96` composite}, build+run N=3 ticks (single-lineage/injected) or to the first division (colony) under `PYTHONHASHSEED` 0 and 1, computing the finite-leaf state fingerprint (reuse the determinism harness: `md5` of sorted finite leaves + inf/nan counts):
- assert the fingerprint is **bit-identical across hash seeds** (determinism preserved), AND
- assert it is **identical before vs. after** A1/A2 (no real edge was dropped — results unchanged). Capture both fingerprints and diff.

Expected: identical in both dimensions. A mismatch before-vs-after means A dropped a genuine producer edge (a dishonest `outputs()`); investigate that process's interface or add `_dep_outputs` per the spec.

- [ ] **Step 4: Record results + commit the measurements note**

Append the measured cycle counts and fingerprints to a short results block at the bottom of the spec file, then:

```bash
git add docs/superpowers/specs/2026-10-21-accurate-step-coupling-design.md
git commit -m "docs(spec): record accurate-step-coupling validation results"
```
