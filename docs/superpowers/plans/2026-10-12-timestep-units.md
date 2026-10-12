# Timestep Units Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a process declare its update `interval` with a time unit (`"1 min"`), auto-convert it to canonical seconds via pint, and display the declared form (with the canonical value on hover) in the composite viewer.

**Architecture:** Units live only in Python (bigraph-schema's existing pint `UnitRegistry`). A process is normalized at realize time to `interval` = canonical **float seconds** (scheduler untouched) plus a sibling `interval_units` string carrying the declared unit. The workbench serves `interval` + `interval_units` + a Python-built `interval_label`; the vendored loom frontend just displays those strings.

**Tech Stack:** Python 3 + pint (via `bigraph_schema.units`); `plum` multimethod dispatch (process-bigraph schema types); FastAPI (workbench API); vendored Vite/React/TypeScript app (bigraph-loom).

**Spec:** `docs/superpowers/specs/2026-10-12-timestep-units-design.md` (in the process-bigraph repo). Executors read both.

## Global Constraints

- **Canonical time base is seconds, fixed.** `process['interval']` is ALWAYS a plain float in seconds. No per-composite override. The scheduler (`composite.py`), `_require_advancing_interval`, and `calculate_timestep` are NOT modified.
- **Backward compatible.** A bare numeric `interval` keeps its exact current behavior and serialization (a number, not a string). Only unit-declaring specs gain the new string form.
- **All pint work is Python-side.** The loom frontend never parses or converts units; it displays strings/numbers computed in Python.
- **Centralize unit logic in `bigraph_schema.units`** — do not duplicate pint calls in process-bigraph or the workbench; import `parse_time_interval` / `format_time_interval`.
- **No AI attribution** in any commit message (no `Co-Authored-By`, no generated-with footer).
- **One worktree + one PR per repo**, off `origin/main`, merged in order: bigraph-schema → process-bigraph → vivarium-workbench. The process-bigraph worktree already exists at `~/code/process-bigraph--timestep-units` (branch `timestep-units`). Create the other two at execution time:
  - `git -C ~/code/bigraph-schema fetch origin main && git -C ~/code/bigraph-schema worktree add ~/code/bigraph-schema--timestep-units -b timestep-units origin/main`
  - `git -C ~/code/vivarium-workbench fetch origin main && git -C ~/code/vivarium-workbench worktree add ~/code/vivarium-workbench--timestep-units -b timestep-units origin/main`
- **Editable installs point at canonical checkouts.** Run each repo's tests with its worktree prepended: `PYTHONPATH=<worktree> pytest ...`, and confirm `python -c "import <pkg>; print(<pkg>.__file__)"` resolves inside the worktree before trusting results.

## Review Focus

- **Non-time unit string** (`interval: "1 meter"`, `interval: "hello"`): must raise a clear `ValueError` naming the value, never silently realize to 0 or crash deep in the scheduler. → Task 1 tests.
- **Multiple processes at different declared units** (one `"1 min"`, one `"30 s"`): the scheduler takes the smallest interval *in seconds* and must advance `global_time` correctly across both. → Task 2 behavior test.
- **Render → realize → render idempotence** for a unit-declared interval: a `"1 min"` process re-serialized must stay `"1 min"` (not drift to `60`, `"60 s"`, or `60.0 minute`). → Task 2 round-trip test.
- **Fractional declared value** (`"1.5 min"`, `"0.5 min"`): converts to the right seconds and formats to a clean label (`"1.5 min"`, not `"1.4999999 min"`). → Task 1 format tests.
- **Legacy bare-number spec unchanged end-to-end**: `interval: 60` still serializes as the number `60`, `interval_units` is absent/None, and it runs identically. → Task 2 tests + Task 3 label fallback (`"60 s"`).

---

## Task 1: Unit helpers in bigraph-schema

**Repo/worktree:** `~/code/bigraph-schema--timestep-units`

**Files:**
- Modify: `bigraph_schema/units.py` (add two functions + inline tests, matching this module's existing `test_*()`-function convention)

**Interfaces:**
- Consumes: `get_quantity_registry()` (already in `units.py`).
- Produces:
  - `parse_time_interval(value) -> tuple[float, str | None]` — `value` is a number → `(float(value), None)`; a pint time string → `(seconds, unit_name)`; anything else / non-time / unparseable → raises `ValueError`.
  - `format_time_interval(seconds: float, unit: str | None) -> str` — `unit is None` → `"{seconds:g} s"`; else the declared magnitude in `unit` + short symbol, e.g. `"1 min"`.

- [ ] **Step 1: Write the failing tests**

Add to `bigraph_schema/units.py` (bottom, before the `if __name__` block):

```python
def test_parse_time_interval():
    # bare number -> seconds, no declared unit
    assert parse_time_interval(60) == (60.0, None)
    assert parse_time_interval(1.5) == (1.5, None)

    # pint strings -> canonical seconds + declared unit name
    assert parse_time_interval('1 min') == (60.0, 'minute')
    assert parse_time_interval('2 minutes') == (120.0, 'minute')
    assert parse_time_interval('30 s') == (30.0, 'second')
    assert parse_time_interval('1.5 min') == (90.0, 'minute')

    # non-time dimensionality is rejected, naming the value
    import pytest
    with pytest.raises(ValueError, match='1 meter'):
        parse_time_interval('1 meter')
    with pytest.raises(ValueError, match='nonsense'):
        parse_time_interval('nonsense')


def test_format_time_interval():
    assert format_time_interval(60.0, 'minute') == '1 min'
    assert format_time_interval(120.0, 'minute') == '2 min'
    assert format_time_interval(90.0, 'minute') == '1.5 min'
    assert format_time_interval(30.0, 'second') == '30 s'
    # no declared unit -> canonical seconds label
    assert format_time_interval(90.0, None) == '90 s'
    assert format_time_interval(0.5, None) == '0.5 s'


def test_time_interval_round_trip():
    # parse then format returns the author's declared form
    seconds, unit = parse_time_interval('1 min')
    assert format_time_interval(seconds, unit) == '1 min'
    seconds, unit = parse_time_interval(60)
    assert format_time_interval(seconds, unit) == '60 s'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=~/code/bigraph-schema--timestep-units python -m pytest bigraph_schema/units.py -v`
Expected: FAIL — `NameError: name 'parse_time_interval' is not defined`.

- [ ] **Step 3: Implement the two helpers**

Add to `bigraph_schema/units.py` (after `set_quantity_registry`, before `render_coefficient`):

```python
def parse_time_interval(value):
    """Normalize a declared timestep to (canonical_seconds, declared_unit).

    ``value`` may be a number (interpreted as seconds, declared_unit None) or a
    pint-parseable time string ("1 min", "2 minutes", "30 s"). A non-time
    quantity or an unparseable string raises ValueError naming the value.
    """
    if isinstance(value, bool):
        raise ValueError(f'interval must be a number or time string, got {value!r}')
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, str):
        ureg = get_quantity_registry()
        try:
            quantity = ureg.Quantity(value)
        except Exception as error:  # pint raises several types; normalize them
            raise ValueError(
                f'could not parse interval {value!r} as a time quantity: {error}')
        if not quantity.check('[time]'):
            raise ValueError(
                f'interval {value!r} is not a time quantity '
                f'(dimensionality {quantity.dimensionality})')
        return float(quantity.to('second').magnitude), str(quantity.units)
    raise ValueError(f'interval must be a number or time string, got {value!r}')


def format_time_interval(seconds, unit):
    """Render the author-facing label for a canonical-seconds interval.

    ``unit is None`` -> "<seconds> s" (canonical). Otherwise convert back into
    the declared ``unit`` and append its short symbol, e.g. "1 min".
    """
    if unit is None:
        return f'{seconds:g} s'
    ureg = get_quantity_registry()
    magnitude = ureg.Quantity(seconds, 'second').to(unit).magnitude
    symbol = f'{ureg.Unit(unit):~}'
    return f'{magnitude:g} {symbol}'
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=~/code/bigraph-schema--timestep-units python -m pytest bigraph_schema/units.py -v`
Expected: PASS (all three new tests, plus the existing `test_units_render` / `test_roots_cycle`).

- [ ] **Step 5: Commit**

```bash
cd ~/code/bigraph-schema--timestep-units
git add bigraph_schema/units.py
git commit -m "feat(units): parse_time_interval + format_time_interval for timestep units"
```

---

## Task 2: Unit-aware interval in process-bigraph

**Repo/worktree:** `~/code/process-bigraph--timestep-units` (already created)

**Depends on Task 1** (imports `parse_time_interval` / `format_time_interval` from `bigraph_schema.units`). During development, point process-bigraph at the Task-1 worktree: `PYTHONPATH=~/code/process-bigraph--timestep-units:~/code/bigraph-schema--timestep-units`.

**Files:**
- Modify: `process_bigraph/types/process.py`
  - `ProcessLink` dataclass (`:41-46`) — add `interval_units` field.
  - `reify_schema(core, schema: ProcessLink, parameters)` (`:127-146`) — parse string declarations.
  - `realize(core, schema: ProcessLink, state, path)` (`:162-182`) — parse string, stamp `interval_units` into the realized state.
  - `render(schema: ProcessLink, defaults)` (`:255-277`) — re-emit the pint string when a unit was declared; skip `interval_units` in the generic field loop.
- Create: `process_bigraph/tests/test_interval_units.py`

**Interfaces:**
- Consumes: `bigraph_schema.units.parse_time_interval`, `bigraph_schema.units.format_time_interval`; existing `Float`, `realize_link`, `reify_schema_link`.
- Produces (runtime contract other repos rely on): a realized process node carries `node['interval']` as a float in seconds and, when a unit was declared, `node['interval_units']` as the pint unit name (e.g. `"minute"`); absent/None when none was declared. A rendered process doc carries `interval` as a pint string (`"1 min"`) when a unit was declared, else the bare number.

- [ ] **Step 1: Write the failing tests**

Create `process_bigraph/tests/test_interval_units.py`:

```python
"""Timestep units: a process may declare interval as a pint string; it is
normalized to canonical seconds with the declared unit preserved as metadata."""

from process_bigraph import Composite, allocate_core


def _doc(interval):
    return {'state': {
        'level': 5.0,
        'grow': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.1},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': interval}}}


def _composite(interval):
    from process_bigraph.processes.examples import IncreaseProcess
    core = allocate_core()
    core.register_link('IncreaseProcess', IncreaseProcess)
    return Composite(_doc(interval), core=core)


def test_string_interval_realizes_to_canonical_seconds():
    c = _composite('1 min')
    grow = c.state['grow']
    assert grow['interval'] == 60.0
    assert isinstance(grow['interval'], float)
    assert grow['interval_units'] == 'minute'


def test_bare_number_interval_unchanged():
    c = _composite(60)
    grow = c.state['grow']
    assert grow['interval'] == 60.0
    assert grow.get('interval_units') is None


def test_string_and_number_intervals_advance_identically():
    by_number = _composite(60)
    by_string = _composite('1 min')
    by_number.run(120)
    by_string.run(120)
    assert by_number.state['global_time'] == by_string.state['global_time'] == 120.0
    assert by_number.state['level'] == by_string.state['level']


def test_multiple_processes_different_units_schedule_in_seconds():
    """Smallest interval wins, measured in canonical seconds — a 30 s process
    and a 1 min process advance global_time by the 30 s step."""
    from process_bigraph.processes.examples import IncreaseProcess
    core = allocate_core()
    core.register_link('IncreaseProcess', IncreaseProcess)
    doc = {'state': {
        'level': 5.0,
        'fast': {'_type': 'process', 'address': 'local:IncreaseProcess',
                 'config': {'rate': 0.1}, 'inputs': {'level': ['level']},
                 'outputs': {'level': ['level']}, 'interval': '30 s'},
        'slow': {'_type': 'process', 'address': 'local:IncreaseProcess',
                 'config': {'rate': 0.1}, 'inputs': {'level': ['level']},
                 'outputs': {'level': ['level']}, 'interval': '1 min'}}}
    c = Composite(doc, core=core)
    assert c.state['fast']['interval'] == 30.0
    assert c.state['slow']['interval'] == 60.0
    c.run(60)
    assert c.state['global_time'] == 60.0


def test_non_time_unit_interval_raises():
    import pytest
    with pytest.raises(ValueError, match='meter'):
        _composite('1 meter')


def test_serialize_schema_round_trips_declared_unit():
    """serialize_schema() runs the ProcessLink render path; a unit-declared
    process re-emits the pint string '1 min' (not 60 / '60 s' / '60.0 minute')."""
    c = _composite('1 min')
    grow = c.serialize_schema()['grow']
    assert grow['interval'] == '1 min'


def test_serialize_schema_keeps_bare_number():
    """A bare-number interval still serializes as the number (no regression)."""
    c = _composite(60)
    grow = c.serialize_schema()['grow']
    assert grow['interval'] == 60.0
```

(`serialize_schema()` is the confirmed render entry point — `composite.serialize_schema()['grow']['interval']` is `60.0` for a bare-number process on current `main`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=~/code/process-bigraph--timestep-units:~/code/bigraph-schema--timestep-units python -m pytest process_bigraph/tests/test_interval_units.py -v`
Expected: FAIL — `test_string_interval_realizes_to_canonical_seconds` errors because a string `interval` cannot realize against `Float` (and `interval_units` is absent).

- [ ] **Step 3: Implement**

In `process_bigraph/types/process.py`:

(a) Add the import near the top (with the other `bigraph_schema` imports):

```python
from bigraph_schema.units import parse_time_interval, format_time_interval
```

(b) Add the `interval_units` field to `ProcessLink` (`:41-46`):

```python
@dataclass(kw_only=True)
class ProcessLink(Link):
    """
    A link type for temporal processes, extending the base Link schema with a time interval.
    """
    interval: Float = field(default_factory=float_default(1.0))
    interval_units: typing.Optional[str] = None
```

(c) In `reify_schema(core, schema: ProcessLink, parameters)` (`:127-146`), replace the numeric-only carry with unit-aware parsing:

```python
    schema = reify_schema_link(core, schema, parameters)

    declared = parameters.get('interval')
    if isinstance(declared, str):
        seconds, unit = parse_time_interval(declared)
        schema.interval = Float(_default=seconds)
        schema.interval_units = unit
    elif isinstance(declared, (int, float)) and not isinstance(declared, bool):
        schema.interval = Float(_default=declared)

    return schema
```

(d) In `realize(core, schema: ProcessLink, state, path=())` (`:162-182`), parse the incoming value before realizing the Float, and stamp the unit into the realized state and schema:

```python
    link_schema, link_state, merges = realize_link(core, schema, state, path=path)

    raw_interval = state.get('interval')
    unit = None
    if raw_interval is not None:
        seconds, unit = parse_time_interval(raw_interval)
    else:
        seconds = None

    _, link_state['interval'], _ = realize(
        core,
        link_schema.interval,
        seconds,
        path + ('interval',))

    if unit is not None:
        link_state['interval_units'] = unit
        link_schema.interval_units = unit

    return link_schema, link_state, merges
```

(e) In `render(schema: ProcessLink, defaults=False)` (`:255-277`), skip `interval_units` in the generic field loop and emit the pint string when a unit is present:

```python
    for field_name in schema.__dataclass_fields__:
        if field_name in ('_default', 'interval_units'):
            continue
        ...  # unchanged body
    ...
    carried = getattr(schema.interval, '_default', None)
    if carried is not None:
        unit = getattr(schema, 'interval_units', None)
        result['interval'] = format_time_interval(carried, unit) if unit else carried

    return wrap_default(schema, result) if defaults else result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=~/code/process-bigraph--timestep-units:~/code/bigraph-schema--timestep-units python -m pytest process_bigraph/tests/test_interval_units.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Run the interval/scheduling regression tests**

Run: `PYTHONPATH=~/code/process-bigraph--timestep-units:~/code/bigraph-schema--timestep-units python -m pytest process_bigraph/tests/test_composite_spec.py process_bigraph/tests/test_run_composite.py -v`
Expected: PASS — existing bare-number interval behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
cd ~/code/process-bigraph--timestep-units
git add process_bigraph/types/process.py process_bigraph/tests/test_interval_units.py
git commit -m "feat(process): unit-aware interval (pint string -> canonical seconds + interval_units)"
```

---

## Task 3: Display declared timestep in the workbench viewer

**Repo/worktree:** `~/code/vivarium-workbench--timestep-units`

**Depends on Task 2** (the resolved process node now carries `interval` seconds + `interval_units`).

**Files:**
- Modify: `vivarium_workbench/env_worker.py` — `_attach_process_docs` (`:924+`), add `interval_label`.
- Modify: `vivarium_workbench/loom/src/types.ts` (`:26`) — add `intervalUnits` / `intervalLabel` to `ProcessNodeData`.
- Modify: `vivarium_workbench/loom/src/convert.ts` (`:338`) — populate the two new fields.
- Modify: `vivarium_workbench/loom/src/nodes/ProcessNode.tsx` (`:592`) — render label + tooltip.
- Modify: `vivarium_workbench/loom/src/panels/InspectorPanel.tsx` (`:53`) — render label + tooltip.
- Test: `tests/test_interval_label.py` (new, Python backend test).

**Interfaces:**
- Consumes: `bigraph_schema.units.format_time_interval`; the process node's `interval` (seconds) + `interval_units` (from Task 2); loom node field `node.interval_units`, `node.interval_label`.
- Produces: each resolved process node payload gains `interval_label` (string, e.g. `"1 min"` or `"60 s"`); loom `ProcessNodeData` gains `intervalUnits?: string | null` and `intervalLabel?: string`.

- [ ] **Step 1: Write the failing backend test**

Create `tests/test_interval_label.py`:

```python
"""_attach_process_docs stamps a human interval_label on process nodes."""

from vivarium_workbench.env_worker import _attach_process_docs


def test_attach_process_docs_adds_interval_label_for_unit():
    doc = {'grow': {'_type': 'process', 'address': 'local:X',
                    'interval': 60.0, 'interval_units': 'minute'}}
    _attach_process_docs(doc)
    assert doc['grow']['interval_label'] == '1 min'


def test_attach_process_docs_adds_seconds_label_for_bare_number():
    doc = {'grow': {'_type': 'process', 'address': 'local:X', 'interval': 90.0}}
    _attach_process_docs(doc)
    assert doc['grow']['interval_label'] == '90 s'


def test_attach_process_docs_no_label_without_interval():
    doc = {'store': {'_type': 'step', 'address': 'local:S'}}
    _attach_process_docs(doc)
    assert 'interval_label' not in doc['store']
```

- [ ] **Step 2: Run the backend test to verify it fails**

Run: `cd ~/code/vivarium-workbench--timestep-units && PYTHONPATH=~/code/vivarium-workbench--timestep-units python -m pytest tests/test_interval_label.py -v`
Expected: FAIL — `KeyError: 'interval_label'`.

- [ ] **Step 3: Implement the backend label**

In `vivarium_workbench/env_worker.py`, inside `_attach_process_docs`'s `walk()`, within the `if node.get("_type") in ("process", "step"):` block (after the existing enrichment, ~`:982`), add:

```python
                # Human-facing timestep label: the author's declared form
                # ("1 min") with canonical seconds available for a tooltip.
                # Guarded so a step (no interval) or an older core is never a crash.
                if "interval" in node and "interval_label" not in node:
                    try:
                        from bigraph_schema.units import format_time_interval
                        node["interval_label"] = format_time_interval(
                            float(node["interval"]), node.get("interval_units"))
                    except Exception:  # noqa: BLE001
                        pass
```

- [ ] **Step 4: Run the backend test to verify it passes**

Run: `cd ~/code/vivarium-workbench--timestep-units && PYTHONPATH=~/code/vivarium-workbench--timestep-units python -m pytest tests/test_interval_label.py -v`
Expected: PASS.

- [ ] **Step 5: Add the loom frontend fields (types + convert)**

In `vivarium_workbench/loom/src/types.ts`, after `interval?: number;` (`:26`):

```ts
  interval?: number;
  intervalUnits?: string | null;
  intervalLabel?: string;
```

In `vivarium_workbench/loom/src/convert.ts`, after `interval: node.interval,` (`:338`):

```ts
          interval: node.interval,
          intervalUnits: (node as { interval_units?: string | null }).interval_units ?? null,
          intervalLabel: (node as { interval_label?: string }).interval_label,
```

- [ ] **Step 6: Render the declared label + tooltip**

In `vivarium_workbench/loom/src/nodes/ProcessNode.tsx` (`:592`), replace:

```tsx
            {data.interval != null && <span> · every {data.interval}</span>}
```

with:

```tsx
            {data.interval != null && (
              <span title={`${data.interval} s`}> · every {data.intervalLabel ?? `${data.interval} s`}</span>
            )}
```

In `vivarium_workbench/loom/src/panels/InspectorPanel.tsx` (`:53`), replace:

```tsx
            {d.interval != null && <> · every {d.interval}</>}
```

with:

```tsx
            {d.interval != null && <span title={`${d.interval} s`}> · every {d.intervalLabel ?? `${d.interval} s`}</span>}
```

- [ ] **Step 7: Rebuild the vendored loom bundle**

The served graph is the gitignored Vite build `vivarium_workbench/loom/_dist`.

Run: `cd ~/code/vivarium-workbench--timestep-units/vivarium_workbench/loom && npm install && npm run build`
Expected: `tsc -b && vite build` completes with no TypeScript errors and refreshes `_dist/`.

- [ ] **Step 8: Verify in the live viewer**

Start the workbench against the tumor-tcell workspace (serve from this worktree via PYTHONPATH so the edited code + rebuilt `_dist` are used), open a composite whose process declares a unit interval (and one that declares bare `60`), open the Composite Explorer (Interactive bigraph), and confirm the process node reads "· every 1 min" (unit case) / "· every 60 s" (bare case) with the canonical seconds in the hover title.

Run: `cd ~/code/vivarium-workbench--timestep-units && PYTHONPATH=~/code/vivarium-workbench--timestep-units:~/code/process-bigraph--timestep-units:~/code/bigraph-schema--timestep-units vivarium-workbench serve --workspace ~/code/viva-tumor-tcell`
Then check `curl -s -o /dev/null -w '%{http_code}' <base>/bigraph-loom/` → 200 and inspect a process node in the Explore panel.
Expected: declared label shown; tooltip shows "<seconds> s".

- [ ] **Step 9: Commit**

```bash
cd ~/code/vivarium-workbench--timestep-units
git add vivarium_workbench/env_worker.py vivarium_workbench/loom/src/types.ts vivarium_workbench/loom/src/convert.ts vivarium_workbench/loom/src/nodes/ProcessNode.tsx vivarium_workbench/loom/src/panels/InspectorPanel.tsx tests/test_interval_label.py
git commit -m "feat(viewer): show declared timestep unit in composite viewer (label + canonical tooltip)"
```

---

## Integration & delivery

- [ ] Open PRs in dependency order (bigraph-schema → process-bigraph → vivarium-workbench). Each PR is independently green; the process-bigraph PR's behavior-equivalence test is the guard that the canonical-seconds invariant holds.
- [ ] No spec migration is bundled: the tumor-tcell composite's `interval: 60` simply renders "60 s". Re-declaring any interval as `"1 min"` etc. is a separate, optional authoring change.
