# Timestep units — design

**Date:** 2026-10-12
**Status:** Approved design, pending implementation plan
**Repos touched:** bigraph-schema, process-bigraph, vivarium-workbench

## Intent

A process's update `interval` is currently a bare number (e.g. the tumor-tcell
composite declares `interval: 60`), rendered in the composite viewer as a
meaningless "· every 60". We want timesteps to carry **time units**:

- A process author can declare `interval: "1 min"` (or `"2 minutes"`, `"30 s"`),
  and the framework automatically converts it to the simulation's canonical time
  base using pint.
- A bare number keeps working and is interpreted as the canonical unit
  (seconds), so existing specs are unchanged and `interval: 60` now reads as
  "60 s".
- The composite viewer shows the author's declared form (e.g. "every 1 min")
  with the canonical value ("60 s") available on hover.

### Decisions locked in brainstorming

1. **Declaration:** `interval` accepts either a pint-parseable string or a bare
   number. A bare number means the canonical unit. One field, backward
   compatible.
2. **Canonical time base:** **seconds, fixed** (not per-composite overridable).
   `global_time` and all scheduler arithmetic count in seconds. Every process
   interval is converted into seconds via pint.
3. **Display:** show the declared form (`every 1 min`) with the canonical value
   (`60 s`) in a tooltip/title.

Chosen approach (of two considered): **normalize at realize, carry the declared
unit as metadata**. The rejected alternative — a polymorphic `interval` carried
verbatim through spec → API → frontend — was rejected because it touches the
scheduler hot path, forces unit parsing into TypeScript (no real JS pint), and
breaks consumers expecting a numeric `interval`.

## Core invariant

> `process['interval']` is **always a canonical float in seconds**, exactly as
> today. The declared unit rides alongside as metadata; it never enters
> scheduler arithmetic.

This is what keeps the scheduler and the frontend unit-blind, and why the change
is low-risk.

## Data flow

```
author writes            realize (process-bigraph)          scheduler        viewer payload (workbench)
-------------            -------------------------          ---------        --------------------------
interval: "1 min"   ──►   interval      = 60.0  (seconds) ──► unchanged ──►  interval: 60.0
                          interval_units = "minute"                          interval_units: "minute"
                                                                             interval_label: "1 min"
interval: 60        ──►   interval      = 60.0                               interval: 60.0
                          interval_units = None                              interval_units: null
                                                                             interval_label: "60 s"
```

All pint work happens in Python (bigraph-schema's existing `UnitRegistry`). The
scheduler sees only seconds. The frontend receives ready-made strings and never
parses units.

## Components & changes

### 1. bigraph-schema — `bigraph_schema/units.py`

All time-unit logic is centralized here (the shared `UnitRegistry` and the
`get_quantity_registry()` accessor already live in this file).

Add two helpers:

- `parse_time_interval(value) -> tuple[float, str | None]`
  - `value` is an `int`/`float` → `(float(value), None)` (None = canonical, no
    explicit unit declared).
  - `value` is a `str` → parse via `get_quantity_registry().Quantity(value)`;
    require time dimensionality; return
    `(quantity.to("second").magnitude, str(quantity.units))`.
  - Non-time dimensionality (e.g. `"1 meter"`) or unparseable string → raise
    `ValueError` naming the offending value (fail-loud).
  - `None` or already-canonical passthrough as needed by callers.

- `format_time_interval(seconds: float, unit: str | None) -> str`
  - `unit is None` → `"{trimmed} s"` (e.g. `"60 s"`).
  - otherwise rebuild `Quantity(seconds, "second").to(unit)` and format the
    magnitude trimmed (no trailing `.0`) with the short unit abbreviation
    (e.g. `"1 min"`).

Tests: a table over number / `"1 min"` / `"2 minutes"` / `"30 s"` / bad-unit
(raises), plus a `format_time_interval` round-trip.

### 2. process-bigraph — `process_bigraph/types/process.py`

- **Schema field:** add `interval_units: Optional[str] = None` to `ProcessLink`
  (around the existing `interval: Float` at `process.py:46`). `CompositeLink`
  inherits it; `StepLink` is unaffected (steps have no interval).

- **`reify_schema` (`process.py:~129-144`)** and **`realize`
  (`process.py:~162-182`)**: before the interval is treated as a `Float`, detect
  a string value and run it through `bigraph_schema.units.parse_time_interval`.
  Set the `Float` magnitude to the returned seconds and set `interval_units` to
  the returned unit. A numeric value stays as-is with `interval_units = None`.

- **`serialize_schema` (`process.py:~255-277`)**: when `interval_units` is set,
  re-emit `interval` as the pint string (reconstruct the declared magnitude via
  `format_time_interval`) so a round-trip of an authored spec is faithful and
  human-readable. When `interval_units` is None, emit the bare number exactly as
  today → **no regression for existing specs**.

- **`default` (`process.py:~150-159`)**: unchanged — numeric default `1.0`, unit
  `None`.

- **Registered `interval` type (`__init__.py:~75`)**: left inheriting `float`;
  string handling is done in the `ProcessLink` multimethods above, not by the
  registered type.

- **Scheduler (`composite.py`)**, `_require_advancing_interval`,
  `calculate_timestep`: **untouched**. They continue to operate on bare seconds
  floats. `calculate_timestep` overrides in user processes already return seconds
  implicitly; this is documented, not changed.

Tests:
- realize a spec with `interval: "1 min"` → `process['interval'] == 60.0` and
  `interval_units == "minute"`.
- serialize round-trips `"1 min"` back to `"1 min"`.
- bare `interval: 60` → `interval == 60.0`, `interval_units is None`, serializes
  as `60`.
- **behavior-equivalence:** a composite with `interval: 60` and one with
  `interval: "1 min"` advance `global_time` identically over a run.

### 3. vivarium-workbench

- **`env_worker.py` (process-node reconstruction, `~906-914`)** and
  **`/api/composite-resolve` (`api/app.py:~1728-1733`)**: surface three fields on
  each process node in the resolved payload — `interval` (seconds, number),
  `interval_units` (string | null), and `interval_label` (declared-form string
  built in Python via `format_time_interval`). The frontend needs no pint.

- **loom (vendored React/TS app, `vivarium_workbench/loom/`):**
  - `src/types.ts` (`ProcessNodeData`, `~:26`): add
    `intervalUnits?: string | null; intervalLabel?: string;`.
  - `src/convert.ts` (`~:338`): populate `intervalUnits: node.interval_units`,
    `intervalLabel: node.interval_label`.
  - `src/nodes/ProcessNode.tsx` (`~:592`): render
    `· every {data.intervalLabel ?? data.interval}` with
    `title={`${data.interval} s`}`.
  - `src/panels/InspectorPanel.tsx` (`~:53`): same treatment on the inspector
    header sub-line.
  - **Rebuild and re-vendor the loom bundle** (loom is a vendored Vite/React
    build; editing `.tsx` sources requires a rebuild to ship to the served UI).

Tests: API payload for a composite containing a minute-interval process carries
the three fields; `convert.ts` maps them; manual viewer verification after the
loom rebuild.

## Error handling

- Non-time unit string (`"1 meter"`) → `ValueError` at realize, naming the value.
- Unparseable string → `ValueError` naming the value.
- Non-positive interval after conversion → caught by the existing
  `_require_advancing_interval` guard (now in seconds).
- Legacy no-unit specs → interpreted as seconds; render "every 60 s".

## Scope boundaries (YAGNI)

- **Fixed canonical = seconds.** No per-composite canonical-unit override (locked
  decision). If that is ever wanted, it is a separate change.
- **No spec migration bundled in.** Authored composite specs are left alone; the
  tumor-tcell `interval: 60` simply starts rendering "60 s". Authors opt into
  `"1 min"` etc. separately when they choose.
- Only the **time** dimension is handled. No general quantity-typed config
  fields in this change.
- No JS-side pint; the frontend only displays strings/numbers computed in Python.

## Delivery

One worktree + one PR per repo, merged in dependency order:

1. **bigraph-schema** — `parse_time_interval` / `format_time_interval` + tests.
2. **process-bigraph** — `ProcessLink.interval_units`, realize/reify/serialize,
   tests. Depends on (1).
3. **vivarium-workbench** — worker/API fields + loom render + rebuild. Depends on
   (2).

Each repo change is independently testable; the behavior-equivalence test in (2)
is the guard that the canonical-seconds invariant holds.
