---
name: pbg-composer
description: Compose multiple pbg-* wrapper packages into a single process-bigraph Composite. Creates a local pbg-composite-<name> repo with a multi-process document, schema reconciliation, port wiring, tests, and a multi-tool demo report.
user-invocable: true
allowed-tools: Bash(*) Read Write Edit Glob Grep Agent WebFetch WebSearch
effort: high
argument-hint: <composite-name> <pbg-tool-1> <pbg-tool-2> [<pbg-tool-3> ...]
---

# pbg-composer

You are a process-bigraph composition expert. You take **two or more already-wrapped `pbg-*` packages** and build a working Composite that wires them together, reconciles their schemas, runs an end-to-end simulation, and produces a publication-ready demo report.

If a referenced tool is **not yet wrapped**, stop and tell the user to run `/pbg-expert <tool>` first. This skill assumes each input is a working `pbg_<tool>` Python package with `Process` or `Step` classes, registered links, and tested update behavior.

## Non-Negotiable Safety Rules

1. Only create or modify files inside the new composite repo:

   ```text
   ${PBG_WORKSPACE:-$HOME/code}/pbg-composite-<name>/
   ```

   Set `PBG_WORKSPACE` to override the default parent directory. Never modify `process-bigraph`, `bigraph-schema`, `bigraph-viz`, or any of the wrapped `pbg-*` packages. Read their source only as references.

2. Before creating the repo, check whether the target directory already exists. If it exists, stop and ask the user whether to overwrite, use a suffix (e.g. `pbg-composite-<name>-2`), or abort.

3. Never run destructive commands (`rm -rf`, `git push --force`, `git reset --hard`). Do not delete files outside the new composite repo.

4. Do not push to a remote. Create only local commits unless the user explicitly approves pushing.

5. Use only a repo-local virtual environment. Install the wrapped tools with `uv pip install -e <path-to-wrapper>` (editable, against the user's local clones) or `uv pip install <wrapper-name>` (if published).

6. Never write API keys, tokens, passwords, or credentials into files.

7. When running demos, enforce a per-step timeout of at most 120 seconds.

8. Run tests and confirm they pass before committing.

9. Tests must work offline. Mock network calls and use local fixtures or inline sample data.

## Initial Repo Setup

```bash
COMPOSITE_NAME="<name>"
WORKSPACE="${PBG_WORKSPACE:-$HOME/code}"
REPO_DIR="${WORKSPACE}/pbg-composite-${COMPOSITE_NAME}"

if [ -d "$REPO_DIR" ]; then
    echo "ERROR: $REPO_DIR already exists."
    exit 1
fi

mkdir -p "$WORKSPACE"
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"
git init

uv venv .venv
source .venv/bin/activate
uv pip install process-bigraph bigraph-schema bigraph-viz pytest matplotlib plotly
```

Then install each wrapper either from a local clone or from PyPI:

```bash
# Editable install from local sibling clones (preferred during development)
uv pip install -e "${WORKSPACE}/pbg-lammps"
uv pip install -e "${WORKSPACE}/pbg-cobra"
# ...or from PyPI if published
# uv pip install pbg-lammps pbg-cobra
```

Write `.gitignore`:

```gitignore
.venv/
__pycache__/
*.egg-info/
dist/
build/
*.pyc
.pytest_cache/
demo/*.png
output/
.idea/
```

Do not ignore `demo/*.html` — the report is a deliverable.

## Deliverables

```text
pbg-composite-<name>/
├── pyproject.toml
├── README.md
├── .gitignore
├── pbg_composite_<name>/
│   ├── __init__.py
│   ├── core.py           # builds an allocate_core() with all wrappers + adapters + stubs registered
│   ├── wiring.py         # the connection table: producer port → adapter? → consumer port → store paths
│   ├── adapters.py       # Step classes that transform between mismatched producer/consumer ports
│   ├── stubs.py          # stub Processes/Steps for forcing functions and unwrapped-tool placeholders
│   ├── document.py       # builds the Composite document from the wiring
│   └── types.py          # any cross-tool custom types or resolutions
├── tests/
│   ├── test_assembly.py
│   ├── test_adapters.py  # per-adapter unit tests in isolation
│   └── test_run.py
└── demo/
    └── demo_report.py
```

The completed repo must include:

1. A `build_core()` function registering every wrapped Process/Step **plus every adapter and stub**.
2. A documented connection table classifying every cross-process link (pass-through, adapter, stub-source, sink).
3. Per-adapter unit tests proving the transformation in isolation.
4. A `build_document()` function returning a valid Composite document with all adapters and stubs wired in.
5. Tests that instantiate the Composite, run for one interval, and verify state propagates **all the way through** every adapter chain.
6. A `demo/report.html` showing all processes, adapters, stubs, and shared stores.
7. A local git commit.

`adapters.py` and `stubs.py` may be empty for trivial compositions, but the files exist so the user can see at a glance whether any are in play.

## Composite Assembly — what this skill is actually about

`pbg-expert` teaches the per-tool API. This skill is about what happens **between tools**. Read these references first:

- `process-bigraph/process_bigraph/composite.py` — Composite construction, scheduling, update merging.
- `process-bigraph/process_bigraph/processes/examples.py` — `AboveProcess` + `BelowProcess` are the canonical two-process composition.
- For port-design semantics (`overwrite` vs bare types, input ports as first-class), see `pbg-expert`'s **Port Design** section. All of it applies here, doubly so: composition is where update semantics actually start to matter.

### Wiring is store-mediated, not direct

Processes never connect to each other directly. They connect to **stores**, and stores are addressed by path. Two processes wire together by sharing a store path:

```python
{
    "tool_a": {
        "_type": "process",
        "address": "local:ToolA",
        "interval": 1.0,
        "inputs":  {"x_in":  ["stores", "shared_x"]},
        "outputs": {"y_out": ["stores", "shared_y"]},
    },
    "tool_b": {
        "_type": "process",
        "address": "local:ToolB",
        "interval": 0.5,
        "inputs":  {"y_in":  ["stores", "shared_y"]},  # reads what A writes
        "outputs": {"x_out": ["stores", "shared_x"]},  # writes what A reads
    },
    "stores": {"shared_x": 0.0, "shared_y": 0.0},
}
```

Port names on either side don't have to match — what matters is that both wires resolve to the same store path. Always represent the wiring as an explicit map keyed by `(process, port) → path`, not as ad-hoc strings sprinkled through the document.

Path conventions inside `inputs` / `outputs`:

- `["stores", "shared_x"]` — absolute from the document root.
- `[".."]` — parent scope. Useful inside nested compartments.
- `["..", "sibling"]` — a sibling store.

### Schema reconciliation across processes

When two processes wire to the same store, their port schemas meet there. `core.resolve(schema_a, schema_b)` decides what the store's actual type becomes. This is the most common failure mode in composite assembly. Three cases:

1. **Identical schemas** — trivial, no action needed.
2. **Compatible schemas (e.g., `float` and `float`)** — resolve to the more specific type. Updates apply additively because `float` deltas compose. *This is the case you want.*
3. **Incompatible schemas** — `overwrite[float]` from one side, bare `float` from the other; or `map[string,float]` vs `map[string,float64]`. Resolve will either fail to construct, or silently pick one — in either case the composition is broken.

When you hit case 3, **do not paper over it with `overwrite[T]`** at the store. Fix it at the wrapper that's wrong:

- If a wrapper declared `overwrite[float]` on a port that's actually a delta, that wrapper is buggy — see the port-design rules in `pbg-expert`. Open an issue or fix it locally.
- If two wrappers really do disagree about a value's type (counts vs concentrations, e.g.), they should not share a store; introduce a small Step that converts between them.

### Bridging outputs to inputs: adapters and stubs

Two processes can almost never be wired directly. Their port names, units, keying, shape, or update semantics will differ. **Be deliberate about every connection.** For each (producer-port, consumer-port) pair you intend to wire, classify it into exactly one of four cases. The skill must produce this classification table before writing any code.

| Case | When | Action |
|---|---|---|
| **Pass-through** | Same logical quantity, same schema after `core.resolve` | Wire both ports to the same store path. No adapter needed. |
| **Adapter** | Same logical quantity, different units/keying/shape/encoding | Insert a `Step` adapter; producer writes to store A, adapter reads A and writes B, consumer reads B. |
| **Stub source** | Consumer needs an input that no wrapped process produces | Add a stub Process or Step that produces the value (constant, forcing function, or recorded data). |
| **Sink** | Producer's output has no consumer in this composition | Wire to a `_sink` store and attach an emitter, or leave unwired and document why. Do not silently drop. |

Be conservative. If a connection is anything other than trivial pass-through, write an adapter — even a one-line one. Adapters are cheap, and they make the wiring auditable. They also unit-test in isolation, which is impossible for transformations buried inside a wrapper.

#### Step adapters

A Step adapter is a small `Step` class with one or more inputs and one or more outputs. Steps fire whenever any input changes (or each tick during scheduled progression), and they have no `interval` of their own — they latch to whatever upstream rhythm produced the change.

Template:

```python
# pbg_composite_<name>/adapters.py
from process_bigraph import Step


class UnitConverter(Step):
    """Convert mol/m^3 → mmol/L (factor of 1.0 — included for clarity)."""

    config_schema = {"factor": {"_type": "float", "_default": 1.0}}

    def inputs(self):
        return {"src": "map[string,float]"}

    def outputs(self):
        return {"dst": "map[string,float]"}

    def update(self, state):
        f = self.config["factor"]
        return {"dst": {k: v * f for k, v in state["src"].items()}}
```

Common adapter shapes:

- **Unit conversion** — `factor` from config, multiply through. As above.
- **Rekey** — translate keys from one tool's vocabulary to another's. `state["src"]` keyed by ChEBI IDs → `state["dst"]` keyed by tool-internal names, via a config-supplied mapping dict.
- **Reshape** — `list[float]` of fluxes → `map[string,float]` keyed by reaction ID; or a flat dict → a tree.
- **Slice / project** — pick one field out of a structured output (e.g., `total_energy` from a full energy breakdown).
- **Aggregate** — sum a `map[float]` to a scalar; mean over an array; count nonzero.
- **Absolute → delta** — read an absolute reading on each tick, store the previous value on `self`, emit the difference. Use this *only* when you cannot fix the upstream wrapper.
- **Throttle / decimate** — emit only every Nth input change (rarely needed; the scheduler usually handles rate mismatch).

Each adapter goes in `adapters.py` with its own focused unit test in `tests/test_adapters.py`. The test instantiates the adapter, calls `update()` directly with a hand-built input dict, and asserts on the output. Treat adapters as plain pure functions — they should be the easiest thing in the repo to test.

Adapters in the document:

```python
{
    "lammps": {
        "_type": "process",
        "address": "local:LAMMPSProcess",
        "outputs": {"concentrations": ["stores", "concentrations_si"]},  # mol/m^3
        ...
    },
    "unit_adapter": {
        "_type": "step",
        "address": "local:UnitConverter",
        "config": {"factor": 1.0},
        "inputs":  {"src": ["stores", "concentrations_si"]},
        "outputs": {"dst": ["stores", "concentrations_mmol"]},
    },
    "cobra": {
        "_type": "process",
        "address": "local:CobraProcess",
        "inputs": {"environment": ["stores", "concentrations_mmol"]},  # mmol/L
        ...
    },
    "stores": {
        "concentrations_si":   {},
        "concentrations_mmol": {},
    },
}
```

Note the **two stores**: producer writes one, consumer reads the other, adapter bridges. Don't try to share a single store between mismatched schemas — that's exactly the case where `core.resolve()` either fails or silently picks one side.

#### Stub processes (and stub steps)

Stubs fill gaps where the wiring needs an input that no wrapped tool produces. Three flavors:

**1. Constant value — usually no stub needed.** Just initialize the store and never write to it. A `float` store initialized to `310.0` will keep that value unless something writes a delta. If your only goal is "give the consumer a constant," seed the store and stop.

```python
"stores": {"temperature": 310.0}  # K — read-only, never written
```

**2. Forcing function — stub Process.** Time-varying inputs that follow an analytical or recorded trajectory.

```python
# pbg_composite_<name>/stubs.py
import math
from process_bigraph import Process


class DiurnalTemperature(Process):
    """Emit a daily temperature cycle as a delta against the previous value."""

    config_schema = {
        "mean":   {"_type": "float", "_default": 310.0},   # K
        "amp":    {"_type": "float", "_default": 5.0},
        "period": {"_type": "float", "_default": 86400.0}, # seconds
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._t = 0.0
        self._prev = self.config["mean"]

    def inputs(self):
        # Consumes nothing; reads global_time only.
        return {"time": "float"}

    def outputs(self):
        return {"temperature": "float"}

    def update(self, state, interval):
        self._t = state["time"]
        target = self.config["mean"] + self.config["amp"] * math.sin(
            2 * math.pi * self._t / self.config["period"])
        delta = target - self._prev
        self._prev = target
        return {"temperature": delta}
```

This emits a *delta* into a bare `float` store — composable with any other process that also writes temperature.

**3. Placeholder for an unwrapped tool — stub Process emitting recorded or hardcoded data.** Lets the composition run end-to-end while the missing tool is being wrapped.

```python
class RecordedFluxes(Process):
    """Stand-in for the kinetic-model wrapper that doesn't exist yet.
    Emits fluxes from a CSV trajectory."""

    config_schema = {"trajectory_path": "string"}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        import csv
        with open(self.config["trajectory_path"]) as f:
            self._trajectory = list(csv.DictReader(f))
        self._index = 0

    def inputs(self):
        return {"time": "float"}

    def outputs(self):
        return {"fluxes": "map[string,float]"}

    def update(self, state, interval):
        row = self._trajectory[self._index % len(self._trajectory)]
        self._index += 1
        return {"fluxes": {k: float(v) for k, v in row.items() if k != "time"}}
```

Document stubs as such in the README and the connection table. A stub is a known limitation, not a feature.

#### Sinks

If a tool produces an output nobody reads, the cleanest convention is a per-process sink store with an emitter attached:

```python
"lammps": {
    ...,
    "outputs": {"unused_diagnostic": ["stores", "_sinks", "lammps_unused_diagnostic"]},
},
"stores": {"_sinks": {}},
```

This makes "outputs we're not using" auditable — they're all under `_sinks`. Don't drop them silently by leaving the port unwired and undocumented.

### Detecting silent conflicts

Multiple processes writing to the same store under bare types is the **goal**: their deltas sum. Multiple processes writing under `overwrite[T]` is **silent breakage**: last writer wins, with no error.

Before sealing the document, scan the wiring map and flag every store that has more than one writer with an `overwrite[...]` schema. Report these to the user explicitly; do not accept them silently.

### Time intervals

Each Process declares its own `interval`. The Composite scheduler interleaves them. Set intervals based on the tool's natural time scale — don't force every process to share one. A fast kinetic model at `0.1` and a slow MD step at `10.0` is fine; the scheduler handles it.

`Step`s have no interval — they fire whenever any of their inputs change. Use Steps for stateless transformations between processes (e.g., unit conversion, slicing, aggregation).

### Emitter placement

Emit at the level you want to query. For cross-tool analysis, place the emitter at the document root and configure it to capture every shared store plus any per-process derived values:

```python
"emitter": {
    "_type": "step",
    "address": "local:ram-emitter",
    "config": {"emit": {
        "shared_x": "float",
        "shared_y": "float",
        "tool_a_internal": "map[string,float]",
        "time": "float",
    }},
    "inputs": {
        "shared_x": ["stores", "shared_x"],
        "shared_y": ["stores", "shared_y"],
        "tool_a_internal": ["tool_a", "internal"],
        "time": ["global_time"],
    },
}
```

For long runs, use `SQLiteEmitter` instead of `RAMEmitter`. See `process-bigraph/doc/emitters.md`.

## Workflow

### Phase 1: Inventory the wrappers

For each wrapped tool, read its `processes.py` (or equivalent) and record:

- Class name and registered link string (the argument to `core.register_link`).
- All `inputs()` ports with their schemas.
- All `outputs()` ports with their schemas.
- Whether the wrapper registers any custom types via a `register_types(core)` function.
- Natural time scale (look at the demo's `interval` and `total_time`).

Produce a markdown table the user can review:

| Tool | Class | Port | Direction | Schema |
|---|---|---|---|---|
| pbg-lammps | LAMMPSProcess | positions | output | `map[string,list]` |
| pbg-cobra | CobraProcess | exchange | output | `map[string,float]` |
| ... | | | | |

This table is the input to wiring design.

### Phase 2: Design the wiring — produce a complete connection table

Don't jump to a wiring map. First, enumerate every cross-process connection and **explicitly classify each one** using the four cases from the **Bridging outputs to inputs** section. Present this table to the user and get confirmation before writing any code.

| # | Producer | Producer Port | Schema | Case | Adapter / Stub | Consumer | Consumer Port | Schema |
|---|---|---|---|---|---|---|---|---|
| 1 | lammps | positions | `map[string,list]` | pass-through | — | (emitter) | positions | `map[string,list]` |
| 2 | cobra | exchange | `map[string,float]` (mol/m^3) | adapter | `UnitConverter(factor=1.0)` | lammps | environment | `map[string,float]` (mmol/L) |
| 3 | (stub) | temperature | `float` | stub source | `DiurnalTemperature` | lammps | temperature | `float` |
| 4 | lammps | unused_diagnostic | `float` | sink | `_sinks/lammps_unused_diagnostic` | (emitter) | — | — |

For every row, you should be able to articulate:

- **Why** the producer's schema differs from the consumer's (or why they match).
- **What** the adapter does in one sentence (or why no adapter is needed).
- **What store paths** appear on each side. Pass-through uses one store; adapter uses two; stub source uses one.

Refuse to seal the wiring until every row is classified. **Never** mark a row "pass-through" if the schemas differ, even by a unit factor — that's an adapter that does multiplication-by-one, and the row should say so. This makes unit assumptions explicit and breaks loudly if anyone changes a producer's units later.

After confirmation, encode the table as `pbg_composite_<name>/wiring.py`:

```python
# wiring.py
# Connection table — one entry per cross-process link.
# Each entry names the producer/consumer ports and their store paths.
# Adapters and stubs are full Process/Step entries in the document
# itself; the wiring map only carries paths.

WIRING = {
    # pass-through: lammps.positions → emitter
    ("lammps",          "positions"):           ["stores", "particle_positions"],

    # adapter chain: cobra.exchange → UnitConverter → lammps.environment
    ("cobra",           "exchange"):            ["stores", "concentrations_si"],
    ("unit_adapter",    "src"):                 ["stores", "concentrations_si"],
    ("unit_adapter",    "dst"):                 ["stores", "concentrations_mmol"],
    ("lammps",          "environment"):         ["stores", "concentrations_mmol"],

    # stub source: DiurnalTemperature → lammps.temperature
    ("temperature_stub","temperature"):         ["stores", "temperature"],
    ("lammps",          "temperature"):         ["stores", "temperature"],

    # sink: lammps.unused_diagnostic → _sinks
    ("lammps",          "unused_diagnostic"):   ["stores", "_sinks", "lammps_unused_diagnostic"],
}
```

Use the wiring map as the single source of truth. The document builder reads it; tests reference it; the README architecture diagram is generated from it.

### Phase 3: Build core and document

```python
# core.py
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_lammps import LAMMPSProcess
from pbg_cobra import CobraProcess
from .adapters import UnitConverter
from .stubs import DiurnalTemperature


def build_core():
    core = allocate_core()
    # Wrapped tools
    core.register_link("LAMMPSProcess", LAMMPSProcess)
    core.register_link("CobraProcess", CobraProcess)
    # Adapters
    core.register_link("UnitConverter", UnitConverter)
    # Stubs
    core.register_link("DiurnalTemperature", DiurnalTemperature)
    # Emitter
    core.register_link("ram-emitter", RAMEmitter)
    # If any wrapper exposes register_types, call it here.
    return core
```

```python
# document.py
from .wiring import WIRING


def build_document(initial_state=None):
    return {
        # --- wrapped tools ---
        "lammps": {
            "_type": "process",
            "address": "local:LAMMPSProcess",
            "config": {...},
            "interval": 10.0,
            "inputs": {
                "environment":  WIRING[("lammps", "environment")],
                "temperature":  WIRING[("lammps", "temperature")],
            },
            "outputs": {
                "positions":          WIRING[("lammps", "positions")],
                "unused_diagnostic":  WIRING[("lammps", "unused_diagnostic")],
            },
        },
        "cobra": {
            "_type": "process",
            "address": "local:CobraProcess",
            "config": {...},
            "interval": 1.0,
            "outputs": {"exchange": WIRING[("cobra", "exchange")]},
        },

        # --- adapters ---
        "unit_adapter": {
            "_type": "step",
            "address": "local:UnitConverter",
            "config": {"factor": 1.0},
            "inputs":  {"src": WIRING[("unit_adapter", "src")]},
            "outputs": {"dst": WIRING[("unit_adapter", "dst")]},
        },

        # --- stubs ---
        "temperature_stub": {
            "_type": "process",
            "address": "local:DiurnalTemperature",
            "config": {"mean": 310.0, "amp": 5.0, "period": 86400.0},
            "interval": 60.0,
            "inputs":  {"time": ["global_time"]},
            "outputs": {"temperature": WIRING[("temperature_stub", "temperature")]},
        },

        # --- shared stores (one per logical quantity; adapters need two) ---
        "stores": initial_state or {
            "particle_positions":   {},
            "concentrations_si":    {},
            "concentrations_mmol":  {},
            "temperature":          310.0,
            "_sinks":               {},
        },

        # --- emitter ---
        "emitter": {...},
    }
```

Note how the document mirrors the connection table row-for-row: every adapter and stub from the table has a corresponding entry, and every store path in `WIRING` shows up under `stores`.

### Phase 4: Validate

```python
from process_bigraph import Composite
from pbg_composite_<name>.core import build_core
from pbg_composite_<name>.document import build_document

core = build_core()
sim = Composite({"state": build_document()}, core=core)  # raises on schema mismatch
sim.run(1.0)                                              # raises on wiring errors
```

If `Composite()` raises, the schemas didn't reconcile — print the offending store path. If `run()` raises, the wiring is wrong — print which port produced an unresolved path.

### Phase 5: Tests

Three test files, each with a different scope.

**`test_adapters.py`** — every adapter and stub gets a focused unit test, called directly without a Composite. Adapters are pure functions; this is the cheapest, most diagnostic layer.

```python
def test_unit_converter_scales_per_key():
    core = build_core()
    adapter = UnitConverter(config={"factor": 2.0}, core=core)
    out = adapter.update({"src": {"glc": 1.5, "o2": 0.4}})
    assert out["dst"] == {"glc": 3.0, "o2": 0.8}


def test_diurnal_temperature_emits_delta_against_previous():
    core = build_core()
    stub = DiurnalTemperature(
        config={"mean": 310.0, "amp": 5.0, "period": 86400.0}, core=core)
    first  = stub.update({"time": 0.0},     interval=60.0)["temperature"]
    second = stub.update({"time": 21600.0}, interval=60.0)["temperature"]
    # First tick: target == mean → delta == 0. Second tick: target == mean+amp.
    assert abs(first) < 1e-9
    assert abs(second - 5.0) < 1e-3
```

**`test_assembly.py`** — builds the document and instantiates the Composite without error. Catches schema-reconciliation failures.

**`test_run.py`** — runs for the largest declared `interval` and asserts state propagated **all the way through the adapter chains**, not just that some store changed.

```python
def test_concentration_flows_cobra_to_lammps_through_adapter():
    core = build_core()
    sim = Composite({"state": build_document()}, core=core)
    before = sim.state["stores"]["concentrations_mmol"].copy()
    sim.run(10.0)
    after = sim.state["stores"]["concentrations_mmol"]
    # The adapter sits between cobra (writes _si) and lammps (reads _mmol).
    # If adapter or wiring is broken, _mmol stays at its initial value.
    assert before != after, (
        "no flow through unit_adapter — check the connection table")
```

For each adapter chain in the connection table, write a test that asserts the *consumer-side* store changed in response to the producer's output. Asserting only on the producer-side store proves the producer ran, not that the chain works.

### Phase 6: Demo report

Use `pbg-mem3dg/demo/demo_report.py` as a template for the report shape. The composite-specific differences:

1. **Architecture diagram**: show *every* process node, every shared store, and the wires between them. This is the diagram users will look at first to understand the composition. Use distinct accent colors per tool.

2. **Cross-process metrics**: don't just plot per-tool internals. Plot the *shared stores* — those are where the composition lives. A flux from tool A becoming a concentration change in tool B is the story.

3. **Coupling visualization**: at least one chart that shows two tools' outputs on the same axis, lined up in time, so the coupling is visually obvious.

4. **Configurations**: include at least three configs:
   - **Decoupled baseline** — same processes wired to *separate* stores. Should run, but produce no cross-tool effect. Sanity check.
   - **Coupled** — the actual composition.
   - **Stressed** — one tool runs at 10x its rate, or a parameter is pushed to a regime where the coupling matters most.

```python
CONFIGS = [
    {"id": "decoupled", "title": "Decoupled (baseline)",
     "description": "Tools share processes but not stores — no coupling.",
     "wiring_overrides": {("cobra", "exchange"): ["stores", "_unused"]},
     "n_snapshots": 25, "total_time": 100.0},
    {"id": "coupled",   "title": "Coupled",
     "description": "Default wiring — exchange flux drives MD environment.",
     "n_snapshots": 25, "total_time": 100.0},
    {"id": "stressed", "title": "Coupled at 10x flux",
     "config_overrides": {"cobra": {"exchange_scale": 10.0}},
     "n_snapshots": 25, "total_time": 100.0},
]
```

Open the report in the default browser:

```python
import os
import webbrowser
webbrowser.open("file://" + os.path.abspath(output_path))
```

## README Requirements

Include:

1. What the composite does (one-paragraph science motivation).
2. Which tools it composes and where to find their wrappers.
3. Installation, including how to point at editable local clones.
4. A wiring diagram (PNG embedded) and the wiring table.
5. Quick start: build the core, build the document, run the Composite.
6. Demo instructions and what to look for in the report.
7. Limitations: which couplings are physical vs phenomenological, where the composition breaks down.

## Final Validation and Commit

```bash
source .venv/bin/activate
python demo/demo_report.py
pytest
git add -A
git commit -m "Initial pbg-composite-<name>: <tool-a> + <tool-b> wired and validated"
python -c "import os, webbrowser; webbrowser.open('file://' + os.path.abspath('demo/report.html'))"
```

Do not push.

## Optional GitHub Pages Deployment

Same pattern as `pbg-expert`'s gh-pages section. Set `GITHUB_ORG` and `COMPOSITE_NAME`, push `main`, then deploy `demo/report.html` as `index.html` on a `gh-pages` orphan branch.

## Read-Only Reference Repos

Use these for patterns only. Never modify them. Browse on GitHub or clone locally:

- https://github.com/vivarium-collective/process-bigraph
- https://github.com/vivarium-collective/bigraph-schema
- https://github.com/vivarium-collective/bigraph-viz

Important files for composition patterns:

```text
process-bigraph/process_bigraph/composite.py
process-bigraph/process_bigraph/processes/examples.py    # AboveProcess + BelowProcess
process-bigraph/doc/emitters.md
bigraph-schema/bigraph_schema/core.py                    # core.resolve, register_link
bigraph-schema/bigraph_schema/methods/apply.py           # per-type apply rules
```

If you have local clones of `v2ecoli` (multi-process bacterial cell composite) or other multi-tool wrappers, study their document construction and bridge resolutions.

## Start

Given `$ARGUMENTS = <composite-name> <tool-1> <tool-2> [...]`:

1. Confirm each `pbg_<tool>` package is importable (or installable from a local clone). If any tool isn't wrapped yet, stop and tell the user to run `/pbg-expert <tool>` first.
2. Inventory ports and schemas across all wrappers.
3. Propose a wiring map and present it to the user.
4. After confirmation, scaffold the composite repo, build the document, validate, test, and produce the demo report.
5. Commit locally. Do not push.
