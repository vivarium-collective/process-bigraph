# Emitters — Recording Simulation Results

**Emitters** are how a process-bigraph simulation records what happens over time. They are a small, specialized kind of **Step** that observes state each tick and writes it somewhere — memory, the console, a JSON file, or a SQLite database — without ever modifying the state they observe.

This guide walks through:

- [How emitters work](#how-emitters-work)
- [The built-in emitters](#the-built-in-emitters)
- [Configuring an emitter inline in a composite](#configuring-an-emitter-inline-in-a-composite)
- [Adding an emitter to an existing composite](#adding-an-emitter-to-an-existing-composite)
- [Retrieving results after a run](#retrieving-results-after-a-run)
- [Filtering what gets recorded](#filtering-what-gets-recorded)
- [Long-term storage with SQLiteEmitter](#long-term-storage-with-sqliteemitter)
- [Writing a custom emitter](#writing-a-custom-emitter)

All examples are self-contained — paste them into a script or a notebook and they run.

---

## How emitters work

An emitter is a `Step` subclass. Like any Step, it:

- declares **input ports** (what parts of state it reads),
- has a `config_schema` describing its configuration,
- is scheduled by the `Composite` and invoked each tick with the wired state.

What makes it an *emitter* rather than an ordinary Step is what it does on each invocation:

- it writes the observed state to some external sink (RAM list, stdout, a file, a database),
- it declares **no outputs**, so it never feeds state back into the simulation,
- it exposes a `query(paths=None)` method so you can read the recorded history afterwards.

The base class lives at `process_bigraph.emitter.Emitter`. Every built-in emitter extends it, and you can extend it yourself — see [Writing a custom emitter](#writing-a-custom-emitter).

A minimal emitter step spec looks like this:

```python
{
    '_type': 'step',
    'address': 'local:RAMEmitter',
    'config': {
        'emit': {'time': 'node', 'x': 'node'},  # schema for observed ports
    },
    'inputs': {
        'time': ['global_time'],   # wire: read from composite.state['global_time']
        'x':    ['Env', 'x'],      # wire: read from composite.state['Env']['x']
    },
}
```

You rarely write that spec by hand — the helpers below build it for you.

---

## The built-in emitters

| Emitter | Address | Stores to | Good for |
|---|---|---|---|
| `ConsoleEmitter` | `local:ConsoleEmitter` | stdout | Quick debugging while you iterate |
| `RAMEmitter` | `local:RAMEmitter` | in-memory list | Short runs, analysis inside the same process |
| `JSONEmitter` | `local:JSONEmitter` | one JSON file per run | Small-to-medium runs you want to keep around |
| `SQLiteEmitter` | `local:SQLiteEmitter` | one SQLite `.db` file, rows keyed by `simulation_id` | Long runs, cross-run analytics, long-term storage |

All of them share the same `query(paths=None)` API, so downstream plotting and analysis code doesn't need to care which one produced the history.

---

## Configuring an emitter inline in a composite

The most common pattern: define the emitter alongside your processes when you build the composite. The `emitter_from_wires` helper builds the spec from a dict of wires.

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import emitter_from_wires

core = allocate_core()

composite = Composite({
    'state': {
        'Env': {'x': 0.0, 'target': 10.0},

        'mover': {
            '_type': 'process',
            'address': 'local:!my_module.MoveToward',
            'config': {'rate': 2.0},
            'interval': 1.0,
            'inputs':  {'x': ['Env', 'x'], 'target': ['Env', 'target']},
            'outputs': {'x': ['Env', 'x']},
        },

        # Record global_time and x every tick into a RAM emitter.
        'emitter': emitter_from_wires({
            'time': ['global_time'],
            'x':    ['Env', 'x'],
        }),
    },
}, core=core)

composite.run(10.0)
history = composite.state['emitter']['instance'].query()

for row in history[:3]:
    print(row)
```

Output:

```
{'time': 0.0, 'x': 0.0}
{'time': 1.0, 'x': 2.0}
{'time': 2.0, 'x': 4.0}
```

`emitter_from_wires(wires, address='local:RAMEmitter')` builds a `step` spec with inputs set to `wires` and an `emit` schema derived from their keys. To use a different backend, pass `address='local:ConsoleEmitter'`, `'local:JSONEmitter'`, or `'local:SQLiteEmitter'`.

---

## Adding an emitter to an existing composite

If you already have a `Composite` and want to attach an emitter — for example, because the composite was built by someone else, or you want to observe every wired-in variable without listing them by hand — use `add_emitter_to_composite`:

```python
from process_bigraph.emitter import add_emitter_to_composite, gather_emitter_results

# 'all' observes every non-process, non-step port in state.
composite = add_emitter_to_composite(composite, core, emitter_mode='all')

composite.run(10.0)

# gather_emitter_results returns {path: history} for every emitter in the composite.
results = gather_emitter_results(composite)
print(next(iter(results.values()))[0])
```

`emitter_mode` accepts:

- `'all'` — observe all valid input ports in state (the default).
- `'none'` — observe only `global_time`.
- `{'paths': [['Env', 'x'], 'target']}` — explicit list of paths.

You can also pass `address='local:SQLiteEmitter'` (or any other emitter address) to change the backend.

---

## Retrieving results after a run

Every emitter exposes the same `query()` API, so downstream code stays backend-agnostic.

```python
emitter = composite.state['emitter']['instance']

# Full history: list of dicts, one per tick.
history = emitter.query()

# Path-filtered: return only specific wires from each tick.
times_only = emitter.query([['time']])
x_and_target = emitter.query([['x'], ['Env', 'target']])
```

For a composite with multiple emitters, `gather_emitter_results(composite, queries=None)` calls `query()` on all of them and returns a `{path: history}` dict. Optionally pass `queries={('emitter',): [['Env', 'x']]}` to drive each emitter's query independently.

---

## Filtering what gets recorded

There are two places to filter:

1. **At emit time**, via the wires you pass to `emitter_from_wires` — the emitter only ever sees what you wire in, so unwired state is never copied.
2. **At query time**, via `emitter.query(paths)` — the emitter kept the full observed tree in storage, but returns only the paths you ask for.

For long runs, prefer filtering at emit time (less memory, smaller files, smaller db rows).

```python
# Only record x and the global clock, nothing else.
emitter_from_wires({
    'time': ['global_time'],
    'x':    ['Env', 'x'],
})
```

---

## Long-term storage with `SQLiteEmitter`

`RAMEmitter` throws everything away when the Python process ends. `JSONEmitter` rewrites the whole history file on every tick, which is fine for small runs but expensive for long ones. `SQLiteEmitter` appends one row per tick into a single `.db` file, indexed by a `simulation_id` — ideal for long runs and for analysis sessions that happen weeks after the simulation.

### Writing history

```python
from process_bigraph.emitter import (
    emitter_from_wires,
    save_simulation_metadata,
)

sim_id = 'run-2026-04-14-001'

composite = Composite({
    'state': {
        # ... processes and state ...
        'emitter': {
            '_type': 'step',
            'address': 'local:SQLiteEmitter',
            'config': {
                'emit':          {'time': 'node', 'x': 'node'},
                'file_path':     './out',            # directory
                'db_file':       'history.db',       # file name inside file_path
                'simulation_id': sim_id,             # defaults to a new UUID
                'name':          'mover_demo',       # optional human-readable label
            },
            'inputs': {'time': ['global_time'], 'x': ['Env', 'x']},
        },
    },
}, core=core)

# Record the config that produced this run, alongside the history.
save_simulation_metadata(
    './out/history.db', sim_id,
    composite_config=composite.state,   # serialize this however you like
    metadata={'notes': 'quick smoke test'},
    name='mover_demo',
)

composite.run(100.0)
```

Multiple runs can share one `.db` file — rows are partitioned by `simulation_id`, so `history.db` becomes a persistent archive of every experiment you've run.

### Thinning high-frequency runs with `subsample`

If your composite fires the emitter on a very short interval (e.g. `interval=0.1` over hours of sim time), you can easily end up writing tens of thousands of rows per run. Most of those rows are redundant for analysis and plotting — and each one costs a JSON-encode plus a SQLite INSERT.

Pass `subsample: N` in the emitter config to keep only every Nth tick:

```python
'emitter': {
    '_type': 'step',
    'address': 'local:SQLiteEmitter',
    'config': {
        'emit':          {'time': 'node', 'x': 'node'},
        'simulation_id': sim_id,
        'subsample':     10,     # record every 10th tick (first tick always kept)
    },
    'inputs': {'time': ['global_time'], 'x': ['Env', 'x']},
}
```

The stored `step` column still reflects the true composite tick number, so time series you build from the history preserve the simulation's real cadence even though the intermediate ticks were not persisted. `subsample` defaults to `1` (record every tick).

### Batching writes for throughput

SQLite commits one transaction per INSERT by default, which means a per-row fsync; for high-frequency runs most of the wall-clock ends up in that fsync. `batch_size: N` buffers up to N recorded rows in memory and flushes them as one transaction:

```python
'config': {
    'emit':          {'time': 'node', 'x': 'node'},
    'simulation_id': sim_id,
    'subsample':     10,
    'batch_size':    100,    # flush in transactions of up to 100 rows
}
```

The emitter guarantees a flush on `close()` and before every `query()`, so readers and clean shutdown always see a consistent picture. A hard crash before flush loses buffered rows only — the on-disk invariants are unaffected. `batch_size` composes with `subsample`: together they cut write volume (subsample) and amortize fsync (batch_size). Default is `1`.

### Reading history back, without any `Composite`

The retrieval helpers take only a db path, so you can analyze runs long after the simulation process has exited — no need to reconstruct the `Composite` or import the original processes.

```python
from process_bigraph.emitter import (
    list_simulations,
    load_history,
    load_simulation_metadata,
)

db_path = './out/history.db'

# What's in this db?
for sim in list_simulations(db_path):
    print(sim['simulation_id'], sim['name'], sim['step_count'])
# -> run-2026-04-14-001 mover_demo 101

# Full history of a specific run (same shape as emitter.query()).
history = load_history(db_path, 'run-2026-04-14-001')

# Or a path-filtered slice.
only_x = load_history(db_path, 'run-2026-04-14-001', paths=[['x']])

# Metadata: when it ran, what config produced it, any notes attached.
meta = load_simulation_metadata(db_path, 'run-2026-04-14-001')
print(meta['started_at'], meta['name'])
print(meta['composite_config'])
print(meta['metadata'])
```

Because it's a plain SQLite file, you can also open it directly and run SQL:

```python
import sqlite3, json
conn = sqlite3.connect('./out/history.db')

# All simulations, newest first.
for row in conn.execute(
    'SELECT simulation_id, name, started_at FROM simulations ORDER BY started_at DESC'
):
    print(row)

# Mean x by simulation.
for sim_id, avg_x in conn.execute('''
    SELECT simulation_id, AVG(json_extract(state, '$.x'))
    FROM history GROUP BY simulation_id
'''):
    print(sim_id, avg_x)
```

The schema:

```sql
CREATE TABLE history (
    simulation_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    global_time REAL,
    state TEXT NOT NULL,             -- JSON
    PRIMARY KEY (simulation_id, step)
);
CREATE INDEX idx_history_sim_time ON history(simulation_id, global_time);

CREATE TABLE simulations (
    simulation_id TEXT PRIMARY KEY,
    name TEXT,
    started_at TEXT NOT NULL,        -- ISO 8601 UTC
    composite_config TEXT,           -- JSON
    metadata TEXT                    -- JSON
);
```

---

## Writing a custom emitter

An emitter is just a `Step` subclass with an `inputs()` method that returns what to observe and an `update(state)` method that records it. Here's a compact example that streams observations to a CSV file:

```python
import csv
from typing import Dict
from process_bigraph.emitter import Emitter


class CSVEmitter(Emitter):
    '''Append each tick to a CSV file.'''

    config_schema = {
        **Emitter.config_schema,
        'file_path':  {'_type': 'string', '_default': 'history.csv'},
        'fieldnames': {'_type': 'list[string]', '_default': None},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.file_path = config.get('file_path', 'history.csv')
        self.fieldnames = config.get('fieldnames') or list(self.inputs().keys())
        self._fh = open(self.file_path, 'a', newline='')
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if self._fh.tell() == 0:
            self._writer.writeheader()

    def update(self, state) -> Dict:
        self._writer.writerow({k: state.get(k) for k in self.fieldnames})
        self._fh.flush()
        return {}

    def query(self, query=None):
        import csv
        with open(self.file_path) as f:
            return list(csv.DictReader(f))

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass
```

To register it so it's reachable at an address like `local:CSVEmitter`:

```python
core.register_link('CSVEmitter', CSVEmitter)
```

Then use it exactly like any other emitter — `emitter_from_wires({...}, address='local:CSVEmitter')`, `add_emitter_to_composite(..., address='local:CSVEmitter')`, or drop its spec directly into a composite.

The contract to satisfy:

- Accept an `emit` key in `config_schema` (inherit from `Emitter.config_schema`) — this is the schema of observed state.
- Implement `update(state)` to persist observations. Returning `{}` keeps it read-only.
- Implement `query(paths=None)` to return the same shape your users expect (a list of per-tick dicts) so it plugs into existing plotting/analysis code.
- If you need to handle live process/edge objects that get wired into state, reuse `process_bigraph.emitter.tree_copy` — it's what `RAMEmitter` and `SQLiteEmitter` use to strip `Edge` instances before persistence.

---

## See also

- `process_bigraph/emitter.py` — the source of all built-in emitters and helpers.
- [Tutorial 1 — Process-Bigraph Basics](https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_1.html) — section 7 introduces emitters in context.
