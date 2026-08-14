# Nextflow Step-Network Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a process-bigraph `Composite`'s Step network as a runnable Nextflow DSL2 workflow on a batch backend, where any node — including a whole nested Composite simulation — becomes one Nextflow task.

**Architecture:** Build on the existing `render_composite()` renderer. Fix its dependency-edge inference, add a `run_composite` per-task runner so a whole Composite is one task, and add a `nextflow_deploy` layer that generates `nextflow.config` executor profiles and shells out to `nextflow run`. State flows between tasks as JSON document files (vEcoli daughter-state pattern).

**Tech Stack:** Python 3.12, process-bigraph, Nextflow DSL2, pytest 9.

**Spec:** `docs/superpowers/specs/2026-08-13-nextflow-step-network-deploy-design.md`

## Global Constraints

- **Worktree:** all work in `/Users/eranagmon/code/process-bigraph--nextflow-deploy` on branch `nextflow-deploy`. Never commit in the canonical `~/code/process-bigraph` checkout.
- **Test-run command (every test step uses this exact prefix):**
  ```
  PYTHONPATH=/Users/eranagmon/code/process-bigraph--nextflow-deploy \
    /Users/eranagmon/code/process-bigraph/.venv/bin/python -m pytest
  ```
  Verify `process_bigraph.__file__` resolves inside the worktree (it does with the above `PYTHONPATH`).
- **Composite construction idiom** (from `tests/test_render_results.py`): `core = allocate_core(); core.register_link('Name', ClassOrFn)`; state nodes use `{'_type': 'step'|'process', 'address': 'local:Name', 'config': {...}, 'inputs': {...}, 'outputs': {...}}`; `Composite({'state': state}, core=core)`.
- **Prefer name refs over line numbers** — the worktree is `origin/main` (e40d0ae); line numbers drift.
- **Commit after every task.** Conventional-commit messages; end each with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **No new runtime dependencies.** Nextflow is an external binary (`/usr/local/bin/nextflow`), invoked via `subprocess`; not a Python dep.

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `process_bigraph/nextflow.py` | Composite → `.nf` string. Edge-fix + composite-node script branch + `python` interpreter option. | modify |
| `process_bigraph/run_composite.py` | Whole-Composite per-task CLI runner. | create |
| `process_bigraph/nextflow_deploy.py` | `nextflow.config` profile generation + `deploy()`/launch. | create |
| `process_bigraph/tests/test_nextflow_render.py` | Renderer: edge fix, composite-node emission. | create |
| `process_bigraph/tests/test_run_composite.py` | `run_composite` runner. | create |
| `process_bigraph/tests/test_nextflow_deploy.py` | Config gen + `nextflow`-gated integration launch. | create |

---

### Task 1: Fix dependency-edge inference in the renderer

The renderer's `_topological_order` infers edges by **exact** `input_path == output_path`, dropping nested-store edges (writer of `('shared',)` → reader of `('shared','x')`). Lift edges from the authoritative `composite.node_dependencies` (`{path: {'before': {writers}, 'after': {readers}}}`, already prefix-propagated by `build_step_network`).

**Files:**
- Modify: `process_bigraph/nextflow.py` (`_topological_order`, and its call in `render_composite`)
- Test: `process_bigraph/tests/test_nextflow_render.py`

**Interfaces:**
- Produces: `_topological_order(step_paths, step_dependencies, node_dependencies=None) -> List[Path]`. When `node_dependencies` is given, edges are `writer → reader` for each path's `before × after`. When `None`, falls back to the existing exact-match behavior (back-compat).

- [ ] **Step 1: Write the failing test**

```python
# process_bigraph/tests/test_nextflow_render.py
from process_bigraph.nextflow import _topological_order


def test_topological_order_respects_nested_store_edges():
    # 'zebra' writes ('shared',); 'alpha' reads the nested ('shared','x').
    # Names chosen so alphabetical (Kahn tie-break) order CONTRADICTS the
    # dependency — only real edge inference puts zebra before alpha.
    step_paths = {('zebra',): {}, ('alpha',): {}}
    step_dependencies = {
        ('zebra',): {'input_paths': [], 'output_paths': [['shared']]},
        ('alpha',): {'input_paths': [['shared', 'x']], 'output_paths': []},
    }
    node_dependencies = {
        ('shared',): {'before': {('zebra',)}, 'after': set()},
        # build_step_network's prefix propagation puts zebra in before(shared/x)
        ('shared', 'x'): {'before': {('zebra',)}, 'after': {('alpha',)}},
    }
    order = _topological_order(step_paths, step_dependencies, node_dependencies)
    assert order.index(('zebra',)) < order.index(('alpha',))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… -m pytest process_bigraph/tests/test_nextflow_render.py::test_topological_order_respects_nested_store_edges -v`
Expected: FAIL — `TypeError: _topological_order() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Implement the fix**

Replace `_topological_order` in `process_bigraph/nextflow.py` with:

```python
def _topological_order(step_paths, step_dependencies, node_dependencies=None):
    """Kahn's algorithm over the step graph.

    Edge model: prefer ``node_dependencies`` (authoritative, prefix-aware).
    For each shared store path, every writer in ``before`` precedes every
    reader in ``after``. Falls back to exact ``input_path == output_path``
    matching when ``node_dependencies`` is absent (back-compat).
    """
    incoming = {sp: set() for sp in step_paths}
    outgoing = {sp: set() for sp in step_paths}

    if node_dependencies:
        for deps in node_dependencies.values():
            writers = [w for w in deps.get('before', ()) if w in step_paths]
            readers = [r for r in deps.get('after', ()) if r in step_paths]
            for w in writers:
                for r in readers:
                    if w != r:
                        outgoing[w].add(r)
                        incoming[r].add(w)
    else:
        producers = {}
        for step_path, info in step_dependencies.items():
            for out_path in info.get('output_paths', []):
                producers[tuple(out_path)] = step_path
        for step_path, info in step_dependencies.items():
            for in_path in info.get('input_paths', []):
                producer = producers.get(tuple(in_path))
                if producer is not None and producer != step_path:
                    incoming[step_path].add(producer)
                    outgoing[producer].add(step_path)

    ordered = []
    remaining = {sp: set(preds) for sp, preds in incoming.items()}
    ready = sorted(sp for sp, preds in remaining.items() if not preds)
    while ready:
        step = ready.pop(0)
        ordered.append(step)
        for consumer in outgoing[step]:
            remaining[consumer].discard(step)
            if not remaining[consumer] and consumer not in ordered and consumer not in ready:
                ready.append(consumer)
        ready.sort()

    if len(ordered) != len(step_paths):
        missing = set(step_paths) - set(ordered)
        raise ValueError(
            f"step graph contains a cycle; could not order: {sorted(missing)!r}")
    return ordered
```

Then, in `render_composite`, pass node_dependencies. Change:
```python
    order = _topological_order(step_paths, step_dependencies)
```
to:
```python
    node_dependencies = getattr(composite, 'node_dependencies', None)
    order = _topological_order(step_paths, step_dependencies, node_dependencies)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `… -m pytest process_bigraph/tests/test_nextflow_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/nextflow.py process_bigraph/tests/test_nextflow_render.py
git commit -m "fix(nextflow): infer DAG edges from node_dependencies (nested-store paths)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `run_composite` — whole-Composite per-task runner

Sibling to `run_step.py`. Loads a composite document, overlays optional initial state, advances the sim, writes the final state document and/or bridge outputs. This is the CLI a Nextflow task invokes to run one whole simulation.

**Files:**
- Create: `process_bigraph/run_composite.py`
- Test: `process_bigraph/tests/test_run_composite.py`

**Interfaces:**
- Produces:
  ```python
  run_composite(document_path: str, *, steps: float,
                initial_state: Optional[dict] = None,
                out_paths: Optional[Dict[str, str]] = None,
                state_out_path: Optional[str] = None) -> Optional[dict]
  ```
  Returns the bridge-output dict (or None). Side effects: writes each `out_paths[port]` and, if given, `state_out_path` (a `{schema, state}` document).
- CLI: `python -m process_bigraph.run_composite --document DOC --steps N [--initial-state @f.json] [--out PORT=PATH]... [--state-out PATH]`

- [ ] **Step 1: Write the failing test**

```python
# process_bigraph/tests/test_run_composite.py
import json
from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Process


class _Incr(Process):
    """Minimal temporal process: pushes 'level' up every unit of time."""
    config_schema = {'rate': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        return {'level': self.config['rate'] * interval}


def _make_core():
    core = allocate_core()
    core.register_link('_Incr', _Incr)
    return core


def _incr_document():
    core = _make_core()
    state = {
        'level': 1.0,
        'incr': {
            '_type': 'process',
            'address': 'local:_Incr',
            'config': {'rate': 2.0},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': 1.0,
        },
    }
    composite = Composite({'state': state}, core=core)
    return {'schema': composite.serialize_schema(),
            'state': composite.serialize_state()}


def test_run_composite_advances_and_writes_state(tmp_path):
    doc_path = tmp_path / 'doc.json'
    doc_path.write_text(json.dumps(_incr_document()))
    out_path = tmp_path / 'final.json'

    from process_bigraph.run_composite import run_composite
    run_composite(str(doc_path), steps=5.0, state_out_path=str(out_path))

    final = json.loads(out_path.read_text())
    # Ran for 5 time units at rate 2.0 → level grew above its start (1.0).
    assert float(final['state']['level']) > 1.0


def test_run_composite_initial_state_overlay(tmp_path):
    doc_path = tmp_path / 'doc.json'
    doc_path.write_text(json.dumps(_incr_document()))
    out_path = tmp_path / 'final.json'

    from process_bigraph.run_composite import run_composite
    run_composite(str(doc_path), steps=0.0,
                  initial_state={'level': 42.0},
                  state_out_path=str(out_path))

    final = json.loads(out_path.read_text())
    assert float(final['state']['level']) == 42.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… -m pytest process_bigraph/tests/test_run_composite.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'process_bigraph.run_composite'`.

- [ ] **Step 3: Write the implementation**

```python
# process_bigraph/run_composite.py
"""Run a whole Composite as one batch task (Nextflow / Snakemake / shell).

Loads a composite document (``{schema, state}``), overlays an optional
initial-state document, advances the simulation, and writes the resulting
state document and/or bridge outputs. The mother→daughter handoff of
vEcoli's Nextflow workflow becomes: one task's ``--state-out`` is the next
task's ``--initial-state``.

CLI::

    python -m process_bigraph.run_composite \\
        --document DOC.json --steps N \\
        [--initial-state @init.json] \\
        [--out PORT=PATH]... [--state-out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively merge ``overlay`` into ``base`` (overlay wins on leaves)."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        for key, value in overlay.items():
            base[key] = _deep_merge(base.get(key), value)
        return base
    return overlay


def _write_json(path: str, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(value, fh, indent=2, default=repr)


def run_composite(document_path: str, *, steps: float,
                  initial_state: Optional[Dict[str, Any]] = None,
                  out_paths: Optional[Dict[str, str]] = None,
                  state_out_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from process_bigraph import Composite, allocate_core

    with open(document_path) as fh:
        document = json.load(fh)
    if initial_state:
        document['state'] = _deep_merge(document.get('state', {}), initial_state)

    # Round-trip through a temp file so Composite.load owns deserialization.
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(document, tmp)
        tmp_path = tmp.name

    core = allocate_core()
    composite = Composite.load(tmp_path, core=core)

    composite.run(float(steps))

    bridge_outputs = composite.read_bridge()
    for port, path in (out_paths or {}).items():
        if not bridge_outputs or port not in bridge_outputs:
            available = sorted(bridge_outputs) if bridge_outputs else []
            raise KeyError(
                f"composite produced no bridge output for port {port!r}; "
                f"available: {available}")
        _write_json(path, bridge_outputs[port])

    if state_out_path is not None:
        _write_json(state_out_path, {
            'schema': composite.serialize_schema(),
            'state': composite.serialize_state()})

    return bridge_outputs


def _parse_out_args(pairs):
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"--out expects PORT=PATH, got {pair!r}")
        port, path = pair.split('=', 1)
        out[port] = path
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m process_bigraph.run_composite',
        description='Run a whole Composite as one batch task.')
    p.add_argument('--document', required=True, help='Composite document JSON')
    p.add_argument('--steps', type=float, required=True,
                   help='Advance simulation time by this amount')
    p.add_argument('--initial-state', dest='initial_state',
                   help='JSON file with a state overlay (or @file.json)')
    p.add_argument('--out', dest='out_pairs', action='append', default=[],
                   metavar='PORT=PATH', help='Per bridge-output destination')
    p.add_argument('--state-out', dest='state_out_path',
                   help='Write the final {schema, state} document here')
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    initial_state = None
    if args.initial_state:
        raw = args.initial_state[1:] if args.initial_state.startswith('@') else args.initial_state
        with open(raw) as fh:
            initial_state = json.load(fh)
    run_composite(
        args.document, steps=args.steps, initial_state=initial_state,
        out_paths=_parse_out_args(args.out_pairs),
        state_out_path=args.state_out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `… -m pytest process_bigraph/tests/test_run_composite.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/run_composite.py process_bigraph/tests/test_run_composite.py
git commit -m "feat(nextflow): run_composite runner — a whole Composite as one task

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Render a whole-Composite node as a `run_composite` task

Extend the renderer so a node whose instance is a `Composite` becomes a Nextflow process whose script runs the whole sim via `run_composite`. Also add a `python` interpreter option (default `'python'`) threaded into both script emitters, so `deploy()` can pin the exact interpreter for subprocess tasks (Task 5).

**Files:**
- Modify: `process_bigraph/nextflow.py` (`render_composite`, `_script_body`; add `_composite_node_script`, `_python_exe` option plumbing)
- Test: `process_bigraph/tests/test_nextflow_render.py`

**Interfaces:**
- Consumes: `run_composite` CLI from Task 2 (`python -m process_bigraph.run_composite --document … --steps …`).
- Produces:
  - `_composite_node_script(instance, doc_ref, steps, inputs_wires, outputs_wires, python='python') -> str` — the `script:` block string for a composite node.
  - `render_composite(composite, options=None)` now also renders `Composite` instances found in `composite.process_paths`; `options` recognizes `python` (default `'python'`), `composite_steps` (default `1000`), and `composite_documents` (`{step_name: doc_path}`, default `f'{step_name}_document.json'`).

- [ ] **Step 1: Write the failing tests**

```python
# append to process_bigraph/tests/test_nextflow_render.py
from types import SimpleNamespace
from process_bigraph import Composite, allocate_core
from process_bigraph.nextflow import _composite_node_script, render_composite


def test_composite_node_script_emits_run_composite():
    script = _composite_node_script(
        instance=None, doc_ref='sim_document.json', steps=1000,
        inputs_wires={'init': ['init_store']},
        outputs_wires={'results': ['results_store']},
        python='python')
    assert 'run_composite' in script
    assert '--document sim_document.json' in script
    assert '--steps 1000' in script


def test_render_composite_emits_composite_node():
    # An outer network with one Composite node (no steps). Fake the outer
    # composite's attributes the renderer reads; the node instance is a real
    # (empty) Composite so the isinstance check fires.
    core = allocate_core()
    inner = Composite({'state': {}}, core=core)
    outer = SimpleNamespace(
        step_paths={},
        step_dependencies={},
        node_dependencies={},
        process_paths={('sim',): {
            'instance': inner,
            'inputs': {'init': ['init_store']},
            'outputs': {'results': ['results_store']},
        }},
        bridge={},
    )
    nf = render_composite(outer)
    assert 'process sim {' in nf
    assert 'run_composite' in nf
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `… -m pytest process_bigraph/tests/test_nextflow_render.py -v`
Expected: FAIL — `ImportError: cannot import name '_composite_node_script'`.

- [ ] **Step 3: Implement**

Add to `process_bigraph/nextflow.py`:

```python
def _composite_node_script(instance, doc_ref, steps, inputs_wires,
                           outputs_wires, python='python'):
    """Emit the ``script:`` block for a whole-Composite node.

    Runs the entire nested simulation via ``run_composite``: the first input
    port (if any) is staged as the initial-state document; the first output
    port (if any) receives the final-state document.
    """
    parts = [
        f'{python} -m process_bigraph.run_composite',
        f'--document {doc_ref}',
        f'--steps {steps}',
    ]
    in_iter = iter(inputs_wires)
    first_in = next(in_iter, None)
    if first_in is not None:
        parts.append(f'--initial-state ${{{first_in}}}')
    out_iter = iter(outputs_wires)
    first_out = next(out_iter, None)
    if first_out is not None:
        parts.append(f'--state-out {first_out}.json')
    cmd = ' \\\n    '.join(parts)
    return f'"""\n{cmd}\n"""'
```

In `_script_body`, add a `python` parameter (default `'python'`) and use it in the `run_step` command instead of the literal `'python -m process_bigraph.run_step'`:
```python
def _script_body(instance, step_name, inputs_wires, outputs_wires, python='python'):
    ...
    parts = [
        f'{python} -m process_bigraph.run_step',
        f'--class {fq}',
    ]
    ...
```

Import `Composite` lazily inside `render_composite` (avoid a top-level cycle) and render process_paths Composite nodes. After the existing `for step_path in order:` loop that emits step blocks, add:

```python
    from process_bigraph.composite import Composite as _Composite
    python = options.get('python', 'python')
    default_steps = options.get('composite_steps', 1000)
    doc_map = options.get('composite_documents', {})

    for node_path, node in (getattr(composite, 'process_paths', {}) or {}).items():
        instance = node.get('instance')
        if not isinstance(instance, _Composite):
            continue
        name = _path_to_step_name(node_path)
        inputs_wires = node.get('inputs') or {}
        outputs_wires = node.get('outputs') or {}
        doc_ref = doc_map.get(name, f'{name}_document.json')

        block_lines = [f'process {name} {{']
        if inputs_wires:
            block_lines.append('    input:')
            for port in inputs_wires:
                block_lines.append(f'    path {port}')
        if outputs_wires:
            block_lines.append('    output:')
            for port in outputs_wires:
                block_lines.append(f'    path "{port}.json"')
        block_lines.append('    script:')
        block_lines.append(_composite_node_script(
            instance, doc_ref, default_steps, inputs_wires, outputs_wires, python))
        block_lines.append('}')
        process_blocks.append('\n'.join(block_lines))

        call_args = [
            _channel_expr_for_input(port, wire, path_to_channel, None, bridge_inputs)
            for port, wire in inputs_wires.items()]
        out_port, out_wire = next(iter(outputs_wires.items()), (None, None))
        if out_wire is not None:
            out_channel = _path_to_channel_name(tuple(out_wire))
            workflow_lines.insert(-1, f'    {out_channel} = {name}({", ".join(call_args)})')
        else:
            workflow_lines.insert(-1, f'    {name}({", ".join(call_args)})')
```

(Also thread `python` into the existing `_process_block`/`_script_body` call so step tasks honor the interpreter: pass `options.get('python', 'python')` down. Keep the default `'python'` so current behavior is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `… -m pytest process_bigraph/tests/test_nextflow_render.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/nextflow.py process_bigraph/tests/test_nextflow_render.py
git commit -m "feat(nextflow): render a whole Composite node as a run_composite task

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Generate `nextflow.config` with executor profiles

A pure function that renders a `nextflow.config` string with `local` + `slurm` profiles (real) and `awsbatch` + `google-batch` (stubbed), plus optional per-label resource directives.

**Files:**
- Create: `process_bigraph/nextflow_deploy.py`
- Test: `process_bigraph/tests/test_nextflow_deploy.py`

**Interfaces:**
- Produces:
  ```python
  generate_nextflow_config(executor: str = 'local',
                           resources: Optional[Dict[str, Dict[str, Any]]] = None,
                           params: Optional[Dict[str, Any]] = None) -> str
  ```
  `resources` maps a Nextflow label → `{'cpus': int, 'memory': str, 'time': str}` rendered as `withLabel:`. `params` become `params { }` entries.

- [ ] **Step 1: Write the failing test**

```python
# process_bigraph/tests/test_nextflow_deploy.py
from process_bigraph.nextflow_deploy import generate_nextflow_config


def test_config_has_requested_profiles_and_resources():
    cfg = generate_nextflow_config(
        executor='slurm',
        resources={'sim': {'cpus': 4, 'memory': '8 GB', 'time': '2h'}},
        params={'publishDir': 'results'})
    assert 'profiles {' in cfg
    assert 'local {' in cfg
    assert 'slurm {' in cfg
    assert "executor = 'slurm'" in cfg
    assert 'withLabel: sim' in cfg
    assert 'cpus = 4' in cfg
    assert "publishDir = 'results'" in cfg


def test_config_default_executor_local():
    cfg = generate_nextflow_config()
    assert 'local {' in cfg
    assert "executor = 'local'" in cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… -m pytest process_bigraph/tests/test_nextflow_deploy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'process_bigraph.nextflow_deploy'`.

- [ ] **Step 3: Implement**

```python
# process_bigraph/nextflow_deploy.py
"""Generate nextflow.config + deploy a Composite's Step network to a backend.

Wraps process_bigraph.nextflow.render_composite (which emits main.nf) with a
nextflow.config profile block and an optional `nextflow run` launch. The
executor abstraction mirrors vEcoli's runscripts/nextflow/config.template:
one `profiles { }` block, backend selected by name.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _resource_lines(resources: Optional[Dict[str, Dict[str, Any]]]) -> str:
    if not resources:
        return ''
    blocks = []
    for label, res in resources.items():
        lines = [f'        withLabel: {label} {{']
        if 'cpus' in res:
            lines.append(f'            cpus = {res["cpus"]}')
        if 'memory' in res:
            lines.append(f'            memory = {res["memory"]!r}')
        if 'time' in res:
            lines.append(f'            time = {res["time"]!r}')
        lines.append('        }')
        blocks.append('\n'.join(lines))
    return '\n'.join(blocks)


def _params_block(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ''
    lines = ['params {']
    for key, value in params.items():
        lines.append(f'    {key} = {value!r}')
    lines.append('}')
    return '\n'.join(lines) + '\n\n'


def generate_nextflow_config(executor: str = 'local',
                             resources: Optional[Dict[str, Dict[str, Any]]] = None,
                             params: Optional[Dict[str, Any]] = None) -> str:
    res = _resource_lines(resources)
    res_block = ('\n' + res) if res else ''
    return f"""{_params_block(params)}profiles {{
    local {{
        process {{
            executor = 'local'{res_block}
        }}
    }}
    slurm {{
        process {{
            executor = 'slurm'
            errorStrategy = {{ task.attempt <= 3 ? 'retry' : 'finish' }}{res_block}
        }}
        executor.queueSize = 100
        executor.submitRateLimit = '20/min'
    }}
    awsbatch {{
        // STUB (untested in v1)
        process {{ executor = 'awsbatch' }}
    }}
    'google-batch' {{
        // STUB (untested in v1)
        process {{ executor = 'google-batch' }}
    }}
}}
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `… -m pytest process_bigraph/tests/test_nextflow_deploy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/nextflow_deploy.py process_bigraph/tests/test_nextflow_deploy.py
git commit -m "feat(nextflow): generate nextflow.config executor profiles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `deploy()` — write files + launch `nextflow run`

Ties it together: write `main.nf` (via `render_composite`, pinning the current interpreter) and `nextflow.config` into `outdir`; optionally shell out to `nextflow run`. Includes a real end-to-end integration test gated on the `nextflow` binary.

**Files:**
- Modify: `process_bigraph/nextflow_deploy.py` (add `deploy`)
- Test: `process_bigraph/tests/test_nextflow_deploy.py`

**Interfaces:**
- Consumes: `generate_nextflow_config` (Task 4); `render_composite(composite, options)` with the `python` option (Task 3).
- Produces:
  ```python
  deploy(composite, *, outdir: str, executor: str = 'local',
         launch: bool = False, resources=None, params=None,
         options=None, work_dir=None) -> Dict[str, str]
  ```
  Returns `{'main_nf': path, 'config': path, 'returncode': int|None}`. Writes the two files; when `launch=True` runs `nextflow -C <config> run <main.nf> -profile <executor> [-work-dir <work_dir>]` and raises `subprocess.CalledProcessError` on failure. The generated scripts use `sys.executable` so Nextflow's subprocess tasks use this interpreter.

- [ ] **Step 1: Write the failing tests**

```python
# append to process_bigraph/tests/test_nextflow_deploy.py
import json
import shutil
import pytest
from types import SimpleNamespace
from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Step
from process_bigraph.nextflow_deploy import deploy


class _EmitStep(Step):
    """Writes a constant to its output store when it fires."""
    def inputs(self):
        return {'seed': 'integer'}

    def outputs(self):
        return {'value': 'integer'}

    def update(self, state):
        return {'value': int(state.get('seed', 0)) + 1}


def _emit_core():
    core = allocate_core()
    core.register_link('_EmitStep', _EmitStep)
    return core


def _emit_composite():
    state = {
        'seed': 3,
        'emit': {
            '_type': 'step',
            'address': 'local:_EmitStep',
            'config': {},
            'inputs': {'seed': ['seed']},
            'outputs': {'value': ['value']},
        },
        'value': 0,
    }
    return Composite({'state': state}, core=_emit_core())


def test_deploy_writes_files(tmp_path):
    composite = _emit_composite()
    result = deploy(composite, outdir=str(tmp_path), executor='local', launch=False)
    assert (tmp_path / 'main.nf').exists()
    assert (tmp_path / 'nextflow.config').exists()
    assert result['returncode'] is None
    # main.nf pins this interpreter for task subprocesses.
    import sys
    assert sys.executable in (tmp_path / 'main.nf').read_text()


@pytest.mark.skipif(shutil.which('nextflow') is None,
                    reason='nextflow binary not on PATH')
def test_deploy_launch_local_end_to_end(tmp_path):
    composite = _emit_composite()
    result = deploy(composite, outdir=str(tmp_path), executor='local',
                    launch=True, work_dir=str(tmp_path / 'work'))
    assert result['returncode'] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `… -m pytest process_bigraph/tests/test_nextflow_deploy.py -v`
Expected: FAIL — `ImportError: cannot import name 'deploy'`.

- [ ] **Step 3: Implement `deploy`**

Append to `process_bigraph/nextflow_deploy.py`:

```python
def deploy(composite, *, outdir: str, executor: str = 'local',
           launch: bool = False, resources=None, params=None,
           options=None, work_dir=None) -> Dict[str, Optional[str]]:
    from process_bigraph.nextflow import render_composite

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    render_options = dict(options or {})
    render_options.setdefault('python', sys.executable)

    main_nf = out / 'main.nf'
    main_nf.write_text(render_composite(composite, render_options))

    config = out / 'nextflow.config'
    config.write_text(generate_nextflow_config(
        executor=executor, resources=resources, params=params))

    returncode: Optional[int] = None
    if launch:
        if shutil.which('nextflow') is None:
            raise RuntimeError('nextflow binary not found on PATH')
        cmd = ['nextflow', '-C', str(config), 'run', str(main_nf),
               '-profile', executor]
        if work_dir is not None:
            cmd += ['-work-dir', str(work_dir)]
        proc = subprocess.run(cmd, cwd=str(out))
        returncode = proc.returncode
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    return {'main_nf': str(main_nf), 'config': str(config),
            'returncode': returncode}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `… -m pytest process_bigraph/tests/test_nextflow_deploy.py -v`
Expected: PASS. (`test_deploy_launch_local_end_to_end` runs for real — `nextflow` is on PATH here. If the toy step network produces no channel output, adjust the emit step wiring so the workflow has at least one process invocation, but do NOT weaken the assertion.)

- [ ] **Step 5: Run the full new suite + commit**

Run: `… -m pytest process_bigraph/tests/test_nextflow_render.py process_bigraph/tests/test_run_composite.py process_bigraph/tests/test_nextflow_deploy.py -v`
Expected: all PASS.

```bash
git add process_bigraph/nextflow_deploy.py process_bigraph/tests/test_nextflow_deploy.py
git commit -m "feat(nextflow): deploy() writes main.nf + config and launches nextflow run

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Part A (deploy/config/profiles) → Tasks 4 + 5. ✓
- Part B (run_composite runner + renderer branch) → Tasks 2 + 3. ✓
- Part C (file/URI handoff via `path` ports) → Task 3 emits `path` in/out + `--initial-state`/`--state-out`; Task 2 implements the overlay/write. ✓
- Fix (full dependency edges) → Task 1. ✓
- Testing (unit + `nextflow`-gated integration; feature had zero tests) → Tasks 1–5, integration in Task 5. ✓
- Non-goals honored (no inline subworkflow, aws/gcb stubbed, no HyperQueue, no study DB). ✓

**Placeholder scan:** No TBD/TODO in task steps. The two `//` STUB markers in the generated config are intentional spec-declared stubs, not plan gaps.

**Type consistency:** `_topological_order(step_paths, step_dependencies, node_dependencies=None)` — defined Task 1, called with 3 args in Task 3's `render_composite`. `_composite_node_script(instance, doc_ref, steps, inputs_wires, outputs_wires, python)` — signature matches its Task 3 test and call site. `run_composite(document_path, *, steps, initial_state, out_paths, state_out_path)` — matches Task 2 tests and Task 3's emitted CLI flags (`--document/--steps/--initial-state/--state-out`). `deploy(...)` returns `{'main_nf','config','returncode'}` — matches Task 5 assertions. `generate_nextflow_config(executor, resources, params)` — matches Task 4/5. ✓

## Open follow-ups (out of scope, noted for later)

- Sweep/fan-out (`sweep={param:[...]}`) — the spec's driver API mentioned it; deferred to a follow-up plan (v1 deploys the network as-is).
- Plain temporal `Process` nodes (non-Composite) in the outer network — v1 renders only `Composite` instances from `process_paths`.
- `-resume` / cache-hash invalidation and cloud-profile testing.
