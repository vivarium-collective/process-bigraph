# Nextflow deployment of the process-bigraph Step network

**Status:** Design approved (brainstorming), pending spec review → implementation plan.
**Date:** 2026-08-13
**Branch:** `nextflow-deploy` (worktree off `origin/main`)

## Goal

Let a process-bigraph `Composite`'s **Step network** be deployed and run as a
**Nextflow DSL2 workflow** on a batch backend (local / SLURM, with cloud
profiles stubbed). Each node in the network — including a **whole nested
Composite simulation** — becomes one Nextflow task. Data flows between tasks
as files/URIs, mirroring how vEcoli drives Nextflow (`runscripts/workflow.py`
+ `runscripts/nextflow/*.nf`).

This is a **deployment/compiler backend**, not an `address:`-resolved transport
like `ray:` / `parallel:`. A Composite is not resolved per-`update()`; the whole
simulation is one opaque DAG node. So it lives alongside the existing renderer,
not under `process_bigraph/protocols/`.

## What already exists (prior art — do not rebuild)

Two commits on `origin/main` already compile the Step network to `.nf`:

- **`process_bigraph/nextflow.py` — `render_composite(composite)`**: two-pass
  interpreter over the step graph. Topologically orders steps
  (`_topological_order`, Kahn), assigns **one channel per shared state-path**,
  emits a `process { }` block per Step and a `workflow { }` block wiring
  producers→consumers. Plumbing Steps (`nextflow_operator`) render as channel
  operators (`.mix`/`.combine`/`.groupTuple`/`.join`).
- **`Composite.nextflow(options)`** (`composite.py:1841`) delegates to it.
- **`process_bigraph/run_step.py`**: the per-task CLI each Nextflow process
  invokes — `python -m process_bigraph.run_step --class M.Cls --in port=val
  --out port=out.json`. Runs one Step's `invoke().update`, writes JSON.
- **Step-level Nextflow annotations**: `_cardinality: 'per_match'` scatter
  ports (`composite.py:630`), `nextflow_operator`, `nextflow_directives`,
  `nextflow_port_decls`, `nextflow_script()` escape hatches.
- **Engine already models a Composite as one DAG node**: `class
  Composite(Process)`; `Composite.update()` projects bridge inputs → runs the
  inner sim → returns bridge outputs. Its docstring (`composite.py:3198`):
  *"the whole simulation is one node… the outer network is a higher-order DAG."*

**Gaps this design fills:** (1) no launch/config/executor layer — the renderer
emits a string only; (2) no runner for a whole-Composite task; (3) the
topological-order edge inference is exact-match and drops nested-store edges;
(4) zero tests.

## Design

Three additions + one fix, all layered on the existing renderer.

### Part A — deploy / launch layer (`process_bigraph/nextflow_deploy.py`, new)

```python
def deploy(composite, *, outdir, executor='local', launch=False,
           sweep=None, resources=None, work_dir=None, options=None) -> DeployResult
```

- Writes `main.nf` (via existing `render_composite`) and a generated
  `nextflow.config` (executor **profiles**) into `outdir`.
- If `launch=True`, shells out: `nextflow -C <config> run <main.nf> -profile
  <executor> -work-dir <work_dir>` via `subprocess.run`, streaming output;
  non-zero exit raises.
- Returns paths (+ the completed process handle when launched).
- `resources`: optional `{label|step: {cpus, mem, time}}` folded into
  `withLabel:` directives in the generated config.
- `sweep`: optional `{param_path: [values]}` → Cartesian product exposed as
  Nextflow `params`; v1 keeps this minimal (single-run if omitted).

**Profiles (v1):** a `config.template`-style profiles block, structurally
derived from vEcoli's `runscripts/nextflow/config.template`:
- `local` — `executor = 'local'` (tested end-to-end).
- `slurm` — `executor = 'slurm'`, retry/queue tuning, `withLabel` resource
  scaling (tested where a scheduler is available; otherwise generate-only).
- `awsbatch`, `google-batch` — emitted but **stubbed / untested** in v1.

### Part B — whole Composite as a task (`process_bigraph/run_composite.py`, new + one renderer branch)

- **`run_composite.py`** — sibling to `run_step.py`. CLI:
  ```
  python -m process_bigraph.run_composite \
      --document doc.json [--in port=@state.json]... \
      [--steps N | --duration T] \
      [--out port=final.json]... [--state-out full_state.json]
  ```
  Loads a composite **document** (`{schema, state}` via `Composite.load`),
  applies `--in` values through the **bridge**, runs the sim (N steps or a
  duration), and writes bridge outputs. This is "one Nextflow task = one whole
  simulation," matching vEcoli's `ecoli_master_sim.py` per-generation task.
- **Renderer branch** (`nextflow.py:_script_body`): a node whose instance is a
  `Composite` (or `Process`) emits a `script:` that calls `run_composite`
  instead of `run_step`, referencing a serialized document artifact. Node
  discovery already exists (`find_instance_paths`).

### Part C — state handoff by file/URI (vEcoli daughter-state pattern)

- Whole-composite state travels as a serialized JSON document at a
  `path`-typed port (`_is_file: True`) → Nextflow **stages it as a file**
  between tasks (the renderer already emits `path <name>` for such ports;
  `nextflow.py:103`).
- Small scalars/lists stay as `val` channel values, unchanged.
- Serialization uses the existing `Composite.save`/`load` and
  `core.serialize`/`deserialize`; a slice can be snapshotted via
  `get_path(state, subpath)` + sub-schema serialize.

### Fix — full dependency edges

`_topological_order` (`nextflow.py:130`) infers edges by **exact**
`input_path == output_path`, silently dropping nested-store edges (a step
writes `P`, another reads `P/X`, or vice-versa) that
`scheduling.py:build_step_network` does capture (its prefix-propagation passes).
Lift DAG edges instead from `composite.node_dependencies`
(`nodes[path]['before'] → nodes[path]['after']`, prefix-aware). Guard with a
**failing test first** (TDD) to confirm the gap before changing behavior.

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `nextflow.py` (existing, extended) | Composite → `.nf` string; edge fix; composite-node script branch | `scheduling` structures on the Composite |
| `nextflow_deploy.py` (new) | `nextflow.config` + profiles; `deploy()`/launch | `nextflow.py`, `subprocess` |
| `run_step.py` (existing) | per-Step task runner | Step class |
| `run_composite.py` (new) | whole-Composite task runner | `Composite.load`/`run`/bridge |

## Testing

- **Unit:** render a 3-step network → assert process blocks, workflow wiring,
  topo order.
- **Unit (TDD for the fix):** nested-store wiring (`P` vs `P/X`) → assert the
  A→B edge appears (fails before the fix).
- **Unit:** a `parca(Step) → sim(Composite) → analyze(Step)` network → assert
  the `sim` node emits `run_composite` with a `path` input/output.
- **Unit:** `nextflow_deploy` generates a `nextflow.config` with the requested
  profile and resource directives (generate-only, no binary needed).
- **Integration:** `deploy(..., executor='local', launch=True)` on a toy step
  network end-to-end; assert output JSON. **Skipped** when the `nextflow`
  binary is absent (`shutil.which('nextflow') is None`).

## Non-goals (v1)

- No inline subworkflow expansion of a nested composite (opaque-node only).
- No tested cloud execution (aws/gcb profiles emitted but unverified).
- No HyperQueue meta-scheduler (vEcoli's `slurm_hq`) — follow-up.
- No study/investigation-DB integration — this is framework-level.

## Open questions

- Naming: keep `nextflow_deploy.py` + `nextflow.py`, or register a
  `protocols/nextflow.py` shim for symmetry with `ray`/`parallel`? (Recommend:
  no shim — the semantics differ.)
- `--steps N` vs a stop-condition on the composite document for `run_composite`
  duration control (default: `--steps`, honor a document-level stop if present).
