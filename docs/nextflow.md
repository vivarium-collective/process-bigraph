# Deploying a Step network to Nextflow

Process-Bigraph can compile a `Composite`'s **Step network** into a
[Nextflow](https://www.nextflow.io/) DSL2 workflow and run it on a batch
backend (your laptop, SLURM, and — as untested stubs — AWS Batch / Google
Batch). Each Step becomes one Nextflow task; the framework's own dependency
graph becomes the Nextflow DAG.

This is **entirely opt-in**. Importing `process_bigraph` does not pull in any
of this; nothing changes about `Composite.run()` (normal in-process
execution). You only touch Nextflow when you call `deploy()`.

## When to use it

- Use **`Composite.run()`** (the default) for a single simulation in one
  process.
- Use the **Ray protocol** (`address: "ray:..."`) for fine-grained,
  per-`update()` parallelism *within* one simulation.
- Use **Nextflow deploy** (this page) to run a **DAG of coarse tasks** across
  a cluster with durable, resumable scheduling — the Step network as a batch
  pipeline.

## Requirements

- The `nextflow` binary on your `PATH` — only needed when `launch=True`.
  Rendering the workflow and generating the config are pure Python and need
  nothing extra (no new Python dependencies are added by this feature).
- **Your Step classes must be importable by fully-qualified name.** Each
  Nextflow task runs `python -m process_bigraph.run_step --class
  <module>.<ClassName> ...`, which imports the class with `importlib`. A Step
  defined in a `__main__` script or a REPL cannot be resolved by a task — put
  your Steps in an importable module.

## Quick start

Define a Step in an importable module:

```python
# mysteps/steps.py
from process_bigraph.composite import Step

class AddOne(Step):
    def inputs(self):
        return {'x': 'integer'}
    def outputs(self):
        return {'y': 'integer'}
    def update(self, state):
        return {'y': int(state.get('x', 0)) + 1}
```

Build a Composite and deploy it:

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.nextflow_deploy import deploy
from mysteps.steps import AddOne

core = allocate_core()
core.register_link('AddOne', AddOne)

state = {
    'x': 41,
    'addone': {
        '_type': 'step',
        'address': 'local:AddOne',
        'config': {},
        'inputs':  {'x': ['x']},
        'outputs': {'y': ['y']},
    },
    'y': 0,
}
composite = Composite({'state': state}, core=core)

result = deploy(
    composite,
    outdir='out/run1',      # main.nf + nextflow.config are written here
    executor='local',       # 'local' | 'slurm' | 'awsbatch' | 'google-batch'
    launch=True,            # actually run `nextflow run`; False = just emit files
    params={'x': 41},       # supply any input store not produced by a Step
    work_dir='out/run1/work',
)
print(result)   # {'main_nf': '.../main.nf', 'config': '.../nextflow.config', 'returncode': 0}
```

This emits and runs the following `main.nf`:

```groovy
nextflow.enable.dsl=2

process addone {
    input:
    val x
    output:
    path "y.json"
    script:
"""
/path/to/python -m process_bigraph.run_step \
    --class mysteps.steps.AddOne \
    --in x="${x}" \
    --out y=y.json
"""
}

workflow  {
    ch_y = addone(params.x)
}
```

Each Step task writes each output port to `<port>.json` (here `y.json`) and
reads inputs via `--in`. Stores that no Step produces (like `x` above) are
surfaced as Nextflow `params.<name>` — supply them through `deploy(params=...)`.

## `deploy()` reference

```python
deploy(composite, *, outdir, executor='local', launch=False,
       resources=None, params=None, options=None, work_dir=None)
    -> {'main_nf': str, 'config': str, 'returncode': int | None}
```

- `outdir` — where `main.nf` and `nextflow.config` are written.
- `executor` — selects the `nextflow.config` profile: `local` and `slurm` are
  real; `awsbatch` and `google-batch` are emitted but **untested stubs** in
  this version.
- `launch` — `True` runs `nextflow -C <config> run <main.nf> -profile
  <executor> [-work-dir <work_dir>]` and raises
  `subprocess.CalledProcessError` on a non-zero exit. `False` only writes the
  files (no `nextflow` binary needed) so you can inspect or run them yourself.
- `resources` — `{label: {'cpus': int, 'memory': '8 GB', 'time': '2h'}}`,
  folded into `withLabel:` directives in the generated config.
- `params` — `{name: value}`, rendered into a `params { }` block (booleans and
  `None` become valid Groovy `true`/`false`/`null`).
- `options` — passed to `render_composite` (see below).

Generated scripts are pinned to the current `sys.executable`, so the Nextflow
tasks run under the same interpreter (and virtualenv) that called `deploy()`.

## Rendering only (no launch)

`Composite.nextflow(options)` returns the `main.nf` string directly, and
`render_composite(composite, options)` is the underlying renderer.
`options` keys:

- `workflow_name` — entry workflow name (`deploy()` defaults this to an
  unnamed entry workflow, because `main` is reserved in Nextflow).
- `header` — leading text (default the DSL2 declaration).
- `python` — interpreter used in emitted task scripts (`deploy()` sets this to
  `sys.executable`).
- `composite_steps` — steps to advance a whole-Composite node (default 1000;
  experimental, see below).
- `composite_documents` — `{step_name: document_path}` for whole-Composite
  nodes (experimental, see below).

## Executor profiles

The generated `nextflow.config` carries one `profiles { }` block:

| Profile | Executor | Status |
|---|---|---|
| `local` | `local` | supported, tested end-to-end |
| `slurm` | `slurm` | supported (retry/queue tuning, `withLabel` resources) |
| `awsbatch` | `awsbatch` | **stub, untested** |
| `google-batch` | `google-batch` | **stub, untested** |

## Whole-Composite tasks — experimental

Process-Bigraph also models a *nested Composite as one node* — a whole
simulation as a single task, run via `python -m process_bigraph.run_composite`
with state handed off between tasks as JSON documents (the vEcoli
mother→daughter pattern). `run_composite` itself is functional and tested:

```
python -m process_bigraph.run_composite \
    --document doc.json --steps 1000 \
    [--initial-state @prev_state.json] [--state-out final_state.json]
```

`--state-out` writes a `{schema, state}` document; feed it straight into the
next task's `--initial-state`.

**However, the renderer branch that turns a Composite node into a Nextflow
task is currently EXPERIMENTAL and not yet runnable end-to-end**: the composite
document is not auto-staged as a task input, and composite nodes are not yet
folded into the topological ordering. The **plain Step-network path above is
fully supported**; the whole-Composite-as-a-task path is scaffolding pending
those two gaps. See
`docs/superpowers/specs/2026-08-13-nextflow-step-network-deploy-design.md`
for the design and status.
