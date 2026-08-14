# process-bigraph workflow execution — architecture (composite-as-DAG + pluggable backends)

**Status:** Governing architecture design, pending approval → reshape implementation plan.
**Date:** 2026-08-14
**Supersedes framing of:** `2026-08-14-parca-node-nextflow-dag-design.md` (its ParCa/rebuild/artifact
substance is retained below; its Nextflow-as-the-mechanism framing is demoted — Nextflow becomes one
backend). Base renderer/deploy work lives on branch `nextflow-deploy`.

## The reframe

The prior design made Nextflow the mechanism. Evaluating against two goals the user raised — **clean
integration into process-bigraph + the vivarium-workbench**, and **visual workflow editing** — a better
architecture emerges: **the process-bigraph composite IS the authoritative, visually-editable workflow**,
and every workflow *engine* (Nextflow, a Python runner, CWL export) is a **pluggable backend** compiled
*from* the composite. No engine is the source of truth; the bigraph is.

Two goals drive this:
1. **Workbench integration** — the workbench is Python + AI-free and already renders bigraphs (Composite
   Explorer, loom). A Python-native default backend is trivial to call from it; shelling out to Nextflow
   and parsing logs is not.
2. **Visual editing** — no workflow *engine* offers visual authoring (Nextflow/Snakemake/Dask/Airflow
   only *visualize* a DAG). The systems that do — CWL (Rabix/Galaxy), Galaxy, KNIME — are *formats/tools*,
   not our runtime. Since a pbg composite already *is* an editable graph, the visual editor is the bigraph
   editor, and CWL is the standards bridge to external editors.

## Principles

- **The composite (a bigraph step-network) is the single source of truth.** Authored/edited as a graph;
  compiled to any backend. `Composite.nextflow()` becomes one of several `Composite.to_<backend>()`.
- **The task model is backend-agnostic.** Build documents, `run_step`/`run_composite --build`,
  `provision_core`, `ArtifactRef`, `CompositeTask`, scatter — none know about any engine. This is the
  reusable core and it is worth building regardless of which backends ship.
- **Determinism/caching lives in the task model, not the engine.** Content-addressed build-doc hashing +
  `ArtifactRef` fingerprints give "skip ParCa / skip finished seeds" under *every* backend — not only
  Nextflow's `-resume`.
- **Backends are opt-in and additive.** Nothing here changes `composite.run()`, v2ecoli's existing Ray
  batch, or any current path. A user who selects no backend keeps today's behavior exactly.

## Layered architecture

```
            ┌─────────────────────────────────────────────┐
  AUTHOR    │  Composite (bigraph step-network DAG)         │  ← source of truth
            │  = workbench graph editor / study → composite │
            └───────────────────────┬─────────────────────-┘
                                    │ compiles to
        ┌───────────────────────────┼───────────────────────────┐
 EXPORT │ CWL renderer   Nextflow renderer   (mermaid/dot viz)   │ ← for editors/tools
        └───────────────────────────┼───────────────────────────┘
                                    │ executes via  (WorkflowBackend)
   RUN  ┌─────────────┬─────────────┼──────────────┬─────────────┐
        │ LocalRunner  │  RayBackend │ NextflowBack │  (cloud …)  │
        │ (default:    │  (single-   │ (HPC/cloud,  │             │
        │  ProcessPool │   node/mini)│  -resume)    │             │
        │  + hash cache)│            │              │             │
        └─────────────┴─────────────┴──────────────┴─────────────┘
                                    │ per-node task body
              python -m process_bigraph.run_step / run_composite --build
```

### The backend-agnostic task model (retained from prior design)
- **Rebuild, don't rehydrate:** a task receives a *build recipe* (`generator + overrides + sim_data
  ArtifactRef + code_version`), not a serialized composite. Validated by Phase-0 probe P0.1.
- **`run_composite --build` / `run_step`** — per-node CLI runners; `provision_core` gives a fresh
  subprocess its configured core via generator-declared `core_extensions` (+ a `--provision` escape hatch).
- **`ArtifactRef` (kind `sim_data`)** — content-addressed produced-once/consumed-many artifact
  (`process_bigraph/artifacts.py`, already exists). Producer = `ParcaBundleStep`.
- **`CompositeTask`** — a node that builds+runs a whole composite from a generator, scattered over a
  parameter axis (seeds). Knows only `generator/overrides/scatter_param` — zero engine or E.coli coupling.

### The backend interface (new)
```python
# process_bigraph/workflow/backend.py
class WorkflowBackend(Protocol):
    name: str
    def run(self, composite, *, outdir, publish_dir=None, code_version=None,
            **opts) -> "RunResult": ...          # execute the DAG
    def available(self) -> bool: ...             # e.g. nextflow binary / ray import present

@dataclass
class RunResult:
    backend: str
    status: str                                   # 'ok' | 'failed'
    outputs: dict                                 # node -> produced paths/refs
    workdir: str
    provenance: dict                              # code_version, per-node cache hits, timings

def get_backend(name: str) -> WorkflowBackend     # registry: 'local' | 'ray' | 'nextflow'
def run_workflow(composite, *, backend='local', **opts) -> RunResult
```
`deploy()` (the existing Nextflow launcher) is refactored to *be* `NextflowBackend.run`. `run_workflow`
is the one public entry the workbench calls.

### Execution backends
- **`LocalRunner` (default).** Topologically runs the step graph (reusing `_topological_order` +
  `node_dependencies`); executes each node as a subprocess (`run_step`/`run_composite --build`); fans out
  a scatter node via `concurrent.futures.ProcessPoolExecutor` bounded by `max_workers`; **content-hash
  cache**: before running a node, hash its build doc + input artifact refs; if a fingerprint matches a
  prior run's output, skip and reuse. Pure Python, no JVM, ~one focused module. Best workbench fit.
- **`RayBackend`.** Adapts v2ecoli's existing `parallel_seeds`/`run_seeds_parallel` as the scatter
  executor for single-node/mini; the per-seed body is the same `run_composite --build` contract.
- **`NextflowBackend`.** The existing `render_composite` → `nextflow.config` → `nextflow run` path
  (branch `nextflow-deploy`), for HPC/cloud (SLURM/AWS Batch/GCB/HyperQueue) and `-resume` parity with
  vEcoli. Unchanged in capability; now behind the interface.

### Export renderers (representation, for editors/tools)
- **CWL renderer — `Composite.to_cwl()` / `render_cwl(composite)`.** Emits a CWL `Workflow` + one
  `CommandLineTool` per node (the tool's `baseCommand` is `run_step`/`run_composite --build`), with CWL
  **`scatter`** for the fan-out. CWL is a *standard*: the output opens in **Rabix Composer** / **Galaxy**
  for visual editing and runs under cwltool/Toil/Arvados. This is the external-visual-editor bridge.
- **Nextflow renderer** (already exists) doubles as an export format.
- **Diagram** — `mermaid`/`dot` for read-only visualization (cheap; both Nextflow and a direct emitter).

### Visual editing
- **Native:** the workbench renders/edits the composite graph directly (extends Composite Explorer / loom
  — the bigraph *is* the workflow). This is the primary authoring surface.
- **Interop:** `to_cwl()` exports to external visual editors (Rabix/Galaxy) and round-trips back as a
  composite (a CWL→composite importer is a later nicety).

## Studies and Investigations ARE workflow composites (the organizing principle)

The unit everything standardizes on is the **Study**, and the hierarchy is **recursive**: a Study is a
workflow composite; an Investigation is a workflow composite whose *nodes are Study-composites*. Both run
through the one `run_workflow`. This is not a new abstraction bolted on — the workbench already sketched
it and left it unwired:

- **`study_interface(spec)`** (`lib/study_spec.py`) already returns `{composite, config, inputs:
  [{from: <producer_study_slug>}], outputs: [<name>]}` — i.e. a study already declares *a generator
  (`composite`), its overrides (`config`), its producer-study edges (`inputs[].from`), and its output*.
- **`resolve_study`** (dead: `lib/artifacts/pipeline.py:91`, only its test imports it) already recurses
  into producer studies first (the prerequisite DAG walk) and content-addresses each study's output via
  `artifact_id(composite_id=iface.composite, config=iface.config, input_ids=sorted(producer ids),
  commit=workspace_commit)` with pull-or-compute over an `ArtifactStore` — **the identical content
  address our `CompositeTask` cache uses (F1).**

So the standardization is: **replace `resolve_study`'s opaque `compute_fn` with `run_workflow` over a
composite compiled from the spec.** The recursion becomes composite wiring; the pull-or-compute becomes
the task model's fingerprint cache; the study DAG and the per-seed sim cache become one content-addressed
mechanism (which is why the W1 hash lock-step fix is load-bearing).

### The recursive model
```
Process ─▶ Composite ─▶ Study-composite ─▶ Investigation-composite      (all run_workflow-able)
```
- **Study = workflow composite.** `study_to_composite(study_spec) -> Composite`:
  `iface.composite` + `iface.config` → the baseline/variant build; seeds/replicates → a `CompositeTask`
  scatter axis; a prerequisite producer (e.g. ParCa) → an upstream `ParcaBundleStep`/producer node wired
  from another study's output; evaluations (acceptance bands / report-card tests) → downstream analysis
  Steps; the verdict/report → the composite's `bridge` output. The **Phase-3 ParCa→per-seed baseline DAG
  is exactly a study's simulation core**, which is why the engine milestone is study-shaped without
  needing the workbench.
- **Investigation = workflow composite of Study-composites.** `investigation_to_composite(inv) ->
  Composite`: each member study is a node (a nested study-composite — composite-as-node); the edges are
  the `inputs[].from` prerequisites (producer study's output artifact → consumer study's input);
  investigation-level aggregation/verdict is the outer bridge. `resolve_study`'s recursion is the
  reference implementation of this wiring.

### Responsibility split (keeps pbg general)
- **process-bigraph (the engine):** `CompositeTask`, nested composites (composite-as-node), `run_workflow`,
  the fingerprint cache. Knows *nothing* about studies/investigations — it runs recursive composites.
  Under `LocalRunner` (tick + composite-as-node, which the native engine already supports) an
  Investigation→Study→sim composite runs in-process today; on `NextflowBackend` the nested-composite case
  needs the parked composite-node staging/topo fix (so Nextflow-of-investigations is Phase 6).
- **vivarium-workbench (the vocabulary):** `study_to_composite` / `investigation_to_composite` (finishing
  `resolve_study` on the pbg substrate); a **run-backend selector** (`local` | `nextflow` | `sms-api`);
  "Run study"/"Run investigation" call `run_workflow(<compiled composite>, backend=…)` and land
  `RunResult.provenance` in `runs_meta`. The detached-process shell (PID/heartbeat/durability/AI-free) is
  kept; only the run *body* becomes `run_workflow`. Studies and investigations become the same verb.

**AI-free preserved:** `run_workflow` is pure Python in pbg; the workbench already depends on
process-bigraph the right direction. Nothing AI-facing enters the run path.

## What is preserved vs new
- **Preserved:** everything on `nextflow-deploy` (the Nextflow renderer + `deploy`) becomes
  `NextflowBackend`. The prior plan's Phases 1–2 (task model: provision, build docs, ArtifactRef,
  ParcaBundleStep) are unchanged and backend-agnostic. Phase-0 probes still hold.
- **New / re-scoped:** the `WorkflowBackend` interface + `run_workflow`; the **`LocalRunner` default
  backend** (replaces "Nextflow is the milestone" — the new milestone runs the ParCa→2-seed DAG under
  `LocalRunner`, no JVM); the **CWL export renderer**; `RayBackend`; the study→composite builder +
  workbench selector.

## Revised phasing
- **Phase 1 — task model (backend-agnostic).** `provision_core`; `run_composite --build/--set/--provision`;
  generator resolution via `apply_core_extensions`; v2ecoli `register_ecoli_core` split. *(= prior plan
  Phase 1; unchanged.)*
- **Phase 2 — sim_data artifact.** `ParcaBundleStep` → `ArtifactRef`; `run_composite --artifact`. *(= prior
  Phase 2; unchanged.)*
- **Phase 3 — LocalRunner backend + `WorkflowBackend` interface (NEW MILESTONE).** `CompositeTask`;
  `backend.py` (interface + registry + `run_workflow`); `LocalRunner` (topo + ProcessPool scatter +
  content-hash cache). **Milestone:** `run_workflow(parca_sims_composite, backend='local')` runs the
  ParCa(fixture)→2-seed baseline DAG locally end-to-end, second run cache-hits — **no Nextflow required.**
- **Phase 4 — Nextflow + CWL backends behind the interface.** Refactor `deploy` → `NextflowBackend`;
  add `render_cwl` / `to_cwl`; `RayBackend` adapter. Same DAG runs under `backend='nextflow'` and exports
  valid CWL (validated with `cwltool --validate`).
- **Phase 5 — workbench/study integration.** `study_to_composite`; workbench run-backend selector; graph
  editor hooks; `RunResult` provenance surfaced in the study report.
- **Phase 6 — scale-out.** variant×seed scatter; multi-generation lineages; cloud executors / fsspec URIs;
  composite-node topo/staging fix; CWL→composite import for external-editor round-trip.

## Risks / prototype-first
- **LocalRunner caching correctness** — the content-hash skip must key on build doc + input artifact
  fingerprints + `code_version`; a missed input → stale reuse. Prototype the hash key on toy nodes first;
  default to *no cache* unless a fingerprint matches exactly.
- **ProcessPool + heavy sims OOM** — bound `max_workers = min(n, cpus//2)`; ban in-memory emitters (as in
  the prior design).
- **CWL fidelity** — scatter + File staging semantics differ from Nextflow; validate emitted CWL with
  `cwltool` in Phase 4; keep CWL export "best-effort / validated", not a hard guarantee, until a real
  external-editor round-trip is proven.
- **Backend behavioral parity** — the same composite under `local` vs `nextflow` must produce the same
  per-seed outputs; add a cross-backend equivalence test on the toy DAG in Phase 4.
- **Scope** — this is a larger architecture; the task model (Phases 1–2) and `LocalRunner` (Phase 3) are
  the load-bearing minimum. Nextflow/CWL/workbench are independent, separately-valuable follow-ons.
