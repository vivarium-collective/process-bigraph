# Workflow Execution — Implementation Plan (Phases 1–3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a process-bigraph composite the authoritative workflow DAG, runnable through a pluggable **`WorkflowBackend`**. Milestone: `run_workflow(parca_sims_composite, backend='local')` runs a `ParCa(fixture) → per-seed E.coli baseline` DAG **locally, pure-Python, with content-hash caching** (second run cache-hits) — no Nextflow.

**Architecture:** "Rebuild, don't rehydrate" task model (build recipe = `generator + overrides + sim_data ArtifactRef`), backend-agnostic. The composite (bigraph) is the source of truth; `LocalRunner` executes it via per-node subprocess task runners (`run_step`/`run_composite --build`) with a `ProcessPoolExecutor` scatter and fingerprint caching. Nextflow/CWL are later backends behind the same interface.

**Tech Stack:** Python 3.12, process-bigraph, v2ecoli, pytest. (Nextflow NOT required for this plan.)

**Spec:** `docs/superpowers/specs/2026-08-14-workflow-execution-architecture-design.md` (governing). Subsumed detail: `2026-08-14-parca-node-nextflow-dag-design.md`. **Supersedes plan:** `2026-08-14-parca-node-nextflow-dag.md` (Nextflow-centric).

**Scope:** Phases 1–3. Phase 4 (Nextflow + CWL backends behind the interface, ray refactor), Phase 5 (workbench/study integration), Phase 6 (scale-out) are later plans.

## Global Constraints

Two repos, two worktrees. Each task is labeled `[pbg]` or `[v2e]`.

- **`[pbg]`** — worktree `/Users/eranagmon/code/process-bigraph--nextflow-deploy`, branch `nextflow-deploy`. Test prefix:
  ```
  PYTHONPATH=/Users/eranagmon/code/process-bigraph--nextflow-deploy \
    /Users/eranagmon/code/process-bigraph/.venv/bin/python -m pytest
  ```
  pbg tests MUST NOT import v2ecoli — use a **toy generator** in the test module.
- **`[v2e]`** — create the worktree before the first `[v2e]` commit:
  ```
  git -C /Users/eranagmon/code/v2ecoli fetch origin main
  git -C /Users/eranagmon/code/v2ecoli worktree add /Users/eranagmon/code/v2ecoli--nextflow-parca -b nextflow-parca origin/main
  ```
  Test prefix (viva-emitters on path; run from the canonical checkout so `out/cache` resolves; worktree on PYTHONPATH so edits are under test):
  ```
  cd /Users/eranagmon/code/v2ecoli && \
  PYTHONPATH=/Users/eranagmon/code/viva-emitters:/Users/eranagmon/code/v2ecoli--nextflow-parca \
    /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <worktree test path>
  ```
  Verify `v2ecoli.__file__` is under the worktree before trusting a `[v2e]` test.
- **Commits** end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. `[pbg]`→`nextflow-deploy`; `[v2e]`→`nextflow-parca`. Never commit in a canonical checkout.
- **Verified APIs (do not re-derive):** `process_bigraph.artifacts`: `ArtifactRef(kind, hash='', store='', context={}, fingerprint='')` with `.to_dict()`, `.coerce(dict)`; `SIM_DATA='sim_data'`; `write_fingerprint(address, value)`; `check_fingerprint(results, observed)`. `process_bigraph.composite_generator`: `@composite_generator(name=..., core_extensions=[fn])`; `_REGISTRY` (`.values()`); `apply_core_extensions(entry, core) -> core`; `build_generator(entry, overrides=None, core=None) -> doc`; resolve by name via `[e for e in _REGISTRY.values() if e.name == name]`. `process_bigraph`: `Composite(doc, core)`, `allocate_core()`, `composite.run(interval)`, `serialize_state()`/`serialize_schema()`, `read_bridge()`.

## File Structure

| File | Responsibility | Repo |
|---|---|---|
| `process_bigraph/provision.py` (new) | `provision_core(core, providers)` | pbg |
| `process_bigraph/build_recipe.py` (new) | `build_from_recipe(build_doc, sets, artifacts, provision) -> (Composite, core)` — shared by run_composite + CompositeTask | pbg |
| `process_bigraph/run_composite.py` (mod) | `--build/--set/--artifact/--provision` | pbg |
| `process_bigraph/run_step.py` (mod) | `--provision` | pbg |
| `process_bigraph/tasks.py` (new) | `CompositeTask` — ProcessPool scatter over `run_composite --build` + fingerprint cache | pbg |
| `process_bigraph/workflow/backend.py` (new) | `WorkflowBackend` protocol, `RunResult`, registry, `run_workflow`, `LocalRunner` | pbg |
| `v2ecoli/core.py` (mod) | split `build_core` → `register_ecoli_core` | v2e |
| `v2ecoli/__init__.py` (mod) | `register_types = register_ecoli_core` | v2e |
| `v2ecoli/composites/ecoli_baseline.py` (mod) | `core_extensions` + `emitter_out_dir` | v2e |
| `v2ecoli/steps/parca_bundle.py` (new) | `ParcaBundleStep` → `ArtifactRef` | v2e |
| `v2ecoli/workflow/build.py` (new) | `build_parca_sim_composite` + `v2ecoli-workflow-run` CLI | v2e |

---

## Phase 1 — backend-agnostic task model (closes core-provisioning blocker)

### Task 1 [pbg]: `provision_core` + `run_step --provision`

**Files:** Create `process_bigraph/provision.py`; modify `run_step.py`; Test `tests/test_provision.py`.
**Produces:** `provision_core(core, providers) -> core`; provider is `"module:attr"` or `(module, attr, args, kwargs)`; imports, calls `fn(core, *a, **kw)`, adopts non-None return. `run_step` gains `--provision` applied after `allocate_core()`.

- [ ] **Step 1 — failing test**
```python
# process_bigraph/tests/test_provision.py
from process_bigraph import allocate_core
from process_bigraph.provision import provision_core
_MARK = {}
def mark_core(core):
    _MARK['called'] = True; core._provisioned = True; return core
def test_provision_core_applies_string_provider():
    core = provision_core(allocate_core(), ['process_bigraph.tests.test_provision:mark_core'])
    assert getattr(core, '_provisioned', False) and _MARK.get('called')
def test_provision_core_noop_on_empty():
    c = allocate_core(); assert provision_core(c, []) is c
```
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: process_bigraph.provision`).
- [ ] **Step 3 — implement `provision.py`**
```python
# process_bigraph/provision.py
from __future__ import annotations
import importlib, sys
from typing import Any, Iterable
def _parse(p):
    if isinstance(p, str):
        if ':' not in p: raise ValueError(f"provider must be 'module:attr', got {p!r}")
        m, a = p.split(':', 1); return m, a, (), {}
    return p[0], p[1], tuple(p[2]) if len(p) > 2 else (), dict(p[3]) if len(p) > 3 else {}
def provision_core(core: Any, providers: Iterable) -> Any:
    for prov in providers or []:
        mod, attr, args, kwargs = _parse(prov)
        try:
            fn = getattr(importlib.import_module(mod), attr)
            r = fn(core, *args, **kwargs)
            if r is not None: core = r
        except Exception as e:
            sys.stderr.write(f'[provision] {mod}:{attr} failed: {type(e).__name__}: {e}\n'); raise
    return core
```
In `run_step.py`: add `--provision` (append), thread into `run_step(...)`, and after `core = allocate_core()` do `core = provision_core(core, provision)`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): provision_core hook + run_step --provision`).

### Task 2 [pbg]: `build_recipe` + `run_composite --build` recipe mode

**Files:** Create `process_bigraph/build_recipe.py`; modify `run_composite.py`; Test `tests/test_run_composite_build.py`.
**Consumes:** `provision_core` (T1). **Produces:** `build_from_recipe(build_doc, sets=None, artifacts=None, provision=None) -> (Composite, core)`. Build doc: `{"build": {"generator": <name>, "import": [<module>...], "overrides": {...}, "provision": [...]}, "artifacts": {<key>: {"kind":..,"map":"store"}}, "run": {"steps": N}}`. Steps: import each `build.import` module; resolve entry by name in `_REGISTRY`; `core = allocate_core()`; `provision_core(core, build.provision + provision)`; `apply_core_extensions(entry, core)`; merge `overrides` with `sets`; `doc = build_generator(entry, overrides, core)`; return `Composite(doc, core), core`. `run_composite(build_path=..., steps=..., sets=..., provision=...)` uses it then `comp.run(steps)` + existing `--out`/`--state-out`. `--document` XOR `--build`. (`--artifact` parsed here, consumed in T5.)

- [ ] **Step 1 — failing test** (toy generator, no v2ecoli)
```python
# process_bigraph/tests/test_run_composite_build.py
import json
from process_bigraph.composite import Process
from process_bigraph.composite_generator import composite_generator
class _Ramp(Process):
    config_schema = {'rate': 'float'}
    def inputs(self):  return {'level': 'float'}
    def outputs(self): return {'level': 'float'}
    def update(self, state, interval): return {'level': self.config['rate'] * interval}
def provision_ramp(core): core.register_link('_Ramp', _Ramp); return core
@composite_generator(name='ramp_toy', core_extensions=[provision_ramp])
def ramp_toy(rate=2.0, start=1.0):
    return {'state': {'level': start,
        'ramp': {'_type':'process','address':'local:_Ramp','config':{'rate':rate},
                 'inputs':{'level':['level']},'outputs':{'level':['level']}}}}
_IMP = ['process_bigraph.tests.test_run_composite_build']
def test_build_uses_generator_and_core_extensions(tmp_path):
    b = tmp_path/'b.json'; b.write_text(json.dumps(
        {'build': {'generator':'ramp_toy','import':_IMP,'overrides':{'rate':3.0},'provision':[]},
         'run': {'steps':4}}))
    out = tmp_path/'f.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(b), steps=4.0, state_out_path=str(out))
    assert float(json.loads(out.read_text())['state']['level']) > 1.0
def test_build_set_override(tmp_path):
    b = tmp_path/'b.json'; b.write_text(json.dumps(
        {'build': {'generator':'ramp_toy','import':_IMP,'overrides':{},'provision':[]},'run':{'steps':0}}))
    out = tmp_path/'f.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(b), steps=0.0, sets={'start':41.0}, state_out_path=str(out))
    assert float(json.loads(out.read_text())['state']['level']) == 41.0
```
- [ ] **Step 2 — run, expect FAIL** (`run_composite()` has no `build_path`).
- [ ] **Step 3 — implement** `build_recipe.py` (verified APIs above; `sets` values parsed via `json.loads` with raw-string fallback) and the `run_composite` build branch + CLI `--build/--set/--provision/--artifact`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): build-recipe runner (run_composite --build)`).

### Task 3 [v2e]: split `build_core` → `register_ecoli_core`; declare `core_extensions`; `emitter_out_dir`

**Files:** modify `v2ecoli/core.py`, `__init__.py`, `composites/ecoli_baseline.py`; Test `v2ecoli/tests/test_workflow_provision.py`.
**Produces:** `register_ecoli_core(core) -> core` (all post-`allocate_core` registration `build_core` did); `build_core() = register_ecoli_core(allocate_core())`; `v2ecoli.register_types = register_ecoli_core`; `ecoli_baseline` declares `core_extensions=[register_ecoli_core]` + an `emitter_out_dir` param (default `''`).

- [ ] **Step 1 — failing test** (the generic runner path must build+run the baseline on a BARE core, provisioned only by `core_extensions` — proves the model works without `build_core`):
```python
# v2ecoli/tests/test_workflow_provision.py
import os, pytest
from process_bigraph import allocate_core, Composite
from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
CACHE = '/Users/eranagmon/code/v2ecoli/out/cache'
@pytest.mark.skipif(not os.path.isdir(CACHE), reason='no ParCa cache')
def test_baseline_builds_on_bare_core_via_core_extensions():
    entry = next(e for e in _REGISTRY.values() if e.name == 'ecoli_baseline')
    assert entry.core_extensions
    core = apply_core_extensions(entry, allocate_core())      # NO build_core()
    comp = Composite(build_generator(entry, overrides={'seed':0,'cache_dir':CACHE}, core=core), core=core)
    comp.run(5.0); assert comp.state.get('global_time') == 5.0
def test_register_types_hook():
    import v2ecoli; assert v2ecoli.register_types is v2ecoli.core.register_ecoli_core
```
- [ ] **Step 2 — run, expect FAIL** (no `core_extensions`; no `register_types`).
- [ ] **Step 3 — implement** the behavior-preserving split; export `register_types`; add `core_extensions` + `emitter_out_dir` to the generator (pass `emitter_out_dir` into the emitter config; `''` = current behavior).
- [ ] **Step 4 — run, expect PASS** (`[v2e]` prefix).
- [ ] **Step 5 — commit** on `nextflow-parca` (`refactor(v2e): register_ecoli_core + core_extensions + emitter_out_dir`).

---

## Phase 2 — sim_data artifact producer + consumption

### Task 4 [v2e]: `ParcaBundleStep` (fixture) → `ArtifactRef`

**Files:** Create `v2ecoli/steps/parca_bundle.py`; Test `v2ecoli/tests/test_parca_bundle.py`.
**Produces:** `class ParcaBundleStep(Step)`, `config_schema = {'mode':'string','cpus':'integer','condition':'maybe[string]','bundle_dir':'string'}`, `outputs() -> {'sim_data': {'_type':'string','_is_file':True}}`, `update(state) -> {'sim_data': <ArtifactRef dict>}`. Fixture/pre-cached only in tests (no full ParCa): point `bundle_dir` at an existing valid cache, `save_sim_input`, `hash = sha256(sim_data_cache.dill) ^ sha256(initial_state.json)` (per-file in `context`), `write_fingerprint`, return `ArtifactRef(kind=SIM_DATA, hash=..., store=bundle_dir).to_dict()`.

- [ ] **Step 1 — failing test**: `ParcaBundleStep({'mode':'fixture','bundle_dir':str(tmp)})`, `invoke({}).update`, assert output dict has `kind=='sim_data'`, non-empty `hash`, and `store` containing `sim_data_cache.dill`.
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3 — implement** (`from process_bigraph.artifacts import ArtifactRef, SIM_DATA, write_fingerprint`; reuse `save_sim_input`/`load_cache_bundle` semantics from `v2ecoli/core.py`).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** on `nextflow-parca` (`feat(v2e): ParcaBundleStep emits a sim_data ArtifactRef`).

### Task 5 [pbg]: `run_composite --artifact` consumption + fingerprint check

**Files:** modify `run_composite.py`, `build_recipe.py`; Test `tests/test_run_composite_build.py` (append).
**Produces:** `build_from_recipe` accepts `artifacts={port: ref_path}`; for each `artifacts` entry in the build doc with `map=='store'`, `ArtifactRef.coerce(json.load(ref_path))` → inject `overrides[key] = ref.store`. Optional `check_fingerprint` (warn, not fail). CLI `--artifact PORT=REF.json` (from T2) wired through.

- [ ] **Step 1 — failing test** (toy): a generator taking `cache_dir` echoing it into state; `run_composite(build_path=..., artifacts={'cache_dir': ref_json}, ...)` where the build doc has `artifacts.cache_dir.map='store'` and `ref_json = {"kind":"sim_data","store":"/x","hash":"h"}`; assert built state saw `cache_dir == "/x"`.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** artifact injection.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): run_composite --artifact injects ArtifactRef.store`).

---

## Phase 3 — LocalRunner backend + WorkflowBackend interface — MILESTONE

### Task 6 [pbg]: `CompositeTask` — scattered whole-Composite node with caching

**Files:** Create `process_bigraph/tasks.py`; Test `tests/test_composite_task.py`.
**Consumes:** `build_from_recipe` (T2), `artifacts`/fingerprints (T5). **Produces:** `class CompositeTask(Step)`, `config_schema = {'generator':'string','import':'list[string]','overrides':'node','artifact_params':'map[string]','scatter_param':'maybe[string]','steps':'float','provision':'list[string]','max_workers':'maybe[integer]'}`. `inputs()`: one `{_type:'string',_is_file:True}` port per `artifact_params` key + (if `scatter_param`) a scatter port `{'_type':'list[integer]','_scatter':True}` named `scatter_param`. `outputs() -> {'results':'node'}` (per-scatter result paths keyed by scatter value). `update(state)`: read the scatter list + artifact refs from `state`; for each scatter value, compose a build doc `{build:{generator,import,overrides,provision}, artifacts:{key:{kind,map:'store'}}, run:{steps}}`, fingerprint it (`artifact_id`-style over generator+overrides+scatter value+artifact hashes), **skip if `check_fingerprint` matches a prior output**, else run `python -m process_bigraph.run_composite --build <doc> --set <scatter>=<val> --artifact <port>=<ref> --state-out <out>` in a subprocess via `ProcessPoolExecutor(max_workers or min(n, cpu//2))`; collect `{val: out_path}`. Also `nextflow_script()` (single-scatter form, for the Phase-4 NextflowBackend) + `nextflow_port_decls={'results':'path "results"'}`.

- [ ] **Step 1 — failing tests**: (a) native scatter — `CompositeTask` over `ramp_toy` (from T2's test module) with `scatter_param='start'`, `update({'starts':[1.0,2.0]})` returns two result paths whose composites ran with the right `start`; (b) **cache** — a second `update` with identical inputs performs zero subprocess runs (assert via a run-counter file or fingerprint hits); (c) emission — `nextflow_script()` contains `run_composite --build` and `--set start=${start}`.
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: process_bigraph.tasks`).
- [ ] **Step 3 — implement** `CompositeTask` (subprocess via `sys.executable -m process_bigraph.run_composite`; write each build doc to a temp/work dir; fingerprints via `process_bigraph.artifacts`).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): CompositeTask — cached ProcessPool scatter over run_composite --build`).

### Task 7 [pbg]: `WorkflowBackend` interface + `run_workflow` + `LocalRunner`

**Files:** Create `process_bigraph/workflow/__init__.py`, `process_bigraph/workflow/backend.py`; Test `tests/test_workflow_backend.py`.
**Consumes:** `CompositeTask` (T6). **Produces:**
```python
# process_bigraph/workflow/backend.py
@dataclass
class RunResult:
    backend: str; status: str; outputs: dict; workdir: str; provenance: dict
class WorkflowBackend(Protocol):
    name: str
    def available(self) -> bool: ...
    def run(self, composite, *, outdir, publish_dir=None, code_version=None, **opts) -> RunResult: ...
class LocalRunner:                       # name='local'
    def available(self): return True
    def run(self, composite, *, outdir, publish_dir=None, code_version=None, **opts) -> RunResult:
        composite.run(opts.get('duration', 0.0) or _infer_duration(composite))  # ticks the step network;
        # ParcaBundleStep + CompositeTask nodes do their own subprocess/scatter work inside update()
        return RunResult('local', 'ok', _collect_outputs(composite), str(outdir),
                         {'code_version': code_version or {}})
_BACKENDS = {'local': LocalRunner()}
def get_backend(name) -> WorkflowBackend: ...
def register_backend(name, backend): ...
def run_workflow(composite, *, backend='local', outdir='.', **opts) -> RunResult:
    return get_backend(backend).run(composite, outdir=outdir, **opts)
```
`LocalRunner` executes by ticking the composite's step network (the nodes are Steps; `ParcaBundleStep` produces the ref, `CompositeTask` consumes it and fans out over seeds via its own `ProcessPoolExecutor`). `_collect_outputs` gathers each terminal node's `results`/`sim_data` from composite state. Keep it minimal — no Nextflow, no Groovy.

- [ ] **Step 1 — failing tests**: (a) a toy composite `producer(Step) → CompositeTask(scatter over [0,1])` run via `run_workflow(comp, backend='local', outdir=tmp)` returns `RunResult(status='ok')` with two per-seed outputs; (b) `get_backend('nope')` raises; (c) `register_backend`/`get_backend` round-trip.
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: process_bigraph.workflow`).
- [ ] **Step 3 — implement** the interface, registry, `run_workflow`, `LocalRunner`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): WorkflowBackend interface + run_workflow + LocalRunner`).

### Task 8 [v2e]: `build_parca_sim_composite` + `v2ecoli-workflow-run` — the milestone

**Files:** Create `v2ecoli/workflow/build.py`; add `[project.scripts] v2ecoli-workflow-run`; Test `v2ecoli/tests/test_workflow_dag.py`.
**Consumes:** everything above. **Produces:** `build_parca_sim_composite(*, seeds, parca_mode='fixture', generator='ecoli_baseline', overrides=None, steps=2700) -> Composite` — a composite whose state wires `parca` (`ParcaBundleStep`, `mode=parca_mode`) → `sims` (`CompositeTask`: `generator=generator`, `import=['v2ecoli']`, `scatter_param='seed'`, `artifact_params={'cache_dir':'sim_data'}`, `steps=steps`) + a `seeds` list store. `main()` builds it and calls `process_bigraph.workflow.run_workflow(comp, backend=args.backend, outdir=args.outdir)`.

- [ ] **Step 1 — failing tests**: (a) unit — `build_parca_sim_composite(seeds=[0,1], steps=2)` returns a Composite whose `sims` node is a `CompositeTask` with `scatter_param=='seed'` and `artifact_params=={'cache_dir':'sim_data'}`; (b) **milestone integration** (skip if no `out/cache`) — `main(['--seeds','2','--parca-mode','fixture','--steps','5','--backend','local','--outdir',str(tmp)])` returns 0, and result outputs contain seed 0 and seed 1 entries; (c) **cache** — a second `main(...)` with the same args re-uses fingerprints (near-zero sim subprocess launches; assert via a counter or the provenance).
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the builder + `main` (keep `steps` small in tests; default 2700).
- [ ] **Step 4 — run, expect PASS** (`[v2e]` prefix; from canonical checkout so `out/cache` resolves).
- [ ] **Step 5 — commit** on `nextflow-parca` (`feat(v2e): v2ecoli-workflow-run — ParCa→per-seed baseline under LocalRunner`).

---

## Self-Review

**Spec coverage (Phases 1–3 of the governing design):** provisioning (T1, T3) ✓; backend-agnostic build-recipe reconstruction (T2, shared `build_recipe.py`) ✓; sim_data `ArtifactRef` producer+consumer (T4, T5) ✓; `CompositeTask` w/ ProcessPool scatter + content-hash cache (T6) ✓; `WorkflowBackend` interface + `run_workflow` + `LocalRunner` (T7) ✓; the ParCa→2-seed **LocalRunner** milestone incl. cache-hit (T8) ✓. Determinism: fingerprints (T4/T6), `code_version` in `RunResult.provenance` (T7). Nextflow/CWL backends + ray refactor + workbench integration = Phase 4/5 (deferred).

**Placeholder scan:** no TBD/TODO; each task carries concrete test + implementation code or a precise recipe against verified APIs.

**Type consistency:** `provision_core(core, providers)` (T1) → used by T2 (`build_from_recipe`) → reused by T6 (`CompositeTask`). `build_from_recipe(build_doc, sets, artifacts, provision) -> (Composite, core)` (T2) consistent with T5 (adds `artifacts`) and T6 (per-scatter build docs). `ArtifactRef` dict shape consistent T4 (produce) ↔ T5/T6 (consume). `CompositeTask` config keys (`generator/import/overrides/artifact_params/scatter_param/steps/provision/max_workers`) consistent T6 → T8. `RunResult`/`run_workflow(backend=...)` (T7) called by T8. `register_ecoli_core` (T3) referenced by the generator `core_extensions` and by T8's `import=['v2ecoli']`.

## Deferred (Phase 4–6, later plans)
Phase 4: refactor `deploy`→`NextflowBackend`; `render_cwl`/`to_cwl` (validated w/ `cwltool`); `RayBackend`; refactor `ray.py` type-providers onto `provision_core`; cross-backend equivalence test. Phase 5: `study_to_composite`; workbench run-backend selector + graph-editor hooks; `RunResult` provenance in study reports. Phase 6: variant×seed scatter; multi-generation lineages; cloud/fsspec; composite-node topo/staging fix; CWL→composite import.
