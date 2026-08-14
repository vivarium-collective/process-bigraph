# Workflow Execution — Implementation Plan (Phases 1–3, Fable-reviewed)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a process-bigraph composite the authoritative workflow DAG, runnable through a pluggable **`WorkflowBackend`**, and demonstrate the program end-to-end. **Milestone (redefined per Fable whole-program streamline §1.2/§5.1):** a real `study.yaml` compiled to a composite runs under `LocalRunner` (the vivarium engine) — `ParCa(fixture) → per-seed sims → ResultsStep → a gating ReportCard → bridge verdict` — **locally, pure-Python, content-hash cached** (re-run cache-hits, a changed `steps` misses), with the study's sim-cache address equal to `resolve_study`'s `artifact_id` for the same spec. Reached by T8 (engine + Evaluate tail) → T9 → **T10 minimal `study_to_composite`**. No Nextflow.

**Program sequence:** see `2026-08-14-workflow-refactor-roadmap.md` §"Program execution order" for the single streamlined order across all repos (this plan's tasks interleave with the post-sim family plan and Task 0).

**Standardization:** the unit is the **Study-as-workflow-composite**, with an Investigation a workflow composite of Study-composites (governing spec §"Studies and Investigations ARE workflow composites"). This plan builds the general engine those compile to; the Phase-3 `ParCa → per-seed baseline` DAG **is a study's simulation core**. The workbench `study_to_composite`/`investigation_to_composite` compilers (completing dead `resolve_study`) are Phase 5.

**Architecture:** "Rebuild, don't rehydrate" task model (build recipe = `generator + overrides + sim_data ArtifactRef + code_version`), backend-agnostic. The composite (bigraph) is the source of truth; `LocalRunner` **ticks the composite** — the native engine already topo-schedules ready Steps at `run(0.0)` and parallelizes layers via `parallel_steps`; `CompositeTask` self-scatters its seed axis. No backend-owned scheduler (that is what NextflowBackend is for, Phase 4).

**Tech Stack:** Python 3.12, process-bigraph, vivarium-workbench (Task 0 only), v2ecoli, pytest. Nextflow NOT required for this plan.

**Spec:** `docs/superpowers/specs/2026-08-14-workflow-execution-architecture-design.md` (governing) + `2026-08-14-workflow-refactor-roadmap.md` (Fable review; this plan folds its must-fixes F1–F6, S1–S7, R1, R2, R4). **Supersedes:** `2026-08-14-parca-node-nextflow-dag.md`.

**Scope:** Task 0 (cross-repo hash fix, prerequisite) + Phases 1–3. Nextflow/CWL backends, `_topological_order` promotion, and the workbench study integration are Phase 4/5 (roadmap doc).

## Global Constraints

Three repos. Each task is labeled `[pbg]`, `[v2e]`, or `[wb]`.

- **`[pbg]`** — worktree `/Users/eranagmon/code/process-bigraph--nextflow-deploy`, branch `nextflow-deploy`. Test prefix:
  ```
  PYTHONPATH=/Users/eranagmon/code/process-bigraph--nextflow-deploy \
    /Users/eranagmon/code/process-bigraph/.venv/bin/python -m pytest
  ```
  pbg tests MUST NOT import v2ecoli — use a **toy generator** in the test module.
- **`[v2e]`** — worktree off canonical `main` (create before first `[v2e]` commit):
  ```
  git -C /Users/eranagmon/code/v2ecoli fetch origin main
  git -C /Users/eranagmon/code/v2ecoli worktree add /Users/eranagmon/code/v2ecoli--nextflow-parca -b nextflow-parca origin/main
  ```
  Test prefix (viva-emitters on path; run from canonical checkout so `out/cache` resolves; worktree on PYTHONPATH):
  ```
  cd /Users/eranagmon/code/v2ecoli && \
  PYTHONPATH=/Users/eranagmon/code/viva-emitters:/Users/eranagmon/code/v2ecoli--nextflow-parca \
    /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <worktree test path>
  ```
- **`[wb]`** — worktree off vivarium-workbench `main` (create before Task 0's commit):
  ```
  git -C /Users/eranagmon/code/vivarium-workbench fetch origin main
  git -C /Users/eranagmon/code/vivarium-workbench worktree add /Users/eranagmon/code/vivarium-workbench--hash-lockstep -b hash-lockstep origin/main
  ```
  Test: use the workbench's own venv/test runner (see its CLAUDE.md); verify the module under edit resolves in the worktree.
- **Commits** end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Route to the matching branch; never commit in a canonical checkout.
- **Task parallelism (S5):** Task 0 ∥ (T1 ∥ T3 ∥ T4). Then T2→T1; T5→(T2,T4); T6→(T2,T5); T7→T6; T8→all. (Independent tasks may be dispatched concurrently only when they touch different repos/files.)
- **Verified APIs (do not re-derive):** `process_bigraph.artifacts`: `canonical(obj)`, `artifact_id(*, composite_id, config, input_ids=(), commit='')` (`artifacts.py:95`), `artifact_exists(address)` (`:147`), `legacy_artifact_id` (`:126`), `write_fingerprint(address, value)`, `check_fingerprint(results, observed)`, `ArtifactRef(kind, hash='', store='', context={}, fingerprint='')` + `.to_dict()`/`.coerce`, `SIM_DATA='sim_data'`, `ARTIFACT_ROOT='.pbg/artifacts'` (`:37`, cwd-relative — must be pinned). `composite_generator`: `@composite_generator(name=..., core_extensions=[fn])`, `_REGISTRY.values()`, `apply_core_extensions(entry, core)`, `build_generator(entry, overrides=None, core=None)`, resolve by `[e for e in _REGISTRY.values() if e.name == name]`. `composite`: `Composite(doc, core)`, `allocate_core()`, `composite.run(interval)`, `read_bridge()`, `serialize_state()/serialize_schema()`, native `_cardinality:'per_match'` scatter (`composite.py:634-726`), `parallel_steps` config.

## File Structure

| File | Responsibility | Repo |
|---|---|---|
| `process_bigraph/workflow/__init__.py` (new) | re-export `run_workflow`, `get_backend`, `register_backend`, `RunResult` | pbg |
| `process_bigraph/workflow/provision.py` (new) | `provision_core(core, providers)` | pbg |
| `process_bigraph/workflow/recipe.py` (new) | `build_from_recipe(...)` — shared by run_composite + CompositeTask | pbg |
| `process_bigraph/workflow/tasks.py` (new) | `CompositeTask` (per_match scatter + ThreadPool + fingerprint cache) | pbg |
| `process_bigraph/workflow/backend.py` (new) | `WorkflowBackend`, `RunResult`, registry, `run_workflow`, `LocalRunner` | pbg |
| `process_bigraph/run_step.py` (mod) | `--provision` (thin CLI shim) | pbg |
| `process_bigraph/run_composite.py` (mod) | `--build/--set/--artifact/--provision` over `workflow.recipe` | pbg |
| `process_bigraph/protocols/ray.py` (mod) | `_apply_type_providers` → shim over `provision_core` (R4) | pbg |
| `process_bigraph/__init__.py` (mod) | export `run_workflow` | pbg |
| `.../vivarium_workbench/lib/artifacts/hashing.py` (mod) | import pbg `canonical`/`artifact_id` (W1/R7) | wb |
| `v2ecoli/core.py`, `__init__.py`, `composites/ecoli_baseline.py` (mod) | `register_ecoli_core` split + `core_extensions` + `emitter_out_dir` | v2e |
| `v2ecoli/steps/parca_bundle.py` (new) | `ParcaBundleStep` → `ArtifactRef` | v2e |
| `v2ecoli/workflow/build.py` (new) | `build_parca_sim_composite` + `v2ecoli-workflow-run` | v2e |

---

## Task 0 [wb]: Fix the broken pbg↔workbench hash lock-step (prerequisite, before Phase 2)

**Why:** `artifacts.py:30-36` declares `hashing.py` LOCK-STEP; it has diverged — pbg pre-walks `narrow_whole_floats` in `canonical()` (`artifacts.py:84-92`) while the workbench copy keeps it only in the dead `default=` hook (`hashing.py:3-9`), so `artifact_id({'seed': 1.0})` differs across writers and caches silently miss. Must land before Phase 2 sends `sim_data` refs between the repos.

**Files:** modify `vivarium_workbench/lib/artifacts/hashing.py`; adjust its tests.
**Produces:** workbench `canonical`/`artifact_id` become re-exports of pbg's; store migration handled by `legacy_artifact_id`.

- [ ] **Step 1:** write a failing test asserting `hashing.artifact_id(config={'seed': 1.0}) == process_bigraph.artifacts.artifact_id(composite_id=..., config={'seed': 1.0}, ...)` for a shared input (fails today — divergent narrowing).
- [ ] **Step 2:** run — expect FAIL (hashes differ).
- [ ] **Step 3:** replace the body of `hashing.py` with `from process_bigraph.artifacts import canonical, artifact_id  # noqa: F401` (+ keep any workbench-only helpers as thin wrappers). Add a `legacy_artifact_id`-based migration for existing cache-store addresses (rename, not recompute).
- [ ] **Step 4:** run — expect PASS; run the workbench artifact/cache test suite for no regression.
- [ ] **Step 5:** commit on `hash-lockstep` (`fix(wb): single-source artifact hashing from process-bigraph`).

---

## Phase 1 — backend-agnostic task model

### Task 1 [pbg]: `workflow/provision.py` + `run_step --provision` + ray shim (R4)

**Files:** Create `process_bigraph/workflow/__init__.py`, `workflow/provision.py`; modify `run_step.py`, `protocols/ray.py`; Test `tests/test_provision.py`.
**Produces:** `provision_core(core, providers) -> core` (provider = `"module:attr"` or `(module, attr, args, kwargs)`; imports, calls `fn(core, *a, **kw)`, adopts non-None return). `run_step` gains `--provision` applied after `allocate_core()`. `protocols/ray.py::_apply_type_providers` becomes `return provision_core(core, providers)` (keeps the legacy 2-tuple branch inside `provision_core._parse`; fixes Ray silently ignoring provider return values).

- [ ] **Step 1 — failing tests**
```python
# process_bigraph/tests/test_provision.py
from process_bigraph import allocate_core
from process_bigraph.workflow.provision import provision_core
_MARK = {}
def mark_core(core): _MARK['n'] = _MARK.get('n', 0) + 1; core._prov = True; return core
def test_string_provider():
    c = provision_core(allocate_core(), ['process_bigraph.tests.test_provision:mark_core'])
    assert getattr(c, '_prov', False)
def test_noop_empty():
    c = allocate_core(); assert provision_core(c, []) is c
def test_ray_shim_delegates():
    from process_bigraph.protocols.ray import _apply_type_providers
    c = allocate_core(); _MARK['n'] = 0
    _apply_type_providers(c, [('process_bigraph.tests.test_provision', 'mark_core', (), {})])
    assert getattr(c, '_prov', False)   # ray path now honors the provider (and its return)
```
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: process_bigraph.workflow.provision`).
- [ ] **Step 3 — implement** `workflow/provision.py`:
```python
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
`workflow/__init__.py`: `from .provision import provision_core` (add backend exports in T7). In `run_step.py`: `--provision` (append) → after `core = allocate_core()`: `core = provision_core(core, provision)`. In `protocols/ray.py`: replace `_apply_type_providers` body with the shim (import `provision_core` lazily to avoid a cycle).
- [ ] **Step 4 — run, expect PASS**; run existing ray protocol tests (no regression).
- [ ] **Step 5 — commit** (`feat(pbg): workflow.provision_core + run_step --provision + ray shim`).

### Task 2 [pbg]: `workflow/recipe.py` + `run_composite --build`

**Files:** Create `process_bigraph/workflow/recipe.py`; modify `run_composite.py`; Test `tests/test_run_composite_build.py`.
**Consumes:** `provision_core` (T1). **Produces:** `build_from_recipe(build_doc, sets=None, artifacts=None, provision=None) -> (Composite, core)`. Build doc: `{"build": {"generator": <name>, "import": [<module>...], "overrides": {...}, "provision": [...]}, "artifacts": {<key>: {"kind":..,"map":"store"}}, "run": {"steps": N}}`. Order (S3): import `build.import` modules → resolve entry by name in `_REGISTRY` → `core = allocate_core()` → **`apply_core_extensions(entry, core)` FIRST** → **then `provision_core(core, build.provision + provision)`** (generator declaration is the foundation; CLI `--provision` supplements) → merge `overrides` with `sets` → `doc = build_generator(entry, overrides, core)` → `Composite(doc, core), core`. `run_composite(build_path=..., steps=..., sets=..., provision=...)` runs it then `comp.run(steps)`. `--document` XOR `--build`. (`--artifact` parsed here; consumed in T5.)

- [ ] **Step 1 — failing test** (toy generator; note the `provision` module ref is the test module)
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
def ramp_toy(rate=2.0, start=1.0, cache_dir=''):
    return {'state': {'level': start,
        'ramp': {'_type':'process','address':'local:_Ramp','config':{'rate':rate},
                 'inputs':{'level':['level']},'outputs':{'level':['level']}}}}
_IMP = ['process_bigraph.tests.test_run_composite_build']
def test_build_via_generator_and_extensions(tmp_path):
    b = tmp_path/'b.json'; b.write_text(json.dumps(
        {'build':{'generator':'ramp_toy','import':_IMP,'overrides':{'rate':3.0},'provision':[]},'run':{'steps':4}}))
    out = tmp_path/'f.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(b), steps=4.0, state_out_path=str(out))
    assert float(json.loads(out.read_text())['state']['level']) > 1.0
def test_build_set_override(tmp_path):
    b = tmp_path/'b.json'; b.write_text(json.dumps(
        {'build':{'generator':'ramp_toy','import':_IMP,'overrides':{},'provision':[]},'run':{'steps':0}}))
    out = tmp_path/'f.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(b), steps=0.0, sets={'start':41.0}, state_out_path=str(out))
    assert float(json.loads(out.read_text())['state']['level']) == 41.0
```
- [ ] **Step 2 — run, expect FAIL** (`run_composite()` has no `build_path`).
- [ ] **Step 3 — implement** `workflow/recipe.py` (order per S3; `sets` values via `json.loads` w/ raw-string fallback) + the `run_composite` build branch + CLI `--build/--set/--provision/--artifact`. `run_composite.py` imports `workflow.recipe` (stays a top-level CLI shim).
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): workflow.build_from_recipe + run_composite --build`).

### Task 3 [v2e]: `register_ecoli_core` split + `core_extensions` + `emitter_out_dir`
*(unchanged from prior plan — behavior-preserving split; the generic runner must build+run the baseline on a BARE core provisioned only via `core_extensions`.)*

**Files:** modify `v2ecoli/core.py`, `__init__.py`, `composites/ecoli_baseline.py`; Test `v2ecoli/tests/test_workflow_provision.py`.
- [ ] **Step 1 — failing test**
```python
# v2ecoli/tests/test_workflow_provision.py
import os, pytest
from process_bigraph import allocate_core, Composite
from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
CACHE = '/Users/eranagmon/code/v2ecoli/out/cache'
@pytest.mark.skipif(not os.path.isdir(CACHE), reason='no ParCa cache')
def test_baseline_on_bare_core_via_core_extensions():
    entry = next(e for e in _REGISTRY.values() if e.name == 'ecoli_baseline')
    assert entry.core_extensions
    core = apply_core_extensions(entry, allocate_core())        # NO build_core()
    comp = Composite(build_generator(entry, overrides={'seed':0,'cache_dir':CACHE}, core=core), core=core)
    comp.run(5.0); assert comp.state.get('global_time') == 5.0
def test_register_types_hook():
    import v2ecoli; assert v2ecoli.register_types is v2ecoli.core.register_ecoli_core
```
- [ ] **Step 2 — FAIL** (no `core_extensions`; no `register_types`).
- [ ] **Step 3 — implement** the behavior-preserving split (`register_ecoli_core(core)` = post-allocation body of `build_core`; `build_core()` calls it); export `register_types`; add `core_extensions=[register_ecoli_core]` + `emitter_out_dir` param (default `''`) to the generator.
- [ ] **Step 4 — PASS** (`[v2e]` prefix).
- [ ] **Step 5 — commit** on `nextflow-parca` (`refactor(v2e): register_ecoli_core + core_extensions + emitter_out_dir`).

---

## Phase 2 — sim_data artifact producer + consumption

### Task 4 [v2e]: `ParcaBundleStep` (fixture) → `ArtifactRef` (F6: concat-hash)

**Files:** Create `v2ecoli/steps/parca_bundle.py`; Test `v2ecoli/tests/test_parca_bundle.py`.
**Produces:** `class ParcaBundleStep(Step)`, `config_schema = {'mode':'string','cpus':'integer','condition':'maybe[string]','bundle_dir':'string'}`, `outputs() -> {'sim_data': {'_type':'string','_is_file':True}}`, `update(state) -> {'sim_data': ArtifactRef(kind=SIM_DATA, hash=..., store=bundle_dir, context={per-file digests}).to_dict()}`. Fixture/pre-cached only in tests. **Hash (F6):** `sha256(b''.join(sorted([sha256(f).digest() for f in bundle_files])))` — concat of sorted per-file digests, NOT XOR. `write_fingerprint(address, hash)`.

- [ ] **Step 1 — failing test:** assert output dict `kind=='sim_data'`, non-empty `hash`, `store` contains `sim_data_cache.dill`; **and** a determinism check: two `ParcaBundleStep` runs over the same bundle produce the same `hash` (guards against XOR self-cancel and ordering bugs).
- [ ] **Step 2 — FAIL.** **Step 3 — implement** (`from process_bigraph.artifacts import ArtifactRef, SIM_DATA, write_fingerprint`; reuse `save_sim_input`/`load_cache_bundle`). **Step 4 — PASS.**
- [ ] **Step 5 — commit** on `nextflow-parca` (`feat(v2e): ParcaBundleStep emits a sim_data ArtifactRef`).

### Task 5 [pbg]: `run_composite --artifact` consumption + fingerprint attestation

**Files:** modify `run_composite.py`, `workflow/recipe.py`; Test `tests/test_run_composite_build.py` (append).
**Produces:** `build_from_recipe` accepts `artifacts={port: ref_path}`; for each build-doc `artifacts` entry with `map=='store'`, `ArtifactRef.coerce(json.load(ref_path))` → `overrides[key] = ref.store`. `--artifact PORT=REF.json` wired. `check_fingerprint` stays **warn-not-fail** (attestation of nondeterminism, not a cache gate).
- [ ] **Step 1 — failing test** (toy generator taking `cache_dir` echoed into state): `run_composite(build_path=..., artifacts={'cache_dir': ref_json}, ...)`, build doc `artifacts.cache_dir.map='store'`, `ref_json={"kind":"sim_data","store":"/x","hash":"h"}` → assert built state saw `cache_dir=="/x"`.
- [ ] **Step 2 — FAIL. Step 3 — implement. Step 4 — PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): run_composite --artifact injects ArtifactRef.store`).

---

## Phase 3 — LocalRunner backend + WorkflowBackend interface — MILESTONE

### Task 6 [pbg]: `CompositeTask` — `per_match` scatter + ThreadPool + content-hash cache

**Files:** Create `process_bigraph/workflow/tasks.py`; Test `tests/test_composite_task.py`.
**Consumes:** `build_from_recipe` (T2), artifacts (T5), `artifacts.artifact_id`/`artifact_exists`. **Produces:** `class CompositeTask(Step)` (`_cache = 'none'` — MUST NOT set `by_hash`). Config: `{'generator':'string','import':'list[string]','overrides':'node','artifact_params':'map[string]','scatter_param':'maybe[string]','steps':'float','provision':'list[string]','code_version':'maybe[string]','max_workers':'maybe[integer]','artifact_root':'maybe[string]','allow_in_memory_emitter':'boolean'}`.
- **Scatter (R2):** declare the scatter input port with `_cardinality: 'per_match'` (named `scatter_param`) + artifact ports (`{_type:'string',_is_file:True}`). Override `invoke()` to run the per-match set through a **bounded `ThreadPoolExecutor`** (`max_workers or min(n, os.cpu_count()//2)`), each match → one subprocess (S2: threads block on `subprocess.run`, so ThreadPool, not ProcessPool). Do NOT introduce `_scatter:True`.
- **Per match:** compose a build doc `{build:{generator,import,overrides,provision}, artifacts:{k:{kind,map:'store'}}, run:{steps}}` with `overrides['emitter_out_dir'] = <workdir>/seed_<val>/results` (F4, when the generator declares it) and `overrides[scatter_param]=val`.
- **Cache key (F1):** `address = artifact_id(composite_id=generator, config={**overrides, **sets, 'steps':steps, 'provision':provision}, input_ids=[ref.hash for artifact refs], commit=code_version)` where `code_version` defaults to `process_bigraph.__version__` + the `import` module's version. Pin `artifact_root` (config, default `<outdir or cwd>/.pbg/artifacts`).
- **Skip (F2):** if `artifact_exists(address)` AND the output payload dir is present → reuse (cache hit); else run the subprocess `sys.executable -m process_bigraph.run_composite --build <doc> --set <scatter>=<val> --artifact <port>=<ref> --state-out <out>` with **`env=os.environ`** (S1); on success `write_fingerprint(address, <output hash>)`.
- **Provenance (F5):** write `<workdir>/<node>/provenance.json` = `{scatter_val: {address, cache_hit, wall_s}}`.
- **Emitter guard (F4):** refuse a build doc whose emitter resolves to `local:RAMEmitter` (or in-memory) unless `allow_in_memory_emitter`.
- `outputs() -> {'results':'node'}` (`{scatter_val: result_path}`). **No `nextflow_script()`** (S6 — deferred to Phase 4).

- [ ] **Step 1 — failing tests:** (a) native scatter — `CompositeTask` over `ramp_toy` with `scatter_param='start'`, feeding a `per_match` match-set `{ '0': ..., '1': ... }` for `start∈{1.0,2.0}`, returns two results with the right `start`; (b) **cache hit** — a second `invoke` with identical inputs launches **zero** subprocesses (assert via a monotonic run-counter file the test's fake `run_composite` shim increments, or via `provenance.json` `cache_hit:true`); (c) **cache MISS on changed `steps`** — same inputs but `steps` changed → subprocess runs again (address differs); (d) emitter guard — a build doc with a RAM emitter raises unless `allow_in_memory_emitter`.
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: process_bigraph.workflow.tasks`).
- [ ] **Step 3 — implement** `CompositeTask` per the contract above.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): CompositeTask — per_match scatter + fingerprint cache`).

### Task 7 [pbg]: `WorkflowBackend` + `run_workflow` + `LocalRunner`

**Files:** Create `process_bigraph/workflow/backend.py`; modify `workflow/__init__.py`, `process_bigraph/__init__.py`; Test `tests/test_workflow_backend.py`.
**Consumes:** `CompositeTask` (T6). **Produces:**
```python
@dataclass
class RunResult: backend: str; status: str; outputs: dict; workdir: str; provenance: dict
class WorkflowBackend(Protocol):
    name: str
    def available(self) -> bool: ...
    def run(self, composite, *, outdir, code_version=None, **opts) -> RunResult: ...
class LocalRunner:                                   # name='local'
    def available(self): return True
    def run(self, composite, *, outdir, code_version=None, **opts):
        try:
            composite.run(0.0)                       # F3: Step DAGs cascade fully at interval 0
            outputs = composite.read_bridge() or {}  # F3: bridge is the output contract (T8 wires it)
            prov = _gather_provenance(outdir)         # F5: aggregate node provenance.json sidecars
            return RunResult('local', 'ok', outputs, str(outdir), {**prov, 'code_version': code_version or {}})
        except Exception as e:                        # F3: failure path
            return RunResult('local', 'failed', {}, str(outdir), {'error': repr(e)})
_BACKENDS = {}
def register_backend(name, backend): _BACKENDS[name] = backend
def get_backend(name): 
    if name not in _BACKENDS: raise KeyError(f'unknown backend {name!r}; have {sorted(_BACKENDS)}')
    return _BACKENDS[name]
register_backend('local', LocalRunner())
def run_workflow(composite, *, backend='local', outdir='.', **opts):
    return get_backend(backend).run(composite, outdir=outdir, **opts)
```
Document (per §2.3): workflow composites should set `parallel_steps: true` for DAG-branch parallelism; `CompositeTask`'s pool covers the scatter axis. Export `run_workflow`, `get_backend`, `register_backend`, `RunResult` from `workflow/__init__.py` and `run_workflow` from `process_bigraph/__init__.py`.

- [ ] **Step 1 — failing tests:** (a) a toy composite `producer(Step) → CompositeTask(scatter [0,1])` with a declared bridge → `run_workflow(comp, backend='local', outdir=tmp)` returns `RunResult(status='ok')` with two per-seed outputs read via `read_bridge`; (b) `get_backend('nope')` raises; (c) a composite that throws → `status='failed'` with `provenance['error']`; (d) `register_backend`/`get_backend` round-trip.
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: process_bigraph.workflow.backend`).
- [ ] **Step 3 — implement.** **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** (`feat(pbg): WorkflowBackend + run_workflow + LocalRunner`).

### Task 8 [v2e]: `build_parca_sim_composite` + `v2ecoli-workflow-run` — the milestone

**Files:** Create `v2ecoli/workflow/build.py`; add `[project.scripts] v2ecoli-workflow-run`; Test `v2ecoli/tests/test_workflow_dag.py`.
**Produces:** `build_parca_sim_composite(*, seeds, parca_mode='fixture', generator='ecoli_baseline', overrides=None, steps=2700, outdir) -> Composite` — state wires `parca` (`ParcaBundleStep`, `mode=parca_mode`) → `sims` (`CompositeTask`: `generator`, `import=['v2ecoli']`, `scatter_param='seed'`, `artifact_params={'cache_dir':'sim_data'}`, `steps`, `artifact_root=<outdir>/.pbg/artifacts`) + a `seeds` list store, **and a `bridge` exposing `sims.results`** (F3: gives `LocalRunner._collect_outputs`/`read_bridge` its output). Set `parallel_steps: true`. `main()` builds it and calls `process_bigraph.run_workflow(comp, backend=args.backend, outdir=args.outdir, code_version={'v2ecoli': <git sha>})`.

- [ ] **Step 1 — failing tests:** (a) unit — `build_parca_sim_composite(seeds=[0,1], steps=2, outdir=tmp)` returns a Composite whose `sims` node is a `CompositeTask` with `scatter_param=='seed'`, `artifact_params=={'cache_dir':'sim_data'}`, and a bridge exposing results; (b) **milestone integration** (skip if no `out/cache`) — `main(['--seeds','2','--parca-mode','fixture','--steps','5','--backend','local','--outdir',str(tmp)])` returns 0 and produces **per-seed result directories** `tmp/.../seed_0/results` and `seed_1/results` (F4 — assert dirs, not just entries); (c) **cache** — a second identical `main(...)` reports `cache_hit:true` for both seeds in provenance (near-zero sim launches); (d) **cache miss** — `main(..., '--steps','7', ...)` re-runs the sims (address changed by `steps`).
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the builder + `main` (git sha via `subprocess`/`importlib.metadata`; keep `steps` small in tests, default 2700).
- [ ] **Step 4 — run, expect PASS** (`[v2e]` prefix; from canonical checkout so `out/cache` resolves).
- [ ] **Step 5 — commit** on `nextflow-parca` (`feat(v2e): v2ecoli-workflow-run — ParCa→per-seed baseline under LocalRunner`).

### Task 9 [v2e]: Evaluate tail on the real DAG — gating ReportCard → verdict

**Depends on:** T8 (the parca→sims composite) **and** post-sim plan T2 (`ResultsStep`/`ResultsHandle` + the post-sim family in `viva_superpowers`). Wires the study's Evaluate stage onto the *real* CompositeTask output (not a fixture).

**Files:** modify `v2ecoli/workflow/build.py`; Test `v2ecoli/tests/test_workflow_dag.py` (extend).
**Produces:** `build_parca_sim_composite` grows an Evaluate tail: `sims (CompositeTask) → ResultsStep (viva_superpowers) → <one real ReportCardStep> → bridge`. The composite's bridge exposes `verdict` (the gating report card's `{status, checks, summary}`), not raw `sims.results`. `v2ecoli-workflow-run` prints `RunResult.outputs['verdict']`.

- [ ] **Step 1 — failing test:** extend the milestone integration test — after the 2-seed run, assert `result.outputs['verdict']['status'] in {'pass','fail','warn'}` and that the ReportCard consumed the `ResultsHandle` produced by `ResultsStep` from the real `CompositeTask` output (not a fixture). Keep the cache-hit / cache-miss-on-`steps` assertions.
- [ ] **Step 2 — run, expect FAIL** (no Evaluate tail yet).
- [ ] **Step 3 — implement** the tail: add `ResultsStep` + a minimal v2ecoli `ReportCardStep` (or reuse `tests_card`) wired downstream of `sims`; the bridge exposes the verdict. `parallel_steps: true` already set.
- [ ] **Step 4 — run, expect PASS** (`[v2e]`; from canonical checkout so `out/cache` resolves; `viva_superpowers` post-sim family installed/on PYTHONPATH).
- [ ] **Step 5 — commit** on `nextflow-parca` (`feat(v2e): study Evaluate tail — ResultsStep → gating ReportCard → verdict`).

### Task 10 [wb]: minimal `study_to_composite` — the end-to-end

**Depends on:** T9. The minimal slice of workbench W5 (pure compiler; NO UI selector, NO detached-run integration, NO codegen retirement — those are Phase 5).

**Files:** Create `vivarium_workbench/lib/study_to_composite.py`; Test `tests/test_study_to_composite.py`. Worktree: reuse `vivarium-workbench--hash-lockstep` or a fresh `study-to-composite` branch off `origin/main`.
**Produces:** `study_to_composite(spec: dict) -> Composite` — a pure function over `study_interface(spec)` (`{composite, config, inputs[].from, outputs}`) producing the Task-9 shape: `composite`+`config` → the CompositeTask build recipe; seeds → its scatter; `inputs[].from` producers → upstream nodes; declared evaluations → `ReportCardStep` selection; verdict → bridge. Reads YAML today; do the one-line `.json`-filename loader branch here only if free.

- [ ] **Step 1 — failing test:** compile a real fixture `study.yaml` (baseline + seeds + one report card) → `run_workflow(study_to_composite(spec), backend='local', outdir=tmp)` → assert `result.outputs['verdict']` gates, AND assert the study's sim-cache address `== artifact_id(composite_id=iface.composite, config=iface.config, input_ids=[], commit=<ws commit>)` (parity with `resolve_study`, `lib/artifacts/pipeline.py`).
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError: study_to_composite`).
- [ ] **Step 3 — implement** the compiler (small: `study_interface` already yields the inputs; assemble the Task-9 composite shape).
- [ ] **Step 4 — run, expect PASS.** **END-TO-END DONE** — a Study is a workflow composite, gated, cached, address-parity with the dead `resolve_study`.
- [ ] **Step 5 — commit** (`feat(wb): study_to_composite — a Study compiled to a runnable workflow composite`).

---

## Self-Review

**Spec coverage:** hash lock-step repair (T0); provisioning + order (T1, T2 S3); backend-agnostic recipe (T2, `workflow/recipe.py`); ray unification (T1 R4); baseline core-extensions (T3); `sim_data` ArtifactRef concat-hash (T4 F6) + consumption (T5); `CompositeTask` `per_match` scatter (R2) + `artifact_id`/`artifact_exists` cache (F1/F2) + emitter guard/`emitter_out_dir` (F4) + provenance (F5) + ThreadPool (S2) + `env` (S1) + pinned `artifact_root` (T6); `WorkflowBackend`/`run_workflow`/`LocalRunner` with failure path + `read_bridge` outputs + `parallel_steps` note (T7 F3); milestone incl. per-seed dirs + cache hit + cache-miss-on-`steps` (T8). Nextflow/CWL backends, `_topological_order` promotion, workbench study integration = roadmap doc (Phase 4/5).

**Placeholder scan:** none. Each task carries concrete test + implementation code or a precise contract against verified APIs. F3's previously-undefined `_collect_outputs`/`_infer_duration` are now specified (`read_bridge`; `0.0`).

**Type consistency:** `provision_core(core, providers)` (T1) → T2/ray. `build_from_recipe(build_doc, sets, artifacts, provision) -> (Composite, core)` (T2) → T5 (adds artifacts) → T6 (per-scatter docs). `ArtifactRef` dict (`kind/hash/store/context/fingerprint`) T4 ↔ T5/T6. `artifact_id(composite_id, config, input_ids, commit)` / `artifact_exists(address)` used identically in T6 and (via import) T0. `CompositeTask` config keys T6 → T8. `RunResult`/`run_workflow(backend=...)` (T7) → T8. `register_ecoli_core` (T3) ← generator `core_extensions` + T8 `import=['v2ecoli']`.

## Deferred (roadmap doc: Phase 4/5/anytime)
Phase 4: `deploy`→`NextflowBackend`; `_topological_order`→`scheduling.py` (R6); renderer `per_match` (finish dead `port_cardinality`); forward `--provision` in rendered `run_step` (S4); CWL `--validate`; cross-backend equivalence test. Phase 5 (workbench): W2 codegen→build docs → W3 detached study runs → W4 rerun unify → W5 `study_to_composite` (finish dead `pipeline.py`) → W6 sweep→scatter → W7 `SmsApiBackend`. Anytime: R5 (`run.py`+`fire`), W8/W9 hygiene. YAGNI: CWL beyond validate, RayBackend, composite-node topo, async APIs, entry-point discovery.
