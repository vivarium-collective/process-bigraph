# ParCa-node Nextflow DAG — Implementation Plan (Phases 1–3)

> **SUPERSEDED by `2026-08-14-workflow-execution-phases-1-3.md`.** The task model (Phases 1–2) is
> unchanged, but the milestone is now a pure-Python `LocalRunner` behind a `WorkflowBackend` interface;
> Nextflow is a later backend, not the Phase-3 deliverable. Do not execute this file — use the successor.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run, on Nextflow, a `ParCa → per-seed E.coli baseline` DAG: an upstream Step produces a cached `sim_data` artifact, and downstream sim tasks fan out over seeds, each rebuilding the baseline from that artifact and running to completion. Milestone: `v2ecoli-nextflow --seeds 2 --parca-mode fixture --launch` runs it locally end-to-end with `-resume`.

**Architecture:** "Rebuild, don't rehydrate" — a sim task receives a *build recipe* (`generator + overrides + sim_data ArtifactRef + code version`), not a serialized WCM. Reuses existing `process_bigraph.artifacts.ArtifactRef` (kind `sim_data`) and `composite_generator.core_extensions`/`apply_core_extensions`. The DAG is a pbg Composite of plain Steps (`parca` = `ParcaBundleStep`, `sims` = `CompositeTask`), carried by the supported Step-network renderer.

**Tech Stack:** Python 3.12, process-bigraph, v2ecoli, Nextflow DSL2, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-parca-node-nextflow-dag-design.md` (Fable review; Phase 0 probes passed — fresh-subprocess rebuild + scatter cardinality both confirmed).

**Scope:** Phases 1–3 only. Phase 4 (robustness + composite-node topo fix) and Phase 5 (variant/generation scale-out) are a later plan.

## Global Constraints

**Two repos, two worktrees, two test environments.** Each task is labeled `[pbg]` or `[v2e]`.

- **`[pbg]` — process-bigraph.** Worktree `/Users/eranagmon/code/process-bigraph--nextflow-deploy`, branch `nextflow-deploy` (already exists; continue on it). Test-run prefix:
  ```
  PYTHONPATH=/Users/eranagmon/code/process-bigraph--nextflow-deploy \
    /Users/eranagmon/code/process-bigraph/.venv/bin/python -m pytest
  ```
  pbg tests MUST NOT import v2ecoli — use a **toy generator** defined in the test module.

- **`[v2e]` — v2ecoli.** Create a dedicated worktree off the current canonical `main` before the first `[v2e]` commit:
  ```
  git -C /Users/eranagmon/code/v2ecoli fetch origin main
  git -C /Users/eranagmon/code/v2ecoli worktree add /Users/eranagmon/code/v2ecoli--nextflow-parca -b nextflow-parca origin/main
  ```
  Test-run prefix (viva-emitters on path — the venv is missing `pbg_emitters`; run from the **canonical** checkout dir so `out/cache` resolves, with the worktree on `PYTHONPATH` so edits are under test):
  ```
  cd /Users/eranagmon/code/v2ecoli && \
  PYTHONPATH=/Users/eranagmon/code/viva-emitters:/Users/eranagmon/code/v2ecoli--nextflow-parca \
    /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <worktree-relative test path>
  ```
  Verify `v2ecoli.__file__` resolves inside the worktree before trusting a `[v2e]` test.

- **Nextflow** binary is at `/usr/local/bin/nextflow` (present). Integration tests that launch it are gated `@pytest.mark.skipif(shutil.which('nextflow') is None, ...)`.
- **Commits** end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. `[pbg]` commits on `nextflow-deploy`; `[v2e]` commits on `nextflow-parca`. Never commit in a canonical checkout.
- **Verified APIs (do not re-derive):** `process_bigraph.artifacts.ArtifactRef(kind, hash='', store='', context={}, fingerprint='')` with `.to_dict()`, `.coerce(dict)`, `SIM_DATA='sim_data'`, `write_fingerprint(address, value)`, `check_fingerprint(results, observed)`. `process_bigraph.composite_generator`: `@composite_generator(name=..., core_extensions=[fn])`; `_REGISTRY` (`.values()`, `.get(id)`, `__contains__`); `apply_core_extensions(entry, core) -> core`; `build_generator(entry, overrides=None, core=None) -> doc`. Resolve a generator by name: `[e for e in _REGISTRY.values() if e.name == name]`.

## File Structure

| File | Responsibility | Repo |
|---|---|---|
| `process_bigraph/provision.py` (new) | `provision_core(core, providers)` — apply `"mod:attr"` hooks; shared by run_step/run_composite/ray | pbg |
| `process_bigraph/run_composite.py` (mod) | add `--build/--set/--artifact/--provision` build-doc mode | pbg |
| `process_bigraph/run_step.py` (mod) | add `--provision` | pbg |
| `process_bigraph/tasks.py` (new) | `CompositeTask` Step (native loop + `nextflow_script()`) | pbg |
| `process_bigraph/nextflow.py` (mod) | `_scatter` port → `Channel.of`; `.first()` on queue producers into a scatter process | pbg |
| `process_bigraph/nextflow_deploy.py` (mod) | `deploy(publish_dir, build_documents, code_version)` | pbg |
| `process_bigraph/protocols/ray.py` (mod) | refactor `_apply_type_providers` onto `provision_core` | pbg |
| `v2ecoli/core.py` (mod) | split `build_core` → `register_ecoli_core` + `allocate_core` | v2e |
| `v2ecoli/__init__.py` (mod) | `register_types = register_ecoli_core` | v2e |
| `v2ecoli/composites/ecoli_baseline.py` (mod) | `core_extensions=[register_ecoli_core]` + `emitter_out_dir` param | v2e |
| `v2ecoli/steps/parca_bundle.py` (new) | `ParcaBundleStep` — ParCa → bundle → `ArtifactRef` | v2e |
| `v2ecoli/workflow/nextflow.py` (new) | `build_parca_sim_workflow` + `v2ecoli-nextflow` CLI | v2e |

---

## Phase 1 — build-document runner + core provisioning (closes blocker #1)

### Task 1 [pbg]: `provision_core` + `run_step --provision`

**Files:** Create `process_bigraph/provision.py`; modify `process_bigraph/run_step.py`; Test `process_bigraph/tests/test_provision.py`.

**Interfaces — Produces:** `provision_core(core, providers) -> core` where each provider is `"module:attr"` (str) or `(module, attr, args, kwargs)` (tuple); imports the module, calls `fn(core, *args, **kwargs)`, and if it returns non-None uses that as the new core. `run_step` gains `--provision MOD:ATTR` (repeatable) applied before instantiating the Step.

- [ ] **Step 1: Write the failing test**

```python
# process_bigraph/tests/test_provision.py
from process_bigraph import allocate_core
from process_bigraph.provision import provision_core

# a module-level provider so "module:attr" import resolves it
_MARK = {}
def mark_core(core):
    _MARK['called'] = True
    core._provisioned = True
    return core

def test_provision_core_applies_string_provider():
    core = allocate_core()
    out = provision_core(core, ['process_bigraph.tests.test_provision:mark_core'])
    assert getattr(out, '_provisioned', False) is True
    assert _MARK.get('called') is True

def test_provision_core_noop_on_empty():
    core = allocate_core()
    assert provision_core(core, []) is core
```

- [ ] **Step 2: Run — expect FAIL** `ModuleNotFoundError: process_bigraph.provision`.
  `… -m pytest process_bigraph/tests/test_provision.py -v`

- [ ] **Step 3: Implement `provision.py`**

```python
# process_bigraph/provision.py
"""Apply core-provisioning hooks to a freshly allocated core.

A provider is a "module:attr" string or a (module, attr, args, kwargs) tuple
naming a callable ``fn(core, *args, **kwargs) -> core | None``. Shared by
run_step, run_composite, and protocols/ray (same contract, two transports).
"""
from __future__ import annotations
import importlib
import sys
from typing import Any, Iterable


def _parse(provider):
    if isinstance(provider, str):
        if ':' not in provider:
            raise ValueError(f"provider must be 'module:attr', got {provider!r}")
        mod, attr = provider.split(':', 1)
        return mod, attr, (), {}
    mod, attr = provider[0], provider[1]
    args = provider[2] if len(provider) > 2 else ()
    kwargs = provider[3] if len(provider) > 3 else {}
    return mod, attr, tuple(args), dict(kwargs)


def provision_core(core: Any, providers: Iterable) -> Any:
    for provider in providers or []:
        mod_name, attr, args, kwargs = _parse(provider)
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr)
            result = fn(core, *args, **kwargs)
            if result is not None:
                core = result
        except Exception as e:
            sys.stderr.write(f'[provision] {mod_name}:{attr} failed: '
                             f'{type(e).__name__}: {e}\n')
            raise
    return core
```

Then in `run_step.py`: add `p.add_argument('--provision', dest='provision', action='append', default=[])`, thread it into `run_step(...)`, and after `core = allocate_core()` call `core = provision_core(core, provision)` (import at top: `from process_bigraph.provision import provision_core`).

- [ ] **Step 4: Run — expect PASS.** `… -m pytest process_bigraph/tests/test_provision.py -v`
- [ ] **Step 5: Commit** (`feat(pbg): provision_core hook + run_step --provision`).

### Task 2 [pbg]: `run_composite --build` (recipe mode) + `--set` + `--provision`

**Files:** modify `process_bigraph/run_composite.py`; Test `process_bigraph/tests/test_run_composite_build.py`.

**Interfaces — Consumes:** `provision_core` (T1). **Produces:** `run_composite(document_path=None, *, build_path=None, steps, sets=None, artifacts=None, provision=None, initial_state=None, out_paths=None, state_out_path=None)`. Build doc schema: `{"build": {"generator": <name>, "overrides": {...}, "provision": ["mod:attr", ...], "import": ["module", ...]}, "run": {"steps": N}}`. Build mode: import each `build.import` module (registers generators via `@composite_generator`), resolve the entry by name in `_REGISTRY`, `core = allocate_core()`, `provision_core(core, build.provision + cli_provision)`, `apply_core_extensions(entry, core)`, merge `overrides` with `--set KEY=JSONVAL`, `doc = build_generator(entry, overrides, core)`, `comp = Composite(doc, core)`, `comp.run(steps)`, then existing `--out`/`--state-out`. `--document` and `--build` are mutually exclusive.

- [ ] **Step 1: Write the failing test** (toy generator — no v2ecoli)

```python
# process_bigraph/tests/test_run_composite_build.py
import json
from process_bigraph.composite import Process
from process_bigraph.composite_generator import composite_generator

class _Ramp(Process):
    config_schema = {'rate': 'float'}
    def inputs(self):  return {'level': 'float'}
    def outputs(self): return {'level': 'float'}
    def update(self, state, interval):
        return {'level': self.config['rate'] * interval}

# provider registers the process class on the task's fresh core
def provision_ramp(core):
    core.register_link('_Ramp', _Ramp)
    return core

@composite_generator(name='ramp_toy', core_extensions=[provision_ramp])
def ramp_toy(rate=2.0, start=1.0):
    return {'state': {
        'level': start,
        'ramp': {'_type': 'process', 'address': 'local:_Ramp', 'config': {'rate': rate},
                 'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}}}}

def test_run_composite_build_uses_generator_and_core_extensions(tmp_path):
    build = tmp_path / 'build.json'
    build.write_text(json.dumps({
        'build': {'generator': 'ramp_toy',
                  'import': ['process_bigraph.tests.test_run_composite_build'],
                  'overrides': {'rate': 3.0}, 'provision': []},
        'run': {'steps': 4}}))
    out = tmp_path / 'final.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(build), steps=4.0, state_out_path=str(out))
    final = json.loads(out.read_text())
    assert float(final['state']['level']) > 1.0   # rate*4 accumulated

def test_run_composite_build_set_override(tmp_path):
    build = tmp_path / 'build.json'
    build.write_text(json.dumps({
        'build': {'generator': 'ramp_toy',
                  'import': ['process_bigraph.tests.test_run_composite_build'],
                  'overrides': {}, 'provision': []},
        'run': {'steps': 0}}))
    out = tmp_path / 'final.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(build), steps=0.0,
                  sets={'start': 41.0}, state_out_path=str(out))
    final = json.loads(out.read_text())
    assert float(final['state']['level']) == 41.0
```

- [ ] **Step 2: Run — expect FAIL** (`run_composite()` has no `build_path` kwarg).
- [ ] **Step 3: Implement** the build branch in `run_composite` (add helper `_build_from_recipe(build_doc, sets, artifacts, provision) -> Composite` using the verified APIs above; keep the existing `--document` path). Add CLI: `--build`, `--set KEY=JSONVAL` (parse value with `json.loads`, fall back to raw string), `--provision`, `--artifact PORT=REF.json` (parse now, consumed in Phase 2). Raise if both `--document` and `--build` given.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** (`feat(pbg): run_composite --build recipe mode (generator + core_extensions)`).

### Task 3 [v2e]: split `build_core` → `register_ecoli_core`; declare `core_extensions`; `emitter_out_dir`

**Files:** modify `v2ecoli/core.py`, `v2ecoli/__init__.py`, `v2ecoli/composites/ecoli_baseline.py`; Test `v2ecoli/tests/test_nextflow_provision.py`.

**Interfaces — Produces:** `v2ecoli.core.register_ecoli_core(core) -> core` (all post-`allocate_core` registration `build_core` did); `build_core()` becomes `register_ecoli_core(allocate_core())` (behavior-preserving). `v2ecoli.register_types = register_ecoli_core`. `ecoli_baseline` generator declares `core_extensions=[register_ecoli_core]` and a new `emitter_out_dir` parameter (default `''`).

- [ ] **Step 1: Write the failing test** — the pbg-native path must build+run the baseline on a bare core provisioned only via `core_extensions` (mirrors P0.1 but through the generic runner):

```python
# v2ecoli/tests/test_nextflow_provision.py
import os, pytest
from process_bigraph import allocate_core
from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
from process_bigraph import Composite

CACHE = '/Users/eranagmon/code/v2ecoli/out/cache'

@pytest.mark.skipif(not os.path.isdir(CACHE), reason='no ParCa cache')
def test_baseline_builds_on_bare_core_via_core_extensions():
    entry = next(e for e in _REGISTRY.values() if e.name == 'ecoli_baseline')
    assert entry.core_extensions, 'ecoli_baseline must declare core_extensions'
    core = allocate_core()                       # BARE — no build_core()
    core = apply_core_extensions(entry, core)    # provisioning only via declaration
    doc = build_generator(entry, overrides={'seed': 0, 'cache_dir': CACHE}, core=core)
    comp = Composite(doc, core=core)
    comp.run(5.0)
    assert comp.state.get('global_time') == 5.0

def test_register_types_hook_exported():
    import v2ecoli
    assert v2ecoli.register_types is v2ecoli.core.register_ecoli_core
```

- [ ] **Step 2: Run — expect FAIL** (no `core_extensions` on the entry; no `register_types` export).
- [ ] **Step 3: Implement** the split in `core.py` (pure refactor — move the post-allocation body of `build_core` into `register_ecoli_core(core)`; `build_core` calls it); export `register_types` in `__init__.py`; add `core_extensions=[register_ecoli_core]` to the `@composite_generator(...)` on `ecoli_baseline` and an `emitter_out_dir` parameter passed to the emitter config (default `''` = existing behavior).
- [ ] **Step 4: Run — expect PASS.** (`[v2e]` prefix; cwd canonical so `out/cache` resolves.)
- [ ] **Step 5: Commit** on `nextflow-parca` (`refactor(v2e): register_ecoli_core + core_extensions + emitter_out_dir`).

### Task 4 [pbg]: refactor `ray.py` onto `provision_core`

**Files:** modify `process_bigraph/protocols/ray.py`; Test: extend `process_bigraph/tests/test_provision.py`.

**Interfaces — Consumes:** `provision_core` (T1). `_apply_type_providers(core, providers)` becomes a thin adapter that maps ray's `(module, attr, args, kwargs)` provider tuples through `provision_core` (preserving ray's stderr logging semantics). No behavior change for ray callers.

- [ ] **Step 1: Write the failing test** — assert `ray._apply_type_providers` and `provision_core` produce identical effects for a shared provider list (a tuple-form provider that tags the core). (Do not require `ray` installed — call `_apply_type_providers` directly; it doesn't need a live actor.)
- [ ] **Step 2: Run — expect FAIL** (only if you assert delegation via a shared spy; otherwise write the test to pin the unified behavior).
- [ ] **Step 3: Implement** the delegation; keep the legacy 2-tuple back-compat and stderr message.
- [ ] **Step 4: Run — expect PASS**, and run the existing ray protocol tests to confirm no regression.
- [ ] **Step 5: Commit** (`refactor(pbg): ray type-providers via shared provision_core`).

---

## Phase 2 — sim_data artifact producer + consumption (formalizes handoff as typed refs)

### Task 5 [v2e]: `ParcaBundleStep` (fixture mode) → `ArtifactRef`

**Files:** Create `v2ecoli/steps/parca_bundle.py`; Test `v2ecoli/tests/test_parca_bundle.py`.

**Interfaces — Produces:** `class ParcaBundleStep(Step)` with `config_schema = {'mode': 'string', 'cpus': 'integer', 'condition': 'maybe[string]', 'bundle_dir': 'string'}`, `outputs() -> {'sim_data': {'_type': 'string', '_is_file': True}}`, and `update(state) -> {'sim_data': <ArtifactRef.to_dict() json path or dict>}`. Fixture mode reuses `models/parca/parca_state.pkl.gz` / the existing cache: write the bundle via `save_sim_input`, hash the bundle files, `write_fingerprint`, and return the `ArtifactRef` (kind `sim_data`, `store=bundle_dir`, `hash=<sha>`). Do NOT run full ParCa in the test — fixture/pre-cached only.

- [ ] **Step 1: Write the failing test** — instantiate `ParcaBundleStep({'mode': 'fixture', 'bundle_dir': str(tmp_path)})`, call `invoke({}).update`, assert the output is an `ArtifactRef`-shaped dict with `kind == 'sim_data'`, a non-empty `hash`, and a `store` that contains `sim_data_cache.dill`.
- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: v2ecoli.steps.parca_bundle`).
- [ ] **Step 3: Implement** `ParcaBundleStep`. Fixture path: if a valid cache already exists (reuse `load_cache_bundle` semantics) copy/point `bundle_dir` at it and `save_sim_input`; compute `hash = sha256(sim_data_cache.dill) ^ sha256(initial_state.json)` recorded per-file in `context`; `from process_bigraph.artifacts import ArtifactRef, SIM_DATA, write_fingerprint`.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** on `nextflow-parca` (`feat(v2e): ParcaBundleStep emits a sim_data ArtifactRef`).

### Task 6 [pbg]: `run_composite --artifact` consumption + fingerprint check

**Files:** modify `process_bigraph/run_composite.py`; Test `process_bigraph/tests/test_run_composite_build.py` (append).

**Interfaces — Consumes:** build doc gains `"artifacts": {<override_key>: {"kind": ..., "map": "store"}}`. `--artifact PORT=REF.json` (from T2) loads the ref file, `ArtifactRef.coerce`, and for each `artifacts[key]` with `map == 'store'` injects `overrides[key] = ref.store` before `build_generator`. Optionally `check_fingerprint` (warn, not fail).

- [ ] **Step 1: Write the failing test** (toy): a build doc whose generator takes a `cache_dir` param and echoes it into state; provide `--artifact cache_dir=ref.json` where `ref.json = {"kind":"sim_data","store":"/some/path","hash":"h"}` and `artifacts.cache_dir.map='store'`; assert the built composite received `cache_dir == "/some/path"`.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** artifact injection in the build branch.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** (`feat(pbg): run_composite --artifact injects ArtifactRef.store into overrides`).

---

## Phase 3 — the DAG end-to-end (closes blocker #3 for the seed axis) — MILESTONE

### Task 7 [pbg]: `CompositeTask` Step (native loop + Nextflow emission)

**Files:** Create `process_bigraph/tasks.py`; Test `process_bigraph/tests/test_composite_task.py`.

**Interfaces — Produces:** `class CompositeTask(Step)` with `config_schema = {'generator': 'string', 'overrides': 'node', 'artifact_params': 'map[string]', 'scatter_param': 'maybe[string]', 'steps': 'float', 'provision': 'list[string]', 'import': 'list[string]'}`. `inputs()`: one `{_type:'string', _is_file:True}` port per `artifact_params` key + a scatter port `{'_type': 'list[integer]', '_scatter': True}` named by `scatter_param`. `outputs() -> {'results': 'path'}`. Native `update(state)`: for each scatter value, build via the same `_build_from_recipe` used by run_composite (reuse it — extract to a shared helper), run `steps`, collect result paths. `nextflow_script()` emits `python -m process_bigraph.run_composite --build <name>_build.json --steps <steps> --set <scatter>=${<scatter>} --artifact <port>=${<port>}`. `nextflow_port_decls = {'results': 'path "results"'}`.

- [ ] **Step 1: Write the failing tests**: (a) native — a `CompositeTask` over the `ramp_toy` generator with `scatter_param='start'`, `update({'starts':[1.0,2.0]})` returns two results; (b) emission — `nextflow_script()` contains `run_composite --build` and `--set start=${start}`.
- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: process_bigraph.tasks`).
- [ ] **Step 3: Implement** `CompositeTask` (reuse the `_build_from_recipe` helper from T2 — refactor it out of `run_composite` into a shared module, e.g. `process_bigraph/build_recipe.py`, imported by both).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** (`feat(pbg): CompositeTask — a scattered whole-Composite task`).

### Task 8 [pbg]: renderer `_scatter` + queue-`.first()` support; `deploy(publish_dir, build_documents, code_version)`

**Files:** modify `process_bigraph/nextflow.py`, `process_bigraph/nextflow_deploy.py`; Test `process_bigraph/tests/test_nextflow_render.py` (append), `test_nextflow_deploy.py` (append).

**Interfaces — Produces:** renderer: a port annotated `_scatter: True` emits its channel as `Channel.of(<state list elements>)` (read from composite state at the wire path); a process consuming a scatter input wraps its *other* single-producer inputs in `.first()` **only when the producer emits a queue channel** (a single-output producer is already a value channel — skip `.first()` to avoid the no-op, per P0.2). `deploy(...)` gains `publish_dir` (→ `params.publishDir` + `publishDir` directives), `build_documents=True` (write `<outdir>/<step>_build.json` for each `CompositeTask`, sourced from its config), and `code_version` (dict stamped into every build doc; default auto from `git rev-parse` + `importlib.metadata`).

- [ ] **Step 1: Write the failing tests**: (a) render a `producer(Step) → CompositeTask(scatter)` composite; assert `Channel.of(` appears for the scatter channel and the CompositeTask process is called with it; (b) `deploy(build_documents=True)` writes a `sims_build.json` whose `code_version` is non-empty.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** the `_scatter`/`.first()` rules in `nextflow.py` and the `deploy` extensions.
- [ ] **Step 4: Run — expect PASS** (render + deploy suites).
- [ ] **Step 5: Commit** (`feat(pbg): scatter-channel rendering + deploy build-docs/publish_dir/code_version`).

### Task 9 [v2e]: `build_parca_sim_workflow` + `v2ecoli-nextflow` CLI — the milestone

**Files:** Create `v2ecoli/workflow/nextflow.py`; add `[project.scripts] v2ecoli-nextflow` to `pyproject.toml`; Test `v2ecoli/tests/test_nextflow_dag.py`.

**Interfaces — Consumes:** everything above. **Produces:** `build_parca_sim_workflow(*, seeds, parca_mode='fixture', generator='ecoli_baseline', overrides=None, steps=2700, publish_dir='out/nf') -> Composite` — a composite whose state wires a `parca` `ParcaBundleStep` → a `sims` `CompositeTask` (`generator=generator`, `scatter_param='seed'`, `artifact_params={'cache_dir': 'sim_data'}`, `import=['v2ecoli']`, `core_extensions` carried by the generator), with a `seeds` list store. `main()` builds it and calls `process_bigraph.nextflow_deploy.deploy(comp, outdir=..., executor='local', launch=args.launch, publish_dir=publish_dir, build_documents=True)`.

- [ ] **Step 1: Write the failing tests**: (a) unit — `build_parca_sim_workflow(seeds=[0,1], steps=2)` returns a Composite whose `sims` node is a `CompositeTask` with `scatter_param=='seed'`; (b) render — `deploy(comp, launch=False)` writes a `main.nf` containing both `parca` and `sims` processes and a `Channel.of(0, 1)`; (c) **integration, nextflow-gated** — `main(['--seeds','2','--parca-mode','fixture','--outdir',str(tmp),'--launch'])` returns 0 and `tmp/out/nf/sims/seed=0` and `seed=1` partitions exist. Use a SMALL `steps` (e.g. 5) in the integration test, not 2700, to keep it fast.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** `build_parca_sim_workflow`, `main`, and the console-script entry. Keep `steps` overridable so the test runs a handful of ticks.
- [ ] **Step 4: Run — expect PASS** (unit + render always; integration when `nextflow` present — it is).
- [ ] **Step 5: Commit** on `nextflow-parca` (`feat(v2e): v2ecoli-nextflow — ParCa→per-seed baseline DAG`).

---

## Self-Review

**Spec coverage (Phases 1–3 of the design):** provisioning (T1, T3, T4) ✓; build-document reconstruction (T2, T7 helper) ✓; sim_data ArtifactRef producer+consumer (T5, T6) ✓; CompositeTask + scatter rendering + deploy build-docs/code_version (T7, T8) ✓; the ParCa→2-seed milestone (T9) ✓. Phase 4/5 explicitly deferred. Determinism: `code_version` stamping (T8), explicit seeds (T7/T9), fingerprints (T5/T6) — the design's robustness points that live in Phases 1–3.

**Placeholder scan:** no TBD/TODO in steps; each task has concrete test + implementation code or a precise construction recipe against verified APIs.

**Type consistency:** `provision_core(core, providers)` (T1) consumed by T2/T4/T7. `run_composite(..., build_path, sets, artifacts, provision, ...)` (T2) reused by T7 via the extracted `_build_from_recipe`/`build_recipe.py`. `ArtifactRef` dict shape (`kind/hash/store/context/fingerprint`) consistent across T5 (produce) and T6 (consume). `CompositeTask` config keys (`generator/overrides/artifact_params/scatter_param/steps/provision/import`) consistent T7 → T9. `deploy(..., publish_dir, build_documents, code_version)` (T8) called by T9. `register_ecoli_core` (T3) referenced by the generator's `core_extensions` and by T9's `import=['v2ecoli']`.

## Deferred (Phase 4–5, later plan)
Composite-node topo-sort inclusion + document staging (blocker #2); retry/memory-escalation + maxForks; real `--parca-mode fast`; slurm smoke on the mini; determinism audit (env-var warnings, fingerprint round-trip); variant×seed tuple scatter; multi-generation (lineage / daughter-state overlay); analysis stage; fsspec URIs for cloud; entry-point provisioning discovery.
