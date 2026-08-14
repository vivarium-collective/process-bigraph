# ParCa-as-a-node Nextflow DAG for process-bigraph — design (Fable review)

**Status:** SUBSUMED by `2026-08-14-workflow-execution-architecture-design.md` — Nextflow is now one
*backend* behind a `WorkflowBackend` interface, not the mechanism. The ParCa/rebuild/ArtifactRef/
CompositeTask substance below is retained and remains accurate; the "Nextflow DAG is the deliverable"
framing is superseded (the new default milestone runs under a pure-Python `LocalRunner`).

Originally: Architecture design (Fable review), pending user approval → writing-plans.
**Date:** 2026-08-14
**Extends:** branch `nextflow-deploy` (append "Part D" to the base nextflow spec).
**Companion spec:** `2026-08-13-nextflow-step-network-deploy-design.md`

## Goal

A general, deterministic process-bigraph capability to run, on Nextflow, a multi-stage DAG:
**ParCa runs as an upstream node producing a cached `sim_data` artifact**, and **downstream E.coli
baseline sims fan out (per seed; later variant×seed×generation) consuming it**, each rebuilding the
baseline from `sim_data` + a seed and running to completion — mirroring vEcoli's `runParca → sim`
DAG. General mechanism in process-bigraph; v2ecoli contributes only declarations + one producer Step.

## Central insight: rebuild, don't rehydrate

A Nextflow sim task never receives a serialized WCM composite — it receives a **recipe**:
`(generator id, overrides, sim_data ArtifactRef, code version)`, and rebuilds the baseline in-task
via the existing generator registry. Mirrors vEcoli (`ecoli_master_sim.py --config --sim_data_path
--seed` is a factory call, never a resumed pickle). Dissolves the fresh-core blocker: the generator
declaration carries its own core provisioning (`core_extensions`), so the subprocess needs no
v2ecoli glue — only a generic "resolve generator, apply core_extensions, build" runner.

## Existing assets that make this small
- `process_bigraph/artifacts.py` — content-addressed `ArtifactRef` with a `SIM_DATA` kind,
  `artifact_id`, `write_fingerprint`/`check_fingerprint`, `register_artifact_loader`. Reuse as-is.
- `process_bigraph/composite_generator.py` — `GeneratorEntry.core_extensions` + `apply_core_extensions`.
  The provisioning substrate (already used by the dashboard subprocess runner).
- `process_bigraph/nextflow.py` renderer escape hatches — `nextflow_script()`, `nextflow_port_decls`.
- v2ecoli `save_sim_input` / `load_cache_bundle` (`v2ecoli/core.py`) — producer/consumer primitives.

## The DAG as a pbg Composite of plain Steps (supported renderer path)

Nodes are plain Steps → the fully-supported Step-network renderer carries it; the experimental
composite-node branch is NOT on the critical path (still fixed, off-path — §Reuse).

- **`parca` node** — v2ecoli `ParcaBundleStep`: run ParCa (fixture/fast/full), `save_sim_input`,
  hash the bundle, emit a small `ArtifactRef` JSON (`{kind:"sim_data", hash, store}`). The 157 MB
  bundle goes to a publish/store location; only the ref travels through Nextflow.
- **`seeds` store** — plain `[0,1,...]` list in composite state.
- **`sims` node** — pbg `CompositeTask` Step. Config: `{generator, overrides, artifact_params,
  scatter_param, steps}`. Native `update()` loops scatter values, builds+runs each composite (same
  semantics, testable without Nextflow). Rendered → one Nextflow `process` fed by a scattered seed
  channel; script = `run_composite --build sims_build.json --set seed=${seed}
  --artifact cache_dir=${sim_data} --steps 2700`.
- (later) **`analysis` node** — plumbing `Collect`/`groupTuple` + analysis Step.

Generated `main.nf` shape:
```
process parca { publishDir "…/parca"; output: path "sim_data.artifact.json";
    script: run_step --class v2ecoli.steps.parca_bundle.ParcaBundleStep --out sim_data=… }
process sims { tag "seed=${seed}"; input: val seed; path sim_data; output: path "results";
    script: run_composite --build sims_build.json --steps 2700 --set seed=${seed} --artifact cache_dir=${sim_data} }
workflow { ch_sim_data = parca(); ch_seed = Channel.of(0,1); ch_results = sims(ch_seed, ch_sim_data.first()) }
```
Nextflow gives fan-out (queue channel → N tasks) and caching (`-resume` keys on staged-input content
+ vals; the ref file contains the bundle hash → ParCa change invalidates all sims).

## Key decisions (with justification)

**Core provisioning — layered.** Primary: generator-declared `core_extensions`, applied by
`run_composite --build` via `apply_core_extensions` (zero CLI). Escape hatch: `--provision
module:attr` (repeatable) for plain state docs. Both funnel through a new shared
`process_bigraph/provision.py::provision_core(core, providers)`; refactor `protocols/ray.py`'s
`_apply_type_providers` onto it ("same contract, two transports" — Ray pickles providers across the
actor boundary, Nextflow serializes them into the build doc/CLI). v2ecoli: split `build_core()` into
`allocate_core()` + `register_ecoli_core(core)`, declare `core_extensions=[register_ecoli_core]` on
`ecoli_baseline`, and export `v2ecoli.register_types = register_ecoli_core`. NOT entry-points (would
import ~39 packages per task, non-deterministic) — defer as optional Phase-5 convenience.

**Composite reconstruction — build document, not state document.** A state doc is not runnable by
design (v2ecoli strips 157 MB configs; a full doc would need to serialize numpy/pint/RandomState +
live `PartitionedProcess` instances — a research program, not a feature). The build doc
`{build:{generator,overrides,provision}, artifacts, run, code_version}` is small, hashable, and IS
the correct cache key (matches `artifact_id`'s input signature). Mirrors vEcoli (ships
config+sim_data URI+seed, never engine state). State docs keep their role for mother→daughter
initial-state overlays (Phase 5). Generator resolved by direct module import from the build doc —
NOT full `discover_generators()` (imports every bigraph package; unacceptable per-task latency).

Build doc schema:
```json
{"build": {"generator": "ecoli_baseline", "overrides": {"emitter": "parquet", "emitter_out_dir": "./results"}, "provision": []},
 "artifacts": {"cache_dir": {"kind": "sim_data", "map": "store"}},
 "run": {"steps": 2700},
 "code_version": {"v2ecoli": "<git sha>", "process-bigraph": "0.x.y"}}
```
`artifacts.cache_dir.map:"store"` = "override `cache_dir` is filled from the staged ref's `store`".

**sim_data as a first-class artifact — reuse `ArtifactRef` (kind sim_data), transported as a small
staged `path` ref file; payload bundle stays at the store location, never staged.** Not a bare
URI+hash val (a typed ref survives groupTuple/collect, self-describing, cache-invalidates for free via
staged-file hashing). Not path-staging the 157 MB payload (waste; shared-FS reads in place; `store`
becomes an fsspec URI for cloud in Phase 5). Not a new type (`artifacts.py` already has SIM_DATA +
content addressing + fingerprints, lock-stepped with the workbench artifact store).

**Determinism/robustness.** Resume keys: every task input is content-bearing (build doc incl.
`code_version`, artifact ref incl. bundle hash, seed val) → `-resume` behaves like vEcoli. `deploy()`
stamps `code_version` (git sha + package versions) — without it, code edits serve stale caches.
Seeds passed explicitly (`Channel.of(0,1)`), not positional indices. Idempotency: each task emits to
task-local `./results` (new `emitter_out_dir` override), declared as the process output; `publishDir
saveAs seed=<n>/…` moves to the shared hive only on success → retries idempotent, no concurrent-writer
races (sidesteps the shared-`out_dir` hazard). Retry: slurm `errorStrategy` + memory-escalation on
137/140; `maxForks` on local. OOM/shared-cache hazards removed by process isolation (private bundle
per task; no long-lived RAMEmitter driver). Two enforced rules: (a) `CompositeTask` rejects
`ram`/in-memory emitters in build docs; (b) emit only declared paths. Determinism holes to close:
`*_RATE_MULTIPLIER` env vars + `enable_features` globals don't enter the cache key → warn in v1,
migrate to declared params later. ParCa nondeterminism: artifact address is an INPUT hash; fingerprint
records actual output; promise "same inputs → same cached artifact served", not byte-stable sim_data.

**Generality.** `CompositeTask` knows only `generator/overrides/artifact_params/scatter_param` —
nothing E.coli. pbg-uq sweeps, viva-biofilm scans, BioModels ensembles use identical machinery. v2ecoli
residue = declarations + one Step.

**Reuse vs new.** Extend `run_composite` (`--build` xor `--document`, `--set`, `--artifact`,
`--provision`) — not a new runner. Reuse renderer via `CompositeTask.nextflow_script()` +
`nextflow_port_decls`. Two general renderer features: (a) `_scatter:True` port → `Channel.of(<state
list>)`; (b) a scatter-input process's other single-producer inputs wrapped in `.first()`
(queue→value). Fix composite-node blocker #2 properly but OFF the critical path (union topo node-set;
stage `<name>_document.json`). Same branch (`nextflow-deploy`); v2ecoli work in its own worktree.

## API surface

process-bigraph:
- `process_bigraph/provision.py` (new): `provision_core(core, providers) -> core`.
- `run_composite.py` (extended): `run_composite(document_path=None, *, build_path=None, steps,
  sets=None, artifacts=None, provision=None, initial_state=None, out_paths=None, state_out_path=None)`;
  CLI adds `--build BUILD.json`, `--set KEY=JSONVAL`, `--artifact PORT=REF.json`, `--provision MOD:ATTR`.
- `run_step.py` (extended): `--provision MOD:ATTR`.
- `process_bigraph/tasks.py` (new): `class CompositeTask(Step)` — config `{generator, overrides,
  artifact_params: map, scatter_param: maybe, steps, provision: list}`; inputs = artifact ports
  (`{_type:string,_is_file:True}`) + scatter port (`{_type:list[integer],_scatter:True}`); outputs
  `{results:path}`; native `update()`; `nextflow_script()`; `nextflow_port_decls={'results':'path "results"'}`.
- `nextflow.py`: `_scatter:True` → `Channel.of(*state_list)`; scatter-input process → `.first()` on
  other inputs; composite nodes join `_topological_order`; composite-node `input: path document` staged.
- `nextflow_deploy.py`: `deploy(..., publish_dir=None, build_documents=True, code_version=None)`.

v2ecoli:
- `core.py`: `register_ecoli_core(core)` (post-allocation registration); `build_core()=register_ecoli_core(allocate_core())`.
- `__init__.py`: `register_types = register_ecoli_core`.
- `ecoli_baseline.py`: `core_extensions=[register_ecoli_core]` + `emitter_out_dir` param.
- `steps/parca_bundle.py` (new): `ParcaBundleStep` — `{mode, cpus, condition, bundle_dir}` →
  save_sim_input + hash + write_fingerprint + emit ArtifactRef.
- `workflow/nextflow.py` (new) + `v2ecoli-nextflow` CLI: `build_parca_sim_workflow(*, seeds,
  parca_mode='fixture', generator='ecoli_baseline', overrides=None, steps=2700, publish_dir='out/nf')`.

## Phased plan
- **Phase 0 — de-risking probes (½ day, no committed code).** P0.1: fresh-subprocess baseline rebuild
  (`python -c` → `build_composite("ecoli_baseline", seed=0, cache_dir="out/cache")`, run 10 steps) —
  proves provisioning + bundle rebuild with zero driver context. P0.2: toy scatter render (producer →
  scattered consumer, hand-patched channels, local `nextflow run`) — proves `Channel.of + .first()`.
- **Phase 1 — build-doc runner + provisioning (closes #1).** provision.py; `run_composite --build/
  --set/--artifact/--provision`; generator resolution via module import + apply_core_extensions;
  refactor ray.py onto provision_core. v2ecoli: register_ecoli_core split + core_extensions +
  emitter_out_dir. Test: `run_composite --build baseline_build.json --steps 10` in a clean subprocess;
  toy-generator unit tests in pbg.
- **Phase 2 — artifact producer + consumption (formalizes #4 as typed refs).** ParcaBundleStep
  (fixture first). pbg: `--artifact` ref parsing + fingerprint check. Test: run_step→ref→run_composite
  chain as subprocesses.
- **Phase 3 — the DAG end-to-end (closes #3 seed axis). MILESTONE.** CompositeTask; `_scatter`/`.first()`
  renderer; `deploy(publish_dir, build_documents, code_version)`. v2ecoli: build_parca_sim_workflow +
  v2ecoli-nextflow. Test: `v2ecoli-nextflow --seeds 2 --parca-mode fixture --launch` → ParCa(fixture)→
  2-seed baseline local e2e; `seed=0/`,`seed=1/` parquet; `-resume` = zero work; override change → sims
  re-run only.
- **Phase 4 — robustness + composite-node fix (closes #2).** Topo-sort inclusion + doc staging (pure-Step
  nested-composite test); retry/mem-escalation; maxForks; real `--parca-mode fast`; slurm smoke on mini;
  determinism audit (env-var warning, fingerprint round-trip).
- **Phase 5 — scale-out (closes #3 fully).** variant×seed tuple scatter; generations (v1: one task = one
  seed lineage via n_generations sequential; v2: per-generation tasks w/ daughter overlay via
  --initial-state); analysis over collect(); fsspec URIs in ArtifactRef.store; optional entry-point discovery.

## Risks (probe-first)
1. **Fresh-subprocess rebuild fidelity (highest; cheapest probe P0.1)** — pint app-registry / unit_bridge
   import order fragile under a bare interpreter; if it fails, fix in register_ecoli_core.
2. Discovery latency — avoided (direct module import); measure `import v2ecoli` in P0.1; lazy-trim if >10s.
3. Emitter path under Nextflow workdirs — override to `./results`; verify relative handling Phase 1.
4. Channel cardinality (`first()`/scatter) — probe P0.2 before renderer code.
5. Local-executor memory — `maxForks=min(n_seeds, cpus//2)`; ban in-memory emitters.
6. Long ParCa restart-from-zero under retry — mitigate w/ fixture/fast + workflow `-resume`; per-step
   checkpoint-to-publishDir is Phase-5.
7. dill hash instability — weakens cross-run dedup only, never correctness; fingerprint + input-keyed
   resume; accept + document.
8. Hidden state escaping cache key (env multipliers, feature globals) — warn v1, migrate later.
