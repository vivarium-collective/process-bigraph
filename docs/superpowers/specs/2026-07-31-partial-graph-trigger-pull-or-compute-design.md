# Partial graph triggering — a cached study result is a filled `results` site

**Status:** Design (Layer 3.5 of the framework-unification stack — sits on the merged `results` port + templates)
**Date:** 2026-07-31
**Repos:** process-bigraph (the engine-native mechanism), vivarium-workbench (the content-addressed artifact store, already shipped), v2ecoli (ParCa as the root node — the proof)
**Companions:** `2026-07-31-study-investigation-templates-design.md` (fill/prune, the same law), `2026-07-29-framework-unification-design.md` (umbrella)
**Decisions locked (user, 2026-07-31):** engine-native filled site (not a workbench resolver); build after the results-wired vertical slice; ParCa-as-root is the headline example.

---

## 1. The problem

An investigation is a composite whose nodes are studies (umbrella Layer 1: each study is one `SimulationStep` node, its output carried by the emitter's `results` port). Today that graph runs **whole or not at all**. Two things are missing:

1. **Trigger part of the graph** — run *one* study, not the entire investigation.
2. **Reuse an upstream result** — rerun a downstream study that consumes a prior study's output **without rerunning the prior study**.

The load-bearing instance is **ParCa**. ParCa is expensive (`--mode full` ≈ 2.5 min, 51 conditions) and produces `sim_data`, the calibration object nearly every downstream simulation consumes. Today ParCa reuse is an ad-hoc directory (`out/cache`) plus a **worktree symlink dance** — the recurring "build failed: out/cache" failure when a worktree has no symlink to the cache. The user's ask, verbatim: *treat ParCa as the first study in the investigation graph and continue from its saved sim_data without always rerunning ParCa.*

## 2. The idea — no new execution concept

We already have `fill` / `is_ground` / `prune` (bigraph-schema 1.4.4, `process_bigraph/templates.py`) and the `results` port carrying a durable `EmitterResults` handle (merged, PR #160). **A cached study result is simply a filled `results` site.** Nothing new to schedule; the existing filling law decides run-vs-pull:

```
study_B.results:   <open>          → COMPUTE  (run study_B's SimulationStep)
study_A.results:   cached@<hash>   → PULL     (a CachedResults step resolves the
                                               artifact and emits the handle; no sim)
parca.results:     cached@<hash>   → PULL     (sim_data; ParCa never runs)
```

- an **open** results site ⇒ the study is non-ground on that axis ⇒ it **computes**;
- a **filled** results site (bound to a cached artifact reference) ⇒ ground ⇒ it **does not run**; downstream consumes the cached handle exactly as if the study had just produced it.

This is the same trick as gating (`2026-07-31-study-investigation-templates-design.md` §4), applied to the *results* axis instead of the *membership* axis.

## 3. The trigger operation — one function on the document

`trigger(investigation_document, target, *, on_missing='error')` builds the sub-document to run:

1. **Resolve prerequisites.** Walk `inputs.from` edges to collect `target`'s transitive ancestors (the DAG edges are already derived from `inputs.from` — study-pipeline Spec 1).
2. **Fill each prerequisite from cache.** Compute its content-address (§5); if `.pbg/artifacts/<hash>/` exists, **fill** its `results` site with a `CachedResults` reference (pull). If it is missing:
   - `on_missing='error'` (default): raise, naming the uncached prerequisite — *"run `parca` first"*;
   - `on_missing='compute'`: leave that prerequisite's site open too (it will compute, recursively pulling *its* satisfied ancestors).
3. **Leave the target open** (it computes).
4. **Prune non-ancestors** (`prune_open_regions`, reused): everything that is neither the target nor an ancestor is absent from the built document — never constructed, never run.

The result is an ordinary ground-or-computing composite. `Composite(...).run(0.0)` executes exactly the target (and any `on_missing='compute'` ancestors), with every cached ancestor resolved, never re-simulated. "Run one study" and "rerun downstream without upstream" are the *same* call with different `target`s.

## 4. `CachedResults` — the pull step

A new `Step` in `process_bigraph.processes` (sibling to `SimulationStep`), the pull-half of pull-or-compute:

```python
class CachedResults(Step):
    """Resolve a content-addressed study artifact and emit its handle —
    the filled counterpart of SimulationStep. Never runs a simulation."""
    config_schema = {'artifact_ref': 'quote'}   # {kind, hash, store, context, fingerprint}
    def inputs(self):  return {}
    def outputs(self): return {'results': 'node'}
    def update(self, state):
        return {'results': ArtifactResults.from_ref(self.config['artifact_ref'])}
```

- It has **no inputs and produces `results` from `update()`** — indistinguishable from `SimulationStep`, which is the counterpart to match. *(Build correction, PR #164: an earlier draft produced at `finalize()` by analogy with an **emitter**. That is wrong — an emitter finalizes because its results only exist post-run, but a study node is a producer **in the step DAG** and `SimulationStep` returns the handle from `update()`. A `CachedResults` that produced only at finalize left its consumers unordered — they fired first and read an empty store. Indistinguishability means matching `SimulationStep` (the step-DAG producer), not matching the emitter. `finalize()` is retained as a secondary route.)*
- `trigger` swaps a prerequisite's `SimulationStep` node for a `CachedResults` node bound to the resolved `artifact_ref`. Same interface (`{results}` out), so the wiring is unchanged.

## 5. Artifact references — the honest generalization ParCa forces

A simulation's `results` is a **zarr trajectory** (`EmitterResults` → `emitter.query()` → DataTree). **`sim_data` is not a trajectory** — it is a build-time calibration object. So a study-node output must generalize beyond "emitter handle" to a **typed content-addressed artifact reference**:

```
ArtifactRef = { kind: 'trajectory' | 'sim_data' | 'figures' | …,
                hash: <content-address>,
                store: <path under .pbg/artifacts/<hash>/>,
                context: {…},
                fingerprint: <for §6 determinism-gating; must survive a store round-trip> }
```

> **Naming (build correction, PR #164):** the registered kind sorts are `results_<kind>`
> (`results_trajectory`, `results_sim_data`), **not** `results:<kind>`. A `_sort` value is
> parsed by the type grammar, where `a:b` is the named-parameter form — a colon in the name
> is swallowed and the registered sort is never found (it silently falls through to
> `core.check`). This is the *third* appearance of the colon-in-a-name trap (after `local:edge`
> and the address work): **any user-facing name that reaches `_sort` or a type position must
> avoid `:`.**

- `kind='trajectory'` resolves via the emitter's `RunReader`/`query()` (what the slice proves).
- `kind='sim_data'` resolves by loading the calibration object from the store (a pickle/zarr/parquet ParCa already writes).
- `ArtifactResults` (the resolved value the `results` node carries) is `EmitterResults` widened to *any* kind — `EmitterResults` becomes the `kind='trajectory'` case. `.resolve()` dispatches on `kind`.

This is a *cleaner* abstraction — "a study produces an artifact; some artifacts are trajectories, one is sim_data" — but it is a real design point, not free: the `results` sort must admit these kinds, and `admits` must type-check a filler's kind against what a consumer needs.

## 6. Content-addressing & determinism (reuse the shipped spine)

**Do not invent hashing** — study-pipeline Spec 1 already defines it (`lib/artifacts/{hashing,store,pipeline}.py`, `resolve_study`):

```
artifact_id = H(composite_id + canonical(config) + sorted(input_ids) + workspace_git_commit)
store        = .pbg/artifacts/<hash>/
```

- **ParCa's address** = `H(parca_config + raw_data_id + workspace_git_commit)` — deterministic and **worktree-independent** (no symlink; any worktree at the same commit resolves the same hash), retiring the `out/cache` symlink dance.
- **Soundness caveat.** A hash-of-inputs cache is only valid if the producer is **deterministic given its inputs**. If ParCa's calibration is stochastic, a hit could serve a result that would not reproduce. Reuse the reproducible-rerun spine's `result_fingerprint` + `provenance_status='nondeterministic'`: the artifact carries a fingerprint; a recompute that diverges from the cached fingerprint at the same address is flagged, not silently served. **ParCa's determinism must be confirmed when building** (seed-controlled ⇒ include seed in the address; genuinely nondeterministic ⇒ address is advisory + fingerprint-gated).

The decision (cache-hit ⇒ fill, miss ⇒ compute) lives **in the graph** (engine-native), not in a pre-build workbench resolver — so the same `.pbg/artifacts/` store the workbench pipeline writes is read by the composite at construction time. One store, two writers already (spine's constraint); this adds one reader.

## 7. Worked example — the comparison-harness investigation (the *real* one)

This mechanism's flagship is not a toy: `workspace/investigations/v2ecoli-vecoli-comparison/investigation.yaml` **already declares exactly this structure** — it is run today by an *imperative loop* (`scripts/_compare/runner.py:run_investigation`), and this spec is what makes it a composite. The conversion is a re-expression, not a redesign, because the data model is already right:

- **`members: [acetate, basal, metabolism_redux, no_oxygen, parca, statistical, succinate, with_aa]`** — the list of per-study configs, *one study per condition*. Each is a top-level `workspace/studies/<name>/study.yaml` with a scalar `condition: <media>`. → the template's **cardinality region** of comparison-study sites, one filled per member.
- **ParCa reuse is already a declared edge:** every condition study carries `inputs: [{artifact: sim_data, from: parca}]`, and `parca` is a member (the root). → the **pulled root**: `trigger`/build fills each condition study's `sim_data` input from the ParCa artifact; ParCa computes once (or is pulled), never per condition.
- **The vEcoli repo pointer** lives in the shared context (`vecoli: {repo: https://github.com/CovertLab/vEcoli, commit: ''}`). → a **`git:` address site** `git:CovertLab/vEcoli@<commit>`, filled once, shared by all studies — replacing today's env var (`V2E_VECOLI_DIR`) + *empty* commit pin (unpinned, and `build_comparison_caches.sh` even defaults to a different `vEcoli-upstream` checkout). The address site makes the compared implementation **pinnable and single-sourced**.

**Two ParCa roots, not one** (a real subtlety, do not model it as one): v2ecoli and vEcoli need *separate* fits — divergent formats, `out/cache_full` vs `out/compare_harness/vecoli_parca`, and v2 cannot load the upstream cache (`build_comparison_caches.sh`). So the composite has **two** content-addressed ParCa-root nodes, each pulled once, each feeding its engine's side of every condition study.

```
comparison investigation (composite):
  parca_v2      : CachedResults<sim_data>   ← pulled once (v2ecoli fit)
  parca_vecoli  : CachedResults<sim_data>   ← pulled once (vEcoli fit, via git: address)
  vecoli.address: <site: git:CovertLab/vEcoli@commit>          (shared)
  studies[ basal, succinate, acetate, no_oxygen, with_aa,      (cardinality region:
           metabolism_redux ]:                                  one per condition)
     each = a comparison study consuming (parca_v2, parca_vecoli) sim_data,
            emitting a matched-timepoint report card (already a pbg Step)
  statistical   : cross-condition gate (Welch-t) consuming the per-condition verdicts
```

**Reuse, don't rebuild the compare side:** the compare + report-card entities are *already* pbg `Step`s (`scripts/_compare/report_cards/`, `@as_step`, `REPORT_CARD_STEPS`; matched-timepoint 5%/10% banding in `comparison_report_card.py`). They drop into the study nodes as flush entities unchanged.

**Fixes a live breakage for free:** the imperative modular runner is *stale* post-migration — `study_spec.load_investigation` reads `data.get("studies", [])` and expects `inv_dir/studies/<name>/`, but the migration moved to `members:` + top-level `workspace/studies/`, so `run_investigation` raises "has no studies" today. A composite that reads `members:` + top-level studies natively repairs this by construction.

**⚠ Decision for the build — per-study vs global compare config.** Per-study config today is minimal (`condition`, `comparison.{seeds,generations,max_steps_per_gen}`, `from_vecoli_config`, `cards`). **Media is derived from `condition`; tolerances + observables are GLOBAL constants** (`comparison_report_card.py: TOL=0.05, TOL2=0.10, OBSERVABLES`). If per-study tolerances/observables are wanted (e.g. looser bands for succinate), the comparison-study template must lift them from global constants to **per-study value sites**; otherwise they stay global. **User decision at build time** — default: keep global (matches today), lift only if asked.

### 7.1 ParCa roots are pull-*or-compute* — and vEcoli's ParCa computes in vEcoli's own venv

§4's `CachedResults` is only the *pull* half. For ParCa to be **deterministically available within the investigation** (user requirement, 2026-07-31), each ParCa root must be **pull-or-compute**: cache-hit → pull the handle; miss (or a stale/incompatible artifact) → **compute it, then cache** — the ordinary open-vs-filled site distinction of §2, applied to a root that has a *compute* recipe.

- **`parca_v2`** computes via v2ecoli's own ParCa (a `SimulationStep`-style compute node running `parca_run.py --mode full`), content-addressed by `(parca_config, V2PARCA_N_SEEDS, workspace_commit)` — note `V2PARCA_N_SEEDS` **must** be in the address (it silently changes the fit and is in no manifest).
- **`parca_vecoli`** computes **inside the fetched vEcoli `git:` venv** — a second `git:` entrypoint `#vecoli_parca:build_sim_data` (`VEcoliParcaBuild`, `outputs = {'artifact_ref': 'quote'}`) that writes `simData.cPickle` and returns its artifact reference. Content-addressed by `(vEcoli_commit, parca_config)` — and, critically, the artifact is **valid only for that resolved env**. *(Build correction: it must call vEcoli's `run_parca` **directly**, NOT `runscripts/parca.py:main` — that `main` is a Nextflow wrapper that **skips the fit** whenever the config carries a `sim_data_path` (vEcoli's `configs/default.json` sets one) and sets `cache_dir` itself. The entrypoint calls `run_parca` and supplies `cache_dir`. Also: record the build env via `sys.prefix`, not `$VIRTUAL_ENV` — the worker inherits the caller's env, so `$VIRTUAL_ENV` names the wrong venv and would defeat the whole point. ParCa returns a SHA256 of the pickled **result** bytes — a fingerprint of the output, not the inputs — so a nondeterministic refit at the same address is detectable.)*

**Why compute-in-venv is load-bearing, not incidental.** The proven blocker to a full live comparison tick was **not** the protocol — it was a **cross-version pickle incompatibility**: the on-disk `vecoli_parca/simData.cPickle` (built 2026-07-21) unpickles with `ModuleNotFoundError: scipy._lib.array_api_compat` in the freshly-resolved vEcoli venv (newer scipy). Regenerating vEcoli's ParCa **inside the same per-SHA venv that will read it** makes the pickle written by the exact scipy that reads it — the incompatibility **cannot arise by construction**. Content-addressing over `(vEcoli_commit)` means a repo/env change invalidates the artifact → recompute in the current venv → always compatible. This is the structural cure for the `#431`-class stale-pickle failures, not a scipy-pin workaround.

**Mechanism note.** A `git:` "compute" entrypoint generalizes the adapter: the same subprocess+per-SHA-venv boundary that resolves `interface()` and ticks a process can also **run a build step and return an artifact reference** (the store path is the boundary — the bulk `simData.cPickle` stays in the venv's checkout, the handle crosses the RPC). `admits`/conformance is unchanged (checked before the run); the compute is gated by the content-address exactly like any pull-or-compute node.

## 8. What runs today vs what this adds

| Piece | State |
|---|---|
| `fill` / `is_ground` / `prune` | ✅ shipped (bigraph-schema 1.4.4, `templates.py`) |
| `results` port + `EmitterResults` durable handle | ✅ merged (#160) |
| `SimulationStep` (compute half) | ✅ merged (#160) |
| content-addressed store + `resolve_study` pull-or-compute | ✅ implemented (workbench Spec 1) — *verify merge state* |
| DAG edges from `inputs.from` | ✅ (Spec 1) |
| **`ArtifactRef` / `ArtifactResults`** (widen `EmitterResults` to kinds) | ❌ this spec |
| **`CachedResults` step** (pull half) | ❌ this spec |
| **`trigger(document, target)`** (fill-upstream-from-cache + prune) | ❌ this spec |
| **ParCa as a study node** with a `sim_data` artifact | ❌ this spec (the proof) |

## 9. Contracts

1. **Indistinguishability.** A downstream consumer sees the *same* `results` handle whether the upstream ran or was pulled — `CachedResults` and `SimulationStep` share the `{results: node}` face; a flush/analysis step is agnostic to which produced it.
2. **`trigger` is filling.** Triggering a study = fill its cached ancestors' results sites + leave it open + prune non-ancestors. No scheduler flag, no "skip" list — a pulled study is *ground*, an absent study is *pruned*. Same law as gating.
3. **Cache soundness.** A pull is served only when the content-address matches *and* (for a nondeterministic producer) the fingerprint matches; a divergent recompute is flagged `nondeterministic`, never silently served.
4. **Worktree independence.** The address is a function of `(config, input_ids, workspace_commit)` — never of a filesystem path — so two worktrees at the same commit share cache hits with no symlink.
5. **Missing prerequisite is explicit.** `on_missing='error'` names the uncached ancestor; it never silently recomputes a "prior" study the user meant to reuse.

## 10. Tests

- **Trigger one study.** A 3-node chain `parca → A → B`; `trigger(doc, 'A')` builds a document whose `process_paths` contains only `A`'s sim (parca pulled, B pruned); running it never invokes ParCa (spy asserts 0 ParCa calls) and A consumes the cached sim_data.
- **Rerun downstream, reuse upstream.** With `parca` and `A` cached, `trigger(doc, 'B')` runs only `B`; both `parca` and `A` are pulled; B's result differs when B's own config changes but neither ancestor recomputes.
- **Missing prerequisite.** `trigger(doc, 'B')` with no cached `parca` raises, naming `parca` (default), or computes it under `on_missing='compute'`.
- **Indistinguishability.** A flush/report-card step produces the same verdict whether its upstream study ran or was pulled from an identical cached artifact.
- **`sim_data` kind.** A `CachedResults` with `kind='sim_data'` resolves the calibration object; a `kind='trajectory'` resolves the DataTree — both via one `ArtifactResults.resolve()` dispatch.
- **Content-address is worktree-independent.** The same `(config, commit)` yields the same hash from two different worktree paths.
- **Nondeterminism flagged.** A stochastic producer whose recompute diverges from the cached fingerprint surfaces `provenance_status='nondeterministic'` rather than serving the stale hit.

## 11. Sequencing & out of scope

**Sequencing:** (1) `ArtifactRef`/`ArtifactResults` (widen `EmitterResults` to kinds); (2) `CachedResults` step; (3) `trigger()` on the investigation document (reusing `prune_open_regions` + the Spec-1 address); (4) **ParCa as a study node** with a `sim_data` artifact + determinism confirmation — the proof; (5) workbench "run this study / continue from here" affordance over `trigger`. Depends on: the merged `results` port (#160), templates.py (#159), and the workbench artifact store (Spec 1 — **verify merge state first**).

**Out of scope v1:** cache eviction/GC (log sizes, don't unbounded-grow); cross-machine cache sharing (address includes `platform`-adjacent commit; coarsen later); migrating the ad-hoc `out/cache` contents (new runs populate `.pbg/artifacts/`; the old cache is abandoned, not imported); the agent "legal system" that decides *which* study to trigger (rerun-spine Spine B, deferred).

**Open for the user (build-time):** ParCa's determinism (seed-controlled vs stochastic) — sets whether seed enters the address or only the fingerprint.
