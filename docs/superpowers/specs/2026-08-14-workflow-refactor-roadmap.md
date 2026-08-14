# Workflow-execution refactor roadmap (Fable review)

**Status:** Reference roadmap distilled from the Fable plan-review + cross-repo streamlining pass
(full report archived in session scratchpad). Companion to
`2026-08-14-workflow-execution-architecture-design.md` and the Phases 1–3 plan.

**Verdict:** the composite-as-DAG + pluggable-backend architecture is right; "tick the composite, nodes
self-execute" is the correct `LocalRunner` model (the native engine already topo-schedules ready steps
at `composite.run(0.0)` and parallelizes layers via `parallel_steps` — a backend-owned scheduler would
duplicate `build_step_network` and belongs only in `NextflowBackend`). The plan is ~85% buildable; the
fixes below are folded into the revised Phases 1–3 plan, and the deferred items are tracked here.

## Must-fix (folded into the Phases 1–3 plan)
- **F1 cache key** — key on `artifact_id(composite_id=generator, config={**overrides, **sets, steps,
  provision}, input_ids=[artifact hashes], commit=code_version)`; reuse `artifacts.artifact_id`
  verbatim. Omitting `code_version`/`steps` = stale cache after a code edit and a 5-step smoke run
  poisoning the 2700-step run.
- **F2 skip semantics** — skip on `artifact_exists(address)` (input hash), NOT `check_fingerprint`
  (output attestation; reserve it for flagging nondeterminism on forced recompute).
- **F3 LocalRunner contract** — try/except → `status='failed'`; `_infer_duration = 0.0` for Step DAGs;
  `_collect_outputs` = `composite.read_bridge()` (T8 wires a bridge).
- **F4 emitter contract** — per-scatter `emitter_out_dir=<workdir>/seed_<val>/results`; refuse
  `local:RAMEmitter` unless opted in (the documented OOM/concurrent-writer hazard, workbench #754).
- **F5 provenance** — `CompositeTask`/`ParcaBundleStep` write a `provenance.json` sidecar
  (`{node, scatter_val: {address, cache_hit, wall_s}}`); `_collect_outputs` aggregates them.
- **F6 bundle hash** — `sha256(concat(sorted per-file digests))`, not XOR (XOR is self-cancelling).
- **S1** subprocess `env=os.environ`. **S2** ThreadPool-around-subprocess, not ProcessPool. **S3**
  `apply_core_extensions` FIRST then `provision_core`. **S5** parallel dispatch T1∥T3∥T4. **S6** drop
  `nextflow_script()` from T6 (→ Phase 4). **pin** `artifact_root` (config, default under `outdir`).
  **T6 must not set** `Step._cache='by_hash'` (a 4th, in-memory cache that would mask the fingerprint test).

## process-bigraph refactor (ranked; ride-along noted)
| # | Item | Value | Effort | When |
|---|---|---|---|---|
| R1 | New modules live in `process_bigraph/workflow/` (`provision, recipe, tasks, backend`); `run_step.py`/`run_composite.py` stay top-level CLI shims over `workflow.recipe`; export `run_workflow` from `__init__` | high | ~0 | plan now |
| R2 | Kill `_scatter:True`; `CompositeTask` scatters via existing `_cardinality:'per_match'` (`composite.py:634-726`) + an `invoke()` override running matches through a bounded ThreadPool; renderer finishes its **dead** `port_cardinality` plumbing (`nextflow.py:320-324`) in Phase 4 | high | S | T6 now / renderer Phase 4 |
| R4 | `protocols/ray.py:150 _apply_type_providers` → shim over `provision_core` (identical tuple contract; also fixes Ray silently ignoring provider return values) | med | ~5 lines | ride T1 |
| R6 | Promote `_topological_order` (`nextflow.py:130`) → `scheduling.topological_order` (public); re-export for back-compat | med | S | ride Phase 4 |
| R5 | Delete legacy `run.py` + its `fire` dep (unused; would be a 4th runner) | low-med | S | anytime |
| — | Document `step_paths`/`node_dependencies`/`parallel_steps`/`read_bridge` as stable public `Composite` API (no code change) | low | ~0 | Phase 4 |

## vivarium-workbench refactor (ranked)
The workbench already invented half this architecture twice, informally: `composite_subprocess`'s
f-string codegen IS a hand-rolled build document (`repr()`-interpolated payload, stdout `@@@RESULTS@@@`
markers, byte-identity source tests), and dead `lib/artifacts/pipeline.py:91` is the proto-`study_to_composite`
+ proto-cache. There are **nine** production "run a composite" paths + two divergent reruns.
| # | Item | Value | When |
|---|---|---|---|
| W1 | `lib/artifacts/hashing.py` → `from process_bigraph.artifacts import canonical, artifact_id` (**lock-step already broken TODAY** — `hashing.py:3-9` keeps the float-narrowing in the dead `default=` position pbg fixed); migrate store via `legacy_artifact_id` | correctness | **BEFORE pbg Phase 2** |
| W2 | Replace `composite_subprocess` codegen (~450 L, `:187-472`) with `run_composite --build` docs; the build doc becomes the sidecar + manifest + cache key | high | Phase 5 (first) |
| W3 | Move study runs onto the detached model (`run_runner` + `spawn_detached` + heartbeat/reconcile); kills 30-min-synchronous-HTTP + stuck-`running` + no-PID | high | Phase 5 |
| W4 | Unify rerun on stored-build-doc replay; delete `cli_runs.rerun`'s self-described lossy variant | high | with W2 |
| W5 | Build `study_to_composite` as the completion of dead `artifacts/pipeline.resolve_study`; then delete/reduce `pipeline.py` | high | Phase 5 |
| W6 | Sweep/seeds variants (`composite_subprocess.py:101` `v2ecoli-workflow` shellout) → `CompositeTask` scatter; gains fingerprint-cached per-seed skip | med-high | Phase 5, after T8 |
| W7 | `SmsApiBackend` behind `WorkflowBackend`; delete legacy `remote_run_jobs.py` after parity | med | Phase 5/6 |
| W8 | Hygiene (anytime): dedupe `_ws_add_to_sys_path` ×3; extract migrate-block helper ×3; `run_study_variant` call `_run_post_run_flush` (kills 50 inlined dup lines); populate `runs_meta.emitter` on generator path; drop dead `detach=` params | med | anytime |
| W9 | Single-source the two `runs_meta` DDLs (`composite_runs.py:80` vs `run_registry.RUNS_META_DDL:32`) | med | anytime |

**Target shape:** keep the workbench's detached-process shell (durability/PID/heartbeat/CSRF/AI-free);
swap the *body* of `run_runner.execute`'s "simulating" phase to `run_workflow(composite, backend=<request.target>)`,
land `RunResult.provenance` in `runs_meta`. Remote/mini become backends, not bespoke pipelines. AI-free is
preserved (pbg dep direction is already correct, `pyproject.toml:20`).

## Explicit YAGNI (recorded so it isn't relitigated)
CWL export beyond `cwltool --validate` until an external-editor round-trip is actually requested;
`RayBackend` while v2ecoli's Ray batch serves; composite-node topo/staging fix (Phase 6 as planned);
async/submit backend APIs; entry-point provisioning discovery; any backend-owned topo scheduler for
`LocalRunner`.

## Phasing of the deferred work
- **Phase 4 (pbg):** `deploy` → `NextflowBackend`; R6 promote `_topological_order`; finish renderer
  `per_match` scatter; forward `--provision` in rendered `run_step` (S4); cross-backend equivalence test.
- **Phase 5 (workbench):** W2 → W3 → W4 → W5 → W6 → W7 (W2 first: independent of `study_to_composite`,
  retires the riskiest engine while behavior pins are fresh).
- **Anytime:** R5, W8, W9.
