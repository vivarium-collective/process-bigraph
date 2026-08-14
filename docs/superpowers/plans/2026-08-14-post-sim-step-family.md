# Post-sim Step family + ResultsStep → viva_superpowers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Make the study **Evaluate** stage a shared, workflow-native substrate: one `ResultsStep`
retrieves emitter output and fans out to pluggable `AnalysisStep` / `VisualizationStep` / `ReportCardStep`
vivarium Steps, run deterministically in the composite. Move the family from `v2ecoli/workflow` to
`viva_superpowers` ("one home"); v2ecoli keeps its concrete subclasses.

**Design:** governing spec `2026-08-14-workflow-execution-architecture-design.md` §"The Evaluate stage".
`sims → ResultsStep(read emitter → ResultsHandle) → {AnalysisStep*, VisualizationStep*, ReportCardStep*} → verdict → bridge`.

**Tech stack:** Python 3.12, process-bigraph, pbg-emitters, pytest. Two repos.

## Global Constraints
- **`[vsp]` — viva-superpowers.** Worktree off `origin/main`:
  ```
  git -C /Users/eranagmon/code/viva-superpowers fetch origin main
  git -C /Users/eranagmon/code/viva-superpowers worktree add /Users/eranagmon/code/viva-superpowers--post-sim-family -b post-sim-family origin/main
  ```
  Test with the repo's own venv/runner (see its CLAUDE.md/AGENTS.md); verify `viva_superpowers.__file__` resolves in the worktree.
- **`[v2e]` — v2ecoli.** Worktree off `origin/main`:
  ```
  git -C /Users/eranagmon/code/v2ecoli worktree add /Users/eranagmon/code/v2ecoli--post-sim-rewire -b post-sim-rewire origin/main
  ```
  Test prefix: `cd /Users/eranagmon/code/v2ecoli && PYTHONPATH=/Users/eranagmon/code/viva-emitters:/Users/eranagmon/code/v2ecoli--post-sim-rewire /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest <test>`. **The `[v2e]` tests need the `[vsp]` change installed** — install the viva-superpowers worktree editable into v2ecoli's venv (or prepend it to PYTHONPATH) so `from viva_superpowers import AnalysisStep` resolves to the worktree.
- Commits end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. `[vsp]`→`post-sim-family`; `[v2e]`→`post-sim-rewire`. Use `git -C <worktree>`; never commit in a canonical checkout.
- **Verified:** `viva_superpowers` deps `process-bigraph>=1.8.2` + `pbg-emitters`. v2ecoli's `V2Step` (`v2ecoli/steps/base.py`) = pbg `Step` + an error-swallowing `invoke()` (returns `SyncUpdate({})` on exception) — the moved bases subclass pbg `Step` and carry that guard, NOT `V2Step`.
- Current sources: `v2ecoli/workflow/post_sim.py` (`Visualization`, `ReportCardStep`, `POST_SIM_REGISTRY`, `REPORT_CARD_REGISTRY`, `register_post_sim`/`iter_post_sim`), `v2ecoli/workflow/analysis.py` (`Analysis` live-conn base, `AnalysisStep` record base, `ANALYSIS_REGISTRY`, `ANALYSIS_SCALES`), `v2ecoli/workflow/report_cards/__init__.py` (`StudyContext`, `write_card`, `prune`, `applicable`, `_sanitize`; imports concrete cards).

## Task 1 [vsp]: shared post-sim Step family (the move)

**Files:** Create `viva_superpowers/post_sim.py`; export from `viva_superpowers/__init__.py`; Test `tests/test_post_sim_family.py`.
**Produces (subclass pbg `Step`, keep the `invoke()` error guard):**
- `AnalysisStep` (record base: reads `results` from state, `analyze(rows)->dict`) + `Analysis` (live base: reads `conn`/`sim_data`, `analyze(**ctx)->dict`) — or a single `AnalysisStep` whose input is the `ResultsHandle` offering both `.records(scale)` and `.conn()` (preferred: one base, see Decisions). `ANALYSIS_REGISTRY`, `ANALYSIS_SCALES`.
- `VisualizationStep` (`render(study)->(html,data)`) + `VISUALIZATION_REGISTRY`.
- `ReportCardStep` (`applies(study)`, `build(study)->(verdict,html)`; verdict conforms to the gating schema `{status: 'pass'|'fail'|'warn', checks: [...], summary: str}`) + `REPORT_CARD_REGISTRY`.
- Unified `POST_SIM_REGISTRY` + `register_post_sim(cls, kind)` + `iter_post_sim(kind=None)`.
- `StudyContext`, `write_card`, `prune`, `applicable`, `_sanitize` (verbatim move).
- `__init_subclass__` on each base auto-registers a named subclass into its registry AND funnels into `POST_SIM_REGISTRY`.

- [ ] Step 1 — failing tests: (a) a toy named `AnalysisStep`/`VisualizationStep`/`ReportCardStep` subclass auto-registers in its registry + `POST_SIM_REGISTRY`; abstract bases (no `name`) do not. (b) `write_card(ctx, name, verdict, html)` writes `<name>.html` + `<name>.verdict.json` (sanitized, `allow_nan=False`). (c) a `ReportCardStep` returning a gating verdict yields `data.status in {pass,fail,warn}`.
- [ ] Step 2 — run, expect FAIL (`ModuleNotFoundError: viva_superpowers.post_sim`).
- [ ] Step 3 — implement `post_sim.py` (move + generalize base to pbg `Step`; drop the `V2Step` import; keep the `invoke()` guard). Export the family from `viva_superpowers/__init__.py`.
- [ ] Step 4 — run, expect PASS.
- [ ] Step 5 — commit on `post-sim-family` (`feat(vsp): shared post-sim Step family (Analysis/Visualization/ReportCard) + StudyContext/write_card`).

## Task 2 [vsp]: `ResultsStep` + `ResultsHandle` + deterministic composite-wiring proof

**Files:** modify `viva_superpowers/post_sim.py` (add `ResultsStep`, `ResultsHandle`); Test `tests/test_results_step.py`.
**Produces:**
- `ResultsHandle` — typed object over a study's emitter output: `.records(scale=None) -> list[dict]` (record slices) and `.conn()` (lazy DuckDB connection) + `.sim_data`/`.paths`. Emitter-agnostic (opens parquet/DuckDB via `pbg-emitters`; zarr where present).
- `ResultsStep(Step)` — reads the study's emitter output location from state (or config) and writes a `ResultsHandle` to the `results` store. `outputs() -> {'results': 'any'}`.
- A **deterministic composite-wiring test**: build a small composite `[fixture results store] → ResultsStep → AnalysisStep + ReportCardStep`, `composite.run(0.0)`, assert both post-sim Steps produced their output from the same handle and the report card's gating verdict landed — proving the family runs deterministically in the step network (topo-ordered after `ResultsStep`).

- [ ] Step 1 — failing tests: (a) `ResultsStep` over a tiny fixture emitter output produces a `ResultsHandle` whose `.records()` returns the expected rows; (b) the composite-wiring test above (deterministic: two runs identical).
- [ ] Step 2 — run, expect FAIL.
- [ ] Step 3 — implement `ResultsStep`/`ResultsHandle` (build the emitter-read on `pbg-emitters`; reference v2ecoli's runner conn-injection for the DuckDB open).
- [ ] Step 4 — run, expect PASS.
- [ ] Step 5 — commit on `post-sim-family` (`feat(vsp): ResultsStep + ResultsHandle — emitter-agnostic Evaluate-stage retrieval`).

## Task 3 [v2e]: rewire v2ecoli onto the shared family

**Files:** modify `v2ecoli/workflow/post_sim.py`, `analysis.py`, `report_cards/__init__.py`; update `v2ecoli/tests/test_report_card_step.py`, `test_tests_card.py`, `test_vs_vecoli_card.py`, `test_vs_literature_card.py` (+ any analysis tests). Concrete cards, `MassFractionSummary`, and `scripts/study_report_cards.py` unchanged (they subclass the re-exported bases).
**Produces:** v2ecoli imports the bases/registries/`StudyContext`/`write_card` from `viva_superpowers` (re-export for back-compat so `from v2ecoli.workflow.post_sim import ReportCardStep` still works); the concrete subclasses register into the shared registries unchanged; the POST_SIM funnel uses the shared `POST_SIM_REGISTRY`.

- [ ] Step 1 — run the EXISTING v2ecoli post-sim test suite (`test_report_card_step`, `test_tests_card`, `test_vs_*`, analysis tests) against the current tree to capture the green baseline.
- [ ] Step 2 — rewire imports: `v2ecoli/workflow/post_sim.py` + `analysis.py` + `report_cards/__init__.py` import the bases/registry/`StudyContext`/`write_card` from `viva_superpowers` and re-export them; delete the now-duplicated base definitions. Keep `Analysis`/`AnalysisStep` concrete-facing behavior identical.
- [ ] Step 3 — run the existing suite (with the `[vsp]` worktree installed/on PYTHONPATH); expect PASS with no behavior change. Fix import fallout only.
- [ ] Step 4 — commit on `post-sim-rewire` (`refactor(v2e): post-sim Step bases import from viva_superpowers (one home)`).

## Decisions (from design approval + Fable whole-program streamline)
- Subclass pbg `Step` (drop `V2Step`); keep the error-swallowing `invoke()` on the shared bases.
- **AMENDED (Fable §1.4): verbatim move NOW (T1, done), collapse LATER (T2).** T1 keeps both the live-`conn` `Analysis` and the record `AnalysisStep` bases (as shipped in commit 5fe61c9). When `ResultsHandle` lands in **T2**, collapse to **one `AnalysisStep` base** whose input is the `ResultsHandle` (offering `.records(scale)` + lazy `.conn()`), keeping `Analysis` as a thin deprecated alias. Doing the collapse in T2 (not T1) avoids reworking the committed move and lets the handle drive the merged surface. v2ecoli's T3 rewire must import the collapsed surface (do T2 before T3 fossilizes it).
- **Emitters import (Fable §1.4): use `viva_emitters`, NOT `pbg_emitters`.** viva-superpowers renamed the emitters dep to `viva-emitters` and dropped the `pbg_emitters` shim (branch base `1dd4adc`, PRs #249/#250). T2's `ResultsStep` imports `viva_emitters`; the T2 implementer verifies the exact module name against the worktree before coding.
- `ResultsHandle` is a typed object (records + lazy conn) **that reconstructs from `{paths, sim_data_ref}` config** (so file-based rehydration is possible for a future Nextflow/CWL Evaluate stage — Fable §1.3) → deterministic, cache-keyable. **v1 Evaluate is LocalRunner-only** (the live handle can't cross a subprocess boundary; Evaluate-under-Nextflow is Phase 6).
- `StudyContext` keeps its workspace-layout knowledge (viva_superpowers owns workspace paths).
- **Merge order (Fable §1.5): T3 (v2ecoli rewire) merges into v2ecoli main BEFORE the Phases-1-3 `nextflow-parca` T8 branch** (disjoint files, small-first avoids rebase load).

## Self-Review
Coverage: family move (T1); ResultsStep/ResultsHandle + deterministic wiring proof (T2); v2ecoli rewire with green existing tests (T3). Determinism: post-sim Steps are pure `(results, config)→output`, topo-ordered after `ResultsStep`. Gating: `ReportCardStep` verdict schema `{status,checks,summary}`. Type consistency: `ResultsHandle` (T2) is the input contract for `AnalysisStep`/`ReportCardStep` (T1); registries populated by `__init_subclass__` consumed by `iter_post_sim`/`applicable`; v2ecoli concretes (T3) subclass the T1 bases.
