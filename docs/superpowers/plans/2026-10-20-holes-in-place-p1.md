# Holes-in-Place Contract Unification (P1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "an open hole whose sort is a `ProcessContract`" the one model for a not-yet-implemented process-slot: surface the existing hole lifecycle as a public API, make `Composite` reject an unfilled hole with a legible contract-and-candidates report, and deprecate (not delete) `DraftProcess`.

**Architecture:** P1 is single-repo (`process-bigraph` only) and additive. It composes machinery that already exists in `bigraph_schema` (contract/assembly/matching/subsumption) and `process_bigraph.templates`; it does **not** reimplement `admits`/`fill_sites`/`find_candidates`/`contract_subsumes`. New surface: a `process_bigraph/holes.py` module (the `UnfilledHoleError` + report builder), top-level re-exports, a dedicated deprecation warning + a pure `draft_to_contract` bridge, docs, and tests.

**Tech Stack:** Python 3.12; `process-bigraph` 1.9.1 / `bigraph-schema` 1.8.1 (`origin/main` = the installed venv at `~/code/viva-genesis/.venv`); pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-holes-in-place-unification-design.md` (read it; this plan argues from it).

## Global Constraints

- **No AI attribution** in commits or PR bodies/titles (no `Co-Authored-By: Claude`, no "Generated with Claude Code" footer).
- **Single-repo:** all changes in `process_bigraph`. **Do not modify `bigraph_schema`.** Re-export bigraph-schema names from their submodules.
- **Additive / non-breaking:** every existing `@draft_process` usage must still build and run inert; only a deprecation warning is added. No change to runtime `contract_strict` semantics.
- **Reuse, don't rebuild:** `admits`, `fill_sites`, `find_candidates`, `contract_subsumes`, `contract_of` are used as-is. A diff that duplicates their logic is wrong.
- **Worktree:** `~/code/process-bigraph--holes-p1`, branch `feat/holes-in-place-p1` (off `origin/main` `acc444c`). Validate with `~/code/viva-genesis/.venv/bin/python` and `PYTHONPATH=~/code/process-bigraph--holes-p1` prepended; confirm `python -c "import process_bigraph; print(process_bigraph.__file__)"` resolves into the worktree before trusting a test run.
- **Tests** go in `process_bigraph/tests/` (the current convention), not root `tests.py`.
- **Candidates are best-effort (Option A):** `find_candidates` is reused unchanged; an empty candidate list is acceptable and must render gracefully.

## Review Focus

- **Unsorted hole (no contract).** A hole whose `_sort` is empty must report "no contract declared" and must not crash the report builder. → Task 2.
- **`find_candidates` empty or raising.** An empty candidate list renders cleanly; any exception from candidate-finding never masks the primary "not ground" error. → Task 2.
- **Face-literal sort vs registered-name sort.** A hole sorted `link[in,out]` and a hole sorted by a registered contract name both report their face correctly. → Task 2 / Task 4.
- **Ground composite with ordinary unwired ports.** A normal composite (processes with unwired ports, no sites) must still build — the guards key on *sites*, not `is_ground`. Must not regress. → Task 3.
- **Downstream `-W error` CI.** A consumer repo running pytest with `filterwarnings = error` must not have its `@draft_process` modules break; the dedicated warning subclass + the audit guard this. → Task 1.

---

## File Structure

- Create `process_bigraph/holes.py` — `UnfilledHoleError(ValueError)`, the per-hole report dataclass, and `build_unfilled_hole_error(core, unfilled_sites)`. Imports `contract_of`/`find_candidates` from bigraph-schema submodules. (Dedicated module so both `composite.py` and `templates.py` import it without a cycle.)
- Modify `process_bigraph/draft_process.py` — add `DraftProcessDeprecationWarning`, fire it once in `draft_process`; add `draft_to_contract(cls)`.
- Modify `process_bigraph/composite.py` — `_raise_not_ground` / `_require_ground_input` / `_require_ground_document` build and raise `UnfilledHoleError`.
- Modify `process_bigraph/templates.py` — `template_document` raises `UnfilledHoleError`.
- Modify `process_bigraph/__init__.py` — re-export the public holes API.
- Modify `README.md`, `docs/concepts/composites-and-templates.md`; create `docs/concepts/designing-with-holes.md`.
- Tests: `process_bigraph/tests/test_holes.py`, `process_bigraph/tests/test_draft_deprecation.py`.

---

### Task 1: Deprecate `DraftProcess` + `draft_to_contract` bridge

**Files:**
- Modify: `process_bigraph/draft_process.py`
- Test: `process_bigraph/tests/test_draft_deprecation.py`

**Interfaces:**
- Produces: `DraftProcessDeprecationWarning(DeprecationWarning)`; `draft_to_contract(cls) -> bigraph_schema.ProcessContract` (pure; face from `DRAFT_INPUTS`/`DRAFT_OUTPUTS`, prose from the free-form dict).
- Consumes: existing `DraftProcess`, `@draft_process`, `_CONTRACT_FIELDS`.

- [ ] **Step 1: Write failing tests.**
```python
def test_draft_process_warns_once_but_still_builds_inert():
    # @draft_process on a DraftProcess subclass emits exactly one
    # DraftProcessDeprecationWarning; the class still has DRAFT_INPUTS/OUTPUTS,
    # describe() still renders "DRAFT —", and base update() returns {}.
def test_draft_to_contract_maps_ports_and_prose():
    # draft_to_contract(cls).face == {'inputs': DRAFT_INPUTS, 'outputs': DRAFT_OUTPUTS}
    # and summary/makes/senses text lands in summary/description (not dropped).
def test_dedicated_warning_is_filterable():
    # warnings.simplefilter('error', DraftProcessDeprecationWarning) makes
    # decoration raise; an unrelated -W error on DeprecationWarning base also
    # catches it (documents the downstream risk the audit addresses).
```
- [ ] **Step 2: Run tests, verify they fail.** `pytest process_bigraph/tests/test_draft_deprecation.py -v`
- [ ] **Step 3: Implement.** Add `DraftProcessDeprecationWarning`; in `draft_process` fire it once via a module-level `_warned` guard (`warnings.warn(..., DraftProcessDeprecationWarning, stacklevel=2)`), message names the holes model + the new guide. Add `draft_to_contract(cls)` building a `ProcessContract` (face from the two port dicts; `summary` from `contract['summary']`; `makes/moves/senses/holds/regulation` folded into `description`; `status` dropped). No registration, no side effects.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Downstream warning-filter audit (non-breaking guard).** Grep the 9 consumer repos under `~/code` (cog_bio_of_the_cell, viva-human-atlas, spatio-flux, viva-paper-demo, meta-modelers-guide, viva-littleb, viva-compiler, + any workspace importing `draft_process`) for `filterwarnings` / `-W error` in `pyproject.toml`/`pytest.ini`/`setup.cfg`/CI yaml. Record findings in the commit message. If any treat warnings as errors, add a one-line note to `docs/concepts/designing-with-holes.md` (Task 5) giving the exact `filterwarnings` ignore for `DraftProcessDeprecationWarning`. (Read-only across repos; do not edit other repos.)
- [ ] **Step 6: Commit.** `feat(draft): deprecate DraftProcess (dedicated warning) + draft_to_contract bridge`

---

### Task 2: `UnfilledHoleError` + report builder (`process_bigraph/holes.py`)

**Files:**
- Create: `process_bigraph/holes.py`
- Test: `process_bigraph/tests/test_holes.py`

**Interfaces:**
- Produces: `UnfilledHoleError(ValueError)` with attribute `.holes: list[HoleReport]`; `HoleReport` dataclass `{path: tuple[str,...], contract: ProcessContract | None, candidates: list}`; `build_unfilled_hole_error(core, unfilled_sites) -> UnfilledHoleError` where `unfilled_sites` is a list of `(path, site)` pairs.
- Consumes: `bigraph_schema.assembly.contract_of`, `bigraph_schema.matching.find_candidates`.

- [ ] **Step 1: Write failing tests.**
```python
def test_report_names_path_contract_and_candidates():
    # core with a registered contract 'PTH' and a conforming registered process;
    # a doc with Site(_sort='PTH') unfilled.
    # build_unfilled_hole_error(...).holes[0] has path, contract.face with the
    # ports, and >=1 candidate (full match) address. str(err) contains all three.
def test_unsorted_hole_reports_no_contract_and_does_not_crash():
    # Site(_sort='') → HoleReport.contract is None, candidates [], str mentions
    # "no contract declared".
def test_find_candidates_failure_never_masks_primary_error():
    # monkeypatch find_candidates to raise → still get an UnfilledHoleError whose
    # str names the hole path; candidates render as empty, not a crash.
def test_is_subclass_of_valueerror():
    assert issubclass(UnfilledHoleError, ValueError)
```
- [ ] **Step 2: Run tests, verify fail.** `pytest process_bigraph/tests/test_holes.py -v`
- [ ] **Step 3: Implement `holes.py`.** `build_unfilled_hole_error` maps each `(path, site)` to a `HoleReport` via `contract_of(core, site)` (None-safe) and `find_candidates(core, site)` wrapped so any exception → `[]`. `UnfilledHoleError.__str__` renders the "composite document is not ground" header + one block per hole: path, required face (ports → `_type`/`_units`/`_min`/`_max`) and conditions, then full-match candidate addresses and near-misses with their failing condition; "no contract declared" when contract is None.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit.** `feat(holes): UnfilledHoleError with contract + candidate report`

---

### Task 3: Wire `UnfilledHoleError` into the Composite groundness guards

**Files:**
- Modify: `process_bigraph/composite.py` (`_raise_not_ground` ~:2902, `_require_ground_input` ~:2912, `_require_ground_document` ~:2960)
- Test: `process_bigraph/tests/test_holes.py` (add cases)

**Interfaces:**
- Consumes: `process_bigraph.holes.build_unfilled_hole_error`.
- The guards currently collect unfilled site *paths*; change them to collect `(path, site)` pairs and pass to the builder.

- [ ] **Step 1: Write failing tests.**
```python
def test_building_composite_with_unfilled_hole_raises_report():
    # Composite(... doc with an unfilled Site(_sort='PTH')) raises UnfilledHoleError;
    # the message names the hole path, its contract, and a candidate.
def test_except_valueerror_still_catches():
    # existing callers that do `except ValueError` still catch it.
def test_ground_composite_with_unwired_ports_still_builds():
    # a normal composite (real process, unwired ports, no sites) builds fine —
    # no false rejection (sites vs is_ground).
```
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement.** In both guards, build the `(path, site)` list (reusing the existing `interfaces(accessed)[0]._places` / `collect_sites` walks, keeping the `_default is None` filter), and raise `build_unfilled_hole_error(self.core, pairs)`. `_raise_not_ground` either delegates to the builder or is replaced; keep the human message text's "not ground … site(s)" lineage so existing string-sniffing tests (if any) survive. Guard the builder import to avoid import cycles (local import inside the method, matching the file's existing pattern).
- [ ] **Step 4: Run tests + the existing composite groundness tests, verify pass.** `pytest process_bigraph/tests/ -k "ground or holes" -v`
- [ ] **Step 5: Commit.** `feat(composite): legible UnfilledHoleError at construction`

---

### Task 4: Public holes API + named-contract authoring example

**Files:**
- Modify: `process_bigraph/__init__.py`
- Test: `process_bigraph/tests/test_holes.py` (add an end-to-end lifecycle test)

**Interfaces:**
- Produces (top-level `process_bigraph` names): `UnfilledHoleError`, `open_sites`, `required_open_sites`, `is_ground_document`, `fill_template`, `template_document`, `find_candidates`, `draft_to_contract`, `DraftProcessDeprecationWarning`.
- Consumes: `process_bigraph.holes`, `process_bigraph.templates`, `bigraph_schema.matching`, `process_bigraph.draft_process`.

- [ ] **Step 1: Write failing end-to-end lifecycle test.**
```python
def test_hole_lifecycle_design_inspect_fill_build():
    # import everything from top-level process_bigraph.
    # 1. core.register_contract('PTH', contract)  (existing Core method)
    # 2. author doc with Site(_sort='PTH'); open_sites/required_open_sites see it;
    #    is_ground_document is False.
    # 3. find_candidates(core, site) returns the registered conforming process.
    # 4. fill_template/fill_sites with that process → is_ground_document True.
    # 5. Composite(...) builds and run() advances.
def test_public_api_importable():
    import process_bigraph as p
    for name in ("UnfilledHoleError","open_sites","required_open_sites",
                 "is_ground_document","fill_template","template_document",
                 "find_candidates","draft_to_contract"):
        assert hasattr(p, name)
```
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement.** Add the re-export lines to `__init__.py` (grouped, `# noqa: F401`, matching the file's style). If — and only if — the lifecycle example needs it to read cleanly, add a thin `register_contract` convenience; otherwise use the existing `Core.register_contract`. Do not export internal helpers.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit.** `feat(api): surface public holes lifecycle from process_bigraph`

---

### Task 5: Docs — holes model + designing-with-holes guide

**Files:**
- Modify: `README.md` (the "draft process" vs "template with an open site" section, ~:103–159)
- Modify: `docs/concepts/composites-and-templates.md`
- Create: `docs/concepts/designing-with-holes.md`

- [ ] **Step 1: Rewrite README draft section** so the two models collapse into one: a draft **is** an open hole sorted by a contract; show `register_contract` → `Site(_sort=…)` → `find_candidates` → `fill` → `Composite`, and the `UnfilledHoleError` you get if you build unfilled. Mark `@draft_process` as deprecated with a one-line migration pointer.
- [ ] **Step 2: Update `composites-and-templates.md`** to the same model; remove the claim that `DraftProcess` is the way to express an unimplemented process; point to the new guide.
- [ ] **Step 3: Write `designing-with-holes.md`** — the full lifecycle walkthrough (the Task 4 example as prose + code), the rejection report shown verbatim, the `draft_to_contract` migration note, and (if Task 1's audit found any) the exact `filterwarnings` ignore for `DraftProcessDeprecationWarning`. Record the downstream docs to rewrite in P2 (viva-docs `templates-and-draft-processes.md`/`glossary.md`/`core-concepts.md`; viva-superpowers `skills/viva-model-build/SKILL.md` + concepts doc; meta-modelers-guide; viva-littleb) as a short "P2 doc sweep" list.
- [ ] **Step 4: Verify the guide's code block runs** (copy it into a scratch script under the scratchpad, run with the venv + `PYTHONPATH`, confirm it builds and the unfilled case prints the report). Fix drift.
- [ ] **Step 5: Commit.** `docs: holes-in-place model + designing-with-holes guide`

---

## Final whole-branch review

After Task 5: run the full `process_bigraph/tests/` suite against the venv with `PYTHONPATH` prepended; confirm no regression in the existing groundness/template/draft tests; dispatch the final whole-branch review. Then `superpowers:finishing-a-development-branch` (verify → commit → push → `gh pr create`, no auto-merge).
