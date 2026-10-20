# Holes-in-Place: Contract Unification (P1) — Design

**Status:** draft for approval
**Date:** 2026-10-20
**Repos touched:** `process-bigraph` **only** — single-repo. Everything P1 needs
already exists in `bigraph_schema` submodules, so `process_bigraph` re-exports it
(`from bigraph_schema.matching import find_candidates`, etc.); **no change to
`bigraph_schema` and no two-repo coordinated release**. (Confirmed: candidate
suggestions use `find_candidates` as-is — best-effort, Option A below — so no
bigraph-schema change is needed.)
**Scope:** P1 of a 3-part program. P2 (migrate ~101 `@draft_process` usages, then
delete the class) and P3 (study↔composite face reconciliation) are sequenced
follow-ons and are **out of scope here**.

---

## Intent (the why)

Today the word "contract" names three different things that don't share code:

1. `bigraph_schema.ProcessContract` — the real, structured, typed, auditable,
   runtime-enforceable interface (face + conditions + math + monotone amendments).
2. `process_bigraph.DraftProcess` — a *free-form dict* (`makes`/`senses`/`holds`/…)
   on an inert registered `Process`, unrelated to `ProcessContract`.
3. A study's `contract.face` JSON-schema block (viva-template) — a third spelling,
   disconnected from the other two.

And there are two parallel ways to say "a process I haven't implemented yet":
`DraftProcess` (an inert *node* with a fake-looking address) and an **open site**
(a real *hole* in a document, filled by `fill_sites`). The second is already
implemented, tested, and documented (`process_bigraph.templates`,
`bigraph_schema.assembly`), and it is the process-bigraph-native one: Milner's
filling discipline, where *"a site's sort IS a contract"* and
`admits(core, site, filler)` decides whether a filler closes a hole.

**The decision (user):** go fully to holes-in-place. A draft is not a kind of
process — it is **a process-slot left as a hole whose sort is a
`ProcessContract`**. "No mechanism to execute yet" just means the hole is
unfilled. The Composite **rejects** an unfilled hole and **reports back** which
hole, what contract it requires, and which registered processes would fill it —
replacing today's opaque `KeyError: 'data'`.

P1 makes holes the one model: it surfaces and completes the existing hole
machinery into a coherent, documented, tested public affordance, wires the one
unused piece (`register_contract`), makes rejection legible, and **deprecates**
(does not yet delete) `DraftProcess` so the ~101 existing usages keep working.

**Payoff:** one contract concept; a legible "what's missing and how to fill it"
error that improves simulation/study robustness; and the groundwork for the
code reduction that P2 banks when the parallel `DraftProcess` model is deleted.

---

## Goals

1. **One public "holes" API.** Surface a single, documented entry point for the
   design→inspect→fill→build lifecycle, reusing what exists:
   - author a hole: a `Site` whose `_sort` names a registered `ProcessContract`
     (or a face literal `link[in,out]`);
   - inspect: `open_sites` / `required_open_sites` / `is_ground_document`;
   - see what fits: `find_candidates(core, site)` (already built in
     `bigraph_schema.matching`) → admissible processes + near-misses with reasons;
   - fill: `fill_sites` / `fill_template`;
   - build: `Composite` once ground.
2. **Wire `register_contract`.** `Core.register_contract(name, contract)` exists
   with **zero callers**. Make named `ProcessContract`s registrable and resolvable
   as a site's sort, so an intended interface is a *named, reusable contract in the
   registry* — recovering the one useful thing `DraftProcess` gave (a named,
   discoverable interface) as a contract rather than as an inert process.
3. **Legible, contract-aware rejection.** When `Composite` construction hits an
   unfilled required hole, raise a structured error that, per hole, reports: its
   path, the contract it requires (face ports with types/units/bounds, plus
   conditions), and the admissible candidates from `find_candidates`. Both
   human-readable (`str`) and machine-readable (structured attribute) so the
   workbench can render it.
4. **Deprecate `DraftProcess` without breaking anyone.** `@draft_process` keeps
   working and inert (composites keep building), now emitting a `DeprecationWarning`
   that points to the holes model. Add a pure bridge
   `draft_to_contract(cls) -> ProcessContract` that turns a draft's
   `DRAFT_INPUTS`/`DRAFT_OUTPUTS` + free-form dict into a real `ProcessContract`
   (face from ports; prose into `summary`/`description`/`assumptions`). This is the
   P2 migration seam and lets deprecated drafts surface a real contract to tooling.
5. **Document and test the whole lifecycle.** Rewrite process-bigraph's own draft
   docs to the holes model, add a "designing with holes" guide, and add tests that
   exercise the full lifecycle and the rejection report end to end.

---

## Non-goals (explicitly P2/P3, not here)

- **Not** migrating the ~101 `@draft_process` subclasses across the 9 consumer
  repos. (P2.)
- **Not** deleting `DraftProcess`/`@draft_process` or the
  `contract_drift._is_draft` exemption. (P2 — after consumers migrate.)
- **Not** changing the runtime behavior of existing `@draft_process` usages: a
  deprecated draft still builds and runs inert. P1 is additive.
- **Not** wiring the study↔composite face reconciliation
  (`contract_spine.contract_face_diff` / `lint_study_contract`). (P3.)
- **Not** rewriting downstream doc repos (viva-docs, viva-superpowers skills,
  meta-modelers-guide). P1 fixes process-bigraph's own docs and records the
  downstream list for a doc-sweep task; broad rewrites ride with P2.

## Anti-goals (would make this a failure even if it "works")

- Reimplementing `admits` / `fill_sites` / `contract_subsumes` / `find_candidates`.
  These exist and are correct; P1 **surfaces and composes** them, it does not
  rewrite them. A diff that duplicates that logic is wrong.
- A rejection error that is merely prettier but still only lists paths. The
  value is the *contract + candidates* per hole; an error without them misses
  the point.
- Breaking the 101 downstream usages. Any change that stops an existing
  `@draft_process` composite from building is a regression, not a cleanup.
- Growing net line count in P1 without the P2 deletion that pays it back — keep
  P1 additions tight; the reduction is banked in P2.

---

## Constraints (copy these verbatim into the plan's Global Constraints)

- **No AI attribution** in commits or PRs (no `Co-Authored-By: Claude`, no
  "Generated with Claude Code" footer).
- **Worktree, off `origin/main`.** Work in `~/code/process-bigraph--holes-p1`
  (branch `feat/holes-in-place-p1`, already created off `origin/main` `acc444c`).
  Any bigraph-schema change gets its own worktree off *its* `origin/main` (the
  canonical `~/code/bigraph-schema` is stale on `chore/bump-1.4.3`).
- **Versions.** Current truth is `bigraph-schema 1.8.1` / `process-bigraph 1.9.1`
  (`origin/main` = the installed venv at `~/code/viva-genesis/.venv`). Validate
  against that venv with `PYTHONPATH=<worktree>` prepended.
- **Additive / non-breaking.** Existing `@draft_process` usages must keep building
  and running unchanged; only a `DeprecationWarning` is added.
- **Reuse, don't rebuild** the assembly/contract/matching/subsumption algorithms.

---

## Current state (verified, with anchors)

Reuse as-is:
- `bigraph_schema/contract.py` — `ProcessContract` (fields incl. `face`,
  `amendments`; `conditions()`/`predicates()`), `resolve_contract`,
  `narrow_condition`, `amend`.
- `bigraph_schema/assembly.py` — `contract_of(core, node)` (:702, *"a site's sort
  IS a contract"*), `admits`/`admits_why`/`contract_admits`/`face_conforms`
  (:764–864), `fill_sites`, `collect_sites`.
- `bigraph_schema/contract_audit.py` — `audit_contract`, `completeness`.
- `bigraph_schema/subsumption.py` — `contract_subsumes`.
- `bigraph_schema/matching.py` — `find_candidates(core, site) ->
  [CandidateMatch(address, match, over_provides, fails)]` (full matches first,
  then near-misses with reasons). **This is the "what fits this hole" enumerator.**
- `bigraph_schema/core.py` — `register_sort` (:400), **`register_contract` (:411,
  zero callers)**, `describe_contract(node)` (:420 → `contract_of`), `admits`
  (:431), `fill_sites` (:436); `contract_registry` seeded at core.py:132.
- `process_bigraph/templates.py` — `open_sites`, `required_open_sites`,
  `is_ground_document`, `fill_template`, `template_document` (raises naming
  unfilled sites), `prune_open_regions`, `investigation_document`,
  `load_template`/`save_template`.
- `process_bigraph/composite.py` — `_require_ground_input` (:2912, pre-realize,
  catches the opaque case B), `_require_ground_document` (:2960, post-realize
  backstop), `_raise_not_ground` (:2902, **names paths only, not contracts**).
- `process_bigraph/contract_drift.py:48` — `_is_draft` exemption (keep in P1).

Gaps P1 closes:
- `register_contract` has **zero callers** — named contracts can't actually be
  registered/used as sorts anywhere.
- `_raise_not_ground` reports **paths only** — not the contract each hole requires,
  nor how to fill it.
- The holes API is **not surfaced**: `process_bigraph.__init__` exports
  `DraftProcess`/`draft_process` but not the `templates` helpers; `bigraph_schema`
  exports only `ProcessContract`/`resolve_contract`, not `find_candidates` /
  `contract_of` / etc.
- No single documented lifecycle; `DraftProcess` is a parallel, weaker model with
  no deprecation path and no bridge to `ProcessContract`.

---

## The design (decided how)

### A. Named contracts as sorts (`register_contract` wired)

A core may register a named `ProcessContract`; a `Site(_sort='<name>')` then
requires it. The resolution path already exists (`contract_of` →
`core.contract_registry[sort]`, assembly.py:727). P1 adds:

- A convenience on the authoring surface so a workspace's `build_core()` can
  register its intended-interface contracts by name alongside its processes.
  (Mechanism + documented example in P1; wholesale workspace wiring is P2.)
- A face-literal path stays supported unchanged (`_sort='link[in,out]'`), for
  holes that need only structural typing without a named contract.

Left to builder: whether registration is a thin helper over
`core.register_contract` or also a decorator/registry-scan convenience — pick the
smallest thing that makes the documented example clean.

### B. Legible rejection: `UnfilledHoleError`

Introduce a `ValueError` subclass (name TBD by builder; suggest
`UnfilledHoleError`) raised by the groundness guards in place of the bare
`_raise_not_ground`. It carries structured, per-hole data and renders readably:

```
for each unfilled required site:
  - path               e.g. "study/model"
  - contract           core.describe_contract(site)  (None if the hole is unsorted)
      · face: ports → {_type, _units, _min, _max}
      · conditions: pre/invariant/post/validity (names + exprs)
  - candidates         find_candidates(core, site)
      · full matches (admissible now)
      · near-misses + the failing condition
```

`str()` gives a scannable report; a structured attribute (e.g. `.holes`, a list of
small dataclasses/dicts) lets the workbench render it. Degrade gracefully: an
unsorted hole reports "no contract declared"; `find_candidates` failures never
mask the primary "not ground" error.

**Candidates are best-effort (Option A, decided).** `find_candidates` is reused
as-is: it reads a candidate's contract via `edge_class({}, core)` and silently
skips any process that needs real config to construct, so for config-heavy
registries the candidate list may be empty. That is acceptable for P1 — the
guaranteed value of the report is the **hole path + required contract**; the
candidate list is an actionable bonus when present. A config-free declared-contract
read path (to make suggestions robust against real registries) is an explicit
fast-follow, taken only if empty lists prove painful in practice; it is **not**
P1, and P1 must not rewrite `find_candidates` to get there.

Wire it into `_require_ground_input` (:2912) and `_require_ground_document`
(:2960); `template_document` (:138) raises the same condition pre-build and should
produce the same report for symmetry.

Left to builder: exact class name, attribute shape, and whether
`_raise_not_ground` is replaced or delegates to the new path.

### C. Public holes API (surface)

Export one coherent set from `process_bigraph.__init__` so an author has an
obvious entry point: the lifecycle (`open_sites`, `required_open_sites`,
`is_ground_document`, `fill_template`, `template_document`), the matcher
(`find_candidates`, re-exported from `bigraph_schema.matching`), and the new
`UnfilledHoleError`. **Do not modify `bigraph_schema.__init__`** — P1 re-exports
the bigraph-schema names it needs (`find_candidates`, `contract_of`,
`audit_contract`, `completeness`, `narrow_condition`, `contract_subsumes`) from
their submodules, keeping P1 single-repo. `Core` methods (`register_contract`,
`describe_contract`, `admits`, `fill_sites`) remain the instance-level spelling.

Left to builder: final export list; don't export internal helpers.

### D. Deprecate `DraftProcess` + bridge

- `@draft_process` and `DraftProcess` keep their current inert-node behavior
  (composites build, run returns `{}`), **plus** a one-time deprecation warning at
  decoration naming the holes model and pointing at the new guide. Use a
  **dedicated `DeprecationWarning` subclass** (e.g. `DraftProcessDeprecationWarning`),
  so a consumer can target an ignore at exactly this warning. (As built: the warning
  is issued per decoration and relies on Python's default per-call-site dedup for
  "seen once" — a module-level guard proved order-dependent and untestable; see the
  ledger ruling.) **Pre-merge task (non-breaking guard):** grep the 9
  consumer repos' pytest/CI configs for `filterwarnings = error` / `-W error`;
  if any treat warnings as errors, ship a documented `filterwarnings` ignore for
  the new category rather than turning their CI red. This protects the hard
  anti-goal "don't break the 101 downstream usages."
- Add a pure function `draft_to_contract(cls) -> ProcessContract`:
  `face['inputs'|'outputs']` from `DRAFT_INPUTS`/`DRAFT_OUTPUTS`; `summary` from the
  dict's `summary`; `makes`/`moves`/`senses`/`holds`/`regulation` folded into
  `description`/`assumptions`; `status` dropped. No side effects, no registration —
  it's the P2 conversion seam and a way for tooling to read a draft's real contract.
- Keep `contract_drift._is_draft` (:48) unchanged in P1.

Left to builder: whether the warning fires at decoration, first build, or both
(pick the least noisy that a consumer will actually see once).

### E. Docs + tests

Docs (process-bigraph only in P1):
- Rewrite the draft sections of `README.md` (:103–126, the "draft process" vs
  "template with an open site" contrast collapses to one model) and
  `docs/concepts/composites-and-templates.md`.
- Add a short "designing with holes" guide: author a contract, register it,
  author a hole sorted by it, `find_candidates`, `fill_sites`, build; show the
  rejection report for an unfilled build.
- Record (do not rewrite) the downstream docs for the P2 sweep: viva-docs
  `templates-and-draft-processes.md` / `glossary.md` / `core-concepts.md`,
  viva-superpowers `skills/viva-model-build/SKILL.md` and the concepts doc,
  meta-modelers-guide, viva-littleb.

Tests (new, in `process_bigraph/tests/` — the current convention):
1. Register a named contract; author `Site(_sort=name)`; `contract_of`/
   `describe_contract` returns it with its face.
2. `find_candidates` returns a registered admissible process as a full match, and
   a near-miss with the failing condition, and omits a non-conforming one.
3. `fill_sites` with an admissible filler grounds the document; `Composite` builds
   and runs.
4. Building/`template_document` with the hole unfilled raises `UnfilledHoleError`;
   assert the message/structured data names the hole path, its contract face, and
   at least one candidate.
5. `admits` rejects a non-conforming filler and the error explains why.
6. `@draft_process` emits a `DeprecationWarning` and still builds/runs inert
   (non-breaking).
7. `draft_to_contract` produces a `ProcessContract` whose face matches
   `DRAFT_INPUTS`/`DRAFT_OUTPUTS` and whose prose carries the free-form fields.

Validate with `~/code/viva-genesis/.venv/bin/python` and
`PYTHONPATH=~/code/process-bigraph--holes-p1` prepended (confirm
`python -c "import process_bigraph, bigraph_schema; print(...__file__)"` resolves
to the worktree). If a bigraph-schema export change is needed, it rides in a
sibling bigraph-schema worktree off its `origin/main` and the two land together.

---

## What's left to the builder (summary)

- Exact name/shape of `UnfilledHoleError` and its structured attribute.
- The smallest clean `register_contract` authoring convenience + its example.
- Final public export lists for both `__init__`s.
- Where the `DeprecationWarning` fires.
- Whether any bigraph-schema change (beyond exports) is actually required, or
  whether everything composes from `Core` methods already present. (Expectation:
  process-bigraph-only plus exports; confirm during the first task.)

## Decisions on record (so they aren't reopened)

- **Reject + report**, not a Composite that holds open holes (draft mode).
- **Single-repo** (process-bigraph only; re-export from bigraph-schema submodules).
- **Deprecate, don't delete** `DraftProcess` in P1; dedicated warning subclass,
  fired once, with a pre-merge audit of downstream warning filters.
- **Option A** for candidate suggestions: reuse `find_candidates` best-effort;
  robust config-free reading is a fast-follow, not P1.

Approved 2026-10-20.
