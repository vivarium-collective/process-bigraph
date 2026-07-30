# Hardening the composite concept + lowering onto the bigraph-schema template

**Status:** Design (Layer 2a of the framework-unification stack)
**Date:** 2026-07-30
**Repo:** process-bigraph
**Umbrella:** `vivarium-workbench/docs/superpowers/specs/2026-07-29-framework-unification-design.md` (PR #676)
**Depends on (Part B only):** bigraph-schema `template`/`slot` primitive (Layer 1, PR #174)
**Supersedes premise of:** "migrate `generate_composite` from viva-superpowers" — that migration already shipped (2026-06-28); this spec addresses what's *left*.

---

## 1. Context — the migration already shipped

Ground-truth (2026-07-30): the composite generator is **already** in process-bigraph.
`CompositeSpec` (`process_bigraph/composite_spec.py:113`, 433 lines, **28 passing
tests**) is the single source of truth; `viva_superpowers` (in the `pbg-superpowers`
repo) is a **thin shim** — `@composite_generator` delegates to `@composite_spec`,
`_REGISTRY` is a live view over the process-bigraph registry. The 2026-06-28
"composite-spec-unified-declaration" plan Tasks 1–8 are done in code; the shim
persisting is **by design** (removing it was out of scope). There is no
`generate_composite()` function anywhere — the public API is:

```
spec = CompositeSpec.from_file(path)            # or @composite_spec / registry get(id)
doc  = spec.to_document(overrides, core)        # → {"schema":.., "state":..}
comp = spec.to_composite(overrides, core)       # → Composite(doc, core)  (runnable Process)
```

`Composite(Process)` (`composite.py:1023`) consumes the document; the dependency is
one-way (composite_spec → Composite, lazily imported). `CompositeSpec` rolls its
**own `${name}` regex substitution** (`substitute_parameters`, composite_spec.py:91)
and does **not** touch bigraph-schema's `Site`/`compose`/sorts.

This spec has two parts. **Part A (now):** harden what the migration left seamed.
**Part B (after Layer 1):** lower `CompositeSpec`'s parameterization onto the
bigraph-schema template primitive so a composite spec *is* a template.

---

## PART A — Harden the composite concept (independently shippable)

### A0. Enforce groundness at the boundary (the hardening this spec is named for)

`Composite.__init__` (`composite.py:1023`) does **not** call `is_ground` on the
document it is handed, so Layer 1's central contract ("filling produces a ground
document") is unenforced at the one place it is consumed. **A0:** `Composite.__init__`
rejects a document with open **required** sites, naming the unfilled sites in the
error. Cheap; turns a contract into a guarantee; and it is the precondition that
makes gating-by-groundness (umbrella Layer 2) *safe* — a blocked study is a
non-ground document, and a non-ground document must fail loudly if something tries
to run it. Ship this first.

### A1. Emitter materialization belongs in process-bigraph (a real bug)

`CompositeSpec.emitters` is a first-class field, but `to_composite()`
(composite_spec.py:218) builds `Composite(doc, core)` and **never installs
emitters** — emitter materialization lives only in the shim
(`viva_superpowers/composite_generator.py`: `install_default_emitters`,
`_emitter_node_from_decl`, `emitter_defaults`, `_validate_emitters`, ~284–411). So
a composite built via the pure process-bigraph API gets **no observation sink**,
while one built via the viva shim (`build_composite_from_spec`) does. This is a
correctness gap, not a style one.

**Fix — reuse the constructors that already exist; do not add a third.**
process-bigraph already has **two** node constructors (`emitter_from_wires`,
`emitter.py:42`; `generate_emitter_state`, `emitter.py:77`) and **one** installer
(`add_emitter_to_composite`, `emitter.py:125`). The shim's `_emitter_node_from_decl`
(`composite_generator.py:332`) is a third that differs only in deriving wires from a
`paths` list and layering Parquet run-partition keys. Porting it verbatim would leave
three. Instead:

- Re-express the *declaration* form as `paths → wires`, then delegate to
  **`emitter_from_wires`**, and install via **`add_emitter_to_composite`** — one
  constructor, one installer. The Parquet run-partition layering becomes a config
  overlay applied by the caller, not a branch inside the constructor.
- Call it from `to_document`/`to_composite` so a spec's declared emitters are
  present in `state`; add an `emit=False` flag for a caller wanting the bare document.
- **The shim's `install_default_emitters` becomes a re-export** of the
  process-bigraph path (behavior-preserving; shim contract unchanged), so
  pbg-superpowers and the dashboard keep working with no source change. Idempotency
  test: a spec built via both paths gets exactly one sink, never two.
- Tests: a `CompositeSpec` with a declared emitter, built via `to_composite`,
  produces a composite whose emitter Step is present and gathers results
  (extend `tests/test_composite_spec.py`).

### A2. Analyses: register the hook, defer execution to the Study spec

`CompositeSpec.analyses` (composite_spec.py:123) has no runner in process-bigraph.
**Do not** build the full flush engine here — that is the Study two-phase composite
(Layer 2b: emitter barrier → extractor → flush DAG). Scope for Part A:

- Validate/normalize `analyses` (and `visualizations`) declarations at spec load
  (shape check only), exactly as `emitters` are validated.
- Expose them on the built document so Layer 2b's flush assembler can consume them
  (`document["analyses"]`, `document["visualizations"]`). No execution here.

This keeps Part A small and avoids pre-empting the Study spec's flush design.

### A3. Consolidate discovery (retire the legacy fallback)

Two discovery paths coexist: the **entry-point group**
`process_bigraph.spec_generators` (composite_spec.py:333, the intended mechanism)
and a **legacy distribution-walk fallback** (`_legacy_discover_generators_fallback`
importing `pbg_superpowers.composite_generator.discover_generators`,
composite_spec.py:395), plus the heavy `_import_bigraph_packages` walk still in the
shim (composite_generator.py:471–570).

- Make the entry-point group authoritative; the viva shim already declares it
  (`pbg-superpowers/pyproject.toml:75`).
- **Deprecate** the legacy fallback (warn on use), gated behind a env/flag, and
  delete once every workspace declares the entry point (tracked, not done here).
- Fold the distribution-walk into a single discovery implementation callable from
  both the entry point and any explicit `discover_specs(workspace)` call.

### A4. Fold `GeneratorEntry` (optional, cross-repo aware)

The shim keeps a parallel `GeneratorEntry` dataclass (composite_generator.py:23)
mirroring `CompositeSpec` fields purely for the dashboard's attribute surface. Add
the missing accessors to `CompositeSpec` (or a `.as_entry()` view) so the dashboard
can read `CompositeSpec` directly; then reduce `GeneratorEntry` to a deprecated
alias. **Cross-repo:** the dashboard (vivarium-workbench) consumes this surface — do
this behind the shim so the dashboard keeps working, and finish the dashboard side
(2026-06-28 plan Tasks 9–11) separately. Lowest priority in Part A.

---

## PART B — Lower `CompositeSpec` onto the bigraph-schema template (after Layer 1)

The umbrella's open question ("does `CompositeSpec` become a thin adapter over the
bigraph-schema template, or is it refactored onto it?") is decided here: **refactor
its parameterization onto the template primitive; keep the `CompositeSpec` surface
as the ergonomic authoring API.**

### B1. The mapping

| CompositeSpec today | Template (Layer 1) |
|---|---|
| `parameters: {name: {type, default, ...}}` + `${name}` substitution | **value slots** (sorted `Site`s of a value type) bound by `bind` |
| a parameter that names a process/composite to drop in | a **process slot** (a sorted `Site` whose sort's formation is interface conformance) |
| `default_n_steps`, counts that scale structure | **cardinality slots** (generative, `tensor`-expanded) |
| `builder`-kind generator (Python fn building state) | a template whose `_body` carries sorted `Site`s; the builder becomes body-construction, binding replaces `**params` |
| `to_document(overrides)` | `core.bind(template, bindings)` → `(schema, state)` |

`substitute_parameters` / `_resolve_value` / `_cast` (composite_spec.py:50–98) —
the bespoke `${name}` engine — is exactly the ~30% that the template's
`bind`/`compose_at` replaces. `_cast`/`normalize_type` become slot sort/type
coercion (bigraph-schema `core.check`/`resolve`).

### B2. The refactor — `${name}` *is* a site

- Add `CompositeSpec.to_template(core)`: replace each **full**-placeholder `${name}`
  occurrence with `Site(_sort=<declared param type>)` **keyed by the parameter name**
  — so value and structural filling are literally the *same* call and the
  "two-mechanism seam" never exists. `_cast`/`normalize_type` (`composite_spec.py:34,47`)
  become the site's sort (`core.check`).
- Re-express `to_document(overrides, core)` as
  `core.fill_sites(self.to_template(core), self._merged_params(overrides))` then
  `core.fill(defaults)` — same signature, same return. **Delete
  `substitute_parameters`/`_resolve_value`/`_cast` (`composite_spec.py:47-95`).**
  (Register is `core.fill_sites`, not `core.bind` — the latter already exists.)
- **Stated limitation:** *inline* interpolation (`"pre_${n}_post"`,
  `_INLINE_PLACEHOLDER`, `composite_spec.py:44`) is string concatenation, not
  substitution, and does **not** lower to a site. The golden corpus (B3/§4) must
  first report how many on-disk specs use it; then either keep a one-function
  string-interp pass for that case (and say so) or restrict authoring to
  full-placeholder form.
- Keep `@composite_spec` / `from_file` / registry / the shim **unchanged on the
  surface** — only the *resolution* mechanism swaps underneath.

### B3. Back-compat

- **Behavior-preserving for `spec`-kind** (static `schema`/`state` + `${name}`):
  every current substitution must produce byte-identical documents. Guard with a
  golden-corpus test (all `*.composite.{yaml,json}` + the 13 v2ecoli generators,
  render-and-compare old vs. new).
- `builder`-kind generators keep working; their Python builder now returns a body
  with `Site`s (or, transitionally, the same dict — a template with zero slots is
  just a document, so unconverted builders still run).
- The `${name}` string syntax stays supported as **sugar that lowers to value
  slots** during `to_template`, so on-disk specs need no rewrite.

---

## 3. Key contracts

1. **Emitter presence (A1).** After `to_composite`, a spec with declared emitters
   yields a composite whose emitter Step(s) are wired and `gather_emitter_results`
   returns data — via the process-bigraph API, no shim.
2. **Discovery single-source (A3).** `discover_specs` and the entry-point group
   resolve the *same* set; the legacy fallback only fires (with a warning) when no
   entry point is present.
3. **Template subsumption (B).** Every `CompositeSpec` lowers to a template such
   that `bind(to_template(spec), overrides)` equals today's `to_document(overrides)`
   for all existing specs (golden corpus). No `CompositeSpec` capability is lost;
   process slots + cardinality are strictly added power.
4. **Surface stability.** `@composite_spec`, `@composite_generator` (shim),
   `from_file`, `to_document`, `to_composite`, `_REGISTRY` view, and the
   entry-point group keep their signatures across A and B.

---

## 4. Tests

- Extend `tests/test_composite_spec.py`: emitter installed via `to_composite`
  (A1); analyses/visualizations validated + surfaced on the document (A2); single
  discovery source + legacy-fallback deprecation warning (A3).
- **Golden corpus is a precondition, not a guard (C4).** Build it — every
  `*.composite.{yaml,json}` fixture + the 13 v2ecoli generators, rendered to
  documents and frozen against *today's* code — and land it as its own commit
  **before** touching resolution. Part B is not started until the corpus is green.
  Then assert `fill_sites(to_template(spec), params)` == legacy
  `to_document(spec, params)` for every entry (byte-identity for `spec`-kind).
- Template mapping unit tests (B): a `${name}` value → value slot; a process
  parameter → process slot rejecting a non-conforming filler; a count → cardinality
  expansion (reuse the Layer-1 conformance/cardinality tests as the substrate).

---

## 5. Migration / compatibility

- **Part A is additive and behavior-preserving** except that composites built via
  the process-bigraph API now (correctly) emit — call that out in release notes.
- **Part B swaps resolution internals only**; on-disk specs and all decorators are
  unchanged. Requires a bigraph-schema version with `bind`/`compose_at` (Layer 1)
  — pin it.
- The shim (`pbg-superpowers`) and dashboard (`vivarium-workbench`) keep working;
  their coordinated changes (fold `GeneratorEntry`, dashboard Tasks 9–11) are
  tracked separately, gated behind the stable shim surface.

---

## 6. Risks

- **Emitter double-install (A1).** The shim currently installs; once process-bigraph
  installs too, ensure the shim's `install_default_emitters` is a *re-export* (one
  implementation) so a spec isn't given two sinks. Idempotency test.
- **Byte-identity for B3.** `${name}` casting edge cases (numeric coercion,
  nested/JSON values) must match under slot binding; the golden corpus is
  load-bearing — build it before touching resolution.
- **Cross-repo choreography.** bigraph-schema (Layer 1) → process-bigraph (this) →
  pbg-superpowers shim → vivarium-workbench dashboard. Each step must state its
  minimum upstream version; do A before B (A has no upstream dep).
- **Scope creep into the Study composite.** Resist building the analyses/flush
  runner here (A2) — it belongs to Layer 2b.

---

## 7. Sequencing

1. **A0** (`Composite.__init__` enforces `is_ground`) — cheap; makes the contract a
   guarantee and gating-by-groundness safe.
2. **A1** (emitter install, reusing `emitter_from_wires`/`add_emitter_to_composite`)
   — the real bug.
3. **A3** (discovery consolidation) + **A2** (validate/surface analyses).
4. **A4** (fold `GeneratorEntry`) — optional, cross-repo, lowest priority.
5. **Land Layer 1** (`fill` + `is_ground` in bigraph-schema) + build the **golden
   corpus** commit.
6. **B** — `to_template` (`${name}` → `Site`) + re-express `to_document` on
   `fill_sites`; delete the substitution engine; guarded by the corpus.

---

## 8. Out of scope

- The **Study two-phase composite** (emitter barrier, extractor substep, flush DAG)
  — Layer 2b, its own spec.
- Removing the `viva_superpowers` shim (permanently out of scope per 2026-06-28).
- Dashboard-side work (vivarium-workbench Tasks 9–11).
- A new emitter format or a new runtime composite type.
