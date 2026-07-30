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

### A1. Emitter materialization belongs in process-bigraph (a real bug)

`CompositeSpec.emitters` is a first-class field, but `to_composite()`
(composite_spec.py:218) builds `Composite(doc, core)` and **never installs
emitters** — emitter materialization lives only in the shim
(`viva_superpowers/composite_generator.py`: `install_default_emitters`,
`_emitter_node_from_decl`, `emitter_defaults`, `_validate_emitters`, ~284–411). So
a composite built via the pure process-bigraph API gets **no observation sink**,
while one built via the viva shim (`build_composite_from_spec`) does. This is a
correctness gap, not a style one.

**Fix:** move emitter materialization into process-bigraph and call it from
`to_composite`/`to_document`:

- New `process_bigraph/emitter_install.py` (or into `emitter.py`) housing
  `emitter_node(decl, core)` and `install_emitters(document, emitters, core)`,
  ported from the shim. process-bigraph already owns the emitter types
  (`emitter.py` — `Emitter(Step)`, RAM/JSON/Console; Parquet/SQLite/XArray via
  pbg-emitters), so this is the correct home.
- `to_document(...)` gains the declared emitter nodes in `state`; `to_composite`
  therefore yields a composite that emits. Add an `emit=True` flag if a
  caller wants the bare document.
- **The shim's `install_default_emitters` becomes a re-export** of the
  process-bigraph function (behavior-preserving; the shim contract is unchanged),
  so pbg-superpowers and the dashboard keep working with no source change.
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

### B2. The refactor

- Add `CompositeSpec.to_template(core) -> template_document`: compile the spec's
  `schema`/`state` + `parameters` into a bigraph-schema **template** — value
  params → value slots at their `${name}` positions; declared model/process params
  → process slots with an interface sort; scalar counts → cardinality slots.
- Re-express `to_document(overrides, core)` as
  `core.bind(self.to_template(core), self._merged_params(overrides))` — same
  signature, same return, substitution engine gone.
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
- New golden-corpus test (B3): for every `*.composite.{yaml,json}` fixture and the
  v2ecoli generators, assert `bind(to_template(spec), params)` == legacy
  `to_document(spec, params)`.
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

1. **A1** (emitter install) — the real bug; ship first, no dependencies.
2. **A3** (discovery consolidation) + **A2** (validate/surface analyses).
3. **A4** (fold `GeneratorEntry`) — optional, cross-repo, lowest priority.
4. **Land Layer 1** (`template`/`slot` in bigraph-schema).
5. **B** — `to_template` + re-express `to_document` on `bind`, guarded by the
   golden corpus.

---

## 8. Out of scope

- The **Study two-phase composite** (emitter barrier, extractor substep, flush DAG)
  — Layer 2b, its own spec.
- Removing the `viva_superpowers` shim (permanently out of scope per 2026-06-28).
- Dashboard-side work (vivarium-workbench Tasks 9–11).
- A new emitter format or a new runtime composite type.
