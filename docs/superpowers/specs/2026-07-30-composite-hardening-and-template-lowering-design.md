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

## PART B — SUPERSEDED: `${name}` does NOT lower onto a site (proven by the corpus)

> **Corrected 2026-07-31.** The original plan — refactor `CompositeSpec.to_document`
> to `fill_sites(to_template, overrides)`, deleting `substitute_parameters` — was
> **disproven by the frozen corpus** (`feat/composite-template-lowering`, commit
> `8aa97b7`). Do **not** implement it. `substitute_parameters` **stays**.

**What the corpus showed.** v2ecoli's 29 registered generators are **builder-kind
with zero `${name}`** — nothing to lower. The real placeholder corpus is 68 static
`*.composite.{yaml,json}` (237 full + 2 inline placeholders). Of the 237:

- **73%** live inside a `config` blob the process constructor reads **raw** — a site
  would inject a schema node into an opaque value;
- **23%** are **edge fields** (`interval`, `priority`) — **not place-graph
  positions**, invisible to `collect_sites`;
- **4%** are actual store positions — the only case the site primitive fits.

And `fill_sites → core.fill` **realizes every edge** (tried to load a COPASI model
during document *construction*; crashed on another spec). Legacy `to_document`
instantiates nothing — which is why all 68 build cheaply. **Byte-identity is
unreachable.** Root mismatch: a **site** fills a place-graph position *in a schema*;
`${name}` fills a value *inside opaque config blobs and typed edge fields in a state
document*. Different things.

### B′ (the real, small win) — typed parameters, keep the dict walk

Give `${name}` params the one thing the lowering actually offered: **typed
validation**. At substitution, coerce/validate each value against its declared type
via `core.check` (replacing `_cast`'s silent coercion), rejecting a mistyped override
with a clear error. `to_document` stays a pure dict walk; `substitute_parameters`
stays; byte-identity is free. This is Part B in full.

### The real template layer (Option 3) — a SEPARATE spec, not a `CompositeSpec` refactor

"Templates for studies and investigations" — and the flagship **comparison-harness**
(value sites + a `CovertLab/vEcoli` **address site**, umbrella §Layer 2) — are the
site-based `fill`/`is_ground` machinery from **Layer 1, which already works**. They
are a **new authoring format** where a parameter *is* a first-class schema **site**
(value / address / model site), **not** a `${}` marker in a state blob. That format
+ a one-time migration is its own spec (`docs/.../study-investigation-template-*`),
built on Layer 1, with **no `${name}` involvement**. The legacy `${name}`
`CompositeSpec` is kept (with B′ typing) and coexists.

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
