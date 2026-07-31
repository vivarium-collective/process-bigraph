# Study & investigation templates — the site-based format (Option 3)

**Status:** Design (Layer 3 of the framework-unification stack)
**Date:** 2026-07-31
**Repos:** process-bigraph (study/investigation documents), bigraph-schema (the site primitive — shipped in 1.4.4), vivarium-workbench (authoring), v2ecoli (the comparison-harness proof)
**Umbrella:** `vivarium-workbench/docs/superpowers/specs/2026-07-29-framework-unification-design.md`

---

## 1. Why this exists (and what it is *not*)

The corpus disproved "make `CompositeSpec` a template by lowering `${name}` onto
sites" (Layer-2a Part B, superseded): a **site** fills a place-graph position *in a
schema*; `${name}` fills a value *inside opaque config blobs and edge fields in a
state document* — different things, and `fill_sites → core.fill` realizes every edge.
So the legacy `${name}` `CompositeSpec` is **kept** (with typed validation, Part B′).

**This spec is the real thing the user asked for: templates for studies and
investigations** — a **new, first-class, site-based format** where a parameter *is* a
schema **site**, built directly on the Layer-1 primitive (all shipped in
bigraph-schema 1.4.4): `fill_sites`, `is_ground`, `admits`, `contract`/`amend`, and
**address injection**. No `${name}`. It coexists with legacy specs.

---

## 2. The building blocks (Layer 1, already shipped)

A template is **a document that is not ground** (has open sites). Its site kinds:

- **value site** — a sorted `Site` of a value type (`float`/`string`/`enum[...]`);
  `admits` = `core.check`. This is a configurable config knob.
- **model / process site** — a `Site` whose sort is a **face** (`link[in,out]`);
  `admits` = the filler's `interface()` conforms. Drop in a conforming composite.
- **address site** — a process edge with a **fixed face** and its **`address` open**
  (Layer-1 §4.6, *a process definition without an implementation*). `fill` injects the
  address; `admits` checks the named process's face conforms.
- **cardinality site** — an `int` driving a `ReactionRule` replication (Layer-1 §4.5),
  e.g. `n_seeds`.

`fill(template, bindings)` → `build` returns a ground `(schema, state)`; `is_ground`
is the runnable predicate; a partially-filled template is still a template.

---

## 3. A study is a template with a model site

A **study template** fixes the analysis-flush sub-network (viz / analyses / report
cards, wired to the emitter's `results` port — umbrella Layer 1) and leaves the
**model** as a site:

```
study.template:
  model:        <site: face = the study's expected model interface>   # fill with a composite
  n_seeds:      <site: int>                                           # cardinality
  media:        <site: enum[...]>                                     # value
  flush:        [ viz_* , analysis_* , report_card_* ]               # fixed
  emitter:      <declared, results port>                             # fixed
```

`fill({model: ecoli_baseline, n_seeds: 8, media: minimal})` → a ground, runnable study
document. Any conforming registered composite drops into `model`.

## 4. An investigation is a template with a site per study; gating = filling

An **investigation template** has one **study site per member**, plus value/address
sites for what's configurable, and **gate edges** that fill downstream sites at
runtime (umbrella Layer 2 — gating is conditional filling): a gate emits `study_B`'s
filler on the upstream `pass`, leaves the site open on `fail`; an open site ⇒
non-ground ⇒ never built.

## 5. Flagship — the comparison-harness investigation template

The v2ecoli ↔ vEcoli comparison, as one document:

```
comparison.template:
  compared:                        # value sites — sweep these
    media:        <site: enum[...]>
    n_seeds:      <site: int>
    timepoints:   <site: list[float]>
    tolerances:   <site: map[float]>
  vecoli:                          # an ADDRESS site (abstract process)
    _type: process
    _inputs / _outputs: <fixed face — vEcoli's I/O contract>
    address:  <site: sort = "a whole-cell model process">    # default: git:CovertLab/vEcoli@main
  v2ecoli: <model site>            # the in-tree model
  compare: <fixed step: matched-timepoint nRMSE, report cards>
```

- **Compare against a different implementation:** `fill(vecoli.address,
  'git:CovertLab/vEcoli@<ref>')` — a fork, a branch, a pinned commit.
- **Sweep the compared configs:** `fill` the value sites.
- One document, no code. This proves **address injection end-to-end on a real
  cross-model harness** — the flagship the umbrella names.

### 5.1 New piece required — a git/remote protocol

The vEcoli `address` names an implementation fetched from a **repo**. Add a
**protocol** `git:<owner>/<repo>@<ref>` (alongside `local:`) that resolves a repo
address to a runnable process: clone/pin the ref, install into an isolated env,
expose vEcoli's entrypoint as an `Edge` whose `interface()` matches the site's face.
`admits` runs against that resolved face (fetch is cached; conformance is checked
before a run, not at document build). This is the one genuinely new capability.

---

## 6. Authoring format & coexistence

- A study/investigation template is a **document with sites** on disk
  (`*.template.{yaml,json}`), authored/rendered via the workbench. A site is written
  as `{_type: site, _sort: <value type | face | address-sort>, _default?: …}` at its
  position — `render`/`access` round-trip it (Layer-1 fixed the `_sort` round-trip).
- **Coexistence:** legacy `${name}` `CompositeSpec` is untouched and keeps working;
  the site format is **opt-in** for new studies/investigations. No mass migration is
  forced — a study authored the new way is a template, an old one stays a spec.
- The workbench's ProcessCard viewer already renders open sites as the "unbound"
  state (presentation side) — a template study/investigation renders for free.

---

## 7. Contracts

1. **A template is `not is_ground`.** `build(template, bindings)` → ground `(schema,
   state)`; the study/investigation runs iff ground.
2. **`admits` per site kind** — value (`core.check`), model/address (face
   conformance via the resolved `interface()`), cardinality (int/range).
3. **Address protocol contract** — `git:` resolves a `(repo, ref)` to an `Edge` whose
   face is checked before run; resolution is cached and pinnable for reproducibility.
4. **Gating = filling** — a failed prerequisite leaves a study site open; the region
   is non-ground and never built.
5. **No `${name}`** — this format never touches the legacy substitution engine.

---

## 8. Tests

- A study template with a model site + value + cardinality sites builds to a runnable
  study when filled with a conforming composite; a non-conforming model → fill error.
- The comparison-harness template: `fill` the vEcoli address with two refs → two
  distinct resolved faces both conform; a non-whole-cell address → rejected.
- Gating: an investigation template where an upstream `fail` leaves the dependent
  study site open → the dependent is not built (asserted via A0's open-site check).
- `git:` protocol: resolves a pinned ref deterministically; caches; a moved ref is
  re-resolved; conformance checked before run.

---

## 9. Sequencing & out of scope

**Sequencing:** (1) the `git:`/remote **protocol** (the new capability); (2) the
study & investigation template **format** + `build`; (3) the **comparison-harness**
template as the proof; (4) workbench authoring UI. Depends on: bigraph-schema ≥1.4.4
(shipped), process-bigraph Part A (A0 open-sites, merged path).

**Out of scope:** migrating the ~250 legacy studies (they stay `${name}` specs; new
ones opt in); rewriting `CompositeSpec`; the read-side renderers.
