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

`fill(template, bindings)` → a ground document; `is_ground` is the runnable
predicate; a partially-filled template is still a template.

> **Use `fill_sites → render → Composite`, not `assembly.build`.** The PoC found
> `build()` finishes with `core.fill`, which realizes every edge and **crashes on a
> process's `config`** (`default(0.5)` — a raw value where `default` expects a
> schema; `default_link` still does `default(schema.config)`). This is the same
> materialization gap 1.4.4 closed for `address`/`inputs`/`outputs` but **not**
> `config`, so `build()` is unusable for any realistic pbg document. `Composite`
> realizes the document itself, so the native path skips the broken step. **A
> one-line bigraph-schema fix to `default_link` should make `build()` usable** — a
> worthwhile upstream follow-up.

---

## 3. A study is a template — model, emitter, AND flush entities are all sites

A study is a **higher-order DAG**. The simulation is *one node*
(`local:SimulationStep`, process-bigraph ≥ the #160 merge) and the flush entities are
ordinary steps downstream of that node's **`results`** output — the emitter's durable
handle (`EmitterResults`: a reference, resolved on demand, never the bulk).

That shape is what makes a flush entity fire **exactly once**. Its `results` input is
unsatisfied until the simulation node's update returns, and the simulation's per-tick
stepping happens *inside* the node rather than beside it — so the flush steps are
never siblings of the simulation's own steps. Ordinary producer/consumer ordering does
all of it: no completion phase, no marker, no scheduler special case.

Everything configurable is a **site**; only the wiring is fixed. A study template's
holes:

```
study.template:
  threshold/media/n_seeds: <value & cardinality sites>     # study-level config
  sim:                                                     # the simulation, ONE node
    _type: step
    address: local:SimulationStep
    config:
      runtime: <float>
      state:                                               # the simulation it runs
        model:   <site: face = the study's expected model interface>
        emitter:                                           # the sink
          address: <site: sort = "an emitter">             # RAM | Parquet | XArray-zarr
          outputs: {results: [results]}                    # fixed — the handle leaves here
    outputs: {results: [results]}                          # fixed — and leaves the node
  # the flush entities — CARDINALITY REGIONS of face-sorted sites (0+ each),
  # every one wired to `results` via an ascending wire:
  visualizations: [ <site: face = Visualization(results→figure)> … ]
  analyses:       [ <site: face = Analysis(results→analysis)>     … ]
  report_cards:   [ <site: face = ReportCard(results→verdict)>    … ]
```

- **Model & emitter sites live inside the simulation node's inner state.** Filling
  them configures *the simulation the study runs*. Sites nested in a node's `config`
  are discoverable and fillable like any other (addressed by path, e.g.
  `study/sim/config/state/model`).
- **Emitter site.** Filling it chooses the sink. Interchangeability is expressed by a
  **registered sort** (`core.register_sort`), not by face conformance: every emitter
  exposes the same `results` port and nothing else, so a face check cannot tell an
  emitter from anything else that happens to expose `results`. The sort states the
  constraint directly — *the address must name a registered `Emitter`* — and refuses
  e.g. `local:IncreaseProcess`.
- **Flush-entity site-regions.** `visualizations`/`analyses`/`report_cards` are
  cardinality regions (Layer-1 §4.5): fill with 0+ conforming Steps. Each sits inside
  its own instance region, so it reaches the shared handle with an **ascending wire**
  (`['..', 'results']`). A report-card site accepts only a Step whose face reads
  `results` and writes a verdict — so you can't wire a viz where a card belongs. Zero
  entities is valid: a study that reports nothing is still a study.
- **What a flush entity observes.** The *resolved trajectory* of a finished run, not
  an instantaneous store value. A report card wired to `results` judges the run's
  final state — something no per-tick step could do — and its firing count is
  independent of how long the simulation ran.

`fill({model: ecoli_baseline, emitter: XArrayEmitter, media: minimal, n_seeds: 8,
visualizations: [mass_trace, growth_curve], analyses: [doubling_time],
report_cards: [mass_conservation, division_timing]})` → a **fully-specified, ground,
runnable study**, no code. Every knob — the model, the sink, and *which* figures /
analyses / verdicts run — is a `fill`.

## 4. An investigation is a template with a site per study; gating = *staged* filling

An **investigation template** has one **study site per member** + value/address
sites. Gating is **conditional filling, staged** (validated in the PoC, `f908bae`):
a gate is a **real edge** (a report-card step) whose verdict decides the **bindings
of the next stage** — on `pass` it emits the downstream study's filler; on `fail` it
does not. An unfilled member is then **pruned** (`prune_open_regions`) and is
**absent** from the constructed document — the strongest form of "never run": the
engine never *decides* not to run something, because what wasn't filled isn't there.

| upstream verdict | blocked | built & run |
|---|---|---|
| `pass` | — | `study_A`, `study_B` |
| `fail` | `study_B` | `study_A` |

**Why staged, not single-run (a real finding, not a compromise).** A site is a hole
in a *schema*, and schemas are consumed at **construction**; the step network runs
**after**. And an open site is **not local** — A0 rejects the *whole* document if any
required site is open (`'investigation/study_B/model' unfilled` → nothing builds). So
a gate edge *cannot* fill a site inside a composite that could never be constructed.
The gate therefore runs **between builds** (verdict → next-stage bindings → build the
now-ground remainder), and unfilled members are pruned before construction. This
preserves contract #4 exactly, with **one** filling mechanism.

*Literal single-run gating (insert a subtree mid-`run()`) is a different operation —
process-bigraph's runtime **structural updates** (`_add`/`_remove`/`_divide`, used for
cell division), not sites. If ever wanted, spec it as that; do **not** conflate it
with filling (the same trap the `${name}`-lowering attempt fell into).*

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

### 6.1 The Layer-4 authoring & fill loop (workbench)

A template is authored and run entirely through the ProcessCard viewer — no code —
because a template *is* a document and the viewer already renders documents:

1. **Author** a study/investigation template: drop a composite into the study's
   **model** face-site, mark the emitter / viz / analysis / report-card holes as
   **sites** (the viewer's config/inputs/outputs regions gain an "open site" affordance
   — a hole with its sort/contract shown, not a value). Saved as `*.template.yaml`
   (sites round-trip via Layer 1).
2. **Fill** it: the viewer's config panel binds each site — a **value** site is a
   typed field; a **face** site (model / viz / report-card) offers the *conforming*
   registry entries (`admits` filters the picker); an **address** site (the vEcoli
   repo) is a text/ref field the `git:` protocol resolves; a **cardinality** site is
   a count. **Apply** = `fill_sites` + re-render (the same Apply already in the card).
3. **Run** it: a fully-filled (ground) template's Run is live; an unfilled required
   site keeps the card in the **unbound** state and disables Run — the *same*
   `is_ground` predicate that gates the engine gates the button. One predicate, one
   picture (viewer handoff §D4).
4. **A gated investigation** renders its blocked members as *pruned/absent* (staged
   filling, §4) — the DAG the viewer draws is the ground remainder, so "what didn't
   fill isn't shown" matches "what didn't fill didn't run."

So the viewer's four regions (config · inputs · contract · outputs) + the pull-down
loom render a *template* as naturally as a composite: a site is just a region whose
value is a **hole with a contract** instead of a value. No new UI concept — the
"unbound" state + `admits`-filtered pickers are the whole Layer-4 surface.

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
