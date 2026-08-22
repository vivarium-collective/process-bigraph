# Composites and Templates

The process-bigraph substrate: what a **composite** is, what a **template** (a composite
document with open **sites**/holes) looks like, and how a template becomes a runnable,
*ground* composite.

This is the formalism layer. For how research metadata sits *on top* of it — studies,
investigations, the dashboard — see viva-superpowers
`docs/concepts/composites-templates-and-the-study-investigation-stack.md`, which
references this document.

---

## 1. The composite

A composite is a **process-bigraph document**: one `state` map whose entries are
**typed nodes**. There are two kinds of node, and a composite is nothing but these two
kinds wired together.

- **Processes** — the moving parts. A node with `_type: process` (or `_type: step`)
  names an `address` (which Process/Step class), a `config`, an `interval`/schedule,
  and an `update` that advances state each step. This is where computation happens.
- **Stores** — the state. A store holds typed values (a concentration field, a molecule
  count, the clock). Processes never talk to each other directly; they read and write
  **shared stores**, and that wiring *is* the coupling.

### Anatomy

Each process node's `inputs`/`outputs` are **paths into shared stores**. Change a path
and you rewire the model; two processes that name the same store path are now coupled.
Nothing else connects them.

```jsonc
"ecoli core dFBA": {
  "_type":    "process",
  "address":  "local:DynamicFBA",           // which Process class, resolved in the registry
  "config":   { "model_file": "textbook", "kinetic_params": { /* … */ } },
  "inputs":   { "substrates": { "glucose": ["fields", "glucose"] },   // ← wiring path
                "biomass": ["fields", "ecoli core"] },                //   into a shared store
  "outputs":  { "substrates": { "glucose": ["fields", "glucose"] }, /* … */ },
  "interval": 1.0
}
```

An **address** is `protocol:path`. `local:` means "resolve this name in the local
registry (`core.link_registry`), which `allocate_core()` populated by walking the
installed packages"; `DynamicFBA` is the registered class name.

Two more structural facts: a composite can **nest** — a node can itself be a
sub-composite, so big models are built from small ones (containment) — and time is
**per-process** (`interval`), so fast and slow processes advance on one shared state.

### Static vs generator composites

A composite comes in two forms:

- **static** — an inline `state` document (the `.composite.json` above).
- **generator** — a `@composite_generator` (`composite_generator.py`) / `CompositeSpec`
  (`composite_spec.py`) function that *builds* the state (e.g. from a cache),
  parameterized by its arguments. `CompositeSpec` also supports `${name}` string
  placeholders substituted at build.

Both resolve to the same thing the engine runs: a `Composite` (`composite.py`).

### Processes and draft processes

A **Process** is a class with typed `inputs()`/`outputs()` and an `update()`. But you
don't always have the mechanism yet — that's what a **draft process** is for.

A `DraftProcess` (`draft_process.py`, ≥ 1.8.3 —
`from process_bigraph import DraftProcess, draft_process`) declares a **contract**:
input/output ports plus a human-readable description of the transformation it is *meant*
to perform — but carries **no `update` dynamics**. It inherits the base no-op update, so
if stepped it stays inert and never fabricates behavior. That lets you drop a node into a
model's topology, wire it to stores, and review the whole thing *before* committing the
mechanism.

```python
from process_bigraph import DraftProcess, draft_process

@draft_process(name="PTH secretion",
    inputs={"ca_sense": "float"}, outputs={"pth_out": "float"},
    contract={"summary": "...", "senses": "...", "makes": "..."})
class PTHSecretion(DraftProcess):
    pass
```

**How it fits in.** A `DraftProcess` is a real Process subclass, so it is a first-class
node in a composite — wired to stores through `inputs`/`outputs` like any process. Two
consequences:

- Because a workspace's `build_core()` walks the package and registers every Process, a
  module-scope draft **auto-registers** and shows in the dashboard (Modules → Processes)
  marked DRAFT, with its ports and contract. Its ports/contract are class-level
  attributes, introspectable **uninitialized** (discovery calls `inputs()`/`outputs()`
  on the class and `describe()` on `cls.__new__(cls)`), so it appears without ever being
  stepped.
- A composite containing only real + draft nodes is still **ground** — it *runs*, and
  the draft simply no-ops. Contrast a **site** (§2): a document with an open site is *not*
  ground and won't run at all.

That gives two complementary placeholders at different stages of a build:

| | What it is | Is the doc runnable? | Use it to… |
|---|---|---|---|
| **Site** (`_type: site`) | an empty *hole* — no node yet | **No** — ungrounded until filled | leave a slot in a *template* for a whole composite/process/value to be plugged in later |
| **Draft process** | a *present but inert* node (ports + contract, no `update`) | **Yes** — runs, the node no-ops | stub a specific mechanism *in place* so the topology is complete and reviewable before the biology is written |

Typical workflow: design the model topology, drop a `DraftProcess` where a mechanism
will go (so the wiring and contract are reviewable and the model already runs), then
replace the draft with a real Process once the mechanism is committed.

---

## 2. Templates: composite documents with holes (sites)

Every composite in §1 is **ground**: every node concrete, ready to run. A **template**
is a composite document that is *not* ground — it has open **sites**.

A site is a **place-graph hole**, written `{"_type": "site", "_sort": <face>}`: a slot
where a whole composite, process, or value plugs in. This is Milner's bigraph site
(Def. 2.1 in *The Space and Motion of Communicating Agents*), implemented in
bigraph-schema:

- `bigraph_schema/schema.py` — the `Site` type (a subclass of `Empty`; carries an
  optional `_sort` place-sort label). "A schema that contains `Site`s is ungrounded: it
  describes a *context* into which another bigraph can be placed, not a runnable state
  tree."
- `bigraph_schema/assembly.py` — the bigraph algebra: `interfaces(schema)` derives the
  inner/outer faces by walking for sites and unwired link ports; `compose(outer, inner)`
  substitutes `inner`'s roots into `outer`'s sites (Milner Def. 2.5); `tensor(a, b)` is
  the side-by-side merge; `fill_sites(core, doc, bindings)` plugs fillers into sites by
  path.

`Composite` **refuses to run a document with any open required site** (`composite.py`,
`_reject_ungrounded`): *"composite document is not ground — required site(s) left
unfilled … an open site is a hole where a process should be."*

### The `process_bigraph.templates` module

`process_bigraph/templates.py` builds and fills templates on the native path
(`fill_sites` → `render` → `Composite`, rather than `assembly.build`, because
`Composite` does its own realization). Key entry points:

| Function | What it does |
|---|---|
| `load_template(path, core)` / `save_template(t, path, core)` | Read/write a `*.template.{yaml,json}` document; `core.access` reconstitutes each site as an open place-graph hole (type/face/`_default` survive the round-trip). |
| `open_sites(document)` | The paths of every site still open. |
| `required_open_sites(document)` | Open sites with no `_default` — the ones that block a build. |
| `is_ground_document(document)` | True when nothing is left to fill — the runnable predicate. |
| `fill_template(core, template, bindings)` | Fill some sites; unbound sites stay open, so filling is incremental. |
| `template_document(core, template, bindings)` | Fill + render a runnable document; **raises naming any required site left unfilled**. |
| `prune_open_regions(document, member_depth)` | Drop every region that still contains an open site; return `(pruned, blocked)`. |
| `investigation_document(core, template, bindings, member_depth)` | Fill + prune: fill a member's site to admit it, leave it open to drop it. |

### What a template looks like

A site is `{"_type": "site", "_sort": <face>}`. A template wires sites into a fixed
network. This is a real study-shaped template (from `tests.py`) — the analysis/emitter/
report-card network is fixed, and the model is left as a hole:

```jsonc
// analysis/emitter network fixed, the model is a hole
{ "study": {
    "threshold": { "_type": "site", "_sort": "float" },          // a VALUE hole
    "sim": { "_type": "step", "address": "local:SimulationStep",
      "config": { "state": {
        "model":   { "_type": "site", "_sort": MODEL_FACE },     // ← plug a COMPOSITE in here
        "emitter": { "address": { "_type": "site", "_sort": "emitter" }, /* … */ } } } },
    "report_cards": {
      "card": { "_type": "site", "_sort": CARD_FACE } } } }      // plug a REPORT CARD in here

open_sites(template) → [ study/threshold,
                         study/sim/config/state/model,           // the model hole
                         study/sim/config/state/emitter/address,
                         study/report_cards/card ]
```

The `_sort` **types the hole** — what may plug in (a model composite, an emitter, a
card, a bare `float`). Filling is `fill_sites(core, template, bindings)`: plug a filler
into a site by path and the hole is gone — "once a site is filled there is no site
anymore." Fill every required site → the document is **ground** → it runs.

### Two template shapes

- **A model-as-a-site template** fixes a network (analysis, emitters, report cards) and
  leaves the model as a single site. Fill the model hole with a composite → ground →
  runnable. Use `template_document`.
- **A one-site-per-member template** has one site per member region (e.g. one per study
  in an investigation). Filling a member's site admits it; leaving it open makes
  `investigation_document` **prune** it. Gating is expressed as *filling*, not
  scheduling: what isn't filled isn't in the built document, so the engine never has to
  decide "don't run this."

### Three things that all look like "blanks" — keep them apart

- **A site** (`_type: site`) is an *empty structural hole* — no node there yet; you plug
  a whole composite/process/value in. **This is the template mechanism.**
- **A draft process** is a node that *is* there but inert (ports + contract, no
  `update`) — a placeholder *node*, not a hole.
- **`config` / `${name}` / a study's `params`** fills a node's *parameters* — not a hole
  and not a node.

So a template is specialized two ways: **fill its sites** (structure) and **set its
parameters** (values).

---

## 3. Where the code lives

### bigraph-schema — the type system + the bigraph algebra

| File | What it is |
|---|---|
| `bigraph_schema/schema.py` | Type definitions incl. the `Site` type (place-graph hole) and the Milner primitives (`Link`, `Interface`, inner/outer names). |
| `bigraph_schema/assembly.py` | `interfaces`, `compose`, `tensor`, `fill_sites`, and the elementary bigraph constructors (`barren`, `merge`, `ion`, `substitution`, `closure`). |
| `bigraph_schema/core.py` | The `TypeSystem` — `access`, `render`, resolution. |
| `bigraph_schema/methods/apply.py` | The store-write law — how a typed update is applied to state. |

### process-bigraph — the engine + composite / template primitives

| File | What it is |
|---|---|
| `process_bigraph/composite.py` | The `Composite` class + `allocate_core()`; the ground-check that rejects unfilled sites. |
| `process_bigraph/templates.py` | `open_sites`, `fill_template`, `template_document`, `investigation_document`, `prune_open_regions`, `load_template`/`save_template`. |
| `process_bigraph/composite_spec.py` | `CompositeSpec` — the unified static/generator front-end; `${name}` placeholders. |
| `process_bigraph/composite_generator.py` | The `@composite_generator` decorator. |
| `process_bigraph/draft_process.py` | `DraftProcess` + `@draft_process`. |
| `process_bigraph/composite_discovery.py` · `scheduling.py` · `emitter.py` | Package discovery/registration; per-process interval scheduler; emitters. |

---

## References

- R. Milner, *The Space and Motion of Communicating Agents* (Cambridge, 2009) — sites
  (Def. 2.1), composition (Def. 2.5), tensor (Def. 2.7), elementary bigraphs
  (Defs. 3.1–3.5).
- viva-superpowers `docs/concepts/composites-templates-and-the-study-investigation-stack.md`
  — the study/investigation layer built on this substrate.
