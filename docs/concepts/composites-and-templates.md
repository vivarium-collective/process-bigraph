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
  "interval": 1.0                           // seconds; or a unit string, e.g. "1 min"
}
```

An **address** is `protocol:path`. `local:` means "resolve this name in the local
registry (`core.link_registry`), which `allocate_core()` populated by walking the
installed packages"; `DynamicFBA` is the registered class name.

Two more structural facts: a composite can **nest** — a node can itself be a
sub-composite, so big models are built from small ones (containment) — and time is
**per-process** (`interval`), so fast and slow processes advance on one shared state.

#### Timestep units

`interval` is optional (default `1.0`) and expressed in the canonical time base,
**seconds**. You may also declare it as a pint time string — `"1 min"`,
`"2 minutes"`, `"30 s"` — which is converted to seconds when the composite is
realized, so the scheduler always counts in seconds and all the stash-model rules
above are unaffected. The declared unit is preserved (as `interval_units`) for
faithful re-serialization and for display in the composite viewer (e.g. "· every
1 min", with "60 s" on hover). A bare number has no declared unit; a non-time or
unparseable string raises an error rather than being silently dropped.

### Update timing with unequal intervals (the stash model)

When processes have **different** intervals, *when* a process reads its inputs
matters, and the implemented contract differs from the idealized description in
the paper. This is the behavior `Composite.run()` actually gives:

- **Read at the start of the step, apply at the end.** When a process becomes
  due at time `t0`, the scheduler reads its input ports from the shared state
  **at `t0`**, computes its update immediately, and holds (*stashes*) the result
  until `global_time` reaches `t0 + interval`, at which point the delta is
  applied. So a process with interval `Δ` sees a fixed snapshot of its inputs
  for the whole step `[t0, t0 + Δ)` and writes the result at the end. This is
  deliberate: every process due at the same instant sees the same pre-update
  state, so the tick has no hidden intra-tick ordering dependence.

- **Why this can differ from "read at the event time".** The paper's formal
  "Process Update" (Supplement 1, §3.8.2) reads a process's inputs at the event
  time `t*` at which its update is *due* — i.e. at the **end** of the interval.
  The two rules **coincide when every process has the same interval** (all
  processes read and write on the same ticks), and **diverge when intervals are
  unequal**: a slow process applies at `t*` a delta computed from its inputs as
  they stood one interval earlier, so it effectively lags a faster neighbor by
  up to one of its own intervals. The implementation follows the *stash model*
  (Supplement 1, §3.8.6), i.e. read-at-start, not the §3.8.2 read-at-`t*`
  idealization; the two Supplement sections describe the same system at
  different levels of detail, and §3.8.6 is the one the code realizes.

  Worked example (deltas returned by each process):
  `x' = x + 0.5·(y − x)` on `p1` with interval 1, `y' = y + 0.25·(x − y)` on
  `p2` with interval 2, starting `x = 1, y = 0`. At `t = 2`, `p2`'s first update
  uses the inputs it read at `t = 0` (`x = 1, y = 0`), giving `y = 0.25` — not
  the `y = 0.125` you would get by reading `x` as it stands at `t = 2`.

- **Chunking caveat.** The read-at-start guarantee holds for a process whose
  **entire next interval fits inside the current `run(interval)` call**. If a
  process's next due time would fall *after* the end of the current `run()`
  window, that process is not scheduled within this call; it is re-evaluated at
  the start of the *next* `run()` call and reads its inputs then. As a result,
  a span computed as a single `run(T)` can differ from the same span computed as
  several shorter `run()` calls when some process interval is larger than a
  chunk. **For chunking-invariant results, call `run()` with an interval at
  least as large as the largest process interval, or keep all intervals equal.**
  (The per-process state stash that §3.8.6 describes to make long intervals
  fully chunking-invariant is only partially realized today — the read half
  exists, the cross-`run()` carry-over does not — so this caveat is a known
  limitation rather than a guarantee.)

### Static vs generator composites

A composite comes in two forms:

- **static** — an inline `state` document (the `.composite.json` above).
- **generator** — a `@composite_generator` (`composite_generator.py`) / `CompositeSpec`
  (`composite_spec.py`) function that *builds* the state (e.g. from a cache),
  parameterized by its arguments. `CompositeSpec` also supports `${name}` string
  placeholders substituted at build.

Both resolve to the same thing the engine runs: a `Composite` (`composite.py`).

### Processes and holes (drafts)

A **Process** is a class with typed `inputs()`/`outputs()` and an `update()`. But you
don't always have the mechanism yet — and when you don't, you leave the slot as a
**hole** rather than inventing a node for it.

A hole is an open **site** (§2) whose `_sort` names a `ProcessContract` — its typed face
(ports with types, units, bounds) plus any conditions. It *is* the draft: the interface
is declared and reviewable, the mechanism is simply absent until the hole is filled. A
real process fills the hole when its own contract *admits* the hole's — the same
filling discipline (`admits` / `fill_sites`) that fills any site.

```python
from bigraph_schema.contract import ProcessContract

# register the intended interface as a named contract …
core.register_contract("PTHSecretion", ProcessContract(
    summary="senses serum Ca, secretes PTH",
    face={"inputs": {"ca_sense": "float"}, "outputs": {"pth_out": "float"}}))

# … and leave the slot as a hole sorted by it
document = {"parathyroid": {"secretion": {"_type": "site", "_sort": "PTHSecretion"}}}
```

**How it fits in.** A composite with an open required hole is **not ground** — it will
not build. `Composite` rejects it with an `UnfilledHoleError` that names the hole, the
contract it requires, and the registered processes `find_candidates` reports as
admissible — an actionable report, not an opaque `KeyError`. Fill the hole with an
admissible process (`fill_sites` / `templates.fill_template`) and the document becomes
ground and runs. The whole lifecycle is walked in
[designing-with-holes.md](designing-with-holes.md).

> **Deprecated — `DraftProcess` / `@draft_process`.** The older model stubbed a mechanism
> as a *present but inert* node (a real Process subclass with a no-op `update` and a
> free-form `contract` dict). It still works but now emits `DraftProcessDeprecationWarning`;
> prefer a hole. `draft_to_contract(cls)` turns an existing draft's ports + free-form
> contract into a real `ProcessContract` you can register and sort a hole by. The one
> affordance it had that a hole lacks — a composite that *runs* (inertly) with the stub in
> place — is deliberately dropped: an unimplemented mechanism should block the build and
> say so, not run as if it were there.

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
