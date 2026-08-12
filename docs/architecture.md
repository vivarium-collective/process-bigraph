# The framework, end to end

This is the map. It explains what the objects are, how they compose, and the
handful of laws everything else follows from. Read it before the per-topic
notes ([tick lifecycle](tick_lifecycle.md), [emitters](emitters.md),
[distributed lifecycles](distributed_lifecycles.md)), which assume it.

The framework spans two packages, and the split is not arbitrary:

| | package | owns |
|---|---|---|
| **Layer 1** | `bigraph-schema` | what a document *is*: types, schemas, places, filling |
| **Layer 2** | `process-bigraph` | what a document *does*: scheduling, running, results |

The dependency runs one way — `process-bigraph` imports `bigraph-schema`,
never the reverse. Anything that needs to know about *running* belongs in
Layer 2; anything true of a document at rest belongs in Layer 1.

---

## 1. The mental model: one object, one operation, one law

Almost everything here is an instance of a single idea.

**One object — the document.** A composite, a study, an investigation, a
template are all *the same kind of thing*: a nested dict describing a place
graph, some of whose positions may be **sites** (holes). There is no separate
"template type" or "investigation type". A template is just a document that
still has holes.

**One operation — fill.** `fill_sites(core, document, bindings)` puts
something into a hole. That is the only structural operation the framework
needs, and it is *incremental*: filling some holes leaves the rest open, so
the result of a fill is another document that may still be a template.

**One law — groundness.** A document is **ground** when no required site is
left open. `Composite` refuses to build a document that is not ground. That
single predicate is what makes everything below expressible:

```
document + holes                     = template
fill(template, bindings)             = document (maybe still a template)
is_ground(document)                  = "this can run"
```

Why this matters: features that would otherwise each need machinery —
optional members, cached results, conditional execution, gating on a
prerequisite — all turn out to be *the same operation applied at a different
place*. The rest of this document is mostly that observation, four times.

---

## 2. Layer 1 — documents, sites, and filling

A **site** is a hole in the place graph, carrying a sort (what may fill it),
optionally a face (the interface a filler must present), and optionally a
`_default` (which makes it *optional* rather than required).

Three relations you need to keep apart, because they are genuinely different
and were conflated early on:

- **formation** — may this node *nest inside* that one? (Milner's sorting
  discipline; about containment.)
- **`admits(core, site, filler)`** — may this filler go in *this hole*?
  Checked *before* substitution. Sort and face compatibility.
- **`compose` / `tensor`** — the algebra on interfaces; defined only when
  faces match.

`fill_sites` checks `admits` and then substitutes **at the site's own
position** — the filler *replaces the hole where it sits*. It is not
forest-spliced into the site's parent. (This was settled with live evidence;
the alternative silently reparents state.)

**Required vs optional.** `required_open_sites(document)` returns the sites
with no `_default`. Those are the ones that block a build.
`is_ground_document` is exactly "that list is empty".

---

## 3. Layer 2 — the higher-order DAG

This is the part most worth internalising, because it is what makes the
scheduling simple instead of special-cased.

**A simulation is one node.** `SimulationStep` takes a whole composite
document as config, runs it to completion *inside its own update*, and emits
a handle to the results. From the outer network's point of view it is an
ordinary step with no inputs and one output.

That framing is what makes downstream work fire **once**:

```
        ┌──────────────────┐
        │  SimulationStep  │   ← the entire inner simulation, all its ticks
        └────────┬─────────┘
                 │ results (a handle, not the data)
      ┌──────────┼──────────┐
      ▼          ▼          ▼
  figure      analysis   report card      ← ordinary steps, ordinary ordering
```

A figure step's `results` input is unsatisfied until the simulation node's
update returns. The simulation's per-tick stepping happens *inside* the node,
not beside it, so the flush steps are never siblings of the simulation's own
steps. **No completion phase, no marker, no scheduler special case** —
ordinary producer/consumer ordering does all of it. The number of times a
figure is drawn is independent of how long the run was.

### `results` is a handle, never the data

Everything that produces results emits a *reference*:

| handle | produced by | resolves to |
|---|---|---|
| `EmitterResults` | a live emitter | its accumulated history |
| `ArtifactResults` | the content-addressed store | a stored artifact |

Both answer the same three things — `kind`, `context`, `resolve()` — so **a
consumer cannot tell whether the study ran or was pulled from cache**. That
indistinguishability is the contract, not an implementation detail: it is
what lets `trigger` swap one for the other without rewiring anything.

Handles are memoized on `resolve()`. A run that has completed does not change
under you, several flush steps resolve the same handle, and a durable
emitter's `query()` can be expensive (re-reading a zarr store, and flushing
buffered rows on the way — which without memoization appends them again on
every read).

### `CachedResults`: the pull half

`SimulationStep` computes results by running. `CachedResults` produces the
same thing by resolving an artifact that already exists. They share the
`{results: node}` face **and both produce from `update()`**.

That last detail is load-bearing and easy to get wrong. An *emitter* produces
`results` at finalize, because its results only exist once the run is over.
But a study node is a producer *in the step DAG*, so it must return its
handle from `update()` — a `CachedResults` that only produced at finalize
would leave its consumers unordered, and they would fire first and read an
empty store. Matching `SimulationStep`, not matching the emitter, is what the
contract requires.

---

## 4. Templates: gating is filling

A **study template** fixes its analysis network and leaves the model as a
site. An **investigation template** has one site per member study. They are
the same on-disk shape (`*.template.{yaml,json}`), so one loader reads both;
only the caller's follow-on call differs.

Now the payoff. "Skip a member whose prerequisite failed" is not a scheduling
decision — it is a hole that was never filled:

```
fill the members that should run  →  prune_open_regions  →  ground document
```

`prune_open_regions` drops every region that still contains an open site and
reports it as blocked. A member that was not filled is not merely unfilled;
it is **absent from the built document**, so it is never constructed and
never runs, while its siblings still do. The engine never has to decide *not*
to run something.

This is why pruning is necessary rather than incidental: `Composite` rejects
a document with *any* open required site, so leaving one open would make the
whole investigation unbuildable rather than skipping one member.

---

## 5. Content addressing: where a result lives

An artifact's address is a hash of **what produced it and what went into it**:

```
artifact_id(composite_id, canonical(config), sorted(input_ids), commit)
```

Never a filesystem path — so two worktrees at the same commit resolve the
same address and share a cache with no symlink between them. An upstream
change invalidates everything downstream of it, because inputs' addresses are
part of the hash.

Three things worth knowing:

- **The formula is a wire format.** Every artifact in every store was placed
  by it. Changing it orphans them all, so it is pinned by golden vectors in
  *both* this repo and the workbench, and re-keying needs a migration.
- **An address is a hash of inputs, so it is only a valid cache key when the
  producer is deterministic given them.** That is what `fingerprint_of` /
  `check_fingerprint` are for: a recompute that disagrees with the stored
  fingerprint is *flagged*, never silently served.
- **Kinds are open.** `results` admits any handle; `results_<kind>` admits
  only one, so a study declaring it needs `sim_data` cannot be wired to a
  trajectory. A workspace registers a loader for its own kind without the
  engine knowing anything about it.

### `trigger`: partial graph execution

`trigger(document, target)` builds the sub-document that runs one study:

- prerequisites already in the store are **pulled** — their simulation node
  is swapped for a `CachedResults`, so they are ground and do not run;
- the target is **left open**, so it computes;
- everything else is **pruned** — absent, never constructed.

"Run this study" and "rerun this study reusing its upstream" are the same
call with different targets. Note that this is section 4's mechanism applied
at a different place: fill the ancestors, leave the target open, prune the
rest.

---

## 6. The `git:` protocol: running someone else's code

```
git:<owner>/<repo>@<ref>#<module>:<callable>
```

resolves a *repository* to a runnable `Edge`. Four properties, each of which
exists because its absence bit us:

1. **Pinned.** A ref resolves to a commit SHA and the SHA is recorded. A
   moving ref re-resolves to a new SHA and is surfaced, never silently rerun.
   A cached checkout is *verified* against its pin, not trusted on sight.
2. **Isolated.** The foreign stack lives in a per-SHA venv and is driven over
   a stdio-RPC boundary. No foreign code is imported into the host
   interpreter.
3. **Frozen.** If the repo ships a `uv.lock`, it is honoured. Re-resolving
   dependencies puts fetched code in an environment its authors never tested,
   which makes an artifact a function of *when* the venv was built — the
   opposite of what pinning a SHA is for. `Materialized.install_mode` reports
   `locked` vs `resolved` so a manifest cannot claim more than it has.
4. **Allow-listed and conformance-checked.** An address whose repo is not on
   the allow-list is refused, not run; and a resolved entrypoint's interface
   must admit the declared face before a run starts.

---

## 7. The laws, collected

These are the invariants. If you are changing the framework and one of them
would stop holding, that is the design discussion.

1. **A document that is not ground cannot run.** `Composite` enforces it.
2. **Filling is incremental and order-independent.** Filling some sites
   leaves the rest open; the result is another document.
3. **A filler must be admitted before it is substituted.** Sort and face are
   checked first.
4. **A consumer cannot tell a computed result from a pulled one.** Same face,
   same handle protocol, both produced from `update()`.
5. **`results` carries a reference, never bulk data.** Resolution is explicit
   and memoized.
6. **The same address means the same bytes.** If a recompute disagrees, that
   is flagged, not served.
7. **A simulation is one node of its outer graph.** Downstream work fires
   once, whatever the run length.
8. **The address is a wire format.** It changes only with a migration.

---

## 8. Where the seams are

Honest notes on the places the model is thinner than the story above.

- **`config` and wires are values, not schemas.** `address`, `config`,
  `inputs`/`outputs` are materialized by `access` and must not be walked as
  schemas. This is the single most common source of bugs in this codebase —
  `default`, `realize`, `resolve` and `render` have each had to learn it
  separately.
- **`SimulationStep` and `CachedResults` are reached by address string**
  (`local:SimulationStep`), so renaming them fails at resolution time, not
  import time. Static analysis will not catch it.
- **The `${name}` legacy parameter format coexists with typed parameters on
  purpose.** It is opt-in, not dead.
- **Lowering `CompositeSpec` onto the template machinery was attempted and
  abandoned.** The real corpus puts most placeholders inside `config`, which
  `core.fill` cannot walk. `substitute_parameters` remains the honest answer;
  do not assume the two systems are convergent.
