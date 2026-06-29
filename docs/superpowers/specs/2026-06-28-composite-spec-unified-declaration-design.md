# CompositeSpec — Unified Composite Declaration — Design

**Date:** 2026-06-28
**Status:** Design (brainstormed + grounded in a process-bigraph / pbg-superpowers / vivarium-dashboard survey; **contract validated against the full ecosystem** — all 13 v2ecoli generators, 93 YAML composites across ~20 pbg-* repos, and ~12 study specs — see §3a; approved section-by-section by user)
**Author:** Eran Agmon (with Claude)
**Repos touched:** `process-bigraph` (new abstraction — center of gravity), `pbg-superpowers` (back-compat shim), `vivarium-dashboard` (resolve against the new contract + display robustness + bug fix). Workspaces (`v2ecoli`, `pbg-autopoiesis`) provide the two proof composites.

## 1. Context & goal

Today a "composite" can be declared two incompatible ways, and a third object (the runtime
`Composite`) knows about neither:

- **`@composite_generator` decorator** (in `pbg-superpowers`) — Python code + a module-level
  `_REGISTRY`. Rich metadata: `parameters`, `visualizations`, `emitters`, `default_n_steps`,
  `core_extensions`. Runs a builder (`build_core`/ParCa) to *generate* state dynamically. This is what
  v2ecoli's `baseline` uses.
- **`*.composite.yaml` / `.json` spec files** — static data. Has `name`/`description`/`parameters`/
  `state`/`emitters`/`tags`/`requires`, with `${param}` substitution. **No** visualizations, analyses,
  or `default_n_steps`. Cannot run code.
- **`process_bigraph.Composite`** — the runtime object. Consumes `{schema, state, interface, bridge,
  …}`. Has `serialize_state()`/`serialize_schema()` but **zero** notion of parameters, runtime,
  visualizations, analyses, or emitters. No registry, no decorator.

The trigger: the dashboard's Composite Explorer cannot robustly display a generator composite. Opening
v2ecoli `baseline` locally without a ParCa `out/cache` makes `build_generator` throw →
`resolve_composite` swallows it to `None` → 404 → the client (`walkthrough.js:4186`) prints a
hardcoded, **misleading** message ("a remote build cannot build generator composites (no local ParCa
cache)") even though the workspace is local.

**Goal:** one declarative descriptor — **`CompositeSpec`** — that lives in `process-bigraph` next to
`Composite`, carries all authoring/UI metadata, knows how to produce a runtime `Composite`, and always
resolves to a **default state** for display without running ParCa. Both existing front-ends (decorator,
YAML file) reconcile onto this one contract and one registry. The dashboard reads exactly one contract.

### Approved decisions (from brainstorming)
1. **Descriptor wrapping `Composite`** — a new `CompositeSpec` class carries the metadata + a state
   source; it *produces* a `Composite` document. `Composite`'s runtime `config_schema` is **not**
   bloated with authoring fields. (Rejected: extending `Composite` itself; a frozen-JSON-only format.)
2. **Two front-ends, one contract** — keep both authoring styles, both producing/registering a
   `CompositeSpec` into one `process-bigraph` registry: a decorator (moved to `process-bigraph`) for
   code/generator composites, and a YAML/JSON file for static ones. "Reconciled" = identical metadata
   fields + one registry + one resolve contract, two ways to author.
3. **Foundation + one proof each** — build the abstraction + a `pbg-superpowers` shim + the dashboard
   resolve/display + the bug fix; prove on **one generator** (v2ecoli `baseline`) and **one static**
   (autopoiesis `growth-division`). Migrating all other composites, an analyses **execution** runner,
   and removing the shim are follow-ups.
4. **Saved default state for display** (replaces a live "skeleton" build) — every `CompositeSpec`
   resolves to a *default state* for display, regardless of kind: a static composite's inline `state`
   *is* its default state; a generator ships a **saved, regenerable default-state artifact** (run the
   builder once with default params, serialize, commit). Display reads that — **no ParCa at display
   time, ever.** (Rejected: a `structure_only`/skeleton builder mode — risked entanglement with
   ParCa-gated process assembly in `build_core`.)
5. **Inline state for static + a sibling file for generators** — a static composite inlines its small
   `state` in the descriptor; a generator references a sibling `<id>.default-state.json` (v2ecoli
   baseline state is thousands of molecules — too big to inline). The descriptor stays small.

## 2. Architecture

```
@composite_spec  ─┐
                  ├─►  CompositeSpec  ──►  registry  ──►  dashboard / runners
*.composite.yaml ─┘         │
                            ├─ .to_composite(overrides, core) ─►  Composite   (full materialize)
                            ├─ .default_state()               ─►  state dict  (display; NO ParCa)
                            ├─ .to_document(overrides)        ─►  {schema, state}  (process-bigraph doc)
                            └─ .to_dict() / .from_dict()       ─►  the portable composite JSON
```

`CompositeSpec` is a `@dataclass` in `process_bigraph` (new module `process_bigraph/composite_spec.py`).
It aligns with `Composite` by **producing its document**, not by sharing its class.

## 3. The `CompositeSpec` contract (the "composite JSON that captures what we need")

```
# identity
id: str                       # canonical "<module>.<name>"
name: str
description: str = ""
tags: list[str] = []

# configuration surface
parameters: dict[str, dict]   # {name: {type, default, description?, choices?}}
                              # `type` ∈ a CANONICAL vocabulary (see "Parameter types" below);
                              # aliases (bool/boolean, float/number, int/integer) normalized on load.
default_n_steps: int | None = None   # a DEFAULT/fallback, NOT a hard bound — studies override per run/variant

# associated artifacts — DECLARATIVE this round (stored + surfaced; execution = follow-up).
# These are an OPEN menu: studies extend freely and are not constrained to these lists.
# `state` may ALSO contain inline emitter/viz/analysis STEPS; these fields are the
# declarative, dashboard-facing surface, not a closed set.
visualizations: list[dict] = []   # study-spec viz dicts
analyses:       list[dict] = []   # NEW field, same {address, config?, …} declarative shape
emitters:       list[dict] = []   # {address, config?, paths?}

requires: dict = {}               # {processes: [...], types: [...]} — process AND external-type deps
schema:  dict = {}                # bigraph-schema type declarations for stores (e.g. {population: tree});
                                  # the static counterpart of a generator document's "schema" envelope key.
                                  # A Composite DOCUMENT is {schema, state, …} — the spec carries both.

# the body — EXACTLY ONE state source (validated):
state:   dict | None = None       # static composites inline this (paired with optional `schema`);
                                  # also serves as the default state for display
builder: Callable | str | None    # generator: a callable in code; a dotted "pkg.mod:fn" in JSON.
                                  # Returns a process-bigraph DOCUMENT dict — {state, schema?, and
                                  # composite-exec keys: skip_initial_steps?, sequential_steps?,
                                  # flow_order?, run_steps_on_init?}. `to_composite`/`to_document`
                                  # forward the WHOLE document to Composite(); `default_state` reads
                                  # only its `state` (from the saved artifact).

# generator-only: where the saved materialized default state lives
default_state_ref: str | None = None   # e.g. "<id>.default-state.json" (sibling artifact)

# code-only (not serialized) — register types/processes on the core (e.g. map[pymunk_agent],
# parca core, millard links). LOAD-BEARING and applied to the core BEFORE the builder runs;
# required for builds in sandboxed/subprocess environments (and for regenerate_default_state).
core_extensions: list[Callable] = []
```

**Parameter types (canonical vocabulary + alias normalization).** The audit found the same
semantic type spelled inconsistently across composites (`bool`/`boolean`, `float`/`number`,
`int`/`integer`). The contract defines a canonical set — `integer`, `float`, `string`, `boolean`,
`list`, `map`/`object` — and the decorator + `from_file` loader **normalize known aliases** to it
(`bool→boolean`, `number→float`, `int→integer`). Unknown types load as-is with a logged warning.
This is part of "make it robust": the dashboard config form keys rendering off a single vocabulary.

**Deep-path overrides convention.** Two override styles exist in studies: flat
`parameter_overrides` (validated against `parameters`) and deep-path `config_overrides` like
`ecoli-polypeptide-elongation.basal_elongation_rate: 28` (targets `<process>.<config-key>`). The
latter is NOT a separate contract mechanism — it is a **builder-provided escape hatch**: a generator
declares a `config_overrides` parameter (type `map`) and applies the patches to its built document
itself (as v2ecoli `baseline()` does, baseline.py:696). `to_composite(overrides)` forwards all
overrides to the builder; flat `parameters` are the validated knobs, `config_overrides` is the typed,
declared escape hatch. No deep-path-aware logic lives in `CompositeSpec`.

**Methods**
- `to_composite(overrides=None, core=None) -> Composite` — apply `core_extensions` to the core FIRST, then
  resolve `${param}` substitution over `{schema, state}` (static) or call the builder with merged
  defaults+overrides (generator); hand the **whole** resolved document to `Composite(document, core)`.
- `to_document(overrides=None) -> dict` — the resolved process-bigraph document, WITHOUT instantiating
  `Composite`. For a static spec: `{schema, state}` after substitution. For a generator: the entire dict
  the builder returns (`state`, optional `schema`, plus any composite-exec keys like `skip_initial_steps`,
  `sequential_steps`, `flow_order`) — passed through verbatim, never narrowed to just state+schema.
- `default_state(base_dir=None) -> dict | None` — the **state** used for display: inline `state` (static)
  or the `state` of the parsed `default_state_ref` artifact (generator). Returns `None` if a generator's
  artifact is missing/not-yet-generated (→ dashboard shows the honest degrade notice). **Never runs the
  builder; needs no core_extensions** (the artifact is already materialized).
- `to_dict() / from_dict(d)` — round-trip the portable JSON. A code `builder` callable serializes to its
  dotted `"pkg.mod:fn"` path; `from_dict` leaves it as a string (resolved lazily on `to_composite`).
- **Validation:** exactly one of `state` / `builder` is set; `schema` may accompany `state` (static) and
  defaults to `{}`; a `builder` may additionally carry a `default_state_ref`; `default_state_ref` without
  a `builder` is an error; parameter `type`s are normalized to the canonical vocabulary on construction.

**Regeneration (generator default-state artifact)**
- `regenerate_default_state(spec, base_dir, core=None) -> Path` — run `spec.to_composite()` with default
  params, serialize via `Composite.serialize_state()`, write `<base_dir>/<id>.default-state.json` with a
  small provenance stamp (`{generated_from_commit, param_signature, generated_with}`). This mirrors the
  existing dashboard `regenerate_composite_states.py` pattern. A CI/freshness check (compare stamp vs
  current commit/param signature) is a **follow-up**, not in this spec.

## 3a. Coverage — validated against the ecosystem

The contract was audited against real usage: **all 13 `@composite_generator` generators in v2ecoli**,
**93 `*.composite.{yaml,json}` files across ~20 pbg-* repos** (autopoiesis, bioreactordesign,
membrane-actin, tyssue, copasi, tellurium, smoldyn, martini, comets, mem3dg, cellpack, yalla, biomodels,
nfsim, readdy, medyan, lammps, compucell3d, parsimony, caspule), and **the study→composite interface
across ~12 `study.yaml` files** (v2ecoli, v2e-invest, autopoiesis, viva-munk, sms-ecoli, bioreactordesign).

| Capability seen in practice | Source | Covered by |
|---|---|---|
| name / description / tags | all | `name`/`description`/`tags` |
| flat config knobs | all generators + most YAML | `parameters` (canonical types) |
| string enums | baseline `emitter` | `parameters[].choices` |
| default run length | v2ecoli `default_n_steps`, incl. dynamic (`len(STEP_ORDER)`) | `default_n_steps` (fallback) |
| declarative viz / emitters / analyses | v2ecoli decorator; tyssue top-level `emitters` | `visualizations`/`emitters`/`analyses` (open menu) |
| inline emitter/viz STEPS in state | membrane-actin (8 viz steps) | `state` (steps live in state; the fields above are the declarative surface) |
| process deps | most YAML `requires.processes` | `requires.processes` |
| **external type deps** | tyssue `requires.types` | **`requires.types` (added)** |
| **store type declarations** | autopoiesis `schema: {population: tree}`; builder `{state, schema}` envelope | **`schema` (added)** |
| **composite-exec doc keys** | v2ecoli builders return `skip_initial_steps`/`sequential_steps`/`flow_order` | **whole-document passthrough (added)** |
| core/type registration | 5 v2ecoli composites (`core_extensions`) | `core_extensions` (load-bearing, pre-builder) |
| flat parameter_overrides | v4 studies | `parameters` (validated) |
| deep-path `config_overrides` | param-uq studies; `baseline(config_overrides=…)` | builder escape-hatch param of type `map` (documented convention) |
| static (data) composite | autopoiesis, all pbg-* wrappers | `state` (+ `schema`) inline |
| generator (code) composite | v2ecoli | `builder` + saved `default_state_ref` |

**Study-level, NOT composite gaps (out of scope by design):** variant/condition declaration,
`parameter_overrides`/`config_overrides` *values*, per-variant seeds/durations, `default_emitter` choice
at run time, `observables`/`readouts`/`behavior_tests`, `pipeline_gate`/`prerequisites`,
`enforced_params` (critical-param locking). The composite exposes the parameter/emitter/analysis
**surface**; the study OWNS which variants to run and what to measure. An optional future
`parameters[].enforced: <reason>` flag (to let a composite mark a knob whose change invalidates the
model, e.g. v2ecoli `mode: full`) is noted but **deferred** — YAGNI for this round.

## 4. Registry & discovery

- A module-level registry in `process_bigraph` (`composite_spec._REGISTRY: dict[str, CompositeSpec]`),
  with `register(spec)`, `get(id)`, `all()`.
- `@composite_spec(...)` registers on import (`builder=<the decorated fn>`).
- `CompositeSpec.from_file(path)` parses a `*.composite.{yaml,json}` into a spec.
- `discover_specs(workspace=None) -> dict[str, CompositeSpec]` — (a) imports installed packages that
  depend on `bigraph-schema` (triggering decorators, same walk `discover_generators` does today) **and**
  (b) scans a workspace's `composites/` directory for spec files, merging both into the one registry.
  Idempotent; safe to call repeatedly.

## 5. Authoring front-ends + the pbg-superpowers shim

- **`@composite_spec(...)`** (in `process-bigraph`) — same kwargs the current `@composite_generator`
  exposes (`name`, `description`, `parameters`, `visualizations`, `emitters`, `default_n_steps`,
  `core_extensions`) **plus** `analyses` and `tags`. Builds + registers a `CompositeSpec` with
  `builder=<fn>`.
- **YAML/JSON loader** — `from_file` accepts the existing spec-file keys (`name`, `description`, `tags`,
  `parameters`, `state`, `emitters`, `requires.processes`) plus the keys the audit surfaced: `analyses`,
  `visualizations`, `default_n_steps`, top-level `schema`, and `requires.types`. A static file's `state`
  (+ optional `schema`) becomes the spec's inline body; parameter `type`s are alias-normalized. A file MAY
  instead declare `builder: "pkg.mod:fn"` + `default_state_ref` for a generator authored as data (specified
  but un-proven this round — no proof composite uses it).
- **pbg-superpowers shim** — `@composite_generator`, `_REGISTRY`, `build_generator`,
  `discover_generators` become **thin delegators** to the `process-bigraph` registry:
  - `@composite_generator(**kw)` maps its kwargs onto `@composite_spec(**kw)` (no `analyses`/`tags` →
    defaults) and registers the same `CompositeSpec`.
  - `_REGISTRY` is a view/alias onto `process_bigraph.composite_spec._REGISTRY` keyed identically, with
    a `GeneratorEntry`-compatible shim object (same attributes the dashboard reads today) so existing
    imports keep working.
  - `build_generator(entry, overrides)` delegates to the spec's `to_document(overrides)`.
  - `discover_generators()` delegates to `discover_specs()`.
  - **v2ecoli's existing `@composite_generator(...)` usage is unchanged.** Removing the shim is a
    follow-up.

## 6. Dashboard resolve + display robustness + the bug fix

- **Resolve against the new contract** — `vivarium_dashboard/lib/composite_resolve.py::resolve_composite`
  resolves through the `process-bigraph` registry (the shim keeps the current `pbg_superpowers` import
  path working during transition). For the Explorer it returns:
  - **metadata** (name, description, parameters, default_n_steps, visualizations, analyses, emitters) —
    straight from the descriptor, **zero build**;
  - **wiring state** from `spec.default_state(base_dir)` — the inline state (static) or the saved
    artifact (generator), **no ParCa**.
- **Structured payload** — distinguish *not-found* (404, `unresolved`) from *found-but-no-default-state*
  (200 + metadata + `wiring_status: "ready" | "unavailable"` + an honest `notice` when a generator's
  artifact hasn't been generated). Display always shows the config form + metadata even when wiring is
  unavailable.
- **Bug fix** — delete the hardcoded "a remote build cannot build generator composites (no local ParCa
  cache)" string at `walkthrough.js:~4186`; render the server's actual `error`/`notice`/`wiring_status`
  instead. The misleading remote-build framing is removed for both local and remote workspaces.

## 7. Components (each independently testable)

- **process-bigraph:** `CompositeSpec` dataclass + validation; `to_composite` / `to_document` /
  `default_state`; `to_dict`/`from_dict` round-trip; `@composite_spec` decorator + registry +
  `discover_specs`; `regenerate_default_state`. Unit-tested with a **fake builder** (no ParCa).
- **pbg-superpowers:** the shim — old `@composite_generator` still registers; `_REGISTRY` view exposes
  the dashboard-read attributes; `build_generator` / `discover_generators` delegate.
- **vivarium-dashboard:** `resolve_composite` returns metadata + default-state wiring + `wiring_status`;
  the structured 404-vs-200 payload; the `walkthrough.js` bug fix.

## 8. Data flow / shape

The dashboard's resolve payload stays **shape-compatible** with today's `CompositeResolvePayload`
(`{id, name, description, parameters, state, svg, kind, module, default_n_steps}`), **plus** new
optional `analyses`, `visualizations`, `emitters`, `schema`, `requires`, `tags`, `wiring_status`, and
`notice` fields. SP-C's config form and the Explorer keep working; the new fields are additive.

## 9. Error handling

| Case | Behavior |
|---|---|
| Composite id not registered | 404 `{error, unresolved:true, ref}` (unchanged) |
| Generator, default-state artifact missing | 200 metadata + `wiring_status:"unavailable"` + honest `notice` ("default state not generated — run regenerate or run the composite"); config form still renders |
| Static spec / generator with artifact present | 200 full payload incl. wiring `state` |
| Spec file malformed | `from_file` raises a typed error; loader skips it with a logged warning (discovery stays robust) |
| `to_composite` builder raises at RUN time | propagates to the run path (not the display path) — display never builds |

## 10. Testing

- **process-bigraph (headless, no ParCa):** construct/validate (exactly-one-of `state`/`builder`;
  `schema` pairs with `state`; `default_state_ref` requires `builder`); parameter-`type` alias
  normalization (`bool→boolean`, `number→float`, `int→integer`); `to_dict`/`from_dict` round-trip incl.
  dotted-builder serialization, `schema`, and `requires.{processes,types}`; `to_document` forwards a
  generator's WHOLE document (extra keys like `skip_initial_steps`/`flow_order` preserved);
  `to_composite` applies `core_extensions` before building and produces a runnable `Composite` (fake
  builder returning a small doc); `default_state` reads inline state and the `state` of a sibling
  artifact, and returns `None` when the artifact is absent; decorator registers into the registry;
  `discover_specs` merges decorator + file specs; `regenerate_default_state` writes the artifact + stamp.
- **pbg-superpowers (headless):** old `@composite_generator` registers a `CompositeSpec`; `_REGISTRY`
  view exposes the attributes the dashboard reads; `build_generator` delegates to `to_document`;
  `discover_generators` delegates.
- **vivarium-dashboard (headless, fakes only):** generator-without-cache resolves to metadata +
  `wiring_status:"unavailable"` + notice (no ParCa); generator-with-artifact resolves full wiring;
  static composite resolves from inline state; the client renders the server notice (no hardcoded
  remote-build string).
- **Proof (in-the-loop where ParCa is needed):** v2ecoli `baseline` — `regenerate_default_state`
  produces + commits `baseline.default-state.json` **once** (needs a ParCa cache, on the machine that
  has it); thereafter the dashboard displays baseline with **no ParCa**. autopoiesis `growth-division`
  — static, resolves from inline `state`. The artifact generation for baseline is the one step that
  needs a real environment; everything else is headless.

## 11. Scope boundaries

**In:** the `process-bigraph` `CompositeSpec` abstraction (class with `schema` + `requires.{processes,
types}` + canonical parameter-type normalization, validation, `to_composite`/`to_document` [whole-document
passthrough]/`default_state`/`to_dict`/`from_dict`, decorator, registry, `discover_specs`,
`regenerate_default_state`); the `pbg-superpowers` shim; the dashboard resolve-against-new-contract +
saved-default-state display + the `walkthrough.js` bug fix; proof on `baseline` (generator) +
`growth-division` (static).

**Out (follow-ups):** migrating all other composites onto the contract; an **analyses execution**
runner (analyses are declarative-only this round); a CI **freshness check** for generator default-state
artifacts; **removing** the pbg-superpowers shim; the deployment/sms-api default-state path (ties into
SP-D — a remote build could ship its default-state artifact in the workspace tarball).

## 12. Realism / build split (READ FIRST)

- **Headless-doable now:** the entire `process-bigraph` abstraction + shim + dashboard wiring + bug
  fix + all unit tests (fake builders, no ParCa). The static proof (`growth-division`) is headless.
- **One in-the-loop step:** generating `baseline.default-state.json` requires a real ParCa cache (run
  `regenerate_default_state` on the machine/worktree that has `out/cache`, e.g. the mini). After that
  one-time artifact commit, the dashboard proof is headless.

## 13. Open questions (resolve in planning)

1. **`_REGISTRY` view shape** — exact `GeneratorEntry`-compatible shim object the dashboard reads
   (`.name`, `.description`, `.parameters`, `.module`, `.default_n_steps`, `.visualizations`,
   `.emitters`). Enumerate every attribute the current dashboard touches and mirror it.
2. **Artifact location convention** — `<id>.default-state.json` beside the spec module vs a workspace
   `composites/_state/` dir vs the dashboard's existing `api/composite-state/<id>.json`. Pick one in
   planning; prefer reusing the dashboard snapshot convention if it fits.
3. **Canonical type vocabulary final list** — confirm the exact set + alias map (the audit found
   `bool`/`boolean`, `float`/`number`, `int`/`integer`, plus `object`/`map`); decide whether `map` or
   `object` is canonical. Normalize the existing v2ecoli composites' types as part of the proof.
