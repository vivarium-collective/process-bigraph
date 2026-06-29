# CompositeSpec — Unified Composite Declaration — Design

**Date:** 2026-06-28
**Status:** Design (brainstormed + grounded in a process-bigraph / pbg-superpowers / vivarium-dashboard survey; approved section-by-section by user)
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
default_n_steps: int | None = None

# associated artifacts — DECLARATIVE this round (stored + surfaced; execution = follow-up)
visualizations: list[dict] = []   # study-spec viz dicts
analyses:       list[dict] = []   # NEW field, same {address, config?, …} declarative shape
emitters:       list[dict] = []   # {address, config?, paths?}
requires: dict = {}               # {processes: [...]} (carried from YAML)

# the body — EXACTLY ONE state source (validated):
state:   dict | None = None       # static composites inline this (also serves as default state)
builder: Callable | str | None    # generator: a callable in code; a dotted "pkg.mod:fn" in JSON

# generator-only: where the saved materialized default state lives
default_state_ref: str | None = None   # e.g. "<id>.default-state.json" (sibling artifact)

# code-only (not serialized)
core_extensions: list[Callable] = []
```

**Methods**
- `to_composite(overrides=None, core=None) -> Composite` — resolve `${param}` substitution (static) or
  call the builder with merged defaults+overrides (generator); hand the document to `Composite(document, core)`.
- `to_document(overrides=None) -> dict` — the resolved `{schema, state}` doc, without instantiating `Composite`.
- `default_state(base_dir=None) -> dict | None` — the state used for **display**: inline `state` (static)
  or the parsed `default_state_ref` artifact (generator). Returns `None` if a generator's artifact is
  missing/not-yet-generated (→ dashboard shows the honest degrade notice). **Never runs the builder.**
- `to_dict() / from_dict(d)` — round-trip the portable JSON. A code `builder` callable serializes to its
  dotted `"pkg.mod:fn"` path; `from_dict` leaves it as a string (resolved lazily on `to_composite`).
- **Validation:** exactly one of `state` / `builder` is set; a `builder` may additionally carry a
  `default_state_ref`; `default_state_ref` without a `builder` is an error.

**Regeneration (generator default-state artifact)**
- `regenerate_default_state(spec, base_dir, core=None) -> Path` — run `spec.to_composite()` with default
  params, serialize via `Composite.serialize_state()`, write `<base_dir>/<id>.default-state.json` with a
  small provenance stamp (`{generated_from_commit, param_signature, generated_with}`). This mirrors the
  existing dashboard `regenerate_composite_states.py` pattern. A CI/freshness check (compare stamp vs
  current commit/param signature) is a **follow-up**, not in this spec.

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
- **YAML/JSON loader** — `from_file` accepts the existing spec-file keys and the new `analyses`. A static
  file's `state` becomes the spec's inline `state`. A file MAY instead declare `builder: "pkg.mod:fn"` +
  `default_state_ref` for a generator authored as data (not required for this round).
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
optional `analyses`, `visualizations`, `emitters`, `wiring_status`, and `notice` fields. SP-C's config
form and the Explorer keep working; the new fields are additive.

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
  `default_state_ref` requires `builder`); `to_dict`/`from_dict` round-trip incl. dotted-builder
  serialization; `to_document`/`to_composite` produce a runnable `Composite` (fake builder returning a
  small doc); `default_state` reads inline state and a sibling artifact, and returns `None` when the
  artifact is absent; decorator registers into the registry; `discover_specs` merges decorator + file
  specs; `regenerate_default_state` writes the artifact + stamp.
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

**In:** the `process-bigraph` `CompositeSpec` abstraction (class, validation, `to_composite`/
`to_document`/`default_state`/`to_dict`/`from_dict`, decorator, registry, `discover_specs`,
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
3. **`from_file` builder authoring** — confirm the data-authored-generator path (`builder:` +
   `default_state_ref` in YAML) is specified but un-proven this round (no proof composite uses it).
