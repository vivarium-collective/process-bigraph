# Process Contract Formalization & Audit — Design

**Date:** 2026-10-03
**Status:** Design (pending review)
**Repos:** `bigraph-schema`, `process-bigraph`, `vivarium-workbench`
**Author:** design brainstorm (Eran Agmon)

## 1. Purpose & intent

Formalize and harden the **Process contract** so that a process's declared
interface and behavior are machine-checkable, and extend that contract beyond
the port interface to the **conditions under which the process is valid**. Four
drivers, all in scope (confirmed):

1. **Runtime validity / in-domain** — catch when a process runs outside the
   regime it was built for (inputs/params beyond its declared bounds).
2. **Composition / swap safety** — guarantee a process can be substituted into a
   slot (its ports/types/units/bounds/invariants match what the slot needs).
3. **Claims-are-true conformance** — a declared contract verified against what
   the process actually does (no silent drift between declaration and code).
4. **Interface formalization** — the loose `inputs()`/`outputs()`/`update()`
   conventions become an explicit, versioned, enforceable contract.

**Audit model (confirmed):** *layered both* — a **static auditor** (always-on,
CI-friendly) plus a **runtime "contract-strict" mode** (opt-in, off by default).

**Template integration (confirmed):** templates have holes/sites where processes
go; a hole declares an **expected contract**, the system **identifies candidate
processes by their contracts**, and **audits** a candidate against the hole's
expectations.

**Composites (confirmed):** a composite *is* a process (`CompositeLink(ProcessLink)`),
so it carries the same contract and is audited the same way, with an added
internal-consistency audit.

## 2. Grounding: what already exists (and the one stale-checkout trap)

A substantial foundation already exists in **installed bigraph-schema 1.6.0**.
**Trap:** the local `~/code/bigraph-schema` checkout is on branch
`chore/bump-1.4.3` and *lacks* this layer — all implementation targets current
bigraph-schema `main` (1.6.0+), not the local tree, which must be updated first.

Existing machinery (cite installed 1.6.0 / process-bigraph):

- **The declaration surface is `Edge`** (`bigraph_schema/edge.py`): `config_schema`,
  `inputs()`, `outputs()`, `interface()`, `description`, and an optional
  `contract` attribute. `Process`/`Step` (`process_bigraph/composite.py`) add
  `update(state, interval)` / `update(state)`. **Enforcement today ≈ none:**
  `inputs`/`outputs`/`update` default to `{}`/no-op; registration validates
  nothing; a process that never implements `update` silently does nothing.
- **Validation primitives exist but are barely wired in:** `core.check(schema, state)`
  and `core.validate(schema, state)` already handle types, `Range` (`_min`/`_max`),
  `Nonnegative`, `Enum`, and `_units` — but are called in exactly one library
  site (`composite_spec.py:184`, for spec params), never on process ports or
  `update()` outputs at runtime.
- **`ProcessContract`** (`bigraph_schema/contract.py`) is already "the full
  interface spec of an edge **or a site**," with a machine-checkable **`face`**
  (the typed port core `admits` reads) and **monotonic `narrow`/`amend`** (a
  contract only gets stricter down through composition).
- **Holes/sites & filling** (`bigraph_schema/schema.py:630` `Site`;
  `assembly.py`): `Site._sort` can **name a contract**; `fill_sites` fills holes;
  before substitution, **`admits`/`contract_admits`/`face_conforms`** check a
  proposed filler against the site's required contract. `face_conforms` =
  structural subtyping (the filler must provide every required port at a
  resolvable type; over-providing is fine). `build(core, template, overrides)`
  fills + ground-checks → runnable.
- **`check_conformance`** (`process_bigraph/protocols/git.py`) is a real pre-run
  interface gate, but scoped to the git remote-process protocol.
- **Templates** (`process_bigraph/templates.py`): a template is any schema with
  `Site` nodes; `interfaces()` finds holes; `required_open_sites` = holes with no
  `_default`; `is_ground_document` = runnable.

**Net:** "a hole declares an expected contract and a candidate is checked against
it" is already built **at the port/face level**. The net-new work is three
things: richer contract conditions, candidate matching by contract, and the two
audit layers — all extending a sound foundation rather than inventing one.

## 3. The extended contract model (bigraph-schema)

Extend `ProcessContract` with four structured, serializable condition groups,
plus its existing `face` and descriptive fields.

### 3.1 Ports — bounds, units, required/optional (no new field)

These ride on the face's port **types**, reusing the type system: `Range`
(`_min`/`_max`), `_units`, `Nonnegative`, `Enum`, `Maybe[...]` = optional. So
`{'glucose': 'positive_float[mM]'}` *is* the bound. What's new is that matching
and audit **read** these (bound + unit compatibility), not just "port exists at a
resolvable type." A port is **required** iff present and non-`Maybe`.

### 3.2 `validity` (new) — the parameter domain

`config`-param → constraint declaring the regime the process is valid in:

```yaml
validity:
  pH: {min: 6, max: 8}
  temperature: {min: 298, max: 313, units: K}
```

References `config_schema` names. Declarative, serializable.

### 3.3 `invariants` (new) — cross-port predicates, per tick

```yaml
invariants:
  - name: mass_conservation
    expr: "abs(sum(outputs.mass) - sum(inputs.mass)) <= tol"
    tol: 1.0e-9
```

Over `inputs.*` / `outputs.*` / `config.*` / `state.*`, each with optional `tol`.
Plus the **escape hatch**: an `invariants(self)` method on the Edge returning
callables for invariants too complex to serialize (marked *code-only*).

### 3.4 `requires` / `guarantees` (new) — pre/post conditions

`requires` = preconditions on input state (what the process needs, e.g.
`"inputs.biomass > 0"`); `guarantees` = postconditions on the update output (what
it promises, e.g. `"all(outputs.mass >= 0)"`). Same declarative-or-callable
pattern.

### 3.5 Two unifying decisions

- **One shared declarative expression language** (used by `validity`,
  `invariants`, `requires`, `guarantees`): name refs (`inputs.<port>`,
  `outputs.<port>`, `config.<param>`, `state.<path>`), comparison + arithmetic,
  reducers (`sum`/`min`/`max`/`abs`/`all`/`any`), `tol`. **Parsed to an AST and
  evaluated against a binding environment — never `eval`.** New module beside
  `contract.py`.
- **The same `ProcessContract` serves both sides.** A *process* declares it via
  the existing `contract` attribute; a *template hole* declares it via
  `Site._sort` naming a registered contract (already supported). "What this
  process provides" and "what this hole requires" are the same object — which is
  what makes candidate-matching a direct comparison.

**Monotonic `narrow` extends to the new fields:** narrowing may tighten a range,
add an invariant, add a precondition — never loosen. (Today `narrow` "only ever
adds required ports"; generalize to "only ever tightens.") Keeps matching sound.

## 4. Candidate matching by contract (bigraph-schema)

The reverse direction that does not exist today: *"nothing enumerates the registry
to return candidates that would pass."*

**Relation — contract satisfaction (subsumption).** A candidate process satisfies
a hole iff its declared contract `C_proc` conforms to the hole's required contract
`C_hole`, generalizing `contract_admits` from "does this filler instance admit" to
"does this process's declaration admit." Per condition group:

| Group | Check | Status |
|---|---|---|
| Face (ports) | `face_conforms` — provides every required port at a resolvable type | exists |
| Bounds/units | candidate range ⊆ required range; units convertible (*bound subsumption*) | new; extends `face_conforms` |
| Validity domain | candidate `validity` **covers** the hole's required regime | new |
| Invariants / pre-post | candidate **declares** the guarantees the hole requires | new (structural only — see limit) |

**Search:** `find_candidates(core, site)` enumerates the process registry
(`link_registry` + discovered composites/generators) and returns those whose
declared contract satisfies the hole's, each result carrying *why*:
`{address, match: full|partial, over_provides: [...], fails: [{condition, reason}]}`.
**Near-misses are included** with the specific failing condition (so an author
sees "`copasi` fits except it doesn't guarantee mass conservation" instead of an
empty list).

**Honest limitation (stated up front):** semantic *implication* between arbitrary
predicates is undecidable. Matching on invariants/pre-post is therefore
**structural** (candidate declares a compatible guarantee by name/shape), not a
theorem prover. Face + bounds/units + validity-domain matching *is* exact and
decidable; predicate conformance is a **sound filter** (won't surface a clearly
incompatible process) with final verification deferred to runtime strict-mode.

**Template integration:** for a template's open sites, `find_candidates` per site
drives "processes that fit this hole" in the authoring surface; the chosen
candidate then goes through the hole-directed audit (§5–6) before it is wired in.

## 5. The static auditor (always-on; well-formed + shape-consistent)

`core.audit_contract(process_or_address)` → `{ok, findings: [{severity, code, where, message}]}`.
Checks, without running the process:

1. **Well-formedness** *(error)* — expressions parse to valid ASTs and every name
   resolves (`inputs.foo` in the face, `config.bar` in `config_schema`); units
   parse (Pint); ranges have `min ≤ max`; `validity` references real params.
2. **Declaration ↔ implementation drift** *(warning)* — best-effort AST
   introspection of `update()`: does it read only declared input ports and write
   only declared output ports? **Plus the no-op trap:** a non-`DraftProcess`
   whose `update()` is still the inherited `{}` default.
3. **Narrow soundness** *(error)* — on a composed template, each `narrow` only
   *tightened* (no loosened bound / dropped required port).
4. **Completeness grade** *(info)* — how much of the contract is actually declared
   vs unconstrained (bare `float` ports, no invariants). A score, not a gate;
   feeds the rigor program.

**CI integration:** an audit command walks all registered processes and fails on
any `error` finding (hard gate).

**Honest limit:** the drift check (2) is a **conservative lint** — Python is
dynamic, so AST introspection flags clear mismatches but cannot *prove* full
conformance statically. Strong "claims-are-true" comes from runtime strict-mode.

## 6. Runtime contract-strict mode (process-bigraph; opt-in)

**Activation:** a core setting `contract_strict ∈ {off, raise, record}`, **off by
default (zero cost)**. When on, the engine wraps each process's call path (around
`Open.invoke` / the scheduler's process step), reading the contract via
`describe_contract()`.

**Per tick, in order:**

1. **`requires`** — preconditions against input state *before* `update()`.
2. **Input port bounds/units** — each input within its port type's
   `Range`/`Nonnegative`/`_units`, via the **existing `core.check`/`core.validate`**.
3. **Validity domain** — config params within `validity` (checked at init + on
   `reconfigure`; per-tick only if declared over state).
4. `update()` → the output delta.
5. **`guarantees` + output port bounds** — against the output.
6. **Invariants** — cross-port predicates over input+output.

Code-escape-hatch callables run here too (runtime-only; the static auditor
skipped them).

**Failure posture:** `raise` halts the run fail-loud (dev/CI/validation — matches
the ecosystem's fail-loud ethos); `record` routes violations to a
listener/audit-log and continues (production monitoring).

**Performance:** declarative expressions are **compiled once** (AST → closure) at
process init, not re-parsed per tick; undeclared ports/predicates are skipped.
Fine for dev/CI/validation runs; not a production hot-loop default.

**Honest framing:** strict mode verifies the contract on the **trajectories
actually run** — a check, not a proof. The three layers compose: static proves
*well-formed + shape-consistent*, strict proves *empirically true on tested
trajectories*, matching proves *admissible*.

## 7. Composite contracts (uniform with processes)

A composite *is* a process (`CompositeLink(ProcessLink)`, carrying `interface`/
`bridge`), so it uses the **same `ProcessContract`**. Composite-specific aspects:

- **Boundary face** — derived from its `interface`/`bridge` (the ports it exposes
  at its boundary). Automatic.
- **Declared richer conditions** — a composite *declares* its own `validity`/
  `invariants`/`requires`/`guarantees` (e.g. a coupled cell+environment composite
  guaranteeing "glucose balance closes to 1e-9").
- **Internal-consistency audit** *(composite-only, new)* — a composite is a
  *template whose holes are filled by its members*, so auditing it **reuses §4's
  conformance**: each member's contract must be satisfied by its wiring, and the
  composite's declared boundary contract must be consistent with its members'
  contracts + wiring (the monotonic-narrow law makes this sound). Optionally a
  `derive_contract(composite)` helper synthesizes a candidate boundary contract
  from members (e.g. conservation invariants that provably compose).

So leaf process, composite-as-process, and template-hole are **one contract
type**; a composite additionally gets the internal compositional audit for free
from the matching machinery.

## 8. Workbench surface (audit status + contract on the cards)

The server already runs `build_core` in a subprocess to build the registry and
serves `/api/registry` + `/api/composites`. **Extend those to run `audit_contract`
there** and include, per process/composite: the contract (ports+bounds, validity,
invariants, pre/post, completeness grade) and the audit result (pass / fail /
incomplete / not-declared, with findings).

Both the **Registry entries and the Catalog/composite cards** then render:

- an **audit badge** — ✓ passed / ✗ failed (N findings) / ◐ incomplete / — no
  contract — at a glance;
- an **expandable contract panel** — declared ports with bounds/units, the
  validity domain, invariants, and pre/post, plus the completeness grade; failing
  audits list findings inline.

Lands on the same `composite-card.js` / registry-entry rendering, and applies
uniformly because composites and processes share the contract + the card chrome.

## 9. Repo boundaries & serialization

**Repo split** (follows the existing dependency direction — process-bigraph
depends on bigraph-schema):

- **bigraph-schema** — declaration + schema-level layer: extended `ProcessContract`
  fields; the declarative expression language (parser+evaluator); `narrow`/`amend`
  extended; `face_conforms`/`contract_admits` extended with bound/unit subsumption
  + validity coverage + structural predicate subsumption; `find_candidates`; the
  schema-level `audit_contract` (well-formedness, narrow soundness, completeness).
- **process-bigraph** — engine + process-class-aware layer: the escape-hatch
  methods on the Process class; the implementation-drift + no-op-`update` audit
  checks; the runtime strict-mode wrapper; the CI audit command.
- **vivarium-workbench** — the audit-API extension + card rendering (§8).

**Serialization:** the **declarative** contract (validity/invariants/requires/
guarantees as data) renders into the serialized process/site node — so a remote /
read-only / git-pinned process carries a contract that is still **auditable and
matchable without importing its code**. The escape-hatch callables are marked
*code-only* in the serialized contract (available only where the class is
importable). Contracts carry a version; monotonic `narrow` + versioning keep old
documents valid.

## 10. Phasing

Each phase is shippable, testable, and **backward-compatible** — a process with
no contract is simply unconstrained: audits as "incomplete," matches on face only,
runs normally.

1. **Contract model + static auditor + expression language** (bigraph-schema).
   Processes *and composites* can declare richer contracts; the auditor lints
   them. **Zero runtime/behavior change.** Composite contracts (§7, same model)
   land here.
2. **Matching + conformance extension** — `find_candidates`, the extended
   subsumption, the drift auditor (process-bigraph), the composite
   internal-consistency audit. Holes find + audit candidates.
3. **Runtime strict-mode** — the `{off, raise, record}` wrapper. "Check if it's
   true" runs.
4. **Workbench surface + adoption** — expose `audit_contract` through
   `/api/registry` + `/api/composites`; render the badge + contract panel on
   Registry and Catalog cards; migrate key processes to declare real contracts;
   wire the CI gate; surface `find_candidates` in the Registry/composite explorer.

**Scope note:** Phases 1–3 are the core-framework ask; Phase 4 is the workbench +
adoption effort and could be split into its own plan if desired.

## 11. Non-goals / explicit limits

- **Not a theorem prover.** Predicate *implication* is undecidable; matching on
  invariants/pre-post is structural, and runtime strict-mode is a check on run
  trajectories, not a proof over all inputs.
- **Static drift check is a lint,** not a conformance proof (Python is dynamic).
- **No new hole mechanism.** Reuse `Site`/`fill_sites`/`admits`; do not invent a
  parallel template/slot system. `CompositeSpec`'s `${param}` placeholders are a
  separate, parameter-level front door and are out of scope here.
- **No change to default runtime behavior.** Strict-mode is off by default;
  contracts are additive; unconstrained processes keep working.
- **Local bigraph-schema checkout (1.4.3) is not the target** — current `main`
  (1.6.0+) is; updating the local checkout is a prerequisite task.
