# Designing with holes

A **hole** is how you say "a process belongs here, but I haven't written it yet."
It is an open **site** whose `_sort` names a `ProcessContract` — the interface the
missing process must satisfy. You design and review the whole topology with the hole
in place; a real process fills it later, *iff* its own contract is admissible. This
replaces the older `DraftProcess` (a present-but-inert node) with one idea: an
unimplemented slot is a contract-bearing hole, and the framework's filling discipline
(`admits` / `fill_sites`) is what turns it into a running model.

The lifecycle is **register a contract → author the hole → inspect → see what fits →
fill → build**.

```python
from bigraph_schema.assembly import interfaces
from bigraph_schema.contract import ProcessContract
from process_bigraph import (
    allocate_core, Process, Composite,
    open_sites, required_open_sites, is_ground_document,
    find_candidates, template_document, UnfilledHoleError,
)

core = allocate_core()

# 1. Declare the intended interface as a named contract.
core.register_contract("PTHSecretion", ProcessContract(
    summary="senses serum Ca, secretes PTH",
    face={"inputs": {"ca_sense": "float"}, "outputs": {"pth_out": "float"}}))

# 2. Design the topology with the mechanism left as a hole.
template = core.access({"parathyroid": {
    "ca": 2.5, "pth": 0.0,
    "secretion": {"_type": "site", "_sort": "PTHSecretion"}}})

# 3. Inspect: the hole is open, and it blocks a build.
print(open_sites(template))           # [('parathyroid', 'secretion')]
print(required_open_sites(template))  # [('parathyroid', 'secretion')]
print(is_ground_document(template))   # False

# 4. See what fits — any registered process the contract admits.
(_path, site), = interfaces(template)[0]._places
for candidate in find_candidates(core, site):
    print(candidate.match, candidate.address)

# ...register a real process whose contract admits the hole...
class PTHSecretionProcess(Process):
    def inputs(self):  return {"ca_sense": "float"}
    def outputs(self): return {"pth_out": "float"}
    def update(self, state, interval):
        return {"pth_out": max(0.0, 3.0 - state["ca_sense"])}

core.register_link("PTHSecretionProcess", PTHSecretionProcess)
# find_candidates(core, site) now lists ✓ PTHSecretionProcess (full)

# 5. Fill the hole and build.
filler = core.access({"_type": "process", "address": "local:PTHSecretionProcess",
    "config": {}, "interval": 1.0,
    "inputs": {"ca_sense": ["ca"]}, "outputs": {"pth_out": ["pth"]}})
document = template_document(core, template, {"parathyroid/secretion": filler})
sim = Composite({"state": document}, core=core)
sim.run(1.0)
```

## The rejection report

Try to build while the hole is open and you get an `UnfilledHoleError` — a
`ValueError` subclass carrying a structured `.holes` list and a readable message —
instead of an opaque `KeyError: 'data'` deep in realization. Run this **after** the
admissible process is registered (step 4) but **before** the hole is filled (step 5),
so the report can name the candidate:

```python
try:
    Composite({"state": template}, core=core)
except UnfilledHoleError as err:
    print(err)
```

(Before any admissible process is registered the report is identical except the
candidates line reads `(no admissible registered process found)` — the hole path and
its required contract are always shown.)

```
composite document is not ground — required site(s) left unfilled:
'parathyroid/secretion'. An open site is a hole where a process must go; fill it
(bigraph_schema `fill_sites` / `process_bigraph.templates.fill_template`) before
constructing the composite.

  • parathyroid/secretion
      requires:
        inputs: ca_sense: float
        outputs: pth_out: float
      candidates:
        ✓ PTHSecretionProcess (full match)
```

Each `HoleReport` on `err.holes` carries the hole `path`, the required `contract`, and
the `candidates`, so tooling (e.g. the workbench) can render the same report
structurally. Candidates are **best-effort**: `find_candidates` reads a process's
contract by instantiating it with empty config and skips any that need real config, so
the list may be empty — the hole path and its required contract are always present.

## Migrating from `DraftProcess`

`@draft_process` / `DraftProcess` are deprecated and emit
`DraftProcessDeprecationWarning`. To migrate a draft to a hole:

```python
from process_bigraph import draft_to_contract

contract = draft_to_contract(SomeDraft)     # face from DRAFT_INPUTS/OUTPUTS; prose preserved
core.register_contract("SomeDraft", contract)
# replace the `local:SomeDraft` process node with {"_type": "site", "_sort": "SomeDraft"}
```

To silence the deprecation warning narrowly (without muting unrelated deprecations):

```python
import warnings
from process_bigraph import DraftProcessDeprecationWarning
warnings.filterwarnings("ignore", category=DraftProcessDeprecationWarning)
```

## Downstream docs to update (P2 doc sweep)

Workspaces and guides still describing the `DraftProcess` node model, to be rewritten
to holes-in-place when their drafts are migrated (P2):

- viva-docs: `docs/compute/templates-and-draft-processes.md`, `docs/reference/glossary.md`,
  `docs/foundations/core-concepts.md`
- viva-superpowers: `skills/viva-model-build/SKILL.md`,
  `docs/concepts/composites-templates-and-the-study-investigation-stack.md`
- meta-modelers-guide, viva-littleb: draft-based composites + their concept docs
