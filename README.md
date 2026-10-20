# Process-Bigraph

[![PyPI](https://img.shields.io/pypi/v/process-bigraph.svg)](https://pypi.org/project/process-bigraph/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Tutorials-brightgreen)](https://vivarium-collective.github.io/process-bigraph/notebooks/index.html)

**Process-Bigraph** is a compositional runtime and protocol for building and executing
**multiscale biological models from interoperable processes**.

It provides a shared architectural layer for:
- declaring **process interfaces**
- wiring processes through **typed shared state**
- orchestrating execution across **heterogeneous timescales**
- supporting **dynamic structure** (workflows, division, graph rewrites)

Process-Bigraph is the execution core of **Vivarium 2.0**, designed to integrate models
built with different formalisms—including ODEs, FBA, agent-based models, spatial solvers,
and machine-learning components—into a single coherent simulation.

<p align="center">
  <img src="https://github.com/vivarium-collective/process-bigraph/blob/main/docs/_static/composition_framework.png?raw=true"
       width="800"
       alt="Process Bigraph composition framework">
</p>

---

## 🧩 What is a Process Bigraph?

A **process bigraph** combines:

- **Typed stores** — hierarchical, schema-validated state defined with
  [**bigraph-schema**](https://github.com/vivarium-collective/bigraph-schema)
- **Processes** — executable components with explicit input/output ports
- **Composites** — encapsulated sub-simulations with their own internal structure
- **Orchestration patterns** — multi-timestepping, directed workflows, and event-driven rewrites

Processes do **not** mutate state directly.
Instead, they emit **typed deltas** that are merged by the runtime.

This allows:
- numerical updates
- structural rewrites
- scheduling and orchestration

to coexist under a single execution semantics.

In this sense, Process-Bigraph is a **composition protocol**, not a domain-specific simulator.

---

## 📄 Paper reference

The conceptual framework and formal semantics of process bigraphs are introduced in:

> **Agmon, E. & Spangler, R. K.**  
> *Process Bigraphs and the Architecture of Compositional Systems Biology*  
> https://arxiv.org/abs/2512.23754

---

## 🚀 Getting Started

### Installation

```console
pip install process-bigraph
```

### Quickstart — composites, drafts & templates in code

A runnable five-minute tour of the core concepts. Every block below runs as-is
against `process-bigraph ≥ 1.8.3`. Concepts explained in
[`docs/concepts/composites-and-templates.md`](docs/concepts/composites-and-templates.md).

**1. A process wired into a composite over a shared store.** A composite is a `state`
map of typed nodes; processes couple only by reading/writing shared stores.

```python
from process_bigraph import Composite, Process, allocate_core
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

class Grow(Process):                                  # a Process = ports + an update
    config_schema = {'rate': 'float'}
    def inputs(self):  return {'level': 'float'}
    def outputs(self): return {'level': 'float'}
    def update(self, state, interval):
        return {'level': state['level'] * self.config['rate'] * interval}  # a delta

core = allocate_core()
core.register_link('Grow', Grow)                      # register it in the local registry

composite = Composite({'state': {
    'level': 1.0,                                     # a shared store
    'grow': {'_type': 'process', 'address': 'local:Grow', 'config': {'rate': 0.5},
             'interval': 1.0,                         # seconds; a unit string like '1 min' also works
             'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}},
    'emitter': emitter_from_wires({'level': ['level'], 'time': ['global_time']}),
}}, core=core)
composite.run(5.0)
print(gather_emitter_results(composite))
# {('emitter',): [{'level': 1.0, 'time': 0.0}, {'level': 1.5, ...}, ... {'level': 7.59, 'time': 5.0}]}
```

**2. A draft — an open hole whose sort is a contract.** A not-yet-implemented
process-slot is a *hole*: an open **site** (step 3) whose `_sort` names a
`ProcessContract` (its typed face + conditions). You design and review the topology
with the hole in place, then fill it with any process the contract *admits*.
`Composite` refuses to build while a required hole is open, and reports **which** hole,
**what** contract it requires, and **which** registered processes would fill it.

```python
from bigraph_schema.contract import ProcessContract
from process_bigraph import find_candidates, UnfilledHoleError

# register the intended interface as a named contract …
core.register_contract('PTHSecretion', ProcessContract(
    summary='senses serum Ca, secretes PTH',
    face={'inputs': {'ca_sense': 'float'}, 'outputs': {'pth_out': 'float'}}))

# … and leave the slot as a hole sorted by it
template = core.access({'parathyroid': {'ca': 2.5, 'pth': 0.0,
    'secretion': {'_type': 'site', '_sort': 'PTHSecretion'}}})

# Composite(..., unfilled) → UnfilledHoleError, not an opaque KeyError:
#   not ground — required site(s) left unfilled: 'parathyroid/secretion'
#     • parathyroid/secretion
#         requires:
#           inputs: ca_sense: float
#           outputs: pth_out: float
#         candidates:
#           ✓ <any registered process whose contract admits the hole>
```

Fill it (step 3) with a process `find_candidates(core, site)` reports as admissible,
and build. The full design → inspect → fill → build lifecycle — and the rejection
report in full — is in
[docs/concepts/designing-with-holes.md](docs/concepts/designing-with-holes.md).

> **Deprecated:** `@draft_process` / `DraftProcess` (an inert placeholder *node*) still
> work but now emit `DraftProcessDeprecationWarning` — model a draft as a hole instead.
> `draft_to_contract(cls)` bridges an existing draft's interface to a `ProcessContract`.

**3. A template — a composite with an open site (hole).** A template is a document that
isn't *ground*: it has open **sites** (`{"_type": "site"}`). `Composite` refuses to run
one until every required site is filled.

```python
from process_bigraph import Step
from process_bigraph.templates import open_sites, is_ground_document, template_document

class ReportCard(Step):                               # a fixed downstream verdict
    config_schema = {'threshold': 'float'}
    def inputs(self):  return {'level': 'float'}
    def outputs(self): return {'verdict': 'string'}
    def update(self, state):
        return {'verdict': 'pass' if state['level'] >= self.config['threshold'] else 'fail'}

core.register_link('ReportCard', ReportCard)
MODEL_FACE = {'_type': 'link', '_inputs': {'level': 'float'}, '_outputs': {'level': 'float'}}

template = core.access({'study': {                     # analysis fixed, the model is a HOLE
    'level': 1.0, 'verdict': 'string',
    'model':  {'_type': 'site', '_sort': MODEL_FACE},
    'report': {'_type': 'step', 'address': 'local:ReportCard', 'config': {'threshold': 2.0},
               'inputs': {'level': ['level']}, 'outputs': {'verdict': ['verdict']}}}})

print(open_sites(template), is_ground_document(template))   # [('study', 'model')] False

def model(rate):                                      # any conforming composite fits the hole
    return core.access({'_type': 'process', 'address': 'local:Grow', 'config': {'rate': rate},
                        'interval': 1.0, 'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}})

sim = Composite({'state': template_document(core, template, {'study/model': model(0.5)})}, core=core)
sim.run(4.0)
print(sim.state['study']['verdict'])                  # pass  (a slower model → fail)
# template_document(core, template, {}) → UnfilledHoleError: not ground — required site 'study/model'
```

**4. An investigation — one site per member study; gating is *filling*.** Fill a
member's site to admit it; leave it open and it is pruned from the built document, so a
blocked prerequisite simply never runs.

```python
from process_bigraph.templates import investigation_document

study = lambda: {'level': 1.0, 'verdict': 'string',
    'model':  {'_type': 'site', '_sort': MODEL_FACE},
    'report': {'_type': 'step', 'address': 'local:ReportCard', 'config': {'threshold': 2.0},
               'inputs': {'level': ['level']}, 'outputs': {'verdict': ['verdict']}}}

inv = core.access({'investigation': {'study_A': study(), 'study_B': study()}})
document, blocked = investigation_document(
    core, inv, {'investigation/study_A/model': model(0.5)}, member_depth=2)  # fill A only
print(blocked, sorted(document['investigation']))     # ['investigation/study_B'] ['study_A']
```

> This same template/site machinery is what viva-superpowers uses to compile a whole
> **investigation into a composite** (one `StudyStep` per member). The study/investigation
> layer is documented in viva-superpowers
> `docs/concepts/composites-templates-and-the-study-investigation-stack.md`.

## 📘 Tutorials

The Process-Bigraph tutorials are executable Jupyter notebooks,
rendered to HTML and published automatically on GitHub Pages.

- 📚 **Tutorial Index (all tutorials)**  
  https://vivarium-collective.github.io/process-bigraph/notebooks/index.html

### Learning Path (Featured Tutorials)

- **Tutorial 0 — A Process in 12 Lines (quickstart)**  
  *The new front door: define a process with the `@process` decorator, wire and
  run it end-to-end on one screen, and watch a `_units`-bearing port auto-convert*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_0_quickstart.html

- **Tutorial 1 — Process-Bigraph Basics**  
  *Processes, Steps, ports, Composites, workflows, and emitters*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_1.html

- **Tutorial 2 — Wrapping an ODE Solver (`odeint`)**  
  *How to expose an existing scientific API as a Process*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_2.html

- **Tutorial 3 — Declarative Math**  
  *Defining mathematical relationships, signal pipelines, and events using `MathExpressionStep`*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_3.html  
  Requires the companion package [`viva-basic-processes`](https://github.com/vivarium-collective/viva-basic-processes), published to PyPI as `pbg-basic-processes` (`pip install "pbg-basic-processes[plotting]>=0.3.0"`), which provides `MathExpressionStep`, the `Tick` process, and the plotting helpers.

- **Tutorial 4 — Composing a Biological Model**  
  *The central dogma from four small processes: composition over shared molecular state, biological units in the port types, and adding regulation by adding a wire*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_4.html

More tutorials are added continuously and appear automatically in the index.

### Architecture

- **The framework, end to end** — *start here to understand the whole system:
  what the objects are (documents, sites, handles), how they compose, the
  higher-order DAG, templates and gating, content-addressed artifacts, the
  `git:` protocol, and the laws everything else follows from.*
  [docs/architecture.md](docs/architecture.md)

### Topic Guides

- **Emitters — Recording Simulation Results**
  *Built-in emitters (RAM, console, JSON, SQLite), how to wire them, retrieve results, store runs long-term, and write your own.*
  [docs/emitters.md](docs/emitters.md)

- **Tick lifecycle** — *how a step network is ordered and advanced, and how a
  protocol runtime batches remote dispatch.*
  [docs/tick_lifecycle.md](docs/tick_lifecycle.md)

- **Distributed lifecycles** — *running parts of a composite off-process.*
  [docs/distributed_lifecycles.md](docs/distributed_lifecycles.md)

- **Deploying a Step network to Nextflow** — *opt-in: compile a composite's
  Step network into a Nextflow DSL2 workflow and run it on a batch backend
  (local / SLURM). Nothing changes about `Composite.run()`.*
  [docs/nextflow.md](docs/nextflow.md)

---

## 🧪 Reference Implementation: spatio-flux

Process-Bigraph is exercised end-to-end in **spatio-flux**, a multiscale reference
model built entirely using the process-bigraph protocol.

spatio-flux composes spatial fields, particle dynamics, and metabolic processes
using typed shared state and declarative orchestration.

GitHub: https://github.com/vivarium-collective/spatio-flux  
Live test report: https://vivarium-collective.github.io/spatio-flux/report/index.html

---

## 🔗 Related Resources

- **Bigraph Schema Basics**  
  https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html  
  *Introduction to the schema language underlying Process-Bigraph*

- **Visualization of Bigraph Document** — diagramming and rendering with
  [**bigraph-viz**](https://github.com/vivarium-collective/bigraph-viz)  
  https://vivarium-collective.github.io/bigraph-viz/notebooks/format.html

- **E. coli Whole-Cell Wiring Diagram**  
  https://raw.githubusercontent.com/vivarium-collective/bigraph-viz/main/doc/_static/ecoli.png

- **Claude Code skills** — wrap simulators, compose `pbg-*` packages, and
  scaffold process-bigraph research workspaces with the
  [**pbg-superpowers**](https://github.com/vivarium-collective/pbg-superpowers)
  Claude Code plugin (install with `/plugin install pbg-superpowers`).

---

## 📜 License

Process-Bigraph is open-source software released under the  
[Apache 2 License](https://github.com/vivarium-collective/process-bigraph/blob/main/LICENSE).
