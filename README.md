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
  <img src="https://github.com/vivarium-collective/process-bigraph/blob/main/doc/_static/composition_framework.png?raw=true"
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

## 📘 Tutorials

The Process-Bigraph tutorials are executable Jupyter notebooks,
rendered to HTML and published automatically on GitHub Pages.

- 📚 **Tutorial Index (all tutorials)**  
  https://vivarium-collective.github.io/process-bigraph/notebooks/index.html

### Learning Path (Featured Tutorials)

- **Tutorial 1 — Process-Bigraph Basics**  
  *Processes, Steps, ports, Composites, workflows, and emitters*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_1.html

- **Tutorial 2 — Wrapping an ODE Solver (`odeint`)**  
  *How to expose an existing scientific API as a Process*  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_2.html

More tutorials are added continuously and appear automatically in the index.

---

## 🔗 Related Resources

- **Bigraph Schema Basics**  
  https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html  
  *Introduction to the schema language underlying Process-Bigraph*

- **Formatting & Visualization** — diagramming and rendering with
  [**bigraph-viz**](https://github.com/vivarium-collective/bigraph-viz)  
  https://vivarium-collective.github.io/bigraph-viz/notebooks/format.html

- **E. coli Whole-Cell Wiring Diagram**  
  https://raw.githubusercontent.com/vivarium-collective/bigraph-viz/main/doc/_static/ecoli.png

---

## 📜 License

Process-Bigraph is open-source software released under the  
[Apache 2 License](https://github.com/vivarium-collective/process-bigraph/blob/main/LICENSE).
