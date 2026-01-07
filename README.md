# Process-Bigraph

[![PyPI](https://img.shields.io/pypi/v/process-bigraph.svg)](https://pypi.org/project/process-bigraph/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Tutorials-brightgreen)](https://vivarium-collective.github.io/process-bigraph/notebooks/index.html)

**Process-Bigraph** is a compositional runtime and protocol for building and executing multiscale models from interoperable processes.

Rather than defining a single modeling method, process-bigraph provides a shared architectural layer for declaring process interfaces, wiring processes through typed shared state, orchestrating execution across heterogeneous timescales, and supporting dynamic structural change such as workflows, division, and graph rewrites. It is the execution core of **Vivarium 2.0**, designed to integrate models built with different formalisms—including ODEs, FBA, agent-based models, spatial solvers, and machine-learning components—into a single coherent simulation.

<p align="center"> <img src="https://github.com/vivarium-collective/process-bigraph/blob/main/doc/_static/composition_framework.png?raw=true" width="800" alt="Process Bigraph"> </p>


## What is a Process Bigraph?

A process bigraph combines typed **stores** (hierarchical state), executable **processes** with explicit input/output ports, **composites** that encapsulate entire sub-simulations behind an interface, and **orchestration patterns** such as multi-timestepping, directed workflows, and event-driven graph rewrites.

Processes do not mutate state directly. Instead, they emit typed deltas that are merged by the runtime, allowing numerical updates, structural rewrites, and scheduling to coexist under a single execution semantics. This makes process-bigraph a **composition protocol**, not a domain-specific simulator.

The conceptual framework and formal semantics of process bigraphs are introduced in:

> Agmon, E. & Spangler, R. K. *Process Bigraphs and the Architecture of Compositional Systems Biology*.  
> https://arxiv.org/abs/2512.23754

## Getting Started

### Installation

```console
pip install process-bigraph
```

## Tutorials

## Tutorials

The Process-Bigraph tutorials are executable Jupyter notebooks rendered to HTML
and published automatically on GitHub Pages.

- 📚 **Tutorial Index (all tutorials)**  
  https://vivarium-collective.github.io/process-bigraph/notebooks/index.html

### Featured Tutorials

- **Tutorial 1 — Process-Bigraph Basics**  
  Processes, Steps, ports, Composites, workflows, and emitters  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_1.html

- **Tutorial 2 — Wrapping an ODE Solver (odeint)**  
  How to expose an existing scientific API as a Process  
  https://vivarium-collective.github.io/process-bigraph/notebooks/tutorial_2.html


### Related Resources

- **Bigraph Schema Basics Tutorial**  
  https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html  

- **Formatting Tutorial**  
  https://vivarium-collective.github.io/bigraph-viz/notebooks/format.html  

- **E. coli Whole-Cell Wiring Diagram**  
  https://raw.githubusercontent.com/vivarium-collective/bigraph-viz/main/doc/_static/ecoli.png


## License

process-bigraph is open-source software released under the [Apache 2 License](https://github.com/vivarium-collective/process-bigraph/blob/main/LICENSE).
