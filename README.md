# Process-Bigraph

[![PyPI](https://img.shields.io/pypi/v/process-bigraph.svg)](https://pypi.org/project/process-bigraph/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Tutorial-brightgreen)](https://vivarium-collective.github.io/process-bigraph/notebooks/process-bigraphs.html)

**Process-Bigraph** is... 

<p align="center">
    <img src="https://github.com/vivarium-collective/process-bigraph/blob/main/doc/_static/process-bigraph.png?raw=true" width="800" alt="Process Bigraph">
</p>

## What are Process Bigraphs?

Process Bigraphs are based on a mathematical formalism introduced by Robin Milner, which was expanded in Vivarium with 
the addition of Processes, and standardized with the introduction of the Schema format. Bigraphs provide a powerful 
framework for compositional modeling due to their ability to represent complex systems through hierarchical structures 
and flexible reconfigurations, thus enabling the seamless composition and decomposition of system components. Variables 
are contained in Stores (circles), which can be embedded in the place graph hierarchy, represented by the dark edges. 
Instead of hyperedges, CBS employs Processes (the rectangles) which have ports (solid black dots) connect via wires 
(dashed edges) to variables within the Stores. Processes are functions that read and write to variables through their 
ports. They can be used to rewrite the bigraph by adding new structure and new processes.

## Getting Started

### Installation

You can install `process-bigraph` using pip:

```console
pip install process-bigraph
```

## Tutorial

To get started with Bigraph-viz, explore our resources: 
* [Process Bigraphs Intro](https://vivarium-collective.github.io/process-bigraph/notebooks/process-bigraphs.html).
* [Bigraph Schema Basics Tutorial](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html). For an introduction to the basic elements of process-bigraph schema.
* [Formatting Tutorial](https://vivarium-collective.github.io/bigraph-viz/notebooks/format.html) for examples
about how to adjust the final look of your bigraph figure.
* [Ecoli](https://raw.githubusercontent.com/vivarium-collective/bigraph-viz/main/doc/_static/ecoli.png) for the wiring
diagraph of a whole-cell E. coli model.

## License

Process-bigraph is open-source software released under the [MIT License](https://github.com/vivarium-collective/process-bigraph/LICENSE).
