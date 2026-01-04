"""
process.py

Schema type definitions and method specializations for process-bigraph.

This module extends bigraph_schema by defining new schema node classes used by process-bigraph.
"""

import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, Empty, Float, Wires, Link, Schema
from bigraph_schema.methods import resolve, realize, realize_link, default, default_link


def float_default(value):
    """Factory-of-a-factory: returns a default_factory that produces Float(_default=value)."""

    def float_factory():
        return Float(_default=value)

    return float_factory


@dataclass(kw_only=True)
class StepLink(Link):
    """A link type used for 'step'-level connectivity."""
    priority: Float = field(default_factory=float_default(0.0))


@dataclass(kw_only=True)
class ProcessLink(Link):
    """
    A link type for temporal processes, extending the base Link schema with a time interval.
    """
    interval: Float = field(default_factory=float_default(1.0))


@dataclass(kw_only=True)
class Bridge(Node):
    """
    Structural node used by `CompositeLink` that declares wiring from inputs and outputs to internal structure.
    """
    inputs: Wires = field(default_factory=Wires)
    outputs: Wires = field(default_factory=Wires)


@dataclass(kw_only=True)
class Interface(Node):
    """
    Declares process I/O schemas: what a process expects (inputs) and produces (outputs).
    """
    inputs: Schema = field(default_factory=Schema)
    outputs: Schema = field(default_factory=Schema)


@dataclass(kw_only=True)
class CompositeLink(ProcessLink):
    """
    Link type for a composite, a process that can have internal structure and processes.

    In addition to being a ProcessLink (and thus having `interval`), composites carry:
      - schema: the schema for the composite state
      - state: a Node describing/holding the composite's state structure
      - interface: explicit I/O schema boundary for the composite
      - bridge: wiring info mapping internal structure to the interface
    """
    schema: Schema = field(default_factory=Schema)
    state: Node = field(default_factory=Node)
    interface: Interface = field(default_factory=Interface)
    bridge: Bridge = field(default_factory=Bridge)


# --- bigraph_schema method specializations -----------------------------------
# These decorators extend the imported multimethod objects (they do not replace them).

@default.dispatch
def default(schema: StepLink):
    """
    Produce a default value for a StepLink.
    """
    link = default_link(schema)

    link['priority'] = default(schema.priority)

    return link


@default.dispatch
def default(schema: ProcessLink):
    """
    Produce a default value for a ProcessLink.
    """
    link = default_link(schema)

    # Use the default of the Float schema node, not a hard-coded numeric literal.
    link['interval'] = default(schema.interval)

    return link


@realize.dispatch
def realize(core, schema: ProcessLink, state, path=()):
    """
    Realize a ProcessLink against a provided state value.
    """
    link_schema, link_state, merges = realize_link(core, schema, state, path=path)

    # Realize the interval field (schema is Float). We pass the incoming value if present.
    _, link_state['interval'], _ = realize(
        core,
        link_schema.interval,
        state.get('interval'),
        path + ('interval',))

    return link_schema, link_state, merges


@realize.dispatch
def realize(core, schema: StepLink, state, path=()):
    """
    Realize a StepLink against a provided state value.
    """
    link_schema, link_state, merges = realize_link(core, schema, state, path=path)

    _, link_state['priority'], _ = realize(
        core,
        link_schema.priority,
        state.get('priority'),
        path + ('priority',))

    return link_schema, link_state, merges


def register_types(core):
    """
    Register process-bigraph schema node classes
    """
    core.register_types({
        'step': StepLink,
        'process': ProcessLink,
        'interface': Interface,
        'bridge': Bridge,
        'composite': CompositeLink})

    return core
