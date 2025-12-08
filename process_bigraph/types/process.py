import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, Empty, Float, Wires, Link, Schema
from bigraph_schema.methods import resolve, deserialize, deserialize_link, default, default_link


@dataclass(kw_only=True)
class StepLink(Link):
    pass


def float_default(value):
    def float_factory():
        return Float(_default=value)
    return float_factory


@dataclass(kw_only=True)
class ProcessLink(Link):
    interval: Float = field(default_factory=float_default(1.0))


@dataclass(kw_only=True)
class CompositeBridge(Node):
    inputs: Wires = field(default_factory=Wires)
    outputs: Wires = field(default_factory=Wires)


@dataclass(kw_only=True)
class CompositeInterface(Node):
    inputs: Schema = field(default_factory=Schema)
    outputs: Schema = field(default_factory=Schema)


@dataclass(kw_only=True)
class CompositeLink(ProcessLink):
    composition: Schema = field(default_factory=Schema)
    state: Node = field(default_factory=Node)
    interface: CompositeInterface = field(default_factory=CompositeInterface)
    bridge: CompositeBridge = field(default_factory=CompositeBridge)

    
@default.dispatch
def default(schema: ProcessLink):
    link = default_link(schema)

    link['interval'] = default(
        schema.interval)

    return link


@deserialize.dispatch
def deserialize(core, schema: ProcessLink, state, path=()):
    link_schema, link_state, merges = deserialize_link(core, schema, state, path=path)

    _, link_state['interval'], _ = deserialize(
        core,
        link_schema.interval,
        state.get('interval'),
        path+('interval',))

    return link_schema, link_state, merges


def register_types(core):
    core.register_types({
        'step': StepLink,
        'process': ProcessLink,
        'composite': CompositeLink})

    return core
