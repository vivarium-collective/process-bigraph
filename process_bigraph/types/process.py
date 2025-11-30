import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Float, Link
from bigraph_schema.methods import resolve, deserialize, deserialize_link, default, default_link


@dataclass(kw_only=True)
class Step(Link):
    pass


@dataclass(kw_only=True)
class Process(Link):
    interval: Float = field(default_factory=Float)


@dispatch
def default(schema: Process):
    link = default_link(schema)

    link['interval'] = default(
        schema.interval)

    return link


@deserialize.dispatch
def deserialize(core, schema: Process, state, path=()):
    link_schema, link_state, merges = deserialize_link(core, schema, state, path=path)

    link_state['interval'] = core.fill(
        link_schema.interval,
        state.get('interval'))

    return link_schema, link_state, merges
