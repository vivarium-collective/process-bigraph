import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Float, Link
from bigraph_schema.methods import resolve, deserialize, deserialize_link, default, default_link


@dataclass(kw_only=True)
class StepLink(Link):
    pass


@dataclass(kw_only=True)
class ProcessLink(Link):
    interval: Float = field(default_factory=Float)


@dispatch
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
