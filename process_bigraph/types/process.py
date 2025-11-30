import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Float, Link
from bigraph_schema.methods import resolve, deserialize


@dataclass(kw_only=True)
class Step(Link):
    pass


@dataclass(kw_only=True)
class Process(Link):
    interval: Float = field(default_factory=Float)


@serialize.dispatch
def deserialize(core, schema: Process, state):
    link = deserialize_link(core, schema, state)

    import ipdb; ipdb.set_trace()

    link['interval'] = core.fill(
        schema.interval,
        state.get(interval))

    return link
