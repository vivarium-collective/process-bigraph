import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, String, Float, Edge
from bigraph_schema.methods import infer, set_default, serialize, deserialize, render, wrap_default


@dataclass(kw_only=True)
class Method(Node):
    instance: object = field(default_factory=object)
    attribute: String = field(default_factory=String)

METHOD_TYPE = type({}.keys)

@infer.dispatch
def infer(core, value: typing.Callable, path: tuple=()):
    import ipdb; ipdb.set_trace()
