import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, String, Float, Edge
from bigraph_schema.methods import infer, set_default, serialize, deserialize, render, wrap_default

from vivarium.core.process import Process as VivariumProcess, Step as VivariumStep
from process_bigraph import Step as BigraphStep, Process as BigraphProcess


@dataclass(kw_only=True)
class Protocol(Node):
    protocol: String = field(default_factory=String)
    data: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class FunctionEdge(Edge):
    address: Protocol = field(default_factory=Protocol)
    config: Node = field(default_factory=Node)

@dataclass(kw_only=True)
class StepEdge(FunctionEdge):
    pass

@dataclass(kw_only=True)
class ProcessEdge(FunctionEdge):
    interval: Float = field(default_factory=Float)


