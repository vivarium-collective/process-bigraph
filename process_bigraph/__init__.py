import pprint
from core_processes.cobra_process import CobraProcess
from core_processes.copasi_process import CopasiProcess
from core_processes.smoldyn_process import SmoldynProcess
from core_processes.tellurium_process import TelluriumProcess, TelluriumStep
from process_bigraph.protocols import local_lookup
from process_bigraph.registry import protocol_registry, process_registry
from process_bigraph.emitter import ConsoleEmitter, RAMEmitter, DatabaseEmitter
from process_bigraph.composite import Process, Step, Composite
from process_bigraph.type_system import types


__all__ = [
    'Process',
    'Step',
    'Composite',
    'types',
    'pp',
    'pf',
    'protocol_registry',
    'process_registry'
]


pretty = pprint.PrettyPrinter(indent=2)


def pp(x):
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x):
    """Format ``x`` for display."""
    return pretty.pformat(x)


# register protocols
protocol_registry.register('local', local_lookup)

# register emitters
process_registry.register('console-emitter', ConsoleEmitter)
process_registry.register('ram-emitter', RAMEmitter)
process_registry.register('database-emitter', DatabaseEmitter)
