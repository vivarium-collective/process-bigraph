import pprint
from process_bigraph.protocols import local_lookup
from process_bigraph.registry import protocol_registry, process_registry, register_process
from process_bigraph.emitter import ConsoleEmitter, RAMEmitter, DatabaseEmitter
from process_bigraph.composite import Process, Step, Composite
from process_bigraph.type_system import ProcessTypes, core


__all__ = [
    'Process',
    'Step',
    'Composite',
    'core',
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
