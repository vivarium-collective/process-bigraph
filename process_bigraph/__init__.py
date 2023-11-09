from process_bigraph.composite import Process, Step, Composite
from process_bigraph.type_system import types
from process_bigraph.protocols import local_lookup
from process_bigraph.registry import protocol_registry, process_registry
from process_bigraph.emitter import ConsoleEmitter, RAMEmitter


# register protocols
protocol_registry.register('local', local_lookup)

# register processes
# TODO
process_registry.register('console-emitter', ConsoleEmitter)
process_registry.register('ram-emitter', RAMEmitter)

