from process_bigraph.protocols import local_lookup
from process_bigraph.core.registry import protocol_registry, process_registry
from process_bigraph.emitter.utils import ConsoleEmitter, RAMEmitter, DatabaseEmitter


# register protocols
protocol_registry.register('local', local_lookup)

# register processes
# TODO
process_registry.register('console-emitter', ConsoleEmitter)
process_registry.register('ram-emitter', RAMEmitter)
process_registry.register('database-emitter', DatabaseEmitter)

