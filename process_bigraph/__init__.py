from process_bigraph.composite import Process, Step, Composite
from process_bigraph.type_system import types
from process_bigraph.protocols import local_lookup
from process_bigraph.registry import protocol_registry, process_registry


# register protocols
protocol_registry.register('local', local_lookup)

# register processes
# TODO