"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

# from process_bigraph.protocols.local import local_lookup, LocalProtocol
from process_bigraph.protocols.parallel import ParallelProtocol, load_protocol as load_parallel_protocol
from process_bigraph.protocols.rest import RestProtocol
from process_bigraph.protocols.ray import RayProtocol


PROCESS_PROTOCOLS = {
    'parallel': ParallelProtocol,
    'rest': RestProtocol,
    'ray': RayProtocol}


def register_types(core):
    core.register_types(PROCESS_PROTOCOLS)
    return core
