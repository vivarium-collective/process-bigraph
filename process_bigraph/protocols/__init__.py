"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

# from process_bigraph.protocols.local import local_lookup, LocalProtocol
from process_bigraph.protocols.parallel import ParallelProtocol, load_protocol as load_parallel_protocol
from process_bigraph.protocols.rest import RestProtocol


PROCESS_PROTOCOLS = {
    'parallel': ParallelProtocol,
    'rest': RestProtocol}

# TODO: remove ProcessTypes
BASE_PROTOCOLS = PROCESS_PROTOCOLS


def register_types(core):
    core.register_types(PROCESS_PROTOCOLS)
    return core
