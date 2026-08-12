"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

from process_bigraph.protocols.parallel import ParallelProtocol, load_protocol as load_parallel_protocol
from process_bigraph.protocols.rest import RestProtocol
from process_bigraph.protocols.ray import RayProtocol
from process_bigraph.protocols.git import GitProtocol


PROCESS_PROTOCOLS = {
    'parallel': ParallelProtocol,
    'rest': RestProtocol,
    'ray': RayProtocol,
    'git': GitProtocol}


def register_types(core):
    core.register_types(PROCESS_PROTOCOLS)
    return core


__all__ = [
    'ParallelProtocol',
    'load_parallel_protocol',
    'RestProtocol',
    'RayProtocol',
    'GitProtocol',
    'PROCESS_PROTOCOLS',
    'register_types',
]
