"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

from process_bigraph.protocols.local import local_lookup, LocalProtocol
from process_bigraph.protocols.parallel import ParallelProtocol
from process_bigraph.protocols.docker import DockerProtocol
from process_bigraph.protocols.rest import RestProtocol


BASE_PROTOCOLS = {
    'local': LocalProtocol,
    'parallel': ParallelProtocol,
    'docker': DockerProtocol,
    'rest': RestProtocol}
