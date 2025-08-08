"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

from process_bigraph.protocols.local import local_lookup, LocalProtocol
from process_bigraph.protocols.parallel import ParallelProtocol


BASE_PROTOCOLS = {
    'local': LocalProtocol,
    'parallel': ParallelProtocol}
