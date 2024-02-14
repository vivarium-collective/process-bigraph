"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

import importlib
import sys
from bigraph_schema.protocols import local_lookup_module


def local_lookup_registry(core, address):
    """Process Registry Protocol

    Retrieves from the process registry
    """
    return core.process_registry.access(address)


def local_lookup(core, address):
    """Local Lookup Protocol

    Retrieves local processes, from the process registry or from a local module
    """
    if address[0] == '!':
        instantiate = local_lookup_module(address[1:])
    else:
        instantiate = local_lookup_registry(core, address)
    return instantiate

