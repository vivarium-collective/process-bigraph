"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

import importlib
import sys
from process_bigraph.registry import process_registry
from bigraph_schema.protocols import local_lookup_module


def local_lookup_registry(address):
    """Process Registry Protocol

    Retrieves from the process registry
    """
    return process_registry.access(address)


def local_lookup(address):
    """Local Lookup Protocol

    Retrieves local processes, from the process registry or from a local module
    """
    if address[0] == '!':
        instantiate = local_lookup_module(address[1:])
    else:
        instantiate = local_lookup_registry(address)
    return instantiate

