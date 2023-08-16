"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

import importlib
import sys
from process_bigraph.registry import process_registry


def local_lookup_module(address):
    """Local Module Protocol

    Retrieves local module
    """
    if '.' in address:
        module_name, class_name = address.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        return getattr(sys.modules[__name__], address)


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

