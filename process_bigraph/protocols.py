"""
Protocol for retrieving processes from address
"""
import importlib
import sys

from process_bigraph.registry import process_registry


def lookup_local(address):
    if '.' in address:
        module_name, class_name = address.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        return getattr(sys.modules[__name__], address)


def lookup_local_process(address, config):
    local = lookup_local(address)
    return local(config)


def lookup_registry(address):
    """Process Registry Protocol

    retrieves address from the process registry
    """
    return process_registry.access(address)


def local_lookup(address):
    if address[0] == '!':
        instantiate = lookup_local(address[1:])
    else:
        instantiate = lookup_registry(address)
    return instantiate

