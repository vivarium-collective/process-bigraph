"""
===============================================
Protocols for retrieving processes from address
===============================================
"""

import importlib
import sys
from process_bigraph.composite import Process
from process_bigraph.registry import protocol_registry

import ray


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


@ray.remote
class RayProcess(Process):
    pass


class RayProcessFactory:
    def __init__(self, instantiate):
        self.instantiate = instantiate


    def __call__(self, config):
        self.instantiate(config)


def ray_lookup(address):
    """Ray Lookup Protocol

    Retrieves processes that operate through Ray's distributed actor system, from the process registry or from a local module
    """
    if address[0] == '!':
        instantiate = local_lookup_module(address[1:])
    else:
        instantiate = local_lookup_registry(address)

    factory = RayProcessFactory(instantiate)
    return factory


protocol_registry.register('ray', ray_lookup)


