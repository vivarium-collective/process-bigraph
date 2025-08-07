"""
===============================================
Protocol for processes running locally
===============================================
"""

from bigraph_schema.protocols import local_lookup_module
from process_bigraph.protocols.protocol import Protocol


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


class LocalProtocol(Protocol):
    @staticmethod
    def interface(core, address):
        if isinstance(address, str):
            return local_lookup(core, address)
        elif isinstance(address, dict):
            if 'address' not in address:
                raise Exception(f'must include address in local protocol: {address}')
            else:
                return local_lookup(core, address['address'])
        else:
            raise Exception(f'address must be str or dict, not {address}')
            
