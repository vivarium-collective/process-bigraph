"""
Process Types
"""

import sys
import importlib
from bigraph_schema import TypeSystem


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


# TODO: implement these
def apply_process(current, update, bindings=None, types=None):
    pass


def divide_process(value, bindings=None, types=None):
    return value


def serialize_process(value, bindings=None, types=None):
    return value


def deserialize_process(serialized, bindings=None, types=None):
    protocol, address = serialized['address'].split(':', 1)

    if protocol == 'local':
        instantiate = lookup_local(address)
    else:
        raise Exception(f'protocol "{protocol}" not implemented')

    config = types.hydrate_state(
        instantiate.config_schema,
        serialized.get('config', {}))

    # this instance always acts like a process no matter
    # where it is running
    process = instantiate(config)
    deserialized = serialized.copy()
    deserialized['instance'] = process

    return deserialized

    # process.address = serialized['address']
    # process.wires = serialized['wires']
    # return process


process_types = {
    'protocol': {
        '_super': 'string'
    },

    'step': {
        '_super': ['edge'],
        '_apply': 'apply_step',
        '_serialize': 'serialize_step',
        '_deserialize': 'deserialize_step',
        '_divide': 'divide_step',
        '_description': '',
        # TODO: support reference to type parameters from other states
        'address': 'protocol',
        'config': 'tree[any]'},

    'process': {
        '_super': ['step'],
        '_apply': 'apply_process',
        '_serialize': 'serialize_process',
        '_deserialize': 'deserialize_process',
        '_divide': 'divide_process',
        '_description': '',
        # TODO: support reference to type parameters from other states
        'timestep': 'float',
    }
}


def register_process_types(types):
    types.apply_registry.register('apply_process', apply_process)
    types.serialize_registry.register('serialize_process', serialize_process)
    types.deserialize_registry.register('deserialize_process', deserialize_process)
    types.divide_registry.register('divide_process', divide_process)

    for process_key, process_type in process_types.items():
        types.type_registry.register(process_key, process_type)

    return types


class ProcessTypes(TypeSystem):
    def __init__(self):
        super().__init__()
        register_process_types(self)

    # def serialize(self, schema, state):
    #     return ''

    # def deserialize(self, schema, encoded):
    #     return {}

    def hydrate_state(self, schema, state):
        if isinstance(state, str) or '_deserialize' in schema:
            result = self.deserialize(schema, state)
        elif isinstance(state, dict):
            result = {
                key: self.hydrate_state(schema[key], state[key])
                for key, value in state.items()}
        return result

    def hydrate(self, schema, state):
        hydrated = self.hydrate_state(schema, state)
        return self.fill(schema, hydrated)

    def dehydrate(self, schema):
        return {}

    def lookup_address(self, address):
        protocol, config = address.split(':')

        if protocol == 'local':
            self.lookup_local(config)


types = ProcessTypes()
