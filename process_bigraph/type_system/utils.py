"""
=============
Process Types
=============
"""


from process_bigraph.core.registry import protocol_registry



process_interval_schema = {
    '_type': 'float',
    '_apply': 'set',
    '_default': '1.0'}


# TODO: implement these
def apply_process(current, update, bindings=None, types=None):
    process_schema = dict(types.access('process'))
    process_schema.pop('_apply')
    return types.apply(
        process_schema,
        current,
        update)


def divide_process(value, bindings=None, types=None):
    return value


def serialize_process(value, bindings=None, types=None):
    # TODO -- need to get back the protocol: address and the config
    return value


DEFAULT_INTERVAL = 1.0


TYPE_SCHEMAS = {
    'float': 'float'}


def deserialize_process(serialized, bindings=None, types=None):
    deserialized = serialized.copy()
    protocol, address = serialized['address'].split(':', 1)

    if 'instance' in deserialized:
        instantiate = type(deserialized['instance'])
    else:
        process_lookup = protocol_registry.access(protocol)
        if not process_lookup:
            raise Exception(f'protocol "{protocol}" not implemented')

        instantiate = process_lookup(address)
        if not instantiate:
            raise Exception(f'process "{address}" not found')

    config = types.hydrate_state(
        instantiate.config_schema,
        serialized.get('config', {}))

    interval = types.deserialize(
        process_interval_schema,
        serialized.get('interval'))

    if not 'instance' in deserialized:
        process = instantiate(config)
        deserialized['instance'] = process

    deserialized['config'] = config
    deserialized['interval'] = interval

    return deserialized


def deserialize_step(serialized, bindings=None, types=None):
    deserialized = serialized.copy()
    protocol, address = serialized['address'].split(':', 1)

    if 'instance' in deserialized:
        instantiate = type(deserialized['instance'])
    else:
        process_lookup = protocol_registry.access(protocol)
        if not process_lookup:
            raise Exception(f'protocol "{protocol}" not implemented')

        instantiate = process_lookup(address)
        if not instantiate:
            raise Exception(f'process "{address}" not found')

    config = types.hydrate_state(
        instantiate.config_schema,
        serialized.get('config', {}))

    if not 'instance' in deserialized:
        process = instantiate(config)
        deserialized['instance'] = process

    deserialized['config'] = config

    return deserialized


process_types = {
    'protocol': {
        '_super': 'string'},

    # TODO: step wires are directional ie, we make a distinction
    #   between inputs and outputs, and the same wire cannot be both
    #   an input and output at the same time
    'step': {
        '_super': ['edge'],
        '_apply': 'apply_process',
        '_serialize': 'serialize_process',
        '_deserialize': 'deserialize_step',
        '_divide': 'divide_process',
        '_description': '',
        # TODO: support reference to type parameters from other states
        'address': 'protocol',
        'config': 'tree[any]'},

    'process': {
        '_super': ['edge'],
        '_apply': 'apply_process',
        '_serialize': 'serialize_process',
        '_deserialize': 'deserialize_process',
        '_divide': 'divide_process',
        '_description': '',
        # TODO: support reference to type parameters from other states
        'interval': process_interval_schema,
        'address': 'protocol',
        'config': 'tree[any]'},
}


def register_process_types(types):
    types.apply_registry.register('apply_process', apply_process)
    types.serialize_registry.register('serialize_process', serialize_process)
    types.deserialize_registry.register('deserialize_process', deserialize_process)
    types.divide_registry.register('divide_process', divide_process)

    types.deserialize_registry.register('deserialize_step', deserialize_step)

    for process_key, process_type in process_types.items():
        types.type_registry.register(process_key, process_type)

    return types


