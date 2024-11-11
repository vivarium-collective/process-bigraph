"""
This module contains the process methods for the process bigraph schema.
"""

import copy

from bigraph_schema import Edge, deep_merge, visit_method


def apply_process(schema, current, update, core):
    """Apply an update to a process."""
    process_schema = schema.copy()
    process_schema.pop('_apply')
    return core.apply(
        process_schema,
        current,
        update)


def check_process(schema, state, core):
    """Check if this is a process."""
    return 'instance' in state and isinstance(
        state['instance'],
        Edge)


def fold_visit(schema, state, method, values, core):
    visit = visit_method(
        schema,
        state,
        method,
        values,
        core)

    return visit


def divide_process(schema, state, values, core):
    # daughter_configs must have a config per daughter

    daughter_configs = values.get(
        'daughter_configs',
        [{} for index in range(values['divisions'])])

    if 'config' not in state:
        return daughter_configs

    existing_config = state['config']

    divisions = []
    for index in range(values['divisions']):
        daughter_config = copy.deepcopy(
            existing_config)
        daughter_config = deep_merge(
            daughter_config,
            daughter_configs[index])

        # TODO: provide a way to override inputs and outputs
        daughter_state = {
            'address': state['address'],
            'config': daughter_config,
            'inputs': copy.deepcopy(state['inputs']),
            'outputs': copy.deepcopy(state['outputs'])}

        if 'interval' in state:
            daughter_state['interval'] = state['interval']

        divisions.append(daughter_state)

    return divisions


def serialize_process(schema, value, core):
    """Serialize a process to a JSON-safe representation."""
    # TODO -- need to get back the protocol: address and the config
    process = value.copy()
    process['config'] = core.serialize(
        process['instance'].config_schema,
        process['config'])
    del process['instance']
    return process


def deserialize_process(schema, encoded, core):
    """Deserialize a process from a serialized state.

    This function is used by the type system to deserialize a process.

    :param encoded: A JSON-safe representation of the process.
    :param bindings: The bindings to use for deserialization.
    :param core: The type system to use for deserialization.

    :returns: The deserialized state with an instantiated process.
    """
    encoded = encoded or {}
    schema = schema or {}

    if not encoded:
        deserialized = core.default(schema)
    else:
        deserialized = encoded.copy()

    if not deserialized.get('address'):
        return deserialized

    protocol, address = deserialized['address'].split(':', 1)

    if 'instance' in deserialized:
        instantiate = type(deserialized['instance'])
    else:
        process_lookup = core.protocol_registry.access(protocol)
        if not process_lookup:
            raise Exception(f'protocol "{protocol}" not implemented')

        instantiate = process_lookup(core, address)
        if not instantiate:
            raise Exception(f'process "{address}" not found')

    config = core.deserialize(
        instantiate.config_schema,
        deserialized.get('config', {}))

    interval = core.deserialize(
        'interval',
        deserialized.get('interval'))

    if interval is None:
        interval = core.default(
            schema.get(
                'interval',
                'interval'))

    if not 'instance' in deserialized:
        process = instantiate(
            config,
            core=core)

        deserialized['instance'] = process

    deserialized['config'] = config
    deserialized['interval'] = interval
    deserialized['_inputs'] = deserialized['instance'].inputs()
    deserialized['_outputs'] = deserialized['instance'].outputs()

    return deserialized


def deserialize_step(schema, encoded, core):
    deserialized = encoded.copy()
    if not encoded['address']:
        return encoded

    protocol = encoded['address'].split(':', 1)
    if len(protocol) == 1:
        return encoded

    protocol, address = encoded['address'].split(':', 1)

    if 'instance' in deserialized:
        instantiate = type(deserialized['instance'])
    else:
        process_lookup = core.protocol_registry.access(protocol)
        if not process_lookup:
            raise Exception(f'protocol "{protocol}" not implemented')

        instantiate = process_lookup(core, address)
        if not instantiate:
            raise Exception(f'process "{address}" not found')

    config = core.deserialize(
        instantiate.config_schema,
        encoded.get('config', {}))

    if not 'instance' in deserialized:
        process = instantiate(config, core=core)
        deserialized['instance'] = process

    deserialized['config'] = config
    deserialized['_inputs'] = deserialized['instance'].inputs()
    deserialized['_outputs'] = deserialized['instance'].outputs()

    return deserialized


"""
Process Types
-------------
This section contains the process types schema
"""

process_types = {
    'protocol': {
        '_type': 'protocol',
        '_inherit': 'string'},

    'emitter_mode': 'enum[none,all,stores,bridge,paths,ports]',

    'interval': {
        '_type': 'interval',
        '_inherit': 'float',
        '_apply': 'set',
        '_default': '1.0'},

    'step': {
        '_type': 'step',
        '_inherit': 'edge',
        '_apply': apply_process,
        '_serialize': serialize_process,
        '_deserialize': deserialize_step,
        '_check': check_process,
        '_fold': fold_visit,
        '_divide': divide_process,
        '_description': '',
        # TODO: support reference to type parameters from other states
        'address': 'protocol',
        'config': 'tree[any]'},

    # TODO: slice process to allow for navigating through a port
    'process': {
        '_type': 'process',
        '_inherit': 'edge',
        '_apply': apply_process,
        '_serialize': serialize_process,
        '_deserialize': deserialize_process,
        '_check': check_process,
        '_fold': fold_visit,
        '_divide': divide_process,
        '_description': '',
        # TODO: support reference to type parameters from other states
        'interval': 'interval',
        'address': 'protocol',
        'config': 'tree[any]'},
}
