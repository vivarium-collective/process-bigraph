"""
=============
Process Types
=============

This module contains the process methods and types for the process bigraph schema.
Additionally, it defines the `ProcessTypes` class, which extends the `TypeSystem`
class to include process types, and maintains a registry of process types, protocols,
and emitters.
"""
import copy

from bigraph_schema import Registry, Edge, TypeSystem, deep_merge, visit_method, get_path

from process_bigraph.protocols import local_lookup, local_lookup_module
from process_bigraph.composite import Composite
from process_bigraph.emitter import BASE_EMITTERS


# ======================
# Process Type Functions
# ======================

def apply_process(schema, current, update, top_schema, top_state, path, core):
    """Apply an update to a process."""
    process_schema = schema.copy()
    process_schema.pop('_apply')
    return core.apply_update(
        process_schema,
        current,
        update,
        top_schema=top_schema,
        top_state=top_state,
        path=path)


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

    default = core.default(schema)
    deserialized = deep_merge(default, encoded)

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
    else:
        process = deserialized['instance']

    # TODO: this mutating the original value directly into
    #   the return value is weird (?)
    shared = deserialized.get('shared', {})
    deserialized['shared'] = {}
    if shared:
        for step_id, step_config in shared.items():
            step = deserialize_step(
                'step',
                step_config,
                core)

            step['instance'].register_shared(
                process)

            deserialized['shared'][step_id] = step

    deserialized['config'] = config
    deserialized['interval'] = interval
    deserialized['_inputs'] = copy.deepcopy(
        deserialized['instance'].inputs())
    deserialized['_outputs'] = copy.deepcopy(
        deserialized['instance'].outputs())

    return deserialized


def deserialize_step(schema, encoded, core):
    default = core.default(schema)
    deserialized = deep_merge(default, encoded)

    if not deserialized['address']:
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

    if not 'instance' in deserialized:
        process = instantiate(config, core=core)
        deserialized['instance'] = process

    deserialized['config'] = config
    deserialized['_inputs'] = copy.deepcopy(
        deserialized['instance'].inputs())
    deserialized['_outputs'] = copy.deepcopy(
        deserialized['instance'].outputs())

    return deserialized


# ===================
# Process Type System
# ===================

class ProcessTypes(TypeSystem):
    """
    ProcessTypes class extends the TypeSystem class to include process types.
    It maintains a registry of process types and provides methods to register
    new process types, protocols, and emitters.
    """

    def __init__(self):
        super().__init__()
        self.process_registry = Registry()
        self.protocol_registry = Registry()

        self.update_types(PROCESS_TYPES)
        self.register_protocols(BASE_PROTOCOLS)
        self.register_processes(BASE_EMITTERS)

        self.register_process('composite', Composite)


    def register_protocols(self, protocols):
        """Register protocols with the core"""
        self.protocol_registry.register_multiple(protocols)


    def register_process(
            self,
            name,
            process_data
    ):
        """
        Registers a new process type in the process registry.

        Args:
            name (str): The name of the process type.
            process_data: The data associated with the process type.
        """
        self.process_registry.register(name, process_data)


    def register_processes(self, processes):
        for process_key, process_data in processes.items():
            self.register_process(
                process_key,
                process_data)


    def initialize_edge_state(self, schema, path, edge):
        """
        Initialize the state for an edge based on the schema and the edge.
        """
        initial_state = edge['instance'].initial_state()
        if not initial_state:
            return initial_state

        input_ports = copy.deepcopy(get_path(schema, path + ('_inputs',)))
        output_ports = copy.deepcopy(get_path(schema, path + ('_outputs',)))
        ports = {
            '_inputs': input_ports,
            '_outputs': output_ports}

        input_state = {}
        if input_ports:
            input_state = self.project_edge(
                ports,
                edge,
                path[:-1],
                initial_state,
                ports_key='inputs')

        output_state = {}
        if output_ports:
            output_state = self.project_edge(
                ports,
                edge,
                path[:-1],
                initial_state,
                ports_key='outputs')

        state = deep_merge(input_state, output_state)

        return state


PROCESS_TYPES = {
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
        'config': 'quote'},

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
        'config': 'quote',
        'shared': 'map[step]'},
}


BASE_PROTOCOLS = {
    'local': local_lookup}
