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

from process_bigraph.protocols import BASE_PROTOCOLS
from process_bigraph.composite import Composite
from process_bigraph.emitter import BASE_EMITTERS


# ======================
# Process Type Functions
# ======================

def apply_process(schema, current, update, top_schema, top_state, path, core):
    """
    Apply an update to a process instance using the core's update mechanism.

    Args:
        schema (dict): The schema for the current process.
        current (dict): The current process state.
        update (dict): The update to apply.
        top_schema (dict): The top-level (composite) schema.
        top_state (dict): The top-level state.
        path (tuple): Path to the current process in the schema tree.
        core: The type system or composition engine.

    Returns:
        dict: Updated process state.
    """
    process_schema = schema.copy()
    process_schema.pop('_apply', None)

    return core.apply_update(
        process_schema,
        current,
        update,
        top_schema=top_schema,
        top_state=top_state,
        path=path
    )


def check_process(schema, state, core):
    """
    Check if a given state belongs to a valid process instance.

    Args:
        schema (dict): The expected schema (unused here).
        state (dict): The process state.
        core: The type system (unused here).

    Returns:
        bool: True if the state contains a valid Edge instance.
    """
    return 'instance' in state and isinstance(state['instance'], Edge)


def fold_visit(schema, state, method, values, core):
    """
    Wrapper for visiting a process state using a specific method.

    Args:
        schema (dict): The process schema.
        state (dict): The process state.
        method (str): The method name to invoke (e.g., 'apply', 'serialize').
        values (dict): Inputs to the visitor method.
        core: The type system.

    Returns:
        Any: Result of the visitor operation.
    """
    return visit_method(schema, state, method, values, core)


def divide_process(schema, state, values, core):
    """
    Divide a process into multiple daughter process states.

    Args:
        schema (dict): The schema for the process.
        state (dict): The current process state.
        values (dict): Division parameters including:
            - 'divisions' (int): Number of daughters
            - 'daughter_configs' (list[dict], optional): Config overrides

    Returns:
        list[dict]: A list of daughter states.
    """
    num_daughters = values['divisions']
    daughter_configs = values.get('daughter_configs', [{} for _ in range(num_daughters)])

    if 'config' not in state:
        return daughter_configs

    existing_config = state['config']
    divisions = []

    for i in range(num_daughters):
        daughter_config = deep_merge(copy.deepcopy(existing_config), daughter_configs[i])

        daughter_state = {
            'address': state['address'],
            'config': daughter_config,
            'inputs': copy.deepcopy(state['inputs']),
            'outputs': copy.deepcopy(state['outputs']),
        }

        if 'interval' in state:
            daughter_state['interval'] = state['interval']

        divisions.append(daughter_state)

    return divisions


def serialize_process(schema, value, core):
    """
    Serialize a process state to a JSON-safe format.

    Args:
        schema (dict): The schema (unused here).
        value (dict): The full process state.
        core: The type system to handle sub-serialization.

    Returns:
        dict: Serialized process data.
    """
    process = value.copy()

    process['config'] = core.serialize(
        process['instance'].config_schema,
        process['config']
    )

    del process['instance']  # Remove the live instance for serialization

    return process


def deserialize_process(schema, encoded, core):
    """
    Deserialize a process from a saved (serialized) state.

    Args:
        schema (dict): The expected process schema.
        encoded (dict): Serialized process state.
        core: The type system, needed to resolve classes and configs.

    Returns:
        dict: Fully rehydrated process state with instance attached.
    """
    encoded = encoded or {}
    schema = schema or {}

    # Base deserialization
    default = core.default(schema)
    deserialized = deep_merge(default, encoded)
    address = deserialized.get('address')

    if not address:
        return deserialized

    # protocol, address = deserialized['address'].split(':', 1)

    # # Determine the process class to instantiate
    # if instance:
    #     instantiate = type(instance)
    # else:
    #     process_lookup = core.protocol_registry.access(protocol)
    #     if not process_lookup:
    #         raise Exception(f'Protocol "{protocol}" not implemented')
    #     instantiate = process_lookup(core, address)
    #     if not instantiate:
    #         raise Exception(f'Process "{address}" not found')

    instantiate = core.parse_protocol(address)

    # Deserialize the configuration
    config = core.deserialize(instantiate.config_schema, deserialized.get('config', {}))
    interval = core.deserialize('interval', deserialized.get('interval'))

    if interval is None:
        interval = core.default(schema.get('interval', 'interval'))

    instance = deserialized.get('instance')
    if not instance:
        instance = instantiate(config, core=core)
        deserialized['instance'] = instance

    # Deserialize shared steps if any
    shared = deserialized.get('shared', {})
    deserialized['shared'] = {}

    for step_id, step_config in shared.items():
        step = deserialize_step('step', step_config, core)
        step['instance'].register_shared(instance)
        deserialized['shared'][step_id] = step

    # Finalize state
    deserialized['config'] = config
    deserialized['interval'] = interval
    deserialized['_inputs'] = copy.deepcopy(instance.inputs())
    deserialized['_outputs'] = copy.deepcopy(instance.outputs())

    return deserialized


def deserialize_step(schema, encoded, core):
    """
    Deserialize a single process step (sub-process in a composite).

    Args:
        schema (str): Schema key, typically 'step'.
        encoded (dict): Serialized step data.
        core: The type system.

    Returns:
        dict: Deserialized step state with process instance.
    """
    default = core.default(schema)
    deserialized = deep_merge(default, encoded)
    address = deserialized.get('address')

    if not deserialized.get('address'):
        return deserialized

    instantiate = core.parse_protocol(address)
    # protocol, address = deserialized['address'].split(':', 1)

    # # Get class or factory function
    # if instance:
    #     instantiate = type(instance)
    # else:
    #     protocol = core.protocol_registry.access(protocol)
    #     if not protocol:
    #         raise Exception(f'Protocol "{protocol}" not implemented')
    #     instantiate = protocol.interface(core, address)
    #     if not instantiate:
    #         raise Exception(f'Process "{address}" not found')

    # Deserialize config and create instance if needed
    config = core.deserialize(instantiate.config_schema, deserialized.get('config', {}))

    instance = deserialized.get('instance')
    if not instance:
        instance = instantiate(config, core=core)
        deserialized['instance'] = instance

    deserialized['config'] = config
    deserialized['_inputs'] = copy.deepcopy(instance.inputs())
    deserialized['_outputs'] = copy.deepcopy(instance.outputs())

    return deserialized


# ===================
# Process Type System
# ===================

class ProcessTypes(TypeSystem):
    """
    Extends the TypeSystem to manage simulation process types,
    including registries for process classes, protocols, and emitters.

    Responsibilities:
    - Registering new process types, protocols, and emitters
    - Initializing edge states for composed processes
    - Providing a default configuration/state template for processes
    """

    def __init__(self):
        super().__init__()

        # Registries to store user-defined and built-in components
        self.process_registry = Registry()
        self.protocol_registry = Registry()

        # Initialize the core type system with known types and protocols
        self.update_types(PROCESS_TYPES)
        self.register_protocols(BASE_PROTOCOLS)
        self.register_processes(BASE_EMITTERS)

        # Explicitly register Composite process type
        self.register_process('composite', Composite)

    def register_protocols(self, protocols):
        """
        Register a dictionary of protocol types with the core type system.

        Args:
            protocols (dict): Mapping of protocol names to protocol definitions.
        """
        self.protocol_registry.register_multiple(protocols)

    def register_process(self, name, process_data):
        """
        Register a new process type into the process registry.

        Args:
            name (str): Unique name for the process.
            process_data: Associated class or factory function for the process.
        """
        self.process_registry.register(name, process_data)

    def register_processes(self, processes):
        """
        Register multiple process types.

        Args:
            processes (dict): Mapping of process names to process data.
        """
        for process_key, process_data in processes.items():
            self.register_process(process_key, process_data)

    def initialize_edge_state(self, schema, path, edge):
        """
        Compute the initial state for a given edge in a composite process.

        This combines the default input and output state projections
        for a process based on its edge mapping and schema.

        Args:
            schema (dict): The complete schema of the composite.
            path (tuple): Path to the process within the schema.
            edge (dict): The edge entry with an instance and port mappings.

        Returns:
            dict: The initial merged state for the edge.
        """
        # Get initial state from the process instance
        initial_state = edge['instance'].initial_state()
        if not initial_state:
            return initial_state

        # Extract and clone port mappings from the schema
        input_ports = copy.deepcopy(get_path(schema, path + ('_inputs',)))
        output_ports = copy.deepcopy(get_path(schema, path + ('_outputs',)))
        ports = {
            '_inputs': input_ports,
            '_outputs': output_ports
        }

        # Project the edge's initial state onto its inputs and outputs
        input_state = self.project_edge(
            ports, edge, path[:-1], initial_state, ports_key='inputs'
        ) if input_ports else {}

        output_state = self.project_edge(
            ports, edge, path[:-1], initial_state, ports_key='outputs'
        ) if output_ports else {}

        return deep_merge(input_state, output_state)

    def parse_protocol(self, address):
        if isinstance(address, str):
            protocol_name, protocol_data = address.split(':', 1)
        else:
            protocol_name = address['protocol']
            protocol_data = address['data']

        protocol = self.protocol_registry.access(protocol_name)
        if not protocol:
            raise Exception(f'Protocol "{protocol_name}" not implemented')

        instantiate = protocol.interface(self, protocol_data)
        if not instantiate:
            raise Exception(f'Process "{address}" not found')

        return instantiate

    def default_state(self, process_class, initial_state=None):
        """
        Construct the default runtime state for a given process class.

        Args:
            process_class (class): The process class to instantiate.
            initial_state (dict, optional): Overrides for the default state.

        Returns:
            dict: The fully constructed default process state.
        """
        # Get default config from the process's schema
        default_config = self.default(process_class.config_schema)

        # Instantiate process with default config
        instance = process_class(default_config, core=self)

        # Build standard process state structure
        state = {
            '_type': 'process',
            'address': f'local:!{process_class.__module__}.{process_class.__name__}',
            'config': default_config,
            'inputs': instance.default_inputs(),
            'outputs': instance.default_outputs()
        }

        # Add default interval if it's a subclass of Process
        if isinstance(process_class, type):
            try:
                from vivarium.core.process import Process  # Delayed import to avoid circularity
                if issubclass(process_class, Process):
                    state['interval'] = 1.0
            except ImportError:
                pass  # Skip interval assignment if Process class is unavailable

        # Apply any user-provided state overrides
        if initial_state:
            state = deep_merge(state, initial_state)

        return state


# ========================
# Process Types Dictionary
# ========================

PROCESS_TYPES = {
    'protocol': {
        '_type': 'protocol',
        '_inherit': 'any'},

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


