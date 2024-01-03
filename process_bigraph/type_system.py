"""
=============
Process Types
=============
"""

from bigraph_schema import Edge, TypeSystem, get_path, establish_path, set_path
from process_bigraph.registry import protocol_registry


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


def check_process(state, bindings, types):
    return 'instance' in state and isinstance(
        state['instance'],
        Edge)


def divide_process(value, bindings=None, types=None):
    return value


def serialize_process(value, bindings=None, types=None):
    """Serialize a process to a JSON-safe representation."""
    # TODO -- need to get back the protocol: address and the config
    process = value.copy()
    del process['instance']
    return process


DEFAULT_INTERVAL = 1.0


TYPE_SCHEMAS = {
    'float': 'float'}


def deserialize_process(serialized, bindings=None, types=None):
    """Deserialize a process from a serialized state.

    This function is used by the type system to deserialize a process.

    :param serialized: A JSON-safe representation of the process.
    :param bindings: The bindings to use for deserialization.
    :param types: The type system to use for deserialization.

    :returns: The deserialized state with an instantiated process.
    """
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

    'step': {
        '_super': ['edge'],
        '_apply': 'apply_process',
        '_serialize': 'serialize_process',
        '_deserialize': 'deserialize_step',
        '_check': 'check_process',
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
        '_check': 'check_process',
        '_divide': 'divide_process',
        '_description': '',
        # TODO: support reference to type parameters from other states
        'interval': process_interval_schema
    }
}


def register_process_types(types):
    types.apply_registry.register('apply_process', apply_process)
    types.serialize_registry.register('serialize_process', serialize_process)
    types.deserialize_registry.register('deserialize_process', deserialize_process)
    types.divide_registry.register('divide_process', divide_process)
    types.check_registry.register('check_process', check_process)

    types.deserialize_registry.register('deserialize_step', deserialize_step)

    for process_key, process_type in process_types.items():
        types.type_registry.register(process_key, process_type)

    return types


class ProcessTypes(TypeSystem):
    def __init__(self):
        super().__init__()
        register_process_types(self)


    def infer_schema(self, schema, state, top_state=None, path=None):
        """
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (processes/steps) during this process
        """

        schema = schema or {}
        # TODO: deal with this
        if schema == '{}':
            schema = {}

        top_state = top_state or state
        path = path or ()

        if isinstance(state, dict):
            if '_type' in state:
                state_type = state['_type']
                state_schema = self.access(state_type)

                hydrated_state = self.deserialize(state_schema, state)
                top_state = set_path(
                    top_state,
                    path,
                    hydrated_state)

                schema = set_path(
                    schema,
                    path,
                    {'_type': state_type})

                # TODO: fix is_descendant
                # if types.type_registry.is_descendant('process', state_schema) or types.registry.is_descendant('step', state_schema):
                if state_type == 'process' or state_type == 'step':
                    port_schema = hydrated_state['instance'].schema()

                    for port_key in ['inputs', 'outputs']:
                        subschema = port_schema.get(
                            port_key, {})

                        schema = set_path(
                            schema,
                            path + (f'_{port_key}',),
                            subschema)

                        ports = hydrated_state.get(
                            port_key, {})

                        schema = self.infer_wires(
                            subschema,
                            hydrated_state,
                            ports,
                            top_schema=schema,
                            path=path[:-1])

            elif '_type' in schema:
                hydrated_state = self.deserialize(schema, state)
                top_state = set_path(
                    top_state,
                    path,
                    hydrated_state)
            else:
                for key, value in state.items():
                    inner_path = path + (key,)
                    if get_path(schema, inner_path) is None or get_path(state, inner_path) is None or (
                            isinstance(value, dict) and '_type' in value):
                        schema, top_state = self.infer_schema(
                            schema,
                            value,
                            top_state=top_state,
                            path=inner_path)

        elif isinstance(state, str):
            pass

        else:
            type_schema = TYPE_SCHEMAS.get(str(type(state)), schema)

            peer = get_path(schema, path)
            destination = establish_path(
                peer,
                path[:-1],
                top=schema,
                cursor=path[:-1])

            path_key = path[-1]
            if path_key in destination:
                # TODO: validate
                pass
            else:
                destination[path_key] = type_schema

        return schema, top_state

    def infer_edge(self, schema, wires):
        schema = schema or {}
        edge = {}

        if isinstance(wires, str):
            import ipdb;
            ipdb.set_trace()

        for port_key, wire in wires.items():
            if isinstance(wire, dict):
                edge[port_key] = self.infer_edge(
                    schema.get(port_key, {}),
                    wire)
            else:
                subschema = get_path(schema, wire)
                edge[port_key] = subschema

        return edge

    def initialize_edge_state(self, schema, path, edge):
        initial_state = edge['instance'].initial_state()
        ports = get_path(schema, path + ('_ports',))

        return self.project_edge(
            ports,
            edge,
            path[:-1],
            initial_state)

    def dehydrate(self, schema):
        return {}

    def lookup_address(self, address):
        protocol, config = address.split(':')

        if protocol == 'local':
            self.lookup_local(config)


types = ProcessTypes()
