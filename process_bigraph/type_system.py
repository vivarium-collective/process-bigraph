"""
=============
Process Types
=============
"""

from bigraph_schema import TypeSystem, get_path, establish_path, set_path
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


class ProcessTypes(TypeSystem):
    def __init__(self):
        super().__init__()
        register_process_types(self)


    # def serialize(self, schema, state):
    #     return ''


    # def deserialize(self, schema, encoded):
    #     return {}


    def infer_wires(self, ports, state, wires, top_schema=None, path=None):
        top_schema = top_schema or {}
        path = path or ()

        for port_key, port_wires in wires.items():
            if isinstance(ports, str):
                import ipdb; ipdb.set_trace()
            port_schema = ports.get(port_key, {})
            # port_wires = wires.get(port_key, ())
            if isinstance(port_wires, dict):
                top_schema = self.infer_wires(
                    ports,
                    state.get(port_key),
                    port_wires,
                    top_schema,
                    path + (port_key,))
            else:
                peer = get_path(
                    top_schema,
                    path[:-1])

                destination = establish_path(
                    peer,
                    port_wires[:-1],
                    top=top_schema,
                    cursor=path[:-1])

                if len(port_wires) == 0:
                    raise Exception(f'no wires at port "{port_key}" in ports {ports} with state {state}')

                destination_key = port_wires[-1]
                if destination_key in destination:
                    # TODO: validate the schema/state
                    pass
                else:
                    destination[destination_key] = port_schema

        return top_schema


    def infer_schema(self, schema, state, top_state=None, path=None):
        '''
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (processes/steps) during this process
        '''

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

                    schema = set_path(
                        schema,
                        path + ('_ports',),
                        port_schema)

                    subwires = hydrated_state['wires']
                    if state_type == 'step':
                        input_subwires = subwires.get('inputs', {})
                        input_port_schema = port_schema.get('inputs', {})
                        schema = self.infer_wires(
                            input_port_schema,
                            hydrated_state,
                            input_subwires,
                            top_schema=schema,
                            path=path[:-1])

                        output_subwires = subwires.get('outputs', {})
                        output_port_schema = port_schema.get('outputs', {})
                        schema = self.infer_wires(
                            output_port_schema,
                            hydrated_state,
                            output_subwires,
                            top_schema=schema,
                            path=path[:-1])
                    else:
                        schema = self.infer_wires(
                            port_schema,
                            hydrated_state,
                            subwires,
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
                    if get_path(schema, inner_path) is None or get_path(state, inner_path) is None or (isinstance(value, dict) and '_type' in value):
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
            import ipdb; ipdb.set_trace()

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

        return edge['instance'].project_state(
            ports,
            edge['wires'],
            path[:-1],
            initial_state)
        

    def hydrate_state(self, schema, state):
        if isinstance(state, str) or '_deserialize' in schema:
            result = self.deserialize(schema, state)
        elif isinstance(state, dict):
            if isinstance(schema, str):
                schema = self.access(schema)
                return self.hydrate_state(schema, state)
            else:
                result = state.copy()
                for key, value in schema.items():
                    if key in schema:
                        subschema = schema[key]
                    else:
                        subschema = schema

                    if key in state:
                        result[key] = self.hydrate_state(
                            subschema,
                            state.get(key))
        else:
            result = state

        return result


    def hydrate(self, schema, state):
        # TODO: support partial hydration (!)
        hydrated = self.hydrate_state(schema, state)
        return self.fill(schema, hydrated)


    def dehydrate(self, schema):
        return {}


    def lookup_address(self, address):
        protocol, config = address.split(':')

        if protocol == 'local':
            self.lookup_local(config)


types = ProcessTypes()
