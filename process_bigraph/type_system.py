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


    def infer_wires(self, schema, ports, state, wires, top_schema=None, path=None):
        top_schema = top_schema or schema
        path = path or ()

        for port_key, port_schema in ports.items():
            port_wires = wires.get(port_key, ())
            if isinstance(port_wires, dict):
                top_schema = self.infer_wires(
                    schema.get(port_key, {}),
                    port_schema,
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

                destination_key = port_wires[-1]
                if destination_key in destination:
                    # TODO: validate the schema/state
                    pass
                else:
                    destination[destination_key] = port_schema

        return top_schema


    def infer_schema(self, schema, state, top_schema=None, top_state=None, path=None):
        '''
        Given a schema fragment and an existing state with _type keys,
        return the full schema required to describe that state,
        and whatever state was hydrated (processes/steps) during this process
        '''

        schema = types.access(schema or {})
        top_schema = top_schema or schema
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

                top_schema = set_path(
                    top_schema,
                    path,
                    {'_type': state_type})

                # TODO: fix is_descendant
                # if types.type_registry.is_descendant('process', state_schema) or types.registry.is_descendant('step', state_schema):
                if state_type == 'process' or state_type == 'step':
                    port_schema = hydrated_state['instance'].schema()
                    top_schema = set_path(
                        top_schema,
                        path + ('_ports',),
                        port_schema)

                    top_schema = self.infer_wires(
                        schema,
                        port_schema,
                        hydrated_state,
                        hydrated_state['wires'],
                        top_schema=top_schema,
                        path=path[:-1])
            else:
                for key, value in state.items():
                    inner_path = path + (key,)
                    top_schema, top_state = self.infer_schema(
                        schema.get(key),
                        value,
                        top_schema=top_schema,
                        top_state=top_state,
                        path=inner_path)

        elif isinstance(state, str):
            pass

        else:
            type_schema = TYPE_SCHEMAS.get(state, schema)

            peer = get_path(top_schema, path)
            destination = establish_path(
                peer,
                path[:-1],
                top=top_schema,
                cursor=path[:-1])

            path_key = path[-1]
            if path_key in destination:
                # TODO: validate
                pass
            else:
                destination[path_key] = type_schema

        return top_schema, top_state
        

    def hydrate_state(self, schema, state):
        if isinstance(state, str) or '_deserialize' in schema:
            result = self.deserialize(schema, state)
        elif isinstance(state, dict):
            result = {
                key: self.hydrate_state(schema[key], state[key])
                for key, value in state.items()}
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
