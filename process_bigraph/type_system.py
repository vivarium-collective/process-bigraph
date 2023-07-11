import importlib

from bigraph_schema import TypeSystem


# TODO: implement these
def apply_process(current, update, bindings=None, types=None):
    pass

def divide_process(value, bindings=None, types=None):
    return value

def serialize_process(value, bindings=None, types=None):
    return value

def deserialize_process(serialized, bindings=None, types=None):
    return value


process_types = {
    'protocol': {
        '_super': 'string'
    },

    'process': {
        '_super': 'edge',
        '_apply': 'apply_process',
        '_serialize': 'serialize_process',
        '_deserialize': 'deserialize_process',
        '_divide': 'divide_process',
        '_description': '',
        # TODO: support reference to type parameters from other states
        'address': 'protocol',
        'config': 'tree[any]',
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


    # TODO: maybe this is just deserialization?
    def lookup_local(self, config):
        module_name, class_name = config.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


    def lookup_address(self, address):
        protocol, config = address.split(':')

        if protocol == 'local':
            self.lookup_local(config)


    
types = ProcessTypes()
