"""
Tests for Process Bigraph
"""

from process_bigraph.composite import Process, Composite
from process_bigraph.composite import merge_collections
from process_bigraph.type_system import types


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'}}

    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'level': 'float'}

    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']}


def test_serialized_composite():
    # This should specify the same thing as above
    composite_schema = {
        '_type': 'process[exchange:float]',
        'address': 'local:!process_bigraph.composite.Composite',
        'config': {
            'state': {
                'increase': {
                    '_type': 'process[level:float]',
                    'address': 'local:!process_bigraph.tests.IncreaseProcess',
                    'config': {'rate': '0.3'},
                    'wires': {'level': ['value']}
                },
                'value': '11.11',
            },
            'schema': {
                'increase': 'process[level:float]',
                # 'increase': 'process[{"level":"float","down":{"a":"int"}}]',
                'value': 'float',
            },
            'bridge': {
                'exchange': 'value'
            },
        }
    }

    composite_instance = types.deserialize(composite_schema, {})
    composite_instance.update()


def test_default_config():
    process = IncreaseProcess()
    assert process.config['rate'] == 0.1


def test_merge_collections():
    a = {('what',): [1, 2, 3]}
    b = {('okay', 'yes'): [3, 3], ('what',): [4, 5, 11]}

    c = merge_collections(a, b)

    assert c[('what',)] == [1, 2, 3, 4, 5, 11]


def test_process():
    process = IncreaseProcess({'rate': 0.2})
    schema = process.schema()
    state = types.fill(schema)
    update = process.update({'level': 5.5}, 1.0)
    new_state = types.apply(schema, state, update)

    assert new_state['level'] == 1.1


def test_composite():
    # TODO: add support for the various vivarium emitters

    # increase = IncreaseProcess({'rate': 0.3})
    # TODO: This is the config of the composite,
    #   we also need a way to serialize the entire composite

    composite = Composite({
        'composition': {
            'increase': 'process[level:float]',
            'value': 'float'},
        'schema': {
            'exchange': 'float'},
        'bridge': {
            'exchange': ['value']},
        'state': {
            'increase': {
                'address': 'local:!process_bigraph.tests.IncreaseProcess',
                'config': {'rate': '0.3'},
                'interval': '1.0',
                'wires': {'level': ['value']}},
            'value': '11.11'}})

    composite.update({'exchange': 3.33}, 10.0)
    assert composite.state['value'] > 199


if __name__ == '__main__':
    test_default_config()
    test_merge_collections()
    test_process()
    test_composite()
