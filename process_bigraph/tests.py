"""
Tests for Process Bigraph
"""

from process_bigraph.composite import Process, Step, Composite
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

    initial_state = {'exchange': 3.33}

    updates = composite.update(initial_state, 10.0)

    final_exchange = sum([
        update['exchange']
        for update in [initial_state] + updates])

    assert composite.state['value'] > 45
    assert 'exchange' in updates[0]
    assert updates[0]['exchange'] == 0.999


def test_infer():
    composite = Composite({
        'state': {
            'increase': {
                '_type': 'process',
                'address': 'local:!process_bigraph.tests.IncreaseProcess',
                'config': {'rate': '0.3'},
                'wires': {'level': ['value']}},
            'value': '11.11'}})

    assert composite.composition['value']['_type'] == 'float'
    assert composite.state['value'] == 11.11


class OperatorStep(Step):
    config_schema = {
        'operator': 'string'}


    def schema(self):
        return {
            'inputs': {
                'a': 'float',
                'b': 'float'},
            'outputs': {
                'c': 'float'}}


    def update(self, inputs):
        a = inputs['a']
        b = inputs['b']

        if self.config['operator'] == '+':
            c = a + b
        elif self.config['operator'] == '*':
            c = a * b
        elif self.config['operator'] == '-':
            c = a - b

        return {'c': c}


def test_step_initialization():
    composite = Composite({
        'state': {
            'A': 13,
            'B': 21,
            'step1': {
                '_type': 'step',
                'address': 'local:!process_bigraph.tests.OperatorStep',
                'config': {
                    'operator': '+'},
                # TODO: avoid inputs/outputs key in wires?
                'wires': {
                    'inputs': {
                        'a': ['A'],
                        'b': ['B']},
                    'outputs': {
                        'c': ['C']}}},
            'step2': {
                '_type': 'step',
                'address': 'local:!process_bigraph.tests.OperatorStep',
                'config': {
                    'operator': '*'},
                'wires': {
                    'inputs': {
                        'a': ['B'],
                        'b': ['C']},
                    'outputs': {
                        'c': ['D']}}}}})


    assert composite.state['D'] == (13 + 21) * 21


# class AddStep(Step):
#     config_schema = {
#         'offset': 'float'}


#     def schema(self):
#         return {
#             'inputs': {
#                 'a': 'float',
#                 'b': 'float'},
#             'outputs': {
#                 'c': 'float'}}


#     def update(self, inputs):
#         output = self.config['offset'] + inputs['a'] + inputs['b']
#         outputs = {'c': output}

#         return outputs


# class SubtractStep(Step):
#     config_schema = {
#         'offset': 'float'}


#     def schema(self):
#         return {
#             'inputs': {
#                 'a': 'float',
#                 'b': 'float'},
#             'outputs': {
#                 'c': 'float'}}


#     def update(self, inputs):
#         output = self.config['offset'] + inputs['a'] - inputs['b']
#         outputs = {'c': output}

#         return outputs


# class MultiplyStep(Step):
#     config_schema = {
#         'scale': {
#             '_type': 'float',
#             '_default': 1.0}}


#     def schema(self):
#         return {
#             'inputs': {
#                 'a': {
#                     '_type': 'float',
#                     '_default': 1.0},
#                 'b': {
#                     '_type': 'float',
#                     '_default': 1.0}},
#             'outputs': {
#                 'c': 'float'}}


#     def update(self, inputs):
#         output = self.config['scale'] * (inputs['a'] * inputs['b'])
#         outputs = {'c': output}

#         return outputs


def test_dependencies():
    operation = {
        'x': 11.111,
        'y': 22.2,
        'z': 555.555,
        'add': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'wires': {
                'inputs': {
                    'a': ['x'],
                    'b': ['y']},
                'outputs': {
                    'c': ['w']}}},
        'subtract': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'wires': {
                'inputs': {
                    'a': ['z'],
                    'b': ['w']},
                'outputs': {
                    'c': ['j']}}},
        'multiply': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '*'},
            'wires': {
                'inputs': {
                    'a': ['j'],
                    'b': ['w']},
                'outputs': {
                    'c': ['k']}}}}

    import ipdb; ipdb.set_trace()

    composite = Composite({'state': operation})


if __name__ == '__main__':
    test_default_config()
    test_merge_collections()
    test_process()
    test_composite()
    test_infer()
    test_step_initialization()
    test_dependencies()
