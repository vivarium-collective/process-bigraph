"""
Tests for Process Bigraph
"""

import random

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


def test_dependencies():
    operation = {
        'a': 11.111,
        'b': 22.2,
        'c': 555.555,

        '1': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'wires': {
                'inputs': {
                    'a': ['a'],
                    'b': ['b']},
                'outputs': {
                    'c': ['e']}}},
        '2.1': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'wires': {
                'inputs': {
                    'a': ['c'],
                    'b': ['e']},
                'outputs': {
                    'c': ['f']}}},
        '2.2': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'wires': {
                'inputs': {
                    'a': ['d'],
                    'b': ['e']},
                'outputs': {
                    'c': ['g']}}},
        '3': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '*'},
            'wires': {
                'inputs': {
                    'a': ['f'],
                    'b': ['g']},
                'outputs': {
                    'c': ['h']}}},
        '4': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'wires': {
                'inputs': {
                    'a': ['e'],
                    'b': ['h']},
                'outputs': {
                    'c': ['i']}}}}

    composite = Composite({'state': operation})

    assert composite.state['h'] == -17396.469884


def test_dependency_cycle():
    # test a step network with cycles in a few ways
    pass


class SimpleCompartment(Process):
    config_schema = {
        'id': 'string'}


    def schema(self):
        return {
            'outer': 'tree[process]',
            'inner': 'tree[process]'}


    def update(self, state, interval):
        choice = random.random()
        update = {}

        outer = state['outer']
        inner = state['inner']

        # TODO: implement divide_state(_)
        divisions = self.types.divide_state(
            self.schema(),
            inner)

        if choice < 0.2:
            # update = {
            #     'outer': {
            #         '_divide': {
            #             'mother': self.config['id'],
            #             'daughters': [
            #                 {'id': self.config['id'] + '0'},
            #                 {'id': self.config['id'] + '1'}]}}}

            # daughter_ids = [self.config['id'] + str(i)
            #     for i in range(2)]

            # update = {
            #     'outer': {
            #         '_react': {
            #             'redex': {
            #                 'inner': {
            #                     self.config['id']: {}}},
            #             'reactum': {
            #                 'inner': {
            #                     daughter_config['id']: {
            #                         '_type': 'process',
            #                         'address': 'local:!process_bigraph.tests.SimpleCompartment',
            #                         'config': daughter_config,
            #                         'inner': daughter_inner,
            #                         'wires': {
            #                             'outer': ['..']}}
            #                     for daughter_config, daughter_inner in zip(daughter_configs, divisions)}}}}}

            update = {
                'outer': {
                    'inner': {
                        '_react': {
                            'reaction': 'divide',
                            'config': {
                                'id': self.config['id'],
                                'daughters': [{
                                        'id': daughter_id,
                                        'state': daughter_state}
                                    for daughter_id, daughter_state in zip(
                                        daughter_ids,
                                        divisions)]}}}}}

        return update


# TODO: create reaction registry, register this under "divide"


def engulf_reaction(config):
    return {
        'redex': {},
        'reactum': {}}


def burst_reaction(config):
    return {
        'redex': {},
        'reactum': {}}


def test_reaction():
    composite = {
        'state': {
            'environment': {
                'concentrations': {},
                'inner': {
                    'agent1': {
                        '_type': 'process',
                        'address': 'local:!process_bigraph.tests.SimpleCompartment',
                        'config': {'id': '0'},
                        'concentrations': {},
                        'inner': {
                            'agent2': {
                                '_type': 'process',
                                'address': 'local:!process_bigraph.tests.SimpleCompartment',
                                'config': {'id': '0'},
                                'inner': {},
                                'wires': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']}}},
                        'wires': {
                            'outer': ['..', '..'],
                            'inner': ['inner']}}}}}}


if __name__ == '__main__':
    test_default_config()
    test_merge_collections()
    test_process()
    test_composite()
    test_infer()
    test_step_initialization()
    test_dependencies()
    # test_reaction()
