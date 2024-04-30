"""
Tests for Process Bigraph
"""

import random
import pytest

from process_bigraph.composite import Process, Step, Composite, merge_collections, ProcessTypes
# from process_bigraph.type_system import ProcessTypes


@pytest.fixture
def core():
    return ProcessTypes()


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'}}


    def inputs(self):
        return {
            'level': 'float'}


    def outputs(self):
        return {
            'level': 'float'}


    def initial_state(self):
        return {
            'inputs': {
                'level': 11.0}}


    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']}


def test_default_config(core):
    process = IncreaseProcess(core=core)

    assert process.config['rate'] == 0.1


def test_merge_collections(core):
    a = {('what',): [1, 2, 3]}
    b = {('okay', 'yes'): [3, 3], ('what',): [4, 5, 11]}

    c = merge_collections(a, b)

    assert c[('what',)] == [1, 2, 3, 4, 5, 11]


@pytest.fixture
def core():
    return ProcessTypes()


def test_process(core):
    process = IncreaseProcess({'rate': 0.2}, core=core)
    interface = process.interface()
    state = core.fill(interface['inputs'])
    state = core.fill(interface['outputs'])
    update = process.update({'level': 5.5}, 1.0)

    new_state = core.apply(
        interface['outputs'],
        state,
        update)

    assert new_state['level'] == 1.1


def test_composite(core):
    # TODO: add support for the various vivarium emitter

    # increase = IncreaseProcess({'rate': 0.3})
    # TODO: This is the config of the composite,
    #   we also need a way to serialize the entire composite

    composite = Composite({
        'composition': {
            'increase': 'process[level:float,level:float]',
            'value': 'float'},
        'interface': {
            'inputs': {
                'exchange': 'float'},
            'outputs': {
                'exchange': 'float'}},
        'bridge': {
            'inputs': {
                'exchange': ['value']},
            'outputs': {
                'exchange': ['value']}},
        'state': {
            'increase': {
                'address': 'local:!process_bigraph.tests.IncreaseProcess',
                'config': {'rate': 0.3},
                'interval': 1.0,
                'inputs': {'level': ['value']},
                'outputs': {'level': ['value']}},
            'value': '11.11'}}, core=core)

    initial_state = {'exchange': 3.33}

    updates = composite.update(initial_state, 10.0)

    final_exchange = sum([
        update['exchange']
        for update in [initial_state] + updates])

    assert composite.state['value'] > 45
    assert 'exchange' in updates[0]
    assert updates[0]['exchange'] == 0.999


def test_infer(core):
    composite = Composite({
        'state': {
            'increase': {
                '_type': 'process',
                'address': 'local:!process_bigraph.tests.IncreaseProcess',
                'config': {'rate': '0.3'},
                'inputs': {'level': ['value']},
                'outputs': {'level': ['value']}},
            'value': '11.11'}}, core=core)

    assert composite.composition['value']['_type'] == 'float'
    assert composite.state['value'] == 11.11


def test_process_type(core):
    assert core.access('process')['_type'] == 'process'


class OperatorStep(Step):
    config_schema = {
        'operator': 'string'}


    def inputs(self):
        return {
            'a': 'float',
            'b': 'float'}


    def outputs(self):
        return {
            'c': 'float'}


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


def test_step_initialization(core):
    composite = Composite({
        'state': {
            'A': 13,
            'B': 21,
            'step1': {
                '_type': 'step',
                'address': 'local:!process_bigraph.tests.OperatorStep',
                'config': {
                    'operator': '+'},
                'inputs': {
                    'a': ['A'],
                    'b': ['B']},
                'outputs': {
                    'c': ['C']}},
            'step2': {
                '_type': 'step',
                'address': 'local:!process_bigraph.tests.OperatorStep',
                'config': {
                    'operator': '*'},
                'inputs': {
                    'a': ['B'],
                    'b': ['C']},
                'outputs': {
                    'c': ['D']}}}}, core=core)

    composite.run(0.0)
    assert composite.state['D'] == (13 + 21) * 21


def test_dependencies(core):
    operation = {
        'a': 11.111,
        'b': 22.2,
        'c': 555.555,

        '1': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'inputs': {
                'a': ['a'],
                'b': ['b']},
            'outputs': {
                'c': ['e']}},
        '2.1': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'inputs': {
                'a': ['c'],
                'b': ['e']},
            'outputs': {
                'c': ['f']}},
        '2.2': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'inputs': {
                'a': ['d'],
                'b': ['e']},
            'outputs': {
                'c': ['g']}},
        '3': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '*'},
            'inputs': {
                'a': ['f'],
                'b': ['g']},
            'outputs': {
                'c': ['h']}},
        '4': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'inputs': {
                'a': ['e'],
                'b': ['h']},
            'outputs': {
                'c': ['i']}}}

    composite = Composite(
        {'state': operation},
        core=core)

    composite.run(0.0)

    assert composite.state['h'] == -17396.469884


def test_dependency_cycle():
    # test a step network with cycles in a few ways
    pass


class SimpleCompartment(Process):
    config_schema = {
        'id': 'string'}


    def interface(self):
        return {
            'outer': 'tree[process]',
            'inner': 'tree[process]'}


    def update(self, state, interval):
        choice = random.random()
        update = {}

        outer = state['outer']
        inner = state['inner']

        # TODO: implement divide_state(_)
        divisions = self.core.divide_state(
            self.interface(),
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
                                'inputs': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']},
                                'outputs': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']}}},
                        'inputs': {
                            'outer': ['..', '..'],
                            'inner': ['inner']},
                        'outputs': {
                            'outer': ['..', '..'],
                            'inner': ['inner']}}}}}}


if __name__ == '__main__':
    core = ProcessTypes()

    test_default_config(core)
    test_merge_collections(core)
    test_process(core)
    test_composite(core)
    test_infer(core)
    test_step_initialization(core)
    test_dependencies(core)
    # test_reaction()
