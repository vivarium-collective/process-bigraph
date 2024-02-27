""" Toy Stochastic Transcription Process
Toy model of Gillespie algorithm-based  transcription,
and a composite with deterministic translation.

Note: This Process is primarily for testing multi-timestepping.
variables and parameters are hard-coded. Do not use this as a
general stochastic transcription.
"""


import numpy as np
import pytest

from process_bigraph import Step, Process, Composite, ProcessTypes


# 'map[float](default:1.0|apply:set)'
# 'float(default:1.0)'


EXPORT = {
    'default 1': {
        '_inherit': 'float',
        '_default': 1.0}}


class GillespieInterval(Step):
    config_schema = {
        'ktsc': {
            '_type': 'float',
            '_default': '5e0'},
        'kdeg': {
            '_type': 'float',
            '_default': '1e-1'}}


    def inputs(self):
        return {
            'DNA': 'map[default 1]',
            'mRNA': {
                'A mRNA': 'default 1',
                'B mRNA': 'default 1'}}

                    # {
                    # '_type': 'map',
                    # '_value': 'float(default:1.0)'},

                    # 'G': {
                    #     '_type': 'float',
                    #     '_default': '1.0'}},


    def outputs(self):
        return {
            'interval': 'interval'}


    def initial_state(self):
        return {
            'mRNA': {
                'A mRNA': 2.0,
                'B mRNA': 3.0}}


    def update(self, input):
        # retrieve the state values
        g = input['DNA']['A gene']
        c = input['mRNA']['A mRNA']

        array_state = np.array([g, c])

        # Calculate propensities
        propensities = [
            self.config['ktsc'] * array_state[0],
            self.config['kdeg'] * array_state[1]]
        prop_sum = sum(propensities)

        # The wait time is distributed exponentially
        interval = np.random.exponential(scale=prop_sum)

        output = {
            'interval': interval}

        print(f'produced interval: {output}')

        return output


class GillespieEvent(Process):
    """stochastic toy transcription"""
    config_schema = {
        'ktsc': {
            '_type': 'float',
            '_default': '5e0'},
        'kdeg': {
            '_type': 'float',
            '_default': '1e-1'}}


    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        self.stoichiometry = np.array([[0, 1], [0, -1]])


    def initial_state(self):
        return {
            'mRNA': {
                'C mRNA': 11.111},
            'DNA': {
                'A gene': 3.0,
                'B gene': 5.0}}


    def inputs(self):
        return {
            'mRNA': 'map[float]',
            'DNA': {
                'A gene': 'float',
                'B gene': 'float'}}

    def outputs(self):
        return {
            'mRNA': 'map[float]'}


    def next_reaction(self, x):
        """get the next reaction and return a new state"""

        propensities = [
            self.config['ktsc'] * x[0],
            self.config['kdeg'] * x[1]]
        prop_sum = sum(propensities)

        # Choose the next reaction
        r_rxn = np.random.uniform()
        i = 0
        for i, _ in enumerate(propensities):
            if r_rxn < propensities[i] / prop_sum:
                # This means propensity i fires
                break
        x += self.stoichiometry[i]
        return x


    def update(self, state, interval):

        # retrieve the state values, put them in array
        g = state['DNA']['A gene']
        c = state['mRNA']['A mRNA']
        array_state = np.array([g, c])

        # calculate the next reaction
        new_state = self.next_reaction(array_state)

        # get delta mRNA
        c1 = new_state[1]
        d_c = c1 - c

        update = {
            'mRNA': {
                'A mRNA': d_c}}

        print(f'received interval: {interval}')

        return update


@pytest.fixture
def core():
    core = ProcessTypes()
    core.import_types(EXPORT)
    return core
    

def test_gillespie_composite(core):
    composite_schema = {
        # This all gets inferred -------------
        # ==================================
        # 'composition': {
        #     'interval': {
        #         '_type': 'step',
        #         '_ports': {
        #             'inputs': {
        #                 'DNA': {
        #                     'G': 'float'},
        #                 'mRNA': {
        #                     'C': 'float'}},
        #             'outputs': {
        #                 'interval': 'float'}}},
        #     'event': {
        #         '_type': 'process',
        #         '_ports': {
        #             'DNA': {
        #                 'G': 'float'},
        #             'mRNA': {
        #                 'C': 'float'}},
        #             'interval': 'float'}}},
        #     'emitter': {
        #         '_type': 'step',
        #         '_ports': {
        #             'inputs': {
        #                 'DNA': {
        #                     'G': 'float'},
        #                 'mRNA': {
        #                     'C': 'float'}}},
        #     'DNA': {
        #         'G': 'float'},
        #     'mRNA': {
        #         'C': 'float'}},
        # 'schema': {
        #     'DNA': {
        #         'G': 'float'},
        #     'mRNA': {
        #         'C': 'float'}},

        'bridge': {
            'inputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']},
            'outputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']}},

        'state': {
            'interval': {
                '_type': 'step',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieInterval',
                'config': {'ktsc': '6e0'},
                'inputs': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'outputs': {
                    'interval': ['event', 'interval']}},

            'event': {
                '_type': 'process',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieEvent',
                'config': {'ktsc': 6e0},
                'inputs': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'outputs': {
                    'mRNA': ['mRNA']},
                'interval': '3.0'},

            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': {
                        'time': 'float',
                        'mRNA': 'map[float]',
                        'interval': 'interval'}},
                'inputs': {
                    'time': ['global_time'],
                    'mRNA': ['mRNA'],
                    'interval': ['event', 'interval']}}}}

                #     'emit': 'any'},
                # 'inputs': ()}}}


            # TODO: provide a way to emit everything:
            # 'emitter': emit_all(
            #     'console-emitter',
            #     exclusions={'DNA': {}}),

            # TODO: make us able to wire to the top with '**'
            # 'ram': {
            #     '_type': 'step',
            #     'address': 'local:ram-emitter',
            #     'config': {
            #         'ports': {
            #             'inputs': 'tree[any]'}},
            #     'wires': {
            #         'inputs': '**'}}}}

            # 'DNA': {
            #     'G': 13.0},

            # 'mRNA': {
            #     'C': '21.0'}}}

    gillespie = Composite(
        composite_schema,
        core=core)

    updates = gillespie.update({
        'DNA': {
            'A gene': 11.0,
            'B gene': 5.0},
        'mRNA': {
            'A mRNA': 33.3,
            'B mRNA': 2.1}},
        1000.0)

    # TODO: make this work
    results = gillespie.gather_results()

    assert 'mRNA' in updates[0]


def test_union_tree(core):
    tree_union = core.access('list[string]~tree[list[string]]')
    assert core.check(
        tree_union,
        {'a': ['what', 'is', 'happening']})


def test_stochastic_deterministic_composite(core):
    # TODO make the demo for a hybrid stochastic/deterministic simulator
    pass


if __name__ == '__main__':
    core = ProcessTypes()
    core.import_types(EXPORT)

    test_gillespie_composite(core)
    test_union_tree(core)
