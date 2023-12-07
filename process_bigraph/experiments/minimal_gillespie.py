""" Toy Stochastic Transcription Process
Toy model of Gillespie algorithm-based  transcription,
and a composite with deterministic translation.

Note: This Process is primarily for testing multi-timestepping.
variables and parameters are hard-coded. Do not use this as a
general stochastic transcription.
"""
import os
import numpy as np

from process_bigraph.composite import types, Step, Process, Composite


class GillespieInterval(Step):
    config_schema = {
        'ktsc': {
            '_type': 'float',
            '_default': '5e0'},
        'kdeg': {
            '_type': 'float',
            '_default': '1e-1'}}


    def schema(self):
        # Step schemas always have 'inputs' and 'outputs' as top level keys
        return {
            'inputs': {
                'DNA': {
                    'G': {
                        '_type': 'float',
                        '_default': '1.0'}},
                'mRNA': {
                    'C': {
                        '_type': 'float',
                        '_default': '1.0'}}},
            'outputs': {
                'interval': 'float'}}


    def initial_state(self):
        return {
            'mRNA': {
                'C': 2.0}}


    def update(self, input):
        # retrieve the state values
        g = input['DNA']['G']
        c = input['mRNA']['C']

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


    def __init__(self, config=None):
        super().__init__(config)
        self.stoichiometry = np.array([[0, 1], [0, -1]])


    def initial_state(self):
        return {
            'mRNA': {'C': 11.111},
            'DNA': {
                'G': 3.0}}


    def schema(self):
        return {
            'DNA': 'tree[float]',
            'mRNA': 'tree[float]'}

            # 'DNA': {
            #     'G': {
            #         '_type': 'float',
            #         '_default': '1.0'}},
            # 'mRNA': {
            #     'C': {
            #         '_type': 'float',
            #         '_default': '1.0'}}}


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
        g = state['DNA']['G']
        c = state['mRNA']['C']
        array_state = np.array([g, c])

        # calculate the next reaction
        new_state = self.next_reaction(array_state)

        # get delta mRNA
        c1 = new_state[1]
        d_c = c1 - c

        update = {
            'mRNA': {
                'C': d_c}}
        return update


def test_gillespie_composite():
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
            'DNA': ['DNA'],
            'mRNA': ['mRNA']},

        'state': {
            'mRNA': {'C': 23.0},
            'interval': {
                '_type': 'step',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieInterval',
                'config': {'ktsc': '6e0'},
                'wires': {
                    'inputs': {
                        'DNA': ['DNA'],
                        'mRNA': ['mRNA']},
                    'outputs': {
                        'interval': ['event', 'interval']}}},

            'event': {
                '_type': 'process',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieEvent',
                'config': {'ktsc': 6e0},
                'wires': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'interval': '3.0'},

            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'ports': {
                        'inputs': {
                            'mRNA': 'tree[float]',
                            'interval': 'float'}}},
                'wires': {
                    'inputs': {
                        'mRNA': ['mRNA'],
                        'interval': ['event', 'interval']}}}}}

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

    gillespie = Composite(composite_schema)

    updates = gillespie.update({
        'DNA': {
            'G': 11.0},
        'mRNA': {
            'C': 5.0}},
        1000.0)

    # TODO: make this work
    results = gillespie.gather_results()

    assert 'mRNA' in updates[0]


def test_stochastic_deterministic_composite():
    # TODO make the demo for a hybrid stochastic/deterministic simulator
    pass


if __name__ == '__main__':
    test_gillespie_composite()
