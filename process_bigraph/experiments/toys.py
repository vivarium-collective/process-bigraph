""" Toy Stochastic Transcription Process
Toy model of Gillespie algorithm-based  transcription,
and a composite with deterministic translation.

Note: This Process is primarily for testing multi-timestepping.
variables and parameters are hard-coded. Do not use this as a
general stochastic transcription.
"""
import os
import numpy as np

from process_bigraph.composite import types, Step, Process, Composite, Generator


class GillespieInterval(Step):
    config_schema = {
        'ktsc': {
            '_type': 'float',
            '_default': '5e0'},
        'kdeg': {
            '_type': 'float',
            '_default': '1e-1'}}
    

    def __init__(self, config=None):
        super().__init__(config)


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

        # self.time_left = None
        # self.event = None


    def schema(self):
        return {
            'DNA': {
                'G': {
                    '_type': 'float',
                    '_default': '1.0'}},
            'mRNA': {
                'C': {
                    '_type': 'float',
                    '_default': '1.0'}}}

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
        # if self.time_left is not None:
        #     if interval >= self.time_left:
        #         event = self.event
        #         self.event = None
        #         self.time_left = None
        #         return event

        #     self.time_left -= interval
        #     return {}

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

        # if self.calculated_interval > interval:
        #     # didn't get all of our time, store the event for later
        #     self.time_left = self.calculated_interval - interval
        #     self.event = update
        #     return {}

        return update





# class TRL(Process):
#     """deterministic toy translation"""

#     defaults = {
#         'ktrl': 1e-2,
#         'kdeg': 1e-4,
#         }

#     def ports_schema(self):
#         return {
#             'mRNA': {
#                 'C': {
#                     '_default': 1.0,
#                     '_emit': True}},
#             'Protein': {
#                 'X': {
#                     '_default': 1.0,
#                     '_emit': True}}}

#     def next_update(self, timestep, states):
#         c = states['mRNA']['C']
#         x = states['Protein']['X']
#         d_x = (
#             self.parameters['ktrl'] * c -
#             self.parameters['kdeg'] * x) * timestep
#         return {
#             'Protein': {
#                 'X': d_x}}


# class TrlConcentration(TRL):
#     """rescale mRNA"""

#     def next_update(self, timestep, states):
#         states['mRNA']['C'] = states['mRNA']['C'] * 1e5
#         return super().next_update(timestep, states)


# class StochasticTscTrl(Composer):
#     """
#     composite toy model with stochastic transcription,
#     deterministic translation.
#     """
#     defaults = {
#         'stochastic_TSC': {'time_step': 10},
#         'TRL': {'time_step': 10},
#     }

#     def generate_processes(self, config):
#         counts_to_molar = process_registry.access(
#             'counts_to_molar')
#         return {
#             'stochastic_TSC': StochasticTSC(config['stochastic_TSC']),
#             'TRL': TrlConcentration(config['TRL']),
#             'concs': counts_to_molar({'keys': ['C']})
#         }

#     def generate_topology(self, config):
#         return {
#             'stochastic_TSC': {
#                 'DNA': ('DNA',),
#                 'mRNA': ('mRNA_counts',)
#             },
#             'TRL': {
#                 'mRNA': ('mRNA',),
#                 'Protein': ('Protein',)
#             },
#             'concs': {
#                 'counts': ('mRNA_counts',),
#                 'concentrations': ('mRNA',)}
#         }



# def test_gillespie_process(total_time=1000):
#     gillespie_process = StochasticTSC({'name': 'process'})

#     # make the experiment
#     exp_settings = {
#         'display_info': False,
#         'experiment_id': 'TscTrl'}
#     composite = gillespie_process.generate()
#     gillespie_experiment = Engine(
#         composite=composite,
#         **exp_settings)

#     # run the experiment in large increments
#     increment = 10
#     for i in range(total_time):
#         if i == total_time - 1:
#             gillespie_experiment.run_for(increment, force_complete=True)
#             # Now the process is at the global time.
#             break
#         gillespie_experiment.run_for(increment)
#         # check that process remains behind global time
#         front = gillespie_experiment.front
#         assert front[('process',)]['time'] < gillespie_experiment.global_time

#     front = gillespie_experiment.front
#     assert front[('process',)]['time'] == gillespie_experiment.global_time

#     gillespie_data = gillespie_experiment.emitter.get_timeseries()
#     return gillespie_data


# def test_gillespie_composite(total_time=10000):
#     stochastic_tsc_trl = StochasticTscTrl().generate()

#     # make the experiment
#     exp_settings = {
#         'experiment_id': 'stochastic_tsc_trl'}
#     stoch_experiment = Engine(
#         composite=stochastic_tsc_trl,
#         **exp_settings)

#     # simulate and retrieve the data from emitter
#     stoch_experiment.update(total_time)
#     data = stoch_experiment.emitter.get_timeseries()

#     return data


# def main():
#     """run the tests and plot"""
#     out_dir = os.path.join(PROCESS_OUT_DIR, 'toy_gillespie')
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     process_output = test_gillespie_process()
#     composite_output = test_gillespie_composite()

#     # plot the simulation output
#     plot_settings = {}
#     plot_simulation_output(
#         process_output, plot_settings, out_dir, filename='process')
#     plot_simulation_output(
#         composite_output, plot_settings, out_dir, filename='composite')


def test_gillespie_composite():
    GillespieComposite = Generator({
        'composition': {
            'interval': {
                '_type': 'step',
                '_ports': {
                    'inputs': {
                        'DNA': {
                            'G': 'float'},
                        'mRNA': {
                            'C': 'float'}},
                    'outputs': {
                        'interval': 'float'}}},
            'event': 'process[DNA.G:float|mRNA.C:float]',
            'DNA': {
                'G': 'float'},
            'mRNA': {
                'C': 'float'}},
        'schema': {
            'DNA': {
                'G': 'float'},
            'mRNA': {
                'C': 'float'}},
        'bridge': {
            'DNA': ['DNA'],
            'mRNA': ['mRNA']},
        'state': {
            'interval': {
                'address': 'local:process_bigraph.experiments.toys.GillespieInterval',
                'config': {'ktsc': '6e0'},
                'wires': {
                    'inputs': {
                        'DNA': ['DNA'],
                        'mRNA': ['mRNA']},
                    'outputs': {
                        'interval': ['event', 'interval']}}},
            'event': {
                'address': 'local:process_bigraph.experiments.toys.GillespieEvent',
                'config': {'ktsc': '6e0'},
                'wires': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'interval': '3.0'},
            'DNA': {
                'G': '13.0'},
            'mRNA': {
                'C': '21.0'}}})

    gillespie = GillespieComposite({})

    gillespie.update({'DNA': {'G': 11.0}, 'mRNA': {'C': 5.0}}, 10000.0)

    import ipdb; ipdb.set_trace()


def test_sed_composite():
    workflow = Composite({
        'composition': {
            'data': 'tree[any]',
            'model': 'sbml',
            'parameters': 'tree[any]',
            'simulation_results': 'tree[any]',
            'analysis_results': 'tree[any]',
            'parameter_estimation': {
                '_type': 'step',
                '_ports': {
                    'inputs': {
                        'data': 'tree[any]',
                        'model': 'sbml'},
                    'outputs': {
                        'parameters': 'tree[any]'}}},
            'simulator': {
                '_type': 'step',
                '_ports': {
                    'inputs': {
                        'model': 'sbml',
                        'parameters': 'tree[any]'},
                    'outputs': {
                        'simulation_results': 'tree[any]'}}},
            'analysis': {
                '_type': 'step',
                '_ports': {
                    'inputs': {
                        'simulation_results': 'tree[any]'},
                    'outputs': {
                        'analysis_results': 'tree[any]'}}}},
        'schema': {
            'results': 'tree[any]'},
        'bridge': {
            'results': ['analysis_results']},
        'state': {
            'data': {},
            'model': 'something.sbml',
            'parameter_estimation': {
                'address': 'local:process_bigraph.experiments.toys.EstimateParameters',
                'config': {},
                'wires': {
                    'inputs': {
                        'data': ['data'],
                        'model': ['model']},
                    'outputs': {
                        'parameters': ['parameters']}}},
            'simulator': {
                'address': 'local:process_bigraph.experiments.toys.UniformTimecourse',
                'config': {},
                'wires': {
                    'inputs': {
                        'model': ['model'],
                        'parameters': ['parameters']},
                    'outputs': {
                        'simulation_results': ['simulation_results']}}},
            'analysis': {
                'address': 'local:process_bigraph.experiments.toys.AnalyzeResults',
                'config': {},
                'wires': {
                    'inputs': {
                        'simulation_results': ['simulation_results'],
                    'outputs': {
                        'analysis_results': ['analysis_results']}}}}}})
    



if __name__ == '__main__':
    test_gillespie_composite()

