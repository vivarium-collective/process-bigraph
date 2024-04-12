import copy
import numpy as np

from bigraph_schema import get_path, set_path
from process_bigraph import Step, Process, Composite, ProcessTypes, interval_time_precision, deep_merge


core = ProcessTypes()


core.register('scannable_process', {
    '_inherit': ['process'],
    '_inputs': {
        'species': {
            '_type': 'array',
            '_data': 'float'}},
    '_outputs': {
        'species': {
            '_type': 'array',
            '_data': 'float'}}})


core.register('ode_config', {
    'stoichiometry': {
        '_type': 'array',
        '_data': 'integer'},
    'rates': 'map[float]',
    'species': 'map[float]'})


class ToySystem(Process):
    config_schema = {
        'rates': {
            '_type': 'map',
            '_value': {
                'kdeg': 'float',
                'ksynth': 'float'}}}


    def inputs(self):
        return {
            'species': 'map[float]'}


    def outputs(self):
        return {
            'species': 'map[float]'}


    def update(self, inputs, interval):
        species = {
            key: input * (self.config['rates'][key]['ksynth'] - self.config['rates'][key]['kdeg'])
            for key, input in inputs['species'].items()}

        return {
            'species': species}


class ODE(Process):
    config_schema = 'ode_config'


    def __init__(self, config, core):
        super().__init__(config, core)

        self.reactions_count = len(self.config['rates'])

        self.species_count = len(
            self.config['species_name'])

        self.config_schema['rates']['_shape'] = (
            self.species_count,
            self.species_count)


    def inputs(self):
        return {
            'species': f'array[({self.species_count}), float]'}


    def outputs(self):
        return {
            'species': f'array[({self.species_count}), float]'}


    def update(self, state, interval):
        total = np.dot(
            self.config['rates'],
            state['species'])

        delta = total - state['species']
        return delta


class RunProcess(Step):
    config_schema = {
        'process_address': 'string',
        'process_config': 'tree[any]',
        'timestep': 'float',
        'runtime': 'float'}


    def __init__(self, config, core):
        super().__init__(config, core)

        self.process = core.deserialize('process', {
            '_type': 'process',
            'address': self.config['process_address'],
            'config': self.config['process_config'],
            'inputs': {},
            'outputs': {}})['instance']

        global_time_precision = interval_time_precision(
            self.config['timestep'])

        self.composite = Composite({
            'global_time_precision': global_time_precision,
            # TODO: support emitter at the composite level
            #   they are a list of emit dicts that describe
            #   which emitter to use and what from the composite
            #   state will be emitted. The schema can be inferred
            #   from the targets. ALSO: support process ports
            #   to be targets
            # 'emit': [{
            #     'address': 'local:mongo-emitter'
            #     'targets': dict({'time': ['global_time']}, **{
            #         key: [key]
            #         for key in self.process.outputs()})}],
            'state': {
                'process': {
                    '_type': 'process',
                    'address': self.config['process_address'],
                    'config': self.config['process_config'],
                    'instance': self.process,
                    'interval': self.config['timestep'],
                    'inputs': {
                        key: [key]
                        for key in self.process.inputs()},
                    'outputs': {
                        key: [key]
                        for key in self.process.outputs()}},
                'emitter': {
                    '_type': 'step',
                    'address': 'local:ram-emitter',
                    'config': {
                        'emit': dict(
                            {'time': 'float'},
                            **self.process.outputs())},
                    'inputs': dict({'time': ['global_time']}, **{
                        key: [key]
                        for key in self.process.outputs()}),
                    'outputs': {}}}})


    def inputs(self):
        return self.process.inputs()


    def outputs(self):
        outputs = self.process.outputs()
        outputs['time'] = 'float'

        return {
            'results': {
                output_key: {
                    '_type': 'list',
                    '_apply': 'set',
                    '_element': output_schema}
                for output_key, output_schema in outputs.items()}}


    def update(self, inputs):
        self.composite.set_state(inputs)

        self.composite.run(
            self.config['runtime'])

        histories = self.composite.gather_results()

        results = {
            key: timeseries_from_history(history)
            for key, history in histories.items()}

        all_results = {}
        for timeseries in results.values():
            all_results = deep_merge(all_results, timeseries)

        return {'results': all_results}


def timeseries_from_history(history):
    results = {}
    for moment in history:
        for key, value in moment.items():
            if key not in results:
                results[key] = []
            results[key].append(value)

    return results


def generate_key(parameters):
    if isinstance(parameters, dict):
        pairs = []
        for key, value in parameters.items():
            pairs.append((key, generate_key(value)))
        tokens = [f'{key}:{value}' for key, value in pairs]
        join = ','.join(tokens)
        return '{' + join + '}'

    elif isinstance(parameters, str):
        return parameters

    else:
        return str(parameters)


class ParameterScan(Step):
    config_schema = {
        'parameter_ranges': 'list[tuple[path,list[float]]]',
        'process_address': 'string',
        'process_config': 'tree[any]',
        'initial_state': 'tree[any]',
        'observables': 'list[path]',
        'timestep': 'float',
        'runtime': 'float'}


    def __init__(self, config, core):
        super().__init__(config, core)

        self.steps_count = int(
            self.config['runtime'] / self.config['timestep']) + 1
        self.observables_count = len(
            self.config['observables'])

        # TODO: test two parameters scanning simultaneously
        self.total_combinations = 1

        results_shape = []
        for parameter_path, parameter_range in self.config['parameter_ranges']:
            ranges_count = len(parameter_range)
            self.total_combinations *= ranges_count
            results_shape.append(ranges_count)

        results_shape.extend([
            self.observables_count,
            self.steps_count])

        self.results_shape = tuple(results_shape)

        self.process_parameters = [
            self.config['process_config']]

        for parameter_path, parameter_range in self.config['parameter_ranges']:
            configs = []
            for process_parameter in self.process_parameters:
                for parameter_value in parameter_range:
                    next_parameters = copy.deepcopy(process_parameter)
                    set_path(next_parameters, parameter_path, parameter_value)
                    configs.append(next_parameters)
            self.process_parameters = configs

        bridge = {'outputs': {}}
        state = {}
        for parameters in self.process_parameters:
            parameters_key = generate_key(parameters)
            bridge['outputs'][f'{parameters_key}'] = [f'{parameters_key}']

            for initial_key, initial_value in self.config['initial_state'].items():
                state[f'{initial_key}_{parameters_key}'] = initial_value

            state[f'process_{parameters_key}'] = {
                '_type': 'step',
                'address': 'local:!process_bigraph.experiments.parameter_scan.RunProcess',
                'config': {
                    'process_address': self.config['process_address'],
                    'process_config': parameters,
                    'timestep': self.config['timestep'],
                    'runtime': self.config['runtime']},
                # TODO: these could be the same if the internal process uses its own state
                #   for calculating history?
                'inputs': {
                    initial_key: [f'{initial_key}_{parameters_key}']
                    for initial_key in self.config['initial_state'].keys()},
                'outputs': {'results': [f'{parameters_key}']}}

        # TODO: perform parallelization on the independent steps
        self.scan = Composite({
            'bridge': bridge,
            'state': state})

        results_schema = {}
        process = self.first_process()
        for parameters in self.process_parameters:
            parameters_key = generate_key(parameters)
            results_schema[parameters_key] = {
                'time': 'float'}

            for observable_path in self.config['observables']:
                observable_schema, _ = self.core.slice(
                    process.outputs(),
                    {},
                    observable_path)

                set_path(
                    results_schema[parameters_key],
                    observable_path,
                    observable_schema)

        self.results_schema = results_schema

        # # TODO: get schema for observables and use them for outputs
        # process = self.first_process()
        # config_schema = {}
        # results_schema = {}
        # for parameter_path, parameter_range in self.config['parameter_ranges']:
        #     parameter_schema, _ = self.core.slice(
        #         process.config_schema,
        #         {},
        #         parameter_path)
        #     set_path(config_schema, parameter_path, parameter_schema)

        # self.results_schema = results_schema

        import ipdb; ipdb.set_trace()


    def first_process(self):
        for key, value in self.scan.state.items():
            if key.startswith('process_'):
                return value['instance'].composite.state['process']['instance']


    def outputs(self):
        return {
            'results': self.results_schema}

        # return {'results': 'tree[any]'}
        # return {
        #     'results': {
        #         '_type': 'array',
        #         '_data': 'float',
        #         '_shape': self.results_shape}}


    def update(self, inputs):
        results = self.scan.update({}, 0.0)

        # TODO: Figure out the right format for returning results -
        #   perhaps more like we are already receiving from the scan,
        #   with parameters and keys present
        result_list = []
        for result in results:
            observable_list = []
            key = list(result.keys())[0]
            values = list(result.values())[0]

            for observable in self.config['observables']:
                value = get_path(values, observable)
                observable_list.append(value)
                    # np.array(list(value)))
                    # np.array(list(result.values())[0][observable]))

            result_list.append(observable_list)

            # result_list.append(
            #     np.array(observable_list))

        import ipdb; ipdb.set_trace()

        return {
            'results': result_list}
            # 'results': np.array(result_list)}


# TODO: support dataframe type?
#   something like this:

# row_schema = {
#     'first name': 'string',
#     'age': 'integer'}

# dataframe_schema = {
#     '_type': 'list',
#     '_element': row_schema}


def test_run_process():
    timestep = 0.1
    runtime = 10.0

    state = {
        'species': {'A': 11.11},
        'run': {
            '_type': 'step',
            'address': 'local:!process_bigraph.experiments.parameter_scan.RunProcess',
            'config': {
                'process_address': 'local:!process_bigraph.experiments.parameter_scan.ToySystem',
                'process_config': {
                    'rates': {
                        'A': {
                            'kdeg': 1.1,
                            'ksynth': 0.9}}},
                'timestep': timestep,
                'runtime': runtime},
            # '_outputs': {'results': {'_emit': True}},
            'inputs': {'species': ['species']},
            'outputs': {'results': ['A_results']}}}

    process = Composite({
        'bridge': {
            'outputs': {
                'results': ['A_results']}},
        'state': state})

    results = process.update({}, 0.0)

    assert results[0]['results']['time'][-1] == runtime

    import ipdb; ipdb.set_trace()


def test_parameter_scan():
    # TODO: make a parameter scan with a biosimulator process,
    #   ie - Copasi

    state = {
        'scan': {
            '_type': 'step',
            'address': 'local:!process_bigraph.experiments.parameter_scan.ParameterScan',
            'config': {
                'parameter_ranges': [
                    (['rates', 'A', 'kdeg'], [0.0, 0.1, 1.0, 10.0])],
                'process_address': 'local:!process_bigraph.experiments.parameter_scan.ToySystem',
                'process_config': {
                    'rates': {
                        'A': {
                            'ksynth': 1.0}}},
                # TODO: allow for observables that live at paths
                'observables': [['species']],
                'initial_state': {
                    'species': {
                        'A': 13.3333}},
                'timestep': 1.0,
                'runtime': 10},
            'outputs': {
                'results': ['results']}}}

    scan = Composite({
        'bridge': {
            'outputs': {
                'results': ['results']}},
        'state': state})
            
    scan.run(0.0)

    import ipdb; ipdb.set_trace()


# scan = {
#     'scan': {
#         '_type': 'step',
#         'address': 'local:!process_bigraph.experiments.parameter_scan.ParameterScan'},

#     'processes': {
#         'process_0': {
#                       '_type': 'step'}}
# }


if __name__ == '__main__':
    test_run_process()
    test_parameter_scan()
