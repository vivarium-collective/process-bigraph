import numpy as np

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

    # 'rates': {
    #     '_type': 'array',
    #     '_data': 'float'},


class ToySystem(Process):
    config_schema = {
        'kdeg': 'float',
        'ksynth': 'float'}


    def inputs(self):
        return {
            'A': 'float'}


    def outputs(self):
        return {
            'A': 'float'}


    def update(self, inputs, interval):
        return {
            'A': inputs['A'] * (self.config['ksynth'] - self.config['kdeg'])}



class ODE(Process):
    config_schema = 'ode_config'
    # config_schema = {
    #     'rates': {
    #         '_type': 'array',
    #         '_data': 'float'},
    #     'species_names': 'list[string]'}


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
        'parameter_ranges': 'map[list[float]]',
        'process_address': 'string',
        'process_config': 'tree[any]',
        'initial_state': 'tree[any]',
        'observables': 'list[string]',
        'timestep': 'float',
        'runtime': 'float'}


    def __init__(self, config, core):
        super().__init__(config, core)

        self.steps_count = int(
            self.config['runtime'] / self.config['timestep']) + 1 # TODO shouldn't need to add 1
        self.observables_count = len(
            self.config['observables'])

        self.total_combinations = 1
        results_shape = []
        for parameter_key, ranges in self.config['parameter_ranges'].items():
            ranges_count = len(ranges)
            self.total_combinations *= ranges_count
            results_shape.append(ranges_count)

        results_shape.extend([
            self.observables_count,
            self.steps_count])

        self.results_shape = tuple(results_shape)

        self.process_parameters = [
            self.config['process_config']]

        for parameter_key, parameter_range in self.config['parameter_ranges'].items():
            configs = []
            for process_parameter in self.process_parameters:
                for parameter_value in parameter_range:
                    configs.append(dict(
                        process_parameter,
                        **{parameter_key: parameter_value}))
            self.process_parameters = configs

        bridge = {'outputs': {}}
        state = {}
        for parameters in self.process_parameters:
            parameters_key = generate_key(parameters)
            bridge['outputs'][f'results_{parameters_key}'] = [f'results_{parameters_key}']

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
                'inputs': {
                    initial_key: [f'{initial_key}_{parameters_key}']
                    for initial_key in self.config['initial_state'].keys()},
                'outputs': {'results': [f'results_{parameters_key}']}}

        self.scan = Composite({
            'bridge': bridge,
            'state': state})


    def outputs(self):
        return {
            'results': {
                '_type': 'array',
                '_data': 'float',
                '_shape': self.results_shape}}


    def update(self, inputs):
        results = self.scan.update({}, 0.0)

        result_list = []
        for result in results:
            observable_list = []
            for observable in self.config['observables']:
                observable_list.append(
                    np.array(list(result.values())[0][observable]))

            result_list.append(
                np.array(observable_list))

        return {
            'results': np.array(result_list)}


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
        'A': 11.11,
        'run': {
            '_type': 'step',
            'address': 'local:!process_bigraph.experiments.parameter_scan.RunProcess',
            'config': {
                'process_address': 'local:!process_bigraph.experiments.parameter_scan.ToySystem',
                'process_config': {
                    'kdeg': 1.1,
                    'ksynth': 0.9},
                'timestep': timestep,
                'runtime': runtime},
            # '_outputs': {'results': {'_emit': True}},
            'inputs': {'A': ['A']},
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
    state = {
        'scan': {
            '_type': 'step',
            'address': 'local:!process_bigraph.experiments.parameter_scan.ParameterScan',
            'config': {
                'parameter_ranges': {
                    'kdeg': [0.0, 0.1, 1.0, 10.0]},
                'process_address': 'local:!process_bigraph.experiments.parameter_scan.ToySystem',
                'process_config': {
                    'ksynth': 1.0},
                'observables': ['A'],
                'initial_state': {'A': 13.3333},
                'timestep': 1.0,
                'runtime': 10},
            'outputs': {
                'results': ['results']}}}

    scan = Composite({
        'bridge': {
            'outputs': {
                'results': ['results']}},
        'state': state})
            
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_run_process()
    test_parameter_scan()
