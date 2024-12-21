import copy
import numpy as np

from bigraph_schema import get_path, set_path, transform_path
from process_bigraph.composite import Step, Process, Composite, interval_time_precision, deep_merge


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

    def initialize(self, config=None):
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
        'observables': 'list[path]',
        'timestep': 'float',
        'runtime': 'float'}

    def initialize(self, config):

        self.process = self.core.deserialize('process', {
            '_type': 'process',
            'address': self.config['process_address'],
            'config': self.config['process_config'],
            'inputs': {},
            'outputs': {}})['instance']

        global_time_precision = interval_time_precision(
            self.config['timestep'])

        process_outputs = self.process.outputs()
        self.observables_schema = {}
        self.results_schema = {}
        self.inputs_config = {}

        for observable in self.config['observables']:
            subschema, _ = self.core.slice(
                process_outputs,
                {},
                observable)

            set_path(
                self.observables_schema,
                observable,
                subschema)

            set_path(
                self.results_schema,
                observable, {
                    '_type': 'list',
                    '_element': subschema})

            set_path(
                self.inputs_config,
                observable,
                observable)
                # [observable[-1]])

        emit_config = dict(
            {'time': 'float'},
            **self.observables_schema)

        composite_config = {
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
                    '_inputs': self.process.inputs(),
                    '_outputs': self.process.outputs(),
                    'inputs': {
                        key: [key]
                        for key in self.process.inputs()},
                    'outputs': {
                        key: [key]
                        for key in process_outputs}},
                'emitter': {
                    '_type': 'step',
                    '_inputs': emit_config,
                    'address': 'local:ram-emitter',
                    'config': {
                        'emit': emit_config},
                    'inputs': dict(
                        {'time': ['global_time']},
                        **self.inputs_config),
                    'outputs': {}}}}

        self.composite = Composite(composite_config, core=self.core)

    def inputs(self):
        return self.process.inputs()

    def outputs(self):
        return {
            'results': dict(
                {'time': 'list[float]'},
                **self.results_schema)}

    def update(self, inputs):
        # TODO: instead of the composite being a reference it is instead read through
        #   some port and lives in the state of the simulation (??)
        self.composite.merge(
            self.inputs(),
            inputs)

        self.composite.run(
            self.config['runtime'])

        histories = self.composite.gather_results()

        results = {
            key: timeseries_from_history(
                history,
                self.config['observables'] + [['time']])
            for key, history in histories.items()}

        all_results = {}
        for timeseries in results.values():
            all_results = deep_merge(all_results, timeseries)

        return {'results': all_results}


def timeseries_from_history(history, observables):
    results = {}
    for moment in history:
        for observable in observables:
            def transform(before):
                if not before:
                    before = []
                value = get_path(moment, observable)
                before.append(value)
                return before

            transform_path(results, observable, transform)

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

    def initialize(self, config=None):

        self.steps_count = int(
            self.config['runtime'] / self.config['timestep']) + 1
        self.observables_count = len(
            self.config['observables'])

        # TODO: test two parameters scanning simultaneously

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
                'address': 'local:RunProcess',
                'config': {
                    'process_address': self.config['process_address'],
                    'process_config': parameters,
                    'observables': self.config['observables'],
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
            'state': state},
            core=self.core)

        results_schema = {}
        process = self.first_process()
        for parameters in self.process_parameters:
            parameters_key = generate_key(parameters)
            results_schema[parameters_key] = {
                'time': 'list[float]'}

            for observable_path in self.config['observables']:
                observable_schema, _ = self.core.slice(
                    process.outputs(),
                    {},
                    observable_path)

                set_path(
                    results_schema[parameters_key],
                    observable_path,
                    {'_type': 'list', '_element': observable_schema})

        self.results_schema = results_schema

    def first_process(self):
        for key, value in self.scan.state.items():
            if key.startswith('process_'):
                return value['instance'].composite.state['process']['instance']

    def outputs(self):
        return {
            'results': self.results_schema}

    def update(self, inputs):
        results = self.scan.update({}, 0.0)

        update = {}
        for result in results:
            observable_list = []
            key = list(result.keys())[0]
            values = list(result.values())[0]
            update[key] = {'time': values['time']}

            for observable in self.config['observables']:
                subschema = self.results_schema[key]
                value_schema, value = self.core.slice(
                    subschema,
                    values,
                    observable)

                set_path(
                    update[key],
                    observable,
                    value)

        return {
            'results': update}

