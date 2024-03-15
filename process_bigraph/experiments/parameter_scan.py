from process_bigraph import Step, Process, Composite, ProcessTypes


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


# TODO: This step during __init__ generates the composite containing all of the processes
#   we are scanning over

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

        self.composite = Composite({'state': {
            'process': {
                '_type': 'process',
                'address': self.config['process_address'],
                'config': self.config['process_config'],
                'instance': self.process,
                'interval': self.config['timestep'],
                'inputs': {},
                'outputs': {}},
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': self.process.outputs()},
                'inputs': {'emit': list(self.process.outputs().keys())},
                'outputs': {}}}})


    def inputs(self):
        return self.process.inputs()


    def outputs(self):
        return {
            output_key: {
                '_type': 'list',
                '_element': output_schema}
            for output_key, output_schema in self.process.outputs()}


    def update(self, inputs):
        # TODO: make method for setting the state of a composite
        self.composite.state = self.core.set(
            self.composite.composition,
            self.composite.state,
            inputs)

        import ipdb; ipdb.set_trace()

        self.composite.run(
            self.config['runtime'])

        results = self.composite.gather_results()

        return results


class ParameterScan(Step):
    config_schema = {
        'parameter_ranges': 'map[list[float]]',
        'process_address': 'string',
        'process_config': 'tree[any]',
        'observables': 'list[string]',
        'timestep': 'float',
        'runtime': 'float'}


    def __init__(self, config, core):
        super().__init__(config, core)

        self.steps_count = int(self.config['runtime'] / self.config['timestep'])
        self.observables_count = len(self.config['observables'])

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


    def outputs(self):
        return {
            'results': {
                '_type': 'array',
                '_data': 'float',
                '_shape': self.results_shape}}


    def update(self, inputs):
        scan = {}

#         for index, parameters in enumerate(self.process_parameters):
#             scan[str(index)] = {
#                 '_type': 'step',
#                 'address': 
#                 'config':
# }

        import ipdb; ipdb.set_trace()


# TODO: support dataframe type?
#   something like this:

# row_schema = {
#     'first name': 'string',
#     'age': 'integer'}

# dataframe_schema = {
#     '_type': 'list',
#     '_element': row_schema}


def test_run_process():
    state = {
        'run': {
            '_type': 'step',
            'address': 'local:!process_bigraph.experiments.parameter_scan.RunProcess',
            'config': {
                'process_address': 'local:!process_bigraph.experiments.parameter_scan.ToySystem',
                'process_config': {
                    'kdeg': 1.0,
                    'ksynth': 1.0},
                'timestep': 0.1,
                'runtime': 10.0}}}

    run = Composite({
        'state': state})

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
                'timestep': 1.0,
                'runtime': 10},
            'outputs': {
                'results': ['results']}}}

    scan = Composite({
        'state': state})
            
    import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    test_run_process()
    test_parameter_scan()
