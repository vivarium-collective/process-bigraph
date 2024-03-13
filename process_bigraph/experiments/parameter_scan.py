from process_bigraph import Step, Process, Composite, ProcessTypes


core = ProcessTypes()


core.register('scannable_process', {
    '_type': 'process',
    '_inputs': {
        'species': {
            '_type': 'array',
            '_data': 'float'}},
    '_outputs': {
        'species': {
            '_type': 'array',
            '_data': 'float'}}})


core.register('ode_config', {
    'rates': {
        '_type': 'array',
        '_data': 'float'},
    'species_names': 'list[string]'})


class ODE(Process):
    config_schema = 'ode_config'
    # config_schema = {
    #     'rates': {
    #         '_type': 'array',
    #         '_data': 'float'},
    #     'species_names': 'list[string]'}


    def __init__(self, config, core):
        super().__init__(config, core)

        self.species_count = len(self.config['species_name'])
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

        return total - state['species']


# TODO: This step during __init__ generates the composite containing all of the processes
#   we are scanning over

class ParameterScan(Step):
    config_schema = {
        'parameter_ranges': 'map[list[float]]',
        'process_address': 'string',
        'process_schema': 'scannable_process',
        'process_config': 'ode_config',
        'timestep': 'float',
        'runtime': 'float'}


    def __init__(self, config, core):
        super().__init__(config, core)

        self.steps_count = int(self.config['runtime'] / self.config['timestep'])
        self.species_count = len(self.config['process_config']['species_names'])

        self.total_combinations = 1
        results_shape = []
        for parameter_key, ranges in self.config['parameter_ranges'].items():
            ranges_count = len(ranges)
            self.total_combinations *= ranges_count
            results_shape.append(ranges_count)

        results_shape.extend([
            self.species_count,
            self.step_count])

        self.results_shape = tuple(results_shape)
        process_parameters = [{}]
        for parameter_key, parameter_range in self.config['parameter_ranges'].items():
            for parameter_value in parameter_range:
                configs = []
                for process_parameter in process_parameters:
                    configs.append(dict(
                        process_parameter,
                        **{parameter_key: parameter_value}))
                process_parameters = configs


    def outputs(self):
        return {
            'results': {
                '_type': 'array',
                '_data': 'float',
                '_shape': self.results_shape}


    def update(self, inputs):
        import ipdb; ipdb.set_trace()


# TODO: support dataframe type?
#   something like this:

# row_schema = {
#     'first name': 'string',
#     'age': 'integer'}

# dataframe_schema = {
#     '_type': 'list',
#     '_element': row_schema}


def test_parameter_scan():
    schema = {
        'scan': 'step'}

    state = {
        'scan': {
            'address': 'local:!process_bigraph.experiments.parameter_scan.ParameterScan',
            'config': {
                ''
}
}
}


if __name__ == '__main__':
    test_parameter_scan()
