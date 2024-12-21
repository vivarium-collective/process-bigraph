"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""
import os
import json

from process_bigraph import ProcessTypes, Composite
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent
from bigraph_schema import is_schema_key


class Vivarium:
    """
    Vivarium is a controlled virtual environment for composite process-bigraph simulations.

    It manages packages and sets up the conditions for running simulations, and collects results through emitters.

    Attributes:
        document (dict): The configuration document for the simulation.
        core (ProcessTypes): The core process types manager.
        composite (Composite): The composite object managing the simulation.
        require (list): List of required packages for the simulation.
    """
    def __init__(self,
                 document=None,
                 processes=None,
                 emitter_config=None,
                 ):
        processes = processes or {}
        emitter_config = emitter_config or {"mode": "all"}

        self.document = document

        # make the core
        self.core = ProcessTypes()

        # register processes
        self.core.register_processes(processes)

        # register other packages
        self.require = document.pop('require', [])
        for require in self.require:
            package = self.find_package(require)
            self.core.register_types(package.get('types', {}))

        # add emitter
        if 'emitter' not in self.document:
            # self.add_emitter(emitter_config)
            self.document['emitter'] = emitter_config

        # make the composite
        self.composite = Composite(
            self.document,
            core=self.core)

    # TODO -- replace Composite's add emitter with this
    # def add_emitter(self, emitter_config=None):
    #     address = emitter_config.get('address', 'local:ram-emitter')
    #     config = emitter_config.get('config', {})
    #     mode = emitter_config.get('mode', 'none')
    #
    #     if mode == 'all':
    #         inputs = {
    #             key: [emitter_config.get('inputs', {}).get(key, key)]
    #             for key in self.composite.state.keys()
    #             if not is_schema_key(key)}
    #
    #     elif mode == 'none':
    #         inputs = emitter_config.get('emit', {})
    #
    #     elif mode == 'bridge':
    #         inputs = {}
    #
    #     elif mode == 'ports':
    #         inputs = {}
    #
    #     if not 'emit' in config:
    #         config['emit'] = {
    #             input: 'any'
    #             for input in inputs}
    #
    #     return {
    #         '_type': 'step',
    #         'address': address,
    #         'config': config,
    #         'inputs': inputs}


    def get_document(self,
                     schema=False,
                     ):
        document = {}

        document['state'] = self.core.serialize(
            self.composite.composition,
            self.composite.state)

        # if schema:
        #     serialized_schema = self.core.representation(
        #         self.composite.composition)
        #     document['composition'] = serialized_schema

        return document


    def save(self,
             filename='simulation.json',
             outdir='out',
             schema=True,
             ):
        # TODO: add in dependent packages and version
        #   maybe packagename.typename?
        # TODO: add in dependent types

        document = self.get_document(schema=schema)

        # save the dictionary to a JSON file
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = os.path.join(outdir, filename)

        # write the new data to the file
        with open(filename, 'w') as json_file:
            json.dump(document, json_file, indent=4)
            print(f"Created new file: {filename}")


    def find_package(self, package):
        pass


    def run(self, interval):
        self.composite.run(interval)


    def step(self):
        self.composite.update({}, 0)


    def get_results(self, queries=None):
        results = self.composite.gather_results(queries=queries)
        return results[('emitter',)]


    def get_timeseries(self, queries=None):
        results = self.composite.gather_results(queries=queries)
        emitter_results = results[('emitter',)]

        def append_to_timeseries(timeseries, state, path=()):
            if isinstance(state, dict):
                if (state.get('address') in ['process', 'step', 'composite']) or state.get('address'):
                    return
                for key, value in state.items():
                    append_to_timeseries(timeseries, value, path + (key,))
            else:
                if path not in timeseries:
                    # TODO -- what if entry appeared in the middle of the simulation? Fill with Nones?
                    timeseries[path] = []
                timeseries[path].append(state)

        # get the timeseries from the emitter results
        timeseries = {}
        for state in emitter_results:
            append_to_timeseries(timeseries, state)

        # Convert tuple keys to string keys for better readability
        timeseries = {'.'.join(key): value for key, value in timeseries.items()}

        return timeseries



def example_package():
    return {
        'name': 'sbml',
        'version': '1.1.33',
        'types': {
            'modelfile': 'string'}}


def example_document():
    return {
        'require': [
            'sbml==1.1.33'],
        'composition': {
            'hello': 'string'},
        'state': {
            'hello': 'world!'}}


def test_vivarium():
    initial_mass = 1.0

    grow_divide = grow_divide_agent(
        {'grow': {'rate': 0.03}},
        {},
        ['environment', '0'])

    environment = {
        'environment': {
            '0': {
                'mass': initial_mass,
                'grow_divide': grow_divide}}}

    document = {
        'state': environment,
    }

    sim = Vivarium(document=document, processes=TOY_PROCESSES)
    sim.run(interval=40.0)
    results = sim.get_timeseries()

    print(results)

    sim.save('test_vivarium.json')


if __name__ == '__main__':
    test_vivarium()
