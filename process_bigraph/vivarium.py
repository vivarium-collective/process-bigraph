"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""

from process_bigraph import ProcessTypes, Composite
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent


class Vivarium:
    def __init__(self,
                 document=None,
                 processes=None,
                 ):
        processes = processes or {}

        self.document = document

        # add emitter
        if 'emitter' not in self.document:
            self.document['emitter'] = {'mode': 'all'}

        # make the core
        self.core = ProcessTypes()
        self.core.register_processes(processes)

        # packages
        self.require = document.pop('require', [])
        for require in self.require:
            package = self.find_package(require)
            self.core.register_types(package.get('types', {}))

        self.composite = Composite(
            self.document,
            core=self.core)

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


if __name__ == '__main__':
    test_vivarium()
