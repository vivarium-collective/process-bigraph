"""
========
Emitters
========

Emitters are steps that observe the state of the system and emit it to an external source.
This could be to a database, to a file, or to the console.
"""
import copy
from typing import Dict

from bigraph_schema import get_path, set_path

from process_bigraph.composite import Step, find_instance_paths


class Emitter(Step):
    """Base emitter class.

    An `Emitter` implementation instance diverts all querying of data to
    the primary historical collection whose type pertains to Emitter child, i.e:
     database-emitter=>`pymongo.Collection`, ram-emitter=>`.RamEmitter.history`(`List`)
    """
    config_schema = {
        'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

    def query(self, query=None):
        return {}

    def update(self, state) -> Dict:
        return {}


class ConsoleEmitter(Emitter):
    """Console emitter class.

    This emitter logs the state to the console.
    """

    def update(self, state) -> Dict:
        print(state)
        return {}


class RAMEmitter(Emitter):
    """RAM emitter class.

    This emitter logs the state to a list in memory.
    """

    def __init__(self, config, core):
        super().__init__(config, core)
        self.history = []


    def update(self, state) -> Dict:
        self.history.append(copy.deepcopy(state))
        return {}


    def query(self, query=None):
        """
        Query the history of the emitter.
        :param query: a list of paths to query from the history. If None, the entire history is returned.
        :return: results of the query in a list
        """
        if isinstance(query, list):
            results = []
            for t in self.history:
                result = {}
                for path in query:
                    element = get_path(t, path)
                    result = set_path(result, path, element)
                results.append(result)
                # element = get_path(self.history, path)
                # result = set_path(result, path, element)
        else:
            results = self.history

        return results


def gather_results(composite, queries=None):
    '''
    a map of paths to emitter --> queries for the emitter at that path
    '''

    emitter_paths = find_instance_paths(
        composite.state,
        instance_type='process_bigraph.emitter.Emitter')

    if queries is None:
        queries = {
            path: None
            for path in emitter_paths.keys()}

    results = {}
    for path, query in queries.items():
        emitter = get_path(composite.state, path)
        results[path] = emitter['instance'].query(query)

        # TODO: unnest the results?
        # TODO: allow the results to be transposed

    return results


BASE_EMITTERS = {
    'console-emitter': ConsoleEmitter,
    'ram-emitter': RAMEmitter}
