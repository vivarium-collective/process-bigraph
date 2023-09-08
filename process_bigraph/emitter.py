import copy
from bigraph_schema import get_path, set_path

from process_bigraph.composite import Step, Process
from process_bigraph import process_registry


class Emitter(Step):
    def query(self, query=None):
        return {}


class ConsoleEmitter(Emitter):
    config_schema = {
        'ports': 'tree[any]'}


    def schema(self):
        return self.config['ports']


    def update(self, state):
        print(state)

        return {}


class RAMEmitter(Emitter):
    config_schema = {
        'ports': 'tree[any]'}


    def __init__(self, config):
        super().__init__(config)

        self.history = []


    def schema(self):
        return self.config['ports']


    def update(self, state):
        self.history.append(copy.deepcopy(state))

        return {}


    def query(self, query=None):
        if isinstance(query, list):
            result = {}
            for path in query:
                element = get_path(self.history, path)
                result = set_path(result, path, element)
        else:
            result = self.history

        return result
