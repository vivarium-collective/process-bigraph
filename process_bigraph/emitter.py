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
        self.history.append(state)

        return {}


    def query(self, query=None):
        return self.history
