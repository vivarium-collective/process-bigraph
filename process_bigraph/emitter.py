from process_bigraph.composite import Step, Process
from process_bigraph import process_registry


class ConsoleEmitter(Step):
    config_schema = {
        'ports': 'tree[any]'}


    def schema(self):
        return self.config['ports']


    def update(self, state):
        print(state)

        return {}


