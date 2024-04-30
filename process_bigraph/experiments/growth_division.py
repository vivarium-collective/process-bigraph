from process_bigraph import Step, Process, Composite, ProcessTypes, interval_time_precision, deep_merge


def Grow(Process):
    config_schema = {
        'rate': 'float'}


    def __init__(self, config, core=None):
        super().__init__(config, core)


    def inputs(self):
        return {
            'mass': 'float'}


    def outputs(self):
        return {
            'mass': 'float'}


    def update(self, state, interval):
        # this calculates a delta

        return {
            'mass': state['mass'] * self.config['rate'] * interval}


example_agent_schema = {
    'id': 'string'
}


# TODO: build composite and divide within it

def Divide(Step):
    # assume the agent_schema has the right divide methods present
    config_schema = {
        'agent_id': 'string',
        'agent_schema': 'schema',
        'threshold': 'float'}


    def __init__(self, config, core=None):
        super().__init__(config, core)


    def inputs(self):
        return {
            'trigger': 'float',
            'self': self.config['agent_schema']}


    def outputs(self):
        return {
            'environment': {
                '_type': 'map',
                '_value': self.config['agent_schema']}}


    def update(self, state):
        if state['trigger'] > self.config['threshold']:
            # # return divide reaction
            # daughters = self.core.fold(
            #     self.config['agent_config'],
            #     state['self'],
            #     'divide',
            #     {'divisions': 2})

            # before = {
            #     self.config['agent_id']: {}}

            # after = {}

            return {
                'environment': {
                    '_react': {
                        'divide': {
                            'path': [self.config['agent_id']]}}}}

            # return {
            #     'environment': {
            #         '_react': {
            #             'replace': {
            #                 'before': before,
            #                 'after': after,
            #                 'path': path}}}}



def grow_divide_composite(core):
    core.register_process('grow', Grow)
    core.register_process('divide', Divide)

    composite = {
        'grow': {
            'address': 'local:grow',
            'config': {
                'rate': 0.1},
            'inputs': {
                'mass': ['mass']},
            'outputs': {
                'mass': ['mass']}},
        'divide': {
            'address': 'local:divide',
            'config': {
                'rate': 0.1},
            'inputs': {
                'mass': ['mass']},
            'outputs': {
                'mass': ['mass']}}}


def test_grow_divide():
    core = ProcessTypes()
    composite = grow_divide_composite(core)


if __name__ == '__main__':
    test_grow_divide()
