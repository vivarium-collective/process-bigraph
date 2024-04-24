from process_bigraph import Step, Process, Composite, ProcessTypes, interval_time_precision, deep_merge


def Growth(Process):
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
    'id': 'string',
    
}


# TODO: build composite and divide within it

def Divide(Step):
    # assume the agent_schema has the right divide methods present
    config_schema = {
        'agent_schema': 'schema',
        'threshold': 'float'}


    def __init__(self, config, core=None):
        super().__init__(config, core)


    def inputs(self):
        return {
            'trigger': 'float'}


    def outputs(self):
        return {
            'self': self.config['agent_schema']}


    def update(self, state):
        if state['trigger'] > self.config['threshold']:
            # return divide reaction
            return {
                'self': {
                    '_fold': {
                        'method': 'divide',
                        'divisions': 2}}}
