import pytest
from process_bigraph import Step, Process, Composite, ProcessTypes, interval_time_precision, deep_merge


class Grow(Process):
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


# TODO: build composite and divide within it

class Divide(Step):
    # assume the agent_schema has the right divide methods present
    config_schema = {
        'agent_id': 'string',
        'agent_schema': 'schema',
        'threshold': 'float',
        'divisions': {
            '_type': 'integer',
            '_default': 2}}


    def __init__(self, config, core=None):
        super().__init__(config, core)


    def inputs(self):
        return {
            'trigger': 'float'}


    def outputs(self):
        return {
            'environment': {
                '_type': 'map',
                '_value': self.config['agent_schema']}}


    # this should be generalized to some function that depends on
    # state from the self.config['agent_schema'] (instead of trigger > threshold)
    def update(self, state):
        if state['trigger'] > self.config['threshold']:
            mother = self.config['agent_id']
            daughters = [(
                f'{mother}_{i}', {
                    'state': {
                        'divide': {
                            'config': {
                                'agent_id': f'{mother}_{i}'}}}})
                for i in range(self.config['divisions'])]

            # return divide reaction
            return {
                'environment': {
                    '_react': {
                        'divide': {
                            'mother': mother,
                            'daughters': daughters}}}}


def generate_bridge_wires(schema):
    return {
        key: [key]
        for key in schema
        if not key.startswith('_')}


def generate_bridge(schema, state, interval=1.0):
    bridge = {
        port: generate_bridge_wires(schema[port])
        for port in ['inputs', 'outputs']}

    config = {
        'state': state,
        'bridge': bridge}

    composite = {
        '_type': 'process',
        'address': 'local:composite',
        'interval': interval,
        'config': config,
        'inputs': generate_bridge_wires(schema['inputs']),
        'outputs': generate_bridge_wires(schema['outputs'])}

    return composite


def grow_divide_agent(config=None, state=None, path=None):
    agent_id = path[-1]

    config = config or {}
    state = state or {}
    path = path or []

    agent_schema = config.get(
        'agent_schema',
        {'mass': 'float'})

    grow_config = {
        'rate': 0.1}

    grow_config = deep_merge(
        grow_config,
        config.get(
            'grow'))

    divide_config = {
        'agent_id': agent_id,
        'agent_schema': agent_schema,
        'threshold': 2.0,
        'divisions': 2}

    divide_config = deep_merge(
        divide_config,
        config.get(
            'divide'))

    grow_divide_state = {
        'grow': {
            '_type': 'process',
            'address': 'local:grow',
            'config': grow_config,
            'inputs': {
                'mass': ['mass']},
            'outputs': {
                'mass': ['mass']}},

        'divide': {
            '_type': 'process',
            'address': 'local:divide',
            'config': divide_config,
            'inputs': {
                'trigger': ['mass']},
            'outputs': {
                'environment': ['environment']}}}

    grow_divide_state = deep_merge(
        grow_divide_state,
        state)

    composite = generate_bridge({
        'inputs': {},
        'outputs': agent_schema},
        grow_divide_state)

    composite['config']['bridge']['outputs']['environment'] = ['environment']
    composite['outputs']['environment'] = ['..']

    return composite


def test_grow_divide(core):
    initial_mass = 1.0

    grow_divide = grow_divide_agent(
        {'grow': {'rate': 0.03}},
        {'mass': initial_mass},
        ['environment', '0'])

    environment = {
        'environment': {
            '0': {
                'mass': initial_mass,
                'grow_divide': grow_divide}}}

    composite = Composite({
        'state': environment},
        core=core)

    updates = composite.update({}, 100.0)


@pytest.fixture
def core():
    core = ProcessTypes()
    core.register_process('grow', Grow)
    core.register_process('divide', Divide)
    return core


if __name__ == '__main__':
    core = ProcessTypes()
    core.register_process('grow', Grow)
    core.register_process('divide', Divide)

    test_grow_divide(core)
