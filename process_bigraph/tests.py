"""
Tests for Process Bigraph
"""
import pytest
import random

from bigraph_schema import default
from process_bigraph import register_types

from process_bigraph.composite import Process, Step, Composite, merge_collections, match_star_path, ProcessTypes

from process_bigraph.processes.growth_division import grow_divide_agent, Grow, Divide
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results


@pytest.fixture
def core():
    core = ProcessTypes()
    return register_types(core)


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'}}

    def inputs(self):
        return {
            'level': 'float'}

    def outputs(self):
        return {
            'level': 'float'}

    def accelerate(self, delta):
        self.config['rate'] += delta

    def initial_state(self):
        return {
            'level': 4.4}

    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']}


class IncreaseRate(Step):
    config_schema = {
        'acceleration': default('float', 0.001)}

    def inputs(self):
        return {
            'level': 'float'}

    def update(self, state):
        # TODO: this is ludicrous.... never do this
        #   probably this feature should only be used for reading
        self.instance.accelerate(
            self.config['acceleration'] * state['level'])


def test_default_config(core):
    process = IncreaseProcess(core=core)

    assert process.config['rate'] == 0.1


def test_merge_collections(core):
    a = {('what',): [1, 2, 3]}
    b = {('okay', 'yes'): [3, 3], ('what',): [4, 5, 11]}

    c = merge_collections(a, b)

    assert c[('what',)] == [1, 2, 3, 4, 5, 11]


def test_process(core):
    process = IncreaseProcess({'rate': 0.2}, core=core)
    interface = process.interface()
    state = core.fill(interface['inputs'])
    state = core.fill(interface['outputs'])
    update = process.update({'level': 5.5}, 1.0)

    new_state = core.apply(
        interface['outputs'],
        state,
        update)

    assert new_state['level'] == 1.1


def test_composite(core):
    # TODO: add support for the various vivarium emitter

    # increase = IncreaseProcess({'rate': 0.3})
    # TODO: This is the config of the composite,
    #   we also need a way to serialize the entire composite

    composite = Composite({
        'composition': {
            'increase': 'process[level:float,level:float]',
            'value': 'float'},
        'interface': {
            'inputs': {
                'exchange': 'float'},
            'outputs': {
                'exchange': 'float'}},
        'bridge': {
            'inputs': {
                'exchange': ['value']},
            'outputs': {
                'exchange': ['value']}},
        'state': {
            'increase': {
                'address': 'local:!process_bigraph.tests.IncreaseProcess',
                'config': {'rate': 0.3},
                'interval': 1.0,
                'inputs': {'level': ['value']},
                'outputs': {'level': ['value']}},
            'value': '11.11'}}, core=core)

    initial_state = {'exchange': 3.33}

    updates = composite.update(initial_state, 10.0)

    final_exchange = sum([
        update['exchange']
        for update in [initial_state] + updates])

    assert composite.state['value'] > 45
    assert 'exchange' in updates[0]
    assert updates[0]['exchange'] == 0.999


def test_infer(core):
    composite = Composite({
        'state': {
            'increase': {
                '_type': 'process',
                'address': 'local:!process_bigraph.tests.IncreaseProcess',
                'config': {'rate': '0.3'},
                'inputs': {'level': ['value']},
                'outputs': {'level': ['value']}},
            'value': '11.11'}}, core=core)

    assert composite.composition['value']['_type'] == 'float'
    assert composite.state['value'] == 4.4


def test_process_type(core):
    assert core.access('process')['_type'] == 'process'


class OperatorStep(Step):
    config_schema = {
        'operator': 'string'}


    def inputs(self):
        return {
            'a': 'float',
            'b': 'float'}


    def outputs(self):
        return {
            'c': 'float'}


    def update(self, inputs):
        a = inputs['a']
        b = inputs['b']

        if self.config['operator'] == '+':
            c = a + b
        elif self.config['operator'] == '*':
            c = a * b
        elif self.config['operator'] == '-':
            c = a - b

        return {'c': c}


def test_step_initialization(core):
    composite = Composite({
        'state': {
            'A': 13,
            'B': 21,
            'step1': {
                '_type': 'step',
                'address': 'local:!process_bigraph.tests.OperatorStep',
                'config': {
                    'operator': '+'},
                'inputs': {
                    'a': ['A'],
                    'b': ['B']},
                'outputs': {
                    'c': ['C']}},
            'step2': {
                '_type': 'step',
                'address': 'local:!process_bigraph.tests.OperatorStep',
                'config': {
                    'operator': '*'},
                'inputs': {
                    'a': ['B'],
                    'b': ['C']},
                'outputs': {
                    'c': ['D']}}}}, core=core)

    composite.run(0.0)
    assert composite.state['D'] == (13 + 21) * 21


def test_dependencies(core):
    operation = {
        'a': 11.111,
        'b': 22.2,
        'c': 555.555,

        '1': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'inputs': {
                'a': ['a'],
                'b': ['b']},
            'outputs': {
                'c': ['e']}},
        '2.1': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'inputs': {
                'a': ['c'],
                'b': ['e']},
            'outputs': {
                'c': ['f']}},
        '2.2': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '-'},
            'inputs': {
                'a': ['d'],
                'b': ['e']},
            'outputs': {
                'c': ['g']}},
        '3': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '*'},
            'inputs': {
                'a': ['f'],
                'b': ['g']},
            'outputs': {
                'c': ['h']}},
        '4': {
            '_type': 'step',
            'address': 'local:!process_bigraph.tests.OperatorStep',
            'config': {
                'operator': '+'},
            'inputs': {
                'a': ['e'],
                'b': ['h']},
            'outputs': {
                'c': ['i']}}}

    composite = Composite(
        {'state': operation},
        core=core)

    composite.run(0.0)

    assert composite.state['h'] == -17396.469884


def test_dependency_cycle():
    # test a step network with cycles in a few ways
    pass


class SimpleCompartment(Process):
    config_schema = {
        'id': 'string'}


    def interface(self):
        return {
            'outer': 'tree[process]',
            'inner': 'tree[process]'}


    def update(self, state, interval):
        choice = random.random()
        update = {}

        outer = state['outer']
        inner = state['inner']

        # TODO: implement divide_state(_)
        divisions = self.core.divide_state(
            self.interface(),
            inner)

        if choice < 0.2:
            # update = {
            #     'outer': {
            #         '_divide': {
            #             'mother': self.config['id'],
            #             'daughters': [
            #                 {'id': self.config['id'] + '0'},
            #                 {'id': self.config['id'] + '1'}]}}}

            # daughter_ids = [self.config['id'] + str(i)
            #     for i in range(2)]

            # update = {
            #     'outer': {
            #         '_react': {
            #             'redex': {
            #                 'inner': {
            #                     self.config['id']: {}}},
            #             'reactum': {
            #                 'inner': {
            #                     daughter_config['id']: {
            #                         '_type': 'process',
            #                         'address': 'local:!process_bigraph.tests.SimpleCompartment',
            #                         'config': daughter_config,
            #                         'inner': daughter_inner,
            #                         'wires': {
            #                             'outer': ['..']}}
            #                     for daughter_config, daughter_inner in zip(daughter_configs, divisions)}}}}}

            update = {
                'outer': {
                    'inner': {
                        '_react': {
                            'reaction': 'divide',
                            'config': {
                                'id': self.config['id'],
                                'daughters': [{
                                        'id': daughter_id,
                                        'state': daughter_state}
                                    for daughter_id, daughter_state in zip(
                                        daughter_ids,
                                        divisions)]}}}}}

        return update


def engulf_reaction(config):
    return {
        'redex': {},
        'reactum': {}}


def burst_reaction(config):
    return {
        'redex': {},
        'reactum': {}}


def test_reaction():
    composite = {
        'state': {
            'environment': {
                'concentrations': {},
                'inner': {
                    'agent1': {
                        '_type': 'process',
                        'address': 'local:!process_bigraph.tests.SimpleCompartment',
                        'config': {'id': '0'},
                        'concentrations': {},
                        'inner': {
                            'agent2': {
                                '_type': 'process',
                                'address': 'local:!process_bigraph.tests.SimpleCompartment',
                                'config': {'id': '0'},
                                'inner': {},
                                'inputs': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']},
                                'outputs': {
                                    'outer': ['..', '..'],
                                    'inner': ['inner']}}},
                        'inputs': {
                            'outer': ['..', '..'],
                            'inner': ['inner']},
                        'outputs': {
                            'outer': ['..', '..'],
                            'inner': ['inner']}}}}}}


def test_emitter(core):
    composite_schema = {
        'bridge': {
            'inputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']},
            'outputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']}},

        'state': {
            'interval': {
                '_type': 'step',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieInterval',
                'config': {'ktsc': '6e0'},
                'inputs': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'outputs': {
                    'interval': ['event', 'interval']}},

            'event': {
                '_type': 'process',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieEvent',
                'config': {'ktsc': 6e0},
                'inputs': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'outputs': {
                    'mRNA': ['mRNA']},
                'interval': '3.0'}},
            'emitter': emitter_from_wires({
                'mRNA': ['mRNA'],
                'interval': ['event', 'interval']})}

    gillespie = Composite(
        composite_schema,
        core=core)

    updates = gillespie.update({
        'DNA': {
            'A gene': 11.0,
            'B gene': 5.0},
        'mRNA': {
            'A mRNA': 33.3,
            'B mRNA': 2.1}},
        1000.0)

    # TODO: make this work
    results = gather_emitter_results(gillespie)

    assert 'mRNA' in updates[0]
    # TODO: support omit as well as emit
    

def test_run_process(core):
    timestep = 0.1
    runtime = 10.0
    initial_A = 11.11

    state = {
        'species': {
            'A': initial_A},
        'run': {
            '_type': 'step',
            'address': 'local:RunProcess',
            'config': {
                'process_address': 'local:ToySystem',
                'process_config': {
                    'rates': {
                        'A': {
                            'kdeg': 1.1,
                            'ksynth': 0.9}}},
                'observables': [['species']],
                'timestep': timestep,
                'runtime': runtime},
            'inputs': {'species': ['species']},
            'outputs': {'results': ['A_results']}}}

    run = Composite({
        'bridge': {
            'outputs': {
                'results': ['A_results']}},
        'state': state},
        core=core)

    results = run.read_bridge()['results']

    assert results['time'][-1] == runtime
    assert results['species'][0]['A'] == initial_A


def test_nested_wires(core):
    timestep = 0.1
    runtime = 10.0
    initial_A = 11.11

    state = {
        'species': {'A': initial_A},
        'run': {
            '_type': 'step',
            'address': 'local:RunProcess',
            'config': {
                'process_address': 'local:ToySystem',
                'process_config': {
                    'rates': {
                        'A': {
                            'kdeg': 1.1,
                            'ksynth': 0.9}}},
                'observables': [['species', 'A']],
                'timestep': timestep,
                'runtime': runtime},
            # '_outputs': {'results': {'_emit': True}},
            'inputs': {'species': ['species']},
            'outputs': {'results': ['A_results']}}}

    process = Composite({
        'bridge': {
            'outputs': {
                'results': ['A_results']}},
        'state': state},
        core=core)

    results = process.update({}, 0.0)

    assert results[0]['results']['time'][-1] == runtime
    assert results[0]['results']['species']['A'][0] == initial_A


def test_parameter_scan(core):
    # TODO: make a parameter scan with a biosimulator process,
    #   ie - Copasi

    state = {
        'scan': {
            '_type': 'step',
            'address': 'local:ParameterScan',
            'config': {
                'parameter_ranges': [(
                    ['rates', 'A', 'kdeg'], [0.0, 0.1, 1.0, 10.0])],
                'process_address': 'local:ToySystem',
                'process_config': {
                    'rates': {
                        'A': {
                            'ksynth': 1.0}}},
                'observables': [
                    ['species', 'A']],
                'initial_state': {
                    'species': {
                        'A': 13.3333}},
                'timestep': 1.0,
                'runtime': 10},
            'outputs': {
                'results': ['results']}}}

    scan = Composite({
        'bridge': {
            'outputs': {
                'results': ['results']}},
        'state': state},
        core=core)

    # TODO: make a method so we can run it directly, provide some way to get the result out
    # result = scan.update({})
    result = scan.update({}, 0.0)


def test_composite_workflow(core):
    # TODO: Make a workflow with a composite inside
    pass


def test_grow_divide(core):
    initial_mass = 1.0

    grow_divide = grow_divide_agent(
        {'grow': {'rate': 0.03}},
        {},
        # {'mass': initial_mass},
        ['environment', '0'])

    environment = {
        'environment': {
            '0': {
                'mass': initial_mass,
                'grow_divide': grow_divide}}}

    composite = Composite({
        'state': environment,
        'bridge': {
            'inputs': {
                'environment': ['environment']}}},
        core=core)

    updates = composite.update({
        'environment': {
            '0': {
                'mass': 1.1}}},
        100.0)

    # TODO: mass is not synchronized between inside and outside the composite?

    assert '0_0_0_0_1' in composite.state['environment']
    assert composite.state['environment']['0_0_0_0_1']['mass'] == composite.state['environment']['0_0_0_0_1']['grow_divide']['instance'].state['mass']

    # check recursive schema reference
    assert id(composite.composition['environment'] == id(composite.composition['environment']['_value']['grow_divide']['_outputs']['environment']))

    composite.save('test_grow_divide_saved.json')

    c2 = Composite.load(
        'out/test_grow_divide_saved.json',
        core=core)
    
    assert id(composite.composition['environment'] == id(composite.composition['environment']['_value']['grow_divide']['_outputs']['environment']))


def test_gillespie_composite(core):
    composite_schema = {
        'bridge': {
            'inputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']},
            'outputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']}},

        'state': {
            'interval': {
                '_type': 'step',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieInterval',
                'config': {'ktsc': '6e0'},
                'inputs': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'outputs': {
                    'interval': ['event', 'interval']}},

            'event': {
                '_type': 'process',
                'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieEvent',
                'config': {'ktsc': 6e0},
                'inputs': {
                    'DNA': ['DNA'],
                    'mRNA': ['mRNA']},
                'outputs': {
                    'mRNA': ['mRNA']},
                'interval': '3.0'},

            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': {
                        'time': 'float',
                        'mRNA': 'map[float]',
                        'interval': 'interval'}},
                'inputs': {
                    'time': ['global_time'],
                    'mRNA': ['mRNA'],
                    'interval': ['event', 'interval']}}}}

    gillespie = Composite(
        composite_schema,
        core=core)

    updates = gillespie.update({
        'DNA': {
            'A gene': 11.0,
            'B gene': 5.0},
        'mRNA': {
            'A mRNA': 33.3,
            'B mRNA': 2.1}},
        1000.0)

    # TODO: make this work
    results = gather_emitter_results(gillespie)

    assert 'mRNA' in updates[0]


def test_union_tree(core):
    tree_union = core.access('list[string]~tree[list[string]]')
    assert core.check(
        tree_union,
        {'a': ['what', 'is', 'happening']})


def test_merge_schema(core):
    state = {'a': 11.0}
    composite = Composite({
        'state': state}, core=core)

    increase_schema = {
        'increase': {
            '_type': 'process',
            'address': default('string', 'local:!process_bigraph.tests.IncreaseProcess'),
            'config': default('quote', {'rate': 0.0001}),
            'inputs': default('wires', {'level': ['b']}),
            'outputs': default('wires', {'level': ['a']})}}

    composite.merge(
        increase_schema,
        {})

    # composite.merge_schema(
    #     increase_schema)

    assert composite.composition['increase']['_type'] == 'process'
    assert isinstance(composite.state['increase']['instance'], Process)

    state = {
        'x': -3.33,
        'atoms': {
            'A': {
                'lll': 55}}}

    composition = {
        'atoms': 'map[lll:integer]'}

    merge = Composite({
        'composition': composition,
        'state': state}, core=core)

    nested_increase_schema = {
        'increase': {
            '_type': 'process',
            'address': default('string', 'local:!process_bigraph.tests.IncreaseProcess'),
            'config': default('quote', {'rate': 0.0001}),
            'inputs': default('wires', {'level': ['..', '..', 'b']}),
            'outputs': default('wires', {'level': ['..', '..', 'a']})}}

    merge.merge(
        {'atoms': {'_value': nested_increase_schema}},
        {})

    # TODO: do we need merge_schema if merge works for schema and state?
    # merge.merge_schema(
    #     nested_increase_schema,
    #     path=['atoms', '_value'])

    assert isinstance(merge.state['atoms']['A']['increase']['instance'], Process)
    assert merge.composition['atoms']['_value']['increase']['_type'] == 'process'
    assert ('atoms', 'A', 'increase') in merge.process_paths

    merge.merge(
        {},
        {'atoms': {'B': {'lll': 11111}}})

    assert isinstance(merge.state['atoms']['B']['increase']['instance'], Process)
    assert ('atoms', 'B', 'increase') in merge.process_paths


def test_shared_steps(core):
    initial_rate = 0.4

    state = {
        'value': 1.1,
        'increase': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.IncreaseProcess',
            'config': {'rate': initial_rate},
            'inputs': {'level': ['value']},
            'outputs': {'level': ['value']},
            'shared': {
                'accelerate': {
                    'address': 'local:!process_bigraph.tests.IncreaseRate',
                    'config': {'acceleration': '3e-20'},
                    'inputs': {'level': ['..', '..', 'value']}}}},
        'emitter': emitter_from_wires({
            'level': ['value']})}

    shared = Composite(
        {'state': state},
        core=core)

    shared.run(100)

    results = gather_emitter_results(shared)

    assert shared.state['increase']['shared']['accelerate']['instance'].instance.config['rate'] == shared.state['increase']['instance'].config['rate']
    assert shared.state['increase']['instance'].config['rate'] > initial_rate


class WriteCounts(Step):
    def inputs(self):
        return {
            'volumes': 'map[float]',
            'concentrations': 'map[map[float]]'}


    def outputs(self):
        return {
            'counts': 'map[map[integer]]'}


    def update(self, state):
        counts = {}

        for key, local in state['concentrations'].items():
            counts[key] = {}
            for substrate, concentration in local.items():
                count = int(concentration * state['volumes'][key])
                counts[key][substrate] = count

        return {
            'counts': counts}


def test_star_update(core):
    composition = {
        'Compartments': {
            '_type': 'map',
            '_value': {
                'Shared Environment': {
                    'counts': 'map[integer]',
                    'concentrations': 'map[float]',
                    'volume': 'float'},
                'position': 'list[float]'}}}

    state = {
        'write': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.WriteCounts',
            'inputs': {
                'volumes': ['Compartments', '*', 'Shared Environment', 'volume'],
                'concentrations': ['Compartments', '*', 'Shared Environment', 'concentrations']},
            'outputs': {
                'counts': ['Compartments', '*', 'Shared Environment', 'counts']}},

        'Compartments': {
            '0': {
                'Shared Environment': {
                    'concentrations': {
                        'acetate': 1.123976466801866,
                        'biomass': 5.484002382436302,
                        'glucose': 5.054266524967003},
                    'counts': {
                        'acetate': 1.123976466801866,
                        'biomass': 5.484002382436302,
                        'glucose': 5.054266524967003},
                    'volume': 100},
                'position': [0.5, 0.5, 0.0]},
            '1': {
                'Shared Environment': {
                    'concentrations': {
                        'acetate': 1.1582833546687243,
                        'biomass': 5.2088139570269405,
                        'glucose': 2.4652858010098577},
                    'counts': {
                        'acetate': 1.1582833546687243,
                        'biomass': 5.2088139570269405,
                        'glucose': 2.4652858010098577},
                    'volume': 200},
                'position': [0.5, 1.5, 0.0]},
            '2': {
                'Shared Environment': {
                    'concentrations': {
                        'acetate': 2.644399921259828,
                        'biomass': 9.63480818091309,
                        'glucose': 2.375172278348736},
                    'counts': {
                        'acetate': 2.644399921259828,
                        'biomass': 9.63480818091309,
                        'glucose': 2.375172278348736},
                    'volume': 300},
                'position': [0.5, 2.5, 0.0]}}}

    star = Composite({
        'composition': composition,
        'state': state}, core=core)

    assert star.state['Compartments']['2']['Shared Environment']['counts']['biomass'] == 2899


class GlobalProcess(Process):
    config_schema = {}


    def initialize(self, config):
        pass


    


def test_default_process_state(core):
    # provide some initial values
    default_rate = {
        'config': {
            'rate': 0.001}}

    # generate a default state for the Grow process
    default_grow = core.default_state(
        Grow,
        default_rate)

    # create a composite from the default process state
    composite = Composite({
        'state': {
            'grow': default_grow,
            'mass': 1.0}},
        core=core)

    # run the composite
    composite.run(10.0)

    # assert the process ran and the mass increased
    assert composite.state['mass'] > 1.0

    default_divide = core.default_state(
        Divide)

    assert 'interval' not in default_divide


def test_update_removal(core):
    return {}


def test_stochastic_deterministic_composite(core):
    # TODO make the demo for a hybrid stochastic/deterministic simulator
    pass


def test_match_star_path(core):
    assert match_star_path(["first", "list", "test"], ["first", "*", "test"])
    assert not match_star_path(["first", "list", "tent"], ["first", "*", "test"])
    assert match_star_path(["first", "list", "test"], ["first", "list", "test"])


if __name__ == '__main__':
    core = ProcessTypes()
    core = register_types(core)

    test_default_config(core)
    test_default_process_state(core)
    test_merge_collections(core)
    test_process(core)
    test_composite(core)
    test_infer(core)
    test_step_initialization(core)
    test_dependencies(core)
    test_emitter(core)
    test_union_tree(core)

    test_gillespie_composite(core)
    test_run_process(core)
    test_nested_wires(core)
    test_parameter_scan(core)
    test_shared_steps(core)

    test_stochastic_deterministic_composite(core)
    test_merge_schema(core)
    test_grow_divide(core)
    test_star_update(core)
    test_match_star_path(core)
