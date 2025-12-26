"""
=========================
Tests for Process Bigraph
=========================
"""

import sys
import random
import inspect
import numpy as np

import pytest
from urllib.parse import urlparse, urlunparse

from bigraph_schema.schema import Path, make_default
from process_bigraph import allocate_core

from process_bigraph.composite import (
    Process, Step, Composite, merge_collections, match_star_path, as_process, as_step,
)

from process_bigraph.emitter import emitter_from_wires, gather_emitter_results, add_emitter_to_composite
from process_bigraph.protocols.rest import rest_get, rest_post
from process_bigraph.types import ProcessLink, StepLink

from process_bigraph.processes.examples import IncreaseProcess
from process_bigraph.processes.growth_division import grow_divide_agent, Grow, Divide


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
    state = core.fill(interface['inputs'], {})
    state = core.fill(interface['outputs'], state)
    update = process.update({'level': 5.5}, 1.0)

    new_state, merges = core.apply(
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
                'address': 'local:IncreaseProcess',
                'config': {'rate': 0.3},
                'interval': 1.0,
                'inputs': {'level': ['value']},
                'outputs': {'level': ['value']}},
            'value': 11.11}}, core=core)

    initial_state = {'exchange': 3.33}

    updates = composite.update(initial_state, 10.0)

    final_exchange = np.sum([
        update['exchange']
        for update in [initial_state] + updates])

    assert composite.state['value'] > initial_state['exchange']
    assert np.isclose(
        composite.state['value'],
        final_exchange)

    assert 'exchange' in updates[0]


def test_infer(core):
    state = {
        'increase': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': '0.3'},
            'inputs': {'level': ['value']},
            'outputs': {'level': ['value']}},
        'value': '11.11'}

    composite = Composite({
        'state': state}, core=core)

    assert core.render(composite.composition['value']).startswith('float')
    assert composite.state['value'] == 11.11


def test_process_type(core):
    assert type(core.access('process')) == ProcessLink


def test_step_initialization(core):
    steps = {
        'state': {
            'A': 13.0,
            'B': 21.0,
            'step1': {
                '_type': 'step',
                'address': 'local:OperatorStep',
                'config': {
                    'operator': '+'},
                'inputs': {
                    'a': ['A'],
                    'b': ['B']},
                'outputs': {
                    'c': ['C']}},
            'step2': {
                '_type': 'step',
                'address': 'local:OperatorStep',
                'config': {
                    'operator': '*'},
                'inputs': {
                    'a': ['B'],
                    'b': ['C']},
                'outputs': {
                    'c': ['D']}}}}

    composite = Composite(
        steps,
        core=core)

    # composite.run(0.0)
    assert composite.state['D'] == (13.0 + 21.0) * 21.0


def test_dependencies(core):
    operation = {
        'a': 11.111,
        'b': 22.2,
        'c': 555.555,

        '1': {
            '_type': 'step',
            'address': 'local:OperatorStep',
            'config': {
                'operator': '+'},
            'inputs': {
                'a': ['a'],
                'b': ['b']},
            'outputs': {
                'c': ['e']}},
        '2.1': {
            '_type': 'step',
            'address': 'local:OperatorStep',
            'config': {
                'operator': '-'},
            'inputs': {
                'a': ['c'],
                'b': ['e']},
            'outputs': {
                'c': ['f']}},
        '2.2': {
            '_type': 'step',
            'address': 'local:OperatorStep',
            'config': {
                'operator': '-'},
            'inputs': {
                'a': ['d'],
                'b': ['e']},
            'outputs': {
                'c': ['g']}},
        '3': {
            '_type': 'step',
            'address': 'local:OperatorStep',
            'config': {
                'operator': '*'},
            'inputs': {
                'a': ['f'],
                'b': ['g']},
            'outputs': {
                'c': ['h']}},
        '4': {
            '_type': 'step',
            'address': 'local:OperatorStep',
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
                        'address': 'local:SimpleCompartment',
                        'config': {'id': '0'},
                        'concentrations': {},
                        'inner': {
                            'agent2': {
                                '_type': 'process',
                                'address': 'local:SimpleCompartment',
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

    process.update({}, 0.0)

    results = process.read_bridge()

    assert results['results']['time'][-1] == runtime
    assert results['results']['species']['A'][0] == initial_A


def test_parameter_scan(core):
    # TODO: make a parameter scan with a biosimulator process,
    #   ie - Copasi

    ranges = core.access('list[tuple[path,list[float]]]')
    assert isinstance(ranges._element._values[0], Path)

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

    assert len(scan.state['results']) == 4


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
        50.0)

    # TODO: mass is not synchronized between inside and outside the composite?

    assert '0_0_0_0_1' in composite.state['environment']
    assert composite.state['environment']['0_0_0_0_1']['mass'] == composite.state['environment']['0_0_0_0_1']['grow_divide']['instance'].state['mass']

    # # check recursive schema reference
    # assert id(composite.composition['environment']['_value']['grow_divide']['_outputs']['environment']) == id(composite.composition['environment']['_value']['grow_divide']['_outputs']['environment']['_value']['grow_divide']['_outputs']['environment'])

    composite.save('test_grow_divide_saved.json')

    # c2 = Composite.load(
    #     'out/test_grow_divide_saved.json',
    #     core=core)

    # assert c2.state['environment'].keys() == composite.state['environment'].keys()

    # assert id(composite.composition['environment']['_value']['grow_divide']['_outputs']['environment']) == id(composite.composition['environment']['_value']['grow_divide']['_outputs']['environment']['_value']['grow_divide']['_outputs']['environment'])


def test_gillespie_composite(core):
    composite_schema = {
        'bridge': {
            'inputs': {
                'DNA': ['DNA'],
                'mRNA': ['mRNA']},
            'outputs': {
                'time': ['global_time'],
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
                'address': 'local:!process_bigraph.emitter.RAMEmitter',
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
            'A mRNA': 33.0,
            'B mRNA': 2.0}},
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
            '_default': {
                'address': 'local:IncreaseProcess',
                'config': {'rate': 0.0001},
                'inputs': {'level': ['b']},
                'outputs': {'level': ['a']}}}}

    composite.merge(
        increase_schema,
        {})

    assert isinstance(composite.composition['increase'], ProcessLink)
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
            '_default': {
                'address': 'local:IncreaseProcess',
                'config': {'rate': 0.0001},
                'inputs': {'level': ['..', '..', 'b']},
                'outputs': {'level': ['..', '..', 'a']}}}}

    merge.merge(
        {'atoms': {
            '_type': 'map',
            '_value': nested_increase_schema}},
        {})

    assert isinstance(merge.state['atoms']['A']['increase']['instance'], Process)
    assert isinstance(merge.composition['atoms']._value['increase'], ProcessLink)
    assert ('atoms', 'A', 'increase') in merge.process_paths

    merge.merge(
        {},
        {'atoms': {'B': {'lll': 11111}}})

    assert isinstance(merge.state['atoms']['B']['increase']['instance'], Process)
    assert ('atoms', 'B', 'increase') in merge.process_paths


def todo_test_shared_steps(core):
    initial_rate = 0.4

    state = {
        'value': 1.1,
        'increase': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': initial_rate},
            'inputs': {'level': ['value']},
            'outputs': {'level': ['value']},
            'shared': {
                'accelerate': {
                    'address': 'local:IncreaseRate',
                    'config': {'acceleration': '3e-20'},
                    'inputs': {'level': ['..', 'value']}}}},
                    # 'inputs': {'level': ['..', '..', 'value']}}}},
        'emitter': emitter_from_wires({
            'level': ['value']})}

    shared = Composite(
        {'state': state},
        core=core)

    shared.run(100)

    results = gather_emitter_results(shared)

    assert shared.state['increase']['shared']['accelerate']['instance'].instance.config['rate'] == shared.state['increase']['instance'].config['rate']
    assert shared.state['increase']['instance'].config['rate'] > initial_rate


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
            '_type': 'step',
            'address': 'local:WriteCounts',
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

    import ipdb; ipdb.set_trace()

    star = Composite({
        'composition': composition,
        'state': state}, core=core)

    assert star.state['Compartments']['2']['Shared Environment']['counts']['biomass'] == 2899


def test_update_removal(core):
    composition = {
        'environment': {
            '_type': 'map',
            '_value': {
                'below': {
                    '_type': 'process',
                    'address': make_default(
                        'string',
                        'local:BelowProcess'),
                    'config': make_default('node', {
                        'creation_probability': 0.01,
                        'annihilation_probability': 0.007}),
                    'inputs': make_default('wires', {
                        'mass': ['mass'],
                        'entropy': ['entropy']}),
                    'outputs': make_default('wires', {
                        'entropy': ['entropy'],
                        'environment': ['..']}),
                    'interval': make_default('float', 0.4)}}}}

    state = {
        'above': {
            '_type': 'process',
            'address': 'local:AboveProcess',
            'config': {
                'rate': 0.001},
            'inputs': {
                'below': ['environment']},
            'outputs': {
                'below': ['environment']},
            'interval': 3.33},
        'environment': {
            '0': {
                'below': {
                    'config': {
                        'id': '0'}},
                'mass': 1.001,
                'entropy': 0.03}}}

    composite = Composite({
        'composition': composition,
        'state': state}, core=core)

    composite.run(50)


def test_stochastic_deterministic_composite(core):
    # TODO make the demo for a hybrid stochastic/deterministic simulator
    pass


def test_match_star_path(core):
    assert match_star_path(["first", "list", "test"], ["first", "*", "test"])
    assert not match_star_path(["first", "list", "tent"], ["first", "*", "test"])
    assert match_star_path(["first", "list", "test"], ["first", "list", "test"])


def test_function_wrappers(core):
    # --- STEP with core ---
    @as_step(inputs={'a': 'float', 'b': 'float'},
             outputs={'sum': 'float'},
             core=core)
    def update_add(state):
        return {'sum': state['a'] + state['b']}

    step = update_add(config={}, core=core)
    out = step.update({'a': 5, 'b': 7})
    assert out == {'sum': 12}
    assert core.access('add')
    print("Step with core:", out)

    # --- PROCESS with core ---
    @as_process(inputs={'x': 'float'},
                outputs={'x': 'float'},
                core=core)
    def update_decay(state, interval):
        return {'x': state['x'] * (1 - 0.2 * interval)}

    proc = update_decay(config={}, core=core)
    out = proc.update({'x': 50.0}, 1.0)
    assert round(out['x'], 2) == 40.0
    assert core.access('decay')
    print("Process with core:", out)

def test_registered_functions_in_composite(core):
    @as_step(inputs={'a': 'float', 'b': 'float'},
             outputs={'sum': 'float'},
             core=core)
    def update_add(state):
        return {'sum': state['a'] + state['b']}

    @as_process(inputs={'x': 'float'},
                outputs={'x': 'float'},
                core=core)
    def update_decay(state, interval):
        return {'x': state['x'] * (1 - 0.1 * interval)}

    # Define Composite
    state = {
        'adder': {
            '_type': 'process',
            'address': 'local:add',
            'inputs': {
                'a': ['Env', 'a'],
                'b': ['Env', 'b']
            },
            'outputs': {
                'sum': ['Env', 'sum']
            }
        },
        'decayer': {
            '_type': 'process',
            'address': 'local:decay',
            'inputs': {
                'x': ['Env', 'sum']
            },
            'outputs': {
                'x': ['Env', 'x']
            }
        },
        'Env': {
            'a': 3.0,
            'b': 2.0,
            'sum': 0.0,
            'x': 0.0
        }
    }

    # Run Composite
    sim = Composite({
        'state': state
    }, core=core)

    sim.run(1.0)  # One time step

    final = sim.state['Env']
    assert round(final['sum'], 2) == 5.0, f"Adder failed: {final}"
    assert round(final['x'], 2) == 4.5, f"Decay failed: {final}"
    print("✅ test_registered_functions_in_composite passed:", final)


def apply_non_negative(schema, current, update, top_schema, top_state, path, core):
    new_value = current + update
    return max(0, new_value)


def apply_non_negative_array(schema, current, update, top_schema, top_state, path, core):
    def recursive_update(result_array, current_array, update_dict, index_path=()):
        if isinstance(update_dict, dict):
            for key, val in update_dict.items():
                recursive_update(result_array, current_array, val, index_path + (key,))
        else:
            if isinstance(current_array, np.ndarray):
                current_value = current_array[index_path]
                result_array[index_path] = np.maximum(0, current_value + update_dict)
            else:
                # Scalar fallback
                return np.maximum(0, current_array + update_dict)

    if not isinstance(current, np.ndarray):
        if isinstance(update, dict):
            raise ValueError("Cannot apply dict update to scalar current")
        return np.maximum(0, current + update)

    result = np.copy(current)
    recursive_update(result, current, update)
    return result


def todo_test_dfba_process(core):
    base_url = urlparse('http://localhost:22222')
    types_url = base_url._replace(path='/list-types')
    types = rest_get(types_url)

    processes_url = base_url._replace(path='/list-processes')
    processes = rest_get(processes_url)

    # # TODO: import types from the server
    # core.register('positive_float', {
    #     '_inherit': 'float',
    #     '_apply': apply_non_negative})

    # core.register('positive_array', {
    #     '_inherit': 'array',
    #     '_apply': apply_non_negative_array})

    # core.register('bounds', {
    #     'lower': 'maybe[float]',
    #     'upper': 'maybe[float]'})

    dfba_name = 'spatio_flux.processes.DynamicFBA'

    schema_url = base_url._replace(
        path=f'/process/{dfba_name}/config-schema')
    dfba_config_schema = rest_get(schema_url)

    dfba_config = {
        'model_file': 'textbook',
        'substrate_update_reactions': {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e'},
        'kinetic_params': {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2)},
        'bounds': {
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}}}

    # assert core.check(
    #     dfba_config_schema,
    #     dfba_config)

    biomass_id = 'biomass'
    substrates = dfba_config['substrate_update_reactions'].keys()

    initial_biomass = 0.1
    initial_fields = {
        'glucose': 2,
        'acetate': 0,
        biomass_id: initial_biomass}

    for substrate in substrates:
        if substrate not in initial_fields:
            initial_fields[substrate] = 10.0

    path = ['fields']

    state = {
        'fields': initial_fields,
        'rest-dfba': {
            '_type': 'process',
            'address': {
                'protocol': 'rest',
                'data': {
                    'process': dfba_name,
                    'host': 'localhost',
                    'port': 22222}},
            'config': dfba_config,
            'inputs': {
                'substrates': {
                    substrate: path + [substrate]
                    for substrate in substrates},
                'biomass': path + [biomass_id]},
            'outputs': {
                'substrates': {
                    substrate: path + [substrate]
                    for substrate in substrates},
                'biomass': path + [biomass_id]},
            'interval': 0.7}}

    composite = Composite({
        'state': state}, core=core)

    composite.run(11.111)

    assert composite.state['fields'][biomass_id] > initial_biomass


def test_rest_process(core):
    state = {
        'mass': 1.0,
        'rest-process': {
            '_type': 'process',
            'address': {
                'protocol': 'rest',
                'data': {
                    'process': 'Grow',
                    'host': 'localhost',
                    'port': 22222}},
            'config': {
                'rate': 0.005},
            'inputs': {
                'mass': ['mass']},
            'outputs': {
                'mass': ['mass']},
            'interval': 0.7}}

    composite = Composite({
        'state': state}, core=core)

    composite.run(11.111)

    assert composite.state['mass'] > 1.0


def test_ram_emitter(core):
    composite_spec = {
        'increase': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.3},
            'inputs': {'level': ['valueA']},
            'outputs': {'level': ['valueA']}},
        'increase2': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.1},
            'inputs': {'level': ['valueB']},
            'outputs': {'level': ['valueB']}},
        'emitter': emitter_from_wires({
            'time': ['global_time'],
            'valueA': ['valueA'],
            'valueB': ['valueB']})}

    composite = Composite({'state': composite_spec}, core=core)
    composite.run(10)

    results = composite.state['emitter']['instance'].query()
    assert len(results) == 11
    assert results[-1]['time'] == 10
    assert 'valueA' in results[0] and 'valueB' in results[0]

    composite_spec['emitter'] = emitter_from_wires({
        'time': ['global_time'],
        'valueA': ['valueA']})
    composite2 = Composite({'state': composite_spec}, core=core)
    composite2.run(10)

    results2 = composite2.state['emitter']['instance'].query()
    assert 'valueA' in results2[0] and 'valueB' not in results2[0]
    print(results2)

def test_json_emitter(core):
    composite_spec = {
        'increase': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.3},
            'interval': 1.0,
            'inputs': {'level': ['value']},
            'outputs': {'level': ['value']}}}
    composite = Composite({'state': composite_spec}, core)
    composite = add_emitter_to_composite(composite, core, emitter_mode='all', address='local:JSONEmitter')
    composite.run(10)

    results = composite.state['emitter']['instance'].query()
    assert len(results) == 10
    assert results[-1]['global_time'] == 10
    print(results)



def make_test_core():
    members = dict(inspect.getmembers(sys.modules[__name__]))
    return allocate_core(
        top=members)


@pytest.fixture
def core():
    return make_test_core()


if __name__ == '__main__':
    core = make_test_core()

    test_default_config(core)
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
    # test_shared_steps(core)

    test_ram_emitter(core)
    test_json_emitter(core)

    test_stochastic_deterministic_composite(core)
    test_merge_schema(core)
    test_grow_divide(core)
    test_star_update(core)
    test_match_star_path(core)
    test_function_wrappers(core)
    test_registered_functions_in_composite(core)
    test_update_removal(core)

    test_rest_process(core)
    # test_dfba_process(core)



    
