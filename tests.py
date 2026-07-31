"""
=========================
Tests for Process Bigraph
=========================
"""

import os
import sys
import random
import inspect
import sqlite3
import tempfile

import numpy as np
import pytest
from urllib.parse import urlparse, urlunparse

from bigraph_schema.schema import Path, make_default
from bigraph_schema import set_path as bs_set_path

from process_bigraph import allocate_core
from process_bigraph.composite import (
    Process, Step, Composite, merge_collections, match_star_path, as_process, as_step,
)
from process_bigraph.emitter import (
    Emitter, EmitterResults, RAMEmitter, emitter_from_wires,
    gather_emitter_results, add_emitter_to_composite,
)

# SQLiteEmitter + friends now live in the focused pbg-emitters library and
# are re-exported from process_bigraph.emitter only when that package is
# installed. Guard the import here so the rest of the test module loads
# cleanly without the optional dep; individual SQLite tests call
# ``pytest.importorskip("pbg_emitters")`` so they skip (rather than fail)
# when pbg-emitters is absent.
try:
    from process_bigraph.emitter import (
        SQLiteEmitter,
        save_simulation_metadata, mark_simulation_finished,
        list_simulations, load_history, load_simulation_metadata,
    )
except ImportError:
    SQLiteEmitter = None
    save_simulation_metadata = mark_simulation_finished = None
    list_simulations = load_history = load_simulation_metadata = None
from process_bigraph.protocols.rest import rest_get, rest_post
from process_bigraph.types import ProcessLink, StepLink

from process_bigraph.processes.examples import IncreaseProcess
from process_bigraph.processes.growth_division import grow_divide_agent, Grow, Divide
from process_bigraph.processes.dynamic_structure import DynamicWorker


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
        'schema': {
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

    assert core.render(composite.schema['value']).startswith('float')
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

    steps['run_steps_on_init'] = True

    composite = Composite(
        steps,
        core=core)
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
    """Test that cross-step dependencies create proper ordering.

    step_a writes to 'y', step_b reads 'y' and writes 'z',
    step_c reads 'z'. This creates a chain: a → b → c.
    """

    execution_log = []

    class ProducerStep(Step):
        config_schema = {'name': 'string'}
        def inputs(self):
            return {'in_val': 'float'}
        def outputs(self):
            return {'out_val': 'float'}
        def update(self, state):
            execution_log.append(self.config['name'])
            return {'out_val': state.get('in_val', 0.0) + 1.0}

    core = allocate_core()
    core.register_link('ProducerStep', ProducerStep)

    composite = Composite({
        'state': {
            'x': 1.0,
            'y': 0.0,
            'z': 0.0,
            'step_a': {
                '_type': 'step',
                'address': 'local:ProducerStep',
                'config': {'name': 'a'},
                'inputs': {'in_val': ['x']},
                'outputs': {'out_val': ['y']},
            },
            'step_b': {
                '_type': 'step',
                'address': 'local:ProducerStep',
                'config': {'name': 'b'},
                'inputs': {'in_val': ['y']},
                'outputs': {'out_val': ['z']},
            },
            'step_c': {
                '_type': 'step',
                'address': 'local:ProducerStep',
                'config': {'name': 'c'},
                'inputs': {'in_val': ['z']},
                'outputs': {'out_val': ['w']},
            },
        }
    }, core=core)

    composite.run(0.0)

    assert execution_log == ['a', 'b', 'c'], (
        f"Expected chain ordering ['a', 'b', 'c'] "
        f"but got {execution_log}."
    )
    # a reads x=1 → writes y=2
    # b reads y=2 → writes z=3
    # c reads z=3 → writes w=4
    assert composite.state['w'] == 4.0


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

    run.run(0.0)
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
            'inputs': {'species': ['species']},
            'outputs': {'results': ['A_results']}}}

    bridge = {
        'outputs': {
            'results': ['A_results']}}

    composition = {
        'bridge': bridge,
        'state': state}

    process = Composite(
        composition,
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

    scan.run(0.0)
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
    # assert id(composite.schema['environment']['_value']['grow_divide']['_outputs']['environment']) == id(composite.schema['environment']['_value']['grow_divide']['_outputs']['environment']['_value']['grow_divide']['_outputs']['environment'])

    composite.save('test_grow_divide_saved.json')

    # c2 = Composite.load(
    #     'out/test_grow_divide_saved.json',
    #     core=core)

    # assert c2.state['environment'].keys() == composite.state['environment'].keys()

    # assert id(composite.schema['environment']['_value']['grow_divide']['_outputs']['environment']) == id(composite.schema['environment']['_value']['grow_divide']['_outputs']['environment']['_value']['grow_divide']['_outputs']['environment'])


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

    assert isinstance(composite.schema['increase'], ProcessLink)
    assert isinstance(composite.state['increase']['instance'], Process)

    state = {
        'x': -3.33,
        'atoms': {
            'A': {
                'lll': 55}}}

    schema = {
        'atoms': 'map[lll:integer]'}

    merge = Composite({
        'schema': schema,
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
    assert isinstance(merge.schema['atoms']._value['increase'], ProcessLink)
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
    schema = {
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

    star = Composite({
        'schema': schema,
        'state': state}, core=core)

    star.run(0.0)
    assert star.state['Compartments']['2']['Shared Environment']['counts']['biomass'] == 2890


def test_update_removal(core):
    schema = {
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
        'schema': schema,
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
             name='add',  # optional but makes intent explicit
             aliases=['add'],  # optional
             )
    def update_add(state):
        return {'sum': state['a'] + state['b']}

    step = update_add(config={}, core=core)
    out = step.update({'a': 5, 'b': 7})

    core.register_link('add', update_add)

    assert out == {'sum': 12}
    assert core.access('add')
    print("Step with core:", out)

    # --- PROCESS with core ---
    @as_process(inputs={'x': 'float'},
                outputs={'x': 'float'},
                name='decay',
                aliases=['decay'],
                )
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
             name='add',  # optional but makes intent explicit
             aliases=['add'],  # optional
             )
    def update_add(state):
        return {'sum': state['a'] + state['b']}

    @as_process(inputs={'x': 'float'},
                outputs={'x': 'float'},
                name='decay',
                aliases=['decay'],
                )
    def update_decay(state, interval):
        return {'x': state['x'] * (1 - 0.1 * interval)}

    core.register_link('add', update_add)
    core.register_link('decay', update_decay)

    # Define Composite
    state = {
        'adder': {
            '_type': 'step',
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

def test_sqlite_emitter(core, tmp_path=None):
    pytest.importorskip('pbg_emitters')
    tmp_dir = tmp_path or tempfile.mkdtemp(prefix='sqlite_emitter_')
    composite_spec = {
        'increase': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.3},
            'interval': 1.0,
            'inputs': {'level': ['value']},
            'outputs': {'level': ['value']}}}
    composite = Composite({'state': composite_spec}, core)

    emitter_spec = {
        '_type': 'step',
        'address': 'local:SQLiteEmitter',
        'config': {
            'emit': {'global_time': 'node', 'value': 'node'},
            'file_path': str(tmp_dir),
            'db_file': 'test_history.db',
        },
        'inputs': {'global_time': ['global_time'], 'value': ['value']},
    }
    composite.merge({}, bs_set_path({}, ('emitter',), emitter_spec))
    _, instance = core.traverse(composite.schema, composite.state, ('emitter',))
    composite.step_paths[('emitter',)] = instance
    composite.build_step_network()

    composite.run(10)

    results = composite.state['emitter']['instance'].query()
    assert len(results) >= 10
    assert results[-1]['global_time'] == 10
    assert 'value' in results[-1]

    # verify the db file exists and survives re-opening
    db_path = os.path.join(str(tmp_dir), 'test_history.db')
    assert os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    (count,) = conn.execute('SELECT COUNT(*) FROM history').fetchone()
    assert count >= 10
    conn.close()


def test_sqlite_emitter_retrieval_helpers(core):
    '''The standalone helpers must let callers inspect and load a run
    without touching a Composite or a core — this is the main post-hoc
    analysis use case.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_retrieval_')
    db_path = os.path.join(tmp, 'history.db')

    # Two runs sharing the same db file, plus metadata for one.
    e1 = SQLiteEmitter({
        'emit': {'global_time': 'node'},
        'file_path': tmp, 'simulation_id': 'run-A', 'name': 'alpha',
    }, core=core)
    for i in range(4):
        e1.update({'global_time': float(i)})
    save_simulation_metadata(
        db_path, 'run-A',
        composite_config={'cells': {}},
        metadata={'experiment': 'alpha', 'notes': 'first run'},
    )
    mark_simulation_finished(db_path, 'run-A', elapsed_seconds=12.5)
    e1.close()

    e2 = SQLiteEmitter({
        'emit': {'global_time': 'node'},
        'file_path': tmp, 'simulation_id': 'run-B',
    }, core=core)
    for i in range(2):
        e2.update({'global_time': float(i)})
    e2.close()

    # list_simulations: ordered newest-first, exposes completion fields.
    sims = {s['simulation_id']: s for s in list_simulations(db_path)}
    assert set(sims) == {'run-A', 'run-B'}
    assert sims['run-A']['step_count'] == 4
    assert sims['run-A']['elapsed_seconds'] == 12.5
    assert sims['run-A']['completed_at'] is not None
    assert sims['run-A']['has_config'] is True
    # run-B had no metadata row written, so it's reported without config info.
    assert sims['run-B']['step_count'] == 2
    assert sims['run-B']['has_config'] is False
    assert sims['run-B']['elapsed_seconds'] is None

    # load_history: same shape as emitter.query(), unfiltered and filtered.
    history = load_history(db_path, 'run-A')
    assert len(history) == 4
    assert history[-1]['global_time'] == 3.0
    filtered = load_history(db_path, 'run-A', paths=[['global_time']])
    assert filtered == [{'global_time': t} for t in (0.0, 1.0, 2.0, 3.0)]

    # load_simulation_metadata: round-trips the composite config + metadata.
    meta = load_simulation_metadata(db_path, 'run-A')
    assert meta['name'] == 'alpha'
    assert meta['composite_config'] == {'cells': {}}
    assert meta['metadata']['experiment'] == 'alpha'
    assert meta['elapsed_seconds'] == 12.5
    assert load_simulation_metadata(db_path, 'nope') is None


def test_sqlite_emitter_query_paths_kwarg(core):
    '''``query()`` should accept the new ``paths`` kwarg and still accept
    the legacy ``query`` kwarg for back-compat.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_paths_kwarg_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'a': 'node', 'b': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
    }, core=core)
    for i in range(3):
        e.update({'global_time': float(i), 'a': i, 'b': i * 2})

    # Unfiltered.
    assert len(e.query()) == 3

    # New kwarg.
    assert e.query(paths=[['a']]) == [{'a': 0}, {'a': 1}, {'a': 2}]

    # Legacy kwarg still honored.
    assert e.query(query=[['a']]) == [{'a': 0}, {'a': 1}, {'a': 2}]

    # When both are given, ``paths`` wins.
    assert e.query(paths=[['a']], query=[['b']]) == [{'a': 0}, {'a': 1}, {'a': 2}]

    e.close()


def test_sqlite_emitter_subsample(core):
    '''subsample=N writes every Nth composite tick (first tick always
    kept) and preserves the original step number in the stored row.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_subsample_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'v': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
        'subsample': 5,
    }, core=core)

    # Feed 20 ticks — ticks at step 0, 5, 10, 15 should land in the db.
    for i in range(20):
        e.update({'global_time': float(i), 'v': i})
    e.close()

    history = load_history(os.path.join(tmp, 'history.db'), 'sim')
    assert len(history) == 4
    assert [row['v'] for row in history] == [0, 5, 10, 15]
    assert [row['global_time'] for row in history] == [0.0, 5.0, 10.0, 15.0]

    # The recorded `step` column is the real composite tick, not a
    # collapsed index — verify via a direct SQL read.
    conn = sqlite3.connect(os.path.join(tmp, 'history.db'))
    try:
        steps = [r[0] for r in conn.execute(
            'SELECT step FROM history WHERE simulation_id = ? ORDER BY step',
            ('sim',),
        ).fetchall()]
    finally:
        conn.close()
    assert steps == [0, 5, 10, 15]


def test_sqlite_emitter_subsample_rejects_bad_value(core):
    '''subsample < 1 is nonsensical — refuse at construction time.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_subsample_bad_')
    with pytest.raises(ValueError):
        SQLiteEmitter({
            'emit': {},
            'file_path': tmp, 'simulation_id': 'sim',
            'subsample': 0,
        }, core=core)


def test_sqlite_emitter_batch_size(core):
    '''batch_size buffers up to N rows and flushes them in one transaction.
    Close and query must flush pending rows so no data is lost.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_batch_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'v': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
        'batch_size': 5,
    }, core=core)

    db_path = os.path.join(tmp, 'history.db')
    conn = sqlite3.connect(db_path)
    try:
        # Feed 3 rows — below the batch threshold, nothing on disk yet.
        for i in range(3):
            e.update({'global_time': float(i), 'v': i})
        (pending,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert pending == 0, f'expected 0 rows before flush, got {pending}'

        # query() forces a flush so the reader sees a consistent view.
        assert len(e.query()) == 3
        (after_query,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert after_query == 3

        # Cross a batch boundary: 3 already flushed by query(), feed 5 more
        # — the 5th triggers auto-flush.
        for i in range(3, 8):
            e.update({'global_time': float(i), 'v': i})
        (after_boundary,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert after_boundary == 8

        # Remaining buffered rows flush on close().
        for i in range(8, 10):
            e.update({'global_time': float(i), 'v': i})
        e.close()
        (final,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert final == 10
    finally:
        conn.close()

    # Final sanity check that the data is intact and in order.
    history = load_history(db_path, 'sim')
    assert [row['v'] for row in history] == list(range(10))


def test_sqlite_emitter_batch_size_rejects_bad_value(core):
    '''batch_size < 1 makes no sense — refuse at construction.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_batch_bad_')
    with pytest.raises(ValueError):
        SQLiteEmitter({
            'emit': {},
            'file_path': tmp, 'simulation_id': 'sim',
            'batch_size': 0,
        }, core=core)


def test_sqlite_emitter_close(core):
    '''close() should release the connection deterministically, make further
    updates fail loudly, and leave the db usable from a fresh connection.'''
    pytest.importorskip('pbg_emitters')
    tmp = tempfile.mkdtemp(prefix='sqlite_close_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
    }, core=core)
    e.update({'global_time': 0.0})
    e.close()

    # Idempotent: calling close again is a no-op.
    e.close()

    # New writes must fail loudly rather than silently dropping data.
    with pytest.raises(RuntimeError):
        e.update({'global_time': 1.0})

    # The file is intact and readable via a fresh connection.
    db_path = os.path.join(tmp, 'history.db')
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute('SELECT COUNT(*) FROM history').fetchone()
        assert count == 1
    finally:
        conn.close()


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
    assert len(results) >= 10
    assert results[-1]['global_time'] == 10
    print(results)



def _pool_agents(state):
    """Return list of agent IDs in the pool (entries with a 'value' key)."""
    pool = state.get('pool', {})
    return [
        k for k, v in pool.items()
        if isinstance(v, dict) and 'value' in v]


def _make_worker_state(process_id, **config_overrides):
    """Build a full worker state dict with address, wires, and config."""
    config = {
        'process_id': process_id,
        'growth_rate': 1.0,
        'spawn_growth_rate': 0.8,
        'propensity_spawn': 1.0,
        'propensity_remove': 1.0,
        'propensity_rewire': 0.0,
        'threshold_spawn': 3.0,
        'threshold_remove': -3.0,
        'threshold_rewire': 4.0,
        'max_pool_size': 15,
        'spawn_value': 0.5,
    }
    config.update(config_overrides)
    return {
        'address': 'local:DynamicWorker',
        'config': config,
        'inputs': {
            'sources': ['..'],
            'self_value': ['value']},
        'outputs': {
            'targets': ['..'],
            'self_value': ['value']},
        'interval': 1.0}


def test_dynamic_structure(core):
    """Test dynamic structure changes: spawn, remove, rewire, nesting,
    and verify compiled link cache invalidation throughout."""

    DynamicWorker._counter = 0

    schema = {
        'pool': {
            '_type': 'map',
            '_value': {
                'value': 'float',
                'worker': {
                    '_type': 'process',
                    'address': make_default('string', 'local:DynamicWorker'),
                    'inputs': make_default('wires', {
                        'sources': ['..'],
                        'self_value': ['value']}),
                    'outputs': make_default('wires', {
                        'targets': ['..'],
                        'self_value': ['value']}),
                    'interval': make_default('float', 1.0)}}}}

    # Start with 3 agents, each reading the entire pool as sources.
    # Address and wires provided explicitly since map _value defaults
    # don't propagate to initial state during realization.
    state = {
        'pool': {
            'a0': {
                'value': 1.0,
                'worker': _make_worker_state('a0', propensity_rewire=1.0)},
            'a1': {
                'value': 1.0,
                'worker': _make_worker_state('a1', propensity_rewire=1.0)},
            'a2': {
                'value': 1.0,
                'worker': _make_worker_state('a2', propensity_rewire=1.0)}}}

    composite = Composite({
        'schema': schema,
        'state': state}, core=core)

    # -- Verify initial state --
    agents = _pool_agents(composite.state)
    assert len(agents) == 3, f"Expected 3 initial agents, got {len(agents)}"

    # Verify initial cache was built
    assert len(composite._compiled_links) > 0, "Initial cache should be populated"
    initial_cache_size = len(composite._compiled_links)

    # -- Phase 1: Growth + rewiring (t=0 to t=10) --
    # Initial agents accumulate value, rewire when source_sum > 4.0,
    # then spawn after rewire disables further rewiring.
    composite.run(10.0)

    agents_after_growth = _pool_agents(composite.state)
    assert len(agents_after_growth) > 3, \
        f"Pool should have grown beyond 3, got {len(agents_after_growth)}"

    # Cache should have been rebuilt to include new process paths
    assert len(composite._compiled_links) > initial_cache_size, \
        "Cache should have grown with new processes"

    # Verify all current process paths have valid compiled links
    for path in composite.process_paths:
        compiled = composite._compiled_links.get(path)
        if compiled is not None:
            core_cached = core.get_compiled_link(path)
            assert core_cached is not None, \
                f"Core cache missing for active process at {path}"

    peak_count = len(agents_after_growth)

    # Cache entry count should match process count
    assert len(composite._compiled_links) == len(composite.process_paths), \
        "Compiled links count should match process paths count"

    # -- Phase 2: Continued growth then shrinkage (t=10 to t=25) --
    # Gen-1 agents spawn gen-2 agents with negative growth.
    # Gen-2 agents accumulate negative value and self-remove.
    composite.run(15.0)

    agents_final = _pool_agents(composite.state)

    # Pool should still have agents but count should have changed
    assert len(agents_final) > 0, "Pool should not be empty"
    assert len(agents_final) != peak_count, \
        f"Pool size should have changed from peak of {peak_count}"

    # Verify remaining agents have valid values (above remove threshold)
    for aid in agents_final:
        agent = composite.state['pool'][aid]
        if isinstance(agent, dict) and 'value' in agent:
            val = agent['value']
            assert val >= -3.0, \
                f"Surviving agent {aid} has value {val} below remove threshold"

    # Verify cache consistency: all process paths have compiled links,
    # and Core cache agrees with Composite cache
    assert len(composite._compiled_links) == len(composite.process_paths), \
        "Final compiled links count should match process paths count"
    for path in composite.process_paths:
        compiled = composite._compiled_links.get(path)
        if compiled is not None:
            core_cached = core.get_compiled_link(path)
            assert core_cached is not None, \
                f"Core cache inconsistent for {path} after structural changes"


class _ArrWriterProcess(Process):
    """Writes an Array-typed output. Used by
    ``test_port_outputs_propagate_to_store_schema``."""

    def inputs(self):
        return {'tick': 'float'}

    def outputs(self):
        return {'arr': 'array[float[64]]'}

    def update(self, state, interval=None):
        import numpy as np
        return {'arr': np.array([1.0, 2.0, 3.0])}


def test_reaction_step(core):
    """ReactionStep fires Milner-style reaction rules on a state subtree.

    The step is standalone (not wired through a Composite) — it reads
    a 'state' input port and returns the modified subtree.
    """
    from bigraph_schema.schema import Site
    from bigraph_schema.assembly import ReactionRule
    from process_bigraph.processes.reaction import ReactionStep

    # B3: agent enters room
    b3 = ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        label='B3')

    step = ReactionStep(
        config={'rules': [b3], 'mode': 'deterministic'},
        core=core)

    # Alice's `mass` and `height` fields are captured by the redex
    # Site `props`. Per Milner's bigraph semantics, a site is a hole
    # in the place graph; when filled, its captured region's
    # children become siblings at the slot position with their
    # original identifiers preserved (rather than nesting under the
    # site name `props`).
    state = {
        'state': {
            'bldg': {
                '_control': 'building',
                'alice': {'_control': 'agent', 'mass': 70.0, 'height': 1.7},
                'lab': {
                    '_control': 'room',
                    'pc': {'_control': 'computer', 'cpu': 3.0}}}}}

    update = step.update(state)
    assert update, 'ReactionStep should have produced an update'
    assert 'state' in update

    bldg = update['state']['bldg']
    assert 'alice' not in bldg, 'alice should no longer be a sibling'
    lab = bldg['lab']
    assert 'alice' in lab, 'alice should be inside lab'
    # mass and height splice into alice as direct fields
    assert lab['alice']['mass'] == 70.0
    assert lab['alice']['height'] == 1.7
    # the room's pc spliced too, with its original key preserved
    assert 'pc' in lab and lab['pc']['_control'] == 'computer'

    # Stochastic mode
    step_stoch = ReactionStep(
        config={'rules': [b3], 'mode': 'stochastic', 'seed': 42},
        core=core)
    update2 = step_stoch.update(state)
    assert update2 and 'state' in update2


def test_reaction_step_in_composite(core):
    """ReactionStep wired through a Composite. The step fires on each
    tick, moving one agent into a room per tick."""
    from bigraph_schema.schema import Site
    from bigraph_schema.assembly import ReactionRule

    b3 = ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        label='B3')

    spec = {
        'schema': {'building': 'tree[node]'},
        'state': {
            'building': {
                '_control': 'building',
                'alice': {'_control': 'agent', 'mass': 70.0, 'height': 1.7},
                'lab': {'_control': 'room',
                        'pc': {'_control': 'computer'}}},
            'reactions': {
                '_type': 'step',
                'address': (
                    'local:!process_bigraph.processes.reaction'
                    '.ReactionStep'),
                'config': {'rules': [b3]},
                'inputs': {'state': ['building']},
                'outputs': {'state': ['building']}}}}

    composite = Composite(spec, core=core)
    building_before = composite.state['building']
    assert 'alice' in building_before

    composite.run(0)

    building_after = composite.state['building']
    # alice should have moved inside the room
    assert 'alice' not in building_after
    lab = building_after['lab']
    assert 'alice' in lab
    # mass/height splice as direct alice fields (Milner site semantics)
    assert lab['alice']['mass'] == 70.0
    assert lab['alice']['height'] == 1.7


def _enter_room_rule(rate=None):
    """A B3-style rule: any free agent enters any room. Reused
    across the BigraphicalReactiveSystem tests."""
    from bigraph_schema.schema import Site
    from bigraph_schema.assembly import ReactionRule

    return ReactionRule(
        redex={
            'a': {'_control': 'agent', 'props': Site()},
            'r': {'_control': 'room', 'contents': Site()}},
        reactum={
            'r': {'_control': 'room',
                  'contents': Site(),
                  'a': {'_control': 'agent', 'props': Site()}}},
        instantiation={'props': 'props', 'contents': 'contents'},
        rate=rate,
        label='enter_room')


def _building_with_agents(n_agents):
    """Building containing `n_agents` free agents and one room.
    Each agent has a distinct identifier so firings consume them
    one at a time."""
    bldg = {'_control': 'building',
            'lab': {'_control': 'room',
                    'pc': {'_control': 'computer'}}}
    for i in range(n_agents):
        bldg[f'agent_{i}'] = {
            '_control': 'agent', 'mass': float(i + 1)}
    return bldg


def test_bigraphical_reactive_system_deterministic(core):
    """BigraphicalReactiveSystem in deterministic mode fires one
    rule match per tick (max_per_tick default = 1) and returns the
    rewritten subtree. Unlike ReactionStep, it's an interval-driven
    Process: `update` takes an explicit `interval` argument."""
    from process_bigraph.processes.bigraphical_reactive_system import (
        BigraphicalReactiveSystem)

    rule = _enter_room_rule()
    proc = BigraphicalReactiveSystem(
        config={'rules': [rule], 'mode': 'deterministic'},
        core=core)

    state = {'state': {'bldg': _building_with_agents(3)}}
    update = proc.update(state, interval=1.0)

    assert update and 'state' in update
    bldg = update['state']['bldg']
    lab = bldg['lab']
    # Exactly one agent should have moved into the lab.
    moved = [k for k in lab if k.startswith('agent_')]
    remaining = [k for k in bldg if k.startswith('agent_')]
    assert len(moved) == 1
    assert len(remaining) == 2
    # And the engine recorded one firing.
    assert len(proc.fired_log) == 1
    assert proc.fired_log[0][1] == 'enter_room'


def test_bigraphical_reactive_system_no_match_returns_empty(core):
    """When no rule matches, the engine returns an empty update —
    no `state` key — so the composite doesn't overwrite the store
    with the unchanged subtree."""
    from process_bigraph.processes.bigraphical_reactive_system import (
        BigraphicalReactiveSystem)

    rule = _enter_room_rule()
    proc = BigraphicalReactiveSystem(
        config={'rules': [rule], 'mode': 'deterministic'},
        core=core)

    # A building with no agents — the redex requires both an agent
    # and a room, so nothing matches.
    state = {'state': {'bldg': {
        '_control': 'building',
        'lab': {'_control': 'room',
                'pc': {'_control': 'computer'}}}}}
    update = proc.update(state, interval=1.0)
    assert update == {}
    assert proc.fired_log == []


def test_bigraphical_reactive_system_max_per_tick(core):
    """`max_per_tick > 1` in non-Gillespie mode lets the engine
    fire multiple rule matches per tick in a single deterministic
    sweep, capped by the configured value."""
    from process_bigraph.processes.bigraphical_reactive_system import (
        BigraphicalReactiveSystem)

    rule = _enter_room_rule()
    proc = BigraphicalReactiveSystem(
        config={
            'rules': [rule],
            'mode': 'deterministic',
            'max_per_tick': 2},
        core=core)

    state = {'state': {'bldg': _building_with_agents(5)}}
    update = proc.update(state, interval=1.0)

    bldg = update['state']['bldg']
    moved = [k for k in bldg['lab'] if k.startswith('agent_')]
    remaining = [k for k in bldg if k.startswith('agent_')]
    assert len(moved) == 2
    assert len(remaining) == 3
    assert len(proc.fired_log) == 2


def test_bigraphical_reactive_system_stochastic(core):
    """Stochastic mode picks one match per tick weighted by rate.
    With a single rule and multiple matches, this is uniform over
    matches — but the path through `_pick_candidate` differs from
    deterministic and uses the seeded RNG."""
    from process_bigraph.processes.bigraphical_reactive_system import (
        BigraphicalReactiveSystem)

    rule = _enter_room_rule(rate=1.0)
    proc = BigraphicalReactiveSystem(
        config={
            'rules': [rule],
            'mode': 'stochastic',
            'seed': 42},
        core=core)

    state = {'state': {'bldg': _building_with_agents(4)}}
    update = proc.update(state, interval=1.0)

    bldg = update['state']['bldg']
    moved = [k for k in bldg['lab'] if k.startswith('agent_')]
    remaining = [k for k in bldg if k.startswith('agent_')]
    assert len(moved) == 1
    assert len(remaining) == 3
    assert len(proc.fired_log) == 1

    # Re-seeded runs are reproducible.
    proc2 = BigraphicalReactiveSystem(
        config={'rules': [rule], 'mode': 'stochastic', 'seed': 42},
        core=core)
    update2 = proc2.update(state, interval=1.0)
    moved2 = [
        k for k in update2['state']['bldg']['lab']
        if k.startswith('agent_')]
    assert moved == moved2


def test_bigraphical_reactive_system_gillespie(core):
    """Gillespie SSA τ-leap mode fires zero or more rules per
    tick, sampling exponential waits with parameter
    ``λ = Σ k_i·|m_i|`` until ``t >= interval``.

    With a high rate and a long interval, the engine should
    exhaust all matches in a single tick: every agent ends up in
    the room."""
    from process_bigraph.processes.bigraphical_reactive_system import (
        BigraphicalReactiveSystem)

    rule = _enter_room_rule(rate=100.0)
    proc = BigraphicalReactiveSystem(
        config={
            'rules': [rule],
            'mode': 'gillespie',
            'seed': 7},
        core=core)

    state = {'state': {'bldg': _building_with_agents(5)}}
    update = proc.update(state, interval=10.0)

    bldg = update['state']['bldg']
    moved = [k for k in bldg['lab'] if k.startswith('agent_')]
    remaining = [k for k in bldg if k.startswith('agent_')]
    assert len(moved) == 5
    assert len(remaining) == 0
    assert len(proc.fired_log) == 5
    # Gillespie writes the per-firing simulation time, not the
    # tick interval. Times should be monotonic non-decreasing and
    # not exceed `interval`.
    times = [t for t, _, _ in proc.fired_log]
    assert all(t <= 10.0 for t in times)
    assert times == sorted(times)


def test_bigraphical_reactive_system_gillespie_max_per_tick(core):
    """`max_per_tick` caps Gillespie firings even when the
    interval would otherwise admit more. With a high rate, a
    long interval, and a cap of 2, only 2 firings happen."""
    from process_bigraph.processes.bigraphical_reactive_system import (
        BigraphicalReactiveSystem)

    rule = _enter_room_rule(rate=100.0)
    proc = BigraphicalReactiveSystem(
        config={
            'rules': [rule],
            'mode': 'gillespie',
            'seed': 1,
            'max_per_tick': 2},
        core=core)

    state = {'state': {'bldg': _building_with_agents(5)}}
    update = proc.update(state, interval=10.0)

    bldg = update['state']['bldg']
    moved = [k for k in bldg['lab'] if k.startswith('agent_')]
    assert len(moved) == 2
    assert len(proc.fired_log) == 2


def test_bigraphical_reactive_system_in_composite(core):
    """BigraphicalReactiveSystem wired through a Composite. Each
    tick advances `global_time` by the configured `interval` and
    fires one deterministic rule match against the building
    store."""
    rule = _enter_room_rule()

    spec = {
        'schema': {'building': 'tree[node]'},
        'state': {
            'building': _building_with_agents(2),
            'brs': {
                '_type': 'process',
                'address': (
                    'local:!process_bigraph.processes'
                    '.bigraphical_reactive_system'
                    '.BigraphicalReactiveSystem'),
                'config': {'rules': [rule], 'mode': 'deterministic'},
                'inputs': {'state': ['building']},
                'outputs': {'state': ['building']},
                'interval': 1.0}}}

    composite = Composite(spec, core=core)
    before = composite.state['building']
    assert sum(1 for k in before if k.startswith('agent_')) == 2

    # Two ticks → two agents moved into the room.
    composite.run(2.0)

    after = composite.state['building']
    lab = after['lab']
    moved = [k for k in lab if k.startswith('agent_')]
    remaining = [k for k in after if k.startswith('agent_')]
    assert len(moved) == 2
    assert len(remaining) == 0


def test_port_outputs_propagate_to_store_schema(core):
    """A process declaring ``_outputs: array[float[64]]`` should cause the
    wired target store to have an Array schema after Composite init, so
    list-typed seed values get coerced to ndarrays through realize.
    """
    import numpy as np

    # Seed the wired store with a Python list default. After realize
    # folds port_merges into the schema, the store's `arr` should be
    # coerced from list → ndarray.
    state = {
        'global_time': 0.0,
        'arr': [0.0, 0.0, 0.0],
        'tick': 1.0,
        'writer': {
            '_type': 'process',
            'address': (
                f'local:!{_ArrWriterProcess.__module__}.'
                f'{_ArrWriterProcess.__name__}'
            ),
            'config': {},
            'inputs': {'tick': ['tick']},
            'outputs': {'arr': ['arr']},
            'interval': 1.0,
        },
    }

    composite = Composite({'schema': {}, 'state': state}, core=core)

    assert isinstance(composite.state['arr'], np.ndarray), (
        f'store at `arr` should be ndarray after realize coerces through '
        f'the _ArrWriterProcess._outputs schema — got '
        f'{type(composite.state["arr"]).__name__}')
    assert composite.state['arr'].dtype == np.dtype('float64')


def test_dynamic_worker_rewire_preserves_instance(core):
    """A rewire should update the worker's `outputs` wires in place —
    the existing process instance is preserved. The `_rewire` sentinel
    triggers re-realize so the compiled link cache is rebuilt with the
    new wiring, but realize_link reuses the existing instance.
    """
    DynamicWorker._counter = 0

    schema = {
        'pool': {
            '_type': 'map',
            '_value': {
                'value': 'float',
                'worker': {
                    '_type': 'process',
                    'address': make_default('string', 'local:DynamicWorker'),
                    'inputs': make_default('wires', {
                        'sources': ['..'],
                        'self_value': ['value']}),
                    'outputs': make_default('wires', {
                        'targets': ['..'],
                        'self_value': ['value']}),
                    'interval': make_default('float', 1.0)}}}}

    # Two agents, only a0 will rewire (a1 has propensity_rewire=0).
    # threshold_rewire=4.0 with source_sum-based trigger; growth_rate=1.0.
    state = {
        'pool': {
            'a0': {
                'value': 1.0,
                'worker': _make_worker_state(
                    'a0',
                    propensity_rewire=1.0,
                    propensity_spawn=0.0,
                    propensity_remove=0.0,
                    threshold_rewire=2.0)},
            'a1': {
                'value': 5.0,
                'worker': _make_worker_state(
                    'a1',
                    propensity_rewire=0.0,
                    propensity_spawn=0.0,
                    propensity_remove=0.0)}}}

    composite = Composite({
        'schema': schema,
        'state': state}, core=core)

    a0_instance_before = composite.state['pool']['a0']['worker']['instance']
    a0_outputs_before = composite.state['pool']['a0']['worker']['outputs']
    assert a0_outputs_before['self_value'] == ['value']

    # One tick is enough: a1.value=5.0 → a0 sees source_sum=5 > 2.0, rewires.
    composite.run(1.0)

    a0_instance_after = composite.state['pool']['a0']['worker']['instance']
    a0_outputs_after = composite.state['pool']['a0']['worker']['outputs']

    assert a0_instance_after is a0_instance_before, (
        'rewire should preserve the existing worker instance')
    assert a0_outputs_after['self_value'] == ['..', 'a1', 'value'], (
        f"expected outputs.self_value to be rewired to a1, "
        f"got {a0_outputs_after['self_value']}")


class _SelfIntervalProcess(Process):
    """Emits a fixed value to its own ``interval`` field via an
    ``overwrite[float]`` output port, and records the interval it
    receives on every ``update()`` call. Used by
    ``test_self_interval_overwrite_does_not_accumulate``.
    """
    config_schema = {'fixed_interval': 'float'}

    def initialize(self, config):
        self.received_intervals = []

    def inputs(self):
        return {'tick': 'float'}

    def outputs(self):
        return {'interval': 'overwrite[float]'}

    def update(self, state, interval):
        self.received_intervals.append(interval)
        return {'interval': self.config['fixed_interval']}


def test_self_interval_overwrite_does_not_accumulate(core):
    """A process wires its ``overwrite[float]`` output back to its own
    ``interval`` field.

    Two things must hold:

    1. After several ticks the field equals the fixed value emitted by
       the process (overwrite), rather than the accumulated sum
       (``initial_interval + fixed_interval * N``) that the additive
       ``apply(Float)`` would produce when the destination's
       ProcessLink.interval Float schema wins over the source's
       Overwrite Wrap during ``promote``.
    2. The runtime actually uses the overwritten interval to schedule
       and invoke the next tick — i.e. the ``interval`` argument
       handed to ``update()`` after the first firing is the fixed
       value, not the original initial interval and not the
       accumulated sum.
    """
    fixed = 0.25
    initial_interval = 1.0

    address = (
        f'local:!{_SelfIntervalProcess.__module__}.'
        f'{_SelfIntervalProcess.__name__}')

    state = {
        'tick': 1.0,
        'self_proc': {
            '_type': 'process',
            'address': address,
            'config': {'fixed_interval': fixed},
            'inputs': {'tick': ['tick']},
            'outputs': {'interval': ['self_proc', 'interval']},
            'interval': initial_interval}}

    composite = Composite({'state': state}, core=core)

    # Advance enough simulated time to fire several updates regardless of
    # whether the runtime ends up using the overwritten or accumulated
    # interval to schedule the next tick.
    composite.run(5.0)

    final = composite.state['self_proc']['interval']

    assert final == fixed, (
        f'expected interval to be overwritten to {fixed}, got {final} '
        f'(initial was {initial_interval}). The process emitted '
        f"'interval' as overwrite[float], so the destination "
        f"ProcessLink.interval should be replaced — not summed via the "
        f"additive Float apply.")

    instance = composite.state['self_proc']['instance']
    received = instance.received_intervals

    # The first firing happens before any update lands, so it must use
    # the originally-configured interval.
    assert received[0] == initial_interval, (
        f'first update() should have received initial interval '
        f'{initial_interval}, got {received[0]}')

    # Every subsequent firing must see the overwritten value. Under the
    # accumulating-Float bug the sequence would be
    # 1.0, 1.25, 1.5, 1.75, ... rather than 1.0, 0.25, 0.25, ...
    assert all(r == fixed for r in received[1:]), (
        f'every update() after the first should have received '
        f'fixed_interval={fixed}, got {received}. The runtime must '
        f'read the post-apply value of process["interval"] when '
        f'scheduling and invoking the next tick.')

    # And we should have actually fired more than once — otherwise the
    # "subsequent" claim is vacuous.
    assert len(received) >= 2, (
        f'expected multiple firings within the 5.0s run window; '
        f'received only {len(received)} intervals: {received}')


def test_partial_process_link_update(core):
    """Updating a single field on a ProcessLink (e.g. only ``interval``)
    must preserve the other fields. The default ``apply(Node)`` walked
    every dataclass field of the schema and recursed with
    ``update.get(key)`` — passing ``None`` for missing keys, which either
    crashed inside Wires/Tree or wiped out address/config/inputs/outputs.
    """
    state = {
        'level': 4.4,
        'increase': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.5},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': 1.0,
        },
    }
    composite = Composite({'state': state}, core=core)

    before = composite.state['increase']
    original_address = before['address']
    original_config = dict(before['config'])
    original_inputs = dict(before['inputs'])
    original_outputs = dict(before['outputs'])
    original_interval = before['interval']

    composite.apply({'increase': {'interval': 0.5}})

    after = composite.state['increase']
    assert after['interval'] == original_interval + 0.5
    assert after['address'] == original_address
    assert after['config'] == original_config
    assert after['inputs'] == original_inputs
    assert after['outputs'] == original_outputs


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

    test_dynamic_structure(core)
    # test_dfba_process(core)

    test_reaction_step(core)
    test_reaction_step_in_composite(core)

    test_bigraphical_reactive_system_deterministic(core)
    test_bigraphical_reactive_system_no_match_returns_empty(core)
    test_bigraphical_reactive_system_max_per_tick(core)
    test_bigraphical_reactive_system_stochastic(core)
    test_bigraphical_reactive_system_gillespie(core)
    test_bigraphical_reactive_system_gillespie_max_per_tick(core)
    test_bigraphical_reactive_system_in_composite(core)



    


# ---------------------------------------------------------------------------
# Tests for parallel_processes / Ray / REST server.
# Kept at the end of the file because they spin up Ray runtime / FastAPI
# TestClient state; running them earlier perturbs other tests' scheduling
# (especially the dynamic-structure tests further up).
# ---------------------------------------------------------------------------
def test_parallel_processes_matches_serial(core):
    """parallel_processes=True must produce bit-exact results vs the serial
    loop. Many independent processes, several timesteps."""
    n = 12
    initial_levels = [float(i + 1) for i in range(n)]

    def build_state():
        # Each cell is an independent IncreaseProcess writing to its own
        # 'level' store. No cross-cell dependencies => trivially parallel.
        state = {f"level_{i}": initial_levels[i] for i in range(n)}
        for i in range(n):
            state[f"grow_{i}"] = {
                "_type": "process",
                "address": "local:IncreaseProcess",
                "config": {"rate": 0.1 + 0.01 * i},
                "inputs":  {"level": [f"level_{i}"]},
                "outputs": {"level": [f"level_{i}"]},
                "interval": 1.0,
            }
        return state

    serial = Composite({"state": build_state()}, core=make_test_core())
    serial.run(10.0)

    parallel = Composite(
        {"state": build_state(), "parallel_processes": True},
        core=make_test_core(),
    )
    parallel.run(10.0)

    for i in range(n):
        s = serial.state[f"level_{i}"]
        p = parallel.state[f"level_{i}"]
        assert s == p, (
            f"level_{i} mismatch: serial={s!r}, parallel={p!r} "
            f"(parallel_processes must be deterministic and identical to serial)"
        )


def test_parallel_processes_single_process_path(core):
    """With just one process, the parallel layer must short-circuit and not
    spin up a thread pool — should match serial without overhead."""
    state = {
        "level": 1.0,
        "grow": {
            "_type": "process",
            "address": "local:IncreaseProcess",
            "config": {"rate": 0.5},
            "inputs":  {"level": ["level"]},
            "outputs": {"level": ["level"]},
            "interval": 1.0,
        },
    }
    sim = Composite({"state": state, "parallel_processes": True},
                    core=make_test_core())
    sim.run(3.0)
    # IncreaseProcess returns level*rate as a delta each tick (additive apply).
    # rate=0.5, three ticks of multiply-and-add: 1.0 + 0.5 + 0.75 + 1.125 = 3.375
    assert abs(sim.state["level"] - 3.375) < 1e-12


def test_by_hash_cache_is_per_instance(core):
    """A ``_cache='by_hash'`` Step must keep its memoization cache on the
    *instance*, not on the class. Two Steps of the same type (e.g. one per
    Composite) running in one process must not share cached results — a
    cache hit on one must never suppress recomputation on the other.

    Regression guard: when the cache lived in a class-level dict, the
    second instance's first invoke with the same (config, state) hash
    silently returned the first instance's result and skipped ``update``,
    leaking results across simulations and growing without bound.
    """
    calls = []

    class _CachedStep(Step):
        _cache = 'by_hash'
        _schema_version = 'v1'

        def inputs(self):
            return {'x': 'float'}

        def outputs(self):
            return {'y': 'float'}

        def update(self, state):
            calls.append(id(self))
            return {'y': state['x'] + 1.0}

    a = _CachedStep(core=core)
    b = _CachedStep(core=core)

    # First invoke on `a`: computes + caches.
    assert a.invoke({'x': 1.0}).get() == {'y': 2.0}
    assert len(calls) == 1
    # Same (config, state) again on `a`: cache HIT, no recompute — the
    # by-hash optimisation must still work *within* a single instance.
    assert a.invoke({'x': 1.0}).get() == {'y': 2.0}
    assert len(calls) == 1

    # `b` is a different instance with an identical cache key. It must NOT
    # see `a`'s cache: it recomputes (correct result, fresh update call).
    assert b.invoke({'x': 1.0}).get() == {'y': 2.0}
    assert len(calls) == 2

    # The caches are distinct dict objects, each holding only its own entry,
    # and the class never accumulates a shared cache.
    assert a._by_hash_cache is not b._by_hash_cache
    assert len(a._by_hash_cache) == 1
    assert len(b._by_hash_cache) == 1
    assert _CachedStep.__dict__.get('_by_hash_cache') is None


class _SharedRegProc:
    """Minimal stand-in for a shared process instance used to exercise the
    per-core ``_shared_processes`` registry through the realize dispatch.

    It deliberately does NOT subclass Edge/Process: ``SharedProcess.realize``
    instantiates the class as ``cls(config)`` with no core, which Edge
    forbids. Only the attributes the dispatch touches are provided.
    """

    def __init__(self, config):
        self.config = config
        self.core = None


def test_shared_process_registry_per_core():
    """The shared-process registry must be scoped per ``core`` so two
    simulations (two Composites / two cores) in one process cannot clobber
    each other's shared process registered under the same id.

    Regression guard: the registry used to be a single module-level dict,
    so the second core's ``realize`` overwrote the first core's entry under
    the same id, and every later lookup resolved to the wrong instance.
    """
    from bigraph_schema.methods import realize as realize_mm, serialize as serialize_mm
    from process_bigraph.types.process import (
        SharedProcess, SharedProcessRef, _core_shared_processes,
        get_shared_process, _shared_processes, _lookup_shared_process_id)

    core1 = make_test_core()
    core2 = make_test_core()
    addr = f'local:!{__name__}._SharedRegProc'
    sp = SharedProcess()

    # Both cores register a shared process under the SAME id 'proc' but with
    # different configs.
    _, r1, _ = realize_mm(
        core1, sp, {'address': addr, 'config': {'rate': 0.1}}, path=('proc',))
    _, r2, _ = realize_mm(
        core2, sp, {'address': addr, 'config': {'rate': 0.9}}, path=('proc',))

    # No cross-core clobbering: each core resolves to its own instance.
    assert r1[0].config['rate'] == 0.1
    assert r2[0].config['rate'] == 0.9
    assert get_shared_process('proc', core=core1).config['rate'] == 0.1
    assert get_shared_process('proc', core=core2).config['rate'] == 0.9

    # Registries are distinct per core, and the module-level fallback dict
    # is not used at all when a core scope is available.
    assert _core_shared_processes(core1) is not _core_shared_processes(core2)
    assert _shared_processes == {}

    # The instance is stamped with its id so reverse-lookup (used by
    # serialize/bundle of a ref, which have no core handle) needs no global.
    assert r1[0]._shared_process_id == 'proc'
    assert _lookup_shared_process_id(r1) == 'proc'

    # SharedProcessRef resolves against its own core's registry.
    spr = SharedProcessRef()
    _, ref1, _ = realize_mm(core1, spr, 'proc', path=('ref',))
    _, ref2, _ = realize_mm(core2, spr, 'proc', path=('ref',))
    assert ref1.config['rate'] == 0.1
    assert ref2.config['rate'] == 0.9
    assert serialize_mm(spr, r1) == 'proc'


@pytest.mark.slow
def test_ray_process_pool_shared_across_clients(core):
    """Two RayProcess clients with the same (class, config) must share one
    actor pool, and both must see correct results from the underlying
    process. Tests pool keying + reuse."""
    pytest.importorskip("ray")
    import ray as _ray
    from process_bigraph.protocols.ray import (
        RayProcess, register_process_class, pool_stats, shutdown_pools)

    register_process_class("IncreaseProcess", IncreaseProcess)
    try:
        a = RayProcess(
            {"process_class": "IncreaseProcess",
             "process_config": {"rate": 0.25}, "pool_size": 2},
            core=core)
        b = RayProcess(
            {"process_class": "IncreaseProcess",
             "process_config": {"rate": 0.25}, "pool_size": 2},
            core=core)
        # Both clients should be backed by the same single 2-actor pool.
        stats = pool_stats()
        assert len(stats) == 1, f"expected one pool, got {stats}"
        assert stats[0]["n_actors"] == 2

        # IncreaseProcess returns level*rate. rate=0.25, level=4.0 → 1.0.
        for client in (a, b):
            result = client.update({"level": 4.0}, interval=1.0)
            assert result == {"level": 1.0}
    finally:
        shutdown_pools()
        # Tear down the Ray runtime so it doesn't leak background threads /
        # scheduler state into subsequent tests in this process.
        if _ray.is_initialized():
            _ray.shutdown()


@pytest.mark.slow
def test_ray_process_distinct_configs_get_separate_pools(core):
    """Different (class, config) pairs must allocate independent pools so
    state doesn't leak between configurations."""
    pytest.importorskip("ray")
    import ray as _ray
    from process_bigraph.protocols.ray import (
        RayProcess, register_process_class, pool_stats, shutdown_pools)

    register_process_class("IncreaseProcess", IncreaseProcess)
    try:
        slow = RayProcess(
            {"process_class": "IncreaseProcess",
             "process_config": {"rate": 0.1}, "pool_size": 1},
            core=core)
        fast = RayProcess(
            {"process_class": "IncreaseProcess",
             "process_config": {"rate": 0.9}, "pool_size": 1},
            core=core)

        assert len(pool_stats()) == 2  # one per distinct config

        slow_result = slow.update({"level": 10.0}, interval=1.0)
        fast_result = fast.update({"level": 10.0}, interval=1.0)
        assert slow_result == {"level": 1.0}
        assert fast_result == {"level": 9.0}
    finally:
        shutdown_pools()
        if _ray.is_initialized():
            _ray.shutdown()


@pytest.mark.slow
def test_ray_protocol_address_batches_per_tick(core):
    """``address: "ray:Foo"`` keeps the per-process node visible in the
    state graph but batches per-tick RPCs through one shared shard pool.

    Verifies:
      - 4 IncreaseProcess clients via "ray:IncreaseProcess" get 4 distinct
        Process instances in the Composite,
      - they all map onto a single RayProtocolRuntime,
      - the per-tick RPC count is bounded by n_shards (≤ 4 here, capped
        by RAY_SHARDS_DEFAULT=2 to make the bound observable),
      - results match a direct local run.
    """
    pytest.importorskip("ray")
    import ray as _ray
    from process_bigraph.protocols.ray import (
        register_process_class, get_or_create_runtime,
        shutdown_all_runtimes,
    )

    register_process_class("IncreaseProcess", IncreaseProcess)
    os.environ["RAY_SHARDS_DEFAULT"] = "2"
    try:
        # Force a fresh runtime so RAY_SHARDS_DEFAULT applies.
        shutdown_all_runtimes()

        state = {
            f"grow_{i}": {
                "_type": "process",
                "address": "ray:IncreaseProcess",
                "config": {"rate": 0.5},
                "inputs":  {"level": [f"level_{i}"]},
                "outputs": {"level": [f"level_{i}"]},
                "interval": 1.0,
            }
            for i in range(4)
        }
        for i in range(4):
            state[f"level_{i}"] = 1.0

        sim = Composite({"state": state, "parallel_processes": True},
                        core=make_test_core())
        runtime = get_or_create_runtime(sim.core)
        sim.run(3.0)

        # Same growth as the local test: 1 -> 1.5 -> 2.25 -> 3.375
        for i in range(4):
            assert abs(sim.state[f"level_{i}"] - 3.375) < 1e-9

        # Bound on actor count: capped by RAY_SHARDS_DEFAULT=2, regardless
        # of the 4 per-process nodes in the graph.
        assert len(runtime._pools) == 1
        only_pool = next(iter(runtime._pools.values()))
        assert len(only_pool.actors) == 2
    finally:
        shutdown_all_runtimes()
        os.environ.pop("RAY_SHARDS_DEFAULT", None)
        if _ray.is_initialized():
            _ray.shutdown()


@pytest.mark.slow
def test_rest_server_initialize_inputs_outputs_update(core):
    """Smoke-test the in-process REST server (no socket): initialize a process,
    query its ports, run an update, end it. Mirrors the round-trip that
    RestProcess does over the network."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from process_bigraph.server.rest import start_server

    core.register_link("IncreaseProcess", IncreaseProcess)
    client = TestClient(start_server(core))

    schema = client.get("/process/IncreaseProcess/config-schema").json()
    assert "rate" in schema

    pid = client.post(
        "/process/IncreaseProcess/initialize", json={"rate": 0.3}
    ).json()
    assert isinstance(pid, str) and len(pid) > 0

    inputs = client.get(f"/process/IncreaseProcess/inputs/{pid}").json()
    outputs = client.get(f"/process/IncreaseProcess/outputs/{pid}").json()
    assert inputs == {"level": "float"}
    assert outputs == {"level": "float"}

    upd = client.post(
        f"/process/IncreaseProcess/update/{pid}",
        json={"state": {"level": 10.0}, "interval": 1.0},
    ).json()
    assert upd == {"level": 3.0}

    end = client.post(f"/process/IncreaseProcess/end/{pid}", json={})
    assert end.status_code == 200


# ---------------------------------------------------------------------------
# ActorPool tests — pool's actors survive multiple acquire/release cycles,
# and module-global registry shares pools across callers.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_session_reconfigures_pool_actors_without_respawn():
    """Validates the central premise of Session: enter calls reconfigure
    with per-sim config; exit returns actors to pool without killing.

    Using a stateful actor where __init__ sets a stable ``base_id`` and
    reconfigure rebinds a ``shard_label`` — we observe both that the
    actor instance is preserved (same base_id) AND that reconfigure
    fired (label changed)."""
    pytest.importorskip("ray")
    import ray as _ray
    from process_bigraph.protocols.pool import (
        get_or_create_pool, shutdown_all_pools,
    )
    from process_bigraph.protocols.session import Session

    if not _ray.is_initialized():
        _ray.init(ignore_reinit_error=True, log_to_driver=False)

    @_ray.remote
    class StatefulActor:
        def __init__(self, _config):
            import uuid
            self.base_id = str(uuid.uuid4())  # set ONCE — pool actor
            self.label = None                  # rebound by reconfigure

        def ping(self):
            return "ready"

        def reconfigure(self, config):
            # Cheap rebind — no re-run of __init__ work.
            self.label = config.get("label")

        def info(self):
            return (self.base_id, self.label)

    try:
        shutdown_all_pools()
        pool = get_or_create_pool(StatefulActor, {}, size=2)
        pool.warm()

        # Session 1: reconfigure with label="A".
        with Session(pool, n_actors=2,
                     sim_config={"label": "A"}) as session_a:
            ids_a, labels_a = zip(*_ray.get([a.info.remote()
                                              for a in session_a.actors]))
            assert all(label == "A" for label in labels_a)
        # After session exit, actors must be back in pool, NOT killed.
        assert pool.stats() == {"size": 2, "warmed": True,
                                 "available": 2, "in_use": 0}

        # Session 2: same pool, reconfigure with label="B".
        with Session(pool, n_actors=2,
                     sim_config={"label": "B"}) as session_b:
            ids_b, labels_b = zip(*_ray.get([a.info.remote()
                                              for a in session_b.actors]))

        # Same actors as session 1 (base_ids preserved).
        assert set(ids_a) == set(ids_b)
        # But reconfigure rebound the label.
        assert all(label == "B" for label in labels_b)

        # Reconfigure can be skipped when sim_config is empty.
        with Session(pool, n_actors=1) as session_c:
            assert session_c.actors  # claimed
            assert session_c.n_actors == 1
        assert pool.stats()["available"] == 2
    finally:
        shutdown_all_pools()
        if _ray.is_initialized():
            _ray.shutdown()


@pytest.mark.slow
def test_actor_pool_reuses_actors_across_acquires():
    """Validates the central premise of ActorPool: one ``warm()`` paid up
    front; subsequent acquire/release cycles do NOT re-spawn actors.

    Done with a stateful counting actor: the count is set in __init__ to
    a unique random id, so if a session got a fresh actor instead of a
    pooled one, the id would change.
    """
    pytest.importorskip("ray")
    import ray as _ray
    from process_bigraph.protocols.pool import (
        ActorPool, get_or_create_pool, shutdown_all_pools,
    )

    if not _ray.is_initialized():
        _ray.init(ignore_reinit_error=True, log_to_driver=False)

    @_ray.remote
    class CountingActor:
        def __init__(self, _config):
            import uuid
            self.id = str(uuid.uuid4())
            self.calls = 0

        def ping(self):
            return "ready"

        def call(self):
            self.calls += 1
            return (self.id, self.calls)

    try:
        shutdown_all_pools()
        pool = get_or_create_pool(CountingActor, {"foo": "bar"}, size=2)
        pool.warm()

        # Session 1
        actors1 = pool.acquire(2)
        ids_1, counts_1 = zip(*_ray.get([a.call.remote() for a in actors1]))
        pool.release(actors1)
        assert pool.stats()["available"] == 2
        assert pool.stats()["in_use"] == 0
        assert all(c == 1 for c in counts_1)

        # Session 2 — same pool, SAME actors (state preserved).
        actors2 = pool.acquire(2)
        ids_2, counts_2 = zip(*_ray.get([a.call.remote() for a in actors2]))
        pool.release(actors2)
        # Actor ids preserved — pool didn't re-spawn.
        assert set(ids_1) == set(ids_2)
        # Counts incremented from where session 1 left off.
        assert all(c == 2 for c in counts_2)

        # Module-global registry: same args → same pool.
        pool_again = get_or_create_pool(CountingActor, {"foo": "bar"}, size=2)
        assert pool_again is pool

        # Different config → different pool.
        other_pool = get_or_create_pool(CountingActor, {"foo": "baz"}, size=2)
        assert other_pool is not pool
    finally:
        shutdown_all_pools()
        if _ray.is_initialized():
            _ray.shutdown()


@pytest.mark.slow
def test_actor_pool_grows_on_acquire_request_larger_than_size():
    """When acquire(n) asks for more actors than the pool currently has,
    the pool grows to fit. Existing pooled actors keep their state —
    only the *new* slots pay the per-actor __init__ cost.

    This covers the multi-grid sweep pattern: first sim wants 2 actors,
    second wants 8, third wants 32 — all from one shared pool keyed
    by (actor_class, config_hash).
    """
    pytest.importorskip("ray")
    import ray as _ray
    from process_bigraph.protocols.pool import (
        get_or_create_pool, shutdown_all_pools)

    if not _ray.is_initialized():
        _ray.init(ignore_reinit_error=True, log_to_driver=False)

    @_ray.remote
    class TaggedActor:
        def __init__(self, _config):
            import uuid
            self.tag = str(uuid.uuid4())
        def ping(self): return "ready"
        def info(self): return self.tag

    try:
        shutdown_all_pools()
        # Pool starts at size 2 (first sweep).
        pool = get_or_create_pool(TaggedActor, {}, size=2)
        actors_2 = pool.acquire(2)
        tags_first_two = set(_ray.get([a.info.remote() for a in actors_2]))
        pool.release(actors_2)
        assert pool.size == 2

        # Second sweep wants 8. Pool should grow to 8 — the original 2
        # actors keep their tags, the new 6 get fresh tags.
        actors_8 = pool.acquire(8)
        assert pool.size == 8
        tags_all = set(_ray.get([a.info.remote() for a in actors_8]))
        # Original 2 tags must still be there (state preserved).
        assert tags_first_two.issubset(tags_all)
        assert len(tags_all) == 8
        pool.release(actors_8)

        # Third sweep wants 4 — that's smaller, pool stays at 8.
        actors_4 = pool.acquire(4)
        assert pool.size == 8
        pool.release(actors_4)

        # get_or_create_pool with a larger size also grows.
        same_pool = get_or_create_pool(TaggedActor, {}, size=12)
        assert same_pool is pool
        assert pool.size == 12
    finally:
        shutdown_all_pools()
        if _ray.is_initialized():
            _ray.shutdown()


def test_tick_lifecycle_applied_skips_apply_updates(core):
    """v3 path: a runtime that mutates ``composite.state`` directly
    returns ``applied=True`` and the framework skips ``apply_updates``
    entirely for the managed group. Verifies state mutations land
    correctly AND that no Defer is placed at any common_path slot.

    This exercises the framework hook used by spatio-flux's
    ShardManager.tick_lifecycle to bypass the per-tick reconcile/apply
    schema walk at scale.
    """
    class DirectMutationRuntime:
        def __init__(self):
            self.tick_count = 0

        def tick_lifecycle(self, *, processes, composite, global_time,
                           end_time, force_complete):
            self.tick_count += 1
            # Mutate composite.state directly — the runtime's whole job.
            next_time = end_time
            for req in processes:
                inc = req['instance'].config.get('increment', 0)
                target = req['instance'].config.get('target')
                composite.state[target] += inc
                future_time = global_time + float(req['interval'])
                if future_time < next_time:
                    next_time = future_time
            return {
                'next_time': next_time,
                'process_paths': [r['path'] for r in processes],
                'applied': True,
            }

    runtime = DirectMutationRuntime()

    class DirectIncrementer(Process):
        config_schema = {'increment': 'integer', 'target': 'string'}

        def initialize(self, config):
            self._protocol_runtime = runtime

        def inputs(self): return {}
        def outputs(self): return {}
        def update(self, state, interval):
            # Should NOT be called — runtime takes over.
            raise AssertionError("update should not fire when applied=True")

    core.register_link('DirectIncrementer', DirectIncrementer)

    state = {
        'a': 0,
        'b': 0,
        'inc_a': {'_type': 'process',
                   'address': 'local:DirectIncrementer',
                   'config': {'increment': 1, 'target': 'a'},
                   'inputs': {}, 'outputs': {}, 'interval': 1.0},
        'inc_b': {'_type': 'process',
                   'address': 'local:DirectIncrementer',
                   'config': {'increment': 5, 'target': 'b'},
                   'inputs': {}, 'outputs': {}, 'interval': 1.0},
    }

    sim = Composite({'state': state}, core=core)
    sim.run(3.0)

    assert runtime.tick_count == 3
    assert sim.state['a'] == 3
    assert sim.state['b'] == 15

    # Front entries for the managed processes should have empty update
    # slots — nothing should have been queued for apply_updates.
    for path in (('inc_a',), ('inc_b',)):
        if path in sim.front:
            assert sim.front[path]['update'] == {}


def test_tick_lifecycle_dispatches_managed_processes(core):
    """Validates the framework hook: a Process whose ``_protocol_runtime``
    implements ``tick_lifecycle`` gets dispatched via that hook (one call
    per runtime per tick) instead of the per-process invoke loop, and the
    combined Defer's projected updates apply correctly.
    """
    # A toy "runtime" that batches multiple incrementing processes into a
    # single tick_lifecycle call. Returns one combined Defer covering all
    # processes' increments.
    class BatchingRuntime:
        def __init__(self):
            self.tick_count = 0
            self.last_process_count = 0

        def tick_lifecycle(self, *, processes, composite, global_time,
                           end_time, force_complete):
            self.tick_count += 1
            self.last_process_count = len(processes)

            raw_defers = []
            next_time = end_time
            for req in processes:
                proc = req['instance']
                interval = float(req['interval']) if req['interval'] else 0.0
                state = composite._cached_view(req['path'])
                d = composite.process_update(
                    req['path'],
                    {'instance': proc, 'interval': interval},
                    state, interval, already_clean=True,
                )
                raw_defers.append(d)
                future_time = global_time + interval
                if future_time < next_time:
                    next_time = future_time

            # Concatenate per-process Defers into one — each .get() emits
            # a list of (schema, state) tuples.
            class _Combined:
                def __init__(self_inner, ds): self_inner._ds = ds
                def get(self_inner):
                    out = []
                    for d in self_inner._ds:
                        s = d.get()
                        if s is None: continue
                        if not isinstance(s, list): s = [s]
                        out.extend(s)
                    return out

            return {
                'common_path': ('counters',),
                'next_time': next_time,
                'process_paths': [r['path'] for r in processes],
                'defer': _Combined(raw_defers),
            }

    # Module-global so the toy class can register itself once per test
    # process. Tests are not concurrent, so the shared singleton is OK.
    runtime = BatchingRuntime()

    class ManagedIncrementer(Process):
        config_schema = {'increment': 'integer'}

        def initialize(self, config):
            # Hook the runtime in — Composite picks this up at
            # find_instance_paths time.
            self._protocol_runtime = runtime

        def inputs(self):
            return {'count': 'integer'}

        def outputs(self):
            return {'count': 'integer'}

        def update(self, state, interval):
            return {'count': self.config['increment']}

    core.register_link('ManagedIncrementer', ManagedIncrementer)

    state = {
        'inc1': {'_type': 'process',
                  'address': 'local:ManagedIncrementer',
                  'config': {'increment': 1},
                  'inputs': {'count': ['counters', 'a']},
                  'outputs': {'count': ['counters', 'a']},
                  'interval': 1.0},
        'inc2': {'_type': 'process',
                  'address': 'local:ManagedIncrementer',
                  'config': {'increment': 10},
                  'inputs': {'count': ['counters', 'b']},
                  'outputs': {'count': ['counters', 'b']},
                  'interval': 1.0},
        'counters': {'a': 0, 'b': 0},
    }

    sim = Composite({'state': state}, core=core)
    sim.run(3.0)

    # tick_lifecycle should have fired ONCE per tick (3 ticks at interval=1).
    # Both processes go through the same runtime → one batched call per tick.
    assert runtime.tick_count == 3
    assert runtime.last_process_count == 2

    # Updates applied: a += 1 each tick, b += 10 each tick.
    assert sim.state['counters']['a'] == 3
    assert sim.state['counters']['b'] == 30


# ==========================================
# A Process must advance time (no run() hang)
# ==========================================

def _increaser(interval=None, rate=0.1):
    """An `IncreaseProcess` node, optionally with an explicit interval."""
    node = {
        '_type': 'process',
        'address': 'local:IncreaseProcess',
        'config': {'rate': rate},
        'inputs': {'level': ['level']},
        'outputs': {'level': ['level']},
    }
    if interval is not None:
        node['interval'] = interval
    return {'proc': node, 'level': 1.0}


@pytest.mark.parametrize('interval', [0, 0.0, -1.0])
def test_non_advancing_interval_raises_instead_of_hanging(interval):
    """A non-positive interval used to spin `run()` forever.

    The scheduler advances `global_time` by the smallest process interval, so
    a zero or negative one leaves time unchanged and the run loop never
    reaches `end_time`. It must fail loudly instead.
    """
    core = allocate_core()
    sim = Composite({'state': _increaser(interval)}, core=core)

    with pytest.raises(ValueError, match='non-advancing interval') as raised:
        sim.run(3.0)

    message = str(raised.value)
    assert "'proc'" in message              # names the offending process
    assert 'must advance time' in message


def test_non_advancing_interval_raises_in_the_parallel_path():
    """The parallel scheduler shares the hazard, so it shares the guard."""
    core = allocate_core()
    good = _increaser(1.0)['proc']
    bad = _increaser(0.0, rate=0.2)['proc']
    state = {'a': good, 'b': bad, 'level': 1.0}

    sim = Composite(
        {'state': state, 'parallel_processes': True}, core=core)
    assert sim._parallel_processes and len(sim.process_paths) > 1

    with pytest.raises(ValueError, match='non-advancing interval') as raised:
        sim.run(3.0)

    assert "'b'" in str(raised.value)       # the bad one, not its neighbour


def test_calculate_timestep_returning_zero_raises():
    """The guard sits after `calculate_timestep`, so a process that computes
    a non-advancing step is caught too — not only a declared one."""
    core = allocate_core()

    class StalledProcess(Process):
        config_schema = {}

        def inputs(self):
            return {'level': 'float'}

        def outputs(self):
            return {'level': 'float'}

        def calculate_timestep(self, interval, state):
            return 0.0

        def update(self, state, interval):
            return {'level': 1.0}

    core.register_link('StalledProcess', StalledProcess)
    state = {
        'proc': {
            '_type': 'process',
            'address': 'local:StalledProcess',
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': 1.0},
        'level': 1.0}

    sim = Composite({'state': state}, core=core)
    with pytest.raises(ValueError, match='non-advancing interval'):
        sim.run(3.0)


def test_omitted_interval_takes_the_schema_default():
    """An omitted interval is not an error — it takes the `process` schema
    default of 1.0 and runs."""
    core = allocate_core()
    sim = Composite({'state': _increaser()}, core=core)

    assert sim.state['proc']['interval'] == 1.0
    sim.run(3.0)
    assert sim.state['global_time'] == 3.0


def test_positive_interval_still_runs():
    core = allocate_core()
    sim = Composite({'state': _increaser(0.5)}, core=core)

    sim.run(2.0)
    assert sim.state['global_time'] == 2.0
    assert sim.state['level'] > 1.0


# ==========================================
# A declared interval survives access/render
# ==========================================

@pytest.mark.parametrize('declared,expected', [(1.0, 1.0), (0.5, 0.5), (None, 1.0)])
def test_declared_interval_survives_access_and_render(declared, expected):
    """`access` used to drop a declared `interval` (keeping the schema
    default) and `render` emitted the bare type `'float'`, so a document that
    round-tripped through the schema layer came back with interval 0 — a
    process that could not advance time.
    """
    core = allocate_core()
    node = {'_type': 'process', 'address': 'local:IncreaseProcess'}
    if declared is not None:
        node['interval'] = declared

    schema = core.access(node)
    assert schema.interval._default == expected

    rendered = core.render(schema)
    assert rendered['interval'] == expected
    assert core.access(rendered).interval._default == expected


def test_round_tripped_document_still_runs():
    """The end-to-end form: a document that goes through `access`/`render`
    must still advance time when handed back to a Composite."""
    core = allocate_core()
    document = _increaser(1.0)

    rendered = core.render(core.access(document))
    assert rendered['proc']['interval'] == 1.0

    sim = Composite({'state': rendered}, core=allocate_core())
    sim.run(2.0)
    assert sim.state['global_time'] == 2.0


# ==========================================
# Layer 2b — the emitter's `results` port
# ==========================================

def _emitting_run(core, emitter_node=None):
    node = emitter_node or {
        '_type': 'step', 'address': 'local:RAMEmitter',
        'config': {'emit': {'level': 'node', 'global_time': 'node'}},
        'inputs': {'level': ['level'], 'global_time': ['global_time']},
        'outputs': {'results': ['results']}}
    doc = {
        'level': 1.0, 'results': {},
        'model': {
            '_type': 'process', 'address': 'local:IncreaseProcess',
            'config': {'rate': 0.5}, 'interval': 1.0,
            'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}},
        'emitter': node}
    sim = Composite({'state': doc}, core=core)
    sim.run(3.0)
    return sim


def test_an_emitter_declares_a_results_port():
    core = allocate_core()
    emitter = RAMEmitter({'emit': {'level': 'node'}}, core)
    assert emitter.outputs() == {'results': 'node'}


def test_a_downstream_step_can_depend_on_the_emitter():
    """The point of the port: the emitter becomes an ordinary producer, so a
    flush step is wired to it as a normal producer/consumer edge instead of
    reaching for the imperative pull."""
    core = allocate_core()

    class Flush(Step):
        config_schema = {}

        def inputs(self):
            return {'results': 'node'}

        def outputs(self):
            return {'rows': 'integer'}

        def update(self, state):
            return {'rows': 0}

    core.register_link('Flush', Flush)
    doc = {
        'level': 1.0, 'results': {}, 'rows': 0,
        'emitter': {
            '_type': 'step', 'address': 'local:RAMEmitter',
            'config': {'emit': {'level': 'node'}},
            'inputs': {'level': ['level']},
            'outputs': {'results': ['results']}},
        'flush': {
            '_type': 'step', 'address': 'local:Flush',
            'inputs': {'results': ['results']},
            'outputs': {'rows': ['rows']}}}

    sim = Composite({'state': doc}, core=core)

    # the emitter now *produces* a path, and the flush consumes it
    assert sim.step_dependencies[('emitter',)]['output_paths'] == [('results',)]
    assert sim.step_dependencies[('flush',)]['input_paths'] == [('results',)]


def test_the_results_handle_is_a_reference_not_the_data():
    core = allocate_core()
    sim = _emitting_run(core)

    handle = sim.finalize()[('emitter',)]

    assert isinstance(handle, EmitterResults)
    assert handle.count == len(gather_emitter_results(sim)[('emitter',)])
    # the summary carries no bulk
    summary = handle.to_dict()
    assert set(summary) == {'address', 'path', 'count', 'context'}
    assert summary['path'] == ['emitter']


def test_the_handle_resolves_to_the_accumulated_data():
    core = allocate_core()
    sim = _emitting_run(core)

    handle = sim.finalize()[('emitter',)]
    rows = handle.resolve()

    assert len(rows) > 1
    assert 'level' in rows[0] and 'global_time' in rows[0]
    assert rows == gather_emitter_results(sim)[('emitter',)]


def test_finalize_writes_the_handle_to_the_wired_store():
    core = allocate_core()
    sim = _emitting_run(core)

    assert sim.state['results'] == {}        # nothing during the run
    sim.finalize()

    handle = sim.state['results']
    assert isinstance(handle, EmitterResults)
    assert len(handle.resolve()) == handle.count


def test_results_are_not_written_per_tick():
    """`results` is a completion-time value. Writing it from `update` would
    fire every downstream consumer on every tick, which is exactly the
    behaviour the port exists to avoid."""
    core = allocate_core()
    sim = _emitting_run(core)

    assert sim.state['results'] == {}
    assert RAMEmitter({'emit': {}}, core).update({}) == {}


def test_finalize_skips_edges_without_results():
    """Additive: an ordinary step contributes nothing to finalize."""
    core = allocate_core()

    class Plain(Step):
        config_schema = {}

        def inputs(self):
            return {'level': 'float'}

        def outputs(self):
            return {'seen': 'integer'}

        def update(self, state):
            return {'seen': 1}

    core.register_link('Plain', Plain)
    doc = {
        'level': 1.0, 'seen': 0,
        'plain': {
            '_type': 'step', 'address': 'local:Plain',
            'inputs': {'level': ['level']}, 'outputs': {'seen': ['seen']}}}

    sim = Composite({'state': doc}, core=core)
    sim.run(0.0)
    assert sim.finalize() == {}


def test_gather_emitter_results_is_unchanged():
    """Back-compat: the imperative pull keeps working exactly as before, with
    or without a wired results port."""
    core = allocate_core()

    unwired = {
        '_type': 'step', 'address': 'local:RAMEmitter',
        'config': {'emit': {'level': 'node', 'global_time': 'node'}},
        'inputs': {'level': ['level'], 'global_time': ['global_time']}}

    sim = _emitting_run(core, emitter_node=unwired)
    rows = gather_emitter_results(sim)[('emitter',)]

    assert len(rows) > 1
    assert rows[0]['level'] == 1.0
    # an emitter with no results wire finalizes harmlessly
    assert isinstance(sim.finalize()[('emitter',)], EmitterResults)


# ==========================================
# Layer 2b — the study as a higher-order DAG
# ==========================================
#
# The flush entities are ORDINARY steps. What they depend on is the whole
# simulation, as a single node. Because the sim's per-tick stepping happens
# *inside* that node rather than in the outer network, the flush steps are
# not siblings of the sim's internal steps — so ordinary producer/consumer
# ordering fires them once, after the sim completes. No finalize marker, no
# scheduler exclusion, no completion phase.

FLUSH_FIRINGS = []


class FlushStep(Step):
    """A report card: an ordinary step that consumes `results`."""

    config_schema = {}

    def inputs(self):
        return {'results': 'node'}

    def outputs(self):
        return {'rows': 'integer'}

    def update(self, state):
        handle = state.get('results')
        FLUSH_FIRINGS.append(type(handle).__name__)
        return {'rows': len(handle.resolve())
                if isinstance(handle, EmitterResults) else -1}


def _sim_state():
    return {
        'level': 1.0, 'results': {},
        'model': {
            '_type': 'process', 'address': 'local:IncreaseProcess',
            'config': {'rate': 0.5}, 'interval': 1.0,
            'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}},
        'emitter': {
            '_type': 'step', 'address': 'local:RAMEmitter',
            'config': {'emit': {'level': 'node', 'global_time': 'node'}},
            'inputs': {'level': ['level'], 'global_time': ['global_time']},
            'outputs': {'results': ['results']}}}


def _study_core_2b():
    # SimulationStep ships in process_bigraph.processes and auto-registers,
    # so a study names it by address like any other node.
    core = allocate_core()
    core.register_link('FlushStep', FlushStep)
    return core


def _study_state(runtime=4.0):
    return {
        'results': {}, 'rows': 0,
        'sim': {
            '_type': 'step', 'address': 'local:SimulationStep',
            'config': {'state': _sim_state(), 'runtime': runtime},
            'inputs': {}, 'outputs': {'results': ['results']}},
        'flush': {
            '_type': 'step', 'address': 'local:FlushStep',
            'inputs': {'results': ['results']},
            'outputs': {'rows': ['rows']}}}


def test_the_study_is_a_higher_order_dag():
    """The outer network is sim-node → results → flush, by ordinary
    producer/consumer wiring."""
    core = _study_core_2b()
    study = Composite({'state': _study_state()}, core=core)

    assert study.step_dependencies[('sim',)]['output_paths'] == [('results',)]
    assert study.step_dependencies[('flush',)]['input_paths'] == [('results',)]
    # the simulation's own processes are inside the node, not in this network
    assert list(study.process_paths) == []


def test_a_flush_step_fires_exactly_once_after_the_simulation():
    """The proof: zero firings while the simulation steps internally, one
    firing after it completes — with the real handle."""
    FLUSH_FIRINGS.clear()

    core = _study_core_2b()
    study = Composite({'state': _study_state(runtime=4.0)}, core=core)
    study.run(0.0)

    # the simulation really did step internally
    assert study.state['sim']['instance'].composite.state['global_time'] == 4.0

    # ...and the flush fired once, not once per inner tick
    assert FLUSH_FIRINGS == ['EmitterResults']
    assert study.state['rows'] > 1


def test_the_flush_step_reads_the_resolved_handle():
    FLUSH_FIRINGS.clear()

    core = _study_core_2b()
    study = Composite({'state': _study_state(runtime=4.0)}, core=core)
    study.run(0.0)

    handle = study.state['results']
    assert isinstance(handle, EmitterResults)
    assert study.state['rows'] == len(handle.resolve()) == handle.count


def test_a_longer_simulation_does_not_fire_the_flush_more():
    """The flush count is independent of how many ticks the simulation
    takes — that is what "the sim is one node" buys."""
    for runtime in (2.0, 8.0, 20.0):
        FLUSH_FIRINGS.clear()

        core = _study_core_2b()
        study = Composite({'state': _study_state(runtime=runtime)}, core=core)
        study.run(0.0)

        inner = study.state['sim']['instance'].composite
        assert inner.state['global_time'] == runtime
        assert FLUSH_FIRINGS == ['EmitterResults'], (
            f'runtime={runtime} fired {len(FLUSH_FIRINGS)} times')


def test_several_flush_steps_all_fire_once():
    """Viz, analyses and report cards are just more consumers of the same
    handle."""
    FLUSH_FIRINGS.clear()

    core = _study_core_2b()
    state = _study_state()
    for name in ('viz', 'analysis'):
        state[f'{name}_rows'] = 0
        state[name] = {
            '_type': 'step', 'address': 'local:FlushStep',
            'inputs': {'results': ['results']},
            'outputs': {'rows': [f'{name}_rows']}}

    study = Composite({'state': state}, core=core)
    study.run(0.0)

    assert len(FLUSH_FIRINGS) == 3               # flush + viz + analysis
    assert set(FLUSH_FIRINGS) == {'EmitterResults'}
    assert study.state['viz_rows'] == study.state['analysis_rows']


def test_results_cross_a_composite_bridge():
    """A composite used as a node surfaces its `results` across the bridge:
    its update completes, so its completion-time outputs are produced."""
    core = allocate_core()
    inner = _sim_state()

    node = Composite(
        {'state': inner, 'bridge': {'outputs': {'results': ['results']}}},
        core=core)
    node.update({}, 4.0)

    handle = node.read_bridge()['results']
    assert isinstance(handle, EmitterResults)
    assert len(handle.resolve()) == handle.count


def test_simulation_step_names_the_emitter_when_ambiguous():
    """With more than one emitter the node cannot guess which supplies
    `results`, so it says so and lists them."""
    core = _study_core_2b()
    inner = _sim_state()
    inner['second_results'] = {}
    inner['second'] = {
        '_type': 'step', 'address': 'local:RAMEmitter',
        'config': {'emit': {'level': 'node'}},
        'inputs': {'level': ['level']},
        'outputs': {'results': ['second_results']}}

    state = _study_state()
    state['sim']['config']['state'] = inner
    study = Composite({'state': state}, core=core)

    with pytest.raises(ValueError, match='name the one') as raised:
        study.run(0.0)
    assert 'emitter' in str(raised.value) and 'second' in str(raised.value)


def test_simulation_step_selects_a_named_emitter():
    core = _study_core_2b()
    inner = _sim_state()
    inner['second_results'] = {}
    inner['second'] = {
        '_type': 'step', 'address': 'local:RAMEmitter',
        'config': {'emit': {'global_time': 'node'}},
        'inputs': {'global_time': ['global_time']},
        'outputs': {'results': ['second_results']}}

    state = _study_state()
    state['sim']['config']['state'] = inner
    state['sim']['config']['emitter'] = 'second'
    study = Composite({'state': state}, core=core)
    study.run(0.0)

    handle = study.state['results']
    assert handle.path == ('second',)
    assert 'global_time' in handle.resolve()[0]


def test_simulation_step_requires_a_wired_emitter():
    """An inner composite with nothing to report is a configuration error,
    reported by name rather than as an empty result."""
    core = _study_core_2b()
    inner = _sim_state()
    del inner['emitter']

    state = _study_state()
    state['sim']['config']['state'] = inner
    study = Composite({'state': state}, core=core)

    with pytest.raises(ValueError, match='produced no results'):
        study.run(0.0)


def test_simulation_step_exposes_the_finished_composite():
    core = _study_core_2b()
    study = Composite({'state': _study_state(runtime=3.0)}, core=core)
    study.run(0.0)

    inner = study.state['sim']['instance'].composite
    assert inner.state['global_time'] == 3.0
    assert inner.state['level'] > 1.0


def test_an_ascending_wire_still_creates_a_dependency_edge():
    """A wire is relative to its edge's parent, so a nested step reaching a
    shared store writes `['..', 'results']`. Runtime views resolve that; the
    dependency graph used to compare the literal path, so the edge was
    silently dropped and the consumer ran before its producer.
    """
    from process_bigraph.scheduling import find_leaves

    assert find_leaves(
        {'results': ['..', 'results']},
        path=('study', 'cards_0')) == [('study', 'results')]


def test_a_nested_flush_step_waits_for_the_simulation():
    """The nested case: report cards inside a region, reaching up to the
    shared `results` store, still fire exactly once and after the sim."""
    FLUSH_FIRINGS.clear()

    core = _study_core_2b()
    state = {
        'sim': {
            '_type': 'step', 'address': 'local:SimulationStep',
            'config': {'state': _sim_state(), 'runtime': 4.0},
            'inputs': {}, 'outputs': {'results': ['results']}},
        'cards_0': {
            'rows': 0,
            'card': {
                '_type': 'step', 'address': 'local:FlushStep',
                'inputs': {'results': ['..', 'results']},
                'outputs': {'rows': ['rows']}}}}

    study = Composite({'state': state}, core=core)
    assert study.step_dependencies[('cards_0', 'card')]['input_paths'] == [
        ('results',)]

    study.run(0.0)

    assert FLUSH_FIRINGS == ['EmitterResults']
    assert study.state['cards_0']['rows'] > 1


# ==========================================
# A0 — groundness at the Composite boundary
# ==========================================

MODEL_FACE = {
    '_type': 'link',
    '_inputs': {'level': 'float'},
    '_outputs': {'level': 'float'}}


def test_unfilled_required_site_is_rejected():
    """A document with an open template hole is not runnable.

    Nothing downstream — wiring, scheduling, emitters — can be built over a
    site, so the composite must say so at construction rather than fail
    obscurely mid-run.
    """
    core = allocate_core()
    document = {'study': {
        'model': {'_type': 'site', '_sort': MODEL_FACE},
        'level': 1.0}}

    with pytest.raises(ValueError, match='not ground') as raised:
        Composite({'state': document}, core=core)

    assert "'study/model'" in str(raised.value)


def test_every_unfilled_site_is_named():
    core = allocate_core()
    document = {'study': {
        'model': {'_type': 'site', '_sort': MODEL_FACE},
        'reference': {'_type': 'site', '_sort': MODEL_FACE}}}

    with pytest.raises(ValueError, match='not ground') as raised:
        Composite({'state': document}, core=core)

    message = str(raised.value)
    assert "'study/model'" in message
    assert "'study/reference'" in message


def test_optional_site_does_not_block_construction():
    """A site carrying a `_default` can supply its own filler, so it is not
    a required hole."""
    core = allocate_core()
    document = {'study': {
        'rate': {'_type': 'site', '_sort': 'float', '_default': 0.5},
        'level': 1.0}}

    sim = Composite({'state': document}, core=core)
    assert sim.state['study']['level'] == 1.0


def test_a_normal_composite_is_still_ground():
    """The check is over *sites*, not `is_ground`: a composite containing a
    process has unwired ports by construction, so `is_ground` would reject
    every real composite. It must not."""
    core = allocate_core()
    document = {
        'proc': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.1},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': 1.0},
        'level': 1.0}

    sim = Composite({'state': document}, core=core)
    sim.run(2.0)
    assert sim.state['global_time'] == 2.0


def test_a_filled_template_constructs():
    """The end-to-end shape: fill the hole, then the composite builds."""
    from bigraph_schema.assembly import fill_sites

    core = allocate_core()
    template = core.access({
        'proc': {'_type': 'site', '_sort': MODEL_FACE},
        'level': 'float'})
    filler = core.access({
        '_type': 'process',
        'address': 'local:IncreaseProcess',
        'config': {'rate': 0.1},
        'inputs': {'level': ['level']},
        'outputs': {'level': ['level']},
        'interval': 1.0})

    filled = fill_sites(core, template, {'proc': filler})

    sim = Composite({'state': core.render(filled)}, core=allocate_core())
    assert list(sim.process_paths) == [('proc',)]
    sim.run(2.0)
    assert sim.state['global_time'] == 2.0


# ==========================================
# A1 — declared emitters are actually installed
# ==========================================

def _emitting_spec(emitters):
    from process_bigraph.composite_spec import CompositeSpec
    return CompositeSpec(
        id='emitting', name='emitting',
        emitters=emitters,
        state={
            'proc': {
                '_type': 'process',
                'address': 'local:IncreaseProcess',
                'config': {'rate': 0.1},
                'inputs': {'level': ['level']},
                'outputs': {'level': ['level']},
                'interval': 1.0},
            'level': 1.0})


def test_declared_emitter_is_installed_and_gathers():
    """`CompositeSpec.emitters` is a first-class field, but `to_composite`
    never installed it — a composite built through the pure process-bigraph
    API had no observation sink at all."""
    spec = _emitting_spec([{'address': 'local:RAMEmitter', 'paths': ['level']}])

    sim = spec.to_composite()
    assert list(sim.step_paths) == [('emitter',)]

    sim.run(3.0)
    results = gather_emitter_results(sim)
    rows = results[('emitter',)]

    assert len(rows) > 1
    assert 'level' in rows[0] and 'global_time' in rows[0]


def test_emit_false_returns_the_bare_document():
    spec = _emitting_spec([{'address': 'local:RAMEmitter', 'paths': ['level']}])

    assert 'emitter' in spec.to_document()['state']
    assert 'emitter' not in spec.to_document(emit=False)['state']


def test_a_spec_with_no_emitters_installs_none():
    spec = _emitting_spec([])
    assert spec.to_document()['state'].keys() == {'proc', 'level'}
    assert list(_emitting_spec([]).to_composite().step_paths) == []


def test_installing_emitters_twice_yields_one_sink():
    """The shim and the dashboard also install; fixed keys mean a second
    install rewrites the same slot instead of adding a second sink."""
    from process_bigraph.emitter import install_emitters

    declarations = [{'address': 'local:RAMEmitter', 'paths': ['level']}]
    state = _emitting_spec(declarations).to_document()['state']

    once = install_emitters(state, declarations)
    twice = install_emitters(once, declarations)

    assert sorted(k for k in twice if k.startswith('emitter')) == ['emitter']
    assert twice['emitter'] == once['emitter']

    sim = Composite({'state': twice}, core=allocate_core())
    assert list(sim.step_paths) == [('emitter',)]


def test_multiple_declared_emitters_get_distinct_slots():
    from process_bigraph.emitter import install_emitters

    state = install_emitters({'level': 1.0}, [
        {'address': 'local:RAMEmitter', 'paths': ['level']},
        {'address': 'local:ConsoleEmitter', 'paths': ['level']}])

    assert sorted(k for k in state if k.startswith('emitter')) == [
        'emitter', 'emitter_1']


def test_declaration_paths_become_wires():
    """A declaration names what to observe; the wiring is computed. Slash-
    and dot-joined paths both address a nested store."""
    from process_bigraph.emitter import emitter_node_from_declaration

    node = emitter_node_from_declaration(
        {'address': 'local:RAMEmitter', 'paths': ['cell/mass', 'cell.volume']})

    assert node['inputs']['cell_mass'] == ['cell', 'mass']
    assert node['inputs']['cell_volume'] == ['cell', 'volume']
    # global_time is always emitted so trajectories carry a time axis.
    assert node['inputs']['global_time'] == ['global_time']
    assert node['config']['emit']['cell_mass'] == 'node'


def test_declaration_uses_the_one_emitter_constructor():
    """A declared sink and a generated one are the same node shape, because
    both come from `emitter_from_wires`."""
    from process_bigraph.emitter import (
        emitter_node_from_declaration, emitter_from_wires)

    declared = emitter_node_from_declaration(
        {'address': 'local:RAMEmitter', 'paths': ['level']})
    generated = emitter_from_wires(
        {'level': ['level'], 'global_time': ['global_time']},
        address='local:RAMEmitter')

    assert declared == generated


def test_unregistered_emitter_address_degrades_to_ram():
    from process_bigraph.emitter import emitter_node_from_declaration

    node = emitter_node_from_declaration(
        {'address': 'local:NoSuchEmitter', 'paths': ['level']},
        core=allocate_core())

    assert node['address'] == 'local:RAMEmitter'


def test_declared_emitter_config_is_preserved():
    from process_bigraph.emitter import emitter_node_from_declaration

    node = emitter_node_from_declaration({
        'address': 'local:RAMEmitter',
        'config': {'subsample': 5},
        'paths': ['level']})

    assert node['config']['subsample'] == 5
    assert 'emit' in node['config']


# ==========================================
# StepLink.priority survives access/render
# ==========================================

@pytest.mark.parametrize('declared,expected', [(5.0, 5.0), (2.5, 2.5), (None, 0.0)])
def test_declared_step_priority_survives_access_and_render(declared, expected):
    """`reify_schema_link` knows only the base `Link` fields, so a declared
    step `priority` was dropped exactly the way a process `interval` was: the
    schema kept its default and `render` emitted the bare type `'float'`, so
    a round-tripped step forgot its scheduling order.
    """
    core = allocate_core()
    node = {'_type': 'step', 'address': 'local:Collect'}
    if declared is not None:
        node['priority'] = declared

    schema = core.access(node)
    assert schema.priority._default == expected

    rendered = core.render(schema)
    assert rendered['priority'] == expected
    assert core.access(rendered).priority._default == expected


def test_declared_step_triggers_survive_access_and_render():
    core = allocate_core()
    schema = core.access({
        '_type': 'step',
        'address': 'local:Collect',
        '_triggers': {'a': ['a']}})

    assert schema._triggers == {'a': ['a']}
    assert core.render(schema)['_triggers'] == {'a': ['a']}


def test_the_scheduler_sees_a_round_tripped_priority():
    """The payoff: priority decides which step runs first when the network
    has a cycle, so it must survive the schema layer."""
    core = allocate_core()

    class Marker(Step):
        config_schema = {'name': 'string'}

        def inputs(self):
            return {'seen': 'list'}

        def outputs(self):
            return {'seen': 'list'}

        def update(self, state):
            return {'seen': state['seen'] + [self.config['name']]}

    core.register_link('Marker', Marker)

    document = {
        'low': {
            '_type': 'step', 'address': 'local:Marker',
            'config': {'name': 'low'}, 'priority': 1.0,
            'inputs': {'seen': ['seen']}, 'outputs': {'seen': ['seen']}},
        'high': {
            '_type': 'step', 'address': 'local:Marker',
            'config': {'name': 'high'}, 'priority': 9.0,
            'inputs': {'seen': ['seen']}, 'outputs': {'seen': ['seen']}},
        'seen': []}

    round_tripped = core.render(core.access(document))
    assert round_tripped['low']['priority'] == 1.0
    assert round_tripped['high']['priority'] == 9.0

    sim = Composite({'state': round_tripped}, core=core)
    priorities = {
        path[-1]: meta['priority']
        for path, meta in sim.step_dependencies.items()}
    assert priorities['high'] == 9.0
    assert priorities['low'] == 1.0


# ==========================================
# Part B' — typed parameter validation
# ==========================================

def _typed_spec(param_type, default):
    from process_bigraph.composite_spec import CompositeSpec
    return CompositeSpec(
        id='typed', name='typed',
        parameters={'rate': {'type': param_type, 'default': default}},
        state={'stores': {'rate': '${rate}'}})


@pytest.mark.parametrize('param_type,override,expected', [
    ('float', 1.0, 1.0),
    ('float', 1, 1.0),          # int for a float widens, as it always has
    ('float', '1.5', 1.5),      # a numeric string from a form field
    ('integer', 5, 5),
    ('integer', 5.0, 5),        # integral float is lossless
    ('integer', '5', 5),
    ('boolean', True, True),
    ('boolean', 'yes', True),
    ('boolean', 'false', False),
    ('string', 'x', 'x'),
    ('string', 5, '5'),
])
def test_a_well_typed_override_coerces_as_before(param_type, override, expected):
    spec = _typed_spec(param_type, override)
    document = spec.to_document(emit=False)
    assert document['state']['stores']['rate'] == expected


@pytest.mark.parametrize('param_type,override,fragment', [
    ('float', 'abc', 'not a number'),
    ('float', {'a': 1}, 'not a number'),
    ('float', True, 'a boolean is not a number'),
    ('integer', 3.7, 'fractional'),
    ('integer', 'x', 'not a whole number'),
    ('boolean', 'banana', 'not a recognized boolean'),
    ('map', [1], 'not a map'),
    ('list', {'a': 1}, 'not a list'),
])
def test_a_mistyped_override_is_rejected(param_type, override, fragment):
    """`_cast` used to coerce silently — `int(3.7)` became 3 and any
    unrecognized string became False — so a typo produced a plausible-looking
    but wrong simulation. It must name the parameter and say why."""
    from process_bigraph.composite_spec import ParameterTypeError

    spec = _typed_spec(param_type, 0 if param_type != 'string' else '')
    with pytest.raises(ParameterTypeError) as raised:
        spec.to_document({'rate': override}, emit=False)

    message = str(raised.value)
    assert "'rate'" in message
    assert param_type in message
    assert repr(override) in message
    assert fragment in message


def test_an_unknown_declared_type_is_left_alone():
    """A workspace type the core does not know is passed through rather than
    guessed at."""
    spec = _typed_spec('some_workspace_type', {'anything': 1})
    assert spec.to_document(emit=False)['state']['stores']['rate'] == {
        'anything': 1}


def test_an_untyped_parameter_is_left_alone():
    from process_bigraph.composite_spec import CompositeSpec

    spec = CompositeSpec(
        id='untyped', name='untyped',
        parameters={'rate': {'default': 'whatever'}},
        state={'stores': {'rate': '${rate}'}})

    assert spec.to_document(emit=False)['state']['stores']['rate'] == 'whatever'


def test_inline_interpolation_still_works():
    """The two real inline uses in the corpus are a solver input deck — string
    templating inside an opaque blob, which typing must not disturb."""
    from process_bigraph.composite_spec import CompositeSpec

    spec = CompositeSpec(
        id='inline', name='inline',
        parameters={'density': {'type': 'float', 'default': 0.8}},
        state={'sim': {'config': {'script': 'lattice sc ${density}\nrun 100'}}})

    script = spec.to_document(emit=False)['state']['sim']['config']['script']
    assert script == 'lattice sc 0.8\nrun 100'


def test_to_document_stays_a_pure_dict_walk():
    """Validation must not drag realization in: a spec naming a process the
    core cannot even load must still produce its document."""
    from process_bigraph.composite_spec import CompositeSpec

    spec = CompositeSpec(
        id='opaque', name='opaque',
        parameters={'interval': {'type': 'float', 'default': 2.0}},
        state={'proc': {
            '_type': 'process',
            'address': 'local:NoSuchProcessAnywhere',
            'config': {'tick': '${interval}'}}})

    document = spec.to_document(emit=False)
    assert document['state']['proc']['config']['tick'] == 2.0
    assert 'instance' not in document['state']['proc']


# ==========================================
# Layer 3 — studies and investigations ARE templates
# ==========================================

MODEL_FACE = {
    '_type': 'link',
    '_inputs': {'level': 'float'},
    '_outputs': {'level': 'float'}}


class ReportCard(Step):
    """A trivial downstream flush: turns the model's final level into a
    verdict, the way a study's report card turns a run into pass/fail."""

    config_schema = {'threshold': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'verdict': 'string'}

    def update(self, state):
        return {'verdict': 'pass' if state['level'] >= self.config['threshold']
                else 'fail'}


def _study_core():
    core = allocate_core()
    core.register_link('ReportCard', ReportCard)
    return core


def _study_region(threshold=2.0):
    """A study: a model **site**, a fixed report step wired to it, stores."""
    return {
        'level': 1.0,
        'verdict': 'string',
        'model': {'_type': 'site', '_sort': MODEL_FACE},
        'report': {
            '_type': 'step', 'address': 'local:ReportCard',
            'config': {'threshold': threshold},
            'inputs': {'level': ['level']},
            'outputs': {'verdict': ['verdict']}}}


def _model(core, rate=0.5):
    return core.access({
        '_type': 'process', 'address': 'local:IncreaseProcess',
        'config': {'rate': rate}, 'interval': 1.0,
        'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}})


def _nonconforming(core):
    """Right kind of thing, wrong face — outputs `mass`, not `level`."""
    return core.access({
        '_type': 'link', '_inputs': {'level': 'float'},
        '_outputs': {'mass': 'float'}})


def test_a_study_template_is_not_ground():
    from process_bigraph.templates import is_ground_document, open_sites

    core = _study_core()
    template = core.access({'study': _study_region()})

    assert not is_ground_document(template)
    assert open_sites(template) == [('study', 'model')]


def test_filling_the_model_site_yields_a_running_study():
    """The whole point: drop a conforming composite into the model site and
    the study builds, constructs and runs — no code."""
    from process_bigraph.templates import template_document, is_ground_document

    core = _study_core()
    template = core.access({'study': _study_region(threshold=2.0)})

    document = template_document(core, template, {'study/model': _model(core)})

    sim = Composite({'state': document}, core=core)
    assert list(sim.process_paths) == [('study', 'model')]
    assert list(sim.step_paths) == [('study', 'report')]

    sim.run(4.0)
    assert sim.state['study']['level'] > 2.0
    assert sim.state['study']['verdict'] == 'pass'


def test_the_same_template_takes_a_different_model():
    """A template is reusable: the site accepts any conforming model, and the
    study's verdict follows the model that was dropped in."""
    from process_bigraph.templates import template_document

    core = _study_core()
    template = core.access({'study': _study_region(threshold=2.0)})

    slow = Composite(
        {'state': template_document(
            core, template, {'study/model': _model(core, rate=0.01)})},
        core=core)
    slow.run(4.0)

    assert slow.state['study']['verdict'] == 'fail'


def test_a_non_conforming_model_is_refused():
    """The site is face-sorted, so only a model with the study's expected
    interface fits — the error names the site and the missing port."""
    from process_bigraph.templates import template_document

    core = _study_core()
    template = core.access({'study': _study_region()})

    with pytest.raises(ValueError, match="site 'study/model'") as raised:
        template_document(core, template, {'study/model': _nonconforming(core)})

    assert 'level' in str(raised.value)


def test_an_unfilled_study_site_is_refused_before_construction():
    from process_bigraph.templates import template_document

    core = _study_core()
    template = core.access({'study': _study_region()})

    with pytest.raises(ValueError, match='not ground') as raised:
        template_document(core, template, {})
    assert "'study/model'" in str(raised.value)


def test_an_unfilled_study_site_is_refused_by_the_composite_too():
    """Belt and braces: even handed straight to `Composite`, an open site is
    rejected at construction (Part A's A0)."""
    core = _study_core()
    template = core.access({'study': _study_region()})

    with pytest.raises(ValueError, match='not ground'):
        Composite({'state': core.render(template)}, core=core)


# ---- investigations: gating as filling --------------------------------

def _investigation_template(core, threshold=2.0):
    return core.access({'investigation': {
        'study_A': _study_region(threshold),
        'study_B': _study_region(threshold)}})


def test_an_investigation_template_has_a_site_per_member():
    from process_bigraph.templates import open_sites

    core = _study_core()
    template = _investigation_template(core)

    assert sorted(open_sites(template)) == [
        ('investigation', 'study_A', 'model'),
        ('investigation', 'study_B', 'model')]


def test_filling_admits_a_member_and_leaving_open_blocks_it():
    """Gating expressed as filling: the member whose site was filled is built
    and runs; the one left open is absent from the document entirely, so the
    engine never has to decide not to run it."""
    from process_bigraph.templates import investigation_document

    core = _study_core()
    template = _investigation_template(core)

    document, blocked = investigation_document(
        core, template,
        {'investigation/study_A/model': _model(core)},
        member_depth=2)

    assert blocked == ['investigation/study_B']
    assert sorted(document['investigation']) == ['study_A']

    sim = Composite({'state': document}, core=core)
    sim.run(4.0)
    assert 'study_B' not in sim.state['investigation']
    assert sim.state['investigation']['study_A']['verdict'] == 'pass'


def test_filling_every_site_builds_the_whole_investigation():
    from process_bigraph.templates import investigation_document

    core = _study_core()
    template = _investigation_template(core)

    document, blocked = investigation_document(
        core, template,
        {'investigation/study_A/model': _model(core),
         'investigation/study_B/model': _model(core)},
        member_depth=2)

    assert blocked == []
    sim = Composite({'state': document}, core=core)
    sim.run(4.0)
    assert sorted(sim.state['investigation']) == ['study_A', 'study_B']


def test_a_gate_edge_decides_the_downstream_filler():
    """The end-to-end gate: an upstream study runs, its report card is a real
    edge in the composite, and its verdict decides whether the downstream
    study's site gets filled.

    The gate runs *between* builds, not inside one: an open site is a
    construction-time condition, so the downstream document is assembled once
    the upstream verdict exists. Filling remains the mechanism — a `fail`
    simply never produces the filler.
    """
    from process_bigraph.templates import (
        template_document, investigation_document)

    def run_gated(rate):
        core = _study_core()

        # -- stage 1: the upstream study, with its gate edge (the report card)
        upstream = core.access({'study': _study_region(threshold=2.0)})
        stage_one = Composite(
            {'state': template_document(
                core, upstream, {'study/model': _model(core, rate=rate)})},
            core=core)
        stage_one.run(4.0)
        verdict = stage_one.state['study']['verdict']

        # -- gating IS filling: a pass emits the downstream filler, a fail
        #    emits nothing, so the site stays open.
        template = _investigation_template(core)
        bindings = {'investigation/study_A/model': _model(core, rate=rate)}
        if verdict == 'pass':
            bindings['investigation/study_B/model'] = _model(core, rate=rate)

        document, blocked = investigation_document(
            core, template, bindings, member_depth=2)
        downstream = Composite({'state': document}, core=core)
        downstream.run(4.0)
        return verdict, blocked, sorted(downstream.state['investigation'])

    verdict, blocked, members = run_gated(rate=0.5)
    assert verdict == 'pass'
    assert blocked == []
    assert members == ['study_A', 'study_B']

    verdict, blocked, members = run_gated(rate=0.01)
    assert verdict == 'fail'
    assert blocked == ['investigation/study_B']
    assert members == ['study_A']


def test_a_partially_filled_template_is_still_a_template():
    """Filling is incremental — a template with one of two sites filled is a
    smaller template, not an error."""
    from process_bigraph.templates import fill_template, open_sites

    core = _study_core()
    template = _investigation_template(core)

    partial = fill_template(
        core, template, {'investigation/study_A/model': _model(core)})

    assert open_sites(partial) == [('investigation', 'study_B', 'model')]

    whole = fill_template(
        core, partial, {'investigation/study_B/model': _model(core)})
    assert open_sites(whole) == []


# ==========================================
# Layer 3 (cont.) — the emitter and the flush entities are sites too
# ==========================================
#
# The study is a higher-order DAG: the simulation is ONE node
# (`local:SimulationStep`) and the flush entities are ordinary steps
# downstream of its `results` handle. So a report card observes the
# *resolved trajectory* of a finished run and fires exactly once — not an
# instantaneous store value once per tick.
#
# The model and emitter sites live inside that node's inner state: filling
# them configures the simulation the study runs.

CARD_FACE = {
    '_type': 'link',
    '_inputs': {'results': 'node'},
    '_outputs': {'verdict': 'string'}}

CARD_FIRINGS = []


class TrajectoryCard(Step):
    """A report card over a finished run.

    It consumes the emitter's handle, resolves it, and judges the run's
    *final* level — something no per-tick step could do.
    """

    config_schema = {'threshold': 'float'}

    def inputs(self):
        return {'results': 'node'}

    def outputs(self):
        return {'verdict': 'string'}

    def update(self, state):
        rows = state['results'].resolve()
        CARD_FIRINGS.append(len(rows))
        return {'verdict': 'pass'
                if rows[-1]['level'] >= self.config['threshold'] else 'fail'}


def _admits_emitter(core, site, filler):
    """A sort for "any emitter".

    Emitters are interchangeable precisely because they share no
    distinguishing face — `Emitter.outputs()` declares only `results` — so
    face conformance cannot pick them out. A registered sort states it
    directly: the address must name a registered `Emitter`.
    """
    from bigraph_schema.schema import normalize_address

    canonical = normalize_address(filler)
    if not isinstance(canonical, dict):
        return False
    edge_class = core.link_registry.get(canonical.get('data'))
    return isinstance(edge_class, type) and issubclass(edge_class, Emitter)


def _flush_core():
    core = _study_core()
    core.register_sort('emitter', _admits_emitter)
    core.register_link('TrajectoryCard', TrajectoryCard)
    return core


def _simulated_study(runtime=4.0):
    """The simulation the study runs: a model site and an emitter address
    site, wired so the emitter's `results` leaves the node."""
    return {
        'level': 1.0,
        'model': {'_type': 'site', '_sort': MODEL_FACE},
        'emitter': {
            '_type': 'step',
            'address': {'_type': 'site', '_sort': 'emitter'},
            'config': {'emit': {'level': 'node', 'global_time': 'node'}},
            'inputs': {'level': ['level'], 'global_time': ['global_time']},
            'outputs': {'results': ['results']}}}


def _flush_template(core, runtime=4.0):
    """A study whose simulation, emitter and report cards are ALL sites."""
    return core.access({'study': {
        'threshold': {'_type': 'site', '_sort': 'float'},
        'sim': {
            '_type': 'step',
            'address': 'local:SimulationStep',
            'config': {'state': _simulated_study(), 'runtime': runtime},
            'inputs': {},
            'outputs': {'results': ['results']}},
        'report_cards': {
            '_control': 'replicate', '_count': 1,
            'verdict': 'string',
            'card': {'_type': 'site', '_sort': CARD_FACE}}}})


MODEL_SITE = 'study/sim/config/state/model'
EMITTER_SITE = 'study/sim/config/state/emitter/address'


def _card(core, threshold):
    """A report card, wired to the simulation's results rather than to a
    store it would otherwise read mid-run."""
    return core.access({
        '_type': 'step', 'address': 'local:TrajectoryCard',
        'config': {'threshold': threshold},
        'inputs': {'results': ['..', 'results']},
        'outputs': {'verdict': ['verdict']}})


def test_the_flush_network_is_made_of_sites():
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import open_sites

    core = _flush_core()
    template = _flush_template(core)

    assert sorted(open_sites(template)) == [
        ('study', 'report_cards', 'card'),
        ('study', 'sim', 'config', 'state', 'emitter', 'address'),
        ('study', 'sim', 'config', 'state', 'model'),
        ('study', 'threshold')]

    # the cardinality region expands into one site per instance
    expanded = replicate(template, {'report_cards': 3})
    assert sorted(
        p for p in open_sites(expanded) if p[1].startswith('report_cards')) == [
        ('study', 'report_cards_0', 'card'),
        ('study', 'report_cards_1', 'card'),
        ('study', 'report_cards_2', 'card')]


def test_each_report_card_fires_once_on_the_resolved_trajectory():
    """The point of wiring the cards to `results` rather than to a store:
    each fires exactly ONCE, on the finished run, seeing every recorded row
    — not once per tick on an instantaneous value."""
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import template_document

    CARD_FIRINGS.clear()
    core = _flush_core()
    expanded = replicate(_flush_template(core), {'report_cards': 2})

    document = template_document(core, expanded, {
        MODEL_SITE: _model(core),
        EMITTER_SITE: 'local:RAMEmitter',
        'study/threshold': 2.0,
        'study/report_cards_0/card': _card(core, 2.0),
        'study/report_cards_1/card': _card(core, 99.0)})

    study = Composite({'state': document}, core=core)
    study.run(0.0)

    # two cards, one firing each — and each saw the WHOLE trajectory
    assert len(CARD_FIRINGS) == 2
    assert len(set(CARD_FIRINGS)) == 1 and CARD_FIRINGS[0] > 1

    assert study.state['study']['threshold'] == 2.0      # value site survived
    assert study.state['study']['report_cards_0']['verdict'] == 'pass'
    assert study.state['study']['report_cards_1']['verdict'] == 'fail'


def test_the_card_count_is_independent_of_simulation_length():
    """A longer simulation does not fire the cards more — that is what
    "the simulation is one node" buys."""
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import template_document

    for runtime in (2.0, 8.0):
        CARD_FIRINGS.clear()
        core = _flush_core()
        expanded = replicate(
            _flush_template(core, runtime=runtime), {'report_cards': 1})

        document = template_document(core, expanded, {
            MODEL_SITE: _model(core),
            EMITTER_SITE: 'local:RAMEmitter',
            'study/threshold': 2.0,
            'study/report_cards_0/card': _card(core, 2.0)})

        Composite({'state': document}, core=core).run(0.0)
        assert len(CARD_FIRINGS) == 1, (
            f'runtime={runtime} fired {len(CARD_FIRINGS)} times')


def test_the_emitter_site_is_interchangeable():
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import template_document

    for address in ('local:RAMEmitter', 'local:ConsoleEmitter'):
        core = _flush_core()
        expanded = replicate(_flush_template(core), {'report_cards': 0})
        document = template_document(core, expanded, {
            MODEL_SITE: _model(core),
            EMITTER_SITE: address,
            'study/threshold': 2.0})

        study = Composite({'state': document}, core=core)
        assert list(study.step_paths) == [('study', 'sim')]
        study.run(0.0)


def test_the_emitter_site_refuses_a_non_emitter():
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import template_document

    core = _flush_core()
    expanded = replicate(_flush_template(core), {'report_cards': 0})

    with pytest.raises(ValueError, match=f"site '{EMITTER_SITE}'"):
        template_document(core, expanded, {
            MODEL_SITE: _model(core),
            EMITTER_SITE: 'local:IncreaseProcess',
            'study/threshold': 2.0})


def test_a_report_card_site_refuses_a_non_report_card():
    """The card site is face-sorted, so only something that turns results
    into a verdict fits."""
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import template_document

    core = _flush_core()
    expanded = replicate(_flush_template(core), {'report_cards': 1})

    with pytest.raises(ValueError,
                       match="site 'study/report_cards_0/card'") as raised:
        template_document(core, expanded, {
            MODEL_SITE: _model(core),
            EMITTER_SITE: 'local:RAMEmitter',
            'study/threshold': 2.0,
            'study/report_cards_0/card': _model(core)})   # a model, not a card

    assert 'results' in str(raised.value)


def test_zero_flush_entities_is_a_valid_study():
    """A study that reports nothing is still a study — the cardinality
    region simply expands to nothing."""
    from bigraph_schema.assembly import replicate
    from process_bigraph.templates import template_document

    core = _flush_core()
    expanded = replicate(_flush_template(core), {'report_cards': 0})

    document = template_document(core, expanded, {
        MODEL_SITE: _model(core),
        EMITTER_SITE: 'local:RAMEmitter',
        'study/threshold': 2.0})

    assert not [k for k in document['study'] if k.startswith('report_cards')]
    study = Composite({'state': document}, core=core)
    study.run(0.0)
    # the simulation still ran inside its node
    assert study.state['study']['sim']['instance'].composite.state['level'] > 1.0


# ==========================================
# Layer 3 (cont.) — the comparison-harness template (local stand-ins)
# ==========================================

class Compare(Step):
    """The harness's verdict: do two models agree within a tolerance?"""

    config_schema = {'tolerance': 'float'}

    def inputs(self):
        return {'a': 'float', 'b': 'float'}

    def outputs(self):
        return {'verdict': 'string', 'difference': 'float'}

    def update(self, state):
        difference = abs(state['a'] - state['b'])
        return {
            'difference': difference,
            'verdict': 'agree' if difference <= self.config['tolerance']
            else 'diverge'}


WHOLE_CELL_FACE = {
    '_type': 'link',
    '_inputs': {'level': 'float'},
    '_outputs': {'level': 'float'}}


def _harness_core():
    """The harness's registry: two *local* stand-ins for the compared models.

    `reference` stands in for an out-of-tree implementation that a `git:`
    address would name once that protocol exists; nothing here fetches or
    executes foreign code.
    """
    core = _study_core()
    core.register_link('Compare', Compare)
    core.register_link('ReferenceModel', IncreaseProcess)
    return core


def _harness_template(core):
    """The §5 comparison-harness structure, with local stand-ins.

    - `compared/*` — value sites, the configs a sweep varies
    - `candidate`  — a model site: the in-tree model
    - `reference`  — an **address site**: fixed face, open implementation
    - `compare`    — fixed
    """
    return core.access({'comparison': {
        'compared': {
            'tolerance': {'_type': 'site', '_sort': 'float'},
            'duration': {'_type': 'site', '_sort': 'float'}},
        'level_a': 1.0,
        'level_b': 1.0,
        'verdict': 'string',
        'difference': 0.0,
        'candidate': {'_type': 'site', '_sort': WHOLE_CELL_FACE},
        'reference': {
            '_type': 'process',
            'address': {'_type': 'site', '_sort': WHOLE_CELL_FACE},
            'config': {'rate': 0.5},
            'interval': 1.0,
            'inputs': {'level': ['level_b']},
            'outputs': {'level': ['level_b']}},
        'compare': {
            '_type': 'step', 'address': 'local:Compare',
            'config': {'tolerance': 0.001},
            'inputs': {'a': ['level_a'], 'b': ['level_b']},
            'outputs': {'verdict': ['verdict'],
                        'difference': ['difference']}}}})


def _candidate(core, rate):
    return core.access({
        '_type': 'process', 'address': 'local:IncreaseProcess',
        'config': {'rate': rate}, 'interval': 1.0,
        'inputs': {'level': ['level_a']}, 'outputs': {'level': ['level_a']}})


def test_the_comparison_harness_is_a_template():
    from process_bigraph.templates import open_sites

    core = _harness_core()
    assert sorted(open_sites(_harness_template(core))) == [
        ('comparison', 'candidate'),
        ('comparison', 'compared', 'duration'),
        ('comparison', 'compared', 'tolerance'),
        ('comparison', 'reference', 'address')]


def test_the_harness_builds_and_runs_with_both_models_filled():
    """Fill the in-tree model, inject the reference implementation by
    address, set the compared configs → one document that runs both models
    side by side and compares them."""
    from process_bigraph.templates import template_document

    core = _harness_core()
    document = template_document(core, _harness_template(core), {
        'comparison/candidate': _candidate(core, rate=0.5),
        'comparison/reference/address': 'local:ReferenceModel',
        'comparison/compared/tolerance': 0.001,
        'comparison/compared/duration': 4.0})

    sim = Composite({'state': document}, core=core)
    assert sorted('/'.join(p) for p in sim.process_paths) == [
        'comparison/candidate', 'comparison/reference']

    sim.run(4.0)

    # identical models, so they agree
    assert sim.state['comparison']['verdict'] == 'agree'
    assert sim.state['comparison']['difference'] < 0.001
    assert sim.state['comparison']['compared']['tolerance'] == 0.001


def test_the_harness_detects_a_divergent_candidate():
    """The same template, a different candidate — the harness is reusable
    because the models are sites."""
    from process_bigraph.templates import template_document

    core = _harness_core()
    document = template_document(core, _harness_template(core), {
        'comparison/candidate': _candidate(core, rate=0.9),
        'comparison/reference/address': 'local:ReferenceModel',
        'comparison/compared/tolerance': 0.001,
        'comparison/compared/duration': 4.0})

    sim = Composite({'state': document}, core=core)
    sim.run(4.0)

    assert sim.state['comparison']['verdict'] == 'diverge'
    assert sim.state['comparison']['difference'] > 0.001


def test_the_reference_address_site_refuses_a_non_conforming_implementation():
    """The address site pins the face, so only an implementation with the
    whole-cell interface can be injected — the check that a `git:` address
    would run against a fetched repo."""
    from process_bigraph.templates import template_document

    core = _harness_core()
    core.register_link('WrongShape', ReportCard)

    with pytest.raises(ValueError,
                       match="site 'comparison/reference/address'"):
        template_document(core, _harness_template(core), {
            'comparison/candidate': _candidate(core, rate=0.5),
            'comparison/reference/address': 'local:WrongShape',
            'comparison/compared/tolerance': 0.001,
            'comparison/compared/duration': 4.0})


def test_the_harness_gates_a_downstream_member():
    """The harness as an investigation member: a downstream study runs only
    when the comparison agrees — gating by filling, staged."""
    from process_bigraph.templates import (
        template_document, investigation_document)

    def run(candidate_rate):
        core = _harness_core()

        harness = Composite(
            {'state': template_document(core, _harness_template(core), {
                'comparison/candidate': _candidate(core, rate=candidate_rate),
                'comparison/reference/address': 'local:ReferenceModel',
                'comparison/compared/tolerance': 0.001,
                'comparison/compared/duration': 4.0})},
            core=core)
        harness.run(4.0)
        verdict = harness.state['comparison']['verdict']

        followup = core.access({'investigation': {
            'downstream': _study_region(threshold=2.0)}})
        bindings = {}
        if verdict == 'agree':
            bindings['investigation/downstream/model'] = _model(core)

        document, blocked = investigation_document(
            core, followup, bindings, member_depth=2)
        members = document.get('investigation')
        # pruning every member leaves an empty region, which renders as the
        # bare type `'node'` rather than `{}`
        members = sorted(members) if isinstance(members, dict) else []
        return verdict, blocked, members

    assert run(0.5) == ('agree', [], ['downstream'])
    assert run(0.9) == ('diverge', ['investigation/downstream'], [])
