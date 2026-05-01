"""
=========================
Tests for Process Bigraph
=========================
"""

import os
import sys
import random
import inspect
import socket
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
    emitter_from_wires, gather_emitter_results, add_emitter_to_composite,
    SQLiteEmitter,
    save_simulation_metadata, mark_simulation_finished,
    list_simulations, load_history, load_simulation_metadata,
)
from process_bigraph.protocols.rest import rest_get, rest_post
from process_bigraph.types import ProcessLink, StepLink

from process_bigraph.processes.examples import IncreaseProcess
from process_bigraph.processes.growth_division import grow_divide_agent, Grow, Divide
from process_bigraph.processes.dynamic_structure import DynamicWorker


def _port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    """Return True if TCP connect succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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


def test_rest_process(core):
    host = "localhost"
    port = 22222

    if not _port_open(host, port):
        pytest.skip(f"REST server not running at {host}:{port} (skipping integration test)")

    state = {
        'mass': 1.0,
        'rest-process': {
            '_type': 'process',
            'address': {
                'protocol': 'rest',
                'data': {
                    'process': 'Grow',
                    'host': host,
                    'port': port}},
            'config': {
                'rate': 0.005},
            'inputs': {'mass': ['mass']},
            'outputs': {'mass': ['mass']},
            'interval': 0.7}}

    composite = Composite({'state': state}, core=core)
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

def test_sqlite_emitter(core, tmp_path=None):
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
            assert val > -3.0, \
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

    # Alice has two non-`_`-prefixed fields (mass, height) so the
    # Site `props` captures via the surplus path, binding them as a dict.
    # With only one field, _match_dict's 1-to-1 path captures the bare
    # value — see test_fire_rule_b3 in bigraph-schema for that case.
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
    assert lab['alice']['props']['mass'] == 70.0
    assert lab['alice']['props']['height'] == 1.7

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
    assert lab['alice']['props']['mass'] == 70.0
    assert lab['alice']['props']['height'] == 1.7


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
    test_rest_process(core)
    # test_dfba_process(core)

    test_reaction_step(core)
    test_reaction_step_in_composite(core)



    
