"""Regression tests for the streamline-core fixes:

- ``match_star_path`` must NOT prefix-match (length check).
- ``trigger_steps`` must NOT mutate the persistent ``step_triggers`` registry
  when a wildcard trigger also matches.
- the default emitter address (``local:RAMEmitter``) must actually resolve so
  ``generate_emitter_state`` / ``add_emitter_to_composite`` work on the default
  path (the old ``local:ram-emitter`` never resolved).
"""

from process_bigraph import allocate_core, Composite
from process_bigraph.composite import match_star_path
from process_bigraph.emitter import (
    RAMEmitter,
    add_emitter_to_composite,
    generate_emitter_state,
    gather_emitter_results,
)
from process_bigraph.processes.examples import IncreaseProcess


# ---------------------------------------------------------------------------
# match_star_path
# ---------------------------------------------------------------------------

def test_match_star_path_exact_and_wildcard():
    assert match_star_path(('cells', 'A', 'growth'), ('cells', '*', 'growth'))
    assert match_star_path(('cells', 'A'), ('cells', '*'))
    assert not match_star_path(('cells', 'A', 'death'), ('cells', '*', 'growth'))


def test_match_star_path_no_prefix_over_match():
    # The docstring's own example: a shorter path must NOT match a longer
    # star_path just because the prefix lines up (zip used to truncate).
    assert not match_star_path(('cells', 'A'), ('cells', '*', 'growth'))
    # ...and the reverse: a longer path must not match a shorter star_path.
    assert not match_star_path(('cells', 'A', 'growth'), ('cells', '*'))


# ---------------------------------------------------------------------------
# trigger_steps must not mutate the persistent registry
# ---------------------------------------------------------------------------

def _minimal_composite(core):
    core.register_link('IncreaseProcess', IncreaseProcess)
    return Composite({
        'state': {
            'increase': {
                'address': 'local:IncreaseProcess',
                'config': {'rate': 0.3},
                'interval': 1.0,
                'inputs': {'level': ['value']},
                'outputs': {'level': ['value']}},
            'value': 11.11}}, core=core)


def test_trigger_steps_does_not_mutate_registry():
    core = allocate_core()
    composite = _minimal_composite(core)

    # Controlled trigger registries: a direct trigger AND a wildcard trigger
    # both keyed on the same path.
    direct_step = ('step_direct',)
    star_step = ('step_star',)
    composite.step_triggers = {('a',): [direct_step]}
    composite.star_triggers = {('a',): [star_step]}

    # Isolate the mutation invariant from the run machinery.
    queued = []
    composite.initialize_steps = lambda steps: queued.extend(steps)
    composite.run_steps = lambda step_paths: None
    composite.to_run = []

    composite.trigger_steps([('a',)])

    # The persistent registry entry must be untouched (length 1), not grown
    # by the wildcard steps.
    assert composite.step_triggers[('a',)] == [direct_step]
    # ...and both the direct and wildcard steps were still queued to run.
    assert direct_step in queued
    assert star_step in queued

    # A second identical trigger must queue the SAME set (no unbounded growth
    # / spurious accumulation across ticks).
    queued.clear()
    composite.trigger_steps([('a',)])
    assert composite.step_triggers[('a',)] == [direct_step]
    assert queued.count(star_step) == 1


# ---------------------------------------------------------------------------
# default emitter address resolves
# ---------------------------------------------------------------------------

def test_default_emitter_address_resolves():
    core = allocate_core()
    composite = _minimal_composite(core)

    # generate_emitter_state on the default path must produce a node whose
    # address resolves to RAMEmitter (the registry keys emitters by class
    # name; 'local:ram-emitter' never resolved).
    emitter_state = generate_emitter_state(composite)
    assert emitter_state['address'] == 'local:RAMEmitter'

    # add_emitter_to_composite on the default path must build a real emitter.
    add_emitter_to_composite(composite, core)
    emitter = composite.state['emitter']['instance']
    assert isinstance(emitter, RAMEmitter)

    composite.update({}, 3.0)
    results = gather_emitter_results(composite)
    assert results  # the default emitter actually recorded a trajectory
