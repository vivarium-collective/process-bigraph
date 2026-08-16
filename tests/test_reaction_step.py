"""ReactionStep in a Composite: genuine structural rewrite, and the untyped-store
diagnostic warning."""
import warnings

import pytest

from process_bigraph import Composite, allocate_core
from bigraph_schema.assembly import ReactionRule

try:
    from bigraph_schema.assembly import Site
except Exception:  # pragma: no cover
    from bigraph_schema.schema import Site


def _division_rule():
    return ReactionRule(
        redex={'cell': {'_control': 'cell', 'contents': Site()}},
        reactum={'daughter_1': {'_control': 'cell', 'contents': Site()},
                 'daughter_2': {'_control': 'cell', 'contents': Site()}},
        instantiation={'contents': 'contents'},
        label='divide')


def _step_node(path):
    return {'_type': 'step', 'address': 'local:ReactionStep',
            'config': {'rules': [_division_rule()], 'mode': 'deterministic'},
            'inputs': {'state': path}, 'outputs': {'state': path}}


def _cells(colony):
    return {k for k, v in colony.items()
            if isinstance(v, dict) and v.get('_control') == 'cell'}


def test_typed_store_fires_a_structural_division():
    """A tree[node]-typed store: the rule fires inside a live Composite and CREATES
    the daughter nodes (one cell -> two)."""
    core = allocate_core()
    state = {
        'colony': {'_type': 'tree[node]',
                   'cell': {'_control': 'cell', 'contents': {'biomass': 1.0}}},
        'divider': _step_node(['colony']),
    }
    sim = Composite({'state': state}, core=core)
    assert sim.state['colony']['cell']['_control'] == 'cell'   # _control preserved
    with warnings.catch_warnings():
        warnings.simplefilter('error')                          # no warning on happy path
        sim.run(1)
    assert _cells(sim.state['colony']) == {'daughter_1', 'daughter_2'}
    assert 'cell' not in sim.state['colony']


def test_untyped_store_warns():
    """An UNTYPED store realizes to a plain dict (no _control); the reaction silently
    no-ops — ReactionStep now warns so this is diagnosable."""
    core = allocate_core()
    state = {
        'colony': {'cell': {'_control': 'cell', 'contents': {'biomass': 1.0}}},  # no _type
        'divider': _step_node(['colony']),
    }
    sim = Composite({'state': state}, core=core)
    with pytest.warns(UserWarning, match="UNTYPED"):
        sim.run(1)
