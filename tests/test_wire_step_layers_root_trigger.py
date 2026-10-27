"""wire_step_layers must wire the root-trigger input port itself.

A layer-0 step whose own ports do not include `global_time` used to fire once
at init and never again, because `_triggers={'global_time': ...}` only matches
a step's *input ports*.
"""

import inspect
import sys

import pytest

from bigraph_schema import allocate_core
from process_bigraph.composite import Process, Step, Composite
from process_bigraph import wire_step_layers


class Tick(Process):
    def inputs(self):
        return {}

    def outputs(self):
        return {'t': 'float'}

    def update(self, state, interval):
        return {'t': interval}


class CountNoClock(Step):
    """Reads/writes a domain store only (no global_time port)."""

    def inputs(self):
        return {'n': 'integer'}

    def outputs(self):
        return {'n': 'integer'}

    def update(self, state):
        return {'n': 1}


class CountWithClock(CountNoClock):
    def inputs(self):
        return {'n': 'integer', 'global_time': 'float'}


@pytest.fixture
def core():
    return allocate_core(top=dict(inspect.getmembers(sys.modules[__name__])))


def _build(core, cls, hand_wired_clock=False, n_steps=1, parallel=False):
    state = {
        'n': 0,
        'm': 0,
        'tick': {'_type': 'process', 'address': 'local:Tick', 'config': {},
                 'interval': 1.0, 'inputs': {}, 'outputs': {'t': ['t']}},
        'count': {'_type': 'step', 'address': f'local:{cls.__name__}', 'config': {},
                  'inputs': {'n': ['n']}, 'outputs': {'n': ['n']}},
        'count2': {'_type': 'step', 'address': f'local:{cls.__name__}', 'config': {},
                   'inputs': {'n': ['m']}, 'outputs': {'n': ['m']}},
    }
    if hand_wired_clock:
        for k in ('count', 'count2'):
            state[k]['inputs']['global_time'] = ['global_time']
    wire_step_layers(state, {'count': [], 'count2': []})
    return Composite({'state': state, 'parallel_steps': parallel}, core=core)


def _run(core, cls, **kw):
    sim = _build(core, cls, **kw)
    sim.run(5.0)
    return sim.state['n'], sim.state['m']


def test_layer0_step_without_clock_port_refires_every_tick(core):
    n, m = _run(core, CountNoClock)
    # init fire + one per tick over 5 ticks; must advance well past 1
    assert n > 1 and m > 1
    assert (n, m) == _run(core, CountWithClock, hand_wired_clock=True)


def test_existing_clock_port_is_unperturbed(core):
    a = _run(core, CountWithClock, hand_wired_clock=True)
    # Hand-wired run is the pre-fix behaviour (port already present, so the
    # fix adds nothing); pin its exact values.
    assert a == (6, 6)


def test_parallel_steps_bit_identical(core):
    assert _run(core, CountNoClock, parallel=True) == _run(core, CountNoClock)


_SPANS = []


class SlowNoClock(CountNoClock):
    def update(self, state):
        import time
        t0 = time.perf_counter()
        time.sleep(0.05)
        _SPANS.append((t0, time.perf_counter()))
        return {'n': 1}


def test_parallel_steps_overlap_and_refire(core):
    _SPANS.clear()
    n, m = _run(core, SlowNoClock, parallel=True)
    par = (n, m)
    spans = sorted(_SPANS)
    assert n > 1 and m > 1
    assert any(b[0] < a[1] for a, b in zip(spans, spans[1:]))  # overlap
    _SPANS.clear()
    assert par == _run(core, SlowNoClock, parallel=False)
    spans = sorted(_SPANS)
    assert not any(b[0] < a[1] for a, b in zip(spans, spans[1:]))  # serial
