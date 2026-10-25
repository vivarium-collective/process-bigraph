"""Regression tests for the event-heap temporal scheduler (`scheduler` config).

The heap path must be BIT-IDENTICAL to the original 'scan' path while cutting
the O(P^2) multi-rate cost to ~O(P log P). These tests pin both: agreement
(heap == scan on state) and the perf guarantee (no wasted per-tick re-slicing
of horizon-overrun processes).
"""

import inspect
import sys

import pytest

from bigraph_schema import allocate_core
from process_bigraph.composite import Process, Composite


class Decay(Process):
    config_schema = {'rate': {'_type': 'float', '_default': '0.99'}}

    def inputs(self):
        return {'x': 'float'}

    def outputs(self):
        return {'x': 'float'}

    def update(self, state, interval):
        return {'x': state['x'] * self.config['rate']}


class MapChurn(Process):
    """Adds and removes `churn` agents per tick — exercises _add/_remove and
    the structural-subtree realize path under the heap."""
    config_schema = {'churn': {'_type': 'integer', '_default': '3'}}

    def inputs(self):
        return {'agents': {'_type': 'map', '_value': 'float'}}

    def outputs(self):
        return {'agents': {'_type': 'map', '_value': 'float'}}

    def update(self, state, interval):
        ids = list(state['agents'].keys())
        c = min(self.config['churn'], len(ids))
        self._n = getattr(self, '_n', 0)
        add = []
        for _ in range(c):
            self._n += 1
            add.append((f'z{self._n}', 1.0))
        return {'agents': {'_remove': ids[:c], '_add': add}}


@pytest.fixture
def core():
    return allocate_core(top=dict(inspect.getmembers(sys.modules[__name__])))


def _timescales(core, k, distinct, scheduler):
    state = {}
    for i in range(k):
        state[f'x_{i}'] = 1.0
        state[f'p_{i}'] = {
            '_type': 'process', 'address': 'local:Decay',
            'config': {'rate': 0.99},
            'interval': 1.0 + (i * 0.017 if distinct else 0.0),
            'inputs': {'x': [f'x_{i}']}, 'outputs': {'x': [f'x_{i}']}}
    return Composite({'state': state, 'scheduler': scheduler}, core=core)


def _final(composite):
    return {k: v for k, v in composite.state.items() if isinstance(v, float)}


@pytest.mark.parametrize('distinct', [True, False])
def test_heap_matches_scan(core, distinct):
    scan = _timescales(core, 40, distinct, 'scan'); scan.run(15.0)
    heap = _timescales(core, 40, distinct, 'heap'); heap.run(15.0)
    assert _final(scan) == _final(heap)


def test_auto_uses_heap_only_when_multirate(core):
    multi = _timescales(core, 10, True, 'auto'); multi.run(5.0)
    single = _timescales(core, 10, False, 'auto'); single.run(5.0)
    assert multi._use_heap is True
    assert single._use_heap is False


def test_heap_avoids_wasted_reslicing(core):
    """On a multi-rate composite the heap must slice ~once per real firing,
    not O(P x ticks). Compare _cached_view call counts heap vs scan."""
    def count_slices(scheduler):
        c = _timescales(core, 60, True, scheduler)
        n = [0]; orig = c._cached_view
        c._cached_view = lambda *a, **k: (n.__setitem__(0, n[0] + 1), orig(*a, **k))[1]
        c.run(15.0)
        return n[0]
    heap_slices = count_slices('heap')
    scan_slices = count_slices('scan')
    # Heap removes the wasted horizon-overrun re-slices: a large fraction fewer.
    assert heap_slices < scan_slices / 3


def test_heap_matches_scan_structural(core):
    """Division/removal (_add/_remove) must agree between heap and scan."""
    def build(scheduler):
        agents = {f'a{i}': 1.0 for i in range(20)}
        return Composite({'state': {
            'agents': agents,
            'churn': {
                '_type': 'process', 'address': 'local:MapChurn',
                'config': {'churn': 3}, 'interval': 1.0,
                'inputs': {'agents': ['agents']},
                'outputs': {'agents': ['agents']}}},
            'scheduler': scheduler}, core=core)
    scan = build('scan'); scan.run(10.0)
    heap = build('heap'); heap.run(10.0)
    assert set(scan.state['agents']) == set(heap.state['agents'])
    assert len(heap.state['agents']) == 20


def test_invalid_scheduler_rejected(core):
    with pytest.raises(ValueError):
        Composite({'state': {}, 'scheduler': 'bogus'}, core=core)
