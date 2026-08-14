"""Tests for ``CompositeTask`` — the per_match scatter + content-hash-cache Step.

Uses a toy ``@composite_generator`` (``ramp_toy``) registered in this module —
pbg's own test suite must not import v2ecoli or any downstream workspace.
"""
import json
import os
import subprocess
from unittest import mock

import pytest
from bigraph_schema import allocate_core

from process_bigraph.composite import Process
from process_bigraph.composite_generator import composite_generator
from process_bigraph.workflow.tasks import CompositeTask


# ── toy generator ────────────────────────────────────────────────────

class _Ramp(Process):
    """Advances ``level`` by ``rate`` per tick, starting at ``start``."""

    config_schema = {'rate': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        return {'level': self.config['rate'] * interval}


def _provision_ramp(core):
    core.register_link('_TaskRamp', _Ramp)
    return core


@composite_generator(
    name='ramp_toy_task',
    core_extensions=[_provision_ramp],
    emitters=[{'address': 'local:JSONEmitter', 'config': {}}],
)
def ramp_toy_task(rate=2.0, start=1.0):
    """A file-backed-emitter toy generator (passes the CompositeTask emitter guard)."""
    return {'state': {
        'level': start,
        'ramp': {
            '_type': 'process', 'address': 'local:_TaskRamp',
            'config': {'rate': rate},
            'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}}}}


@composite_generator(name='ramp_toy_ram', core_extensions=[_provision_ramp])
def ramp_toy_ram(rate=2.0, start=1.0):
    """No declared emitter → resolves to the framework's RAMEmitter default."""
    return {'state': {
        'level': start,
        'ramp': {
            '_type': 'process', 'address': 'local:_TaskRamp',
            'config': {'rate': rate},
            'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}}}}


_IMPORT = ['process_bigraph.tests.test_composite_task']


def _core():
    core = allocate_core()
    core.register_link('_TaskRamp', _Ramp)
    return core


def _task(tmp_path, **overrides):
    config = {
        'generator': 'ramp_toy_task',
        'import': _IMPORT,
        'overrides': {'rate': 3.0},
        'artifact_params': {},
        'scatter_param': 'start',
        'steps': 2.0,
        'provision': [],
        'allow_in_memory_emitter': True,
        'artifact_root': str(tmp_path / '.pbg' / 'artifacts'),
    }
    config.update(overrides)
    return CompositeTask(config, core=_core())


def _match_state():
    return {'start': {'0': 1.0, '1': 2.0}}


def _read_state(result_path):
    """Open a match's returned result and return its ``state`` dict.

    Every test generator here has no ``emitter_out_dir`` param, so the
    returned result is always the ``--state-out`` file (a directory would
    only happen for a generator that wires ``emitter_out_dir`` — see
    ``CompositeTask._run_match``).
    """
    assert os.path.isfile(result_path), f'expected a state file, got: {result_path!r}'
    with open(result_path) as fh:
        return json.load(fh)['state']


# ── (a) native scatter ───────────────────────────────────────────────

def test_scatter_returns_one_result_per_match(tmp_path):
    task = _task(tmp_path)  # overrides={'rate': 3.0}, steps=2.0
    update = task.invoke(_match_state()).get()
    results = update['results']
    assert set(results) == {'0', '1'}

    # Not just "non-empty paths" — each match must have actually run the
    # composite parameterized by ITS OWN scatter value: level = start +
    # rate * steps for this toy generator (rate=3.0, steps=2.0 here), so a
    # mixed-up or shared match would show up as the wrong level.
    level_0 = _read_state(results['0'])['level']
    level_1 = _read_state(results['1'])['level']
    assert level_0 == pytest.approx(1.0 + 3.0 * 2.0)  # start=1.0
    assert level_1 == pytest.approx(2.0 + 3.0 * 2.0)  # start=2.0
    assert level_0 != level_1


def test_declares_per_match_scatter_port():
    task = _task_no_run()
    schema = task.inputs()
    assert schema['start']['_cardinality'] == 'per_match'
    assert task.scatter_port() == 'start'


def _task_no_run():
    config = {
        'generator': 'ramp_toy_task', 'import': _IMPORT, 'overrides': {},
        'artifact_params': {'sim_data': 'sim_data'}, 'scatter_param': 'start',
        'steps': 1.0, 'provision': [], 'allow_in_memory_emitter': True,
    }
    return CompositeTask(config, core=_core())


def test_artifact_param_ports_declared():
    task = _task_no_run()
    schema = task.inputs()
    assert schema['sim_data'] == {'_type': 'string', '_is_file': True}
    assert task.outputs() == {'results': 'node'}


# ── (b) cache hit — zero subprocess launches ────────────────────────

def test_cache_hit_launches_zero_subprocesses(tmp_path):
    task = _task(tmp_path)
    state = _match_state()

    first = task.invoke(state).get()['results']

    with mock.patch(
            'process_bigraph.workflow.tasks.subprocess.run',
            wraps=subprocess.run) as spy:
        second = task.invoke(state).get()['results']
        assert spy.call_count == 0, (
            f'expected zero subprocess launches on cache hit, got {spy.call_count}')

    assert second == first

    node_dir = task._workdir_root() if hasattr(task, '_workdir_root') else None
    # provenance.json (F5) records the cache hit explicitly
    prov_path = task._provenance_path()
    provenance = json.loads(prov_path.read_text() if hasattr(prov_path, 'read_text')
                            else open(prov_path).read())
    assert provenance['0']['cache_hit'] is True
    assert provenance['1']['cache_hit'] is True


# ── (c) cache miss on changed steps ─────────────────────────────────

def test_changed_steps_misses_cache_and_reruns(tmp_path):
    task_a = _task(tmp_path, steps=2.0)
    state = _match_state()
    task_a.invoke(state)
    prov_a = json.loads(open(task_a._provenance_path()).read())

    task_b = _task(tmp_path, steps=3.0)
    with mock.patch(
            'process_bigraph.workflow.tasks.subprocess.run',
            wraps=subprocess.run) as spy:
        task_b.invoke(state)
        assert spy.call_count == 2, (
            f'expected a real subprocess per match on a steps-changed cache miss, '
            f'got {spy.call_count}')
    prov_b = json.loads(open(task_b._provenance_path()).read())

    assert prov_a['0']['address'] != prov_b['0']['address']
    assert prov_b['0']['cache_hit'] is False


# ── regression: cache payload must be address-keyed, not scatter-value-keyed ──
#
# Sweep-then-revert is the most common real pattern: run at steps=2, run
# again at steps=3 (a different address, correctly a miss), then run again
# at steps=2. If the payload a match writes is keyed only by its scatter
# value (e.g. a `seed_<val>/` directory) rather than by its cache address,
# the steps=3 run overwrites steps=2's on-disk payload at that shared
# val-keyed location. The third call then sees address A's marker (still
# present — it was never invalidated) sitting next to the steps=3 payload
# and wrongly reports a cache hit carrying the WRONG (steps=3) data. The fix
# is for both the cache-hit check and the returned result path to read from
# the address-keyed artifact-store directory, never the scatter-value path.

def test_cache_payload_is_address_keyed_not_scatter_val_keyed(tmp_path):
    state = {'start': {'0': 1.0}}

    task_2a = _task(tmp_path, steps=2.0)
    first = task_2a.invoke(state).get()['results']
    first_level = _read_state(first['0'])['level']
    assert first_level == pytest.approx(1.0 + 3.0 * 2.0)  # 7.0

    task_3 = _task(tmp_path, steps=3.0)  # different address — correct miss
    third_party = task_3.invoke(state).get()['results']
    other_level = _read_state(third_party['0'])['level']
    assert other_level == pytest.approx(1.0 + 3.0 * 3.0)  # 10.0
    assert other_level != first_level

    task_2b = _task(tmp_path, steps=2.0)  # back to the FIRST address
    with mock.patch(
            'process_bigraph.workflow.tasks.subprocess.run',
            wraps=subprocess.run) as spy:
        third = task_2b.invoke(state).get()['results']
        assert spy.call_count == 0, (
            f'expected a cache hit (zero subprocess launches) reverting to '
            f'an already-computed address, got {spy.call_count}')

    third_level = _read_state(third['0'])['level']
    assert third_level == pytest.approx(first_level), (
        f'cache hit returned the WRONG payload: expected the steps=2 result '
        f'({first_level}) but got {third_level} — the payload was keyed by '
        f'scatter value instead of by cache address')
    assert third_level != pytest.approx(other_level)

    prov = json.loads(open(task_2b._provenance_path()).read())
    assert prov['0']['cache_hit'] is True


# ── duplicate scatter value in one match-set shares an address safely ──

def test_duplicate_scatter_value_shares_address_without_racing(tmp_path):
    task = _task(tmp_path, steps=2.0)
    # Two distinct match keys resolving to the SAME address (same start,
    # same everything else) — must not race/tear the shared payload dir,
    # and both must resolve to the one real run.
    update = task.invoke({'start': {'0': 5.0, '1': 5.0}}).get()
    results = update['results']
    assert results['0'] == results['1']
    level_0 = _read_state(results['0'])['level']
    level_1 = _read_state(results['1'])['level']
    assert level_0 == level_1 == pytest.approx(5.0 + 3.0 * 2.0)


# ── (d) emitter guard ────────────────────────────────────────────────

def test_emitter_guard_raises_for_in_memory_emitter(tmp_path):
    config = {
        'generator': 'ramp_toy_ram', 'import': _IMPORT, 'overrides': {},
        'artifact_params': {}, 'scatter_param': 'start', 'steps': 1.0,
        'provision': [], 'allow_in_memory_emitter': False,
        'artifact_root': str(tmp_path / '.pbg' / 'artifacts'),
    }
    task = CompositeTask(config, core=_core())
    with pytest.raises(ValueError, match='(?i)in-memory|RAMEmitter'):
        task.invoke({'start': {'0': 1.0}})


def test_emitter_guard_allows_when_opted_in(tmp_path):
    config = {
        'generator': 'ramp_toy_ram', 'import': _IMPORT, 'overrides': {},
        'artifact_params': {}, 'scatter_param': 'start', 'steps': 1.0,
        'provision': [], 'allow_in_memory_emitter': True,
        'artifact_root': str(tmp_path / '.pbg' / 'artifacts'),
    }
    task = CompositeTask(config, core=_core())
    update = task.invoke({'start': {'0': 1.0}}).get()
    assert '0' in update['results']
