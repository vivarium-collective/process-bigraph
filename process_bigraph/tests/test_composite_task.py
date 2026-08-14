"""Tests for ``CompositeTask`` — the per_match scatter + content-hash-cache Step.

Uses a toy ``@composite_generator`` (``ramp_toy``) registered in this module —
pbg's own test suite must not import v2ecoli or any downstream workspace.
"""
import json
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


# ── (a) native scatter ───────────────────────────────────────────────

def test_scatter_returns_one_result_per_match(tmp_path):
    task = _task(tmp_path)
    update = task.invoke(_match_state()).get()
    results = update['results']
    assert set(results) == {'0', '1'}
    for path in results.values():
        assert path  # non-empty result location for each match


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
