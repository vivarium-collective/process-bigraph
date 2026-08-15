"""Tests for ``WorkflowBackend`` / ``run_workflow`` / ``LocalRunner``.

Uses a toy ``@composite_generator`` (``wf_ramp_toy``) plus a toy producer
``Step``, both registered in this module — pbg's own test suite must not
import v2ecoli or any downstream workspace.
"""
import json
import os

import pytest
from bigraph_schema import allocate_core

from process_bigraph.composite import Composite, Process, Step
from process_bigraph.composite_generator import composite_generator
from process_bigraph.workflow.backend import (
    LocalRunner, RunResult, get_backend, register_backend, run_workflow,
)
from process_bigraph.workflow.tasks import CompositeTask


# ── toy generator (consumed by CompositeTask) ───────────────────────────

class _WFRamp(Process):
    """Advances ``level`` by ``rate`` per tick, starting at ``start``."""

    config_schema = {'rate': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        return {'level': self.config['rate'] * interval}


def _provision_wf_ramp(core):
    core.register_link('_WFRamp', _WFRamp)
    return core


@composite_generator(
    name='wf_ramp_toy',
    core_extensions=[_provision_wf_ramp],
    emitters=[{'address': 'local:JSONEmitter', 'config': {}}],
)
def wf_ramp_toy(rate=2.0, start=1.0):
    """A file-backed-emitter toy generator (passes the CompositeTask emitter guard)."""
    return {'state': {
        'level': start,
        'ramp': {
            '_type': 'process', 'address': 'local:_WFRamp',
            'config': {'rate': rate},
            'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}}}}


_IMPORT = ['process_bigraph.tests.test_workflow_backend']


@pytest.fixture(autouse=True)
def _ensure_wf_ramp_toy_registered():
    """A full-suite run can have another test module's autouse fixture
    (test_composite_generator.py's ``_clear_registry``) wipe the global
    composite-generator registry between tests. Re-apply the decorator
    here so ``wf_ramp_toy`` is registered before every test in this
    module, regardless of cross-file run order/pollution.
    """
    composite_generator(
        name='wf_ramp_toy',
        core_extensions=[_provision_wf_ramp],
        emitters=[{'address': 'local:JSONEmitter', 'config': {}}],
    )(wf_ramp_toy)


# ── toy producer Step (feeds CompositeTask's per_match scatter port) ────

class _WFProducer(Step):
    """Emits a fixed two-value scatter match-set."""

    def inputs(self):
        return {}

    def outputs(self):
        return {'starts': 'node'}

    def update(self, state):
        return {'starts': {'0': 1.0, '1': 2.0}}


class _ThrowingStep(Step):
    """A Step that always raises — exercises the failure path."""

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def update(self, state):
        raise RuntimeError('boom')


def _outer_core():
    core = allocate_core()
    core.register_link('_WFProducer', _WFProducer)
    core.register_link('_WFThrowing', _ThrowingStep)
    core.register_link('_WFTask', CompositeTask)
    return core


def _toy_workflow_composite(tmp_path):
    """producer(Step) -> CompositeTask(scatter over 2 values), bridged out."""
    task_config = {
        'generator': 'wf_ramp_toy',
        'import': _IMPORT,
        'overrides': {'rate': 3.0},
        'artifact_params': {},
        'scatter_param': 'start',
        'steps': 2.0,
        'provision': [],
        'allow_in_memory_emitter': True,
        'artifact_root': str(tmp_path / '.pbg' / 'artifacts'),
    }
    state = {
        'starts': {},
        'task_results': {},
        'producer': {
            '_type': 'step', 'address': 'local:_WFProducer', 'config': {},
            'inputs': {}, 'outputs': {'starts': ['starts']},
        },
        'task': {
            '_type': 'step', 'address': 'local:_WFTask', 'config': task_config,
            'inputs': {'start': ['starts']},
            'outputs': {'results': ['task_results']},
        },
    }
    document = {'state': state, 'bridge': {'outputs': {'results': ['task_results']}}}
    return Composite(document, core=_outer_core())


def _throwing_composite():
    state = {
        'boom': {
            '_type': 'step', 'address': 'local:_WFThrowing', 'config': {},
            'inputs': {}, 'outputs': {},
        },
    }
    return Composite({'state': state}, core=_outer_core())


# ── (a) success path: two per-scatter outputs read via read_bridge ──────

def test_run_workflow_local_success_returns_bridge_outputs(tmp_path):
    composite = _toy_workflow_composite(tmp_path)
    result = run_workflow(composite, backend='local', outdir=str(tmp_path))

    assert isinstance(result, RunResult)
    assert result.backend == 'local'
    assert result.status == 'ok'
    assert result.workdir == str(tmp_path)

    results = result.outputs.get('results')
    assert results is not None
    assert set(results) == {'0', '1'}

    # Cross-check against the composite's own read_bridge().
    bridge = composite.read_bridge()
    assert bridge == result.outputs


# ── (b) unknown backend raises ───────────────────────────────────────────

def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_backend('nope')


# ── (c) failure path: status='failed' with provenance['error'] ─────────

def test_run_workflow_local_failure_returns_error_provenance(tmp_path):
    composite = _throwing_composite()
    result = run_workflow(composite, backend='local', outdir=str(tmp_path))

    assert result.status == 'failed'
    assert result.outputs == {}
    assert 'error' in result.provenance
    assert 'boom' in result.provenance['error']


# ── (d) register_backend / get_backend round-trip ───────────────────────

def test_register_and_get_backend_round_trip():
    class _DummyBackend:
        name = 'dummy'

        def available(self):
            return True

        def run(self, composite, *, outdir, code_version=None, **opts):
            return RunResult('dummy', 'ok', {}, str(outdir), {})

    dummy = _DummyBackend()
    register_backend('dummy', dummy)
    try:
        assert get_backend('dummy') is dummy
    finally:
        from process_bigraph.workflow.backend import _BACKENDS
        _BACKENDS.pop('dummy', None)


def test_local_runner_registered_by_default():
    backend = get_backend('local')
    assert isinstance(backend, LocalRunner)
    assert backend.name == 'local'
    assert backend.available() is True
