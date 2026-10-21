"""Pluggable orchestration registry (RFC step 1): the seam exists and defaults
to a behaviour-preserving ``temporal`` strategy."""
import pytest

from process_bigraph import Composite, allocate_core
from process_bigraph.orchestration import (
    OrchestrationStrategy, TemporalOrchestration,
    register_orchestration, get_orchestration)


def test_temporal_registered_by_default():
    assert get_orchestration("temporal") is TemporalOrchestration


def test_register_resolve_and_errors():
    class _Noop(OrchestrationStrategy):
        name = "noop-test"
        def advance(self, interval, force_complete, run_start):
            return None
    register_orchestration("noop-test", _Noop)
    assert get_orchestration("noop-test") is _Noop
    with pytest.raises(KeyError):
        get_orchestration("missing-xyz")
    with pytest.raises(TypeError):
        register_orchestration("bad", object)


def test_composite_defaults_to_temporal_and_runs_unchanged():
    sim = Composite({'state': {}}, core=allocate_core())
    assert sim.config.get('orchestration') == 'temporal'
    t0 = sim.state['global_time']
    sim.run(1.0)
    assert sim.state['global_time'] >= t0  # advanced via the temporal strategy


def test_custom_strategy_is_invoked_through_run():
    calls = []
    class _Recording(TemporalOrchestration):  # delegate to temporal; just record
        name = "recording-test"
        def advance(self, interval, force_complete, run_start):
            calls.append(interval)
            return super().advance(interval, force_complete, run_start)
    register_orchestration("recording-test", _Recording)
    sim = Composite({'state': {}, 'orchestration': 'recording-test'}, core=allocate_core())
    sim.run(1.0)
    assert calls == [1.0]
    assert isinstance(sim._orchestration_strategy(), _Recording)


# ── workflow strategy: pure step DAG, result-equivalent to temporal ──────────
from process_bigraph import Step
from process_bigraph.orchestration import WorkflowOrchestration


class _Producer(Step):
    def inputs(self): return {}
    def outputs(self): return {'out': 'float'}
    def update(self, state): return {'out': 2.0}


class _Doubler(Step):
    def inputs(self): return {'x': 'float'}
    def outputs(self): return {'y': 'float'}
    def update(self, state): return {'y': state['x'] * 10.0}


def _step_dag_core():
    core = allocate_core()
    core.register_link('_Producer', _Producer)
    core.register_link('_Doubler', _Doubler)
    return core


def _step_dag_doc(orchestration):
    return {
        'state': {
            'a': 0.0, 'b': 0.0,
            'producer': {'_type': 'step', 'address': 'local:_Producer', 'config': {},
                         'inputs': {}, 'outputs': {'out': ['a']}},
            'doubler': {'_type': 'step', 'address': 'local:_Doubler', 'config': {},
                        'inputs': {'x': ['a']}, 'outputs': {'y': ['b']}},
        },
        'orchestration': orchestration,
    }


def test_workflow_registered():
    assert get_orchestration("workflow") is WorkflowOrchestration


def test_workflow_runs_step_dag_result_equivalent_to_temporal():
    sim_t = Composite(_step_dag_doc('temporal'), core=_step_dag_core()); sim_t.run(1.0)
    sim_w = Composite(_step_dag_doc('workflow'), core=_step_dag_core()); sim_w.run(1.0)
    assert isinstance(sim_w._orchestration_strategy(), WorkflowOrchestration)
    # producer ran, then doubler consumed its output (DAG order) — identical result
    assert sim_w.state['a'] == sim_t.state['a'] == 2.0
    assert sim_w.state['b'] == sim_t.state['b'] == 20.0


def test_composite_exposes_primitive_interface():
    """The streamlined surface strategies compose: run_step_network + advance_tick."""
    sim = Composite({'state': {}}, core=allocate_core())
    assert callable(sim.run_step_network) and callable(sim.advance_tick)
    sim.run_step_network()
    halted, fc = sim.advance_tick(1.0, False)      # one tick toward end_time=1.0
    assert halted is False and fc is False
    assert sim.state['global_time'] == 1.0         # no processes -> jump to end_time
