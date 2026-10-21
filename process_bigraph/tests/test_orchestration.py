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
