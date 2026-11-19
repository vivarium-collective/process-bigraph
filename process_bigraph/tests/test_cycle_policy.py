"""Issue #30 — opt-in ``cycle_policy`` to WARN instead of raise on a
non-deterministic step cycle.

``build_step_network`` rejects a zero-time step cycle whose members share a
priority (its evaluation order would otherwise be ambiguous — issue #233). That
hard raise is right as the default, but it blocks a document whose step wiring
genuinely forms such a cycle and that the author accepts (v2ecoli's division /
counts_deriver wiring, which cannot be re-wired without rewriting ~45 steps).

``cycle_policy`` lets a Composite opt into a warning instead of the raise:

  * ``'raise'`` (default) — today's behavior, unchanged.
  * ``'warn'``           — log a warning and build anyway; the cycle is then
                           ordered by the scheduler's priority/``repr`` fallback,
                           which is DETERMINISTIC across processes (#284/#285)
                           though arbitrary — distinct priorities still give a
                           meaningful order.

Resolution order: the Composite ``config['cycle_policy']`` wins; if unset, the
``PROCESS_BIGRAPH_CYCLE_POLICY`` env var; otherwise ``'raise'``.
"""
import warnings

import pytest

from process_bigraph import Composite, Step, allocate_core


class _A(Step):  # reads x, writes y
    config_schema = {}

    def inputs(self):
        return {"x": "float"}

    def outputs(self):
        return {"y": "float"}

    def update(self, state):
        return {"y": state["x"]}


class _B(Step):  # reads y, writes x -> mutual zero-time cycle, equal priority
    config_schema = {}

    def inputs(self):
        return {"y": "float"}

    def outputs(self):
        return {"x": "float"}

    def update(self, state):
        return {"x": state["y"]}


def _make(config_extra=None):
    core = allocate_core()
    core.register_link("A", _A)
    core.register_link("B", _B)
    state = {
        "x": 1.0, "y": 2.0,
        "a": {"_type": "step", "address": "local:A",
              "inputs": {"x": ["x"]}, "outputs": {"y": ["y"]}},
        "b": {"_type": "step", "address": "local:B",
              "inputs": {"y": ["y"]}, "outputs": {"x": ["x"]}},
    }
    config = {"state": state}
    if config_extra:
        config.update(config_extra)
    return Composite(config, core=core)


def test_priority_tie_cycle_raises_by_default():
    """Default is unchanged: the non-deterministic cycle is fatal."""
    with pytest.raises(ValueError, match="non-deterministic step dependency cycle"):
        _make()


def test_explicit_raise_policy_still_raises():
    """``cycle_policy='raise'`` is the explicit form of the default."""
    with pytest.raises(ValueError, match="non-deterministic step dependency cycle"):
        _make({"cycle_policy": "raise"})


def test_cycle_policy_warn_builds_and_only_warns():
    """``cycle_policy='warn'`` builds the Composite, emitting a warning that
    names the cycle instead of raising."""
    with pytest.warns(UserWarning, match="step dependency cycle"):
        sim = _make({"cycle_policy": "warn"})
    # it is a usable composite, not a half-built one
    assert sim.state["x"] is not None
    sim.run(0.0)  # a zero-time step pass must not raise


def test_env_var_enables_warn_when_config_is_unset(monkeypatch):
    """With no ``cycle_policy`` in config, ``PROCESS_BIGRAPH_CYCLE_POLICY=warn``
    selects warn mode."""
    monkeypatch.setenv("PROCESS_BIGRAPH_CYCLE_POLICY", "warn")
    with pytest.warns(UserWarning, match="step dependency cycle"):
        _make()


def test_config_wins_over_env(monkeypatch):
    """An explicit ``cycle_policy`` in config overrides the env var: config
    ``'raise'`` still raises even though the env says warn."""
    monkeypatch.setenv("PROCESS_BIGRAPH_CYCLE_POLICY", "warn")
    with pytest.raises(ValueError, match="non-deterministic step dependency cycle"):
        _make({"cycle_policy": "raise"})


def test_unknown_policy_is_fail_closed(monkeypatch):
    """A typo'd policy fails CLOSED (raises), never silently continues."""
    monkeypatch.delenv("PROCESS_BIGRAPH_CYCLE_POLICY", raising=False)
    with pytest.raises(ValueError, match="non-deterministic step dependency cycle"):
        _make({"cycle_policy": "wrn"})


def test_warn_mode_runs_deterministically_across_rebuilds():
    """In warn mode the cycle is ordered by the deterministic fallback, so two
    independently built composites reach the same state after a zero-time pass."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _make({"cycle_policy": "warn"})
        a.run(0.0)
        b = _make({"cycle_policy": "warn"})
        b.run(0.0)
    assert (a.state["x"], a.state["y"]) == (b.state["x"], b.state["y"])
