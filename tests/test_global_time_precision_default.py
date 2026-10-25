"""A plain Composite derives a global_time_precision default (#266).

CompositeSpec already derives a global_time_precision from a document's edge
intervals (#227). A document handed straight to Composite(...) — the usual path
for @composite_generator output — got no such default, so float accumulation
deferred the last scheduled update: 100 steps of 0.02 reach 2.0000000000000013,
which is > 2.0, so the 100th update slipped to the next run.
"""
from process_bigraph import Process, Composite, allocate_core


class _Ticker(Process):
    config_schema = {"interval": {"_type": "float", "_default": 0.02}}

    def interval(self):
        return self.config["interval"]

    def inputs(self):
        return {}

    def outputs(self):
        return {"n": "float"}

    def update(self, state, interval):
        return {"n": 1.0}


def _build(interval=0.02, **top):
    core = allocate_core()
    core.register_link("Ticker", _Ticker)
    return Composite(
        {"state": {"t": {"_type": "process", "address": "local:Ticker",
                         "interval": interval, "outputs": {"n": ["n"]}}},
         **top},
        core=core)


def test_plain_composite_applies_every_update_without_float_drift():  # #266
    sim = _build()
    assert sim.global_time_precision == 2       # derived from interval 0.02
    sim.run(2.0)
    assert sim.state["n"] == 100.0              # was 99.0 before the fix


def test_explicit_global_time_precision_still_wins():
    sim = _build(global_time_precision=9)
    assert sim.global_time_precision == 9       # explicit config is honored
    sim.run(2.0)
    assert sim.state["n"] == 100.0


def test_mixed_interval_grids_leave_precision_unset():
    """Two processes on different decimal grids must not be rounded to a grid
    that does not fit both — the derived precision stays None."""
    core = allocate_core()
    core.register_link("Ticker", _Ticker)
    sim = Composite({"state": {
        "a": {"_type": "process", "address": "local:Ticker",
              "interval": 0.02, "outputs": {"n": ["na"]}},
        "b": {"_type": "process", "address": "local:Ticker",
              "interval": 0.1, "outputs": {"n": ["nb"]}},
    }}, core=core)
    assert sim.global_time_precision is None
