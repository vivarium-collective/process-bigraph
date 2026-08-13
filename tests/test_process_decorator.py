"""Tests for the ``@process`` / ``@step`` decorators (config_schema inferred
from typed keyword-only defaults).

These live in ``tests/`` and are fully self-contained (they allocate their own
core) so they need none of the fixtures in the root ``tests.py`` suite.
"""

import pytest

from process_bigraph import allocate_core, Composite, process, step, Process


# ---------------------------------------------------------------------------
# (a) the inferred config_schema matches the keyword-only args
# ---------------------------------------------------------------------------

def test_inferred_config_schema_matches_kwargs():
    @process(
        inputs={"state": "float", "neighbor_secretory": "float"},
        outputs={"fate": "overwrite[integer]"},
    )
    def BooleanSubcell(state, interval, *,
                       stemness_threshold: float = 0.4,
                       goblet_type: int = 3,
                       absorptive_type: int = 2):
        s = float((state or {}).get("state", 1.0))
        if s >= stemness_threshold:
            return {"fate": 0}
        neigh = float((state or {}).get("neighbor_secretory", 0))
        return {"fate": goblet_type if neigh == 0 else absorptive_type}

    assert BooleanSubcell.config_schema == {
        "stemness_threshold": {"_type": "float", "_default": 0.4},
        "goblet_type": {"_type": "integer", "_default": 3},
        "absorptive_type": {"_type": "integer", "_default": 2},
    }
    # discovery metadata is stamped like the older as_process decorator
    assert BooleanSubcell.__pb_kind__ == "process"
    assert BooleanSubcell.__pb_aliases__ == ["BooleanSubcell"]
    assert BooleanSubcell.__name__ == "BooleanSubcell"


def test_all_scalar_types_infer():
    @process(inputs={}, outputs={"out": "float"})
    def AllTypes(state, interval, *,
                 f: float = 1.5, i: int = 2, b: bool = True, s: str = "hi"):
        return {"out": 0.0}

    assert AllTypes.config_schema == {
        "f": {"_type": "float", "_default": 1.5},
        "i": {"_type": "integer", "_default": 2},
        "b": {"_type": "boolean", "_default": True},
        "s": {"_type": "string", "_default": "hi"},
    }


# ---------------------------------------------------------------------------
# (b) decorated process runs in a Composite, same result as the class version
# ---------------------------------------------------------------------------

def _run_decay(register, config):
    core = allocate_core()
    register(core)
    sim = Composite({"state": {
        "decay": {"_type": "process", "address": "local:decay",
                  "config": config,
                  "inputs": {"S": ["S"]}, "outputs": {"S": ["S"]}},
        "S": 10.0,
    }}, core=core)
    sim.run(1.0)
    return sim.state["S"]


def test_decorated_process_matches_class_based():
    @process(inputs={"S": "float"}, outputs={"S": "float"})
    def decay(state, interval, *, rate: float = 0.1):
        return {"S": -rate * state["S"] * interval}

    class DecayClass(Process):
        config_schema = {"rate": {"_type": "float", "_default": 0.1}}

        def initialize(self, config):
            self.rate = float(config["rate"])

        def inputs(self):
            return {"S": "float"}

        def outputs(self):
            return {"S": "float"}

        def update(self, state, interval):
            return {"S": -self.rate * state["S"] * interval}

    decorated = _run_decay(lambda c: c.register_link("decay", decay), {"rate": 0.2})
    classic = _run_decay(lambda c: c.register_link("decay", DecayClass), {"rate": 0.2})

    assert decorated == classic == 8.0


def test_default_config_used_when_no_override():
    @process(inputs={"S": "float"}, outputs={"S": "float"})
    def decay(state, interval, *, rate: float = 0.1):
        return {"S": -rate * state["S"] * interval}

    # no config -> the inferred default rate=0.1 is used: 10 - 0.1*10*1 = 9.0
    result = _run_decay(lambda c: c.register_link("decay", decay), {})
    assert result == 9.0


# ---------------------------------------------------------------------------
# (c) config overrides via config= work (and coerce/refuse through _coerce)
# ---------------------------------------------------------------------------

def test_config_override():
    @process(inputs={"S": "float"}, outputs={"S": "float"})
    def decay(state, interval, *, rate: float = 0.1):
        return {"S": -rate * state["S"] * interval}

    # override rate=0.5: 10 - 0.5*10*1 = 5.0
    result = _run_decay(lambda c: c.register_link("decay", decay), {"rate": 0.5})
    assert result == 5.0


def test_string_override_coerced():
    # a numeric string override is coerced to float (reuses composite_spec._coerce)
    @process(inputs={"S": "float"}, outputs={"S": "float"})
    def decay(state, interval, *, rate: float = 0.1):
        return {"S": -rate * state["S"] * interval}

    result = _run_decay(lambda c: c.register_link("decay", decay), {"rate": "0.2"})
    assert result == 8.0


def test_bad_override_refuses():
    @process(inputs={"S": "float"}, outputs={"S": "float"})
    def decay(state, interval, *, rate: float = 0.1):
        return {"S": -rate * state["S"] * interval}

    core = allocate_core()
    core.register_link("decay", decay)
    with pytest.raises(Exception):
        # a non-numeric override is refused rather than silently mangled
        decay(config={"rate": "not-a-number"}, core=core)


# ---------------------------------------------------------------------------
# @step analogue
# ---------------------------------------------------------------------------

def test_step_decorator_infers_and_runs():
    @step(inputs={"a": "float", "b": "float"}, outputs={"sum": "float"})
    def weighted_add(state, *, weight: float = 1.0):
        return {"sum": weight * (state["a"] + state["b"])}

    assert weighted_add.config_schema == {"weight": {"_type": "float", "_default": 1.0}}
    assert weighted_add.__pb_kind__ == "step"

    s = weighted_add(config={"weight": 2.0}, core=allocate_core())
    assert s.update({"a": 3.0, "b": 4.0}) == {"sum": 14.0}


# ---------------------------------------------------------------------------
# escape hatch: an un-inferrable annotation raises a named, actionable error
# ---------------------------------------------------------------------------

def test_uninferrable_type_raises_actionable_error():
    class Weird:
        pass

    with pytest.raises(TypeError, match="cannot infer a config type"):
        @process(inputs={}, outputs={})
        def bad(state, interval, *, thing: Weird):  # no default, non-scalar type
            return {}


def test_verbatim_string_annotation_used_as_bigraph_type():
    # advanced escape hatch: a string annotation is taken as a bigraph type name
    @process(inputs={}, outputs={"out": "float"})
    def adv(state, interval, *, weights: "map[float]" = None):
        return {"out": 0.0}

    assert adv.config_schema["weights"]["_type"] == "map[float]"
