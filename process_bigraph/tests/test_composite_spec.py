import pytest
from process_bigraph.composite_spec import CompositeSpec, normalize_type, CANONICAL_TYPES, substitute_parameters


def test_normalize_type_aliases():
    assert normalize_type("int") == "integer"
    assert normalize_type("number") == "float"
    assert normalize_type("bool") == "boolean"
    assert normalize_type("object") == "map"
    assert normalize_type("array") == "list"
    assert normalize_type("string") == "string"
    assert normalize_type("unknownX") == "unknownX"  # pass-through


def test_static_spec_normalizes_param_types():
    s = CompositeSpec(id="m.c", name="c", state={"x": 1},
                      parameters={"seed": {"type": "int", "default": 0},
                                  "rate": {"type": "number", "default": 1.0}})
    assert s.parameters["seed"]["type"] == "integer"
    assert s.parameters["rate"]["type"] == "float"
    assert s.kind == "spec"


def test_generator_spec_kind():
    s = CompositeSpec(id="m.g", name="g", builder="m:g")
    assert s.kind == "generator"


def test_exactly_one_of_state_or_builder():
    with pytest.raises(ValueError, match="exactly one"):
        CompositeSpec(id="m.x", name="x")  # neither
    with pytest.raises(ValueError, match="exactly one"):
        CompositeSpec(id="m.x", name="x", state={}, builder="m:x")  # both


def test_schema_only_with_state():
    with pytest.raises(ValueError, match="schema"):
        CompositeSpec(id="m.g", name="g", builder="m:g", schema={"pop": "tree"})


def test_default_state_ref_requires_builder():
    with pytest.raises(ValueError, match="default_state_ref"):
        CompositeSpec(id="m.c", name="c", state={"x": 1}, default_state_ref="x.json")


def test_to_dict_from_dict_round_trip_static():
    s = CompositeSpec(id="m.c", name="c", description="d", tags=["t"],
                      state={"x": "${seed}"}, schema={"pop": "tree"},
                      parameters={"seed": {"type": "integer", "default": 0}},
                      requires={"processes": ["P"], "types": ["ty"]},
                      emitters=[{"address": "local:RAMEmitter"}], default_n_steps=10)
    d = s.to_dict()
    assert d["state"] == {"x": "${seed}"} and d["schema"] == {"pop": "tree"}
    assert d["requires"] == {"processes": ["P"], "types": ["ty"]}
    s2 = CompositeSpec.from_dict(d)
    assert s2 == s


def test_to_dict_serializes_builder_callable_to_dotted():
    def fn(core=None):
        return {"state": {}}
    s = CompositeSpec(id="m.g", name="g", builder=fn, module="m")
    d = s.to_dict()
    # a callable serializes to "<module>:<qualname>"; core_extensions are dropped
    assert isinstance(d["builder"], str) and d["builder"].endswith(":fn")
    assert "core_extensions" not in d


def test_substitute_full_and_inline():
    params = {"seed": {"type": "integer", "default": 0}, "tag": {"type": "string", "default": "x"}}
    state = {"a": "${seed}", "b": "v-${tag}", "c": 5}
    out = substitute_parameters(state, params, {"seed": 7, "tag": "z"})
    assert out == {"a": 7, "b": "v-z", "c": 5}  # full placeholder typed; inline stringified


def test_to_document_static_substitutes_schema_and_state():
    s = CompositeSpec(id="m.c", name="c", state={"x": "${seed}"}, schema={"pop": "tree"},
                      parameters={"seed": {"type": "integer", "default": 3}})
    doc = s.to_document()
    assert doc == {"schema": {"pop": "tree"}, "state": {"x": 3}}


def test_to_document_generator_passes_whole_document():
    def build(core=None, *, seed=0):
        return {"state": {"s": seed}, "skip_initial_steps": True, "flow_order": ["a"]}
    s = CompositeSpec(id="m.g", name="g", builder=build,
                      parameters={"seed": {"type": "integer", "default": 0}})
    doc = s.to_document({"seed": 9})
    assert doc == {"state": {"s": 9}, "skip_initial_steps": True, "flow_order": ["a"]}


def test_to_document_rejects_unknown_override():
    s = CompositeSpec(id="m.c", name="c", state={}, parameters={"seed": {"type": "integer", "default": 0}})
    with pytest.raises(KeyError):
        s.to_document({"nope": 1})


def test_to_composite_static_returns_runnable_composite():
    s = CompositeSpec(id="m.c", name="c", state={"v": 1.0})
    comp = s.to_composite()
    from process_bigraph import Composite
    assert isinstance(comp, Composite)


def test_to_composite_applies_core_extensions_before_build():
    seen = {}
    def ext(core):
        seen["ran"] = True
        return core
    def build(core=None):
        seen["built_after_ext"] = seen.get("ran", False)
        return {"state": {}}
    s = CompositeSpec(id="m.g", name="g", builder=build, core_extensions=[ext])
    s.to_composite()
    assert seen["ran"] and seen["built_after_ext"]


def test_default_state_inline():
    s = CompositeSpec(id="m.c", name="c", state={"v": 1})
    assert s.default_state() == {"v": 1}
