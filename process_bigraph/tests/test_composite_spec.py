import pytest
import json
import textwrap
from process_bigraph.composite_spec import CompositeSpec, normalize_type, substitute_parameters
import process_bigraph.composite_spec as cs


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


def test_decorator_registers_generator():
    from process_bigraph import composite_spec as cs
    cs.clear_registry()

    @cs.composite_spec(name="demo", description="d",
                       parameters={"seed": {"type": "int", "default": 0}},
                       emitters=[{"address": "local:RAMEmitter"}], default_n_steps=5)
    def demo(core=None, *, seed=0):
        return {"state": {"s": seed}}

    spec_id = f"{demo.__module__}.demo"
    spec = cs.get(spec_id)
    assert spec is not None and spec.kind == "generator"
    assert spec.parameters["seed"]["type"] == "integer"  # normalized
    assert spec.default_n_steps == 5 and spec.builder is demo
    assert demo(seed=2) == {"state": {"s": 2}}  # decorator returns the original fn


def test_register_get_all_clear():
    from process_bigraph import composite_spec as cs_mod
    cs_mod.clear_registry()
    s = CompositeSpec(id="m.c", name="c", state={})
    cs_mod.register(s)
    assert cs_mod.get("m.c") is s
    assert "m.c" in cs_mod.all_specs()
    cs_mod.clear_registry()
    assert cs_mod.get("m.c") is None


def test_from_file_static_yaml(tmp_path):
    p = tmp_path / "g.composite.yaml"
    p.write_text(textwrap.dedent("""
        name: growth
        description: demo
        tags: [a]
        requires:
          processes: [P]
          types: [ty]
        schema:
          population: tree
        parameters:
          seed: {type: int, default: 0}
        state:
          v: "${seed}"
    """), encoding="utf-8")
    s = CompositeSpec.from_file(p)
    assert s.kind == "spec" and s.name == "growth"
    assert s.schema == {"population": "tree"}
    assert s.requires == {"processes": ["P"], "types": ["ty"]}
    assert s.parameters["seed"]["type"] == "integer"  # normalized
    assert s.id.endswith(".growth")


def test_from_file_generator_with_builder_ref(tmp_path):
    p = tmp_path / "b.composite.json"
    p.write_text(json.dumps({
        "name": "bgen", "builder": "process_bigraph.composite_spec:normalize_type",
        "default_state_ref": "bgen.default-state.json",
        "parameters": {}}), encoding="utf-8")
    s = CompositeSpec.from_file(p)
    assert s.kind == "generator" and s.default_state_ref == "bgen.default-state.json"


def test_discover_specs_scans_workspace_files(tmp_path):
    cs.clear_registry()
    comp = tmp_path / "composites"
    comp.mkdir()
    (comp / "x.composite.yaml").write_text("name: xc\nstate: {a: 1}\n", encoding="utf-8")
    found = cs.discover_specs(workspace=tmp_path)
    assert any(s.name == "xc" for s in found.values())


def test_register_spec_generator_hook_fires_on_discover(tmp_path):
    """The imperative hook surface: a registered zero-arg callable runs during
    discover_specs and can populate the registry (no pbg_superpowers import)."""
    cs.clear_registry()

    calls = []

    def hook():
        calls.append(True)
        cs.register(CompositeSpec(id="hooked.g", name="hooked",
                                  state={"a": 1}))
    try:
        cs.register_spec_generator(hook)
        found = cs.discover_specs()
        assert calls, "hook was not invoked by discover_specs"
        assert any(s.name == "hooked" for s in found.values())
    finally:
        # keep global hook registry clean for other tests
        cs._SPEC_GENERATOR_HOOKS.remove(hook)


def test_register_spec_generator_is_idempotent():
    cs._SPEC_GENERATOR_HOOKS.clear()

    def hook():
        pass
    try:
        cs.register_spec_generator(hook)
        cs.register_spec_generator(hook)
        assert cs._SPEC_GENERATOR_HOOKS.count(hook) == 1
    finally:
        cs._SPEC_GENERATOR_HOOKS.clear()


def test_discover_specs_runs_entry_point_group(monkeypatch):
    """discover_specs loads + runs each entry point in the
    process_bigraph.spec_generators group."""
    cs.clear_registry()
    cs._SPEC_GENERATOR_HOOKS.clear()

    ran = []

    class _FakeEP:
        name = "fake_generator"

        def load(self):
            def _run():
                ran.append(True)
                cs.register(CompositeSpec(id="ep.g", name="ep_gen",
                                          state={"a": 1}))
            return _run

    monkeypatch.setattr(cs, "_iter_spec_generator_entry_points",
                        lambda: [_FakeEP()])
    found = cs.discover_specs()
    assert ran, "entry point was not loaded/run"
    assert any(s.name == "ep_gen" for s in found.values())


def test_discover_specs_logs_entry_point_failure(monkeypatch, caplog):
    """A failing entry point is logged (not silently swallowed) and does not
    abort discovery."""
    cs.clear_registry()
    cs._SPEC_GENERATOR_HOOKS.clear()

    class _BadEP:
        name = "bad_generator"

        def load(self):
            def _boom():
                raise RuntimeError("kaboom")
            return _boom

    monkeypatch.setattr(cs, "_iter_spec_generator_entry_points",
                        lambda: [_BadEP()])
    import logging
    with caplog.at_level(logging.WARNING):
        cs.discover_specs()  # must not raise
    assert any("bad_generator" in r.message for r in caplog.records)


def test_composite_spec_module_has_no_module_level_yaml_import():
    # yaml must be a lazy/optional import (PyYAML is not a process-bigraph dep);
    # importing the module must not require it, only from_file on a .yaml does.
    import process_bigraph.composite_spec as m
    assert not hasattr(m, "yaml"), "yaml must not be imported at module level"


def test_regenerate_and_read_default_state(tmp_path):
    def build(core=None, *, seed=0):
        return {"state": {"count": seed + 1}}
    s = CompositeSpec(id="m.g", name="g", builder=build,
                      parameters={"seed": {"type": "integer", "default": 4}},
                      default_state_ref="g.default-state.json")
    from process_bigraph.composite_spec import regenerate_default_state
    out = regenerate_default_state(s, tmp_path)
    assert out.exists()
    # display reads the saved state, no rebuild
    state = s.default_state(base_dir=tmp_path)
    assert state is not None
    assert state["count"] == 5


def test_default_state_missing_artifact_returns_none(tmp_path):
    s = CompositeSpec(id="m.g", name="g", builder=lambda core=None: {"state": {}},
                      default_state_ref="absent.json")
    assert s.default_state(base_dir=tmp_path) is None


def test_top_level_exports():
    import process_bigraph as pbg
    assert hasattr(pbg, "CompositeSpec")
    assert hasattr(pbg, "composite_spec")      # the decorator
    assert hasattr(pbg, "discover_specs")


# ---------------------------------------------------------------------------
# Typed parameter validation: "unknown type" and "broken checker" are not
# the same answer
# ---------------------------------------------------------------------------

def test_unknown_declared_type_passes_through_unvalidated():
    """A workspace type this core does not define is not ours to judge."""
    from process_bigraph.composite_spec import _coerce

    assert _coerce({'a': 1}, 'some_workspace_type', name='p') == {'a': 1}


def test_known_declared_type_is_still_enforced():
    from process_bigraph.composite_spec import _coerce

    assert _coerce('2.5', 'float', name='rate') == 2.5
    with pytest.raises(Exception):
        _coerce('not-a-number', 'float', name='rate')


def test_a_broken_type_checker_is_not_read_as_permission():
    """The regression this guards.

    `_coerce` used to wrap the final `core.check` in a blanket
    `except Exception: return coerced`. That made "this core doesn't know
    that type" and "the type checker itself broke" the same outcome — the
    value passed through unvalidated either way — so a genuine bug in
    `check` would silently disable parameter validation for every spec
    while every test still passed.
    """
    from process_bigraph.composite_spec import _coerce, _validation_core

    class BrokenCore:
        def access(self, declared):
            return _validation_core(None).access(declared)

        def check(self, declared, value):
            raise RuntimeError('the type checker is broken')

    with pytest.raises(RuntimeError, match='type checker is broken'):
        _coerce(1.0, 'float', name='rate', core=BrokenCore())


# ---------------------------------------------------------------------------
# The runtime partition is structural, so it is cached
# ---------------------------------------------------------------------------

def _one_process_composite():
    from process_bigraph import Composite, allocate_core
    from process_bigraph.processes.examples import IncreaseProcess

    core = allocate_core()
    core.register_link('IncreaseProcess', IncreaseProcess)
    return Composite(_one_process_document(), core=core)


def _one_process_document():
    return {'state': {
        'level': 5.0,
        'grow': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.1},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': 1.0}}}


def test_runtime_partition_is_cached_across_ticks():
    """It is a function of the process network's shape, which does not
    change between ticks — it used to be rediscovered by a full walk of
    every process on every tick."""
    composite = _one_process_composite()
    first = composite._partition_processes_by_runtime()
    assert composite._partition_processes_by_runtime() is first

    composite.run(2.0)
    assert composite._partition_processes_by_runtime() is first


def test_runtime_partition_cache_is_dropped_on_structural_change():
    """Same lifetime as the layer walk: a document whose process network
    changed must not keep answering from the old shape."""
    composite = _one_process_composite()
    first = composite._partition_processes_by_runtime()

    composite.expire_layer_walk_cache()
    assert composite._partition_processes_by_runtime() is not first

    rebuilt = composite._partition_processes_by_runtime()
    composite._invalidate_caches()
    assert composite._partition_processes_by_runtime() is not rebuilt


def test_a_document_with_no_protocol_runtime_partitions_to_nothing():
    managed, groups = _one_process_composite(
        )._partition_processes_by_runtime()
    assert managed == set()
    assert groups == []
