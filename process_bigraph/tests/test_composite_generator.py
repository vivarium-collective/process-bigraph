"""Tests for process_bigraph.composite_generator (moved from viva-superpowers)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from process_bigraph.composite_generator import (
    build_generator, composite_generator, GeneratorEntry, _REGISTRY,
    apply_core_extensions, emitter_defaults,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test starts with an empty registry."""
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


def test_decorator_registers_function():
    @composite_generator(
        name="my-composite",
        description="A test composite.",
        parameters={"rate": {"type": "float", "default": 1.0}},
    )
    def builder(core=None, *, rate=1.0):
        return {"stores": {"level": rate}}

    # Function is registered with its module-qualified id
    entry_id = f"{builder.__module__}.my-composite"
    assert entry_id in _REGISTRY
    entry = _REGISTRY[entry_id]
    assert isinstance(entry, GeneratorEntry)
    assert entry.name == "my-composite"
    assert entry.description == "A test composite."
    assert entry.parameters == {"rate": {"type": "float", "default": 1.0}}
    assert entry.func is builder
    # Wrapped function is unchanged-ish: callable with same signature
    assert builder(rate=2.0) == {"stores": {"level": 2.0}}
    # Sidecar on the function for introspection
    assert builder._composite_generator_entry is entry


def test_decorator_accepts_default_n_steps():
    @composite_generator(
        name="dn", description="", parameters={}, default_n_steps=200,
    )
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.dn"]
    assert entry.default_n_steps == 200


def test_decorator_default_n_steps_optional():
    @composite_generator(name="dn-opt", description="", parameters={})
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.dn-opt"]
    assert entry.default_n_steps is None


def test_decorator_core_extensions_default_empty():
    @composite_generator(name="ce-default", description="", parameters={})
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.ce-default"]
    assert entry.core_extensions == []


def test_decorator_stores_core_extensions():
    """v2ecoli friction #16: a generator can declare register_* callables so
    the subprocess runner can register the package's types on the core it
    actually builds against."""
    def register_pymunk_types(core):
        core.setdefault("types", []).append("pymunk_agent")
        return core

    def register_processes(core):
        core.setdefault("procs", []).append("Attachment")
        return core

    @composite_generator(
        name="ce-stored", description="", parameters={},
        core_extensions=[register_pymunk_types, register_processes],
    )
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.ce-stored"]
    assert entry.core_extensions == [register_pymunk_types, register_processes]


def test_apply_core_extensions_runs_each_in_order():
    calls = []

    def ext_a(core):
        calls.append("a")
        core["a"] = True
        return core

    def ext_b(core):
        calls.append("b")
        core["b"] = True
        return core

    entry = GeneratorEntry(
        id="x.y", name="y", description="", parameters={},
        func=lambda core=None: {}, module="x",
        core_extensions=[ext_a, ext_b],
    )
    core = {}
    out = apply_core_extensions(entry, core)
    assert calls == ["a", "b"]
    assert out == {"a": True, "b": True}


def test_apply_core_extensions_threads_replacement_core():
    """An extension that returns a NEW core (not the one passed in) should
    have that replacement threaded to the next extension and returned."""
    sentinel_a = {"id": "a"}
    sentinel_b = {"id": "b"}

    def ext_replaces(core):
        return sentinel_b

    def ext_observes(core):
        assert core is sentinel_b  # got the replacement, not the original
        return None  # returning None keeps the current core

    entry = GeneratorEntry(
        id="x.y", name="y", description="", parameters={},
        func=lambda core=None: {}, module="x",
        core_extensions=[ext_replaces, ext_observes],
    )
    out = apply_core_extensions(entry, sentinel_a)
    assert out is sentinel_b


def test_apply_core_extensions_none_return_keeps_core():
    def ext_inplace(core):
        core["touched"] = True
        return None  # mutate-in-place convention

    entry = GeneratorEntry(
        id="x.y", name="y", description="", parameters={},
        func=lambda core=None: {}, module="x",
        core_extensions=[ext_inplace],
    )
    core = {}
    out = apply_core_extensions(entry, core)
    assert out is core
    assert core == {"touched": True}


def test_apply_core_extensions_does_not_swallow_failures():
    """A missing/broken registration must surface, not be silently skipped —
    otherwise the Composite build fails later with a cryptic type error."""
    def ext_boom(core):
        raise RuntimeError("register_pymunk_types blew up")

    entry = GeneratorEntry(
        id="x.y", name="y", description="", parameters={},
        func=lambda core=None: {}, module="x",
        core_extensions=[ext_boom],
    )
    with pytest.raises(RuntimeError, match="register_pymunk_types blew up"):
        apply_core_extensions(entry, {})


def test_apply_core_extensions_empty_is_noop():
    entry = GeneratorEntry(
        id="x.y", name="y", description="", parameters={},
        func=lambda core=None: {}, module="x",
    )
    core = {"unchanged": True}
    assert apply_core_extensions(entry, core) is core


def test_decorator_accepts_visualizations():
    """``visualizations`` ships canonical Study-spec viz entries with the
    composite; dashboards can merge them into a Study without the author
    having to hand-author each."""
    viz_list = [
        {
            "name": "level-trace",
            "address": "local:TimeSeriesPlot",
            "config": {"observable": "level"},
        },
        {
            "name": "topology",
            "address": "local:NetworkVisualization",
            "config": {},
        },
    ]

    @composite_generator(
        name="vz",
        description="",
        parameters={},
        visualizations=viz_list,
    )
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.vz"]
    assert entry.visualizations == viz_list
    # Defensive copy — mutating the caller's list shouldn't change the entry.
    viz_list.append({"name": "intruder"})
    assert len(entry.visualizations) == 2


def test_decorator_visualizations_optional():
    @composite_generator(name="vz-opt", description="", parameters={})
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.vz-opt"]
    assert entry.visualizations == []


def test_decorator_accepts_emitters():
    """``emitters`` ships the composite's default observation sink(s) — a
    lightweight {address, config, paths} selection, parallel to
    ``visualizations``."""
    emitters = [
        {
            "address": "local:ParquetEmitter",
            "config": {"out_dir": "out/parquet"},
            "paths": ["stores.x"],
        },
    ]

    @composite_generator(name="em", description="", parameters={}, emitters=emitters)
    def builder(core=None):
        return {"stores": {"x": 0.0}}

    entry = _REGISTRY[f"{builder.__module__}.em"]
    assert entry.emitters == emitters
    # Defensive copy — mutating the caller's list/dict shouldn't reach the entry.
    emitters.append({"address": "local:SQLiteEmitter"})
    assert len(entry.emitters) == 1
    emitters_inner = entry.emitters[0]
    assert emitters_inner["config"]["out_dir"] == "out/parquet"


def test_decorator_emitters_optional():
    @composite_generator(name="em-opt", description="", parameters={})
    def builder(core=None):
        return {}

    entry = _REGISTRY[f"{builder.__module__}.em-opt"]
    assert entry.emitters == []


def test_emitter_defaults_accessor_reads_function_and_entry():
    emitters = [{"address": "local:ParquetEmitter", "config": {"out_dir": "d"}}]

    @composite_generator(name="em-acc", description="", parameters={}, emitters=emitters)
    def builder(core=None):
        return {}

    # From the decorated function (sidecar) and from the entry directly.
    assert emitter_defaults(builder) == emitters
    entry = _REGISTRY[f"{builder.__module__}.em-acc"]
    assert emitter_defaults(entry) == emitters
    # Non-generators yield [] so callers can use it unconditionally.
    assert emitter_defaults(object()) == []
    assert emitter_defaults(lambda: None) == []


@pytest.mark.parametrize(
    "bad, match",
    [
        ([{"config": {}}], "address"),                       # missing address
        ([{"address": ""}], "address"),                      # empty address
        ([{"address": 1}], "address"),                       # non-string address
        ([{"address": "local:X", "config": []}], "config"),  # config not a dict
        ([{"address": "local:X", "paths": "a.b"}], "paths"),  # paths not a list
        ([{"address": "local:X", "paths": [1]}], "paths"),    # paths not strings
        (["not-a-dict"], "must be a dict"),                  # entry not a dict
    ],
)
def test_decorator_emitters_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        @composite_generator(name="em-bad", description="", parameters={}, emitters=bad)
        def builder(core=None):
            return {}


def _make_entry(parameters, body):
    @composite_generator(name="t", description="", parameters=parameters)
    def _fn(core=None, **kw):
        return body(**kw)
    entry_id = f"{_fn.__module__}.t"
    return _REGISTRY[entry_id]


def test_build_generator_applies_defaults():
    entry = _make_entry(
        {"rate": {"type": "float", "default": 0.25}},
        lambda **kw: {"got": kw},
    )
    assert build_generator(entry) == {"got": {"rate": 0.25}}


def test_build_generator_applies_overrides():
    entry = _make_entry(
        {"rate": {"type": "float", "default": 0.25}},
        lambda **kw: {"got": kw},
    )
    assert build_generator(entry, overrides={"rate": 9.0}) == {"got": {"rate": 9.0}}


def test_build_generator_rejects_unknown_overrides():
    entry = _make_entry(
        {"rate": {"type": "float", "default": 0.25}},
        lambda **kw: {"got": kw},
    )
    with pytest.raises(ValueError, match="bogus"):
        build_generator(entry, overrides={"bogus": 1})


def test_build_generator_passes_core_when_present():
    seen = {}

    @composite_generator(name="t2", description="", parameters={})
    def builder(core=None):
        seen["core"] = core
        return {}

    entry = _REGISTRY[f"{builder.__module__}.t2"]
    sentinel = object()
    build_generator(entry, core=sentinel)
    assert seen["core"] is sentinel


FIXTURE_PKG = Path(__file__).parent / "fixtures" / "fake_generator_pkg"


def _install_cmd(action: str, target: str) -> list[str]:
    """Build a pip install/uninstall command that targets ``sys.executable``.

    Prefers ``uv pip --python <sys.executable>`` because it (a) pins the
    target interpreter explicitly — bare ``uv pip`` would otherwise pick
    the project's ``.venv`` instead of the pyenv interpreter that's
    actually running pytest — and (b) doesn't require pip to be present
    in the target env, which matters in CI where ``uv venv`` creates a
    pip-less ``.venv``.

    Falls back to ``sys.executable -m pip`` when ``uv`` isn't on PATH,
    for contributors who run tests without uv installed.
    """
    if shutil.which("uv"):
        if action == "install":
            return ["uv", "pip", "install", "--python", sys.executable,
                    "-q", "-e", target]
        return ["uv", "pip", "uninstall", "--python", sys.executable, target]
    if action == "install":
        return [sys.executable, "-m", "pip", "install", "-q", "-e", target]
    return [sys.executable, "-m", "pip", "uninstall", "-q", "-y", target]


@pytest.fixture
def installed_fake_pkg():
    """Install the fixture package into the running test interpreter's env.

    Uses ``uv pip install --python sys.executable`` so the install lands
    in the same environment that runs the test, regardless of whether
    that env has pip available (CI's ``uv venv`` does not) and regardless
    of whether ``uv``'s default target ``.venv`` differs from
    ``sys.executable`` (e.g. running under pyenv locally). Falls back to
    ``sys.executable -m pip`` if uv is not installed.

    Also patches ``sys.path`` directly so the editable ``.pth`` file
    added by hatchling takes effect inside the already-running process
    (``site`` only processes ``.pth`` files at startup).
    """
    subprocess.run(_install_cmd("install", str(FIXTURE_PKG)), check=True)
    # Editable installs write a .pth file that is only processed at Python
    # startup. Manually add the package root so importlib can find it now.
    pkg_root = str(FIXTURE_PKG)
    inserted = pkg_root not in sys.path
    if inserted:
        sys.path.insert(0, pkg_root)
    import importlib
    importlib.invalidate_caches()
    yield
    # Teardown: remove from sys.path and evict from sys.modules so subsequent
    # tests (or re-runs) don't see stale state.
    if inserted and pkg_root in sys.path:
        sys.path.remove(pkg_root)
    for mod_name in list(sys.modules):
        if mod_name == "fake_generator_pkg" or mod_name.startswith("fake_generator_pkg."):
            del sys.modules[mod_name]
    importlib.invalidate_caches()
    subprocess.run(_install_cmd("uninstall", "fake-generator-pkg"), check=True)


def test_discover_generators_finds_decorated_function_in_installed_pkg(
        installed_fake_pkg):
    from process_bigraph.composite_generator import discover_generators
    # Discovery must import the package — it can't rely on the test having
    # already done so.
    found = discover_generators()
    expected_id = "fake_generator_pkg.composites.demo"
    assert expected_id in found
    entry = found[expected_id]
    assert entry.name == "demo"
    # "int" is normalised to "integer" by CompositeSpec.__post_init__
    assert entry.parameters == {"x": {"type": "integer", "default": 7}}


def test_discover_all_merges_specs_and_generators(tmp_path, installed_fake_pkg):
    # Create a tiny static spec in a tmp dir so discover_composites picks it up
    spec_file = tmp_path / "baseline.composite.yaml"
    spec_file.write_text("name: baseline\nstate: {}\n")

    from process_bigraph.composite_discovery import discover_all
    _REGISTRY.clear()
    merged = discover_all(extra_search_paths=[tmp_path])

    # Spec entry tagged spec
    spec_keys = [k for k, v in merged.items() if v.get("kind") == "spec"]
    assert any(k.endswith(".baseline") for k in spec_keys)

    # Generator entry tagged generator
    gen_id = "fake_generator_pkg.composites.demo"
    assert gen_id in merged
    assert merged[gen_id]["kind"] == "generator"
    assert merged[gen_id]["name"] == "demo"
    # default_n_steps is always propagated (None when the generator omits it).
    assert "default_n_steps" in merged[gen_id]
    # visualizations is always propagated as a list (empty when the generator
    # omits it) so dashboard callers can rely on the key existing.
    assert merged[gen_id].get("visualizations") == []
    # emitters likewise — the default observation sink(s) travel with the entry.
    assert merged[gen_id].get("emitters") == []


# ---------------------------------------------------------------------------
# Slice H / friction #22: subpackage @composite_generator decorators are
# discovered even when the top-level __init__.py doesn't eagerly import
# the subpackage.
# ---------------------------------------------------------------------------


def test_discover_generators_walks_subpackages_without_eager_import(tmp_path):
    """A workspace package that ships its generators in
    `<pkg>/composites/__init__.py` should have those decorators fire on
    discovery — even when `<pkg>/__init__.py` doesn't do
    `from . import composites`. Before the walk_packages fix, the
    subpackage was invisible to discover_generators() and the dashboard
    Run handler would silently reject the composite as "not in registry."
    """
    import sys
    import importlib
    from process_bigraph.composite_generator import (
        _REGISTRY, discover_generators,
    )

    # Build a minimal package on disk:
    #   tmp/_walkpkg_demo/__init__.py          (NO eager subpackage import)
    #   tmp/_walkpkg_demo/composites/__init__.py  (@composite_generator decorator)
    pkg_dir = tmp_path / "_walkpkg_demo"
    sub_dir = pkg_dir / "composites"
    sub_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        '"""Top-level package with NO eager subpackage import — the walk\n'
        'should pick up the subpackage decorators anyway."""\n'
    )
    (sub_dir / "__init__.py").write_text(
        "from process_bigraph.composite_generator import composite_generator\n"
        "\n"
        "@composite_generator(name='walkpkg_demo_baseline')\n"
        "def walkpkg_demo_baseline():\n"
        "    return {'state': {'x': 1.0}}\n"
    )

    sys.path.insert(0, str(tmp_path))
    _REGISTRY.clear()
    try:
        # Without the walk_packages fix, this returns {} because importing
        # `_walkpkg_demo` alone doesn't fire decorators in `_walkpkg_demo.composites`.
        found = discover_generators(extra_packages=["_walkpkg_demo"])
        # The expected entry id is "<pkg>.composites.<name>"; the @composite_generator
        # decorator builds it from func.__module__ + entry.name.
        matching = [eid for eid in found if eid.endswith(".walkpkg_demo_baseline")]
        assert matching, (
            f"discover_generators didn't find the subpackage decorator. "
            f"Got: {sorted(found.keys())}"
        )
    finally:
        sys.path.remove(str(tmp_path))
        for k in list(sys.modules):
            if k == "_walkpkg_demo" or k.startswith("_walkpkg_demo."):
                del sys.modules[k]
        _REGISTRY.clear()
        importlib.invalidate_caches()


def test_discover_generators_traps_sys_exit_from_imported_subpackage(tmp_path):
    """v2ecoli friction #4: a subpackage that calls sys.exit() at module
    level (typical for CLI scripts without `if __name__ == "__main__":`)
    used to take the whole subprocess down. Now it warns + continues."""
    import sys
    import importlib
    from process_bigraph.composite_generator import (
        _REGISTRY, discover_generators,
    )

    # Build a pkg whose subpackage exits at import time.
    pkg_dir = tmp_path / "_exiter_demo"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        '"""Top-level package; subpackage below sys.exits at import."""\n'
    )
    (pkg_dir / "ill_behaved.py").write_text(
        "import sys\n"
        "sys.exit(0)\n"   # this used to kill the discovery subprocess
    )

    sys.path.insert(0, str(tmp_path))
    _REGISTRY.clear()
    try:
        # Must NOT raise SystemExit — the trap converts it to a warning.
        found = discover_generators(extra_packages=["_exiter_demo"])
        # The badly-behaved subpackage is skipped; nothing registers.
        assert all(not k.startswith("_exiter_demo.") for k in found.keys())
    finally:
        sys.path.remove(str(tmp_path))
        for k in list(sys.modules):
            if k == "_exiter_demo" or k.startswith("_exiter_demo."):
                del sys.modules[k]
        _REGISTRY.clear()
        importlib.invalidate_caches()


def test_composite_generator_registers_into_process_bigraph_registry():
    from process_bigraph import composite_spec as cs
    from process_bigraph.composite_generator import composite_generator, build_generator
    cs.clear_registry()

    @composite_generator(name="shimdemo", parameters={"seed": {"type": "int", "default": 1}})
    def shimdemo(core=None, *, seed=1):
        return {"state": {"s": seed}}

    spec_id = f"{shimdemo.__module__}.shimdemo"
    spec = cs.get(spec_id)
    assert spec is not None and spec.parameters["seed"]["type"] == "integer"
    # build_generator delegates to the spec's document
    assert build_generator(spec, overrides={"seed": 3}) == {"state": {"s": 3}}


def test_build_generator_descriptive_error_for_unregistered_id():
    from process_bigraph import composite_spec as cs
    from process_bigraph.composite_generator import build_generator, GeneratorEntry
    cs.clear_registry()
    ge = GeneratorEntry(id="missing.x", name="x", description="", parameters={},
                        func=None, module="missing")
    import pytest
    with pytest.raises(ValueError, match="no registered composite"):
        build_generator(ge)


def test_registry_view_supports_clean_alias_assignment():
    import dataclasses
    from process_bigraph import composite_spec as cs
    from process_bigraph.composite_generator import composite_generator, _REGISTRY
    cs.clear_registry()

    @composite_generator(name="aliasme", parameters={"seed": {"type": "int", "default": 0}})
    def aliasme(core=None, *, seed=0):
        return {"state": {"s": seed}}

    full_id = f"{aliasme.__module__}.aliasme"
    orig = _REGISTRY[full_id]                       # GeneratorEntry from the view
    _REGISTRY["aliasme"] = dataclasses.replace(orig, id="aliasme")  # the v2ecoli pattern
    assert "aliasme" in _REGISTRY
    assert _REGISTRY["aliasme"].id == "aliasme"
    assert cs.get("aliasme") is not None and cs.get("aliasme").kind == "generator"
    assert len(_REGISTRY) >= 2                      # exercises __len__


def test_registry_view_dict_conversion_works():
    """M2: _RegistryView.keys() enables dict(_REGISTRY) to work."""
    from process_bigraph import composite_spec as cs
    from process_bigraph.composite_generator import composite_generator, _REGISTRY
    cs.clear_registry()
    @composite_generator(name="dictme", parameters={})
    def dictme(core=None):
        return {"state": {}}
    d = dict(_REGISTRY)            # requires keys()
    assert f"{dictme.__module__}.dictme" in d


def test_discover_generators_skips_scripts_subpackage(tmp_path):
    """v2ecoli friction #4: a `scripts/` subpackage holds CLI tools, not
    library code; discovery should skip it entirely to avoid importing
    argparse-driven modules that crash under bare import."""
    import sys
    import importlib
    from process_bigraph.composite_generator import (
        _REGISTRY, discover_generators, composite_generator,
    )

    pkg_dir = tmp_path / "_libscripts_demo"
    scripts_dir = pkg_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    # A generator in the LIBRARY half — must be discovered.
    (pkg_dir / "lib_module.py").write_text(
        "from process_bigraph.composite_generator import composite_generator\n"
        "\n"
        "@composite_generator(name='libscripts_demo_baseline')\n"
        "def libscripts_demo_baseline():\n"
        "    return {'state': {}}\n"
    )
    # A would-be generator in scripts/ — must be SKIPPED even though it's
    # otherwise valid.
    (scripts_dir / "__init__.py").write_text("")
    (scripts_dir / "would_register.py").write_text(
        "from process_bigraph.composite_generator import composite_generator\n"
        "\n"
        "@composite_generator(name='libscripts_demo_should_be_skipped')\n"
        "def libscripts_demo_should_be_skipped():\n"
        "    return {'state': {}}\n"
    )

    sys.path.insert(0, str(tmp_path))
    _REGISTRY.clear()
    try:
        found = discover_generators(extra_packages=["_libscripts_demo"])
        libs = [k for k in found if "libscripts_demo" in k]
        # Library generator landed.
        assert any("baseline" in k for k in libs)
        # scripts/ generator did not.
        assert not any("should_be_skipped" in k for k in libs)
    finally:
        sys.path.remove(str(tmp_path))
        for k in list(sys.modules):
            if k == "_libscripts_demo" or k.startswith("_libscripts_demo."):
                del sys.modules[k]
        _REGISTRY.clear()
        importlib.invalidate_caches()


def test_discover_generators_skips_tests_subpackage_before_import(tmp_path):
    """process-bigraph itself declares `bigraph-schema` as a dependency, so
    its own distribution is always a walk target — and (unlike
    viva-superpowers, whose tests/ sits *outside* the package) this repo
    keeps tests *inside* the package at process_bigraph/tests/. A naive
    walk would import the whole test suite (and anything nested under
    tests/fixtures/) as an uncontrolled side effect of composite discovery.

    This mirrors test_discover_generators_skips_scripts_subpackage, but for
    a `tests/` subpackage, and additionally proves the skip happens BEFORE
    any import is attempted (not just a post-hoc skip of an already-walked
    entry) by nesting a decorator-bearing module two levels under tests/ —
    exactly the shape of tests/fixtures/fake_generator_pkg/fake_generator_pkg/.
    """
    import sys
    import importlib
    from process_bigraph.composite_generator import (
        _REGISTRY, discover_generators, composite_generator,
    )

    pkg_dir = tmp_path / "_libtests_demo"
    tests_dir = pkg_dir / "tests"
    nested_dir = tests_dir / "fixtures" / "would_be_pkg"
    nested_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    # A generator in the LIBRARY half — must be discovered.
    (pkg_dir / "lib_module.py").write_text(
        "from process_bigraph.composite_generator import composite_generator\n"
        "\n"
        "@composite_generator(name='libtests_demo_baseline')\n"
        "def libtests_demo_baseline():\n"
        "    return {'state': {}}\n"
    )
    # A would-be generator nested under tests/ — must be SKIPPED, and the
    # skip must happen before any of these modules are ever imported.
    (tests_dir / "__init__.py").write_text(
        "raise RuntimeError('tests/__init__.py must never be imported by discovery')\n"
    )
    (nested_dir / "__init__.py").write_text(
        "from process_bigraph.composite_generator import composite_generator\n"
        "\n"
        "@composite_generator(name='libtests_demo_should_be_skipped')\n"
        "def libtests_demo_should_be_skipped():\n"
        "    return {'state': {}}\n"
    )

    sys.path.insert(0, str(tmp_path))
    _REGISTRY.clear()
    try:
        # Must not raise — proves tests/__init__.py (which raises on import)
        # was never imported, i.e. the skip happens before descent, not
        # merely after pkgutil has already recursed into it.
        found = discover_generators(extra_packages=["_libtests_demo"])
        libs = [k for k in found if "libtests_demo" in k]
        assert any("baseline" in k for k in libs)
        assert not any("should_be_skipped" in k for k in libs)
        assert "_libtests_demo.tests" not in sys.modules
        assert "_libtests_demo.tests.fixtures.would_be_pkg" not in sys.modules
    finally:
        sys.path.remove(str(tmp_path))
        for k in list(sys.modules):
            if k == "_libtests_demo" or k.startswith("_libtests_demo."):
                del sys.modules[k]
        _REGISTRY.clear()
        importlib.invalidate_caches()


def test_discover_generators_does_not_walk_process_bigraph_own_tests():
    """Regression for a full-suite-only flaky failure: discover_generators()
    walks every installed bigraph-schema-dependent distribution, and
    process-bigraph's own distribution always qualifies (it declares
    bigraph-schema as a dependency). In an editable/dev install, that
    self-walk would otherwise recurse into process_bigraph/tests/ — this
    asserts none of process-bigraph's own test modules end up registered
    as discovered generators."""
    from process_bigraph.composite_generator import discover_generators
    found = discover_generators()
    assert not any("process_bigraph.tests" in sid for sid in found), (
        f"discover_generators walked into process_bigraph's own test suite: "
        f"{[sid for sid in found if 'process_bigraph.tests' in sid]}"
    )
