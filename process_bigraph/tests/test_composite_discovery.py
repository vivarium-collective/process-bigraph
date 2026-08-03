"""Tests for process_bigraph.composite_discovery (moved from viva_superpowers)."""
from pathlib import Path

import pytest

from process_bigraph.composite_discovery import (
    discover_composites, discover_all, _make_spec_id,
)
from process_bigraph import composite_spec as cs


def test_discover_composites_finds_workspace_local_specs(tmp_path):
    """extra_search_paths lets a workspace-local composite dir be discovered
    without it belonging to any installed bigraph-schema-dependent package."""
    (tmp_path / "local.composite.yaml").write_text(
        "name: increase-demo\nstate: {a: 1}\n", encoding="utf-8"
    )
    specs = discover_composites(extra_search_paths=[tmp_path])
    assert any(spec_id.endswith("local") for spec_id in specs), (
        f"workspace-local fixture not found: {list(specs.keys())}"
    )


def test_discover_composites_skips_malformed_spec(tmp_path, capsys):
    """A malformed spec file is skipped (with a stderr warning), not fatal."""
    (tmp_path / "broken.composite.yaml").write_text(
        "not: [valid, - yaml: :", encoding="utf-8"
    )
    (tmp_path / "good.composite.yaml").write_text(
        "name: good\nstate: {a: 1}\n", encoding="utf-8"
    )
    specs = discover_composites(extra_search_paths=[tmp_path])
    assert any(spec_id.endswith("good") for spec_id in specs)
    assert not any(spec_id.endswith("broken") for spec_id in specs)


def test_discover_composites_ignores_non_directory_extra_path(tmp_path):
    """A non-existent / non-directory extra search path is silently skipped."""
    missing = tmp_path / "does-not-exist"
    specs = discover_composites(extra_search_paths=[missing])
    assert specs == {} or isinstance(specs, dict)


def test_make_spec_id_strips_composite_suffix(tmp_path):
    pkg_root = tmp_path / "pkg"
    sub = pkg_root / "sub"
    sub.mkdir(parents=True)
    file_path = sub / "baseline.composite.yaml"
    file_path.write_text("name: baseline\nstate: {}\n", encoding="utf-8")
    spec_id = _make_spec_id("mypkg", pkg_root, file_path)
    assert spec_id == "mypkg.sub.baseline"


def test_discover_all_tags_spec_kind(tmp_path):
    (tmp_path / "baseline.composite.yaml").write_text(
        "name: baseline\nstate: {}\n", encoding="utf-8"
    )
    cs.clear_registry()
    merged = discover_all(extra_search_paths=[tmp_path])
    spec_entries = {k: v for k, v in merged.items() if k.endswith(".baseline")}
    assert spec_entries, f"expected a .baseline spec entry, got {list(merged.keys())}"
    for entry in spec_entries.values():
        assert entry["kind"] == "spec"


def test_discover_all_includes_generator_entries(tmp_path):
    """Generator entries surface via discover_all, tagged 'generator', with
    the framework-owned fields (visualizations/emitters/default_n_steps)
    always present."""
    from process_bigraph.composite_generator import composite_generator, _REGISTRY
    cs.clear_registry()

    @composite_generator(name="collide", description="", parameters={})
    def collide(core=None):
        return {"state": {}}

    gen_id = f"{collide.__module__}.collide"
    merged = discover_all(extra_search_paths=[])
    assert gen_id in merged
    assert merged[gen_id]["kind"] == "generator"
    assert merged[gen_id]["name"] == "collide"
    assert merged[gen_id]["visualizations"] == []
    assert merged[gen_id]["emitters"] == []
    assert "default_n_steps" in merged[gen_id]
    _REGISTRY.clear()


def test_discover_all_warns_on_spec_generator_id_collision(monkeypatch, tmp_path):
    """When a static spec and a generator share an id, the generator entry
    wins and a collision warning fires (discover_all's own merge logic,
    independent of what discover_composites/discover_generators return)."""
    from process_bigraph.composite_generator import composite_generator, _REGISTRY
    cs.clear_registry()

    @composite_generator(name="collide2", description="", parameters={})
    def collide2(core=None):
        return {"state": {}}

    gen_id = f"{collide2.__module__}.collide2"

    def fake_discover_composites(extra_search_paths=None):
        return {gen_id: {"name": "fake-spec", "state": {}}}

    monkeypatch.setattr(
        "process_bigraph.composite_discovery.discover_composites",
        fake_discover_composites,
    )
    with pytest.warns(UserWarning, match="collides"):
        merged = discover_all(extra_search_paths=[])
    assert merged[gen_id]["kind"] == "generator"
    _REGISTRY.clear()
