"""Tests for process_bigraph.core_introspection (moved from viva-superpowers)."""
import pytest

from process_bigraph.core_introspection import (
    list_processes, list_types, registry_snapshot,
)


class FakeCore:
    """Exercises the callable-method probe path of `_try`."""

    def __init__(self):
        self._links = {"ProcA": object(), "StepB": object()}
        self._types = {"my_type": {"_inherit": "float", "_default": 0.0}}

    def list_processes(self):
        return list(self._links.keys())

    def list_types(self):
        return list(self._types.keys())

    def access(self, name):
        return self._types.get(name, {})


class DictOnlyCore:
    """Exercises the dict-attribute fallback path of `_try` (no callable methods)."""

    _links = {"P": object()}
    _types = {"T": {}}


class EmptyCore:
    """No registry attributes — must raise RuntimeError."""


def test_list_processes_returns_registered():
    assert sorted(list_processes(FakeCore())) == ["ProcA", "StepB"]


def test_list_types_returns_registered():
    assert "my_type" in list_types(FakeCore())


def test_registry_snapshot_is_stable_dict():
    snap = registry_snapshot(FakeCore())
    assert "processes" in snap and "types" in snap
    assert sorted(snap["processes"]) == ["ProcA", "StepB"]
    assert sorted(snap["types"]) == ["my_type"]


def test_dict_only_core_processes():
    """Fallback path: only `_links` dict attribute, no callable methods."""
    assert list_processes(DictOnlyCore()) == ["P"]


def test_dict_only_core_types():
    assert list_types(DictOnlyCore()) == ["T"]


def test_empty_core_raises_runtime_error():
    """Loud failure when no registry attributes exist — message names probed attrs."""
    with pytest.raises(RuntimeError, match="list_processes"):
        list_processes(EmptyCore())
    with pytest.raises(RuntimeError, match="list_types"):
        list_types(EmptyCore())
