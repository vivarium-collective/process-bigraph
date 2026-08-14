"""Tests for process_bigraph.workflow.provision (provision_core)."""
from process_bigraph import allocate_core
from process_bigraph.workflow.provision import provision_core

_MARK = {}

def mark_core(core):
    _MARK['n'] = _MARK.get('n', 0) + 1
    core._prov = True
    return core


def test_string_provider():
    c = provision_core(allocate_core(), ['process_bigraph.tests.test_provision:mark_core'])
    assert getattr(c, '_prov', False)


def test_noop_empty():
    c = allocate_core()
    assert provision_core(c, []) is c


def test_ray_shim_delegates():
    from process_bigraph.protocols.ray import _apply_type_providers
    c = allocate_core()
    _MARK['n'] = 0
    _apply_type_providers(c, [('process_bigraph.tests.test_provision', 'mark_core', (), {})])
    assert getattr(c, '_prov', False)  # ray path now honors the provider (and its return)
