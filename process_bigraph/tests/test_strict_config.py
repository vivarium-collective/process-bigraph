"""Tests for opt-in strict-config mode (issue #232).

Unknown / typo'd config keys passed to a Process or Step are silently ignored
by default. These tests pin both halves of the contract:

* default (strict off) -> undeclared keys are accepted, as before;
* strict on -> undeclared keys raise a clear error naming the key and the
  process type, while declared keys are always accepted.
"""

import pytest

from process_bigraph import allocate_core
from process_bigraph.composite import Process, Step, validate_config_keys


class _KnownStep(Step):
    config_schema = {'a': 'integer', 'b': 'float'}

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def update(self, state):
        return {}


class _KnownProcess(Process):
    config_schema = {'rate': 'float'}

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def update(self, state, interval):
        return {}


# --- default OFF: current behavior is preserved -----------------------------

def test_unknown_key_ignored_when_strict_off_step():
    core = allocate_core()
    # no error — undeclared 'bogus_key' is silently accepted (legacy behavior)
    step = _KnownStep({'a': 1, 'b': 2.0, 'bogus_key': 99}, core=core)
    assert step.config['a'] == 1


def test_unknown_key_ignored_when_strict_off_process():
    core = allocate_core()
    proc = _KnownProcess({'rate': 1.5, 'typo': 'x'}, core=core)
    assert proc.config['rate'] == 1.5


def test_default_allocate_core_is_not_strict():
    core = allocate_core()
    assert getattr(core, 'strict_config', False) is False


# --- strict ON: undeclared keys raise ---------------------------------------

def test_unknown_key_raises_when_strict_on_step():
    core = allocate_core(strict=True)
    with pytest.raises(ValueError) as exc:
        _KnownStep({'a': 1, 'b': 2.0, 'bogus_key': 99}, core=core)
    msg = str(exc.value)
    assert 'bogus_key' in msg          # names the offending key
    assert '_KnownStep' in msg         # names the process type


def test_unknown_key_raises_when_strict_on_process():
    core = allocate_core(strict=True)
    with pytest.raises(ValueError) as exc:
        _KnownProcess({'rate': 1.0, 'wrongkey': 1}, core=core)
    assert 'wrongkey' in str(exc.value)


def test_strict_reports_all_unknown_keys():
    core = allocate_core(strict=True)
    with pytest.raises(ValueError) as exc:
        _KnownStep({'a': 1, 'nope1': 1, 'nope2': 2}, core=core)
    msg = str(exc.value)
    assert 'nope1' in msg and 'nope2' in msg


# --- strict ON: declared keys are always accepted ---------------------------

def test_known_keys_accepted_when_strict_on_step():
    core = allocate_core(strict=True)
    step = _KnownStep({'a': 1, 'b': 2.0}, core=core)
    assert step.config['a'] == 1 and step.config['b'] == 2.0


def test_known_keys_accepted_when_strict_on_process():
    core = allocate_core(strict=True)
    proc = _KnownProcess({'rate': 2.0}, core=core)
    assert proc.config['rate'] == 2.0


def test_empty_config_accepted_when_strict_on():
    core = allocate_core(strict=True)
    # defaults fill in; no undeclared keys present
    _KnownStep({}, core=core)
    _KnownStep(None, core=core)


# --- enabling mechanism -----------------------------------------------------

def test_strict_toggle_on_existing_core():
    core = allocate_core()
    core.strict_config = True
    with pytest.raises(ValueError):
        _KnownStep({'bogus': 1}, core=core)


def test_allocate_core_strict_sets_flag():
    assert allocate_core(strict=True).strict_config is True


# --- helper unit ------------------------------------------------------------

def test_validate_config_keys_noop_without_core():
    # no core -> no-op (edge construction handles the missing-core error)
    validate_config_keys(_KnownStep, {'anything': 1}, None)


def test_validate_config_keys_noop_when_not_strict():
    core = allocate_core()
    # strict off -> no raise even with undeclared keys
    validate_config_keys(_KnownStep, {'anything': 1}, core)
