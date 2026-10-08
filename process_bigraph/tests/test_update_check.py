"""Debug update checking (issue #99): ``check_updates`` in {off, record, raise}."""
import math

import numpy as np
import pytest

from process_bigraph import Composite, Process, Step, allocate_core
from process_bigraph import update_check
from process_bigraph.update_check import (
    UpdateViolation, check_update, has_structural_keys, iter_non_finite, validate_mode)


# ---------------------------------------------------------------------------
# unit: check_update
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def core():
    return allocate_core()


PORTS = {'x': 'float', 'counts': 'map[float]', 'n': 'integer'}


def _codes(violations):
    return [code for code, _ in violations]


def test_clean_update_has_no_violations(core):
    assert check_update({'x': 1.5, 'counts': {'a': 1.0}, 'n': 2}, PORTS, core) == []


def test_none_and_empty_updates_are_clean(core):
    assert check_update(None, PORTS, core) == []
    assert check_update({}, PORTS, core) == []


def test_non_dict_update_is_flagged(core):
    assert _codes(check_update([1.0], PORTS, core)) == ['not_a_dict']


def test_undeclared_port_is_flagged_and_names_declared_ports(core):
    violations = check_update({'xx': 1.0}, PORTS, core)
    assert _codes(violations) == ['undeclared_port']
    reason = violations[0][1]
    assert "'xx'" in reason and 'counts, n, x' in reason


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), np.float32('nan')])
def test_non_finite_scalar_is_flagged(core, bad):
    assert 'non_finite' in _codes(check_update({'x': bad}, PORTS, core))


def test_non_finite_nested_value_reports_its_path(core):
    violations = check_update({'counts': {'a': 1.0, 'b': math.inf}}, PORTS, core)
    non_finite = [reason for code, reason in violations if code == 'non_finite']
    assert len(non_finite) == 1 and "'counts.b'" in non_finite[0]


def test_non_finite_array_element_reports_first_index():
    found = list(iter_non_finite(np.array([[1.0, 2.0], [np.nan, np.inf]]), ('arr',)))
    assert len(found) == 1
    path, value = found[0]
    assert path == ('arr', (1, 0)) and math.isnan(value)


def test_integer_bool_and_string_values_are_not_non_finite():
    assert list(iter_non_finite({'a': 1, 'b': True, 'c': 'nan', 'd': np.array([1, 2])})) == []


def test_complex_non_finite_is_flagged():
    assert len(list(iter_non_finite(complex(1.0, math.nan)))) == 1


def test_type_mismatch_is_flagged_with_rendered_schema(core):
    violations = check_update({'x': 'oops'}, PORTS, core)
    assert _codes(violations) == ['type_mismatch']
    assert "'oops'" in violations[0][1] and 'schema float' in violations[0][1]


def test_integer_port_rejects_float(core):
    assert _codes(check_update({'n': 2.5}, PORTS, core)) == ['type_mismatch']


def test_structural_delta_skips_type_check_but_not_non_finite(core):
    # ``_add`` is an instruction to the map type, not a map value: no type_mismatch.
    assert check_update({'counts': {'_add': {'z': 1.0}}}, PORTS, core) == []
    # ...but a NaN carried inside it is still caught.
    assert _codes(check_update({'counts': {'_add': {'z': math.nan}}}, PORTS, core)) == ['non_finite']


def test_has_structural_keys():
    assert has_structural_keys({'_remove': ['a']})
    assert has_structural_keys({'a': [{'_add': {}}]})
    assert not has_structural_keys({'a': {'b': 1.0}})


def test_validate_mode():
    assert validate_mode(None) == 'off'
    assert validate_mode('record') == 'record'
    with pytest.raises(ValueError):
        validate_mode('loud')


# ---------------------------------------------------------------------------
# integration: the Composite seam
# ---------------------------------------------------------------------------

class Buggy(Process):
    """Writes ``x``; ``mode`` selects how the update goes wrong."""
    config_schema = {'mode': 'string{ok}'}

    def inputs(self):
        return {'x': 'float'}

    def outputs(self):
        return {'x': 'float'}

    def update(self, state, interval):
        mode = self.config['mode']
        if mode == 'typo':
            return {'xx': 1.0}
        if mode == 'nan':
            return {'x': float('nan')}
        if mode == 'type':
            return {'x': 'oops'}
        return {'x': 1.0}


class NanStep(Step):
    def inputs(self):
        return {'x': 'float'}

    def outputs(self):
        return {'y': 'float'}

    def update(self, state):
        return {'y': float('nan')}


@pytest.fixture
def buggy_core():
    core = allocate_core()
    core.register_link('Buggy', Buggy)
    core.register_link('NanStep', NanStep)
    return core


def _composite(core, mode, check_updates=None):
    doc = {
        'x': 0.0,
        'p': {
            '_type': 'process', 'address': 'local:Buggy', 'config': {'mode': mode},
            'interval': 1.0, 'inputs': {'x': ['x']}, 'outputs': {'x': ['x']}}}
    config = {'state': doc}
    if check_updates is not None:
        config['check_updates'] = check_updates
    return Composite(config, core=core)


def test_off_by_default_and_typo_is_silently_dropped(buggy_core):
    """Documents the failure #99 exists for: without checking, a misspelled
    port is dropped and the run looks fine."""
    composite = _composite(buggy_core, 'typo')
    composite.run(3)
    assert composite._check_updates == 'off'
    assert composite.state['x'] == 0.0
    assert composite.update_violations == []


def test_clean_process_runs_unchanged_with_checks_on(buggy_core):
    composite = _composite(buggy_core, 'ok', check_updates='raise')
    composite.run(3)
    assert composite.state['x'] == 3.0
    assert composite.update_violations == []


@pytest.mark.parametrize('mode,code', [
    ('typo', 'undeclared_port'), ('nan', 'non_finite'), ('type', 'type_mismatch')])
def test_raise_mode_names_process_and_problem(buggy_core, mode, code):
    composite = _composite(buggy_core, mode, check_updates='raise')
    with pytest.raises(UpdateViolation) as excinfo:
        composite.run(1)
    message = str(excinfo.value)
    assert "Buggy at 'p'" in message and f'[{code}]' in message


def test_record_mode_collects_and_continues(buggy_core):
    composite = _composite(buggy_core, 'nan', check_updates='record')
    composite.run(3)   # keeps running despite the NaN
    assert len(composite.update_violations) == 3
    first = composite.update_violations[0]
    assert first['path'] == ['p'] and first['cls'] == 'Buggy'
    assert first['violations'][0]['code'] == 'non_finite'


def test_record_mode_is_bounded(buggy_core, monkeypatch):
    monkeypatch.setattr(update_check, 'MAX_RECORDED', 2)
    composite = _composite(buggy_core, 'nan', check_updates='record')
    composite.run(5)
    assert len(composite.update_violations) == 2
    assert composite.update_violations_dropped == 3


def test_step_updates_are_checked(buggy_core):
    doc = {
        'x': 1.0, 'y': 0.0,
        's': {'_type': 'step', 'address': 'local:NanStep',
              'inputs': {'x': ['x']}, 'outputs': {'y': ['y']}}}
    with pytest.raises(UpdateViolation) as excinfo:
        Composite({'state': doc, 'check_updates': 'raise', 'run_steps_on_init': True},
                  core=buggy_core)
    assert "NanStep at 's'" in str(excinfo.value)


def test_core_wide_setting_turns_checking_on(buggy_core):
    buggy_core.check_updates = 'raise'
    composite = _composite(buggy_core, 'typo')
    with pytest.raises(UpdateViolation):
        composite.run(1)


def test_unknown_mode_is_rejected_at_construction(buggy_core):
    with pytest.raises(ValueError):
        _composite(buggy_core, 'ok', check_updates='loud')
