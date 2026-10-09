"""Debug update checking (issue #99): ``check_updates`` in {off, record, raise}."""
import copy
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


def test_undeclared_port_message_names_the_port_set(core):
    violations = check_update({'zz': 1.0}, PORTS, core, ports_key='inputs')
    assert 'not declared in inputs()' in violations[0][1]


# Updates that ``core.check`` rejects as *states* but that ``apply`` accepts as
# *updates* (review on #231). Each row is (schema, state before, update). The
# test asserts both that apply succeeds and that check_update stays quiet, so
# the rule is tied to what apply actually does.
APPLIES_CLEANLY = [
    ('float', 1.0, 2),                                   # int to a float port
    ('float', 1.0, np.float32(1.5)),
    ('float', 1.0, np.int64(2)),
    ('float', 1.0, np.array(2.0)),                       # 0-d array
    ('float[32]', np.float32(1.0), np.float32(1.0)),     # the declared numpy width
    ('integer', 1, np.int64(2)),
    ('integer', 1, np.uint8(2)),
    ('boolean', False, np.bool_(True)),
    ('range[0,1]', 0.5, -0.25),                          # a change, not a state:
    ('nonnegative', 3.0, -2.0),                          # bounds are contract_strict's job
    ('list[float]', [1.0, 1.0], np.array([1.0, 2.0])),
    ('list[float]', [1.0, 1.0], np.array([1, 2])),
    ('list[float]', [1.0], [2]),
    ('list[integer]', [1, 1], np.array([1, 2], dtype=np.uint8)),
    ('map[float]', {'a': 1.0}, {'a': 2}),
    ('map[float]', {'a': 1.0}, {'a': np.float32(2.0)}),
    ('tuple[float,integer]', (1.0, 1), (1, np.int32(2))),
    ('tree[float]', {'a': {'b': 1.0}}, {'a': {'b': 2}}),
    ('maybe[float]', 1.0, np.float16(1.0)),
    ('overwrite[float]', 1.0, 2),
    ({'u': 'float', 'v': 'integer'}, {'u': 1.0, 'v': 1}, {'u': 1}),   # partial update
    ('array[(3),float]', np.zeros(3), np.array([1, 2, 3])),           # int dtype
    ('array[(3),float]', np.zeros(3), [(0, 1.0)]),                    # sparse update
    ('array[(3),float]', np.zeros(3), {0: 1.0}),                      # per-index update
    ('array[(3),float]', np.zeros(3), 2.0),                           # broadcast scalar
]


@pytest.mark.parametrize('schema,before,delta', APPLIES_CLEANLY,
                         ids=[f'{s}<-{type(d).__name__}' for s, _, d in APPLIES_CLEANLY])
def test_updates_that_apply_cleanly_are_not_type_mismatches(core, schema, before, delta):
    resolved = core.access(schema)
    core.apply(resolved, copy.deepcopy(before), delta)   # raises if apply disagrees
    violations = check_update({'p': delta}, {'p': schema}, core)
    assert 'type_mismatch' not in _codes(violations), violations


# The wrong kind of value for the port. Several of these *do* go through
# apply, but only by storing the wrong type (an integer store becomes 3.5, a
# float store becomes 'z', a pair loses a value), which is what the check is
# for.
STILL_FLAGGED = [
    ('float', 'oops'), ('float', [1.0]), ('float', {'a': 1.0}), ('float', True),
    ('integer', 2.5), ('integer', np.float64(2.0)), ('integer', 'x'),
    ('boolean', 1),
    ('list[float]', 'abc'), ('list[float]', 3.0), ('list[float]', np.array(['a'])),
    ('list[integer]', np.array([1.5])),
    ('map[float]', [1.0]), ('map[float]', {'a': 'z'}),
    ('tuple[float,integer]', (1.0, 2.5)), ('tuple[float,integer]', (1.0,)),
    ('tree[float]', {'a': 'z'}),
    ('overwrite[float]', 'z'),
    ({'u': 'float', 'v': 'integer'}, {'u': 'z'}),
    ('array[(3),float]', 'abc'),
]


@pytest.mark.parametrize('schema,delta', STILL_FLAGGED,
                         ids=[f'{s}<-{d!r}' for s, d in STILL_FLAGGED])
def test_wrong_kind_of_value_is_still_flagged(core, schema, delta):
    assert 'type_mismatch' in _codes(check_update({'p': delta}, {'p': schema}, core))


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


class NumpyFlavoured(Process):
    """Returns the numpy and int updates real models produce."""

    def inputs(self):
        return {'x': 'float', 'v': 'list[float]', 'n': 'integer'}

    def outputs(self):
        return {'x': 'float', 'v': 'list[float]', 'n': 'integer'}

    def update(self, state, interval):
        return {'x': np.float32(0.5), 'v': np.array([1, 2]), 'n': np.int64(1)}


class IntToFloat(Process):
    def inputs(self):
        return {'x': 'float'}

    def outputs(self):
        return {'x': 'float'}

    def update(self, state, interval):
        return {'x': 1}


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
    core.register_link('NumpyFlavoured', NumpyFlavoured)
    core.register_link('IntToFloat', IntToFloat)
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


def test_numpy_and_int_updates_run_clean_in_raise_mode(buggy_core):
    """Review on #231: correct numpy-heavy updates must not halt a run."""
    doc = {
        'x': 0.0, 'v': [0.0, 0.0], 'n': 0,
        'np': {'_type': 'process', 'address': 'local:NumpyFlavoured', 'interval': 1.0,
               'inputs': {'x': ['x'], 'v': ['v'], 'n': ['n']},
               'outputs': {'x': ['x'], 'v': ['v'], 'n': ['n']}},
        'it': {'_type': 'process', 'address': 'local:IntToFloat', 'interval': 1.0,
               'inputs': {'x': ['x']}, 'outputs': {'x': ['x']}}}
    composite = Composite({'state': doc, 'check_updates': 'raise'}, core=buggy_core)
    composite.run(3)
    assert composite.update_violations == []
    assert composite.state['x'] == pytest.approx(4.5)   # 3 x (0.5 + 1)
    assert composite.state['n'] == 3


def test_core_wide_setting_turns_checking_on(buggy_core):
    buggy_core.check_updates = 'raise'
    composite = _composite(buggy_core, 'typo')
    with pytest.raises(UpdateViolation):
        composite.run(1)


def test_unknown_mode_is_rejected_at_construction(buggy_core):
    with pytest.raises(ValueError):
        _composite(buggy_core, 'ok', check_updates='loud')
