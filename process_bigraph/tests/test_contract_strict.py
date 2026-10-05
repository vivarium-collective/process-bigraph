from bigraph_schema.contract import ProcessContract, narrow_condition
from process_bigraph.contract_strict import compile_contract, CompiledContract, ContractViolation, _eval_conditions, _compile_condition, check_pre, check_post


def _contract():
    c = ProcessContract(face={
        'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 10}},
        'outputs': {'m': {'_type': 'float', '_min': 0}}})
    c = narrow_condition(c, 'invariant', 'outputs.m - inputs.m <= tol', name='conservation', tol=1e-9)
    c = narrow_condition(c, 'pre', 'inputs.m >= 0', name='nonneg_in')
    return c


def test_compile_contract_parses_conditions_and_bounds():
    compiled = compile_contract(_contract())
    assert isinstance(compiled, CompiledContract)
    kinds = {cc.kind for cc in compiled.conditions}
    assert kinds == {'invariant', 'pre'}
    # every compiled condition carries a parsed ast + its referenced names
    conservation = next(cc for cc in compiled.conditions if cc.name == 'conservation')
    assert ('outputs', 'm') in conservation.names and ('inputs', 'm') in conservation.names
    assert compiled.input_bounds['m'] == (0, 10)
    assert compiled.output_bounds['m'] == (0, None)


def test_compile_contract_none_is_none():
    assert compile_contract(None) is None


def _cond(kind, expr, tol=0.0, name='c'):
    return _compile_condition({'kind': kind, 'name': name, 'expr': expr, 'tol': tol})


def test_eval_conditions_flags_failures():
    conds = [_cond('invariant', 'outputs.m - inputs.m <= tol', tol=1e-9, name='cons')]
    env = {('inputs', 'm'): 1.0, ('outputs', 'm'): 5.0}   # grew by 4 → violates
    fails = _eval_conditions(conds, env)
    assert len(fails) == 1 and fails[0][0].name == 'cons'


def test_eval_conditions_passes_when_satisfied():
    conds = [_cond('invariant', 'outputs.m - inputs.m <= tol', tol=1e-9)]
    env = {('inputs', 'm'): 1.0, ('outputs', 'm'): 1.0}
    assert _eval_conditions(conds, env) == []


def test_missing_binding_is_skipped():
    conds = [_cond('pre', 'inputs.ghost >= 0')]
    assert _eval_conditions(conds, {('inputs', 'm'): 1.0}) == []   # 'ghost' absent → skip, no crash


def test_check_pre_flags_precondition_and_bounds():
    c = ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 10}}, 'outputs': {}})
    c = narrow_condition(c, 'pre', 'inputs.m >= 1', name='min_in')
    compiled = compile_contract(c)
    # m = 20 → over the _max=10 bound AND satisfies >=1
    fails = check_pre(compiled, {'m': 20.0})
    assert any(code == 'input_bounds' for code, _ in fails)
    # m = 0 → within bounds but violates the >=1 precondition
    fails2 = check_pre(compiled, {'m': 0.0})
    assert any('min_in' in reason for _, reason in fails2)


def test_check_pre_clean():
    c = ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 10}}, 'outputs': {}})
    assert check_pre(compile_contract(c), {'m': 5.0}) == []


def test_check_pre_int_value_not_flagged():
    c = ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0, '_max': 10}}, 'outputs': {}})
    assert check_pre(compile_contract(c), {'m': 5}) == []   # int within range → ok (lenient)


def _conserving():
    c = ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0}},
                              'outputs': {'m': {'_type': 'float', '_min': 0}}})
    return compile_contract(narrow_condition(c, 'invariant', 'outputs.m - inputs.m <= tol',
                                             name='cons', tol=1e-9))


def test_post_conservation_pass_and_fail():
    compiled = _conserving()
    # inputs m=5, delta +0 → outputs m=5, conserved → clean
    assert check_post(compiled, {'m': 5.0}, {'m': 0.0}) == []
    # inputs m=5, delta +4 → outputs m=9 > 5 → violates conservation
    fails = check_post(compiled, {'m': 5.0}, {'m': 4.0})
    assert any('cons' in reason for _, reason in fails)


def test_negative_delta_within_bounds_not_flagged():
    # Nonnegative output port; delta is -3 (a decrement). Reconstructed post = 10-3 = 7 ≥ 0 → OK.
    c = ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0}},
                              'outputs': {'m': {'_type': 'float', '_min': 0}}})
    compiled = compile_contract(c)
    assert check_post(compiled, {'m': 10.0}, {'m': -3.0}) == []   # NOT flagged (the delta alone is negative)


def test_output_bounds_flag_reconstructed_out_of_range():
    c = ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0}},
                              'outputs': {'m': {'_type': 'float', '_min': 0}}})
    compiled = compile_contract(c)
    # inputs 2, delta -5 → reconstructed -3 < 0 → flagged
    fails = check_post(compiled, {'m': 2.0}, {'m': -5.0})
    assert any(code == 'output_bounds' for code, _ in fails)


def test_sentinel_delta_skipped():
    c = ProcessContract(face={'inputs': {}, 'outputs': {'m': {'_type': 'float', '_min': 0}}})
    compiled = compile_contract(c)
    assert check_post(compiled, {}, {'m': {'_add': {'x': 1}}}) == []   # sentinel → skip, no false violation


import pytest
from process_bigraph.contract_strict import handle_violations


class _RecordingEmitter:
    def __init__(self): self.events = []
    def event(self, name, level='info', **payload): self.events.append((name, level, payload))


def test_raise_mode_raises():
    with pytest.raises(ContractViolation) as e:
        handle_violations([('pre', "pre 'x' failed: inputs.x=-1")], mode='raise',
                          phase='pre', path=('p',), cls='P', emitter=None, global_time=0.0)
    assert 'x' in str(e.value)


def test_record_mode_emits_and_continues():
    em = _RecordingEmitter()
    handle_violations([('invariant', "invariant 'cons' failed")], mode='record',
                      phase='post', path=('p',), cls='P', emitter=em, global_time=1.0)
    assert em.events and em.events[0][0] == 'contract.violation' and em.events[0][1] == 'warning'


def test_no_violations_is_noop():
    em = _RecordingEmitter()
    handle_violations([], mode='raise', phase='pre', path=('p',), cls='P', emitter=em, global_time=0.0)
    assert em.events == []


# --- integration: gated hooks in the live Composite run path ---------------
from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Process


class _Grower(Process):
    """Adds `rate` to level each tick, but its contract FALSELY claims conservation."""
    contract = narrow_condition(
        ProcessContract(face={'inputs': {'level': 'float'}, 'outputs': {'level': 'float'}}),
        'invariant', 'outputs.level - inputs.level <= tol', name='conservation', tol=1e-9)
    config_schema = {'rate': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        return {'level': self.config['rate']}


def _composite(strict):
    core = allocate_core()
    core.register_link('_Grower', _Grower)
    doc = {'state': {
        'grow': {'_type': 'process', 'address': 'local:_Grower', 'config': {'rate': 1.0},
                 'inputs': {'level': ['level']}, 'outputs': {'level': ['level']},
                 'interval': 1.0},
        'level': 0.0}}
    if strict is not None:
        doc['contract_strict'] = strict
    return Composite(doc, core=core)


def test_default_is_off():
    sim = _composite(None)
    assert sim._contract_strict == 'off'
    assert sim._compiled_contracts == {}


def test_off_mode_does_not_check():
    sim = _composite('off')
    sim.run(3.0)
    assert sim.state['level'] > 0
    # TODO: assert a contract.violation event reaches the composite emitter
    # (needs a recording event sink wired into the test composite).


def test_raise_mode_halts_on_violation():
    sim = _composite('raise')
    with pytest.raises(ContractViolation):
        sim.run(3.0)


def test_record_mode_continues():
    sim = _composite('record')
    sim.run(3.0)
    assert sim.state['level'] > 0


def test_eval_conditions_swallows_evaluation_errors():
    # a condition that divides by zero must be SKIPPED, not raise out
    conds = [_cond('invariant', 'inputs.a / inputs.b <= tol', tol=0.0, name='div')]
    env = {('inputs', 'a'): 1.0, ('inputs', 'b'): 0.0}
    assert _eval_conditions(conds, env) == []   # ZeroDivisionError swallowed → skip
