from bigraph_schema.contract import ProcessContract, narrow_condition
from process_bigraph.contract_strict import compile_contract, CompiledContract, ContractViolation, _eval_conditions, _compile_condition


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
