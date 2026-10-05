from bigraph_schema.core import allocate_core
from bigraph_schema.edge import Edge
from bigraph_schema.contract import ProcessContract, narrow_condition
from process_bigraph.audit_contracts import audit_all


class _Good(Edge):
    contract = narrow_condition(
        ProcessContract(face={'inputs': {'m': {'_type': 'float', '_min': 0}}, 'outputs': {'m': {'_type': 'float', '_min': 0}}}),
        'post', 'outputs.m >= 0', name='nonneg')
    def inputs(self): return {'m': {'_type': 'float', '_min': 0}}
    def outputs(self): return {'m': {'_type': 'float', '_min': 0}}

class _Lying(Edge):
    contract = narrow_condition(ProcessContract(face={'inputs': {'m': 'float'}, 'outputs': {'m': 'float'}}),
                                'post', 'outputs.ghost >= 0', name='bogus')
    def inputs(self): return {'m': 'float'}
    def outputs(self): return {'m': 'float'}

class _Bare(Edge):
    def inputs(self): return {'m': 'float'}
    def outputs(self): return {'m': 'float'}


def test_clean_core_passes():
    core = allocate_core(); core.register_link('good', _Good)
    out = audit_all(core)
    assert out['exit_code'] == 0 and out['errors'] == 0
    assert any(r['status'] == 'pass' for r in out['reports'] if r['cls'].endswith('_Good'))

def test_lying_contract_fails_gate():
    core = allocate_core(); core.register_link('good', _Good); core.register_link('lying', _Lying)
    out = audit_all(core)
    assert out['exit_code'] == 1 and out['errors'] >= 1
    lie = next(r for r in out['reports'] if r['cls'].endswith('_Lying'))
    assert lie['status'] == 'fail' and any('ghost' in f.get('message', '') for f in lie['findings'])

def test_bare_is_not_declared_not_failure():
    core = allocate_core(); core.register_link('bare', _Bare)
    out = audit_all(core)
    bare = next(r for r in out['reports'] if r['cls'].endswith('_Bare'))
    assert bare['status'] in ('not-declared', 'incomplete') and out['exit_code'] == 0

def test_alias_dedup():
    core = allocate_core(); core.register_link('good', _Good); core.register_link('pkg.Good', _Good)
    out = audit_all(core)
    goods = [r for r in out['reports'] if r['cls'].endswith('_Good')]
    assert len(goods) == 1   # one class, audited once despite two addresses

def test_require_declared_floor():
    core = allocate_core(); core.register_link('good', _Good)
    base = audit_all(core)['declared']
    assert audit_all(core, require_declared=base)['exit_code'] == 0
    assert audit_all(core, require_declared=base + 1)['exit_code'] == 1
