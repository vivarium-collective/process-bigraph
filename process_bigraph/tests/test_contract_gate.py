from bigraph_schema.core import allocate_core
from bigraph_schema.edge import Edge
from bigraph_schema.contract import ProcessContract, narrow_condition
from process_bigraph.audit_contracts import audit_all
from process_bigraph import audit_contracts


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


def test_main_returns_zero_on_clean(monkeypatch, capsys):
    core = allocate_core(); core.register_link('good', _Good)
    monkeypatch.setattr('process_bigraph.audit_contracts.allocate_core', lambda *a, **k: core)
    assert audit_contracts.main([]) == 0
    assert 'declared' in capsys.readouterr().out   # the summary report was printed


def test_main_returns_one_on_lying(monkeypatch):
    core = allocate_core(); core.register_link('lying', _Lying)
    monkeypatch.setattr('process_bigraph.audit_contracts.allocate_core', lambda *a, **k: core)
    assert audit_contracts.main([]) == 1


def test_main_json(monkeypatch, capsys):
    core = allocate_core(); core.register_link('good', _Good)
    monkeypatch.setattr('process_bigraph.audit_contracts.allocate_core', lambda *a, **k: core)
    audit_contracts.main(['--json'])
    import json
    data = json.loads(capsys.readouterr().out)
    assert 'reports' in data and 'exit_code' in data


def test_main_require_declared(monkeypatch):
    core = allocate_core(); core.register_link('good', _Good)
    monkeypatch.setattr('process_bigraph.audit_contracts.allocate_core', lambda *a, **k: core)
    assert audit_contracts.main(['--require-declared', '5']) == 1


def test_installed_registry_has_no_contract_errors():
    """Regression gate: no process installed in THIS venv ships a contract with
    an error finding. (Mostly a guard today — few processes declare contracts —
    but it fails loudly if a broken contract ever lands.)"""
    core = allocate_core()
    out = audit_all(core)
    bad = [(r['cls'], r['findings']) for r in out['reports']
           if any(f['severity'] == 'error' for f in r['findings'])]
    assert bad == [], f"processes with contract errors: {bad}"
