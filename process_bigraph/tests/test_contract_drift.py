from process_bigraph import allocate_core
from process_bigraph.composite import Process, Step
from process_bigraph.draft_process import DraftProcess
from process_bigraph.contract_drift import audit_process_drift, audit_registry_drift, Finding, AuditReport, _declared_ports, _update_ast, _state_param, _read_ports, _written_ports


class _BareProcess(Process):     # no update override → inherits the base no-op
    def inputs(self): return {'x': 'float'}
    def outputs(self): return {'y': 'float'}


class _RealProcess(Process):
    def inputs(self): return {'x': 'float'}
    def outputs(self): return {'y': 'float'}
    def update(self, state, interval): return {'y': state['x']}


def test_noop_update_is_flagged():
    report = audit_process_drift(_BareProcess)
    assert isinstance(report, AuditReport)
    assert any(f.code == 'noop_update' and f.severity == 'warning' for f in report.findings)


def test_real_process_update_not_flagged_as_noop():
    report = audit_process_drift(_RealProcess)
    assert not any(f.code == 'noop_update' for f in report.findings)


def test_draft_process_noop_exempt():
    class _Draft(DraftProcess):
        DRAFT_INPUTS = {'x': 'float'}
        DRAFT_OUTPUTS = {'y': 'float'}
    report = audit_process_drift(_Draft)       # must not raise
    assert not any(f.code == 'noop_update' for f in report.findings)


def test_declared_ports_from_class_and_instance():
    core = allocate_core()
    ins, outs = _declared_ports(_RealProcess, core)
    assert ins == {'x'} and outs == {'y'}
    inst = _RealProcess({}, core=core)
    ins2, outs2 = _declared_ports(inst, core)
    assert ins2 == {'x'} and outs2 == {'y'}


class _Reader(Process):
    def inputs(self): return {'a': 'float', 'b': 'float'}
    def outputs(self): return {'c': 'float'}
    def update(self, state, interval):
        total = state['a'] + state.get('b', 0.0) + state['ghost']
        return {'c': total}


def test_read_ports_and_state_param():
    fn = _update_ast(_Reader)
    assert fn is not None
    assert _state_param(fn) == 'state'
    reads, unan = _read_ports(fn, 'state')
    assert reads == {'a', 'b', 'ghost'} and unan == []


class _Opaque(Process):
    def inputs(self): return {'a': 'float'}
    def outputs(self): return {'c': 'float'}
    def update(self, state, interval):
        return {'c': self.core.helper(state)}      # state passed whole


def test_opaque_state_use_is_unanalyzable():
    fn = _update_ast(_Opaque)
    reads, unan = _read_ports(fn, _state_param(fn))
    assert unan and any('whole' in r for r in unan)


class _LiteralWriter(Process):
    def inputs(self): return {'a': 'float'}
    def outputs(self): return {'c': 'float'}
    def update(self, state, interval):
        return {'c': state['a'], 'd': 1.0}


def test_written_ports_literal():
    writes, unan = _written_ports(_update_ast(_LiteralWriter))
    assert writes == {'c', 'd'} and unan == []


class _LocalWriter(Process):
    def inputs(self): return {'a': 'float'}
    def outputs(self): return {'c': 'float'}
    def update(self, state, interval):
        out = {'c': state['a']}
        return out


def test_written_ports_single_local_dict():
    writes, unan = _written_ports(_update_ast(_LocalWriter))
    assert writes == {'c'} and unan == []


class _ComputedWriter(Process):
    def inputs(self): return {'a': 'float'}
    def outputs(self): return {'c': 'float'}
    def update(self, state, interval):
        return dict(self._compute(state))        # not a dict literal


def test_written_ports_computed_is_unanalyzable():
    writes, unan = _written_ports(_update_ast(_ComputedWriter))
    assert unan and writes == set()


def test_undeclared_read_and_write_flagged():
    report = audit_process_drift(_Reader, allocate_core())   # reads 'ghost' (undeclared input)
    assert any(f.code == 'undeclared_input_read' and 'ghost' in f.message for f in report.findings)
    assert report.ok is True   # advisory, no error


def test_clean_process_has_no_warnings():
    report = audit_process_drift(_RealProcess, allocate_core())
    assert not any(f.severity == 'warning' for f in report.findings)


def test_undeclared_output_write_flagged():
    report = audit_process_drift(_LiteralWriter, allocate_core())  # writes 'd' (undeclared output)
    assert any(f.code == 'undeclared_output_write' and 'd' in f.message for f in report.findings)


def test_source_unavailable_is_info_not_crash(monkeypatch):
    import process_bigraph.contract_drift as cd
    monkeypatch.setattr(cd, '_update_ast', lambda cls: None)
    report = audit_process_drift(_RealProcess, allocate_core())   # must not raise
    assert any(f.code == 'unanalyzable_update' and f.severity == 'info' for f in report.findings)


def test_registry_driver_audits_registered_processes():
    core = allocate_core()
    core.register_link('bare_proc', _BareProcess)
    core.register_link('real_proc', _RealProcess)
    reports = audit_registry_drift(core)
    assert 'bare_proc' in reports and 'real_proc' in reports
    assert any(f.code == 'noop_update' for f in reports['bare_proc'].findings)
    assert not any(f.severity == 'warning' for f in reports['real_proc'].findings)


def test_end_to_end_drift_surfaced():
    """A process whose update reads/writes ports it never declared is flagged
    on exactly those ports; a faithful one is clean."""
    core = allocate_core()
    report = audit_process_drift(_Reader, core)
    codes = {(f.code, f.severity) for f in report.findings}
    assert ('undeclared_input_read', 'warning') in codes
    assert report.ok is True


def test_empty_declared_ports_skips_drift_as_info():
    class _EmptyFace(Process):
        def inputs(self): return {}
        def outputs(self): return {}
        def update(self, state, interval): return {'y': state['x']}
    report = audit_process_drift(_EmptyFace, allocate_core())
    assert not any(f.severity == 'warning' for f in report.findings)
    assert any(f.code == 'unanalyzable_update' for f in report.findings)


def test_invoke_delegating_noop_exempt():
    class _InvokeWorker(Process):
        def inputs(self): return {'x': 'float'}
        def outputs(self): return {'y': 'float'}
        def invoke(self, interval=None): return {'y': 1.0}
    report = audit_process_drift(_InvokeWorker, allocate_core())
    assert not any(f.code == 'noop_update' for f in report.findings)


def test_base_process_and_step_not_flagged_noop():
    assert not any(f.code == 'noop_update' for f in audit_process_drift(Process).findings)
    assert not any(f.code == 'noop_update' for f in audit_process_drift(Step).findings)


def test_state_store_subscript_not_counted_as_read():
    class _Mutator(Process):
        def inputs(self): return {'x': 'float'}
        def outputs(self): return {'y': 'float'}
        def update(self, state, interval):
            state['tmp'] = 1.0
            return {'y': state['x']}
    report = audit_process_drift(_Mutator, allocate_core())
    assert not any(f.code == 'undeclared_input_read' for f in report.findings)
