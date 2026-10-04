from process_bigraph import allocate_core
from process_bigraph.composite import Process, Step
from process_bigraph.draft_process import DraftProcess
from process_bigraph.contract_drift import audit_process_drift, Finding, AuditReport, _declared_ports


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
