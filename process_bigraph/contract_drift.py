"""Declaration↔implementation drift auditor (spec §5 item 2).

Best-effort static lint: AST-introspect a Process/Step `update()` and flag
reads of undeclared input ports, writes of undeclared output ports, the no-op
trap (a non-Draft process left on the inherited ``return {}``), and cases too
dynamic to analyze. Advisory only — Python is dynamic, so this flags clear
mismatches but does not prove conformance; strong "claims-are-true" is runtime
strict-mode (Phase 3). Purely additive; no runtime path changes.
"""
import ast
import inspect
import textwrap

try:
    from bigraph_schema.contract_audit import Finding, AuditReport
except ImportError:  # bigraph-schema predates contract_audit (PyPI 1.6.0)
    from dataclasses import dataclass, field

    @dataclass
    class Finding:
        severity: str   # 'error' | 'warning' | 'info'
        code: str
        where: str
        message: str

    @dataclass
    class AuditReport:
        ok: bool
        findings: list = field(default_factory=list)


def _is_noop_update(cls):
    """True iff cls has not overridden update — it is still the base no-op."""
    from process_bigraph.composite import Process, Step
    return cls.update is Process.update or cls.update is Step.update


def _is_draft(cls):
    """A DraftProcess's no-op is intentional; exempt it."""
    try:
        from process_bigraph.draft_process import DraftProcess
    except ImportError:
        return False
    return isinstance(cls, type) and issubclass(cls, DraftProcess)


def _declared_ports(proc_or_cls, core):
    """(inputs, outputs) declared port-name sets. Accepts a class or instance;
    never raises — an unreadable face yields an empty set on that side.
    """
    instance = proc_or_cls
    if isinstance(proc_or_cls, type):
        try:
            instance = proc_or_cls({}, core=core) if core is not None else proc_or_cls.__new__(proc_or_cls)
        except Exception:  # noqa: BLE001 - construction may need real config; fall back
            instance = proc_or_cls.__new__(proc_or_cls)
    try:
        inputs = set((instance.inputs() or {}).keys())
    except Exception:  # noqa: BLE001 - face read may fail on a bare __new__'d instance
        inputs = set()
    try:
        outputs = set((instance.outputs() or {}).keys())
    except Exception:  # noqa: BLE001
        outputs = set()
    return inputs, outputs


def audit_process_drift(proc_or_cls, core=None):
    """Audit one process (class or instance) for declaration/implementation
    drift. Returns an AuditReport; never raises on a malformed process.
    Drift findings are advisory (warning/info), so ok stays True.
    """
    cls = proc_or_cls if isinstance(proc_or_cls, type) else type(proc_or_cls)
    findings = []

    if _is_noop_update(cls) and not _is_draft(cls):
        findings.append(Finding('warning', 'noop_update', f'{cls.__name__}.update',
                                 'non-draft process inherits the base no-op update (returns {})'))
        return AuditReport(ok=True, findings=findings)

    # Port-drift analysis is added in Tasks 2-5.
    return AuditReport(ok=True, findings=findings)
