"""CI audit gate for declared process contracts (spec section 5).

Audits every registered process's DECLARED contract (static — no instantiation)
and fails (exit 1) on any error-severity finding, or on a declared-coverage
floor. Read the class `.contract` directly (contract_of drops amendments). In
process-bigraph's own CI few processes declare contracts, so the error-gate is
largely a regression guard; the real value is downstream workspaces running
`python -m process_bigraph.audit_contracts` against their registries, and the
--require-declared floor makes coverage visible.
"""
try:
    from bigraph_schema.contract_audit import audit_contract, completeness
    from bigraph_schema.assembly import contract_of
    from bigraph_schema.contract import resolve_contract
    AUDIT_AVAILABLE = True
except Exception:  # noqa: BLE001
    AUDIT_AVAILABLE = False

_INCOMPLETE_BELOW = 0.5


def _class_key(cls):
    return f'{getattr(cls, "__module__", "?")}.{getattr(cls, "__qualname__", getattr(cls, "__name__", "?"))}'


def _contract_for(core, address, cls):
    """Declared contract of a class, amendments preserved. Reads the class
    `.contract` first (contract_of drops amendments), then falls back."""
    declared = getattr(cls, 'contract', None)
    if declared is not None:
        try:
            return resolve_contract(cls)   # normalizes ProcessContract/dict; never raises
        except Exception:  # noqa: BLE001
            pass
    try:
        return contract_of(core, address)
    except Exception:  # noqa: BLE001
        return None


def audit_all(core, *, require_declared=0, strict_resolve=False):
    from process_bigraph.core_introspection import list_processes
    reports = {}
    for address in list_processes(core):
        try:
            cls = core.link_registry.get(address)
        except Exception:  # noqa: BLE001 - per-entry import failure
            cls = None
        if cls is None:
            reports.setdefault(f'unresolvable:{address}',
                               {'address': address, 'cls': address, 'status': 'unresolvable',
                                'grade': None, 'findings': []})
            continue
        key = _class_key(cls)
        if key in reports:
            continue   # alias dedup
        contract = _contract_for(core, address, cls)
        if contract is None:
            reports[key] = {'address': address, 'cls': key, 'status': 'not-declared', 'grade': None, 'findings': []}
            continue
        try:
            report = audit_contract(core, contract)
            grade = completeness(contract)
            findings = [{'severity': f.severity, 'code': f.code, 'where': f.where, 'message': f.message}
                        for f in report.findings]
        except Exception as error:  # noqa: BLE001
            reports[key] = {'address': address, 'cls': key, 'status': 'unresolvable', 'grade': None,
                            'findings': [{'severity': 'error', 'code': 'audit_raised', 'where': key, 'message': str(error)}]}
            continue
        declared = bool(contract.conditions()) or grade >= _INCOMPLETE_BELOW
        if any(f['severity'] == 'error' for f in findings):
            status = 'fail'
        elif not declared:
            status = 'not-declared'
        elif grade < _INCOMPLETE_BELOW:
            status = 'incomplete'
        else:
            status = 'pass'
        reports[key] = {'address': address, 'cls': key, 'status': status, 'grade': round(grade, 3), 'findings': findings}

    rows = list(reports.values())
    errors = sum(1 for r in rows for f in r['findings'] if f['severity'] == 'error')
    if strict_resolve:
        errors += sum(1 for r in rows if r['status'] == 'unresolvable')
    declared_count = sum(1 for r in rows if r['status'] in ('pass', 'incomplete'))
    exit_code = 1 if (errors > 0 or (require_declared > 0 and declared_count < require_declared)) else 0
    return {'reports': rows, 'declared': declared_count, 'errors': errors, 'exit_code': exit_code}
