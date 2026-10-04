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
    """True iff cls is a concrete process that has not overridden update AND
    does not do its work via an invoke() override. The base Process/Step
    classes and invoke-delegating processes (e.g. CompositeTask) are exempt."""
    from process_bigraph.composite import Process, Step
    if cls is Process or cls is Step:
        return False
    if not (cls.update is Process.update or cls.update is Step.update):
        return False
    base_invokes = {getattr(Process, 'invoke', None), getattr(Step, 'invoke', None)}
    cls_invoke = getattr(cls, 'invoke', None)
    if cls_invoke is not None and cls_invoke not in base_invokes:
        return False   # work delegated to invoke()
    return True


def _is_draft(cls):
    """A DraftProcess's no-op is intentional; exempt it."""
    try:
        from process_bigraph.draft_process import DraftProcess
    except ImportError:
        return False
    return isinstance(cls, type) and issubclass(cls, DraftProcess)


def _update_ast(cls):
    """The FunctionDef for cls.update, or None if source is unavailable/unparsable."""
    try:
        source = inspect.getsource(cls.update)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'update':
            return node
    return None


def _state_param(fn_node):
    """The name of the state parameter (first after self), or None."""
    args = fn_node.args.args
    return args[1].arg if len(args) > 1 else None


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _read_ports(fn_node, state_name):
    """Ports read off the state param, plus reasons the read is unanalyzable.

    Catches ``state['x']`` and ``state.get('x')`` with constant keys. A
    non-constant subscript or the whole ``state`` passed to a call is recorded
    as unanalyzable rather than a port.
    """
    reads = set()
    unanalyzable = []
    if not state_name:
        return reads, ['no state parameter']
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == state_name:
            if isinstance(node.ctx, ast.Store):
                continue
            key = _const_str(node.slice)
            if key is not None:
                reads.add(key)
            else:
                unanalyzable.append('non-constant subscript on state')
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == state_name
              and node.func.attr == 'get' and node.args):
            key = _const_str(node.args[0])
            if key is not None:
                reads.add(key)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == state_name:
                    unanalyzable.append('state passed whole to a call')
    return reads, unanalyzable


def _dict_keys(dict_node):
    """(constant-string keys, has_dynamic) for an ast.Dict."""
    keys = set()
    has_dynamic = False
    for key in dict_node.keys:
        if key is None:        # ** spread
            has_dynamic = True
            continue
        name = _const_str(key)
        if name is not None:
            keys.add(name)
        else:
            has_dynamic = True
    return keys, has_dynamic


def _written_ports(fn_node):
    """Ports written by the update's return, plus unanalyzable reasons.

    Resolves a returned dict literal and a single local assigned a dict literal
    then returned. A computed/delegated/absent return is recorded as
    unanalyzable rather than guessed.
    """
    writes = set()
    unanalyzable = []
    dict_locals = {}
    for node in ast.walk(fn_node):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            dict_locals[node.targets[0].id] = _dict_keys(node.value)

    returns = [n for n in ast.walk(fn_node) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return writes, ['update returns nothing']

    for ret in returns:
        value = ret.value
        if isinstance(value, ast.Dict):
            keys, dynamic = _dict_keys(value)
            writes |= keys
            if dynamic:
                unanalyzable.append('returned dict has a non-constant or ** key')
        elif isinstance(value, ast.Name) and value.id in dict_locals:
            keys, dynamic = dict_locals[value.id]
            writes |= keys
            if dynamic:
                unanalyzable.append('returned local dict built with a non-constant or ** key')
        else:
            unanalyzable.append('return value is not a dict literal')
    return writes, unanalyzable


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

    declared_inputs, declared_outputs = _declared_ports(proc_or_cls, core)

    fn_node = _update_ast(cls)
    if fn_node is None:
        findings.append(Finding('info', 'unanalyzable_update', f'{cls.__name__}.update',
                                 'update source unavailable for static drift analysis'))
        return AuditReport(ok=True, findings=findings)

    state_name = _state_param(fn_node)
    reads, read_unanalyzable = _read_ports(fn_node, state_name)
    if declared_inputs:
        for port in sorted(reads - declared_inputs):
            findings.append(Finding('warning', 'undeclared_input_read', f'{cls.__name__}.update',
                                     f'reads state[{port!r}] but {port!r} is not a declared input port'))
    else:
        findings.append(Finding('info', 'unanalyzable_update', f'{cls.__name__}.update',
                                 'declared input ports are empty or indeterminate; skipping input-drift check'))
    for reason in read_unanalyzable:
        findings.append(Finding('info', 'unanalyzable_update', f'{cls.__name__}.update', reason))

    writes, write_unanalyzable = _written_ports(fn_node)
    if declared_outputs:
        for port in sorted(writes - declared_outputs):
            findings.append(Finding('warning', 'undeclared_output_write', f'{cls.__name__}.update',
                                     f'writes {port!r} but {port!r} is not a declared output port'))
    else:
        findings.append(Finding('info', 'unanalyzable_update', f'{cls.__name__}.update',
                                 'declared output ports are empty or indeterminate; skipping output-drift check'))
    for reason in write_unanalyzable:
        findings.append(Finding('info', 'unanalyzable_update', f'{cls.__name__}.update', reason))

    return AuditReport(ok=True, findings=findings)


def audit_registry_drift(core):
    """Audit every registered process for drift. Returns {address: AuditReport}.

    Skips registry entries that cannot be resolved to a class, rather than
    failing the whole sweep.
    """
    from process_bigraph.core_introspection import list_processes
    reports = {}
    for address in list_processes(core):
        edge_class = core.link_registry.get(address)
        if edge_class is None:
            continue
        try:
            reports[address] = audit_process_drift(edge_class, core)
        except Exception as error:  # noqa: BLE001 - one bad process must not sink the sweep
            reports[address] = AuditReport(
                ok=True, findings=[Finding('info', 'unanalyzable_update', address, f'audit raised: {error}')])
    return reports
