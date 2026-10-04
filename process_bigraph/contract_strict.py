"""Opt-in runtime contract checking (spec §6).

A core setting ``contract_strict`` in {off, raise, record}; off by default and
zero-cost. When on, the Composite seam checks each process's ``pre`` conditions
and input-port bounds before update(), and ``post``/``invariant`` conditions and
output-port bounds after. Output values are reconstructed post-state
(inputs[port] + delta[port]) for numeric deltas; non-numeric/sentinel deltas are
skipped. Units are not enforced at runtime (metadata; handled at wire-compile).

Scope (v1): only the Composite.process_update / _project_process_update seam is
checked. _run_tick_lifecycle, direct run_step invokes, and pooled/ray reconfigure
validity re-checks are NOT checked. Advisory framing: strict mode verifies the
contract on the trajectories actually run — a check, not a proof.
"""
from dataclasses import dataclass, field

from bigraph_schema.contract_expr import parse, names_in, evaluate, ExprError


class ContractViolation(Exception):
    """Raised (in 'raise' mode) when a process violates its contract at runtime."""


@dataclass
class CompiledCondition:
    kind: str
    name: str
    ast: object
    names: frozenset
    tol: float


@dataclass
class CompiledContract:
    conditions: list = field(default_factory=list)   # pre/post/invariant only
    input_bounds: dict = field(default_factory=dict)   # {port: (lo, hi)}
    output_bounds: dict = field(default_factory=dict)


def _bounds(port_schema):
    """(lo, hi) numeric bounds from a port schema (raw dict or resolved Range);
    (None, None) when unbounded or non-numeric."""
    if isinstance(port_schema, dict):
        lo, hi = port_schema.get('_min'), port_schema.get('_max')
    else:
        lo, hi = getattr(port_schema, '_min', None), getattr(port_schema, '_max', None)
    lo = lo if isinstance(lo, (int, float)) else None
    hi = hi if isinstance(hi, (int, float)) else None
    return lo, hi


def _compile_condition(condition):
    try:
        ast = parse(condition['expr'])
    except (ExprError, KeyError):
        return None
    return CompiledCondition(
        kind=condition['kind'], name=condition.get('name', '?'),
        ast=ast, names=frozenset(names_in(ast)), tol=condition.get('tol', 0.0) or 0.0)


def compile_contract(contract):
    """Pre-parse a ProcessContract's conditions + port bounds once, for fast
    per-tick checking. Returns None if there is no contract."""
    if contract is None:
        return None
    compiled = CompiledContract()
    for condition in contract.conditions():
        cc = _compile_condition(condition)
        if cc is None or cc.kind == 'validity':   # validity (config-domain) deferred in v1
            continue
        compiled.conditions.append(cc)
    face = getattr(contract, 'face', None) or {}
    for port, schema in (face.get('inputs') or {}).items():
        lo, hi = _bounds(schema)
        if lo is not None or hi is not None:
            compiled.input_bounds[port] = (lo, hi)
    for port, schema in (face.get('outputs') or {}).items():
        lo, hi = _bounds(schema)
        if lo is not None or hi is not None:
            compiled.output_bounds[port] = (lo, hi)
    return compiled


def _eval_conditions(conditions, env):
    """Evaluate each compiled condition against env; return (condition, reason)
    for the ones that fail. A condition referencing a name absent from env is
    skipped (cannot be evaluated). Never raises out of this function.
    """
    fails = []
    for cc in conditions:
        if not cc.names.issubset(env.keys()):
            continue   # missing binding → unevaluable → skip
        try:
            ok = evaluate(cc.ast, env, tol=cc.tol)
        except ExprError:
            continue   # defensive: treat an unexpected binding error as a skip
        if not ok:
            fails.append((cc, f'{cc.kind} {cc.name!r} failed: {_fmt(cc, env)}'))
    return fails


def _fmt(cc, env):
    referenced = {f'{".".join(n)}={env.get(n)!r}' for n in cc.names if n in env}
    return ', '.join(sorted(referenced))


def _bounds_violations(bounds, values, direction):
    """(code, reason) for each port whose numeric value is out of [lo, hi].
    Non-numeric values are skipped (can't bound-check)."""
    code = f'{direction}_bounds'
    out = []
    for port, (lo, hi) in bounds.items():
        value = values.get(port)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if lo is not None and value < lo:
            out.append((code, f'{direction} {port!r}={value} below _min {lo}'))
        if hi is not None and value > hi:
            out.append((code, f'{direction} {port!r}={value} above _max {hi}'))
    return out


def check_pre(compiled, inputs):
    """Violations from `pre` conditions + input-port bounds, against the input
    state (port-keyed dict). Returns [(code, reason)]."""
    env = {('inputs', port): value for port, value in inputs.items()}
    fails = [('pre', reason) for _cc, reason in
             _eval_conditions([c for c in compiled.conditions if c.kind == 'pre'], env)]
    fails.extend(_bounds_violations(compiled.input_bounds, inputs, 'input'))
    return fails


def _reconstruct_outputs(inputs, delta):
    """Post-state values per output port. For a numeric delta, inputs[port] +
    delta (or the delta itself when the port has no input). A non-numeric,
    sentinel, or list delta is omitted (its output check is skipped)."""
    out = {}
    if not isinstance(delta, dict):
        return out
    for port, change in delta.items():
        if isinstance(change, bool) or not isinstance(change, (int, float)):
            continue   # sentinel / dict / list / non-numeric → skip this port
        base = inputs.get(port)
        out[port] = (base + change) if isinstance(base, (int, float)) and not isinstance(base, bool) else change
    return out


def check_post(compiled, inputs, delta):
    """Violations from `post`+`invariant` conditions and output-port bounds,
    against the reconstructed post-state. Returns [(code, reason)]."""
    outputs = _reconstruct_outputs(inputs, delta)
    env = {('inputs', port): value for port, value in inputs.items()}
    env.update({('outputs', port): value for port, value in outputs.items()})
    checked = [c for c in compiled.conditions if c.kind in ('post', 'invariant')]
    fails = [(cc.kind, reason) for cc, reason in _eval_conditions(checked, env)]
    fails.extend(_bounds_violations(compiled.output_bounds, outputs, 'output'))
    return fails


def handle_violations(violations, *, mode, phase, path, cls, emitter, global_time):
    """Dispatch contract violations. raise → ContractViolation (fail-loud);
    record → one contract.violation event, then continue. No-op if empty."""
    if not violations:
        return
    summary = '; '.join(f'[{code}] {reason}' for code, reason in violations)
    where = '.'.join(str(p) for p in (path or ()))
    if mode == 'raise':
        raise ContractViolation(f'{cls} at {where!r} ({phase}): {summary}')
    if mode == 'record' and emitter is not None:
        emitter.event('contract.violation', level='warning', component='contract',
                      path=list(path or ()), cls=cls, phase=phase,
                      violations=[{'code': c, 'reason': r} for c, r in violations],
                      global_time=global_time)
