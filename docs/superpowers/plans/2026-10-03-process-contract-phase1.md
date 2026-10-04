# Process Contract — Phase 1 (contract model + expression language + static auditor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In bigraph-schema, extend `ProcessContract` with declarative validity conditions (per-port bounds already ride on the type system; invariants / pre / post / parameter-validity become structured predicates on `narrow` amendments), a safe expression language to express them, and a static `audit_contract` that lints a contract — all with **zero runtime/behavior change**.

**Architecture:** Reuse the existing contract machinery: `ProcessContract` already has a `face` (ports→types) and a predicate mechanism via `narrow` amendments (`predicates()`, monotonicity enforced by `amend`). We (a) define a structured predicate `{kind, name, expr, tol}` carried in an amendment's `detail['predicate']`, (b) add a safe parse-to-AST + evaluate expression language (never `eval`), and (c) add `audit_contract` which checks well-formedness, narrow soundness, and a completeness grade. No parallel field set, no changes to `amend`'s monotonicity rules, no engine changes.

**Tech Stack:** Python 3.12, bigraph-schema (current `main`, 1.6.0+), `plum` dispatch (existing), `pint` (existing, for units), `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-03-process-contract-formalization-design.md`

## Global Constraints

- Target **current bigraph-schema `main` (1.6.0+)** — NOT the local `chore/bump-1.4.3` checkout (it lacks `contract.py`'s amendment model, `Site`, `fill_sites`). Task 0 updates the checkout.
- **No AI attribution** in commit messages (no `Co-Authored-By`, no generated-with footer).
- **Never use `eval`/`exec`** on contract expressions — parse to an AST and evaluate against an explicit binding environment.
- **Zero runtime/behavior change** in Phase 1: nothing added here is called during a simulation run; `amend`'s existing monotonicity (narrow may not redefine a face port) is unchanged; a process with no contract still audits as "incomplete," never "error."
- **Pure/append-only contract ops:** like the existing `amend`, every new contract helper returns a NEW contract and never mutates its input.
- bigraph-schema's lint gate is narrow (ruff `F` + `E722`); keep imports used and avoid bare `except:`.

## Review Focus

The spec implies these inputs/failure modes; each line's test is added to the task that owns the code.

- **A predicate expression referencing a name that is not in the face or config** (typo `inputs.glucse`) must be a well-formedness **error**, not a silent pass → Task 7.
- **A malformed expression** (`"sum(outputs.mass =="`, unbalanced) must fail to parse with a clear message, never raise an opaque exception to the caller → Task 2.
- **`eval`-style injection** (`expr: "__import__('os').system('rm -rf /')"`) must be rejected by the grammar (only the whitelisted node kinds parse), never executed → Task 2.
- **A range with `min > max`, or units that don't parse** must be an audit **error** → Task 8.
- **A contract with no declared conditions** (bare `float` ports, no predicates) must audit as **incomplete (info)** and score low, never `error` and never crash → Task 9.

---

## File Structure

- **Create `bigraph_schema/contract_expr.py`** — the safe expression language: `parse(expr) -> Expr` (AST), `names_in(ast) -> set[str]`, `evaluate(ast, env) -> value`. One responsibility: turn a declarative string into a checkable/evaluable, name-introspectable object. No contract knowledge.
- **Modify `bigraph_schema/contract.py`** — add the structured predicate helpers on top of the existing amendment model: a `Predicate` shape + `narrow_predicate(contract, kind, expr, *, name, tol)`, and `ProcessContract.predicates(kind=None)` filtering by kind. Reuses `amend`/`Amendment`.
- **Create `bigraph_schema/contract_audit.py`** — `audit_contract(core, contract_or_instance) -> AuditReport`: the static linter (well-formedness, narrow soundness surfacing, completeness grade). Separate file because it's a distinct concern (analysis, not declaration) and will grow (process-bigraph adds drift checks in Phase 2 by extending the report, not this module).
- **Create tests:** `bigraph_schema/tests/test_contract_expr.py`, `bigraph_schema/tests/test_contract_predicates.py`, `bigraph_schema/tests/test_contract_audit.py`.

---

### Task 0: Refresh the bigraph-schema checkout to current main

**Files:**
- Modify (working tree): `~/code/bigraph-schema` (branch/worktree)

The local checkout is stale (`chore/bump-1.4.3`); all later tasks edit files that only exist on current `main`.

- [ ] **Step 1: Create a fresh worktree off current origin/main**

```bash
git -C ~/code/bigraph-schema fetch origin main
git -C ~/code/bigraph-schema worktree add ~/code/bigraph-schema--process-contract -b feat/process-contract origin/main
cd ~/code/bigraph-schema--process-contract
```

- [ ] **Step 2: Verify the amendment model is present (fails on 1.4.3, passes on 1.6.0+)**

Run: `grep -c "def amend" bigraph_schema/contract.py && grep -c "class Site" bigraph_schema/schema.py`
Expected: both print a non-zero count. If `contract.py` is absent, you are on the wrong base — stop and re-fetch.

- [ ] **Step 3: Install editable + confirm tests run green on the untouched base**

Run: `uv sync && uv run pytest -q`
Expected: the existing suite passes (baseline before any change).

---

### Task 1: Expression language — parse to a safe AST

**Files:**
- Create: `bigraph_schema/contract_expr.py`
- Test: `bigraph_schema/tests/test_contract_expr.py`

**Interfaces:**
- Produces: `parse(expr: str) -> Expr` where `Expr` is a small frozen dataclass tree; raises `ExprError(str)` on a syntax error or a disallowed construct. The grammar admits ONLY: name paths (`inputs.mass`, `config.pH`, `state.a.b`), the bareword `tol`, number/bool/string literals, binary comparisons (`== != < <= > >=`), binary arithmetic (`+ - * /`), unary `-`, and calls to the whitelisted reducers `sum min max abs all any`. Nothing else (no attribute calls, no `__`, no lambdas, no comprehensions).

- [ ] **Step 1: Write the failing tests**

```python
# bigraph_schema/tests/test_contract_expr.py
import pytest
from bigraph_schema.contract_expr import parse, ExprError

def test_parses_a_conservation_expression():
    ast = parse("abs(sum(outputs.mass) - sum(inputs.mass)) <= tol")
    assert ast is not None  # structure asserted via names_in/evaluate in later tasks

def test_parses_a_name_path():
    assert parse("inputs.biomass > 0") is not None

def test_rejects_unbalanced_parens():
    with pytest.raises(ExprError):
        parse("sum(outputs.mass ==")

def test_rejects_dunder_and_calls_outside_the_whitelist():
    with pytest.raises(ExprError):
        parse("__import__('os')")
    with pytest.raises(ExprError):
        parse("outputs.mass.conjugate()")  # attribute call, not a whitelisted reducer

def test_rejects_an_unknown_reducer():
    with pytest.raises(ExprError):
        parse("total(outputs.mass) == 0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py -q`
Expected: FAIL — `ModuleNotFoundError: bigraph_schema.contract_expr`.

- [ ] **Step 3: Implement the parser over Python's `ast` with a strict whitelist**

Use the stdlib `ast` module to tokenize/parse, then walk the tree rejecting any node kind not in the whitelist (this is the standard "safe expression" technique — `ast.parse(expr, mode='eval')` then validate node classes; never compile/eval). Map the accepted subset to the small `Expr` tree.

```python
# bigraph_schema/contract_expr.py
"""A tiny, safe expression language for contract predicates.

parse() -> an Expr tree (frozen); names_in() -> referenced port/config/state
paths; evaluate() -> value against a binding environment. NEVER uses eval/exec:
the string is parsed with the stdlib ``ast`` module and every node is checked
against a whitelist before a hand-written evaluator walks it.
"""
from __future__ import annotations
import ast as _ast
from dataclasses import dataclass

class ExprError(ValueError):
    """A contract expression is malformed or uses a disallowed construct."""

REDUCERS = frozenset({'sum', 'min', 'max', 'abs', 'all', 'any'})
_CMP = {_ast.Eq: '==', _ast.NotEq: '!=', _ast.Lt: '<', _ast.LtE: '<=', _ast.Gt: '>', _ast.GtE: '>='}
_BIN = {_ast.Add: '+', _ast.Sub: '-', _ast.Mult: '*', _ast.Div: '/'}

@dataclass(frozen=True)
class Expr:
    kind: str           # 'name' | 'lit' | 'cmp' | 'bin' | 'neg' | 'call'
    value: object = None
    op: str = ''
    args: tuple = ()

def parse(expr: str) -> Expr:
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError('expression must be a non-empty string')
    try:
        tree = _ast.parse(expr, mode='eval').body
    except SyntaxError as error:
        raise ExprError(f'could not parse {expr!r}: {error.msg}') from None
    return _convert(tree, expr)

def _convert(node, expr):
    if isinstance(node, _ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return Expr('lit', value=node.value)
        raise ExprError(f'unsupported literal in {expr!r}')
    if isinstance(node, _ast.Name):
        if node.id == 'tol':
            return Expr('name', value=('tol',))
        raise ExprError(f'bare name {node.id!r} — use a path like inputs.x (only `tol` is bare)')
    if isinstance(node, _ast.Attribute):
        return Expr('name', value=_path(node, expr))
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
        return Expr('neg', args=(_convert(node.operand, expr),))
    if isinstance(node, _ast.BinOp) and type(node.op) in _BIN:
        return Expr('bin', op=_BIN[type(node.op)], args=(_convert(node.left, expr), _convert(node.right, expr)))
    if isinstance(node, _ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
            raise ExprError(f'only a single simple comparison is allowed in {expr!r}')
        return Expr('cmp', op=_CMP[type(node.ops[0])], args=(_convert(node.left, expr), _convert(node.comparators[0], expr)))
    if isinstance(node, _ast.Call):
        if not isinstance(node.func, _ast.Name) or node.func.id not in REDUCERS:
            raise ExprError(f'only the reducers {sorted(REDUCERS)} may be called, not {_ast.dump(node.func)}')
        if node.keywords:
            raise ExprError('reducers take no keyword arguments')
        return Expr('call', op=node.func.id, args=tuple(_convert(a, expr) for a in node.args))
    raise ExprError(f'disallowed construct {type(node).__name__} in {expr!r}')

def _path(node, expr):
    parts = []
    while isinstance(node, _ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, _ast.Name):
        raise ExprError(f'a name path must start at a bare root (inputs/outputs/config/state) in {expr!r}')
    parts.append(node.id)
    return tuple(reversed(parts))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract_expr.py bigraph_schema/tests/test_contract_expr.py
git commit -m "feat(contract): safe expression parser for contract predicates"
```

---

### Task 2: Expression language — name extraction

**Files:**
- Modify: `bigraph_schema/contract_expr.py`
- Test: `bigraph_schema/tests/test_contract_expr.py`

**Interfaces:**
- Produces: `names_in(ast: Expr) -> set[tuple[str, ...]]` — every name path referenced (e.g. `{('inputs','mass'), ('outputs','mass')}`), excluding the bare `('tol',)`. Used by the auditor to resolve references and by the evaluator to build its environment.

- [ ] **Step 1: Write the failing test**

```python
from bigraph_schema.contract_expr import parse, names_in

def test_names_in_collects_paths_and_excludes_tol():
    names = names_in(parse("abs(sum(outputs.mass) - sum(inputs.mass)) <= tol"))
    assert names == {('outputs', 'mass'), ('inputs', 'mass')}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py::test_names_in_collects_paths_and_excludes_tol -q`
Expected: FAIL — `names_in` not defined.

- [ ] **Step 3: Implement**

```python
def names_in(ast: Expr) -> set:
    found = set()
    def walk(node):
        if node.kind == 'name' and node.value != ('tol',):
            found.add(node.value)
        for arg in node.args:
            walk(arg)
    walk(ast)
    return found
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py::test_names_in_collects_paths_and_excludes_tol -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract_expr.py bigraph_schema/tests/test_contract_expr.py
git commit -m "feat(contract): names_in — referenced paths of a contract expression"
```

---

### Task 3: Expression language — evaluate against a binding environment

**Files:**
- Modify: `bigraph_schema/contract_expr.py`
- Test: `bigraph_schema/tests/test_contract_expr.py`

**Interfaces:**
- Produces: `evaluate(ast: Expr, env: dict, *, tol: float = 0.0) -> value`. `env` maps a name path tuple to its value (e.g. `{('inputs','mass'): [1.0, 2.0], ('outputs','mass'): 3.0}`). Reducers operate on list/scalar; `==`/`!=` on floats use `tol` (`abs(a-b) <= tol`). A referenced name missing from `env` raises `ExprError` (the evaluator never invents a value). Used by the runtime strict-mode (Phase 3) and by the auditor's optional dry-eval.

- [ ] **Step 1: Write the failing tests**

```python
from bigraph_schema.contract_expr import parse, evaluate, ExprError
import pytest

def test_evaluate_conservation_true_within_tol():
    ast = parse("abs(sum(outputs.flux) - sum(inputs.flux)) <= tol")
    env = {('outputs', 'flux'): [1.0, 2.0], ('inputs', 'flux'): [3.0]}
    assert evaluate(ast, env, tol=1e-9) is True

def test_evaluate_precondition():
    assert evaluate(parse("inputs.biomass > 0"), {('inputs', 'biomass'): 0.5}) is True
    assert evaluate(parse("inputs.biomass > 0"), {('inputs', 'biomass'): 0.0}) is False

def test_evaluate_missing_name_raises():
    with pytest.raises(ExprError):
        evaluate(parse("inputs.x > 0"), {})
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py -k evaluate -q`
Expected: FAIL — `evaluate` not defined.

- [ ] **Step 3: Implement**

```python
_REDUCE = {'sum': sum, 'min': min, 'max': max, 'abs': abs, 'all': all, 'any': any}

def evaluate(ast: Expr, env: dict, *, tol: float = 0.0):
    def ev(node):
        if node.kind == 'lit':
            return node.value
        if node.kind == 'name':
            if node.value == ('tol',):
                return tol
            if node.value not in env:
                raise ExprError(f'no binding for {".".join(node.value)}')
            return env[node.value]
        if node.kind == 'neg':
            return -ev(node.args[0])
        if node.kind == 'bin':
            a, b = ev(node.args[0]), ev(node.args[1])
            return {'+': a + b, '-': a - b, '*': a * b, '/': a / b}[node.op]
        if node.kind == 'call':
            values = [ev(a) for a in node.args]
            return _REDUCE[node.op](*values) if node.op == 'abs' else _REDUCE[node.op](values[0])
        if node.kind == 'cmp':
            a, b = ev(node.args[0]), ev(node.args[1])
            if node.op == '==':
                return abs(a - b) <= tol if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b
            if node.op == '!=':
                return not (abs(a - b) <= tol) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a != b
            return {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[node.op]
        raise ExprError(f'cannot evaluate node kind {node.kind!r}')
    return ev(ast)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py -k evaluate -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract_expr.py bigraph_schema/tests/test_contract_expr.py
git commit -m "feat(contract): safe evaluator for contract expressions"
```

---

### Task 4: Structured predicates on the contract

**Files:**
- Modify: `bigraph_schema/contract.py`
- Test: `bigraph_schema/tests/test_contract_predicates.py`

**Interfaces:**
- Produces:
  - `narrow_predicate(contract, kind, expr, *, name=None, tol=0.0) -> ProcessContract` — returns a NEW contract with one more `narrow` amendment whose `detail['predicate']` is `{'kind': kind, 'name': name or <auto>, 'expr': expr, 'tol': tol}`. `kind ∈ {'invariant','pre','post','validity'}`. Raises `ValueError` on an unknown kind or an unparseable `expr` (parsed via `contract_expr.parse` at declaration time — fail at authoring, not at runtime).
  - `ProcessContract.predicates(kind=None) -> list[dict]` — the predicate dicts (optionally filtered by kind). Extends the existing `predicates()` which currently returns raw predicate values.

Note: this reuses the existing `amend`/`Amendment` monotonicity — a predicate is only ever *added*, never removed, so a contract only gets stricter. Per-port bounds/units are NOT added here; they already live on the face port types.

- [ ] **Step 1: Write the failing tests**

```python
# bigraph_schema/tests/test_contract_predicates.py
import pytest
from bigraph_schema.contract import ProcessContract, narrow_predicate

def _base():
    return ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})

def test_add_invariant_is_monotone_and_readable_by_kind():
    c0 = _base()
    c1 = narrow_predicate(c0, 'invariant', 'abs(sum(outputs.mass) - sum(inputs.mass)) <= tol', name='mass', tol=1e-9)
    assert c0.predicates() == []                       # input unchanged (pure)
    inv = c1.predicates(kind='invariant')
    assert len(inv) == 1 and inv[0]['name'] == 'mass' and inv[0]['tol'] == 1e-9
    assert c1.predicates(kind='pre') == []

def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        narrow_predicate(_base(), 'whatever', 'inputs.mass > 0')

def test_unparseable_expr_rejected_at_declaration():
    with pytest.raises(ValueError):
        narrow_predicate(_base(), 'pre', 'inputs.mass >')
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest bigraph_schema/tests/test_contract_predicates.py -q`
Expected: FAIL — `narrow_predicate` not importable.

- [ ] **Step 3: Implement in contract.py**

```python
# add near the other module functions in bigraph_schema/contract.py
from bigraph_schema.contract_expr import parse as _parse_expr, ExprError as _ExprError

PREDICATE_KINDS = frozenset({'invariant', 'pre', 'post', 'validity'})

def narrow_predicate(contract, kind, expr, *, name=None, tol=0.0):
    """Return a NEW contract with one structured predicate added as a narrow
    amendment. kind in PREDICATE_KINDS; expr is parsed now so a typo fails at
    authoring, not mid-run."""
    if kind not in PREDICATE_KINDS:
        raise ValueError(f'unknown predicate kind {kind!r}; use one of {sorted(PREDICATE_KINDS)}')
    try:
        _parse_expr(expr)
    except _ExprError as error:
        raise ValueError(f'invalid {kind} expression {expr!r}: {error}') from None
    predicate = {'kind': kind, 'name': name or f'{kind}_{len(contract.amendments)}', 'expr': expr, 'tol': tol}
    return amend(contract, Amendment(op='narrow', detail={'predicate': predicate}))
```

Then update `ProcessContract.predicates` to accept `kind` and return the structured dicts (keep back-compat: a predicate may be a bare value or the new dict):

```python
    def predicates(self, kind=None):
        """Structured predicate dicts contributed by narrow amendments, optionally filtered by kind."""
        out = []
        for amendment in self.amendments:
            predicate = (amendment.detail or {}).get('predicate')
            if isinstance(predicate, dict) and 'kind' in predicate:
                if kind is None or predicate['kind'] == kind:
                    out.append(predicate)
        return out
```

- [ ] **Step 4: Run to verify they pass (and the existing contract suite still passes)**

Run: `uv run pytest bigraph_schema/tests/test_contract_predicates.py -q && uv run pytest bigraph_schema/tests/ -k contract -q`
Expected: PASS; no regression in the existing contract tests.

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract.py bigraph_schema/tests/test_contract_predicates.py
git commit -m "feat(contract): structured invariant/pre/post/validity predicates via narrow amendments"
```

---

### Task 5: Serialization round-trip for predicates

**Files:**
- Modify: `bigraph_schema/contract.py` (only if `to_dict`/`from` drops the predicate dict — verify first)
- Test: `bigraph_schema/tests/test_contract_predicates.py`

**Interfaces:**
- Produces: a contract carrying structured predicates survives `to_dict()` → reconstruct, so it travels in a serialized document (spec §9). `Amendment.to_dict` already serializes `detail`; this task PINS that the predicate dict round-trips and that reconstruction preserves `predicates(kind=...)`.

- [ ] **Step 1: Write the failing test**

```python
from bigraph_schema.contract import ProcessContract, narrow_predicate, Amendment

def test_predicate_survives_to_dict_round_trip():
    c = narrow_predicate(ProcessContract(face={'inputs': {}, 'outputs': {}}), 'post', 'all(outputs.mass >= 0)', name='nonneg')
    data = c.to_dict()
    # amendments serialize their detail; the predicate dict must be intact
    preds = [ (a.get('detail') or {}).get('predicate') for a in data['amendments'] ]
    assert {'kind': 'post', 'name': 'nonneg', 'expr': 'all(outputs.mass >= 0)', 'tol': 0.0} in preds
    # and reconstructing amendments restores predicates(kind=...)
    rebuilt = ProcessContract(face=data['face'], amendments=[Amendment(**a) for a in data['amendments']])
    assert len(rebuilt.predicates(kind='post')) == 1
```

- [ ] **Step 2: Run to verify it fails or passes**

Run: `uv run pytest bigraph_schema/tests/test_contract_predicates.py::test_predicate_survives_to_dict_round_trip -q`
Expected: If `to_dict`/`Amendment(**a)` already round-trips (likely, since `detail` is plain JSON-ish), this PASSES immediately — then this task is a *characterization* test locking the behavior; keep it. If it FAILS (e.g. `_is_jsonish` strips the dict), proceed to Step 3.

- [ ] **Step 3: If failing, ensure the predicate dict is treated as JSON-ish detail**

Inspect `_is_jsonish` and `Amendment.to_dict` in `contract.py`; a `{'kind','name','expr','tol'}` dict of str/float is already JSON-ish, so the fix (if any) is to not special-case-drop `detail['predicate']`. Make the minimal change so the Step-1 test passes. (If Step 2 already passed, skip.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest bigraph_schema/tests/test_contract_predicates.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract.py bigraph_schema/tests/test_contract_predicates.py
git commit -m "test(contract): predicates round-trip through to_dict (serialization pin)"
```

---

### Task 6: Audit report shape + completeness grade

**Files:**
- Create: `bigraph_schema/contract_audit.py`
- Test: `bigraph_schema/tests/test_contract_audit.py`

**Interfaces:**
- Produces:
  - `Finding = dataclass(severity: str, code: str, where: str, message: str)` — `severity ∈ {'error','warning','info'}`.
  - `AuditReport = dataclass(ok: bool, findings: list[Finding])` where `ok` is `True` iff no `error` finding.
  - `completeness(contract) -> float` in `[0,1]`: fraction of the contract that is constrained — scores declared port *bounds/units* (a face port whose type is a bare `'float'`/`'int'` counts as unconstrained; a `Range`/`_units`/`Nonnegative`/`Enum`-typed port counts as constrained) plus a bonus if any predicate exists. Pure heuristic; used as the `info` grade, never a gate.

- [ ] **Step 1: Write the failing tests**

```python
# bigraph_schema/tests/test_contract_audit.py
from bigraph_schema.contract import ProcessContract, narrow_predicate
from bigraph_schema.contract_audit import completeness, Finding, AuditReport

def test_bare_float_ports_score_low():
    c = ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})
    assert completeness(c) < 0.5

def test_constrained_ports_and_a_predicate_score_higher():
    c = narrow_predicate(
        ProcessContract(face={'inputs': {'mass': 'positive_float[mM]'}, 'outputs': {'mass': 'positive_float[mM]'}}),
        'invariant', 'abs(sum(outputs.mass) - sum(inputs.mass)) <= tol', tol=1e-9)
    assert completeness(c) > completeness(ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}))

def test_report_ok_iff_no_error():
    assert AuditReport(ok=True, findings=[Finding('info', 'x', 'y', 'z')]).ok is True
    assert AuditReport(ok=False, findings=[Finding('error', 'x', 'y', 'z')]).ok is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest bigraph_schema/tests/test_contract_audit.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the report shape + completeness**

`completeness` decides "is a port type constrained?" by asking the core whether the resolved type carries a bound/unit. To avoid coupling to private schema internals, treat a port type string as constrained if it is NOT one of the bare base names and/or carries `[...]`/`{...}` decoration; refine with `core` in Task 7's signature. For this task use a string heuristic (bare `'float'/'int'/'integer'/'number'/'string'/'boolean'` = unconstrained).

```python
# bigraph_schema/contract_audit.py
"""Static audit of a ProcessContract: well-formedness, narrow soundness, and a
completeness grade. Pure analysis — never runs the process, never raises on a
well-formed-but-incomplete contract."""
from __future__ import annotations
from dataclasses import dataclass, field

_BARE = frozenset({'float', 'int', 'integer', 'number', 'string', 'boolean', 'bool', 'any'})

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

def _port_constrained(port_type) -> bool:
    if isinstance(port_type, dict):
        keys = set(port_type)
        return bool(keys & {'_min', '_max', '_units', '_values'}) or port_type.get('_type') not in _BARE | {None}
    if isinstance(port_type, str):
        base = port_type.split('[', 1)[0].split('{', 1)[0].strip()
        return base not in _BARE or ('[' in port_type)
    return False

def completeness(contract) -> float:
    face = contract.face or {}
    ports = list((face.get('inputs') or {}).values()) + list((face.get('outputs') or {}).values())
    if not ports:
        port_score = 0.0
    else:
        port_score = sum(1 for p in ports if _port_constrained(p)) / len(ports)
    predicate_bonus = 0.25 if contract.predicates() else 0.0
    return min(1.0, 0.75 * port_score + predicate_bonus)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest bigraph_schema/tests/test_contract_audit.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract_audit.py bigraph_schema/tests/test_contract_audit.py
git commit -m "feat(contract): audit report shape + completeness grade"
```

---

### Task 7: Audit — predicate well-formedness (names resolve)

**Files:**
- Modify: `bigraph_schema/contract_audit.py`
- Test: `bigraph_schema/tests/test_contract_audit.py`

**Interfaces:**
- Produces: `audit_contract(core, contract) -> AuditReport`. First check: every predicate's expression parses (already guaranteed by `narrow_predicate`, but a contract built by hand may skip it) AND every referenced `inputs.<p>`/`outputs.<p>` names a port in the face, and every `config.<p>` is plausible (non-empty segment). An unresolved reference is an `error` finding. `core` is accepted for later checks (unit parsing, type resolution) and unused refs are fine in Phase 1.

- [ ] **Step 1: Write the failing tests**

```python
from bigraph_schema.contract import ProcessContract, narrow_predicate
from bigraph_schema.contract_audit import audit_contract

class _Core:  # a stand-in; audit_contract must not require a real core in Phase 1
    pass

def test_predicate_referencing_unknown_port_is_an_error():
    c = narrow_predicate(ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}),
                         'invariant', 'outputs.flux == 0')  # no 'flux' port
    report = audit_contract(_Core(), c)
    assert report.ok is False
    assert any(f.severity == 'error' and 'flux' in f.message for f in report.findings)

def test_predicate_referencing_declared_ports_is_clean():
    c = narrow_predicate(ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}}),
                         'invariant', 'abs(outputs.mass - inputs.mass) <= tol', tol=1e-9)
    report = audit_contract(_Core(), c)
    assert not [f for f in report.findings if f.severity == 'error']
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest bigraph_schema/tests/test_contract_audit.py -k audit_contract -q`
Expected: FAIL — `audit_contract` not defined.

- [ ] **Step 3: Implement**

```python
from bigraph_schema.contract_expr import parse, names_in, ExprError

def audit_contract(core, contract) -> AuditReport:
    findings = []
    face = contract.face or {}
    in_ports = set((face.get('inputs') or {}).keys())
    out_ports = set((face.get('outputs') or {}).keys())
    for predicate in contract.predicates():
        name = predicate.get('name', '?')
        try:
            ast = parse(predicate['expr'])
        except ExprError as error:
            findings.append(Finding('error', 'expr_parse', f'predicate:{name}', str(error)))
            continue
        for path in names_in(ast):
            root, port = path[0], (path[1] if len(path) > 1 else None)
            if root == 'inputs' and port not in in_ports:
                findings.append(Finding('error', 'unknown_port', f'predicate:{name}', f'inputs.{port} is not a declared input port'))
            elif root == 'outputs' and port not in out_ports:
                findings.append(Finding('error', 'unknown_port', f'predicate:{name}', f'outputs.{port} is not a declared output port'))
            elif root not in {'inputs', 'outputs', 'config', 'state'}:
                findings.append(Finding('error', 'unknown_root', f'predicate:{name}', f'{root} is not a valid reference root'))
    grade = completeness(contract)
    findings.append(Finding('info', 'completeness', 'contract', f'completeness grade {grade:.2f}'))
    return AuditReport(ok=not any(f.severity == 'error' for f in findings), findings=findings)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest bigraph_schema/tests/test_contract_audit.py -q`
Expected: PASS (all audit tests).

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract_audit.py bigraph_schema/tests/test_contract_audit.py
git commit -m "feat(contract): audit_contract — predicate well-formedness + name resolution"
```

---

### Task 8: Audit — per-port bound/unit sanity

**Files:**
- Modify: `bigraph_schema/contract_audit.py`
- Test: `bigraph_schema/tests/test_contract_audit.py`

**Interfaces:**
- Produces: `audit_contract` additionally flags a face port whose declared type has `min > max`, or a `_units` string that `pint` cannot parse, as an `error`. Uses the installed `pint` registry (the same `bigraph_schema/units.py` the type system uses) for unit parsing.

- [ ] **Step 1: Write the failing tests**

```python
from bigraph_schema.contract import ProcessContract
from bigraph_schema.contract_audit import audit_contract

class _Core: pass

def test_min_greater_than_max_is_an_error():
    c = ProcessContract(face={'inputs': {'t': {'_type': 'float', '_min': 10, '_max': 0}}, 'outputs': {}})
    report = audit_contract(_Core(), c)
    assert any(f.code == 'bad_range' and f.severity == 'error' for f in report.findings)

def test_unparseable_units_is_an_error():
    c = ProcessContract(face={'inputs': {'g': {'_type': 'float', '_units': 'not_a_unit_xyz'}}, 'outputs': {}})
    report = audit_contract(_Core(), c)
    assert any(f.code == 'bad_units' and f.severity == 'error' for f in report.findings)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest bigraph_schema/tests/test_contract_audit.py -k "range or units" -q`
Expected: FAIL — no such findings yet.

- [ ] **Step 3: Implement — add a port-sanity pass into `audit_contract`**

```python
# at top of contract_audit.py
try:
    from bigraph_schema.units import units as _unit_registry  # the shared pint registry
except Exception:  # noqa: BLE001 — units optional; absence is not an audit error
    _unit_registry = None

# inside audit_contract, before the completeness info finding:
    for direction in ('inputs', 'outputs'):
        for port, port_type in (face.get(direction) or {}).items():
            if not isinstance(port_type, dict):
                continue
            lo, hi = port_type.get('_min'), port_type.get('_max')
            if lo is not None and hi is not None and lo > hi:
                findings.append(Finding('error', 'bad_range', f'{direction}.{port}', f'_min {lo} > _max {hi}'))
            unit = port_type.get('_units')
            if unit and _unit_registry is not None:
                try:
                    _unit_registry.Unit(unit)
                except Exception:  # noqa: BLE001 — a bad unit string is the finding
                    findings.append(Finding('error', 'bad_units', f'{direction}.{port}', f'cannot parse units {unit!r}'))
```

(Place this block before the `grade = completeness(...)` line so the info finding stays last.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest bigraph_schema/tests/test_contract_audit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bigraph_schema/contract_audit.py bigraph_schema/tests/test_contract_audit.py
git commit -m "feat(contract): audit per-port bound/unit sanity"
```

---

### Task 9: End-to-end audit of a real, incomplete, and lying contract

**Files:**
- Test: `bigraph_schema/tests/test_contract_audit.py`

**Interfaces:**
- Consumes: everything above. This is the integration test pinning the three-way behavior: a *good* contract audits clean, an *incomplete* one audits clean-but-low-grade (never `error`), a *malformed* one audits `error`.

- [ ] **Step 1: Write the failing/also-passing integration test**

```python
from bigraph_schema.contract import ProcessContract, narrow_predicate
from bigraph_schema.contract_audit import audit_contract

class _Core: pass

def test_good_incomplete_and_lying_contracts():
    good = narrow_predicate(
        ProcessContract(face={'inputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'mg'}},
                              'outputs': {'mass': {'_type': 'float', '_min': 0, '_units': 'mg'}}}),
        'invariant', 'abs(outputs.mass - inputs.mass) <= tol', tol=1e-9)
    r_good = audit_contract(_Core(), good)
    assert r_good.ok and not [f for f in r_good.findings if f.severity == 'error']

    incomplete = ProcessContract(face={'inputs': {'mass': 'float'}, 'outputs': {'mass': 'float'}})
    r_incomplete = audit_contract(_Core(), incomplete)
    assert r_incomplete.ok                                  # incomplete is never an error
    assert any(f.code == 'completeness' for f in r_incomplete.findings)

    lying = narrow_predicate(incomplete, 'post', 'outputs.ghost >= 0')  # no 'ghost' port
    r_lying = audit_contract(_Core(), lying)
    assert not r_lying.ok and any(f.severity == 'error' for f in r_lying.findings)
```

- [ ] **Step 2: Run the whole Phase-1 suite**

Run: `uv run pytest bigraph_schema/tests/test_contract_expr.py bigraph_schema/tests/test_contract_predicates.py bigraph_schema/tests/test_contract_audit.py -q`
Expected: PASS (all).

- [ ] **Step 3: Run the FULL bigraph-schema suite — confirm zero regressions**

Run: `uv run pytest -q`
Expected: PASS; the existing suite is unaffected (Phase 1 adds only new modules + additive helpers).

- [ ] **Step 4: Commit**

```bash
git add bigraph_schema/tests/test_contract_audit.py
git commit -m "test(contract): end-to-end audit of good/incomplete/lying contracts"
```

- [ ] **Step 5: Open the PR**

```bash
git push -u origin feat/process-contract
gh pr create -R vivarium-collective/bigraph-schema --title "feat(contract): Phase 1 — contract conditions, expression language, static audit" --body "Implements Phase 1 of docs/superpowers/specs/2026-10-03-process-contract-formalization-design.md: a safe contract expression language (contract_expr.py), structured invariant/pre/post/validity predicates on narrow amendments (contract.py), and a static audit_contract (contract_audit.py). Additive, zero runtime/behavior change; full suite green."
```

---

## Self-Review

**Spec coverage (Phase 1 scope — spec §3, §5, §9, §10 Phase 1):**
- §3.1 per-port bounds/units (on face types) → read by Task 6 (`completeness`) + Task 8 (sanity). ✓
- §3.2 validity / §3.3 invariants / §3.4 requires-guarantees → Task 4 (`narrow_predicate` kinds `validity/invariant/pre/post`). ✓
- §3.5 expression language → Tasks 1–3. ✓
- §3.5 monotonicity → reused from existing `amend`; Task 4 test asserts purity. ✓
- §5 static auditor: well-formedness → Task 7; bound/unit sanity → Task 8; completeness grade → Task 6; **narrow soundness** → inherited from `amend` (raises `AmendmentError`); *(the implementation-drift / no-op-`update` check is process-bigraph → Phase 2, correctly out of this plan)*. ✓
- §9 serialization → Task 5. ✓
- Out of Phase 1 (correctly deferred): candidate matching (§4 → Phase 2), runtime strict-mode (§6 → Phase 3), workbench (§8 → Phase 4), composite internal-audit (§7 → Phase 2).

**Placeholder scan:** no TBD/TODO; every code step has real code. ✓

**Type consistency:** `parse`/`names_in`/`evaluate`/`Expr`/`ExprError` (Tasks 1–3) used consistently; `narrow_predicate`/`predicates(kind=)` (Task 4) used in Tasks 6–9; `Finding`/`AuditReport`/`completeness`/`audit_contract` (Tasks 6–8) used in Task 9. ✓

**Review Focus coverage:** unknown-name typo → Task 7; malformed expr → Task 2; eval-injection → Task 2; bad range/units → Task 8; empty/incomplete contract → Task 6 + Task 9. ✓
