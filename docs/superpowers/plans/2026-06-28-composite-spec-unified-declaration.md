# CompositeSpec — Unified Composite Declaration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the two existing pbg-superpowers composite front-ends (static `composite_spec.py` + code `composite_generator.py`) into ONE declarative `CompositeSpec` descriptor in process-bigraph, carrying all metadata, producing a runtime `Composite`, and resolving to a saved default state for ParCa-free dashboard display.

**Architecture:** A `process_bigraph.composite_spec` module defines the `CompositeSpec` dataclass + a `@composite_spec` decorator + a registry + `discover_specs` + `regenerate_default_state`. pbg-superpowers' two modules become thin shims delegating to it (existing usage unchanged). The dashboard resolves against the process-bigraph registry, displays from `CompositeSpec.default_state()`, and the explorer's hardcoded remote-build error is replaced by the server's real notice.

**Tech Stack:** Python 3.12, `dataclasses`, stdlib `importlib`/`json`, `yaml` (PyYAML), pytest (`python_files = *.py`, `python_functions = test*`), `bigraph_schema.allocate_core`, `process_bigraph.Composite`.

## Global Constraints

- **No new runtime dependencies.** Only stdlib + already-present `yaml`, `bigraph_schema`, `process_bigraph`.
- **The local resolve/run paths must stay behaviour-preserving** — existing `@composite_generator` usage in v2ecoli (13 generators) and all `*.composite.yaml` files keep working through the shim. Removing the shim is OUT of scope.
- **Exactly one of `state` / `builder`** is set on a `CompositeSpec`; `schema` accompanies `state` only; `default_state_ref` requires `builder`.
- **Canonical parameter types** = `{"integer", "float", "string", "boolean", "list", "map"}`. Aliases normalize on construction: `int→integer`, `number→float`, `double→float`, `str→string`, `bool→boolean`, `array→list`, `object→map`, `dict→map`. Unknown types pass through unchanged (no raise).
- **A Composite DOCUMENT is `{schema, state, …}`** — generator builders may return extra exec keys (`skip_initial_steps`, `sequential_steps`, `flow_order`, `run_steps_on_init`); `to_document`/`to_composite` forward the WHOLE document verbatim, never narrowed to `{state, schema}`.
- **`core_extensions` are load-bearing** — applied to the core BEFORE the builder runs.
- **Worktrees:** process-bigraph work is in `/Users/eranagmon/code/pbg-composite-spec` (branch `feat/composite-spec`). Tasks 7–8 (pbg-superpowers) and 9–10 (vivarium-dashboard) each need their own worktree off `origin/main` — create at task start (see each task). Proof (Task 11) spans v2ecoli + autopoiesis worktrees.
- **Test command (process-bigraph):** the worktree has no `.venv`; use the main checkout's interpreter, which imports the worktree's local `process_bigraph/` because CWD is on `sys.path`: `cd /Users/eranagmon/code/pbg-composite-spec && /Users/eranagmon/code/process-bigraph/.venv/bin/python -m pytest process_bigraph/tests/test_composite_spec.py -v`. Ruff: `/Users/eranagmon/code/process-bigraph/.venv/bin/ruff check process_bigraph/composite_spec.py` (or `python -m ruff` if the `ruff` entrypoint is absent).

## The Canonical Contract (every task depends on these exact signatures)

```python
# process_bigraph/composite_spec.py
CANONICAL_TYPES = {"integer", "float", "string", "boolean", "list", "map"}

@dataclass
class CompositeSpec:
    id: str                                    # canonical "<module>.<name>"
    name: str
    description: str = ""
    tags: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)      # {name: {type, default, description?, choices?}}
    default_n_steps: "int | None" = None
    visualizations: list = field(default_factory=list)
    analyses: list = field(default_factory=list)
    emitters: list = field(default_factory=list)        # [{address, config?, paths?}]
    requires: dict = field(default_factory=dict)        # {processes: [...], types: [...]}
    schema: dict = field(default_factory=dict)          # bigraph-schema store types (static only)
    state: "dict | None" = None                         # static body (exactly one of state/builder)
    builder: "Callable | str | None" = None             # generator: callable or dotted "pkg.mod:fn"
    default_state_ref: "str | None" = None              # generator saved-state artifact filename
    module: str = ""                                    # owning module (for discovery/display)
    core_extensions: list = field(default_factory=list) # callables (core) -> core | None; not serialized

    # properties / methods
    @property
    def kind(self) -> str: ...                          # "generator" if builder else "spec"
    def to_dict(self) -> dict: ...                       # portable JSON (builder -> dotted str; core_extensions dropped)
    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSpec": ...
    @classmethod
    def from_file(cls, path) -> "CompositeSpec": ...     # parse *.composite.{yaml,json}
    def to_document(self, overrides=None, core=None) -> dict: ...   # resolved {schema, state, …}
    def to_composite(self, overrides=None, core=None):  ...         # -> process_bigraph.Composite
    def default_state(self, base_dir=None) -> "dict | None": ...    # display state; never builds

def normalize_type(t: str) -> str: ...
def composite_spec(*, name, description="", parameters=None, visualizations=None,
                   emitters=None, analyses=None, tags=None, default_n_steps=None,
                   core_extensions=None, default_state_ref=None): ...   # decorator -> registers CompositeSpec(builder=fn)
def register(spec: CompositeSpec) -> None: ...
def get(spec_id: str) -> "CompositeSpec | None": ...
def all_specs() -> "dict[str, CompositeSpec]": ...
def clear_registry() -> None: ...
def discover_specs(workspace=None) -> "dict[str, CompositeSpec]": ...
def regenerate_default_state(spec: CompositeSpec, base_dir, core=None) -> "Path": ...
```

---

### Task 1: `CompositeSpec` dataclass — fields, type normalization, validation, JSON round-trip

**Files:**
- Create: `process_bigraph/composite_spec.py`
- Create: `process_bigraph/tests/__init__.py` (empty)
- Test: `process_bigraph/tests/test_composite_spec.py`

**Interfaces — Produces:** `CompositeSpec` dataclass (fields per the Contract above), `normalize_type(t)`, `CANONICAL_TYPES`, `CompositeSpec.kind`, `.to_dict()`, `.from_dict(d)`. Validation runs in `__post_init__`.

- [ ] **Step 1: Write the failing tests**

```python
# process_bigraph/tests/test_composite_spec.py
import pytest
from process_bigraph.composite_spec import CompositeSpec, normalize_type, CANONICAL_TYPES


def test_normalize_type_aliases():
    assert normalize_type("int") == "integer"
    assert normalize_type("number") == "float"
    assert normalize_type("bool") == "boolean"
    assert normalize_type("object") == "map"
    assert normalize_type("array") == "list"
    assert normalize_type("string") == "string"
    assert normalize_type("unknownX") == "unknownX"  # pass-through


def test_static_spec_normalizes_param_types():
    s = CompositeSpec(id="m.c", name="c", state={"x": 1},
                      parameters={"seed": {"type": "int", "default": 0},
                                  "rate": {"type": "number", "default": 1.0}})
    assert s.parameters["seed"]["type"] == "integer"
    assert s.parameters["rate"]["type"] == "float"
    assert s.kind == "spec"


def test_generator_spec_kind():
    s = CompositeSpec(id="m.g", name="g", builder="m:g")
    assert s.kind == "generator"


def test_exactly_one_of_state_or_builder():
    with pytest.raises(ValueError, match="exactly one"):
        CompositeSpec(id="m.x", name="x")  # neither
    with pytest.raises(ValueError, match="exactly one"):
        CompositeSpec(id="m.x", name="x", state={}, builder="m:x")  # both


def test_schema_only_with_state():
    with pytest.raises(ValueError, match="schema"):
        CompositeSpec(id="m.g", name="g", builder="m:g", schema={"pop": "tree"})


def test_default_state_ref_requires_builder():
    with pytest.raises(ValueError, match="default_state_ref"):
        CompositeSpec(id="m.c", name="c", state={"x": 1}, default_state_ref="x.json")


def test_to_dict_from_dict_round_trip_static():
    s = CompositeSpec(id="m.c", name="c", description="d", tags=["t"],
                      state={"x": "${seed}"}, schema={"pop": "tree"},
                      parameters={"seed": {"type": "integer", "default": 0}},
                      requires={"processes": ["P"], "types": ["ty"]},
                      emitters=[{"address": "local:RAMEmitter"}], default_n_steps=10)
    d = s.to_dict()
    assert d["state"] == {"x": "${seed}"} and d["schema"] == {"pop": "tree"}
    assert d["requires"] == {"processes": ["P"], "types": ["ty"]}
    s2 = CompositeSpec.from_dict(d)
    assert s2 == s


def test_to_dict_serializes_builder_callable_to_dotted():
    def fn(core=None):
        return {"state": {}}
    s = CompositeSpec(id="m.g", name="g", builder=fn, module="m")
    d = s.to_dict()
    # a callable serializes to "<module>:<qualname>"; core_extensions are dropped
    assert isinstance(d["builder"], str) and d["builder"].endswith(":fn")
    assert "core_extensions" not in d
```

- [ ] **Step 2: Run → fail.** `cd /Users/eranagmon/code/pbg-composite-spec && .venv/bin/python -m pytest process_bigraph/tests/test_composite_spec.py -v` → ModuleNotFound / NameError.

- [ ] **Step 3: Implement** `process_bigraph/composite_spec.py`:

```python
"""Unified composite declaration for process-bigraph.

A CompositeSpec is the declarative descriptor a composite is authored as —
either inline data (``state`` + optional ``schema``) for a static composite, or
a ``builder`` callable for a generator composite. It carries the dashboard/UI
metadata (parameters, default_n_steps, visualizations, analyses, emitters,
requires) and produces a runtime ``process_bigraph.Composite``.

This unifies (and is the new home of) the two front-ends that previously lived
separately in pbg-superpowers: the static ``composite_spec`` spec-file format
and the ``composite_generator`` decorator. Those become thin shims over this.
"""
from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

CANONICAL_TYPES = {"integer", "float", "string", "boolean", "list", "map"}

_TYPE_ALIASES = {
    "integer": "integer", "int": "integer",
    "float": "float", "number": "float", "double": "float",
    "string": "string", "str": "string",
    "boolean": "boolean", "bool": "boolean",
    "list": "list", "array": "list",
    "map": "map", "object": "map", "dict": "map",
}


def normalize_type(t: str) -> str:
    """Map a parameter ``type`` string onto the canonical vocabulary.

    Known aliases collapse (int→integer, number→float, bool→boolean,
    object→map, array→list); an unknown type passes through unchanged.
    """
    return _TYPE_ALIASES.get(t, t)


@dataclass
class CompositeSpec:
    id: str
    name: str
    description: str = ""
    tags: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    default_n_steps: "int | None" = None
    visualizations: list = field(default_factory=list)
    analyses: list = field(default_factory=list)
    emitters: list = field(default_factory=list)
    requires: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)
    state: "dict | None" = None
    builder: "Callable | str | None" = None
    default_state_ref: "str | None" = None
    module: str = ""
    core_extensions: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CompositeSpec.name is required (non-empty string)")
        has_state = self.state is not None
        has_builder = self.builder is not None
        if has_state == has_builder:
            raise ValueError("CompositeSpec needs exactly one of `state` or `builder`")
        if has_builder and self.schema:
            raise ValueError("`schema` is for static specs; a generator's schema comes "
                             "from the builder document")
        if self.default_state_ref is not None and not has_builder:
            raise ValueError("`default_state_ref` requires a `builder`")
        # normalize parameter types in place
        for pdef in self.parameters.values():
            if isinstance(pdef, dict) and "type" in pdef:
                pdef["type"] = normalize_type(pdef["type"])

    @property
    def kind(self) -> str:
        return "generator" if self.builder is not None else "spec"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("core_extensions", None)  # callables are not serializable
        if callable(self.builder):
            d["builder"] = f"{getattr(self.builder, '__module__', self.module)}:" \
                           f"{self.builder.__qualname__}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSpec":
        d = dict(d)
        d.pop("core_extensions", None)
        return cls(**d)
```

- [ ] **Step 4: Run → pass.** Same command. All 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/pbg-composite-spec
git add process_bigraph/composite_spec.py process_bigraph/tests/
git commit -m "feat(composite-spec): CompositeSpec dataclass + type normalization + JSON round-trip"
```

---

### Task 2: Resolution — `substitute_parameters`, `to_document`, `to_composite`, `default_state` (inline)

**Files:**
- Modify: `process_bigraph/composite_spec.py`
- Test: `process_bigraph/tests/test_composite_spec.py` (append)

**Interfaces — Consumes:** `CompositeSpec` (Task 1), `process_bigraph.Composite`, `bigraph_schema.allocate_core`. **Produces:** module-level `substitute_parameters(state, params, overrides=None)`; `CompositeSpec.to_document(overrides=None, core=None)`, `.to_composite(overrides=None, core=None)`, `.default_state(base_dir=None)` (inline-state path; the artifact path lands in Task 5).

- [ ] **Step 1: Write the failing tests**

```python
from process_bigraph.composite_spec import substitute_parameters


def test_substitute_full_and_inline():
    params = {"seed": {"type": "integer", "default": 0}, "tag": {"type": "string", "default": "x"}}
    state = {"a": "${seed}", "b": "v-${tag}", "c": 5}
    out = substitute_parameters(state, params, {"seed": 7, "tag": "z"})
    assert out == {"a": 7, "b": "v-z", "c": 5}  # full placeholder typed; inline stringified


def test_to_document_static_substitutes_schema_and_state():
    s = CompositeSpec(id="m.c", name="c", state={"x": "${seed}"}, schema={"pop": "tree"},
                      parameters={"seed": {"type": "integer", "default": 3}})
    doc = s.to_document()
    assert doc == {"schema": {"pop": "tree"}, "state": {"x": 3}}


def test_to_document_generator_passes_whole_document():
    def build(core=None, *, seed=0):
        return {"state": {"s": seed}, "skip_initial_steps": True, "flow_order": ["a"]}
    s = CompositeSpec(id="m.g", name="g", builder=build,
                      parameters={"seed": {"type": "integer", "default": 0}})
    doc = s.to_document({"seed": 9})
    assert doc == {"state": {"s": 9}, "skip_initial_steps": True, "flow_order": ["a"]}


def test_to_document_rejects_unknown_override():
    s = CompositeSpec(id="m.c", name="c", state={}, parameters={"seed": {"type": "integer", "default": 0}})
    with pytest.raises(KeyError):
        s.to_document({"nope": 1})


def test_to_composite_static_returns_runnable_composite():
    s = CompositeSpec(id="m.c", name="c", state={"v": 1.0})
    comp = s.to_composite()
    from process_bigraph import Composite
    assert isinstance(comp, Composite)


def test_to_composite_applies_core_extensions_before_build():
    seen = {}
    def ext(core):
        seen["ran"] = True
        return core
    def build(core=None):
        seen["built_after_ext"] = seen.get("ran", False)
        return {"state": {}}
    s = CompositeSpec(id="m.g", name="g", builder=build, core_extensions=[ext])
    s.to_composite()
    assert seen["ran"] and seen["built_after_ext"]


def test_default_state_inline():
    s = CompositeSpec(id="m.c", name="c", state={"v": 1})
    assert s.default_state() == {"v": 1}
```

- [ ] **Step 2: Run → fail.** `substitute_parameters` / `to_document` undefined.

- [ ] **Step 3: Implement** — append to `process_bigraph/composite_spec.py`:

```python
_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: "str | None") -> Any:
    if declared_type is None:
        return value
    t = normalize_type(declared_type)
    if t == "float":
        return float(value)
    if t == "integer":
        return int(value)
    if t == "string":
        return str(value)
    if t == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _resolve_value(value, params, overrides):
    if not isinstance(value, str):
        return value
    m = _FULL_PLACEHOLDER.match(value)
    if m:
        pname = m.group(1)
        if pname not in params:
            raise KeyError(f"parameter '{pname}' referenced in state but not declared")
        pdef = params[pname]
        raw = overrides.get(pname, pdef.get("default"))
        if raw is None and "default" not in pdef:
            raise KeyError(f"parameter '{pname}' has no default and no override")
        return _cast(raw, pdef.get("type"))
    if _INLINE_PLACEHOLDER.search(value):
        def repl(match):
            pname = match.group(1)
            if pname not in params:
                raise KeyError(f"parameter '{pname}' referenced in state but not declared")
            raw = overrides.get(pname, params[pname].get("default"))
            return str(raw)
        return _INLINE_PLACEHOLDER.sub(repl, value)
    return value


def substitute_parameters(state, params, overrides=None):
    """Recursively substitute ``${name}`` placeholders. Returns a new structure."""
    overrides = overrides or {}
    if isinstance(state, dict):
        return {k: substitute_parameters(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [substitute_parameters(v, params, overrides) for v in state]
    return _resolve_value(state, params, overrides)


def _resolve_builder(builder, module):
    """Resolve a builder callable; a dotted ``pkg.mod:fn`` string is imported."""
    if callable(builder):
        return builder
    mod_name, _, qual = str(builder).partition(":")
    mod = importlib.import_module(mod_name or module)
    obj = mod
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj
```

Then add these methods to the `CompositeSpec` class body (place after `from_dict`):

```python
    def _merged_params(self, overrides):
        overrides = overrides or {}
        unknown = set(overrides) - set(self.parameters)
        if unknown:
            raise KeyError(f"unknown override(s): {sorted(unknown)}")
        merged = {k: v.get("default") for k, v in self.parameters.items()}
        merged.update(overrides)
        return merged

    def to_document(self, overrides=None, core=None) -> dict:
        if self.kind == "spec":
            return {
                "schema": substitute_parameters(self.schema, self.parameters, overrides),
                "state": substitute_parameters(self.state, self.parameters, overrides),
            }
        fn = _resolve_builder(self.builder, self.module)
        return fn(core=core, **self._merged_params(overrides))

    def to_composite(self, overrides=None, core=None):
        from process_bigraph import Composite, allocate_core
        if core is None:
            core = allocate_core()
        for ext in self.core_extensions:
            ext(core)
        doc = self.to_document(overrides, core=core)
        return Composite(doc, core=core)

    def default_state(self, base_dir=None) -> "dict | None":
        if self.state is not None:
            return self.state
        return None  # generator artifact path implemented in Task 5
```

Note `_merged_params` rejects unknown override keys (matches the test). A builder that
accepts a `config_overrides` map declares it as a parameter, so deep-path overrides
flow through the normal merge — no special handling here.

- [ ] **Step 4: Run → pass.** All Task-2 tests green; Task-1 tests still green.

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/composite_spec.py process_bigraph/tests/test_composite_spec.py
git commit -m "feat(composite-spec): to_document/to_composite/default_state + parameter substitution"
```

---

### Task 3: `@composite_spec` decorator + registry

**Files:**
- Modify: `process_bigraph/composite_spec.py`
- Test: `process_bigraph/tests/test_composite_spec.py` (append)

**Interfaces — Produces:** module registry `_REGISTRY: dict[str, CompositeSpec]`; `register(spec)`, `get(spec_id)`, `all_specs()`, `clear_registry()`; `composite_spec(*, name, description="", parameters=None, visualizations=None, emitters=None, analyses=None, tags=None, default_n_steps=None, core_extensions=None)` decorator returning the original fn and registering a `CompositeSpec(id="<module>.<name>", builder=fn, …)`.

- [ ] **Step 1: Write the failing tests**

```python
from process_bigraph import composite_spec as cs


def test_decorator_registers_generator(monkeypatch):
    cs.clear_registry()

    @cs.composite_spec(name="demo", description="d",
                       parameters={"seed": {"type": "int", "default": 0}},
                       emitters=[{"address": "local:RAMEmitter"}], default_n_steps=5)
    def demo(core=None, *, seed=0):
        return {"state": {"s": seed}}

    spec_id = f"{demo.__module__}.demo"
    spec = cs.get(spec_id)
    assert spec is not None and spec.kind == "generator"
    assert spec.parameters["seed"]["type"] == "integer"  # normalized
    assert spec.default_n_steps == 5 and spec.builder is demo
    assert demo(seed=2) == {"state": {"s": 2}}  # decorator returns the original fn


def test_register_get_all_clear():
    cs.clear_registry()
    s = CompositeSpec(id="m.c", name="c", state={})
    cs.register(s)
    assert cs.get("m.c") is s
    assert "m.c" in cs.all_specs()
    cs.clear_registry()
    assert cs.get("m.c") is None
```

- [ ] **Step 2: Run → fail.** `composite_spec`/`register` undefined.

- [ ] **Step 3: Implement** — append to `process_bigraph/composite_spec.py`:

```python
_REGISTRY: "dict[str, CompositeSpec]" = {}


def register(spec: CompositeSpec) -> None:
    _REGISTRY[spec.id] = spec


def get(spec_id: str) -> "CompositeSpec | None":
    return _REGISTRY.get(spec_id)


def all_specs() -> "dict[str, CompositeSpec]":
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()


def composite_spec(*, name, description="", parameters=None, visualizations=None,
                   emitters=None, analyses=None, tags=None, default_n_steps=None,
                   core_extensions=None, default_state_ref=None):
    """Decorator: register a generator function as a CompositeSpec.

    The wrapped fn becomes the spec's ``builder``; its id is
    ``"<fn.__module__>.<name>"``. Returns the original fn unchanged.
    """
    def decorate(fn):
        spec = CompositeSpec(
            id=f"{fn.__module__}.{name}",
            name=name,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            tags=list(tags or []),
            parameters=dict(parameters or {}),
            default_n_steps=default_n_steps,
            visualizations=list(visualizations or []),
            analyses=list(analyses or []),
            emitters=list(emitters or []),
            builder=fn,
            module=fn.__module__,
            default_state_ref=default_state_ref,
            core_extensions=list(core_extensions or []),
        )
        register(spec)
        return fn
    return decorate
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/composite_spec.py process_bigraph/tests/test_composite_spec.py
git commit -m "feat(composite-spec): @composite_spec decorator + module registry"
```

---

### Task 4: `from_file` loader + `discover_specs`

**Files:**
- Modify: `process_bigraph/composite_spec.py`
- Test: `process_bigraph/tests/test_composite_spec.py` (append)

**Interfaces — Consumes:** `yaml`. **Produces:** `CompositeSpec.from_file(path)` (parses a `*.composite.{yaml,json}` into a static-or-generator spec, normalizing types, carrying `schema`/`requires.{processes,types}`/`analyses`/`visualizations`/`default_n_steps`/`emitters`); `discover_specs(workspace=None)` — imports bigraph-schema-dependent packages (triggering decorators) and scans `<workspace>/**/*.composite.{yaml,json}`, merging both into `_REGISTRY` and returning it.

- [ ] **Step 1: Write the failing tests**

```python
import textwrap


def test_from_file_static_yaml(tmp_path):
    p = tmp_path / "g.composite.yaml"
    p.write_text(textwrap.dedent("""
        name: growth
        description: demo
        tags: [a]
        requires:
          processes: [P]
          types: [ty]
        schema:
          population: tree
        parameters:
          seed: {type: int, default: 0}
        state:
          v: "${seed}"
    """), encoding="utf-8")
    s = CompositeSpec.from_file(p)
    assert s.kind == "spec" and s.name == "growth"
    assert s.schema == {"population": "tree"}
    assert s.requires == {"processes": ["P"], "types": ["ty"]}
    assert s.parameters["seed"]["type"] == "integer"  # normalized
    assert s.id.endswith(".growth")


def test_from_file_generator_with_builder_ref(tmp_path):
    p = tmp_path / "b.composite.json"
    p.write_text(json.dumps({
        "name": "bgen", "builder": "process_bigraph.composite_spec:normalize_type",
        "default_state_ref": "bgen.default-state.json",
        "parameters": {}}), encoding="utf-8")
    s = CompositeSpec.from_file(p)
    assert s.kind == "generator" and s.default_state_ref == "bgen.default-state.json"


def test_discover_specs_scans_workspace_files(tmp_path):
    cs.clear_registry()
    comp = tmp_path / "composites"
    comp.mkdir()
    (comp / "x.composite.yaml").write_text("name: xc\nstate: {a: 1}\n", encoding="utf-8")
    found = cs.discover_specs(workspace=tmp_path)
    assert any(s.name == "xc" for s in found.values())
```

- [ ] **Step 2: Run → fail.** `from_file`/`discover_specs` undefined.

- [ ] **Step 3: Implement** — add `from_file` as a `CompositeSpec` classmethod and `discover_specs` at module level:

```python
    @classmethod
    def from_file(cls, path) -> "CompositeSpec":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"composite spec {path} must parse to a dict")
        name = raw.get("name")
        # id: prefer an explicit id, else "<file-stem-stripped>.<name>"; keep stable + simple
        stem = path.name.replace(".composite.yaml", "").replace(".composite.json", "")
        spec_id = raw.get("id") or f"{stem}.{name}"
        builder = raw.get("builder")
        return cls(
            id=spec_id, name=name, description=raw.get("description", ""),
            tags=list(raw.get("tags") or []),
            parameters=dict(raw.get("parameters") or {}),
            default_n_steps=raw.get("default_n_steps"),
            visualizations=list(raw.get("visualizations") or []),
            analyses=list(raw.get("analyses") or []),
            emitters=list(raw.get("emitters") or []),
            requires=dict(raw.get("requires") or {}),
            schema=dict(raw.get("schema") or {}) if builder is None else {},
            state=raw.get("state") if builder is None else None,
            builder=builder,
            default_state_ref=raw.get("default_state_ref"),
            module=raw.get("module", ""),
        )
```

```python
def discover_specs(workspace=None) -> "dict[str, CompositeSpec]":
    """Populate + return the registry: import decorator-registered generators
    AND scan a workspace for ``*.composite.{yaml,json}`` files."""
    try:
        from pbg_superpowers.composite_generator import discover_generators
        discover_generators()  # fires @composite_spec / @composite_generator on import
    except Exception:
        pass  # discovery of code generators is best-effort
    if workspace is not None:
        for pat in ("*.composite.yaml", "*.composite.json"):
            for fp in Path(workspace).rglob(pat):
                try:
                    register(CompositeSpec.from_file(fp))
                except Exception:
                    continue  # a malformed file must not break discovery
    return all_specs()
```

(The `discover_generators` import is the transition bridge; after the shim lands — Task 7 —
it triggers the same `@composite_spec` registrations. The file scan is independent of it.)

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/composite_spec.py process_bigraph/tests/test_composite_spec.py
git commit -m "feat(composite-spec): from_file loader + discover_specs (decorator + file merge)"
```

---

### Task 5: `regenerate_default_state` + artifact-backed `default_state`

**Files:**
- Modify: `process_bigraph/composite_spec.py`
- Test: `process_bigraph/tests/test_composite_spec.py` (append)

**Interfaces — Consumes:** `CompositeSpec.to_composite` (Task 2), `Composite.serialize_state()`. **Produces:** `regenerate_default_state(spec, base_dir, core=None) -> Path` (runs the builder with defaults, writes `<base_dir>/<default_state_ref>` containing `{"state": <serialized>, "_provenance": {param_signature}}`); `CompositeSpec.default_state(base_dir=None)` extended so a generator reads its artifact's `state`.

- [ ] **Step 1: Write the failing tests**

```python
def test_regenerate_and_read_default_state(tmp_path):
    def build(core=None, *, seed=0):
        return {"state": {"count": seed + 1}}
    s = CompositeSpec(id="m.g", name="g", builder=build,
                      parameters={"seed": {"type": "integer", "default": 4}},
                      default_state_ref="g.default-state.json")
    from process_bigraph.composite_spec import regenerate_default_state
    out = regenerate_default_state(s, tmp_path)
    assert out.exists()
    # display reads the saved state, no rebuild
    assert s.default_state(base_dir=tmp_path) == {"count": 5}


def test_default_state_missing_artifact_returns_none(tmp_path):
    s = CompositeSpec(id="m.g", name="g", builder=lambda core=None: {"state": {}},
                      default_state_ref="absent.json")
    assert s.default_state(base_dir=tmp_path) is None
```

- [ ] **Step 2: Run → fail.** `regenerate_default_state` undefined; `default_state` returns None for the artifact case.

- [ ] **Step 3: Implement** — replace `default_state` and add `regenerate_default_state`:

```python
    def default_state(self, base_dir=None) -> "dict | None":
        if self.state is not None:
            return self.state
        if self.default_state_ref and base_dir is not None:
            artifact = Path(base_dir) / self.default_state_ref
            if artifact.is_file():
                data = json.loads(artifact.read_text(encoding="utf-8"))
                return data.get("state", data)
        return None
```

```python
def regenerate_default_state(spec: CompositeSpec, base_dir, core=None) -> "Path":
    """Run a generator's builder with default params, serialize the materialized
    state, and write the ``default_state_ref`` artifact (+ a provenance stamp).

    This is the one step that needs the heavy build environment (e.g. a ParCa
    cache for v2ecoli baseline). Display thereafter reads the artifact, no build.
    """
    if spec.kind != "generator" or not spec.default_state_ref:
        raise ValueError("regenerate_default_state requires a generator with default_state_ref")
    comp = spec.to_composite(core=core)
    state = comp.serialize_state()
    param_sig = {k: v.get("default") for k, v in spec.parameters.items()}
    out = Path(base_dir) / spec.default_state_ref
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"state": state, "_provenance": {"param_signature": param_sig}},
                              indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/composite_spec.py process_bigraph/tests/test_composite_spec.py
git commit -m "feat(composite-spec): regenerate_default_state + artifact-backed default_state"
```

---

### Task 6: Export from `process_bigraph.__init__`

**Files:**
- Modify: `process_bigraph/__init__.py:3-8`
- Test: `process_bigraph/tests/test_composite_spec.py` (append)

**Interfaces — Produces:** top-level imports `from process_bigraph import CompositeSpec, composite_spec, discover_specs`.

- [ ] **Step 1: Write the failing test**

```python
def test_top_level_exports():
    import process_bigraph as pbg
    assert hasattr(pbg, "CompositeSpec")
    assert hasattr(pbg, "composite_spec")      # the decorator
    assert hasattr(pbg, "discover_specs")
```

- [ ] **Step 2: Run → fail.** AttributeError.

- [ ] **Step 3: Implement** — add to `process_bigraph/__init__.py` after the existing `from process_bigraph.composite import (...)` block (around line 6):

```python
from process_bigraph.composite_spec import (
    CompositeSpec, composite_spec, discover_specs, regenerate_default_state,
    register, get, all_specs, clear_registry, normalize_type, CANONICAL_TYPES,
)
```

- [ ] **Step 4: Run → pass.** Also run the WHOLE process-bigraph suite to confirm no import cycle: `.venv/bin/python -m pytest -q` (composite_spec imports `process_bigraph.Composite` lazily inside methods, so no cycle at import time).

- [ ] **Step 5: Commit**

```bash
git add process_bigraph/__init__.py process_bigraph/tests/test_composite_spec.py
git commit -m "feat(composite-spec): export CompositeSpec API from process_bigraph"
```

---

### Task 7: pbg-superpowers shim — `composite_generator` delegates to process-bigraph

**Worktree (create first):** `git -C /Users/eranagmon/code/pbg-superpowers worktree add -b feat/composite-spec-shim /Users/eranagmon/code/pbg-spec-shim origin/main` — work in `/Users/eranagmon/code/pbg-spec-shim`.

**Files:**
- Modify: `pbg_superpowers/composite_generator.py` (decorator + `build_generator` + `discover_generators` delegate; keep `GeneratorEntry`, `emitter_defaults`, `apply_core_extensions`, `install_default_emitters` intact for the dashboard's reads)
- Test: `tests/test_composite_generator.py` (append delegation tests)

**Interfaces — Consumes:** `process_bigraph.composite_spec` (Tasks 1–6). **Produces:** unchanged public surface — `@composite_generator(...)` still works and registers into the process-bigraph registry; `_REGISTRY` reflects those registrations as `GeneratorEntry`-compatible objects; `build_generator(entry, overrides)` returns the builder document; `discover_generators()` returns the entries.

**Constraint:** the dashboard reads these `entry` attributes today — preserve them all: `.id, .name, .description, .parameters, .module, .default_n_steps, .visualizations, .emitters, .core_extensions, .func`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composite_generator.py (append)
def test_composite_generator_registers_into_process_bigraph_registry():
    from process_bigraph import composite_spec as cs
    from pbg_superpowers.composite_generator import composite_generator, build_generator
    cs.clear_registry()

    @composite_generator(name="shimdemo", parameters={"seed": {"type": "int", "default": 1}})
    def shimdemo(core=None, *, seed=1):
        return {"state": {"s": seed}}

    spec_id = f"{shimdemo.__module__}.shimdemo"
    spec = cs.get(spec_id)
    assert spec is not None and spec.parameters["seed"]["type"] == "integer"
    # build_generator delegates to the spec's document
    assert build_generator(spec, overrides={"seed": 3}) == {"state": {"s": 3}}
```

- [ ] **Step 2: Run → fail.** Old `composite_generator` writes its own `_REGISTRY`, not process-bigraph's.

- [ ] **Step 3: Implement** — in `pbg_superpowers/composite_generator.py`:
  1. Keep the `GeneratorEntry` dataclass (the dashboard's read shape) and `emitter_defaults`/`apply_core_extensions`/`install_default_emitters` as-is.
  2. Make `composite_generator(...)` build a `process_bigraph.composite_spec.CompositeSpec` (via `cs.composite_spec(...)`), then ALSO register a `GeneratorEntry` view so `_REGISTRY` exposes the dashboard-read attributes. Implement `_REGISTRY` as a module object whose `get`/`__contains__`/`values`/`items` read from `cs.all_specs()` and wrap each `CompositeSpec` in a `GeneratorEntry` (mapping `spec.builder→.func`, `spec.*→.*`). Simplest correct form:

```python
from process_bigraph import composite_spec as _cs

def composite_generator(*, name, description="", parameters=None, visualizations=None,
                        emitters=None, default_n_steps=None, core_extensions=None,
                        default_state_ref=None):
    _validate_emitters(emitters, name)
    def decorate(fn):
        _cs.composite_spec(
            name=name, description=description, parameters=parameters,
            visualizations=visualizations, emitters=emitters,
            default_n_steps=default_n_steps, core_extensions=core_extensions,
            default_state_ref=default_state_ref,
        )(fn)
        return fn
    return decorate


def _entry_for(spec):
    """Adapt a process-bigraph CompositeSpec to the GeneratorEntry the dashboard reads."""
    return GeneratorEntry(
        id=spec.id, name=spec.name, description=spec.description,
        parameters=spec.parameters, func=_cs._resolve_builder(spec.builder, spec.module),
        module=spec.module, default_n_steps=spec.default_n_steps,
        visualizations=spec.visualizations, emitters=spec.emitters,
        core_extensions=spec.core_extensions,
    )


class _RegistryView:
    def get(self, k, default=None):
        s = _cs.get(k)
        return _entry_for(s) if s is not None else default
    def __getitem__(self, k):
        s = _cs.get(k)
        if s is None:
            raise KeyError(k)
        return _entry_for(s)
    def __contains__(self, k):
        return _cs.get(k) is not None
    def values(self):
        return [_entry_for(s) for s in _cs.all_specs().values()]
    def items(self):
        return [(sid, _entry_for(s)) for sid, s in _cs.all_specs().items()]
    def __iter__(self):
        return iter(_cs.all_specs())
    def __bool__(self):
        return bool(_cs.all_specs())
    def clear(self):
        _cs.clear_registry()

_REGISTRY = _RegistryView()


def build_generator(entry, overrides=None, core=None):
    """Delegate to the CompositeSpec document. ``entry`` may be a GeneratorEntry
    (has ``.id``) or a CompositeSpec."""
    spec = _cs.get(getattr(entry, "id", None)) or entry
    return spec.to_document(overrides, core=core)


def discover_generators(extra_search_paths=None):
    # keep the existing distribution-walking import logic; then return entries
    _import_bigraph_packages(extra_search_paths)   # existing helper that imports packages
    return {sid: _entry_for(s) for sid, s in _cs.all_specs().items()
            if s.kind == "generator"}
```

  (Keep the existing distribution-walk body of `discover_generators` — extract it into `_import_bigraph_packages` if not already separate. `apply_core_extensions(entry, core)` keeps working: it reads `entry.core_extensions`.)

- [ ] **Step 4: Run → pass.** `cd /Users/eranagmon/code/pbg-spec-shim && python -m pytest tests/test_composite_generator.py -v`. Also run the full `pytest -q` to confirm no regression in the dashboard-facing reads.

- [ ] **Step 5: Commit** `feat(shim): composite_generator delegates to process-bigraph CompositeSpec registry`.

---

### Task 8: pbg-superpowers shim — `composite_spec` (static) + `composite_discovery` delegate

**Worktree:** same `/Users/eranagmon/code/pbg-spec-shim`.

**Files:**
- Modify: `pbg_superpowers/composite_spec.py` (delegate parsing/substitution/build to process-bigraph; keep `install_default_emitters` call for behaviour parity)
- Modify: `pbg_superpowers/composite_discovery.py` (`discover_all` returns specs via `CompositeSpec.from_file`)
- Test: `tests/test_composite_spec.py` (append)

**Interfaces — Consumes:** `process_bigraph.composite_spec`. **Produces:** unchanged signatures — `load_spec(path)`, `validate_spec(spec)`, `substitute_parameters(state, params, overrides)`, `build_composite_from_spec(spec, overrides=None, core=None)`; `discover_all(...)` still returns `{id: {kind, …}}`.

**Constraint:** `build_composite_from_spec` must preserve its existing emitter behaviour (it calls `install_default_emitters` before constructing the Composite) and its `requires.processes` preflight.

- [ ] **Step 1: Write the failing/championing test**

```python
def test_build_composite_from_spec_uses_unified_substitution():
    from pbg_superpowers.composite_spec import build_composite_from_spec
    spec = {"name": "c", "parameters": {"seed": {"type": "int", "default": 2}},
            "state": {"v": "${seed}"}}
    comp = build_composite_from_spec(spec, overrides={"seed": 9})
    # substitution happened via the unified engine (typed int)
    assert comp.state["v"] == 9 if hasattr(comp, "state") else True


def test_substitute_parameters_delegates():
    from pbg_superpowers import composite_spec as legacy
    out = legacy.substitute_parameters({"a": "${x}"}, {"x": {"type": "int", "default": 0}}, {"x": 5})
    assert out == {"a": 5}
```

- [ ] **Step 2: Run → fail / verify.** (If behaviour already matches, make `substitute_parameters` a re-export to prove single-sourcing.)

- [ ] **Step 3: Implement** — re-point the static module at the unified engine, keeping the wrapper behaviour:

```python
from process_bigraph.composite_spec import substitute_parameters  # single source

# load_spec stays (file IO). validate_spec stays but widen the allowed types to the
# canonical vocabulary + aliases (accept what normalize_type accepts) so existing
# files using integer/boolean/number/object don't raise.

def build_composite_from_spec(spec, overrides=None, core=None):
    validate_spec(spec)
    from process_bigraph import Composite, allocate_core
    from process_bigraph.composite_spec import CompositeSpec
    if core is None:
        core = allocate_core()
    # requires.processes preflight (unchanged)
    requires = spec.get("requires") or {}
    proc_required = requires.get("processes") or []
    link_registry = getattr(core, "link_registry", {}) or {}
    missing = [p for p in proc_required if p not in link_registry]
    if missing:
        raise RuntimeError(f"composite spec '{spec.get('name')}' requires processes "
                           f"not in registry: {missing}.")
    cspec = CompositeSpec(
        id=spec.get("id") or f"spec.{spec.get('name')}", name=spec.get("name"),
        state=spec.get("state") or {}, schema=dict(spec.get("schema") or {}),
        parameters=dict(spec.get("parameters") or {}), requires=requires,
        emitters=list(spec.get("emitters") or []))
    doc = cspec.to_document(overrides)
    state = doc["state"]
    from pbg_superpowers.composite_generator import install_default_emitters
    state = install_default_emitters(state, spec, core=core)
    return Composite({"schema": doc.get("schema") or {}, "state": state}, core=core)
```

  Update `validate_spec` line 57 to accept the canonical+alias type set:
  `if "type" in pdef and normalize_type(pdef["type"]) not in CANONICAL_TYPES:` (import both from `process_bigraph.composite_spec`).
  In `composite_discovery.discover_all`, build spec entries via `CompositeSpec.from_file(fp).to_dict()` tagged `kind` (generator entries already come from the generator shim).

- [ ] **Step 4: Run → pass.** `python -m pytest tests/test_composite_spec.py tests/test_composite_generator.py -v` + full `pytest -q`.

- [ ] **Step 5: Commit** `feat(shim): static composite_spec + discovery delegate to process-bigraph; widen type vocab`.

---

### Task 9: Dashboard — resolve against the new contract + structured wiring payload

**Worktree (create first):** `git -C /Users/eranagmon/code/vivarium-dashboard worktree add -b feat/composite-spec-resolve /Users/eranagmon/code/vdash-composite-spec origin/main` — work in `/Users/eranagmon/code/vdash-composite-spec`.

**Files:**
- Modify: `vivarium_dashboard/lib/composite_resolve.py` (resolve via process-bigraph registry; add `default_state` wiring + `wiring_status`/`notice`)
- Modify: `vivarium_dashboard/api/app.py` (`/api/composite-resolve` returns 200-with-metadata when wiring is unavailable, not a swallowed 404)
- Modify: `vivarium_dashboard/lib/models.py` (`CompositeResolvePayload`: add optional `analyses`, `visualizations`, `emitters`, `schema`, `requires`, `tags`, `wiring_status`, `notice`)
- Test: `tests/test_composite_resolve_dispatch.py` (or the existing composite-resolve test file)

**Interfaces — Consumes:** `process_bigraph.composite_spec.{discover_specs,get}`, `CompositeSpec.default_state`. **Produces:** `resolve_composite(ws_root, spec_id, overrides=None) -> dict | None` returning `{id, name, description, parameters, state, schema, requires, tags, visualizations, analyses, emitters, kind, module, default_n_steps, wiring_status, notice}`. `wiring_status ∈ {"ready", "unavailable"}`.

**Constraint:** a generator whose default-state artifact is missing returns **200 + metadata + `wiring_status:"unavailable"` + honest `notice`**, NOT a swallowed None→404. A genuinely-unregistered id still returns None→404.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_generator_without_artifact_degrades(tmp_path, monkeypatch):
    from process_bigraph import composite_spec as cs
    from vivarium_dashboard.lib import composite_resolve as cr
    cs.clear_registry()
    cs.register(cs.CompositeSpec(id="m.g", name="g", builder=lambda core=None: {"state": {}},
                                 default_state_ref="m.g.default-state.json",
                                 parameters={"seed": {"type": "integer", "default": 0}}))
    monkeypatch.setattr(cr, "discover_specs", lambda ws=None: cs.all_specs())
    out = cr.resolve_composite(tmp_path, "m.g")
    assert out["wiring_status"] == "unavailable" and out["notice"]
    assert out["parameters"]["seed"]["type"] == "integer"   # metadata present without build
    assert out["state"] is None


def test_resolve_static_ready(tmp_path, monkeypatch):
    from process_bigraph import composite_spec as cs
    from vivarium_dashboard.lib import composite_resolve as cr
    cs.clear_registry()
    cs.register(cs.CompositeSpec(id="m.c", name="c", state={"v": 1}, schema={"v": "float"}))
    monkeypatch.setattr(cr, "discover_specs", lambda ws=None: cs.all_specs())
    out = cr.resolve_composite(tmp_path, "m.c")
    assert out["wiring_status"] == "ready" and out["state"] == {"v": 1}


def test_resolve_unregistered_returns_none(tmp_path, monkeypatch):
    from process_bigraph import composite_spec as cs
    from vivarium_dashboard.lib import composite_resolve as cr
    cs.clear_registry()
    monkeypatch.setattr(cr, "discover_specs", lambda ws=None: {})
    assert cr.resolve_composite(tmp_path, "absent") is None
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — rewrite `resolve_composite` in `composite_resolve.py` to use the unified registry (keep `resolve_composite_for_request` from SP-D1 wrapping it for remote builds):

```python
from process_bigraph.composite_spec import discover_specs, get as _get_spec  # module-level for monkeypatch

def resolve_composite(ws_root, spec_id, overrides=None):
    ws_root = Path(ws_root)
    _ws_add_to_sys_path(ws_root)
    specs = discover_specs(workspace=ws_root)
    spec = specs.get(spec_id)
    if spec is None:
        return None
    base_dir = _artifact_base_dir(ws_root, spec)   # where default-state artifacts live
    state = spec.default_state(base_dir=base_dir)
    wiring_status = "ready" if state is not None else "unavailable"
    notice = None
    if wiring_status == "unavailable":
        notice = (f"default state for generator '{spec.name}' is not generated yet — "
                  f"run it, or regenerate its default-state artifact to see wiring.")
    try:
        from vivarium_dashboard.lib.process_docs import attach_process_docs
        if state is not None:
            attach_process_docs(state)
    except Exception:
        pass
    return {
        "id": spec.id, "name": spec.name, "description": spec.description,
        "parameters": spec.parameters, "state": state, "schema": spec.schema,
        "requires": spec.requires, "tags": spec.tags,
        "visualizations": spec.visualizations, "analyses": spec.analyses,
        "emitters": spec.emitters, "kind": spec.kind, "module": spec.module,
        "default_n_steps": spec.default_n_steps, "svg": None,
        "wiring_status": wiring_status, "notice": notice,
    }
```

  Add `_artifact_base_dir(ws_root, spec)` returning the dashboard's existing
  `api/composite-state/` dir if present else `ws_root` (resolve in code review against
  the §13 artifact-location decision). In `app.py`, the `/api/composite-resolve` handler
  returns `CompositeResolvePayload.model_validate(result)` for a non-None result (200,
  including `wiring_status:"unavailable"`), and only 404s when `result is None`. Add the
  new optional fields to `CompositeResolvePayload` in `models.py`.

- [ ] **Step 4: Run → pass.** Test command: `cd /Users/eranagmon/code/vdash-composite-spec && PYTHONPATH=$PWD:/Users/eranagmon/code/investigation-contracts /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/test_composite_resolve_dispatch.py -v`.

- [ ] **Step 5: Commit** `feat(dashboard): resolve composites via process-bigraph CompositeSpec + structured wiring payload`.

---

### Task 10: Dashboard — Explorer bug fix (remove the hardcoded remote-build error)

**Worktree:** same `/Users/eranagmon/code/vdash-composite-spec`.

**Files:**
- Modify: `vivarium_dashboard/static/walkthrough.js:~4176-4190` (`_ceFetch`)
- Modify: `vivarium_dashboard/static/walkthrough.js` (the explorer render path — surface `wiring_status`/`notice`)
- Test: add a small JS-logic test if the repo has a JS test harness; otherwise a Python smoke test asserting the served file no longer contains the hardcoded string.

**Interfaces — Consumes:** the Task 9 payload (`wiring_status`, `notice`). **Produces:** the explorer renders metadata + the server `notice` for `wiring_status:"unavailable"`; no hardcoded remote-build text.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_explorer_no_hardcoded_remote_build.py
from pathlib import Path
def test_walkthrough_has_no_hardcoded_remote_build_message():
    js = Path("vivarium_dashboard/static/walkthrough.js").read_text(encoding="utf-8")
    assert "remote build cannot build generator composites" not in js
```

- [ ] **Step 2: Run → fail.** The string is present.

- [ ] **Step 3: Implement** — in `_ceFetch`, replace the hardcoded fallback message with the server's real error, and render `wiring_status`/`notice`:

```javascript
var msg = (d && (d.error || d.detail || d.notice)) ? (d.error || d.detail || d.notice)
  : ('HTTP ' + r.status + ' — could not resolve this composite.');
return { error: msg };
```

  In the explorer render path, when a resolved payload has `wiring_status === 'unavailable'`,
  show the config form + metadata and render `notice` as an inline info banner (not an error),
  instead of treating the missing wiring as a hard failure.

- [ ] **Step 4: Run → pass.** Plus a manual check: open the Explorer on a local generator with no artifact → metadata + honest notice, no "remote build" text.

- [ ] **Step 5: Commit** `fix(dashboard): replace hardcoded remote-build explorer error with the server notice`.

---

### Task 11: Proof — migrate `baseline` (generator) + `growth-division` (static); generate baseline artifact

**Worktrees:** v2ecoli (`/Users/eranagmon/code/v2e-main` or a fresh worktree off origin/main) + pbg-autopoiesis (fresh worktree off origin/main). The artifact-generation step needs a ParCa cache (run on the mini or a checkout with `out/cache`).

**Files:**
- Modify: `v2ecoli/v2ecoli/composites/baseline.py` — the `@composite_generator` decorator now flows through the shim (no source change required); ADD `default_state_ref="baseline.default-state.json"` once the decorator/shim supports passing it through (extend the generator shim decorator to accept + forward `default_state_ref` — a one-line addition in Task 7's `composite_generator`).
- Create: `v2ecoli/.../composites/baseline.default-state.json` (generated artifact, committed)
- Modify (if needed): `pbg-autopoiesis/.../growth-division.composite.yaml` — confirm it loads via `CompositeSpec.from_file` unchanged (it already has `schema: {population: tree}` + `parameters` + `state`).
- Test: `v2ecoli` smoke test + autopoiesis smoke test.

**Interfaces — Consumes:** the full stack (Tasks 1–10).

- [ ] **Step 1: Static proof (headless).** In an autopoiesis worktree with process-bigraph + the shims installed, assert the YAML loads + resolves:

```python
def test_growth_division_loads_as_composite_spec():
    from process_bigraph.composite_spec import CompositeSpec
    p = "pbg_autopoiesis/composites/growth-division.composite.yaml"
    s = CompositeSpec.from_file(p)
    assert s.kind == "spec" and s.schema == {"population": "tree"}
    assert s.default_state() == s.state  # static display state present, no build
```

Run → pass.

- [ ] **Step 2: Generator metadata proof (headless).** Assert baseline registers with the new contract and exposes metadata WITHOUT a build:

```python
def test_baseline_registers_with_metadata():
    import v2ecoli.composites.baseline  # triggers @composite_generator
    from process_bigraph import composite_spec as cs
    spec = cs.get("v2ecoli.composites.baseline.baseline")  # id = "<module>.<name>"
    assert spec is not None and spec.kind == "generator"
    assert spec.default_n_steps == 2700
    assert all(p["type"] in cs.CANONICAL_TYPES for p in spec.parameters.values())  # normalized
    assert spec.default_state_ref == "baseline.default-state.json"
```

Run → pass (after adding `default_state_ref` to the baseline decorator + the shim forward).

- [ ] **Step 3: Generate the artifact (IN-THE-LOOP — needs a ParCa cache).** On the mini (or a checkout with `out/cache`):

```bash
cd <v2ecoli-with-cache>
.venv/bin/python -c "import v2ecoli.composites.baseline; \
from process_bigraph import composite_spec as cs, regenerate_default_state; \
spec = cs.get('v2ecoli.composites.baseline.baseline'); \
print(regenerate_default_state(spec, 'v2ecoli/composites'))"
```

Commit the generated `baseline.default-state.json`.

- [ ] **Step 4: Dashboard display proof.** Start the dashboard on the v2ecoli workspace; open the Composite Explorer for `baseline` with NO ParCa cache on the dashboard host → it renders the config form + metadata + wiring from the artifact, no ParCa, no "remote build" error. Open `growth-division` → renders from inline state.

- [ ] **Step 5: Commit** the artifact + any `default_state_ref` wiring + smoke tests: `feat(proof): baseline + growth-division on CompositeSpec; commit baseline default-state artifact`.

---

## Self-Review

**Spec coverage:**
- §2 architecture + §3 contract → Tasks 1–6 (dataclass, types, resolution, decorator, registry, discovery, regenerate, exports). ✓
- §3 `schema` + `requires.{processes,types}` + whole-document passthrough + canonical types → Task 1 (validation/normalization) + Task 2 (`to_document` passthrough) + Task 4 (`from_file` carries schema/requires). ✓
- §5 two front-ends + shim → Tasks 7 (generator) + 8 (static + discovery). ✓
- §4 registry & discovery → Tasks 3 + 4. ✓
- §6 dashboard resolve + display robustness + bug fix → Tasks 9 + 10. ✓
- §3a/§4 saved default state → Task 5 (`regenerate_default_state` + artifact `default_state`) + Task 9 (display from it). ✓
- §10 testing + §11 proof → Task 11 (static headless + generator metadata headless + the one in-the-loop artifact gen + display proof). ✓
- §3a config_overrides convention → Task 2 note (`_merged_params` forwards a declared `config_overrides` param; no special logic). ✓

**Placeholder scan:** Task 9's `_artifact_base_dir` and Task 11's `default_state_ref` decorator-forward are the two spots that defer a small decision to code review (artifact location, §13 OQ2) — both have a concrete default specified, not a TODO. No "add error handling"/"TBD" placeholders.

**Type consistency:** `CompositeSpec` field names + method signatures are fixed in the Contract block and reused verbatim in Tasks 1–11. Registry fns named `register/get/all_specs/clear_registry` consistently (note: `all_specs`, not `all`, to avoid shadowing the builtin — used the same in tests). `discover_specs(workspace=...)`, `regenerate_default_state(spec, base_dir, core=None)`, `default_state(base_dir=None)`, `to_document(overrides=None, core=None)` consistent across the dashboard task and the proof task. The generator shim maps `CompositeSpec.builder → GeneratorEntry.func` consistently in `_entry_for`.

## Execution Handoff

Pre-flight conflict: Task 11 requires the generator shim decorator (Task 7) to accept + forward a `default_state_ref` kwarg — fold that one-line addition into Task 7 (note added there). No other cross-task contradictions found.
