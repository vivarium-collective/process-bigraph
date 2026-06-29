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

import re
import json
import importlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Any

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
            module = self.module if self.module else getattr(self.builder, '__module__', '')
            d["builder"] = f"{module}:{self.builder.__name__}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSpec":
        d = dict(d)
        d.pop("core_extensions", None)
        return cls(**d)

    @classmethod
    def from_file(cls, path) -> "CompositeSpec":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            import yaml
            raw = yaml.safe_load(text)
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

    def _merged_params(self, overrides):
        overrides = overrides or {}
        unknown = set(overrides) - set(self.parameters)
        if unknown:
            raise KeyError(f"unknown override(s): {sorted(unknown)}")
        merged = {k: v.get("default") for k, v in self.parameters.items()}
        merged.update(overrides)
        return merged

    def to_document(self, overrides=None, core=None) -> dict:
        # Validate overrides for both static and generator specs
        self._merged_params(overrides)
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
        if self.default_state_ref and base_dir is not None:
            artifact = Path(base_dir) / self.default_state_ref
            if artifact.is_file():
                data = json.loads(artifact.read_text(encoding="utf-8"))
                return data.get("state", data)
        return None


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
