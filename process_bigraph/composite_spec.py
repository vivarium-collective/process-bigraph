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


class ParameterTypeError(TypeError):
    """A parameter value that does not fit its declared type."""


_TRUE_WORDS = ("true", "1", "yes", "on")
_FALSE_WORDS = ("false", "0", "no", "off")

_VALIDATION_CORE = None


def _validation_core(core=None):
    """A core to type-check parameter values against.

    Prefers the caller's core; otherwise builds a base-types-only core once.
    ``allocate_core()`` costs seconds because it walks every installed
    package, and checking ``float``/``integer``/``string``/``boolean`` needs
    none of that — keeping ``to_document`` the cheap dict walk it has always
    been.
    """
    if core is not None:
        return core

    global _VALIDATION_CORE
    if _VALIDATION_CORE is None:
        from bigraph_schema import Core
        from bigraph_schema.schema import BASE_TYPES
        _VALIDATION_CORE = Core(BASE_TYPES)
    return _VALIDATION_CORE


def _reject(name, declared, value, why):
    label = f"parameter '{name}'" if name else "parameter"
    raise ParameterTypeError(
        f"{label} is declared '{declared}' but got {value!r} "
        f"({type(value).__name__}): {why}")


def _coerce(value, declared_type, name=None, core=None):
    """Coerce a parameter value to its declared type, refusing to mangle it.

    ``_cast`` coerced silently: ``int(3.7)`` quietly became ``3`` and any
    unrecognized string quietly became ``False``, so a typo in an override
    produced a plausible-looking but wrong simulation. The coercions callers
    genuinely rely on still pass — an ``int`` for a ``float`` parameter, a
    numeric string from a form field — but a lossy or meaningless one now
    raises, naming the parameter, its declared type and the offending value.

    The coerced result is then checked against the declared type with
    ``core.check``. A declared type this core does not know is left alone
    rather than guessed at.
    """
    if declared_type is None:
        return value

    declared = normalize_type(declared_type)

    if value is None:
        _reject(name, declared, value, "no value was supplied")

    # bool is a subclass of int, so it would silently satisfy the numeric
    # coercions; a boolean where a number belongs is a mistake, not a widening.
    if declared in ("float", "integer") and isinstance(value, bool):
        _reject(name, declared, value, "a boolean is not a number")

    if declared == "float":
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                _reject(name, declared, value, "not a number")
        elif not isinstance(value, (int, float)):
            _reject(name, declared, value, "not a number")

    elif declared == "integer":
        if isinstance(value, float) and value != int(value):
            _reject(name, declared, value,
                    "would lose its fractional part")
        elif isinstance(value, str):
            try:
                int(value)
            except ValueError:
                _reject(name, declared, value, "not a whole number")
        elif not isinstance(value, (int, float)):
            _reject(name, declared, value, "not a whole number")

    elif declared == "boolean":
        if isinstance(value, str):
            if value.strip().lower() not in _TRUE_WORDS + _FALSE_WORDS:
                _reject(name, declared, value,
                        f"not a recognized boolean "
                        f"(use one of {list(_TRUE_WORDS + _FALSE_WORDS)})")
        elif not isinstance(value, (bool, int)):
            _reject(name, declared, value, "not a boolean")

    elif declared in ("list", "map"):
        expected = list if declared == "list" else dict
        if not isinstance(value, expected):
            _reject(name, declared, value, f"not a {declared}")

    coerced = _cast(value, declared_type)

    # Final guarantee: the coerced value really is of the declared type.
    # An unrecognized declared type (a workspace type, say) is left alone.
    try:
        valid = _validation_core(core).check(declared, coerced)
    except Exception:
        return coerced

    if not valid:
        _reject(name, declared, value,
                f"resolved to {coerced!r}, which is not a valid {declared}")

    return coerced


def _resolve_value(value, params, overrides, core=None):
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
        return _coerce(raw, pdef.get("type"), name=pname, core=core)
    if _INLINE_PLACEHOLDER.search(value):
        def repl(match):
            pname = match.group(1)
            if pname not in params:
                raise KeyError(f"parameter '{pname}' referenced in state but not declared")
            raw = overrides.get(pname, params[pname].get("default"))
            return str(raw)
        return _INLINE_PLACEHOLDER.sub(repl, value)
    return value


def substitute_parameters(state, params, overrides=None, core=None):
    """Recursively substitute ``${name}`` placeholders. Returns a new structure.

    Each substituted value is type-checked against its declared parameter type
    (see :func:`_coerce`); ``core`` is only needed to resolve a declared type
    the base types do not cover.
    """
    overrides = overrides or {}
    if isinstance(state, dict):
        return {k: substitute_parameters(v, params, overrides, core) for k, v in state.items()}
    if isinstance(state, list):
        return [substitute_parameters(v, params, overrides, core) for v in state]
    return _resolve_value(state, params, overrides, core)


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
        # normalize parameter types (non-mutating rebuild)
        self.parameters = {
            k: ({**v, "type": normalize_type(v["type"])}
                if isinstance(v, dict) and "type" in v else v)
            for k, v in self.parameters.items()
        }

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

    def to_document(self, overrides=None, core=None, emit=True) -> dict:
        """Build the composite document.

        ``emitters`` is a first-class field of a spec, so the document it
        produces carries the declared sinks — otherwise a composite built
        through this API observes nothing. Pass ``emit=False`` for the bare
        document (e.g. to inspect or re-emit it under different sinks).
        """
        # Validate overrides for both static and generator specs
        self._merged_params(overrides)
        if self.kind == "spec":
            doc = {
                "schema": substitute_parameters(self.schema, self.parameters, overrides, core),
                "state": substitute_parameters(self.state, self.parameters, overrides, core),
            }
        else:
            fn = _resolve_builder(self.builder, self.module)
            doc = fn(core=core, **self._merged_params(overrides))

        return self._with_emitters(doc, core) if emit else doc

    def _with_emitters(self, doc, core=None):
        """Install this spec's declared emitters into a built document.

        The declared ``emitters`` are a *default* observation sink, applied
        only when the builder produced a document that observes nothing. A
        builder that already installs and configures its own emitter — e.g. a
        workspace generator that resolves a run-specific ``out_dir`` and nests
        a fully-built emitter instance inside an agent sub-composite — keeps
        it untouched: reinstalling the bare declaration on top would add a
        second, differently-configured sink (which, for a ParquetEmitter,
        realizes with an empty ``out_dir`` and fails). A composite built
        through the pure API whose builder installs no sink still gets the
        declared one, so the pure-API path is not left observing nothing.

        Installing is idempotent — the declared sinks land at fixed
        ``emitter`` / ``emitter_<i>`` keys — so a caller that installs the same
        declaration again rewrites the same slots rather than adding a second
        sink.
        """
        from process_bigraph.emitter import install_emitters, document_has_emitter

        if not self.emitters or not isinstance(doc, dict):
            return doc

        if "state" in doc:
            state = doc["state"] or {}
            if document_has_emitter(state, core):
                return doc
            return {**doc, "state": install_emitters(state, self.emitters, core=core)}

        # A builder that returned a bare state tree.
        if document_has_emitter(doc, core):
            return doc
        return install_emitters(doc, self.emitters, core=core)

    def to_composite(self, overrides=None, core=None, emit=True):
        from process_bigraph import Composite, allocate_core
        if core is None:
            core = allocate_core()
        for ext in self.core_extensions:
            ext(core)
        doc = self.to_document(overrides, core=core, emit=emit)
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
