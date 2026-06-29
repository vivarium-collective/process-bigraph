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

from dataclasses import dataclass, field, asdict
from typing import Callable

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
            module = self.module if self.module else getattr(self.builder, '__module__', '')
            d["builder"] = f"{module}:{self.builder.__name__}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeSpec":
        d = dict(d)
        d.pop("core_extensions", None)
        return cls(**d)
