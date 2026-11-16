"""Holes-in-place: a legible, contract-aware rejection for an unfilled composite.

A *draft* in this framework is a process-slot left as an open **hole** — a
``Site`` whose ``_sort`` is a :class:`~bigraph_schema.contract.ProcessContract`.
``Composite`` refuses to build while a required hole is open; instead of the
opaque ``KeyError: 'data'`` an unfilled address site used to raise, it raises an
:class:`UnfilledHoleError` that names, per hole, its path and the contract it
requires (typed face + conditions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from bigraph_schema.assembly import contract_of


@dataclass
class HoleReport:
    """One unfilled hole: where it is and what contract it requires."""

    path: Tuple[str, ...]
    contract: Optional[Any] = None          # ProcessContract | None


class UnfilledHoleError(ValueError):
    """Raised at ``Composite`` construction when a required hole is still open.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    catching it. Carries the structured ``.holes`` (a list of :class:`HoleReport`)
    for machine rendering (e.g. the workbench) and a scannable ``str``.
    """

    def __init__(self, holes: Sequence[HoleReport], core: Any = None):
        self.holes: List[HoleReport] = list(holes)
        self._core = core  # used only to render resolved face-literal types by name
        super().__init__(self._render())

    def _render(self) -> str:
        named = ", ".join(repr("/".join(h.path)) for h in self.holes)
        lines = [
            "composite document is not ground — required site(s) left "
            f"unfilled: {named}. An open site is a hole where a process must "
            "go; fill it (bigraph_schema `fill_sites` / "
            "`process_bigraph.templates.fill_template`) before constructing "
            "the composite.",
        ]
        for hole in self.holes:
            lines.append("")
            lines.append(f"  • {'/'.join(hole.path)}")
            lines.extend(_render_requires(hole.contract, self._core))
        return "\n".join(lines)


def build_unfilled_hole_error(
    core: Any, unfilled_sites: Sequence[Tuple[Sequence[str], Any]]
) -> UnfilledHoleError:
    """Build an :class:`UnfilledHoleError` from ``(path, site)`` pairs.

    For each hole, resolve the contract it requires (``contract_of``), wrapped so
    a failure never masks the primary "not ground" report; an unsorted hole simply
    reports no contract.
    """
    reports = []
    for path, site in unfilled_sites:
        contract = _safe_contract_of(core, site)
        reports.append(HoleReport(tuple(path), contract))
    return UnfilledHoleError(reports, core=core)


def _safe_contract_of(core, site):
    try:
        return contract_of(core, site)
    except Exception:  # noqa: BLE001 - a hole we cannot read is reported as unsorted
        return None


def _render_requires(contract, core=None) -> List[str]:
    if contract is None:
        return ["      requires: no contract declared (unsorted hole)"]
    lines = ["      requires:"]
    face = getattr(contract, "face", None) or {}
    for direction in ("inputs", "outputs"):
        ports = face.get(direction) or {}
        if ports:
            rendered = ", ".join(
                f"{port}: {_fmt_port(schema, core)}" for port, schema in ports.items())
            lines.append(f"        {direction}: {rendered}")
    conditions = contract.conditions() if hasattr(contract, "conditions") else []
    for condition in conditions:
        lines.append(
            f"        condition [{condition.get('kind', '?')}] "
            f"{condition.get('name', '?')}: {condition.get('expr', '')}")
    return lines


def _fmt_port(schema, core=None) -> str:
    """A port schema as ``type (units, [min, max])``; a bare type as itself.

    A ``dict`` schema (the registered-contract path) renders its ``_type`` with
    any units/bounds. A plain string renders as-is. Anything else is a *resolved*
    type object (the face-literal path, where ``contract_of`` resolves the face
    to live type objects such as ``Float(...)``): render it to its type name via
    ``core.render`` when a core is available, else fall back to its class name so
    the report never shows a raw ``Float(_default=None, ...)`` repr.
    """
    if isinstance(schema, dict):
        base = schema.get("_type", "any")
        extras = []
        if schema.get("_units"):
            extras.append(f"units={schema['_units']}")
        lo, hi = schema.get("_min"), schema.get("_max")
        if lo is not None or hi is not None:
            extras.append(f"[{lo}, {hi}]")
        return f"{base} ({', '.join(extras)})" if extras else base
    if isinstance(schema, str):
        return schema
    if core is not None:
        try:
            return str(core.render(schema))
        except Exception:  # noqa: BLE001 - fall back to the class name, never a repr
            pass
    return type(schema).__name__.lower()
