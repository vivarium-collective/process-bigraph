"""DraftProcess — a placeholder / draft process.

A *draft* process declares a **contract** — its input and output ports (names +
bigraph types) plus a human-readable description of the transformation it is
*intended* to perform — but has **no update dynamics yet**. It lets the model
topology (which process connects which stores) be designed and reviewed
*before* anyone commits to a mechanism.

Define a concrete, registrable draft with the :func:`draft_process` decorator::

    from process_bigraph import DraftProcess, draft_process

    @draft_process(
        name="PTH secretion",
        inputs={"ca_sense": "float"},
        outputs={"pth_out": "float"},
        contract={
            "summary": "Chief cells synthesize PTH de novo and secrete it ...",
            "makes": "PTH (de novo) -> parathyroid vessel",
            "senses": "parathyroid vessel Ca2+",
            "regulation": "inversely Ca2+-dependent ...",
        },
    )
    class PTHSecretion(DraftProcess):
        pass

Because it is a ``Process`` subclass defined at module scope, discovery that
walks a package and registers every Process/Step (e.g. a workspace
``build_core()``) auto-registers it. It therefore shows up in registry-driven
tooling *by name*, with its ports, and — because ``describe()`` renders the
whole contract prefixed ``DRAFT —`` — its contract text and a DRAFT marker.

There is intentionally no ``update`` override: a draft inherits the base
``Process.update`` no-op (returns ``{}``). It builds and renders but is inert if
stepped — it never fabricates dynamics it does not have.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

from bigraph_schema.contract import ProcessContract

from process_bigraph.composite import Process

#: Contract fields, in the order ``describe()`` / ``_render_contract`` render them.
_CONTRACT_FIELDS = ("summary", "makes", "moves", "senses", "holds", "regulation", "status")

#: Free-form prose fields folded into a bridged contract's ``description``.
_PROSE_FIELDS = ("makes", "moves", "senses", "holds", "regulation")

#: Default status stamped onto every draft's contract.
_DRAFT_STATUS = "draft - no update dynamics yet"


class DraftProcessDeprecationWarning(DeprecationWarning):
    """``DraftProcess`` / ``@draft_process`` is deprecated in favour of
    holes-in-place: model a not-yet-implemented process as an open *hole* whose
    sort is a :class:`~bigraph_schema.contract.ProcessContract`. A dedicated
    category so a consumer can silence exactly this warning
    (``warnings.filterwarnings("ignore",
    category=process_bigraph.DraftProcessDeprecationWarning)``) without muting
    unrelated deprecations."""


class DraftProcess(Process):
    """Interface-only placeholder process: ports + contract, no dynamics.

    Concrete drafts are ``DraftProcess`` subclasses carrying class-level
    ``DRAFT_INPUTS`` / ``DRAFT_OUTPUTS`` / ``contract`` (normally injected by the
    :func:`draft_process` decorator). Registry-style introspection reads the
    class uninitialized — it calls ``inputs()``/``outputs()`` with the *class*
    as ``self`` and ``describe()`` on an ``__new__``'d instance — so every
    method here reads class attributes and never requires a configured instance.
    """

    # Filled by @draft_process (or overridden on a subclass).
    DRAFT_NAME: str = ""
    DRAFT_INPUTS: Dict[str, Any] = {}
    DRAFT_OUTPUTS: Dict[str, Any] = {}
    #: Class-level contract dict (also the ``.contract`` convention used by
    #: other processes to advertise their intended interface).
    contract: Dict[str, Any] = {}

    # A draft has no config beyond a nominal (unused) interval.
    config_schema = {"interval": {"_type": "float", "_default": 1.0}}

    def inputs(self) -> Dict[str, Any]:
        return dict(self.DRAFT_INPUTS or {})

    def outputs(self) -> Dict[str, Any]:
        return dict(self.DRAFT_OUTPUTS or {})

    def describe(self) -> str:
        """A ``DRAFT —`` prefixed rendering of the whole contract.

        Registry tooling surfaces this as the process's description, so the whole
        contract (and the DRAFT marker) is visible wherever the description is.
        """
        contract = self.contract if isinstance(self.contract, dict) else {}
        label = self.DRAFT_NAME or type(self).__name__
        return _render_contract(label, contract)

    # No ``update`` override on purpose — draft. Inherits the base no-op.


def _render_contract(label: str, contract: Dict[str, Any]) -> str:
    """One-line ``DRAFT — summary · key: value · …`` rendering of a contract.

    A single line (fields joined by "·") so it stays readable in a registry
    card whether or not the surrounding markup preserves newlines.
    """
    parts = ["DRAFT — " + (contract.get("summary") or label)]
    for key in _CONTRACT_FIELDS:
        if key == "summary":
            continue
        value = contract.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "  ·  ".join(parts)


def draft_to_contract(cls) -> ProcessContract:
    """Bridge a ``DraftProcess`` subclass to a real :class:`ProcessContract`.

    The typed ``face`` comes from ``DRAFT_INPUTS`` / ``DRAFT_OUTPUTS``; the
    free-form ``summary`` becomes the contract ``summary`` and the remaining
    prose fields (``makes`` / ``moves`` / ``senses`` / ``holds`` /
    ``regulation``) are folded into ``description`` so nothing is dropped. The
    draft ``status`` marker is not carried. Pure: no registration, no side
    effects — this is the P2 migration seam (draft → hole) and a way for tooling
    to read a draft's interface as a contract.
    """
    inputs = dict(getattr(cls, "DRAFT_INPUTS", {}) or {})
    outputs = dict(getattr(cls, "DRAFT_OUTPUTS", {}) or {})
    declared = getattr(cls, "contract", {})
    declared = declared if isinstance(declared, dict) else {}

    summary = declared.get("summary") or getattr(cls, "DRAFT_NAME", "") or cls.__name__
    prose = [
        f"{key}: {declared[key]}"
        for key in _PROSE_FIELDS
        if declared.get(key)]

    return ProcessContract(
        summary=summary,
        description="\n".join(prose),
        face={"inputs": inputs, "outputs": outputs})


def draft_process(
    *,
    name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    contract: Dict[str, Any],
):
    """Class decorator turning a bare ``DraftProcess`` subclass into a
    registrable draft with fixed ports + contract.

    Args:
        name: Human-readable draft name (shown in ``describe()``).
        inputs / outputs: ``{port_name: bigraph_type}`` port schemas.
        contract: Free-form contract dict (``summary`` + optional ``makes`` /
            ``moves`` / ``senses`` / ``holds`` / ``regulation``). A ``status``
            of "draft" is stamped in automatically.
    """

    def decorate(cls):
        if not (isinstance(cls, type) and issubclass(cls, DraftProcess)):
            raise TypeError("@draft_process can only decorate DraftProcess subclasses")
        warnings.warn(
            "DraftProcess / @draft_process is deprecated; model a "
            "not-yet-implemented process as an open hole whose sort is a "
            "ProcessContract (see docs/concepts/designing-with-holes.md). "
            "draft_to_contract(cls) bridges an existing draft's interface.",
            DraftProcessDeprecationWarning,
            stacklevel=2,
        )
        cls.DRAFT_NAME = name
        cls.DRAFT_INPUTS = dict(inputs)
        cls.DRAFT_OUTPUTS = dict(outputs)
        cls.contract = {"status": _DRAFT_STATUS, **dict(contract)}
        cls.description = _render_contract(name, cls.contract)
        return cls

    return decorate
