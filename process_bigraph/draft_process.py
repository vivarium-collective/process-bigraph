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

from typing import Any, Dict

from process_bigraph.composite import Process

#: Contract fields, in the order ``describe()`` / ``_render_contract`` render them.
_CONTRACT_FIELDS = ("summary", "makes", "moves", "senses", "holds", "regulation", "status")

#: Default status stamped onto every draft's contract.
_DRAFT_STATUS = "draft - no update dynamics yet"


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
        cls.DRAFT_NAME = name
        cls.DRAFT_INPUTS = dict(inputs)
        cls.DRAFT_OUTPUTS = dict(outputs)
        cls.contract = {"status": _DRAFT_STATUS, **dict(contract)}
        cls.description = _render_contract(name, cls.contract)
        return cls

    return decorate
