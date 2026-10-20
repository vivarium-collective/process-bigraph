"""Task 1 — DraftProcess deprecation + draft_to_contract bridge.

DraftProcess is being retired in favour of holes-in-place (a Site whose sort is
a ProcessContract). P1 keeps it working but deprecated, and adds a pure bridge
turning a draft's declared ports + free-form contract into a real
``ProcessContract`` (the P2 migration seam).
"""
import warnings

import pytest

from process_bigraph import allocate_core
from process_bigraph.draft_process import (
    DraftProcess,
    DraftProcessDeprecationWarning,
    draft_process,
    draft_to_contract,
)


def _make_draft():
    @draft_process(
        name="PTH secretion",
        inputs={"ca_sense": "float"},
        outputs={"pth_out": "float"},
        contract={
            "summary": "Chief cells secrete PTH",
            "makes": "PTH (de novo)",
            "senses": "parathyroid vessel Ca2+",
        },
    )
    class PTHSecretion(DraftProcess):
        pass

    return PTHSecretion


def _make_draft_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DraftProcessDeprecationWarning)
        return _make_draft()


def test_draft_process_warns_on_decoration_but_still_builds_inert():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = _make_draft()

    draft_warnings = [
        w for w in caught if isinstance(w.message, DraftProcessDeprecationWarning)
    ]
    assert len(draft_warnings) == 1

    # Still a working, inert draft (non-breaking): ports intact, DRAFT marker,
    # base no-op update.
    assert cls.DRAFT_INPUTS == {"ca_sense": "float"}
    assert cls.DRAFT_OUTPUTS == {"pth_out": "float"}
    core = allocate_core()
    instance = cls({}, core=core)
    assert "DRAFT —" in instance.describe()
    assert instance.update({}, 1.0) == {}


def test_draft_to_contract_maps_ports_and_prose():
    cls = _make_draft_silent()
    contract = draft_to_contract(cls)

    assert contract.face == {
        "inputs": {"ca_sense": "float"},
        "outputs": {"pth_out": "float"},
    }
    assert "Chief cells secrete PTH" in contract.summary
    # The free-form fields are preserved in readable prose, not dropped.
    blob = f"{contract.summary}\n{contract.description}"
    assert "PTH (de novo)" in blob
    assert "parathyroid vessel Ca2+" in blob


def test_dedicated_warning_is_filterable():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DraftProcessDeprecationWarning)
        with pytest.raises(DraftProcessDeprecationWarning):
            _make_draft()
