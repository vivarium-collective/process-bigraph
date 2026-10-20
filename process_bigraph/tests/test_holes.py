"""Holes-in-place: UnfilledHoleError + the contract-aware report builder.

A composite that still has an open required hole is rejected with a report that
names, per hole, its path, the contract it requires (face + conditions), and the
registered processes that would fill it — replacing the opaque ``KeyError:
'data'`` an unfilled address site used to raise.
"""
import pytest

from process_bigraph import allocate_core
from process_bigraph.composite import Composite, Process
from process_bigraph.templates import template_document
from process_bigraph.holes import (
    HoleReport,
    UnfilledHoleError,
    build_unfilled_hole_error,
)
from bigraph_schema.assembly import interfaces
from bigraph_schema.contract import ProcessContract


def _core():
    """A core with one named contract 'PTH' and a process that fills it."""
    core = allocate_core()
    core.register_contract(
        "PTH",
        ProcessContract(
            summary="PTH secretion",
            face={"inputs": {"ca_sense": "float"}, "outputs": {"pth_out": "float"}},
        ),
    )

    class PTHReal(Process):
        def inputs(self):
            return {"ca_sense": "float"}

        def outputs(self):
            return {"pth_out": "float"}

        def update(self, state, interval):
            return {"pth_out": 0.0}

    core.register_link("PTHReal", PTHReal)
    return core


def _pairs(core, document):
    """The (path, site) pairs the Composite groundness guards collect."""
    accessed = core.access(document)
    return [(tuple(path), site) for path, site in interfaces(accessed)[0]._places]


def test_is_subclass_of_valueerror():
    # Existing callers catch ValueError; the new error must stay caught.
    assert issubclass(UnfilledHoleError, ValueError)


def test_report_names_path_contract_and_candidates():
    core = _core()
    pairs = _pairs(core, {"model": {"_type": "site", "_sort": "PTH"}})

    err = build_unfilled_hole_error(core, pairs)

    assert isinstance(err, UnfilledHoleError)
    assert len(err.holes) == 1
    hole = err.holes[0]
    assert isinstance(hole, HoleReport)
    assert hole.path == ("model",)
    assert hole.contract is not None
    assert hole.contract.face["inputs"] == {"ca_sense": "float"}
    assert any(c.address == "PTHReal" and c.match == "full" for c in hole.candidates)

    text = str(err)
    assert "not ground" in text
    assert "model" in text
    assert "ca_sense" in text
    assert "PTHReal" in text


def test_unsorted_hole_reports_no_contract_and_does_not_crash():
    core = _core()
    pairs = _pairs(core, {"model": {"_type": "site"}})  # no _sort → no contract

    err = build_unfilled_hole_error(core, pairs)

    hole = err.holes[0]
    assert hole.contract is None
    assert hole.candidates == []
    assert "no contract declared" in str(err)


def test_find_candidates_failure_never_masks_primary_error(monkeypatch):
    core = _core()
    import process_bigraph.holes as holes_mod

    def boom(*args, **kwargs):
        raise RuntimeError("candidate finder exploded")

    monkeypatch.setattr(holes_mod, "find_candidates", boom)
    pairs = _pairs(core, {"model": {"_type": "site", "_sort": "PTH"}})

    err = build_unfilled_hole_error(core, pairs)  # must not raise

    assert err.holes[0].path == ("model",)
    assert err.holes[0].candidates == []
    assert "model" in str(err)
    assert "not ground" in str(err)


# ── Task 3: the Composite groundness guards raise the report ──────────

def _unfilled_composite_doc():
    return {"state": {"study": {"model": {"_type": "site", "_sort": "PTH"}}}}


def test_building_composite_with_unfilled_hole_raises_report():
    core = _core()
    with pytest.raises(UnfilledHoleError) as excinfo:
        Composite(_unfilled_composite_doc(), core=core)
    text = str(excinfo.value)
    assert "study/model" in text
    assert "ca_sense" in text
    assert "PTHReal" in text


def test_except_valueerror_still_catches():
    core = _core()
    try:
        Composite(_unfilled_composite_doc(), core=core)
        assert False, "expected an unfilled-hole raise"
    except ValueError as error:
        assert isinstance(error, UnfilledHoleError)


def test_ground_composite_with_unwired_ports_still_builds():
    core = _core()
    doc = {
        "state": {
            "ca": 3.0,
            "pth": 0.0,
            "pth_proc": {
                "_type": "process",
                "address": "local:PTHReal",
                "config": {},
                "inputs": {"ca_sense": ["ca"]},
                "outputs": {"pth_out": ["pth"]},
            },
        }
    }
    sim = Composite(doc, core=core)  # must not raise — unwired ports are not holes
    assert sim is not None


def test_template_document_unfilled_raises_report():
    core = _core()
    template = core.access({"study": {"model": {"_type": "site", "_sort": "PTH"}}})
    with pytest.raises(UnfilledHoleError) as excinfo:
        template_document(core, template, {})
    assert "study/model" in str(excinfo.value)
