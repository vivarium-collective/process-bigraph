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
    _render_candidates,
    _render_requires,
)
from bigraph_schema.assembly import interfaces
from bigraph_schema.contract import ProcessContract, narrow_condition
from bigraph_schema.matching import CandidateMatch


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


# ── Task 4: public API surface + end-to-end lifecycle ─────────────────

def test_public_api_importable():
    import process_bigraph as pbg

    for name in (
        "UnfilledHoleError",
        "open_sites",
        "required_open_sites",
        "is_ground_document",
        "fill_template",
        "template_document",
        "find_candidates",
        "draft_to_contract",
        "DraftProcessDeprecationWarning",
    ):
        assert hasattr(pbg, name), f"process_bigraph.{name} is not exported"


def test_hole_lifecycle_design_inspect_fill_build():
    import process_bigraph as pbg

    core = _core()  # registers contract 'PTH' + process 'PTHReal'

    # 1. design: a composite whose model slot is a hole sorted by the contract.
    template = core.access(
        {"study": {"ca": 3.0, "pth": 0.0, "model": {"_type": "site", "_sort": "PTH"}}}
    )

    # 2. inspect: the hole is open and blocks a build.
    assert pbg.open_sites(template) == [("study", "model")]
    assert pbg.required_open_sites(template) == [("study", "model")]
    assert pbg.is_ground_document(template) is False

    # 3. see what fits: an admissible registered process.
    (_path, site), = interfaces(template)[0]._places
    candidates = pbg.find_candidates(core, site)
    assert any(c.address == "PTHReal" and c.match == "full" for c in candidates)

    # 4. fill with the admissible process → ground.
    filler = core.access(
        {
            "_type": "process",
            "address": "local:PTHReal",
            "config": {},
            "interval": 1.0,
            "inputs": {"ca_sense": ["ca"]},
            "outputs": {"pth_out": ["pth"]},
        }
    )
    document = pbg.template_document(core, template, {"study/model": filler})
    assert pbg.is_ground_document(core.access(document))

    # 5. build + run.
    sim = pbg.Composite({"state": document}, core=core)
    sim.run(1.0)
    assert sim is not None


# ── Fix pass: report-quality (I1 face-literal type; M1 near-miss; M2 conditions) ──

def test_face_literal_hole_renders_clean_type():
    # A hole sorted by a face literal (a first-class authoring path) must report
    # its ports as clean types, not resolved-type reprs like "Float(...)".
    core = allocate_core()
    model_face = {
        "_type": "link",
        "_inputs": {"ca_sense": "float"},
        "_outputs": {"pth_out": "float"},
    }
    doc = {"state": {"study": {"model": {"_type": "site", "_sort": model_face}}}}
    with pytest.raises(UnfilledHoleError) as excinfo:
        Composite(doc, core=core)
    text = str(excinfo.value)
    assert "ca_sense: float" in text
    assert "Float(" not in text


def test_near_miss_candidate_renders_failing_condition():
    # A partial candidate must render its failing condition, not a raw dict.
    candidate = CandidateMatch(
        address="PartialProc",
        match="partial",
        over_provides=[],
        fails=[{"condition": "bound.ca_sense", "detail": "below _min"}],
    )
    joined = "\n".join(_render_candidates([candidate]))
    assert "PartialProc" in joined
    assert "bound.ca_sense" in joined
    assert "{'condition'" not in joined  # no raw dict dump


def test_contract_condition_renders_in_report():
    # A narrowed contract's condition appears in the 'requires' block.
    contract = ProcessContract(
        face={"inputs": {"ca_sense": "float"}, "outputs": {"pth_out": "float"}}
    )
    contract = narrow_condition(contract, "pre", "inputs.ca_sense >= 0", name="nonneg")
    joined = "\n".join(_render_requires(contract, None))
    assert "nonneg" in joined
    assert "inputs.ca_sense" in joined
