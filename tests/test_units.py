"""Demonstrate the (already-built) ``_units`` fast-path: a wire between two
ports with compatible-but-different units is auto-converted by the engine.

This does NOT change the units engine (``bigraph_schema`` ``_compute_unit_scale``
/ ``project_ports_fast``'s ``scaled_leaf`` branch) — it exercises and documents
the existing feature, which until now no real process declared.
"""

from process_bigraph import allocate_core, Composite, process


def test_output_units_autoconvert_across_wire():
    # Producer emits mass in femtograms (fg) into a store declared in
    # picograms (pg). 1000 fg == 1 pg, so the engine scales by 1e-3 at the wire.
    @process(inputs={}, outputs={"mass": {"_type": "float", "_units": "fg"}})
    def emit_fg(state, interval, *, amount: float = 1000.0):
        return {"mass": amount}

    core = allocate_core()
    core.register_link("emit_fg", emit_fg)
    sim = Composite({"state": {
        "mass": {"_type": "float", "_units": "pg"},
        "emitter": {"_type": "process", "address": "local:emit_fg",
                    "config": {"amount": 1000.0},
                    "inputs": {}, "outputs": {"mass": ["mass"]}},
    }}, core=core)
    sim.run(1.0)

    # 1000 fg auto-converted to 1.0 pg
    assert sim.state["mass"] == 1.0


def test_no_units_passes_through_unscaled():
    # Control: with no _units declared, the raw value is written unscaled,
    # which is what makes the conversion in the test above observable.
    @process(inputs={}, outputs={"mass": "float"})
    def emit_plain(state, interval, *, amount: float = 1000.0):
        return {"mass": amount}

    core = allocate_core()
    core.register_link("emit_plain", emit_plain)
    sim = Composite({"state": {
        "mass": {"_type": "float"},
        "emitter": {"_type": "process", "address": "local:emit_plain",
                    "config": {"amount": 1000.0},
                    "inputs": {}, "outputs": {"mass": ["mass"]}},
    }}, core=core)
    sim.run(1.0)

    assert sim.state["mass"] == 1000.0
