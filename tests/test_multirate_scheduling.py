"""Pin the multi-rate update-timing contract (issue #234).

A process reads its inputs at the **start** of its interval and the resulting
delta is applied at the **end** — the "stash" model of Supplement 1, §3.8.6 —
rather than reading at the event time ``t*`` described in §3.8.2. The two rules
agree when all intervals are equal and diverge otherwise. These tests lock the
behavior so it cannot change silently.

Self-contained (own ``allocate_core``), like the other files in ``tests/``.
"""

from process_bigraph import allocate_core, Composite, process


# x' = x + 0.5 (y - x) on p1; y' = y + 0.25 (x - y) on p2 (each returns a delta)
@process(inputs={"x": "float", "y": "float"}, outputs={"x": "float"})
def _P1(state, interval):
    return {"x": 0.5 * (state["y"] - state["x"])}


@process(inputs={"x": "float", "y": "float"}, outputs={"y": "float"})
def _P2(state, interval):
    return {"y": 0.25 * (state["x"] - state["y"])}


def _build(i1, i2):
    core = allocate_core()
    core.register_link("P1", _P1)
    core.register_link("P2", _P2)
    doc = {"state": {
        "x": 1.0,
        "y": 0.0,
        "p1": {"_type": "process", "address": "local:P1", "config": {},
               "interval": i1,
               "inputs": {"x": ["x"], "y": ["y"]}, "outputs": {"x": ["x"]}},
        "p2": {"_type": "process", "address": "local:P2", "config": {},
               "interval": i2,
               "inputs": {"x": ["x"], "y": ["y"]}, "outputs": {"y": ["y"]}},
    }}
    return Composite(doc, core=core)


def test_unequal_intervals_read_at_start_of_interval():
    """p2 (interval 2) reads x,y at t=0 and applies at t=2.

    Read-at-start (§3.8.6): p2's first delta uses x=1, y=0, giving
    y = 0.25 at t=2 — not the y = 0.125 you would get by reading x at t=2
    (the §3.8.2 read-at-t* rule). Single ``run`` so the whole interval fits
    the window.
    """
    sim = _build(1.0, 2.0)
    sim.run(2.0)
    assert sim.state["x"] == 0.25
    assert sim.state["y"] == 0.25


def test_equal_intervals_are_chunking_invariant():
    """With equal intervals, read-at-start == read-at-t*, and the result does
    not depend on how ``run`` is chunked."""
    single = _build(1.0, 1.0)
    single.run(4.0)

    chunked = _build(1.0, 1.0)
    for _ in range(4):
        chunked.run(1.0)

    assert (single.state["x"], single.state["y"]) == \
           (chunked.state["x"], chunked.state["y"])
    # sanity: this case actually evolves (not a trivial fixed point)
    assert single.state["x"] != 1.0
