"""Regression test for the #259 pure-reader ordering bug.

A2/A3 drop an inout accumulator (a step that read-modify-writes a store) from
the ``before`` producer set of that store's leaves — correct for accumulator
PEERS (they do not produce what each other consumes), but it left a PURE READER
of the store (reads the settled value, never writes it) unordered, so the reader
could run mid-accumulation and read a stale value (sms-ecoli ``contract_out`` read
``dry_mass`` before ``division``/listeners settled it).

The fix records a per-step ``_extra_predecessors`` edge accumulator -> pure
reader, ordering the reader after the accumulators WITHOUT ordering the
accumulators among themselves (which would rebuild the cycle A2/A3 removed), and
only where it stays acyclic (a reader that also feeds an accumulator is genuine
zero-time feedback, left to priority resolution).
"""
from process_bigraph.scheduling import build_step_network


class _Step:
    def __init__(self, inputs_schema, outputs_schema):
        self._in, self._out = inputs_schema, outputs_schema

    def interface(self):
        return {"inputs": self._in, "outputs": self._out}

    def inputs(self):
        return self._in

    def outputs(self):
        return self._out


def _acc(store):
    # read-modify-write the same store (an inout accumulator)
    return {"instance": _Step({"s": "float"}, {"s": "float"}),
            "inputs": {"s": [store]}, "outputs": {"s": [store]}, "priority": 0.0}


def test_pure_reader_ordered_after_inout_accumulators_but_peers_are_not():
    steps = {
        ("acc",): _acc("S"),
        ("peer",): _acc("S"),
        # pure reader: reads S, writes a DIFFERENT store R (never writes S)
        ("reader",): {"instance": _Step({"s": "float"}, {"r": "float"}),
                      "inputs": {"s": ["S"]}, "outputs": {"r": ["R"]}, "priority": 0.0},
    }
    ancestors, nodes = build_step_network(steps)

    # Both writers of S are inout — A2 suppresses them from S's producer set.
    assert not nodes[("S",)]["before"]

    # The pure reader must run after BOTH accumulators.
    assert ancestors[("reader",)]["_extra_predecessors"] == frozenset({("acc",), ("peer",)})

    # The accumulators are NOT ordered among themselves (no peer cycle).
    assert not ancestors[("acc",)]["_extra_predecessors"]
    assert not ancestors[("peer",)]["_extra_predecessors"]


def test_feedback_reader_is_left_to_priority_not_hard_ordered():
    # A "reader" that also feeds the accumulator (writes a store the accumulator
    # reads) is genuine zero-time feedback: a hard accumulator->reader edge would
    # close a cycle, so it must be skipped (priority resolves the loop instead).
    steps = {
        # accumulator reads+writes S, and also reads F (the feedback store)
        ("acc",): {"instance": _Step({"s": "float", "f": "float"}, {"s": "float"}),
                   "inputs": {"s": ["S"], "f": ["F"]}, "outputs": {"s": ["S"]},
                   "priority": 0.0},
        # reader reads S (making it a candidate pure reader) but writes F, which
        # the accumulator reads -> reader -> acc edge already exists.
        ("reader",): {"instance": _Step({"s": "float"}, {"f": "float"}),
                      "inputs": {"s": ["S"]}, "outputs": {"f": ["F"]}, "priority": 0.0},
    }
    ancestors, _ = build_step_network(steps)
    # No hard acc->reader edge: reader already reaches acc (reader writes F, acc
    # reads F), so adding it would close a cycle.
    assert not ancestors[("reader",)]["_extra_predecessors"]
