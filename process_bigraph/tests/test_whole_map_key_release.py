"""Regression tests for process-bigraph #284.

A Step that writes a map WHOLE (output ``('P',)``) must release every Step that
reads a KEY of that map (trigger path ``('P', 'x')`` — a DESCENDANT of the
output) once it has run. ``determine_steps`` only walked each output's prefixes
(``explode_path(output)[1:]`` = the output and its ANCESTORS), so a descendant
trigger path kept the writer in its wait-set forever. The blocked readers then
fell through to the equal-priority cycle fallback, whose ``max(set, ...)`` tie
was decided by Python's hash seed — so an *acyclic* network ran in hash-seed
order, breaking the paper's "deterministic zero-time DAG evaluation"
(Supplement 1, 3.8.4; cycles were restored in #233).

Two fixes, both in ``determine_steps``:
  1. release a just-run step from every trigger path AT OR BELOW each of its
     outputs, not only the prefixes;
  2. break the cycle-fallback tie deterministically (priority, then
     ``repr``-sorted path), matching ``find_step_cycle``'s existing ``key=repr``.
"""
from process_bigraph import Composite, Process, Step, allocate_core
from process_bigraph.scheduling import determine_steps


# ---------------------------------------------------------------------------
# Fix 1 — descendant release (the reliable failing-first unit test)
# ---------------------------------------------------------------------------

def test_whole_map_writer_releases_a_descendant_key_reader():
    """After the whole-map writer runs it must be cleared from the DESCENDANT
    trigger path ``('P', 'x')``, not only the output ``('P',)`` and its
    prefixes. Before the fix ``fulfilled[('P', 'x')]`` still held the writer, so
    the key-reading Step stayed blocked."""
    steps = {
        ("writer",): {"input_paths": [("tick",)], "output_paths": [("P",)], "priority": 0.0},
    }
    remaining = {("writer",)}
    fulfilled = {
        ("tick",): set(),
        ("P",): {("writer",)},
        ("P", "x"): {("writer",)},  # second-pass producer edge onto the leaf
    }
    to_run, remaining, fulfilled = determine_steps(steps, remaining, fulfilled)

    assert to_run == [("writer",)]
    assert fulfilled[("P",)] == set()       # output + prefixes (already worked)
    assert fulfilled[("P", "x")] == set()   # <-- the fix; {('writer',)} before it


# ---------------------------------------------------------------------------
# Fix 2 — deterministic cycle-fallback tie-break
# ---------------------------------------------------------------------------

def test_cycle_fallback_tie_is_deterministic():
    """With no ready step and equal priorities, the fallback picks the same
    step regardless of set-iteration order — the ``repr``-smallest path, the
    convention ``find_step_cycle`` already sorts by."""
    steps = {
        ("a",): {"input_paths": [("x",)], "output_paths": [("y",)], "priority": 0.0},
        ("b",): {"input_paths": [("y",)], "output_paths": [("x",)], "priority": 0.0},
    }
    # a 2-cycle: a waits on b's output, b waits on a's output -> nothing ready
    fulfilled = {("x",): {("b",)}, ("y",): {("a",)}}
    chosen, _, _ = determine_steps(steps, set(steps), dict(fulfilled))
    assert chosen == [min(steps, key=repr)]


def test_cycle_fallback_still_prefers_higher_priority():
    """The tie-break only applies to EQUAL priorities: a higher-priority member
    of a cycle still wins outright."""
    steps = {
        ("a",): {"input_paths": [("x",)], "output_paths": [("y",)], "priority": 0.0},
        ("b",): {"input_paths": [("y",)], "output_paths": [("x",)], "priority": 5.0},
    }
    fulfilled = {("x",): {("b",)}, ("y",): {("a",)}}
    chosen, _, _ = determine_steps(steps, set(steps), dict(fulfilled))
    assert chosen == [("b",)]


# ---------------------------------------------------------------------------
# End-to-end regression (the issue's own repro)
# ---------------------------------------------------------------------------

class _Clock(Process):
    config_schema = {}

    def inputs(self):
        return {"tick": "float"}

    def outputs(self):
        return {"tick": "float"}

    def update(self, state, interval):
        return {"tick": interval}


class _Writer(Step):
    config_schema = {}

    def inputs(self):
        return {"tick": "float"}

    def outputs(self):
        return {"P": "map[overwrite[float]]"}

    def update(self, state):
        return {"P": {"x": state["tick"]}}


class _Copy(Step):
    config_schema = {}

    def inputs(self):
        return {"source": "float", "trigger": "float"}

    def outputs(self):
        return {"target": "overwrite[float]"}

    def update(self, state):
        return {"target": state["source"]}


def test_repro_z_tracks_tick_for_an_acyclic_writer_reader_late_chain():
    """writer (writes map ``P`` whole) -> reader (reads ``P/x``) -> late (reads
    ``y``) is acyclic, so ``y`` and ``z`` must equal the tick at every row — no
    longer lagging ``z`` by a tick on some hash seeds."""
    core = allocate_core()
    for name, link in {"Clock": _Clock, "Writer": _Writer, "Copy": _Copy}.items():
        core.register_link(name, link)
    state = {
        "tick": 0.0, "P": {"x": 0.0}, "y": 0.0, "z": 0.0,
        "clock": {"_type": "process", "address": "local:Clock", "interval": 1.0,
                  "inputs": {"tick": ["tick"]}, "outputs": {"tick": ["tick"]}},
        "writer": {"_type": "step", "address": "local:Writer",
                   "inputs": {"tick": ["tick"]}, "outputs": {"P": ["P"]}},
        "reader": {"_type": "step", "address": "local:Copy",
                   "inputs": {"source": ["P", "x"], "trigger": ["tick"]},
                   "outputs": {"target": ["y"]}},
        "late": {"_type": "step", "address": "local:Copy",
                 "inputs": {"source": ["y"], "trigger": ["y"]},
                 "outputs": {"target": ["z"]}},
    }
    sim = Composite({"state": state}, core=core)
    rows = []
    for _ in range(3):
        sim.run(1.0)
        rows.append((sim.state["tick"], sim.state["y"], sim.state["z"]))

    assert rows == [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)]
