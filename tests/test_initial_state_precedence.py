"""initial_state() outranks realize-filled type defaults, not the document (#200).

Composite.initialize realizes the document first (filling every store with its
type default), then combines each edge's initial_state() in. The type defaults
were applied as the *update*, so for overwrite[float] and boolean output ports
the default (0.0, False) outranked the process's initial_state(). A boolean flag
that must start True could not be seeded this way.

A type default that realize filled in must not outrank initial_state(); a value
the document states explicitly still takes precedence over both.
"""
from process_bigraph import Process, Composite, allocate_core

_UNSET = object()


def _seeded(port_type, value, doc_value=_UNSET):
    class P(Process):
        def inputs(self):
            return {}

        def outputs(self):
            return {"x": port_type}

        def initial_state(self):
            return {"x": value}

        def update(self, state, interval):
            return {}

    core = allocate_core()
    core.register_link("P", P)
    state = {"p": {"_type": "process", "address": "local:P",
                   "outputs": {"x": ["s", "x"]}}}
    if doc_value is not _UNSET:
        state["s"] = {"x": doc_value}          # document sets the store explicitly
    sim = Composite({"state": state}, core=core)
    return sim.state["s"]["x"]


def test_initial_state_seeds_store_across_port_types():  # #200
    assert _seeded("float", 42.0) == 42.0
    assert _seeded("overwrite[float]", 42.0) == 42.0     # was 0.0 before the fix
    assert _seeded("maybe[float]", 42.0) == 42.0
    assert _seeded("boolean", True) == True              # was False before the fix


def test_document_explicit_value_still_outranks_initial_state():
    assert _seeded("float", 42.0, doc_value=7.0) == 7.0
    assert _seeded("overwrite[float]", 42.0, doc_value=7.0) == 7.0
    assert _seeded("boolean", True, doc_value=False) == False
