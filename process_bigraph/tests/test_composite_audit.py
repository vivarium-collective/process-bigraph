from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Process
from process_bigraph.composite_audit import _member_wirings, MemberWiring, _store_graph, _resolve_wire


class _Src(Process):   # produces 'level'
    def inputs(self): return {}
    def outputs(self): return {'level': 'float'}
    def update(self, state, interval): return {'level': 1.0}


class _Sink(Process):  # consumes 'level'
    def inputs(self): return {'level': 'float'}
    def outputs(self): return {}
    def update(self, state, interval): return {}


def _two_process_composite():
    core = allocate_core()
    core.register_link('_Src', _Src)
    core.register_link('_Sink', _Sink)
    return Composite({'state': {
        'src': {'_type': 'process', 'address': 'local:_Src', 'config': {},
                'inputs': {}, 'outputs': {'level': ['level']}},
        'snk': {'_type': 'process', 'address': 'local:_Sink', 'config': {},
                'inputs': {'level': ['level']}, 'outputs': {}},
        'level': 0.0}}, core=core), core


def test_member_wirings_enumerates_members():
    composite, _ = _two_process_composite()
    wirings = {w.address.split(':')[-1].split('.')[-1]: w for w in _member_wirings(composite)}
    assert set(wirings) >= {'_Src', '_Sink'}
    src = next(w for w in _member_wirings(composite) if w.output_wires)
    assert src.output_wires == {'level': ['level']}
    snk = next(w for w in _member_wirings(composite) if w.input_wires)
    assert snk.input_wires == {'level': ['level']}
    assert isinstance(src, MemberWiring)


def test_store_graph_pairs_writer_and_reader():
    composite, _ = _two_process_composite()
    graph, unanalyzable = _store_graph(_member_wirings(composite))
    assert ('level',) in graph
    assert any(port == 'level' for _, port, _ in graph[('level',)]['writers'])
    assert any(port == 'level' for _, port, _ in graph[('level',)]['readers'])
    assert unanalyzable == []


def test_nested_wire_is_unanalyzable():
    assert _resolve_wire((), ['level']) == ('level',)
    assert _resolve_wire(('a',), ['x']) == ('a', 'x')
    assert _resolve_wire((), {'sub': ['level']}) is None    # nested dict wire
    assert _resolve_wire((), ['..', 'level']) is None        # relative wire


from process_bigraph.composite_audit import _check_store_compat, _check_boundary_override

def _graph(writer_schema, reader_schema):
    return {('s',): {'writers': [('W', 'out', writer_schema)],
                     'readers': [('R', 'in', reader_schema)]}}

def test_store_bounds_mismatch_flagged():
    # writer produces [0,100]; reader requires [0,5] → 100 doesn't fit → warning
    findings = _check_store_compat(_graph(
        {'_type': 'float', '_min': 0, '_max': 100}, {'_type': 'float', '_min': 0, '_max': 5}))
    assert any(f.severity == 'warning' and 's' in f.where for f in findings)

def test_store_bounds_compatible_clean():
    # writer [0,3] ⊆ reader [0,5] → ok
    findings = _check_store_compat(_graph(
        {'_type': 'float', '_min': 0, '_max': 3}, {'_type': 'float', '_min': 0, '_max': 5}))
    assert findings == []

def test_store_units_mismatch_flagged():
    findings = _check_store_compat(_graph({'_type': 'float', '_units': 'mg'},
                                          {'_type': 'float', '_units': 'second'}))
    assert any(f.severity == 'warning' for f in findings)

def test_unbounded_members_are_quiet():
    findings = _check_store_compat(_graph('float', 'float'))
    assert findings == []

def test_store_with_only_writer_is_not_flagged():
    findings = _check_store_compat({('s',): {'writers': [('W', 'out', {'_min': 0, '_max': 100})], 'readers': []}})
    assert findings == []


class _FakeComposite:
    """Minimal stand-in exposing the two things _check_boundary_override reads:
    the declared interface-output override and the bridge-output wiring."""
    def __init__(self, declared_outputs, bridge_outputs):
        self.config = {'interface': {'outputs': declared_outputs}}
        self._bridge_outputs = bridge_outputs
    def read_bridge_outputs(self):
        return self._bridge_outputs

def test_boundary_override_underclaims_flagged():
    # boundary declares output 'y' bounded [0,5], but it is wired to store ('s',)
    # whose writer produces [0,100] → boundary under-claims → warning
    graph = {('s',): {'writers': [('W', 'out', {'_min': 0, '_max': 100})], 'readers': []}}
    comp = _FakeComposite(declared_outputs={'y': {'_type': 'float', '_min': 0, '_max': 5}},
                          bridge_outputs={'y': ['s']})
    findings = _check_boundary_override(comp, graph)
    assert any(f.severity == 'warning' and 'y' in f.where for f in findings)

def test_no_override_no_finding():
    comp = _FakeComposite(declared_outputs={}, bridge_outputs={})
    assert _check_boundary_override(comp, {}) == []


from process_bigraph.composite_audit import audit_composite

def test_audit_requires_built_composite_and_is_clean_when_consistent():
    composite, _ = _two_process_composite()   # float↔float, no bounds → clean
    report = audit_composite(composite)
    assert report.ok is True
    assert not any(f.severity == 'warning' for f in report.findings)

def test_end_to_end_bounds_mismatch_surfaced():
    """A producer whose declared output range exceeds a consumer's declared
    input range on a shared store is flagged — the check build-time type-merge
    does not perform."""
    core = allocate_core()

    class _WideSrc(Process):
        def inputs(self): return {}
        def outputs(self): return {'level': {'_type': 'float', '_min': 0, '_max': 100}}
        def update(self, state, interval): return {'level': 1.0}

    class _NarrowSink(Process):
        def inputs(self): return {'level': {'_type': 'float', '_min': 0, '_max': 5}}
        def outputs(self): return {}
        def update(self, state, interval): return {}

    core.register_link('_WideSrc', _WideSrc)
    core.register_link('_NarrowSink', _NarrowSink)
    composite = Composite({'state': {
        'src': {'_type': 'process', 'address': 'local:_WideSrc', 'config': {},
                'inputs': {}, 'outputs': {'level': ['level']}},
        'snk': {'_type': 'process', 'address': 'local:_NarrowSink', 'config': {},
                'inputs': {'level': ['level']}, 'outputs': {}},
        'level': 0.0}}, core=core)
    report = audit_composite(composite)
    assert report.ok is True
    assert any(f.code == 'store_bounds_mismatch' for f in report.findings)
