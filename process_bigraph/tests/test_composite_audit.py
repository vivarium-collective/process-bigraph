from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Process
from process_bigraph.composite_audit import _member_wirings, MemberWiring


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
