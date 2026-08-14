import json
from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Process


class _Incr(Process):
    """Minimal temporal process: pushes 'level' up every unit of time."""
    config_schema = {'rate': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        return {'level': self.config['rate'] * interval}


def _incr_document():
    core = allocate_core()
    state = {
        'level': 1.0,
        'incr': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.test_run_composite._Incr',
            'config': {'rate': 2.0},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': 1.0,
        },
    }
    composite = Composite({'state': state}, core=core)
    return {'schema': composite.serialize_schema(),
            'state': composite.serialize_state()}


def test_run_composite_advances_and_writes_state(tmp_path):
    doc_path = tmp_path / 'doc.json'
    doc_path.write_text(json.dumps(_incr_document()))
    out_path = tmp_path / 'final.json'

    from process_bigraph.run_composite import run_composite
    run_composite(str(doc_path), steps=5.0, state_out_path=str(out_path))

    final = json.loads(out_path.read_text())
    # Ran for 5 time units at rate 2.0 → level grew above its start (1.0).
    assert float(final['state']['level']) > 1.0


def test_run_composite_initial_state_overlay(tmp_path):
    doc_path = tmp_path / 'doc.json'
    doc_path.write_text(json.dumps(_incr_document()))
    out_path = tmp_path / 'final.json'

    from process_bigraph.run_composite import run_composite
    run_composite(str(doc_path), steps=0.0,
                  initial_state={'level': 42.0},
                  state_out_path=str(out_path))

    final = json.loads(out_path.read_text())
    assert float(final['state']['level']) == 42.0


def test_run_composite_state_out_is_best_effort_on_serialization_failure(tmp_path, monkeypatch):
    # A WCM composite's string-typed LabeledArray can't serialize under the
    # pinned bigraph_schema==1.6.0 ('str' object has no attribute 'fields'),
    # even though composite.run() itself succeeded and the real scientific
    # output (emitter/parquet) was already written. --state-out must be
    # best-effort: write a marker document and NOT crash the subprocess.
    doc_path = tmp_path / 'doc.json'
    doc_path.write_text(json.dumps(_incr_document()))
    out_path = tmp_path / 'final.json'

    from process_bigraph.composite import Composite as CompositeClass

    def _boom(self):
        raise AttributeError("'str' object has no attribute 'fields'")

    monkeypatch.setattr(CompositeClass, 'serialize_state', _boom)

    from process_bigraph.run_composite import run_composite
    # Must not raise: composite.run() succeeded, so the subprocess exits 0.
    run_composite(str(doc_path), steps=5.0, state_out_path=str(out_path))

    assert out_path.exists()
    marker = json.loads(out_path.read_text())
    assert 'note' in marker
    assert 'error' in marker
    assert "'str' object has no attribute 'fields'" in marker['error']


def test_run_composite_state_out_roundtrips_as_initial_state(tmp_path):
    # The composite-node renderer chains one task's --state-out into the
    # next task's --initial-state. --state-out writes a FULL {schema, state}
    # document, not a bare state dict. The overlay must unwrap it, so the
    # handoff lands at state['level'] (not nested under state['state']).
    doc_path = tmp_path / 'doc.json'
    doc_path.write_text(json.dumps(_incr_document()))
    first_out = tmp_path / 'first.json'

    from process_bigraph.run_composite import run_composite
    run_composite(str(doc_path), steps=5.0, state_out_path=str(first_out))

    handoff_document = json.loads(first_out.read_text())
    assert 'schema' in handoff_document and 'state' in handoff_document

    second_out = tmp_path / 'second.json'
    run_composite(str(doc_path), steps=0.0,
                  initial_state=handoff_document,
                  state_out_path=str(second_out))

    final = json.loads(second_out.read_text())
    expected_level = float(handoff_document['state']['level'])
    assert float(final['state']['level']) == expected_level
