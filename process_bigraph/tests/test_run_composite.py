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
