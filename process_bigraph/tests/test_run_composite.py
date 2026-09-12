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


def _boom_document():
    return {
        'state': {
            'boom': {'_type': 'process',
                     'address': 'local:!process_bigraph.tests.test_events._Boom',
                     'config': {'at': 1.0}, 'interval': 1.0,
                     'inputs': {'level': ['level'], 'global_time': ['global_time']},
                     'outputs': {'level': ['level']}},
            'level': 0.0,
        }
    }


def test_run_composite_writes_failure_json_and_exits_nonzero(tmp_path):
    """A failing run leaves a machine-readable ``failure.json`` next to its
    output (engine context included) and still propagates the exception, so
    the CLI's exit code is unchanged."""
    import subprocess
    import sys
    from process_bigraph import events
    from process_bigraph.run_composite import run_composite
    doc = tmp_path / 'doc.json'
    doc.write_text(json.dumps(_boom_document()))
    events.set_emitter(None)
    import os
    import pytest
    before = os.environ.get('PBG_TRACEPARENT')
    with pytest.raises(ZeroDivisionError):
        run_composite(str(doc), steps=3.0, state_out_path=str(tmp_path / 'out' / 'state.json'))
    assert os.environ.get('PBG_TRACEPARENT') == before    # the library never mutates env
    record = json.loads((tmp_path / 'out' / 'failure.json').read_text())
    assert record['exc_type'] == 'ZeroDivisionError'
    assert record['pbg_context']['path'] == 'boom'
    assert record['pbg_context']['global_time'] == 1.0
    assert 'ZeroDivisionError' in record['traceback_tail']

    # The CLI: non-zero exit, events on stdout by default, failure.json where asked.
    proc = subprocess.run(
        [sys.executable, '-m', 'process_bigraph.run_composite', '--document', str(doc),
         '--steps', '3', '--failure-out', str(tmp_path / 'cli' / 'failure.json')],
        capture_output=True, text=True, env={**__import__('os').environ, 'PBG_EVENT_HEARTBEAT_S': '3600'})
    assert proc.returncode != 0
    stdout_events = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith('{')]
    names = [e['event'] for e in stdout_events]
    assert 'task.start' in names and 'process.exception' in names and 'task.end' in names
    assert [e for e in stdout_events if e['event'] == 'task.end'][0]['payload']['status'] == 'error'
    assert all(e['component'] == 'process_bigraph' for e in stdout_events)
    assert (tmp_path / 'cli' / 'failure.json').exists()


def test_run_composite_summary_out_on_success(tmp_path):
    from process_bigraph import events
    from process_bigraph.run_composite import run_composite
    doc = tmp_path / 'doc.json'
    doc.write_text(json.dumps(_incr_document()))
    events.set_emitter(None)
    run_composite(str(doc), steps=3.0, summary_out=str(tmp_path / 'summary.json'))
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['status'] == 'ok' and summary['global_time'] == 3.0
    assert summary['total'] >= summary['process_time'] >= 0.0


def test_run_step_writes_failure_json_and_reraises(tmp_path):
    from process_bigraph import events
    from process_bigraph.run_step import run_step
    import pytest
    events.set_emitter(None)
    with pytest.raises(ZeroDivisionError):
        run_step('process_bigraph.tests.test_run_composite._BoomStep', config={},
                 state={}, update_json_path=str(tmp_path / 'u' / 'update.json'))
    record = json.loads((tmp_path / 'u' / 'failure.json').read_text())
    assert record['exc_type'] == 'ZeroDivisionError' and record['step_class'].endswith('_BoomStep')


from process_bigraph.composite import Step as _Step


class _BoomStep(_Step):
    def inputs(self):
        return {}

    def outputs(self):
        return {'x': 'float'}

    def update(self, state):
        raise ZeroDivisionError('step boom')
