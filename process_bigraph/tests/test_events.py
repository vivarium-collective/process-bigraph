"""Engine-level observability (``process_bigraph.events``).

The invariant every test here protects, stated once: enabling instrumentation
adds output and never changes a simulation's result or exception type.
"""
import json
import sys
import types

import numpy as np
import pytest

from process_bigraph import Composite, allocate_core, events
from process_bigraph.composite import Process


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _increaser(rate=0.1):
    return {
        '_type': 'process',
        'address': 'local:IncreaseProcess',
        'config': {'rate': rate},
        'inputs': {'level': ['level']},
        'outputs': {'level': ['level']},
    }


def _two_increasers():
    return {'a': _increaser(1.0), 'b': _increaser(0.2), 'level': 1.0}


class _Boom(Process):
    """Raises a distinctive exception once ``global_time`` passes ``at``."""
    config_schema = {'at': 'float'}

    def inputs(self):
        return {'level': 'float', 'global_time': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        if state['global_time'] >= self.config['at']:
            raise ZeroDivisionError('boom at %s' % state['global_time'])
        return {'level': 1.0}


class _RaisingSink(events.EventSink):
    def __init__(self):
        self.calls = 0

    def emit(self, event):
        self.calls += 1
        raise IOError('disk on fire')


class _ListSink(events.EventSink):
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


@pytest.fixture(autouse=True)
def _silent_emitter():
    """Every test starts from the library default (no sinks) and leaves it so."""
    events.set_emitter(None)
    yield
    events.set_emitter(None)


def _lines(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith('{')]


def _boom_state(at=2.0):
    return {
        'boom': {'_type': 'process', 'address': 'local:!process_bigraph.tests.test_events._Boom',
                 'config': {'at': at}, 'interval': 1.0,
                 'inputs': {'level': ['level'], 'global_time': ['global_time']},
                 'outputs': {'level': ['level']}},
        'level': 0.0,
    }


# ---------------------------------------------------------------------------
# defaults and sinks
# ---------------------------------------------------------------------------

def test_events_off_by_default_emits_nothing(capsys, monkeypatch):
    monkeypatch.delenv('PBG_EVENT_SINKS', raising=False)
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(3.0)
    assert not events.get_emitter().enabled
    assert _lines(capsys) == []


def test_stdout_sink_emits_run_start_and_run_end_with_baggage(capsys):
    events.configure('stdout', env={
        'PBG_TRACE_BAGGAGE': json.dumps({'campaign': '946', 'cell': 0, 'replicate': 3}),
        'PBG_EVENT_TAGS': json.dumps({'backend': 'test'}),
        'PBG_EVENT_HEARTBEAT_S': '3600'})
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(3.0)
    recs = _lines(capsys)
    names = [r['event'] for r in recs]
    assert names[0] == 'run_start' and names[-1] == 'run_end'
    assert 'tick' not in names[1:-1] or names.count('tick') == 1   # first tick only
    for r in recs:
        assert r['v'] == events.SCHEMA_VERSION
        assert r['layer'] == 'engine'
        assert r['baggage'] == {'campaign': 946, 'cell': 0, 'replicate': 3}   # opaque, ints coerced
        assert r['tags'] == {'backend': 'test'}
        assert set(r) == {'v', 'ts', 'seq', 'layer', 'event', 'level', 'trace_id', 'span_id',
                          'parent_span_id', 'global_time', 'wall_time', 'source',
                          'baggage', 'tags', 'payload'}
        assert len(r['trace_id']) == 32
    end = recs[-1]['payload']
    assert end['status'] == 'ok' and recs[-1]['global_time'] == 3.0
    assert end['total'] >= end['process_time'] >= 0.0


def test_heartbeat_is_wall_clock_throttled(capsys):
    events.configure('stdout', env={'PBG_EVENT_HEARTBEAT_S': '0'})
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(3.0)
    every = [r for r in _lines(capsys) if r['event'] == 'tick']
    assert len(every) >= 3
    assert [r['global_time'] for r in every][:3] == [0.0, 1.0, 2.0]
    assert every[-1]['payload']['ticks_total'] == len(every)

    events.configure('stdout', env={'PBG_EVENT_HEARTBEAT_S': '3600'})
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(3.0)
    rare = [r for r in _lines(capsys) if r['event'] == 'tick']
    assert len(rare) == 1           # the first tick always reports, then silence


def test_exception_event_names_process_and_reraises_original_type(capsys):
    events.configure('stdout', env={'PBG_EVENT_HEARTBEAT_S': '3600'})
    sim = Composite({'state': _boom_state(at=2.0)}, core=allocate_core())
    with pytest.raises(ZeroDivisionError) as raised:
        sim.run(5.0)
    exc = raised.value
    assert exc.pbg_context['path'] == 'boom'
    assert exc.pbg_context['global_time'] == 2.0
    assert exc.pbg_context['cls'] == '_Boom'
    recs = _lines(capsys)
    ev = [r for r in recs if r['event'] == 'exception']
    assert len(ev) == 1
    p = ev[0]['payload']
    assert p['exc_type'] == 'ZeroDivisionError'
    assert p['path'] == 'boom' and p['is_step'] is False and p['interval'] == 1.0
    assert p['state_summary']['level'] == 2.0
    end = [r for r in recs if r['event'] == 'run_end'][0]
    assert end['payload']['status'] == 'error'
    assert end['level'] == 'error'
    assert 'ZeroDivisionError' in end['payload']['error']


def test_a_raising_sink_never_breaks_the_sim():
    bad = _RaisingSink()
    good = _ListSink()
    events.set_emitter(events.EventEmitter([bad, good], heartbeat_s=0))
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(3.0)                                  # no exception
    assert bad.calls == 1                         # disabled after the first failure
    errors = [e for e in good.events if e['event'] == 'sink_error']
    assert len(errors) == 1 and errors[0]['payload']['sink'] == '_RaisingSink'
    assert [e['event'] for e in good.events][-1] == 'run_end'


def test_sink_resolution_registry_entry_point_and_module_attr(monkeypatch, tmp_path):
    seen = {}

    def factory(spec):
        seen['spec'] = spec
        return _ListSink()
    events.register_sink_factory('demo', factory)
    assert isinstance(events.resolve_sink('demo://bucket/prefix'), _ListSink)
    assert seen['spec'] == 'demo://bucket/prefix'

    # entry point group
    class _EP:
        name = 'ep'

        @staticmethod
        def load():
            return lambda spec: _ListSink()

    class _EPs:
        def select(self, group):
            return [_EP()] if group == events.ENTRY_POINT_GROUP else []
    import importlib.metadata as md
    monkeypatch.setattr(md, 'entry_points', lambda: _EPs())
    assert isinstance(events.resolve_sink('ep:anything'), _ListSink)

    # module:attr
    mod = types.ModuleType('pbg_test_sink_mod')
    mod.make = lambda spec: _ListSink()
    monkeypatch.setitem(sys.modules, 'pbg_test_sink_mod', mod)
    assert isinstance(events.resolve_sink('pbg_test_sink_mod:make'), _ListSink)

    # builtins + unknown
    assert isinstance(events.resolve_sink('stdout'), events.StdoutSink)
    fs = events.resolve_sink(f'file:{tmp_path}/e.jsonl')
    assert isinstance(fs, events.FileSink)
    fs.close()
    with pytest.warns(UserWarning, match='dropping sink'):
        assert events.resolve_sink('nope:x') is None
    assert events.resolve_sink('none') is None


def test_file_sink_and_deprecated_aliases(tmp_path, capsys):
    trace = tmp_path / 'trace.jsonl'
    em = events.configure(env={'PROCESS_BIGRAPH_TRACE_FILE': str(trace),
                               'PROCESS_BIGRAPH_PROFILE_PROCESSES': '1',
                               'PBG_EVENT_HEARTBEAT_S': '3600'})
    assert em.detail_invoke and em.detail_timing
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(2.0)
    em.flush()
    recs = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(r['event'] == 'invoke' and r['payload']['path'] in ('a', 'b') for r in recs)
    assert [r for r in recs if r['event'] == 'run_end'][0]['payload']['top5']
    assert _lines(capsys) == []          # nothing leaked to stdout


# ---------------------------------------------------------------------------
# trace / span context
# ---------------------------------------------------------------------------

def test_traceparent_parsed_and_stamped_on_every_event(capsys):
    trace = 'a' * 32
    parent = 'b' * 16
    events.configure('stdout', env={'PBG_TRACEPARENT': f'00-{trace}-{parent}-01',
                                    'PBG_EVENT_HEARTBEAT_S': '3600'})
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(1.0)
    recs = _lines(capsys)
    assert recs and all(r['trace_id'] == trace for r in recs)
    assert all(r['parent_span_id'] == parent for r in recs)   # no run span without detail=spans
    assert events.parse_traceparent(f'{trace}-{parent}') == (trace, parent)
    assert events.parse_traceparent('garbage') is None


def test_spans_nest_and_span_end_carries_start_ts(capsys):
    events.configure('stdout', env={'PBG_EVENT_DETAIL': 'spans', 'PBG_EVENT_HEARTBEAT_S': '3600'})
    em = events.get_emitter()
    with em.span('task', name='t'):
        outer = em.current_context()
        sim = Composite({'state': _two_increasers()}, core=allocate_core())
        sim.run(1.0)
    recs = _lines(capsys)
    starts = [r for r in recs if r['event'] == 'span_start']
    ends = [r for r in recs if r['event'] == 'span_end']
    assert [s['payload']['name'] for s in starts] == ['task', 'run']
    run_start = starts[1]
    assert run_start['parent_span_id'] == outer.span_id
    run_end = [e for e in ends if e['payload']['name'] == 'run'][0]
    assert run_end['payload']['start_ts'] == run_start['payload']['start_ts']
    assert run_end['payload']['duration_s'] >= 0 and run_end['payload']['status'] == 'ok'
    # events inside the run carry the run span
    assert [r for r in recs if r['event'] == 'run_end'][0]['span_id'] == run_start['span_id']
    assert em.current_context() is None            # restored after the task span


def test_baggage_and_tags_accept_w3c_key_value_form_and_stay_opaque():
    """Dispatchers render env through ``docker --env`` (no quotes/whitespace),
    so the primary form is W3C baggage; JSON stays accepted for laptops. The
    engine interprets no key: whatever a caller puts there comes out as-is,
    with int-looking values coerced."""
    em = events.configure(env={
        'PBG_TRACE_BAGGAGE': 'campaign=946,label=run3%20pilot,cell=0,replicate=3,stage=x',
        'PBG_EVENT_TAGS': 'backend=nextflow,job=abc-123;prop=ignored,attempt=2',
        'PBG_TRACEPARENT': '00-' + '0123456789abcdef' * 2 + '-' + 'fedcba9876543210' + '-01'})
    assert em.baggage == {'campaign': 946, 'label': 'run3 pilot', 'cell': 0, 'replicate': 3, 'stage': 'x'}
    assert em.tags == {'backend': 'nextflow', 'job': 'abc-123', 'attempt': 2}
    assert em.trace_id == '0123456789abcdef' * 2 and em.root_span_id == 'fedcba9876543210'
    assert events.parse_baggage('{"a": "1", "b": "two"}') == {'a': 1, 'b': 'two'}
    assert events.parse_baggage('') == {} and events.parse_baggage('novalue,=x') == {}
    assert events.parse_baggage('{not json') == {}                      # never raises
    # bind() adds/overwrites/removes, any key
    em.bind(stage='y', extra=7, cell=None)
    assert em.baggage == {'campaign': 946, 'label': 'run3 pilot', 'replicate': 3, 'stage': 'y', 'extra': 7}


def test_local_run_mints_trace_id_when_no_traceparent():
    em = events.configure(env={})
    assert len(em.trace_id) == 32 and em.root_span_id is None
    tp = em.current_traceparent()
    assert tp.startswith('00-' + em.trace_id + '-')


def test_span_context_survives_the_parallel_layer():
    sink = _ListSink()
    events.set_emitter(events.EventEmitter([sink], heartbeat_s=3600, detail=('spans', 'invoke')))
    state = _two_increasers()
    sim = Composite({'state': state, 'parallel_processes': True}, core=allocate_core())
    assert sim._parallel_processes
    sim.run(2.0)
    run_span = [e for e in sink.events if e['event'] == 'span_start'][0]['span_id']
    invokes = [e for e in sink.events if e['event'] == 'invoke']
    assert len(invokes) >= 2
    assert all(e['span_id'] == run_span for e in invokes)


def test_tree_reconstructable_from_jsonl(capsys):
    events.configure('stdout', env={'PBG_EVENT_DETAIL': 'spans', 'PBG_EVENT_HEARTBEAT_S': '3600'})
    em = events.get_emitter()
    with em.span('task'):
        for _ in range(2):
            Composite({'state': _two_increasers()}, core=allocate_core()).run(1.0)
    recs = _lines(capsys)

    # A 20-line reference reconstructor: spans from span_start/span_end, events
    # attached to their span_id, children by parent_span_id.
    spans, children = {}, {}
    for r in recs:
        if r['event'] == 'span_start':
            spans[r['span_id']] = {'name': r['payload']['name'], 'parent': r['parent_span_id'],
                                   'events': [], 'end': None}
            children.setdefault(r['parent_span_id'], []).append(r['span_id'])
    for r in recs:
        if r['event'] == 'span_end':
            spans[r['span_id']]['end'] = r['payload']
        elif r['event'] != 'span_start' and r['span_id'] in spans:
            spans[r['span_id']]['events'].append(r['event'])
    roots = [sid for sid, s in spans.items() if s['parent'] not in spans]
    assert len(roots) == 1 and spans[roots[0]]['name'] == 'task'
    kids = children[roots[0]]
    assert [spans[k]['name'] for k in kids] == ['run', 'run']
    for k in kids:
        assert spans[k]['events'][0] == 'run_start' and spans[k]['events'][-1] == 'run_end'
        assert spans[k]['end']['status'] == 'ok'


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def test_summarize_state_flags_nan_and_negatives():
    s = events.summarize_state({
        'bulk': np.array([1.0, -2.0, np.nan]),
        'counts': np.array([3, 4], dtype=int),
        'nested': {'x': {'y': 1}},
        'text': 'ok',
    })
    assert s['bulk']['nan_count'] == 1 and s['bulk']['neg_count'] == 1
    assert s['bulk']['min'] == -2.0 and s['bulk']['max'] == 1.0
    assert s['counts']['neg_count'] == 0 and s['counts']['nan_count'] == 0
    assert s['nested'] == {'x': {'y': 1}}
    assert s['text'] == 'ok'
    # never raises on odd input
    assert events.summarize_state(object()).startswith('<')


def test_failure_record_carries_engine_context():
    events.configure(env={})
    sim = Composite({'state': _boom_state(at=1.0)}, core=allocate_core())
    with pytest.raises(ZeroDivisionError) as raised:
        sim.run(3.0)
    rec = events.failure_record(raised.value, task='t')
    assert rec['exc_type'] == 'ZeroDivisionError'
    assert rec['pbg_context']['path'] == 'boom' and rec['task'] == 't'
    assert 'ZeroDivisionError' in rec['traceback_tail']


# ---------------------------------------------------------------------------
# the invariant
# ---------------------------------------------------------------------------

def test_instrumentation_does_not_change_results(tmp_path):
    events.set_emitter(None)
    off = Composite({'state': _two_increasers()}, core=allocate_core())
    off.run(10.0)
    off_state = json.dumps(off.serialize_state(), sort_keys=True, default=str)

    events.configure(f'stdout,file:{tmp_path}/e.jsonl',
                     env={'PBG_EVENT_DETAIL': 'timing,invoke,spans', 'PBG_EVENT_HEARTBEAT_S': '0'})
    on = Composite({'state': _two_increasers()}, core=allocate_core())
    on.run(10.0)
    on_state = json.dumps(on.serialize_state(), sort_keys=True, default=str)
    events.get_emitter().close()
    assert on_state == off_state
    assert on.state['level'] == off.state['level']
    assert (tmp_path / 'e.jsonl').read_text().count('"invoke"') >= 20


# ---------------------------------------------------------------------------
# Ray runtime (optional dependency)
# ---------------------------------------------------------------------------

def test_ray_batch_update_error_names_the_proc_id():
    pytest.importorskip('ray')
    from process_bigraph.protocols import ray as ray_protocol
    actor_cls = ray_protocol._batch_actor_class()
    plain = getattr(getattr(actor_cls, '__ray_metadata__', None), 'modified_class', None) or actor_cls
    actor = plain.__new__(plain)

    class _Bad:
        def update(self, inputs, interval):
            raise ValueError('cell exploded')
    actor.composites = {7: _Bad()}
    with pytest.raises(RuntimeError, match=r'proc_id=7 class=_Bad .*ValueError: cell exploded'):
        actor.batch_update([(7, {})], 1.0)


# ---------------------------------------------------------------------------
# contract examples: the engine's own toy composites, events on
# ---------------------------------------------------------------------------

def _loud(capsys_env=None, **extra_env):
    env = {'PBG_EVENT_DETAIL': 'timing,invoke,spans', 'PBG_EVENT_HEARTBEAT_S': '0'}
    env.update(extra_env)
    return events.configure('stdout', env=env)


def _grow_divide_composite(core):
    from process_bigraph.processes.growth_division import grow_divide_agent
    grow_divide = grow_divide_agent({'grow': {'rate': 0.03}}, {}, ['environment', '0'])
    return Composite({
        'state': {'environment': {'0': {'mass': 1.0, 'grow_divide': grow_divide}}},
        'bridge': {'inputs': {'environment': ['environment']}}},
        core=core)


def test_contract_two_increasers_stream(capsys):
    """run_start -> tick... -> run_end, invoke per process per tick, one run span."""
    _loud()
    sim = Composite({'state': _two_increasers()}, core=allocate_core())
    sim.run(3.0)
    names = [r['event'] for r in _lines(capsys)]
    assert names[:3] == ['span_start', 'run_start', 'tick']
    assert names[-2:] == ['run_end', 'span_end']
    assert names.count('tick') == 3 and names.count('invoke') == 6


def test_contract_grow_divide_division_is_a_structural_change(capsys):
    """A real division (grow_divide_agent, tests.py::test_grow_divide) shows up as
    ``structural_change`` between ticks, and the on/off states are identical."""
    core = allocate_core()
    events.set_emitter(None)
    quiet = _grow_divide_composite(core)
    quiet.update({'environment': {'0': {'mass': 1.1}}}, 50.0)
    quiet_keys = sorted(quiet.state['environment'])

    _loud(PBG_EVENT_HEARTBEAT_S='0')
    loud = _grow_divide_composite(allocate_core())
    loud.update({'environment': {'0': {'mass': 1.1}}}, 50.0)
    recs = _lines(capsys)
    names = [r['event'] for r in recs]
    assert names[0] == 'span_start' and names[1] == 'run_start'
    assert 'structural_change' in names
    first_div = names.index('structural_change')
    assert 'tick' in names[:first_div]                  # ticks before the division
    assert names[-2:] == ['run_end', 'span_end']
    div = recs[first_div]
    assert div['global_time'] is not None
    assert any('environment' in p for p in div['payload']['sample_paths'])
    end = [r for r in recs if r['event'] == 'run_end'][-1]['payload']
    assert end['status'] == 'ok' and end['top5']

    assert '0_0_0_0_1' in loud.state['environment']
    assert sorted(loud.state['environment']) == quiet_keys       # invariance under events
    assert loud.state['environment']['0_0_0_0_1']['mass'] == \
        quiet.state['environment']['0_0_0_0_1']['mass']


def test_contract_gillespie_composite_stream(capsys):
    """tests.py::test_gillespie_composite with events on: steps and processes
    both appear as invokes; the emitter step is a Step (interval -1)."""
    _loud(PBG_EVENT_HEARTBEAT_S='3600')
    gillespie = Composite({
        'bridge': {'inputs': {'DNA': ['DNA'], 'mRNA': ['mRNA']},
                   'outputs': {'time': ['global_time'], 'DNA': ['DNA'], 'mRNA': ['mRNA']}},
        'state': {
            'interval': {'_type': 'step',
                         'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieInterval',
                         'config': {'ktsc': '6e0'},
                         'inputs': {'DNA': ['DNA'], 'mRNA': ['mRNA']},
                         'outputs': {'interval': ['event', 'interval']}},
            'event': {'_type': 'process',
                      'address': 'local:!process_bigraph.experiments.minimal_gillespie.GillespieEvent',
                      'config': {'ktsc': 6e0},
                      'inputs': {'DNA': ['DNA'], 'mRNA': ['mRNA']},
                      'outputs': {'mRNA': ['mRNA']}, 'interval': '3.0'},
            'emitter': {'_type': 'step', 'address': 'local:!process_bigraph.emitter.RAMEmitter',
                        'config': {'emit': {'time': 'float', 'mRNA': 'map[float]', 'interval': 'interval'}},
                        'inputs': {'time': ['global_time'], 'mRNA': ['mRNA'],
                                   'interval': ['event', 'interval']}}}},
        core=allocate_core())
    gillespie.update({'DNA': {'A gene': 11.0, 'B gene': 5.0},
                      'mRNA': {'A mRNA': 33.0, 'B mRNA': 2.0}}, 100.0)
    recs = _lines(capsys)
    invokes = [r['payload'] for r in recs if r['event'] == 'invoke']
    assert {i['path'] for i in invokes} >= {'event', 'interval', 'emitter'}
    assert all(i['interval'] == -1.0 for i in invokes if i['path'] in ('interval', 'emitter'))
    assert [r['event'] for r in recs][-2:] == ['run_end', 'span_end']


def test_contract_injected_raising_process(capsys):
    """An injected process that raises mid-run: exactly one exception event with
    the process path and a state summary; the original type propagates."""
    _loud(PBG_EVENT_HEARTBEAT_S='3600')
    state = _two_increasers()
    state.update(_boom_state(at=5.0))
    sim = Composite({'state': state}, core=allocate_core())
    with pytest.raises(ZeroDivisionError):
        sim.run(10.0)
    recs = _lines(capsys)
    ex = [r for r in recs if r['event'] == 'exception']
    assert len(ex) == 1
    assert ex[0]['payload']['path'] == 'boom' and ex[0]['global_time'] == 5.0
    assert set(ex[0]['payload']['state_summary']) == {'level', 'global_time'}
    assert [r['event'] for r in recs][-2:] == ['run_end', 'span_end']
    assert recs[-2]['payload']['status'] == 'error'
    assert recs[-1]['payload']['status'] == 'error'
