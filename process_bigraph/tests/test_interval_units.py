"""Timestep units: a process may declare interval as a pint string; it is
normalized to canonical seconds with the declared unit preserved as metadata."""

from process_bigraph import Composite, allocate_core


def _doc(interval):
    return {'state': {
        'level': 5.0,
        'grow': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': 0.1},
            'inputs': {'level': ['level']},
            'outputs': {'level': ['level']},
            'interval': interval}}}


def _composite(interval):
    from process_bigraph.processes.examples import IncreaseProcess
    core = allocate_core()
    core.register_link('IncreaseProcess', IncreaseProcess)
    return Composite(_doc(interval), core=core)


def test_string_interval_realizes_to_canonical_seconds():
    c = _composite('1 min')
    grow = c.state['grow']
    assert grow['interval'] == 60.0
    assert isinstance(grow['interval'], float)
    assert grow['interval_units'] == 'minute'


def test_bare_number_interval_unchanged():
    c = _composite(60)
    grow = c.state['grow']
    assert grow['interval'] == 60.0
    assert grow.get('interval_units') is None


def test_interval_is_optional_defaults_to_one_second():
    """`interval` is optional: a process that declares none keeps the schema
    default (1.0 seconds) and carries no declared unit."""
    from process_bigraph.processes.examples import IncreaseProcess
    core = allocate_core()
    core.register_link('IncreaseProcess', IncreaseProcess)
    doc = {'state': {
        'level': 5.0,
        'grow': {'_type': 'process', 'address': 'local:IncreaseProcess',
                 'config': {'rate': 0.1}, 'inputs': {'level': ['level']},
                 'outputs': {'level': ['level']}}}}  # no 'interval' key
    c = Composite(doc, core=core)
    grow = c.state['grow']
    assert grow['interval'] == 1.0
    assert grow.get('interval_units') is None


def test_string_and_number_intervals_advance_identically():
    by_number = _composite(60)
    by_string = _composite('1 min')
    by_number.run(120)
    by_string.run(120)
    assert by_number.state['global_time'] == by_string.state['global_time'] == 120.0
    assert by_number.state['level'] == by_string.state['level']


def test_multiple_processes_different_units_schedule_in_seconds():
    """Smallest interval wins, measured in canonical seconds — a 30 s process
    and a 1 min process advance global_time by the 30 s step."""
    from process_bigraph.processes.examples import IncreaseProcess
    core = allocate_core()
    core.register_link('IncreaseProcess', IncreaseProcess)
    doc = {'state': {
        'level': 5.0,
        'fast': {'_type': 'process', 'address': 'local:IncreaseProcess',
                 'config': {'rate': 0.1}, 'inputs': {'level': ['level']},
                 'outputs': {'level': ['level']}, 'interval': '30 s'},
        'slow': {'_type': 'process', 'address': 'local:IncreaseProcess',
                 'config': {'rate': 0.1}, 'inputs': {'level': ['level']},
                 'outputs': {'level': ['level']}, 'interval': '1 min'}}}
    c = Composite(doc, core=core)
    assert c.state['fast']['interval'] == 30.0
    assert c.state['slow']['interval'] == 60.0
    c.run(60)
    assert c.state['global_time'] == 60.0


def test_non_time_unit_interval_raises():
    import pytest
    with pytest.raises(ValueError, match='meter'):
        _composite('1 meter')


def test_serialize_schema_round_trips_declared_unit():
    """serialize_schema() runs the ProcessLink render path; a unit-declared
    process re-emits the pint string '1 min' (not 60 / '60 s' / '60.0 minute')."""
    c = _composite('1 min')
    grow = c.serialize_schema()['grow']
    assert grow['interval'] == '1 min'


def test_serialize_schema_keeps_bare_number():
    """A bare-number interval still serializes as the number (no regression)."""
    c = _composite(60)
    grow = c.serialize_schema()['grow']
    assert grow['interval'] == 60.0
