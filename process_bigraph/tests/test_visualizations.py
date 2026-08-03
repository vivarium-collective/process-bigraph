"""Tests for the 5 default Visualization classes (v2: update(state))."""
from process_bigraph.visualizations import TimeSeriesPlot


def _trajectory_state():
    """One run's trajectory of ``observable`` and ``time``."""
    return {
        'observable': [1.0, 2.0, 4.0, 8.0],
        'time': [0.0, 1.0, 2.0, 3.0],
    }


def _multi_run_state():
    """Two runs' trajectories — orchestrator passes list-of-lists for sweeps."""
    return {
        'observable': [[1.0, 2.0, 4.0], [3.0, 6.0, 12.0]],
        'time': [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        '_run_labels': ['rate=1.0', 'rate=3.0'],
    }


def test_time_series_plot_single_run():
    inst = object.__new__(TimeSeriesPlot)
    inst.config = {'title': 'Test'}
    html = inst.update(_trajectory_state())
    assert 'html' in html
    assert 'Plotly.newPlot' in html['html']
    assert 'Test' in html['html']


def test_time_series_plot_multi_run():
    inst = object.__new__(TimeSeriesPlot)
    inst.config = {'title': ''}
    html = inst.update(_multi_run_state())
    assert 'Plotly.newPlot' in html['html']
    assert 'rate=1.0' in html['html']
    assert 'rate=3.0' in html['html']


from process_bigraph.visualizations import ParamVsObservable


def test_param_vs_observable():
    inst = object.__new__(ParamVsObservable)
    inst.config = {'title': 'Sweep'}
    state = {
        'sweep_param_values': [0.1, 0.5, 1.0],
        'reduced_observable':  [3.0, 7.5, 15.0],
    }
    out = inst.update(state)
    assert 'html' in out
    assert 'Plotly.newPlot' in out['html']
    assert '15' in out['html']


from process_bigraph.visualizations import Distribution


def test_distribution_histogram():
    inst = object.__new__(Distribution)
    inst.config = {'title': 'Hist'}
    state = {'samples': [10.0, 10.3, 10.6, 10.9, 11.2]}
    out = inst.update(state)
    assert 'Plotly.newPlot' in out['html']
    assert 'histogram' in out['html'].lower()


from process_bigraph.visualizations import PhaseSpace


def test_phase_space():
    inst = object.__new__(PhaseSpace)
    inst.config = {'title': 'XY'}
    state = {'x': [0.0, 1.0, 2.0, 3.0], 'y': [0.0, 1.0, 4.0, 9.0]}
    out = inst.update(state)
    assert 'Plotly.newPlot' in out['html']


from process_bigraph.visualizations import Heatmap


def test_heatmap():
    inst = object.__new__(Heatmap)
    inst.config = {'title': 'Grid'}
    state = {
        'x_params': [1.0, 2.0, 3.0],
        'y_params': [10.0, 20.0],
        'z_values': [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]],
    }
    out = inst.update(state)
    assert 'Plotly.newPlot' in out['html']
    assert 'heatmap' in out['html'].lower()
