"""Tests for process_bigraph.visualization.render_results (moved from viva-superpowers).

Mirrors the shape of ``process_bigraph.emitter.gather_emitter_results``:
returns a path-keyed dict whose values are the per-viz ``{'html': str}`` dicts.

Covers both code paths:

- legacy vizes (subclass overrides ``update``) — end-of-run mode reads the
  last value the runtime wrote to the html port; replay mode calls
  ``instance.update(results)``.
- new-style vizes (subclass implements ``accumulate`` + ``render``) —
  end-of-run mode calls ``instance.render()``; replay mode accumulates then
  renders.
"""
from __future__ import annotations

from process_bigraph import Composite, allocate_core

from process_bigraph.visualization import (
    Visualization,
    as_visualization,
    render_results,
)


# --- legacy viz fixtures (as_visualization → overrides update) -------------
@as_visualization(inputs={'k': 'string'}, name='_RR_EchoViz')
def update__rr_echo_viz(state):
    return {'html': '<x>' + state.get('k', '') + '</x>'}


@as_visualization(inputs={'k': 'string'}, name='_RR_LabelViz')
def update__rr_label_viz(state):
    return {'html': '<label>' + state.get('k', '') + '</label>'}


# --- new-style viz fixture (accumulate + render) ---------------------------
class _RR_CountViz(Visualization):
    """Counts ticks via accumulate(); renders the final count once."""

    def inputs(self):
        return {'k': 'string'}

    def accumulate(self, state):
        self._n = getattr(self, '_n', 0) + 1
        self._last = state.get('k', '')

    def render(self):
        return f'<count n="{self._n}">{self._last}</count>'


_RR_CountViz.__pb_kind__ = 'visualization'
_RR_CountViz.__pb_aliases__ = ['_RR_CountViz']


def _make_core():
    core = allocate_core()
    core.register_link('_RR_EchoViz', update__rr_echo_viz)
    core.register_link('_RR_LabelViz', update__rr_label_viz)
    core.register_link('_RR_CountViz', _RR_CountViz)
    return core


def _state_with_echo():
    return {
        'k_store': 'streamed',
        'viz1': {
            '_type': 'step',
            'address': 'local:_RR_EchoViz',
            'config': {},
            'inputs': {'k': ['k_store']},
            'outputs': {'html': ['viz_html_store']},
        },
        'viz_html_store': '',
    }


def _state_with_counter():
    return {
        'k_store': 'tick',
        'viz_count': {
            '_type': 'step',
            'address': 'local:_RR_CountViz',
            'config': {},  # default render_mode='end'
            'inputs': {'k': ['k_store']},
            'outputs': {'html': ['count_html']},
        },
        'count_html': '',
    }


# ---------------------------------------------------------------------------
# replay mode


def test_render_results_replay_mode_legacy():
    """Replay mode on a legacy viz calls ``update(results)`` directly."""
    composite = Composite({'state': _state_with_echo()}, core=_make_core())
    out = render_results(composite, results={'k': 'replay'})
    assert ('viz1',) in out
    assert 'replay' in out[('viz1',)]['html']
    # Replay does not depend on the wired store being populated.
    assert composite.state['viz_html_store'] == ''


def test_render_results_replay_mode_new_style():
    """Replay mode on a new-style viz accumulates the provided state then
    renders. The wired store stays untouched."""
    composite = Composite({'state': _state_with_counter()}, core=_make_core())
    out = render_results(composite, results={'k': 'replayed'})
    assert ('viz_count',) in out
    html = out[('viz_count',)]['html']
    assert 'replayed' in html and 'n="1"' in html


# ---------------------------------------------------------------------------
# end-of-run mode


def test_render_results_finds_nothing_when_no_viz():
    """A composite with no Visualization instances returns an empty dict."""
    core = _make_core()
    state = {'k_store': 'hello'}
    composite = Composite({'state': state}, core=core)
    out = render_results(composite)
    assert out == {}


def test_render_results_end_mode_legacy_reads_html_port():
    """For legacy vizes, end-mode returns whatever the runtime wrote to the
    wired html store."""
    core = _make_core()
    state = {
        'k_store': 'streamed',
        'viz_a': {
            '_type': 'step',
            'address': 'local:_RR_EchoViz',
            'config': {},
            'inputs': {'k': ['k_store']},
            'outputs': {'html': ['viz_a_html']},
        },
        'viz_a_html': '',
        'viz_b': {
            '_type': 'step',
            'address': 'local:_RR_LabelViz',
            'config': {},
            'inputs': {'k': ['k_store']},
            'outputs': {'html': ['viz_b_html']},
        },
        'viz_b_html': '',
    }
    composite = Composite({'state': state}, core=core)
    composite.run(1)
    out = render_results(composite)
    assert set(out.keys()) == {('viz_a',), ('viz_b',)}
    assert '<x>streamed</x>' == out[('viz_a',)]['html']
    assert '<label>streamed</label>' == out[('viz_b',)]['html']


def test_render_results_end_mode_new_style_calls_render():
    """For new-style vizes, end-mode invokes ``instance.render()`` once.

    Critically: during the run the html port stays empty (because the
    baseclass returns ``{'html': ''}`` in 'end' mode), so render_results is
    the only way to materialize the final HTML — and it does so without
    re-running the simulation.

    We don't pin the exact number of accumulate() calls: bigraph schedules
    Steps independently of simulation time, so ``composite.run(N)`` doesn't
    map 1:1 to N update ticks. The contract under test is that render()
    fires once at the end with state from the accumulator buffer.
    """
    composite = Composite({'state': _state_with_counter()}, core=_make_core())
    composite.run(5)
    # Per-tick output to the wired html store stayed empty.
    assert composite.state['count_html'] == ''
    out = render_results(composite)
    assert ('viz_count',) in out
    html = out[('viz_count',)]['html']
    # render() was called once and produced HTML from the accumulated state.
    assert html.startswith('<count n="') and 'tick' in html
