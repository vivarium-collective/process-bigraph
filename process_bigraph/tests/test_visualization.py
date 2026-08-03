"""Tests for process_bigraph.visualization.Visualization (moved from viva_superpowers).

The baseclass now orchestrates accumulate/render with ``render_mode='end'``
as the default. Subclasses MAY override ``update(state)`` directly to keep
the legacy streaming contract; both paths are exercised here.
"""
import pytest

from process_bigraph import Step
from process_bigraph.visualization import Visualization, as_visualization


# --- legacy subclass: overrides update() directly --------------------------

class _Echo(Visualization):
    """Legacy-style subclass that echoes the input as html each tick."""

    def inputs(self):
        return {'msg': 'string'}

    def update(self, state):
        return {'html': '<p>' + state.get('msg', '') + '</p>'}


# --- new-style subclass: implements accumulate + render --------------------

class _Counter(Visualization):
    """New-style subclass that accumulates a count then renders at end."""

    def inputs(self):
        return {'msg': 'string'}

    def accumulate(self, state):
        self._n = getattr(self, '_n', 0) + 1
        self._last_msg = state.get('msg', '')

    def render(self):
        return f'<p>n={self._n}, last={self._last_msg}</p>'


def _make(cls):
    """Helper: instantiate without running through bigraph wiring."""
    return object.__new__(cls)


# ---------------------------------------------------------------------------
# baseclass shape


def test_visualization_is_step_subclass():
    assert issubclass(Visualization, Step)


def test_visualization_outputs_default_html():
    inst = _make(Visualization)
    assert inst.outputs() == {'html': 'string'}


def test_visualization_inputs_default_empty():
    inst = _make(Visualization)
    assert inst.inputs() == {}


def test_visualization_marker_classmethod():
    assert _Echo.is_visualization() is True
    assert _Counter.is_visualization() is True


def test_render_mode_in_config_schema():
    """Default render_mode must be 'end' so test-suite-style runs don't
    pay the per-tick render cost."""
    assert Visualization.config_schema['render_mode']['_default'] == 'end'
    assert Visualization.config_schema['sample_every']['_default'] == 1


# ---------------------------------------------------------------------------
# legacy contract: subclass.update() overrides orchestrator


def test_legacy_subclass_update_returns_html_dict():
    inst = _make(_Echo)
    out = inst.update({'msg': 'hello'})
    assert out == {'html': '<p>hello</p>'}


# ---------------------------------------------------------------------------
# new-style contract: baseclass.update() orchestrates accumulate + render


def test_new_style_end_mode_accumulates_silently():
    """In default ('end') mode, update() accumulates but emits empty html."""
    inst = _make(_Counter)
    inst.config = {}
    assert inst.update({'msg': 'a'}) == {'html': ''}
    assert inst.update({'msg': 'b'}) == {'html': ''}
    assert inst.update({'msg': 'c'}) == {'html': ''}
    # Accumulation happened across all three ticks.
    assert inst.render() == '<p>n=3, last=c</p>'


def test_new_style_stream_mode_renders_each_tick():
    """``render_mode='stream'`` keeps the legacy per-tick HTML behavior."""
    inst = _make(_Counter)
    inst.config = {'render_mode': 'stream'}
    assert inst.update({'msg': 'a'}) == {'html': '<p>n=1, last=a</p>'}
    assert inst.update({'msg': 'b'}) == {'html': '<p>n=2, last=b</p>'}


def test_new_style_sample_every_skips_intermediate_ticks():
    """``sample_every=N`` accumulates only every Nth tick."""
    inst = _make(_Counter)
    inst.config = {'sample_every': 3}
    for msg in ['t0', 't1', 't2', 't3', 't4', 't5', 't6']:
        inst.update({'msg': msg})
    # Ticks 0, 3, 6 were accumulated → n == 3, last == 't6'.
    assert inst.render() == '<p>n=3, last=t6</p>'


def test_new_style_default_accumulate_snapshots_last_state():
    """The baseclass default ``accumulate`` stashes the latest state on
    ``_last_state`` so stateless renderers don't need their own buffer."""

    class _SnapOnly(Visualization):
        def render(self):
            return f'<p>{self._last_state.get("msg")}</p>'

    inst = _make(_SnapOnly)
    inst.config = {}
    inst.update({'msg': 'a'})
    inst.update({'msg': 'b'})
    assert inst.render() == '<p>b</p>'


def test_new_style_missing_render_raises():
    """New-style subclass that forgets ``render()`` blows up only when the
    orchestrator actually tries to render — i.e. in stream mode or via
    render_results."""

    class _NoRender(Visualization):
        def accumulate(self, state):
            pass

    inst = _make(_NoRender)
    inst.config = {}  # 'end' mode — accumulate-only, never calls render()
    assert inst.update({}) == {'html': ''}

    # Stream mode triggers render() and surfaces the NotImplementedError.
    inst.config = {'render_mode': 'stream'}
    with pytest.raises(NotImplementedError, match='render'):
        inst.update({})


# ---------------------------------------------------------------------------
# as_visualization decorator — legacy-style by definition


def test_as_visualization_synthesizes_subclass():
    @as_visualization(inputs={'x': 'list[float]'}, name='MyViz', demo={'x': [1.0, 2.0]})
    def update_my_viz(state):
        return {'html': '<p>x=' + str(state['x']) + '</p>'}

    assert issubclass(update_my_viz, Visualization)
    assert update_my_viz.__name__ == 'MyViz'
    assert update_my_viz.__pb_kind__ == 'visualization'
    assert 'MyViz' in update_my_viz.__pb_aliases__
    inst = _make(update_my_viz)
    assert inst.inputs() == {'x': 'list[float]'}
    assert inst.outputs() == {'html': 'string'}
    assert inst.update({'x': [1.0, 2.0]}) == {'html': '<p>x=[1.0, 2.0]</p>'}


def test_as_visualization_demo_dict():
    @as_visualization(inputs={'x': 'list[float]'},
                      demo={'x': [3.0, 4.0]})
    def update_demo_dict(state):
        return {'html': str(state['x'])}

    assert update_demo_dict.demo() == {'x': [3.0, 4.0]}


def test_as_visualization_demo_callable():
    @as_visualization(inputs={'x': 'list[float]'},
                      demo=lambda: {'x': [5.0, 6.0]})
    def update_demo_callable(state):
        return {'html': str(state['x'])}

    assert update_demo_callable.demo() == {'x': [5.0, 6.0]}


def test_as_visualization_function_name_validation():
    with pytest.raises(AssertionError, match='update_'):
        @as_visualization(inputs={})
        def bad_name(state):
            return {'html': ''}


def test_as_visualization_default_name_from_function():
    @as_visualization(inputs={'x': 'list[float]'})
    def update_inferred_name(state):
        return {'html': ''}

    assert update_inferred_name.__name__ == 'inferred_name'
    assert 'inferred_name' in update_inferred_name.__pb_aliases__


def test_as_visualization_aliases():
    @as_visualization(inputs={}, name='Primary', aliases=['alt1', 'alt2'])
    def update_aliased(state):
        return {'html': ''}

    assert 'Primary' in update_aliased.__pb_aliases__
    assert 'alt1' in update_aliased.__pb_aliases__
    assert 'alt2' in update_aliased.__pb_aliases__


# ----------------------------------------------------------------------------
# F4 — as_visualization is deprecated in favor of subclassing
# ----------------------------------------------------------------------------


def test_as_visualization_emits_deprecation_warning():
    """The decorator continues to work but warns at decoration time so
    authors are nudged toward the canonical subclass form."""
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        @as_visualization(inputs={'x': 'list[float]'}, name='_F4_TestViz')
        def update__f4_test_viz(state):
            return {'html': '<x>' + str(state['x']) + '</x>'}

    dep = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert dep, "expected a DeprecationWarning from as_visualization"
    msg = str(dep[0].message)
    assert "deprecated" in msg.lower()
    assert "subclass Visualization" in msg
    # The decorated class is still a real Visualization subclass with the
    # pb-discovery markers; instantiation requires a core (not the point
    # of this test — covered by the older test_as_visualization_synthesizes_subclass).
    assert update__f4_test_viz.__name__ == '_F4_TestViz'
    assert update__f4_test_viz.__pb_kind__ == 'visualization'
    assert '_F4_TestViz' in update__f4_test_viz.__pb_aliases__
    assert issubclass(update__f4_test_viz, Visualization)


def test_visualization_subclass_does_not_warn():
    """The canonical subclass path produces no DeprecationWarning. Anchors
    the recommended pattern as the silent one — workspaces that adopt it
    don't see migration nudges."""
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        class _F4_CanonicalViz(Visualization):
            def inputs(self):
                return {'x': 'list[float]'}

            def update(self, state):
                return {'html': '<x>' + str(state.get('x', [])) + '</x>'}

    dep = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert not dep, f"subclass form must be silent; got {[str(w.message) for w in dep]}"


# ---------------------------------------------------------------------------
# stable_div_id helper (mem3dg-readdy friction #28)
#
# The method only reads `type(self).__name__` and `self.config`; it doesn't
# touch any process_bigraph machinery. Instantiating a real Visualization
# requires a `core` (process_bigraph framework requirement) which is heavy
# for a pure-string test. Call the method against a minimal duck object
# instead — same code path, no framework dependency.
# ---------------------------------------------------------------------------


def _div_id(name: str, config: dict | None, *parts: str) -> str:
    """Invoke Visualization.stable_div_id against a duck object whose
    class name is `name` (set via dynamic class creation so the formatted
    prefix is deterministic)."""
    cls = type(name, (object,), {})
    obj = cls()
    if config is not None:
        obj.config = config
    return Visualization.stable_div_id(obj, *parts)


def test_stable_div_id_is_deterministic_per_class_and_title():
    """Two duck objects with the same class name + title produce the SAME
    div_id — the whole point. The friction #28 case used id(self), which
    is CPython's object address; stable_div_id is content-addressed."""
    a = _div_id("DivIDViz", {"title": "growth"})
    b = _div_id("DivIDViz", {"title": "growth"})
    assert a == b
    # Format: lowercased-classname-8hex.
    import re
    assert re.fullmatch(r"dividviz-[0-9a-f]{8}", a)


def test_stable_div_id_differs_by_title():
    """Same class, different config title → different ids. A workspace
    shipping two instances of one Viz class with distinct titles (e.g.
    'baseline' vs 'variant') gets two distinct DOM ids without manual
    discriminators."""
    a = _div_id("DivIDViz", {"title": "baseline"})
    b = _div_id("DivIDViz", {"title": "variant"})
    assert a != b


def test_stable_div_id_accepts_extra_discriminator_parts():
    """Callers can pass extra `parts` (e.g. a per-run id) to disambiguate
    further. Distinct parts → distinct ids; repeated parts → same id."""
    p1 = _div_id("DivIDViz", {"title": "x"}, "run-1")
    p2 = _div_id("DivIDViz", {"title": "x"}, "run-2")
    p1_again = _div_id("DivIDViz", {"title": "x"}, "run-1")
    assert p1 != p2
    assert p1 == p1_again


def test_stable_div_id_handles_missing_config():
    """A subclass that doesn't set self.config (e.g. mid-init) shouldn't
    crash — the helper falls back to empty title."""
    result = _div_id("NoConfigViz", config=None)
    assert result.startswith("noconfigviz-")
