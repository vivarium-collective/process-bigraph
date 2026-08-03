"""Visualization Step base class — accumulate during the run, render at the end.

Visualization is a process_bigraph.Step. Subclasses declare typed input ports
via ``inputs()`` and produce HTML by implementing two hooks:

- ``accumulate(state)``: called each tick (or every Nth tick in 'sample' mode)
  with the current per-step state. Subclasses mutate their own buffers here.
- ``render() -> str``: builds the final HTML from the accumulated buffers.

The baseclass ``update(state)`` orchestrates these based on the
``render_mode`` config:

- ``'end'`` (default): accumulate each tick, return ``{'html': ''}`` from
  every ``update()`` call. The dashboard / test harness calls
  :func:`render_results` after the run to materialize the HTML once.
- ``'stream'``: accumulate AND render every tick. Use when the runtime is
  short (dashboard's Composite Explorer Run tab) and per-tick HTML is wanted.
- ``'sample'``: accumulate only every ``sample_every`` ticks, defer rendering
  to end-of-run (same as ``'end'`` but cheaper accumulation).

Subclasses that need the old per-tick streaming contract can still override
``update(state)`` directly — the baseclass detects this and skips the
orchestrator. This keeps every existing Visualization subclass working
unchanged.

Discovery: Visualization extends Step extends Edge, so subclasses are
auto-discovered via ``bigraph_schema.package.discover`` and registered in
``core.link_registry``.

This module also exposes :func:`render_results`, the visualization analogue
of ``process_bigraph.emitter.gather_emitter_results``: it walks a Composite's
state for Visualization instances and returns a path-keyed dict of their
rendered HTML, calling ``render()`` once per viz.
"""
from __future__ import annotations
from typing import Any

from process_bigraph import Step


class Visualization(Step):
    """Base class for renderable Visualization Steps.

    New-style contract:
      - override ``accumulate(state)`` to buffer per-tick state
      - override ``render() -> str`` to build the final HTML
      - leave ``update(state)`` to the baseclass orchestrator

    Legacy contract (still supported):
      - override ``update(state) -> {'html': str}`` directly; the orchestrator
        steps aside.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': ''},
        # New-style controls. Ignored when a subclass overrides update().
        'render_mode':  {'_type': 'string',  '_default': 'end'},
        'sample_every': {'_type': 'integer', '_default': 1},
    }

    # Pluggable units resolver: a callable ``path -> unit_str | None``.
    # Workspaces (e.g. v2ecoli) assign this; left None elsewhere -> no-op.
    units_resolver = None

    @classmethod
    def resolve_unit(cls, path):
        """Resolve the unit for an observable path via the pluggable resolver."""
        resolver = cls.units_resolver
        if resolver is None or not path:
            return None
        try:
            return resolver(path) or None
        except Exception:
            return None

    @staticmethod
    def _append_unit(label, unit):
        """Append ``(unit)`` to a label, idempotently. None unit -> unchanged."""
        if not unit:
            return label
        text = (label or "").rstrip()
        if text.endswith(f"({unit})"):
            return text
        return f"{text} ({unit})".strip()

    @classmethod
    def finalize_figure(cls, fig, axis_units=()):
        """Append schema units to matplotlib axis labels in-place.

        ``axis_units`` is an iterable of ``(ax, which, path)`` where ``which`` is
        ``'x'`` or ``'y'`` and ``path`` is the observable dotted path that axis
        displays. Axes whose path has no unit are left unchanged. Returns ``fig``.
        """
        for ax, which, path in axis_units:
            unit = cls.resolve_unit(path)
            if not unit:
                continue
            if which == "y" and hasattr(ax, "set_ylabel"):
                ax.set_ylabel(cls._append_unit(ax.get_ylabel(), unit))
            elif which == "x" and hasattr(ax, "set_xlabel"):
                ax.set_xlabel(cls._append_unit(ax.get_xlabel(), unit))
        return fig

    @classmethod
    def figure_to_html(cls, fig, axis_units=(), *, dpi=150, close=True):
        """Finalize axis units, then serialize a matplotlib figure to an <img>.

        One-stop replacement for per-viz ``_fig_to_b64`` + manual <img> wrapping.
        """
        import base64
        import io
        cls.finalize_figure(fig, axis_units)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        if close:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'

    def inputs(self) -> dict[str, Any]:
        """Typed input ports — keys are port names; values are bigraph-schema
        type strings. Subclasses override.
        """
        return {}

    def outputs(self) -> dict[str, Any]:
        """All visualizations expose a single ``html`` string port."""
        return {'html': 'string'}

    def accumulate(self, state: dict) -> None:
        """New-style hook: buffer per-tick state into ``self`` for later render.

        Default implementation snapshots the latest state on ``self._last_state``,
        which is enough for stateless renderers that only need the most recent
        frame. Override for genuine history accumulation.
        """
        self._last_state = state

    def render(self) -> str:
        """New-style hook: produce the final HTML from accumulated buffers.

        Subclasses MUST implement this (or override ``update(state)`` directly
        for the legacy streaming contract).
        """
        raise NotImplementedError(
            f'{type(self).__name__}: implement render() (and optionally '
            f'accumulate(state)), or override update(state) directly.'
        )

    def update(self, state: dict) -> dict:
        """Default orchestrator. Subclasses MAY override directly (legacy).

        New-style subclasses leave this alone; the baseclass calls
        ``accumulate(state)`` per tick and only emits HTML in
        ``render_mode='stream'``. End-of-run rendering happens via
        :func:`render_results`.
        """
        cfg = getattr(self, 'config', None) or {}
        sample_every = max(1, int(cfg.get('sample_every', 1) or 1))
        tick = getattr(self, '_tick', -1) + 1
        self._tick = tick
        if tick % sample_every == 0:
            self.accumulate(state)
        mode = cfg.get('render_mode', 'end') or 'end'
        if mode == 'stream':
            return {'html': self.render()}
        return {'html': ''}

    def stable_div_id(self, *parts: str) -> str:
        """Stable, collision-resistant DOM id for the rendered HTML container.

        Use this instead of ``id(self)`` when building Plotly / Vega div ids
        (mem3dg-readdy friction #28 — ``id(self)`` happens to be the
        CPython object address and can collide when two viz instances on the
        same page get the same address after GC reuses an id slot).

        Hashes the class name plus the config title plus any extra ``parts``
        the caller passes in (e.g. a per-instance discriminator). Returns
        an 8-char hex suffix prefixed with the lowercased class name so the
        id is human-debuggable in devtools::

            <div id="couplingtrace-3f9c1ab8">

        Pure stdlib; safe to call from ``__init__`` or ``render``.
        """
        import hashlib
        cls_name = type(self).__name__.lower()
        cfg = getattr(self, 'config', None) or {}
        title = str(cfg.get('title') or '')
        payload = "|".join([type(self).__name__, title, *parts]).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()[:8]
        return f"{cls_name}-{digest}"

    @classmethod
    def is_visualization(cls) -> bool:
        """Marker for dashboard filtering: distinguishes viz Steps from Emitters."""
        return True


def _is_new_style(instance) -> bool:
    """True iff the subclass relies on the baseclass orchestrator.

    A subclass is "new-style" when it does NOT override ``update`` itself —
    i.e. the orchestrator owns the per-tick path and ``render()`` is the
    single source of HTML.
    """
    return type(instance).update is Visualization.update


def as_visualization(inputs, name=None, demo=None, aliases=None):
    """**Deprecated** — prefer subclassing ``Visualization`` directly.

    Decorator: convert an ``update_*`` pure function into a Visualization subclass.
    The function must be named ``update_<viz_name>`` and accept
    ``state: dict`` -> ``{'html': str}``.

    Why deprecated (F4 of the framework cleanup):

    - Every shipped Visualization in ``process_bigraph.visualizations.*`` and
      every Visualization in real workspaces (v2ecoli) uses explicit
      subclassing. The decorator path is only exercised by tests.
    - The decorator registers TWO names per class (PascalCase from ``name``
      AND the snake_case ``update_<x>`` suffix), forcing the workspace lint
      to grep for both forms when resolving ``local:Foo`` addresses.
    - Subclassing makes the input/output contract explicit at the class
      definition site instead of buried in decorator kwargs, which is
      easier to read and to extend (accumulate/render new-style).

    Migration::

        # before
        @as_visualization(inputs={'x': 'list[float]'}, name='MyViz')
        def update_my_viz(state):
            return {'html': '<x>' + str(state['x']) + '</x>'}

        # after
        class MyViz(Visualization):
            def inputs(self):
                return {'x': 'list[float]'}

            def update(self, state):
                return {'html': '<x>' + str(state['x']) + '</x>'}

    The decorator continues to work for back-compat — existing call sites
    don't need to migrate immediately. A DeprecationWarning fires at
    decoration time so authors see the nudge.

    Args:
        inputs:  typed input port map (same shape as Visualization.inputs()).
                 Keys are port names; values are bigraph-schema type strings.
        name:    class name override (default: derived from function name).
        demo:    sample state dict (or callable returning one) for dashboard previews.
        aliases: extra registration aliases for bigraph-schema discovery.

    Returns the synthesized Visualization subclass, ready to be registered by
    ``bigraph_schema.discover_packages()`` when the enclosing module is walked.
    """
    import warnings as _warnings
    _warnings.warn(
        "as_visualization is deprecated; subclass Visualization directly. "
        "See the docstring for a migration example. The decorator continues "
        "to work but new code should use the subclass form for clarity and "
        "to avoid the double-name (snake_case + PascalCase) registration "
        "the decorator emits.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(func):
        if not func.__name__.startswith("update_"):
            raise AssertionError(
                f"as_visualization expects a function named update_<viz_name>; "
                f"got '{func.__name__}'"
            )
        viz_name = name or func.__name__[len("update_"):]
        _demo = demo

        class FunctionVisualization(Visualization):
            def inputs(self):
                return inputs

            def outputs(self):
                return {'html': 'string'}

            def update(self, state):
                return func(state)

            @classmethod
            def demo(cls):
                if callable(_demo):
                    return _demo()
                return dict(_demo or {})

        FunctionVisualization.__name__ = viz_name
        FunctionVisualization.__qualname__ = viz_name
        FunctionVisualization.__module__ = func.__module__
        FunctionVisualization.__doc__ = func.__doc__
        FunctionVisualization.__pb_kind__ = "visualization"
        FunctionVisualization.__pb_aliases__ = [viz_name] + list(aliases or [])
        FunctionVisualization.__pb_wrapped__ = func
        return FunctionVisualization
    return decorator


def render_results(composite, results=None):
    """Gather rendered HTML from every Visualization step in ``composite``.

    Returns a dict ``{step_path: {'html': '<rendered>'}}`` mirroring the shape
    of ``process_bigraph.emitter.gather_emitter_results``.

    Two modes:

    - ``results=None`` — end-of-run rendering. For new-style vizes (subclass
      did not override ``update``), call ``instance.render()`` once per viz.
      For legacy vizes, fall back to reading the last value the runtime wrote
      to the html output port.

    - ``results=<dict>`` — replay mode. For new-style vizes, accumulate the
      provided state then render. For legacy vizes, call ``instance.update``
      directly. Useful for re-rendering against a saved SQLiteEmitter dump.
    """
    from process_bigraph.composite import find_instance_paths

    viz_paths = find_instance_paths(
        composite.state,
        'process_bigraph.visualization.Visualization',
    )
    out = {}
    for path in viz_paths:
        node = _get_path(composite.state, path)
        if node is None:
            continue
        instance = node.get('instance') if isinstance(node, dict) else None
        if instance is None:
            continue
        new_style = _is_new_style(instance)
        try:
            if results is not None:
                # Replay mode.
                if new_style:
                    instance.accumulate(results)
                    html = instance.render()
                else:
                    rendered = instance.update(results) or {}
                    html = rendered.get('html', '') if isinstance(rendered, dict) else ''
            else:
                # End-of-run.
                if new_style:
                    html = instance.render()
                else:
                    html = _read_last_html(composite, path) or ''
            rendered = {'html': html}
        except Exception as e:  # noqa: BLE001
            rendered = {'html': f'<pre style="color:#c00">render failed: {e}</pre>'}
        out[path] = rendered
    return out


def _get_path(state, path_tuple):
    node = state
    for p in path_tuple:
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def _read_last_html(composite, path_tuple):
    """Best-effort: walk the composite's bigraph for the html output store
    wired to this visualization, and return its current value (string).

    bigraph wires ``outputs.html`` to a store somewhere in the composite. The
    Visualization step's node has an ``outputs: {html: <store-path>}`` mapping
    we can use to look up the value. If the wiring isn't found or the store
    doesn't hold a string, returns None.
    """
    node = _get_path(composite.state, path_tuple)
    if not isinstance(node, dict):
        return None
    outputs = node.get('outputs') or {}
    html_target = outputs.get('html')
    if html_target is None:
        return None
    if isinstance(html_target, (list, tuple)):
        value = _get_path(composite.state, tuple(html_target))
    else:
        value = None
    return value if isinstance(value, str) else None
