"""TimeSeriesPlot — observable(s) vs time, one line per run."""
from __future__ import annotations
import html as _html
import json

from process_bigraph.visualization import Visualization


_PALETTE = ['#6366f1', '#10b981', '#f43f5e', '#f59e0b',
            '#8b5cf6', '#06b6d4', '#84cc16', '#ec4899']


class TimeSeriesPlot(Visualization):
    """Plot one or more observables vs time.

    Inputs (declared types):
      observable: list[float]   — trajectory values, or list-of-lists for multi-run
      time:       list[float]   — same shape as observable
    """

    def inputs(self):
        return {'observable': 'list[float]', 'time': 'list[float]'}

    def update(self, state):
        obs = state.get('observable')
        ts = state.get('time')
        if obs is None or ts is None:
            return {'html': '<p style="color:#991b1b">missing observable or time input</p>'}

        if obs and isinstance(obs[0], list):
            runs = list(zip(ts, obs))
        else:
            runs = [(ts, obs)]
        labels = state.get('_run_labels') or [''] * len(runs)
        overlays = state.get('_overlays') or []

        traces = []
        for i, (xs, ys) in enumerate(runs):
            traces.append({
                'x': xs, 'y': ys, 'type': 'scatter', 'mode': 'lines',
                'name': labels[i] if i < len(labels) else f'run {i}',
                'line': {'color': _PALETTE[i % len(_PALETTE)], 'width': 2},
            })

        shapes = []
        annotations = []
        for ov in overlays:
            kind = ov.get('kind')
            if kind == 'reference-range':
                y_min, y_max = ov.get('y_min'), ov.get('y_max')
                if y_min is not None and y_max is not None:
                    shapes.append({
                        'type': 'rect', 'xref': 'paper', 'yref': 'y',
                        'x0': 0, 'x1': 1, 'y0': y_min, 'y1': y_max,
                        'fillcolor': '#fef3c7', 'opacity': 0.3,
                        'line': {'width': 0},
                    })
                    annotations.append({
                        'xref': 'paper', 'yref': 'y',
                        'x': 0.02, 'y': y_max,
                        'text': _html.escape(ov.get('label', 'reference range')),
                        'showarrow': False,
                        'font': {'size': 11, 'color': '#92400e'},
                    })
            elif kind == 'experimental-points':
                pts = ov.get('points') or []
                if pts:
                    traces.append({
                        'x': [p['x'] for p in pts],
                        'y': [p['y'] for p in pts],
                        'type': 'scatter', 'mode': 'markers',
                        'name': ov.get('label', 'experimental'),
                        'marker': {'color': '#000', 'size': 8, 'symbol': 'circle-open'},
                    })

        title = (getattr(self, 'config', None) or {}).get('title', '')
        layout = {
            'title': {'text': _html.escape(title), 'font': {'size': 14}},
            'xaxis': {'title': {'text': 'time'}},
            'margin': {'l': 55, 'r': 15, 't': 40, 'b': 40},
            'legend': {'orientation': 'h', 'y': -0.2},
            'shapes': shapes,
            'annotations': annotations,
        }
        return {'html': (
            '<div id="viz" style="height:380px"></div>'
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            '<script>Plotly.newPlot("viz", '
            + json.dumps(traces) + ', ' + json.dumps(layout)
            + ', {responsive:true, displayModeBar:false});</script>'
        )}
