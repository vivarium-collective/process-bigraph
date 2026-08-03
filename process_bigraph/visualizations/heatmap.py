"""Heatmap — 2D parameter sweep, color = reduced observable."""
from __future__ import annotations
import html as _html
import json

from process_bigraph.visualization import Visualization


class Heatmap(Visualization):
    """Color matrix over a 2D parameter sweep.

    Inputs (declared types):
      x_params: list[float]
      y_params: list[float]
      z_values: list[list[float]]  — z[y_idx][x_idx]
    """

    def inputs(self):
        return {
            'x_params': 'list[float]',
            'y_params': 'list[float]',
            'z_values': 'list[list[float]]',
        }

    def update(self, state):
        xs = state.get('x_params') or []
        ys = state.get('y_params') or []
        zs = state.get('z_values') or []
        title = (getattr(self, 'config', None) or {}).get('title', '')
        traces = [{
            'z': zs, 'x': xs, 'y': ys, 'type': 'heatmap',
            'colorscale': 'Viridis',
        }]
        layout = {
            'title': {'text': _html.escape(title), 'font': {'size': 14}},
            'margin': {'l': 55, 'r': 60, 't': 40, 'b': 40},
        }
        return {'html': (
            '<div id="viz" style="height:420px"></div>'
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            '<script>Plotly.newPlot("viz", '
            + json.dumps(traces) + ', ' + json.dumps(layout)
            + ', {responsive:true, displayModeBar:false});</script>'
        )}
