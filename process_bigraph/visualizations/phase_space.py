"""PhaseSpace — two observables plotted against each other."""
from __future__ import annotations
import html as _html
import json

from process_bigraph.visualization import Visualization


class PhaseSpace(Visualization):
    """XY trajectory of two observables.

    Inputs (declared types):
      x: list[float]
      y: list[float]
    """

    def inputs(self):
        return {'x': 'list[float]', 'y': 'list[float]'}

    def update(self, state):
        xs = state.get('x') or []
        ys = state.get('y') or []
        title = (getattr(self, 'config', None) or {}).get('title', '')
        traces = [{
            'x': xs, 'y': ys, 'type': 'scatter', 'mode': 'lines+markers',
            'line': {'color': '#6366f1', 'width': 2},
            'marker': {'size': 5},
        }]
        layout = {
            'title': {'text': _html.escape(title), 'font': {'size': 14}},
            'margin': {'l': 55, 'r': 15, 't': 40, 'b': 40},
            'showlegend': False,
        }
        return {'html': (
            '<div id="viz" style="height:380px"></div>'
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            '<script>Plotly.newPlot("viz", '
            + json.dumps(traces) + ', ' + json.dumps(layout)
            + ', {responsive:true, displayModeBar:false});</script>'
        )}
