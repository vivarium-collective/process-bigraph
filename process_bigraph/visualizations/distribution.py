"""Distribution — histogram of a sample list.

Orchestrator collects the samples (e.g., final-step values across N seed
runs) into a flat list; this Step renders a histogram.
"""
from __future__ import annotations
import html as _html
import json

from process_bigraph.visualization import Visualization


class Distribution(Visualization):
    """Histogram of an observable's distribution.

    Inputs (declared types):
      samples: list[float] — the values to bin
    """

    def inputs(self):
        return {'samples': 'list[float]'}

    def update(self, state):
        samples = state.get('samples') or []
        title = (getattr(self, 'config', None) or {}).get('title', '')
        traces = [{
            'x': samples, 'type': 'histogram',
            'marker': {'color': '#6366f1'},
            'opacity': 0.85,
        }]
        layout = {
            'title': {'text': _html.escape(title), 'font': {'size': 14}},
            'yaxis': {'title': {'text': 'count'}},
            'margin': {'l': 55, 'r': 15, 't': 40, 'b': 40},
            'showlegend': False,
            'bargap': 0.05,
        }
        return {'html': (
            '<div id="viz" style="height:380px"></div>'
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            '<script>Plotly.newPlot("viz", '
            + json.dumps(traces) + ', ' + json.dumps(layout)
            + ', {responsive:true, displayModeBar:false});</script>'
        )}
