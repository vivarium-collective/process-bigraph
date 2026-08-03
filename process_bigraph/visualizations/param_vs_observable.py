"""ParamVsObservable — sweep parameter value vs reduced observable.

The orchestrator does the reduction (final/mean/max/...) before populating
the input store; this Step just plots ``y vs x`` as a line+marker chart.
"""
from __future__ import annotations
import html as _html
import json

from process_bigraph.visualization import Visualization


class ParamVsObservable(Visualization):
    """Plot reduced observable values across a sweep.

    Inputs (declared types):
      sweep_param_values: list[float] — x-axis (one value per run in the sweep)
      reduced_observable: list[float] — y-axis (reduced trajectory per run)
    """

    def inputs(self):
        return {
            'sweep_param_values': 'list[float]',
            'reduced_observable': 'list[float]',
        }

    def update(self, state):
        xs = state.get('sweep_param_values') or []
        ys = state.get('reduced_observable') or []
        title = (getattr(self, 'config', None) or {}).get('title', '')
        if xs and ys:
            pairs = sorted(zip(xs, ys))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
        traces = [{
            'x': xs, 'y': ys, 'type': 'scatter', 'mode': 'lines+markers',
            'line': {'color': '#6366f1', 'width': 2},
            'marker': {'color': '#6366f1', 'size': 8},
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
