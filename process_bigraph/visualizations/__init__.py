"""Default Visualization classes shipped with process-bigraph.

All five inherit ``process_bigraph.visualization.Visualization`` and implement
``render_final(results, *, config) -> str``. They are auto-discovered via
``bigraph_schema.package.discover`` so workspaces don't need to register them
manually.

Usage (from a composite or investigation spec):
    visualizations:
      - name: trajectory
        address: "local:TimeSeriesPlot"
        config: {observable: free_DnaA, sources: [baseline]}
"""
from process_bigraph.visualizations.time_series import TimeSeriesPlot
from process_bigraph.visualizations.timeseries_from_observables import (
    TimeSeriesFromObservables,
)
from process_bigraph.visualizations.param_vs_observable import ParamVsObservable
from process_bigraph.visualizations.distribution import Distribution
from process_bigraph.visualizations.phase_space import PhaseSpace
from process_bigraph.visualizations.heatmap import Heatmap

__all__ = [
    "TimeSeriesPlot",
    "TimeSeriesFromObservables",
    "ParamVsObservable",
    "Distribution",
    "PhaseSpace",
    "Heatmap",
]
