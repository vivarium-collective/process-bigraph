"""TimeSeriesFromObservables — self-contained multi-observable time-series.

Backs Finding #26 in the v2ecoli investigation walkthrough notes:

  > Critical missing piece: a generic ``TimeSeriesFromObservables``
  > Visualization Step that consumes any list of observable specs +
  > a runs.db and produces a time-series plot. Without this, studies
  > that declare ``address: local:TimeSeriesPlot`` have no backing
  > code and can't be previewed.

The existing :class:`TimeSeriesPlot` is the right *renderer*, but its
``observable: list[float]`` / ``time: list[float]`` inputs need to be
plumbed in by an upstream Step. The dashboard's ``build_viz_composite``
handles that plumbing for *single*-observable cases via an
``inputs_map`` convention, but most study visualizations want to plot
several observables together with shared time + per-run colors — and
the wiring boilerplate has to be written per study, which the field
notes flagged as the main reason "Visualizations tab is empty" was
the default state.

This class fixes it by being **self-contained**:

  - Its only required config is ``observables: list[str]`` — the names
    of observables (matching ``study.yaml.observables[].name``) to plot.
  - The dashboard's renderer injects ``_runs_db_path`` and
    ``_study_yaml_path`` (the two private config keys this class
    reads) when it builds the composite. Both are optional; the class
    degrades gracefully when either is missing.
  - The class reads runs.db itself (stdlib sqlite3 — no extra deps)
    and renders one Plotly trace per (observable × run) pair with
    units pulled from ``study.yaml.observables[].units`` when present.

The result is a single Viz address (``local:TimeSeriesFromObservables``)
that lights up most studies' declared visualizations without any
per-study code. The note recommends shipping this in the framework so
every workspace gets it; that's why this lives in
``process_bigraph.visualizations`` alongside the other defaults.
"""
from __future__ import annotations

import html as _html
import json
import sqlite3
from pathlib import Path

from process_bigraph.visualization import Visualization


_PALETTE = [
    "#6366f1", "#10b981", "#f43f5e", "#f59e0b",
    "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899",
    "#14b8a6", "#dc2626", "#3b82f6", "#a855f7",
]


class TimeSeriesFromObservables(Visualization):
    """Multi-observable time series, self-contained.

    Config keys (user-set in ``study.yaml.visualizations[].config``):

      - ``observables: list[str]`` (required) — observable names to plot.
        Matched against ``study.yaml.observables[].name`` for units
        + against the ``observables`` map in runs.db ``history.state``
        for data.
      - ``title: str`` (optional) — chart title. Defaults to ``""``.
      - ``sources: list[str]`` (optional) — restrict to runs whose
        ``sim_name`` is in this list. Default: all runs.

    Config keys injected by the renderer (private; users don't set
    these):

      - ``_runs_db_path: str`` — absolute path to the study's runs.db.
        When absent, the chart renders an empty-data placeholder.
      - ``_study_yaml_path: str`` — absolute path to study.yaml. Used
        to look up per-observable ``units``. When absent, units are
        omitted from axis labels.

    Inputs: none (the class plumbs its own data). The renderer's
    ``build_viz_composite`` skips inputs_map plumbing when a viz class
    declares no inputs.
    """

    config_schema = {
        # Inherits title, render_mode, sample_every from Visualization.
        **Visualization.config_schema,
        "observables": {"_type": "list[string]", "_default": []},
        "sources": {"_type": "list[string]", "_default": []},
        # Private — injected by the dashboard renderer; the schema
        # accepts string so YAML stays clean.
        "runs_db_path": {"_type": "string", "_default": ""},
        "run_id": {"_type": "string", "_default": ""},
        "study_yaml_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        # Self-contained: no bigraph inputs. The renderer detects this
        # and skips inputs_map plumbing.
        return {}

    # The class is rendered in one shot (legacy contract). Override
    # update() directly so the baseclass orchestrator stays out of the
    # way — we don't accumulate per-tick state; runs.db already has it.
    def update(self, state: dict) -> dict:
        cfg = dict(getattr(self, "config", None) or {})
        return {"html": _render_html(cfg)}


# ---------------------------------------------------------------------------
# Pure helpers (no Visualization runtime — easy to unit-test)
# ---------------------------------------------------------------------------


def _load_study_observable_meta(study_yaml_path: str | None) -> dict[str, dict]:
    """Return ``{observable_name: {units, description, store_path}}``.

    Tolerant of missing file, malformed YAML, or absent ``observables``
    block — returns an empty dict in any of those cases.
    """
    if not study_yaml_path:
        return {}
    p = Path(study_yaml_path)
    if not p.is_file():
        return {}
    # PyYAML is not a hard process-bigraph dependency (same convention as
    # composite_spec.py); import lazily so importing this module never
    # requires it unless a caller actually supplies a study_yaml_path.
    import yaml
    try:
        # study.yaml is UTF-8 (often non-ASCII prose); decode explicitly so a
        # bare-CLI render under an ASCII locale doesn't crash on read.
        spec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    out: dict[str, dict] = {}
    for o in (spec.get("observables") or []):
        if isinstance(o, dict) and isinstance(o.get("name"), str):
            out[o["name"]] = {
                "units": o.get("units") or "",
                "description": o.get("description") or "",
                "store_path": o.get("store_path") or "",
            }
    return out


def _load_runs(runs_db_path: str | None, sources: list[str], run_id: str = "") -> list[dict]:
    """Read runs.db and return a list of ``{run_id, sim_name, params,
    observables, time}`` dicts.

    ``observables`` is ``{name: list[number]}`` keyed by the names the
    emitter wrote (typically the observable name as declared in
    ``study.yaml``). ``time`` is the trajectory's time axis.

    Returns ``[]`` if the db file is missing, malformed, or empty.
    Restricts to runs in ``sources`` when provided (matching
    ``runs_meta.sim_name``).
    """
    if not runs_db_path:
        return []
    p = Path(runs_db_path)
    if not p.is_file():
        return []

    sources_set = set(sources or [])

    try:
        conn = sqlite3.connect(str(p))
    except sqlite3.Error:
        return []
    conn.row_factory = sqlite3.Row
    try:
        try:
            meta_rows = conn.execute(
                "SELECT run_id, sim_name, params_json FROM runs_meta"
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        has_history = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='history'"
        ).fetchone() is not None
        if not has_history:
            return []

        out: list[dict] = []
        for r in meta_rows:
            if run_id and r["run_id"] != run_id:
                continue
            sim_name = r["sim_name"] or "default"
            if sources_set and sim_name not in sources_set:
                continue
            try:
                params = json.loads(r["params_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                params = {}
            history_rows = conn.execute(
                "SELECT step, global_time, state FROM history "
                "WHERE simulation_id=? ORDER BY step ASC",
                (r["run_id"],),
            ).fetchall()
            observables: dict[str, list] = {}
            time_axis: list[float] = []
            for hrow in history_rows:
                try:
                    s = json.loads(hrow["state"]) if hrow["state"] else {}
                except (json.JSONDecodeError, TypeError):
                    continue
                for k, v in s.items():
                    observables.setdefault(k, []).append(v)
                if "time" in s:
                    # Already captured under observables["time"]; mirror
                    # to the dedicated axis for convenience.
                    time_axis.append(s["time"])
                else:
                    time_axis.append(hrow["global_time"])
            out.append({
                "run_id": r["run_id"],
                "sim_name": sim_name,
                "params": params,
                "observables": observables,
                "time": time_axis,
            })
        return out
    finally:
        conn.close()


def _label_for_run(run: dict, index: int) -> str:
    """Compact label for a run, preferring params; falling back to
    sim_name; finally the last 6 chars of run_id."""
    params = run.get("params") or {}
    if params:
        return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    sim_name = run.get("sim_name") or ""
    if sim_name and sim_name != "default":
        return sim_name
    rid = run.get("run_id") or f"run-{index}"
    return rid[-6:]


def _build_traces(
    runs: list[dict],
    observable_names: list[str],
) -> list[dict]:
    """Build Plotly traces — one line per (observable × run) pair.

    Lines are colored by observable; runs of the same observable share
    a color but get distinct dash patterns when there are multiple
    runs, so an expert can tell apart calibration sweeps at a glance.
    """
    traces: list[dict] = []
    dash_patterns = ["solid", "dash", "dot", "dashdot", "longdash"]
    for obs_idx, name in enumerate(observable_names):
        color = _PALETTE[obs_idx % len(_PALETTE)]
        runs_with_obs = [
            (i, r) for i, r in enumerate(runs)
            if name in (r.get("observables") or {})
        ]
        for trace_idx, (run_idx, run) in enumerate(runs_with_obs):
            y = run["observables"][name]
            x = run.get("time") or list(range(len(y)))
            dash = dash_patterns[trace_idx % len(dash_patterns)] \
                if len(runs_with_obs) > 1 else "solid"
            label = name
            if len(runs_with_obs) > 1:
                label = f"{name} — {_label_for_run(run, run_idx)}"
            traces.append({
                "x": x, "y": y, "type": "scatter", "mode": "lines",
                "name": label,
                "line": {"color": color, "width": 2, "dash": dash},
            })
    return traces


def _y_axis_label(meta: dict[str, dict], names: list[str]) -> str:
    """Compose a Y axis label from the requested observables.

    - Single observable, units known: ``"<name> (<units>)"``.
    - Multiple observables, all sharing a unit: ``"<units>"``.
    - Multiple observables, mixed units: ``"value"`` (no label).
    - No units known: ``""``.
    """
    units = [meta.get(n, {}).get("units") or "" for n in names]
    units_set = {u for u in units if u}
    if len(names) == 1:
        if units[0]:
            return f"{names[0]} ({units[0]})"
        return names[0]
    if len(units_set) == 1:
        return next(iter(units_set))
    return "value" if any(units) else ""


def _render_html(cfg: dict) -> str:
    """Pure renderer: build the Plotly HTML from a viz config dict.

    Exposed as a free function so tests can drive it without
    constructing a Visualization instance.
    """
    observable_names = list(cfg.get("observables") or [])
    if not observable_names:
        return (
            '<div style="padding:12px;color:#991b1b">'
            'TimeSeriesFromObservables: no observables declared in config. '
            'Set <code>observables: [name1, name2]</code> in the viz config.'
            '</div>'
        )

    runs = _load_runs(cfg.get("runs_db_path") or cfg.get("_runs_db_path"), cfg.get("sources") or [], cfg.get("run_id") or "")
    meta = _load_study_observable_meta(cfg.get("study_yaml_path") or cfg.get("_study_yaml_path"))

    if not runs:
        return (
            '<div style="padding:12px;color:#92400e;background:#fef3c7;'
            'border:1px solid #fcd34d;border-radius:4px">'
            '<strong>No run data yet.</strong><br>'
            'Run a baseline (or wait for a Simulate-phase run to complete) '
            'and re-render. The viz config is valid; the data side just '
            'isn\'t populated.'
            '</div>'
        )

    traces = _build_traces(runs, observable_names)
    if not traces:
        return (
            '<div style="padding:12px;color:#991b1b">'
            'None of the declared observables '
            f'({", ".join(observable_names)}) were found in any run\'s '
            'emitted state. Check that the run\'s emitter is configured '
            'to record these names.'
            '</div>'
        )

    title = cfg.get("title") or ""
    y_label = _y_axis_label(meta, observable_names)
    layout = {
        "title": {"text": _html.escape(title), "font": {"size": 14}},
        "xaxis": {"title": {"text": "time"}},
        "yaxis": {"title": {"text": _html.escape(y_label)}},
        "margin": {"l": 60, "r": 15, "t": 40, "b": 50},
        "legend": {"orientation": "h", "y": -0.2},
    }
    return (
        '<div id="viz" style="height:380px"></div>'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        '<script>Plotly.newPlot("viz", '
        + json.dumps(traces) + ", " + json.dumps(layout)
        + ", {responsive:true, displayModeBar:false});</script>"
    )
