"""``WorkflowBackend`` — pluggable execution of a workflow ``Composite``.

A *workflow composite* is an ordinary process-bigraph :class:`Composite`
whose Step DAG is the authoritative execution plan (see
``process_bigraph.workflow.tasks.CompositeTask`` for the scatter-axis node
and ``process_bigraph.workflow.recipe`` for how such a composite gets
built). This module is the thin execution boundary on top of that: given a
built composite, ``run_workflow`` hands it to a named :class:`WorkflowBackend`
and normalizes the result into a single :class:`RunResult` shape, regardless
of *how* the DAG actually ran (in-process today via :class:`LocalRunner`;
out-of-process — Nextflow, Ray, a batch scheduler — for future backends
registered the same way).

Two parallelism axes, deliberately orthogonal (F-note from the workflow
design):

- **Scatter axis** — many parameterizations of the *same* sub-composite
  (e.g. a parameter sweep). ``CompositeTask`` covers this itself, fanning
  out ``run_composite --build`` subprocesses across a bounded
  ``ThreadPoolExecutor`` (see ``workflow.tasks`` module docstring).
- **DAG-branch axis** — independent *Step* nodes in the outer composite
  that don't depend on each other (e.g. two unrelated ``CompositeTask``
  nodes, or a ``CompositeTask`` alongside an unrelated post-processing
  Step). A workflow composite's document should set
  ``parallel_steps: true`` (a top-level :class:`Composite` config key) to
  let independent branches of that DAG run concurrently within one layer;
  ``CompositeTask``'s own thread pool then further covers the scatter axis
  *within* whichever branch it sits on. Setting only one of the two still
  runs correctly — it just leaves the other axis serialized.

F3 (bridge is the output contract): a workflow composite communicates its
result to the outside world exclusively through its declared
``bridge.outputs`` wiring (see ``Composite.read_bridge``). Backends should
never reach into internal state paths directly — a Step DAG can restructure
its internals freely as long as the bridge contract holds.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


# ── result shape ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    """The normalized result of running a workflow composite on some backend."""

    backend: str
    status: str
    outputs: dict
    workdir: str
    provenance: dict


# ── backend protocol ─────────────────────────────────────────────────

class WorkflowBackend(Protocol):
    """Interface every workflow backend (local, Nextflow, ...) implements."""

    name: str

    def available(self) -> bool:
        """Whether this backend can run in the current environment."""
        ...

    def run(self, composite, *, outdir, code_version=None, **opts) -> RunResult:
        """Run ``composite`` to completion and return a :class:`RunResult`."""
        ...


# ── provenance aggregation (F5) ──────────────────────────────────────

def _gather_provenance(outdir) -> Dict[str, Any]:
    """Aggregate ``provenance.json`` sidecars found under ``outdir``.

    ``CompositeTask`` (see ``workflow.tasks._write_provenance``) writes one
    ``provenance.json`` per node, under a ``<workdir_root>/<generator>/``
    directory that is a *sibling* of the artifact store — not necessarily
    nested under the run's ``outdir`` at all (the artifact store is shared
    and content-addressed across runs, so it's rooted independently). This
    walks whatever tree the caller points at (typically ``outdir``, or a
    caller-controlled workdir root that contains it) and merges every
    ``provenance.json`` it finds, keyed by the immediate parent directory
    name (the generator/node name that wrote it).

    Best-effort: a missing or unreadable ``outdir`` yields an empty dict
    rather than raising — provenance is diagnostic, never load-bearing for
    the run's own success/failure determination.
    """
    aggregated: Dict[str, Any] = {}
    if not outdir or not os.path.isdir(outdir):
        return aggregated

    for root, _dirs, files in os.walk(outdir):
        if 'provenance.json' not in files:
            continue
        path = os.path.join(root, 'provenance.json')
        try:
            with open(path) as fh:
                content = json.load(fh)
        except (OSError, ValueError):
            continue
        key = os.path.basename(root) or path
        aggregated[key] = content

    return aggregated


# ── local (in-process) backend ────────────────────────────────────────

class LocalRunner:
    """Runs a workflow composite in-process, synchronously.

    ``run()`` drives the composite's own Step DAG via ``composite.run(0.0)``
    — a zero-length interval triggers no time-scheduled Processes but still
    cascades the full Step dependency graph to completion (F3: Steps run
    off their trigger graph, not off elapsed time), so a workflow composite
    built entirely of Steps (``CompositeTask`` nodes, post-processing Steps,
    etc.) reaches its final state in one call.
    """

    name = 'local'

    def available(self) -> bool:
        return True

    def run(self, composite, *, outdir, code_version=None, **opts) -> RunResult:
        try:
            composite.run(0.0)
            outputs = composite.read_bridge() or {}
            provenance = _gather_provenance(outdir)
            return RunResult(
                'local', 'ok', outputs, str(outdir),
                {**provenance, 'code_version': code_version or {}})
        except Exception as e:  # noqa: BLE001 — F3: failure path, never raise
            return RunResult('local', 'failed', {}, str(outdir), {'error': repr(e)})


# ── registry ──────────────────────────────────────────────────────────

_BACKENDS: Dict[str, WorkflowBackend] = {}


def register_backend(name: str, backend: WorkflowBackend) -> None:
    """Register ``backend`` under ``name`` (overwrites any existing entry)."""
    _BACKENDS[name] = backend


def get_backend(name: str) -> WorkflowBackend:
    """Look up a registered backend by name, or raise with the available set."""
    if name not in _BACKENDS:
        raise KeyError(f'unknown backend {name!r}; have {sorted(_BACKENDS)}')
    return _BACKENDS[name]


register_backend('local', LocalRunner())


def run_workflow(composite, *, backend: str = 'local', outdir: str = '.',
                  **opts) -> RunResult:
    """Run ``composite`` on the named backend (default: in-process ``local``)."""
    return get_backend(backend).run(composite, outdir=outdir, **opts)
