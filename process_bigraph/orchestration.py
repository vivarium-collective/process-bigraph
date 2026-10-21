"""Pluggable orchestration strategies for :class:`~process_bigraph.composite.Composite`.

An *orchestration strategy* owns how a Composite advances — its top-level
execution pattern. The engine ships ``temporal`` (the default: the
event-scheduling tick lifecycle implemented by ``Composite._run_inner``).
Downstream packages register additional patterns — e.g. a ``workflow`` DAG pass
or an ``investigation`` member-study dispatcher — via :func:`register_orchestration`,
exactly as processes and types are registered.

A strategy decides *ordering and dispatch*; it reuses the Composite's engine
primitives (the per-process ``front``, ``apply_updates``, ``determine_steps``)
rather than reimplementing the typed delta/apply model.

Design: ``docs/superpowers/specs/2026-10-20-pluggable-orchestration-registry.md``.
This module is step 1 of that RFC — a behaviour-preserving seam: the default
``temporal`` strategy delegates verbatim to the existing ``_run_inner`` loop, so
a Composite with no declared orchestration runs exactly as before.
"""
from __future__ import annotations


class OrchestrationStrategy:
    """Base class: owns a Composite's top-level advance.

    Subclasses implement :meth:`advance`. The strategy holds a reference to the
    Composite it drives and may call its engine primitives.
    """

    name = "orchestration"

    def __init__(self, composite):
        self.composite = composite

    def advance(self, interval, force_complete, run_start):
        """Advance ``self.composite`` by ``interval`` (mutating its state through
        the engine's apply path). ``force_complete`` and ``run_start`` match the
        existing ``Composite._run_inner`` contract."""
        raise NotImplementedError


class TemporalOrchestration(OrchestrationStrategy):
    """The default event-scheduling tick lifecycle.

    Delegates verbatim to ``Composite._run_inner`` — the loop is unchanged; this
    only routes the call through the registry seam so alternative strategies can
    be plugged in without touching the engine loop.
    """

    name = "temporal"

    def advance(self, interval, force_complete, run_start):
        return self.composite._run_inner(interval, force_complete, run_start)


class WorkflowOrchestration(OrchestrationStrategy):
    """Pure zero-interval step DAG — no temporal event loop.

    Runs the ready-step network to completion: ``run_steps`` self-cascades
    through ``cycle_step_state``/``trigger_steps``, so one call drains the whole
    DAG in dependency order. For composites that are entirely ``step`` nodes
    (analysis / workflow pipelines, e.g. the investigation member-study DAG).
    ``global_time`` is advanced to the requested horizon in one move, since a
    workflow has no sub-interval temporal dynamics.

    This strategy composes only two Composite primitives — the ready-step queue
    (``to_run``) and ``run_steps`` — a much smaller surface than the temporal
    loop, which is what makes the clean engine-primitive interface visible.
    """

    name = "workflow"

    def advance(self, interval, force_complete, run_start):
        composite = self.composite
        if composite.to_run:
            composite.run_steps(composite.to_run)
            composite.to_run = []
        composite.state['global_time'] = composite.state.get('global_time', 0.0) + interval


ORCHESTRATION_REGISTRY: dict = {}


def register_orchestration(name: str, strategy: type) -> None:
    """Register an orchestration strategy class under ``name``.

    Overwrites an existing registration (like process registration), so a
    downstream ``build_core`` can replace or add patterns idempotently.
    """
    if not (isinstance(strategy, type) and issubclass(strategy, OrchestrationStrategy)):
        raise TypeError(
            "orchestration strategy must be an OrchestrationStrategy subclass, "
            f"got {strategy!r}")
    ORCHESTRATION_REGISTRY[name] = strategy


def get_orchestration(name: str) -> type:
    """Resolve a registered orchestration strategy class by ``name``."""
    try:
        return ORCHESTRATION_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown orchestration {name!r}; registered: "
            f"{sorted(ORCHESTRATION_REGISTRY)}") from None


# The built-in default. Downstream packages register their own (e.g.
# vivarium-workbench registers 'investigation') at build_core time.
register_orchestration("temporal", TemporalOrchestration)
register_orchestration("workflow", WorkflowOrchestration)
