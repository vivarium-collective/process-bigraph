"""Run a whole Composite as one batch task (Nextflow / Snakemake / shell).

Loads a composite document (``{schema, state}``), overlays an optional
initial-state document, advances the simulation, and writes the resulting
state document and/or bridge outputs. The mother→daughter handoff of
vEcoli's Nextflow workflow becomes: one task's ``--state-out`` is the next
task's ``--initial-state``.

CLI::

    python -m process_bigraph.run_composite \\
        --document DOC.json --steps N \\
        [--initial-state @init.json] \\
        [--out PORT=PATH]... [--state-out PATH]

A composite can also be built on the fly from a *build recipe* (a named
``@composite_generator`` plus overrides/provisioning) instead of a literal
document — see ``workflow.recipe.build_from_recipe``::

    python -m process_bigraph.run_composite \\
        --build RECIPE.json [--steps N] \\
        [--set KEY=JSONVAL]... [--provision MODULE:ATTR]... \\
        [--artifact PORT=REF.json]... \\
        [--out PORT=PATH]... [--state-out PATH]

``--document`` and ``--build`` are mutually exclusive.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from process_bigraph.workflow import recipe as _recipe


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively merge ``overlay`` into ``base`` (overlay wins on leaves)."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        for key, value in overlay.items():
            base[key] = _deep_merge(base.get(key), value)
        return base
    return copy.deepcopy(overlay)


def _write_json(path: str, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(value, fh, indent=2, default=repr)


def run_composite(document_path: Optional[str] = None, *, steps: float,
                  build_path: Optional[str] = None,
                  sets: Optional[Dict[str, Any]] = None,
                  artifacts: Optional[Dict[str, Any]] = None,
                  provision: Optional[Iterable] = None,
                  initial_state: Optional[Dict[str, Any]] = None,
                  out_paths: Optional[Dict[str, str]] = None,
                  state_out_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if bool(document_path) == bool(build_path):
        raise ValueError(
            "run_composite: exactly one of document_path or build_path is required")

    if build_path is not None:
        with open(build_path) as fh:
            build_doc = json.load(fh)
        composite, core = _recipe.build_from_recipe(
            build_doc, sets=sets, artifacts=artifacts, provision=provision)
        if initial_state:
            overlay = initial_state
            if isinstance(overlay, dict) and 'state' in overlay and 'schema' in overlay:
                overlay = overlay['state']
            composite.merge({}, overlay)
    else:
        from process_bigraph import Composite, allocate_core

        with open(document_path) as fh:
            document = json.load(fh)
        if initial_state:
            # --initial-state accepts EITHER a bare state dict OR a full
            # {schema, state} document (the exact shape --state-out writes) —
            # unwrap the latter so a producer's --state-out can be chained
            # directly into a consumer's --initial-state without the whole
            # document landing nested under document['state']['state'].
            overlay = initial_state
            if isinstance(overlay, dict) and 'state' in overlay and 'schema' in overlay:
                overlay = overlay['state']
            document['state'] = _deep_merge(document.get('state', {}), overlay)

        core = allocate_core()
        composite = Composite(document, core=core)

    composite.run(float(steps))

    bridge_outputs = composite.read_bridge()
    for port, path in (out_paths or {}).items():
        if not bridge_outputs or port not in bridge_outputs:
            available = sorted(bridge_outputs) if bridge_outputs else []
            raise KeyError(
                f"composite produced no bridge output for port {port!r}; "
                f"available: {available}")
        _write_json(path, bridge_outputs[port])

    if state_out_path is not None:
        _write_json(state_out_path, {
            'schema': composite.serialize_schema(),
            'state': composite.serialize_state()})

    return bridge_outputs


def _parse_out_args(pairs):
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"--out expects PORT=PATH, got {pair!r}")
        port, path = pair.split('=', 1)
        out[port] = path
    return out


def _parse_set_args(pairs):
    """Parse ``--set KEY=JSONVAL`` pairs into a dict.

    The value is parsed with ``json.loads`` so ``--set steps=3`` or
    ``--set flag=true`` produce native types; when the value isn't valid
    JSON (e.g. a bare identifier or path) it falls back to the raw string.
    """
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"--set expects KEY=JSONVAL, got {pair!r}")
        key, raw = pair.split('=', 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _parse_artifact_args(pairs):
    """Parse ``--artifact PORT=REF.json`` pairs into a dict.

    Parsed here (so the CLI surface exists now); wiring the referenced
    artifact into the composite is consumed by a later task.
    """
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"--artifact expects PORT=REF.json, got {pair!r}")
        port, ref = pair.split('=', 1)
        out[port] = ref
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m process_bigraph.run_composite',
        description='Run a whole Composite as one batch task.')
    doc_group = p.add_mutually_exclusive_group(required=True)
    doc_group.add_argument('--document', help='Composite document JSON')
    doc_group.add_argument('--build', dest='build_path',
                            help='Build recipe JSON (see workflow.recipe.build_from_recipe)')
    p.add_argument('--steps', type=float, default=None,
                   help='Advance simulation time by this amount '
                        '(optional with --build: falls back to the recipe\'s run.steps)')
    p.add_argument('--set', dest='set_pairs', action='append', default=[],
                   metavar='KEY=JSONVAL',
                   help='Build-recipe parameter override (--build only)')
    p.add_argument('--provision', dest='provision_specs', action='append', default=[],
                   metavar='MODULE:ATTR',
                   help='Additional core provider, appended after the recipe\'s own '
                        '(--build only)')
    p.add_argument('--artifact', dest='artifact_pairs', action='append', default=[],
                   metavar='PORT=REF.json', help='Artifact reference for a bridge port')
    p.add_argument('--initial-state', dest='initial_state',
                   help='JSON file with a state overlay (or @file.json)')
    p.add_argument('--out', dest='out_pairs', action='append', default=[],
                   metavar='PORT=PATH', help='Per bridge-output destination')
    p.add_argument('--state-out', dest='state_out_path',
                   help='Write the final {schema, state} document here')
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    initial_state = None
    if args.initial_state:
        raw = args.initial_state[1:] if args.initial_state.startswith('@') else args.initial_state
        with open(raw) as fh:
            initial_state = json.load(fh)

    steps = args.steps
    if args.build_path is not None and steps is None:
        with open(args.build_path) as fh:
            build_doc = json.load(fh)
        steps = build_doc.get('run', {}).get('steps')
    if steps is None:
        raise ValueError(
            "--steps is required unless --build's recipe declares run.steps")

    run_composite(
        args.document, build_path=args.build_path, steps=steps,
        sets=_parse_set_args(args.set_pairs),
        artifacts=_parse_artifact_args(args.artifact_pairs),
        provision=args.provision_specs,
        initial_state=initial_state,
        out_paths=_parse_out_args(args.out_pairs),
        state_out_path=args.state_out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
