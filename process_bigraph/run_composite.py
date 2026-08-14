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
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


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


def run_composite(document_path: str, *, steps: float,
                  initial_state: Optional[Dict[str, Any]] = None,
                  out_paths: Optional[Dict[str, str]] = None,
                  state_out_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from process_bigraph import Composite, allocate_core

    with open(document_path) as fh:
        document = json.load(fh)
    if initial_state:
        document['state'] = _deep_merge(document.get('state', {}), initial_state)

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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m process_bigraph.run_composite',
        description='Run a whole Composite as one batch task.')
    p.add_argument('--document', required=True, help='Composite document JSON')
    p.add_argument('--steps', type=float, required=True,
                   help='Advance simulation time by this amount')
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
    run_composite(
        args.document, steps=args.steps, initial_state=initial_state,
        out_paths=_parse_out_args(args.out_pairs),
        state_out_path=args.state_out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
