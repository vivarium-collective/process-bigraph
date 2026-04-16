"""
Unified runner: invoke a Step class from the command line.

This is the bridge between a composite document and any execution
environment that runs processes as separate subprocesses — Nextflow,
Snakemake, a shell pipeline, or an ad-hoc test. The same Step class
that runs inside a native ``Composite`` runs here, driven by files on
disk instead of the in-memory state tree.

CLI
---

::

    python -m process_bigraph.run_step \\
        --class MODULE.PATH.ClassName \\
        [--config CONFIG_JSON] \\
        [--state STATE_JSON] \\
        [--in PORT=VALUE]... \\
        [--out PORT=PATH]... \\
        [--update-json UPDATE_JSON]

``--class``
    Fully qualified Step class, resolved by ``importlib.import_module``.

``--config``
    Optional JSON file holding the Step's config dict. Falls back to
    ``{}`` if omitted.

``--state``
    Optional JSON file holding the complete input state (``{port:
    value}``). Merged with any ``--in`` overrides; ``--in`` wins.

``--in PORT=VALUE``
    Per-port input. ``VALUE`` is either a JSON literal (``"wt"``,
    ``42``, ``[1,2,3]``) or ``@FILE.json`` to read from a file. Repeatable.

``--out PORT=PATH``
    Per-port output destination. The runner writes JSON at ``PATH``
    containing just that port's value from the update dict. Repeatable.

``--update-json PATH``
    If given, the entire update dict (all output ports) is written here
    in addition to any per-port ``--out`` files.

Exit code 0 on success. On error, the exception propagates with a
non-zero exit code and a traceback on stderr — Nextflow / Snakemake can
see the failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _resolve_class(fq_name: str) -> type:
    """Import ``module.sub.Class`` and return the class object."""
    if '.' not in fq_name:
        raise ValueError(
            f"--class must be fully qualified (module.Class), got {fq_name!r}")
    module_name, class_name = fq_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"class {class_name!r} not found in module {module_name!r}"
        ) from e


def _load_json_file(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _parse_in_value(raw: str) -> Any:
    """Parse a single ``--in PORT=VALUE`` right-hand side.

    ``@FILE.json`` reads JSON from disk; everything else is parsed as a
    JSON literal. Bare strings without quotes fall back to the raw text
    so users don't have to quote-shell-quote simple identifiers like
    ``wt`` or ``ko``.
    """
    if raw.startswith('@'):
        return _load_json_file(raw[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_in_args(pairs: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"--in expects PORT=VALUE, got {pair!r}")
        port, raw = pair.split('=', 1)
        out[port] = _parse_in_value(raw)
    return out


def _parse_out_args(pairs: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"--out expects PORT=PATH, got {pair!r}")
        port, path = pair.split('=', 1)
        out[port] = path
    return out


def _write_json(path: str, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(value, fh, indent=2, default=repr)


def run_step(
    fq_class: str,
    config: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    out_paths: Optional[Dict[str, str]] = None,
    update_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Instantiate the Step, run ``update(state)``, write outputs.

    Returns the full update dict. Side effects: writes JSON to each
    path in ``out_paths`` and (if given) ``update_json_path``.
    """
    from bigraph_schema import allocate_core
    core = allocate_core()

    cls = _resolve_class(fq_class)
    instance = cls(config or {}, core=core)

    update = instance.invoke(state or {}).update

    for port, path in (out_paths or {}).items():
        if port not in update:
            raise KeyError(
                f"Step {fq_class!r} produced no output for port {port!r}; "
                f"available ports: {sorted(update.keys())}")
        _write_json(path, update[port])

    if update_json_path is not None:
        _write_json(update_json_path, update)

    return update


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m process_bigraph.run_step',
        description=__doc__.split('\n\n')[0],
    )
    p.add_argument('--class', dest='fq_class', required=True,
                   help='Fully qualified Step class (module.ClassName)')
    p.add_argument('--config', dest='config_path',
                   help='JSON file with the Step config dict')
    p.add_argument('--state', dest='state_path',
                   help='JSON file with the full input state dict')
    p.add_argument('--in', dest='in_pairs', action='append', default=[],
                   metavar='PORT=VALUE',
                   help='Per-port input (JSON literal or @file.json); repeatable')
    p.add_argument('--out', dest='out_pairs', action='append', default=[],
                   metavar='PORT=PATH',
                   help='Per-port output destination; repeatable')
    p.add_argument('--update-json', dest='update_json_path',
                   help='Write the full update dict to this path')
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)

    config = _load_json_file(args.config_path) if args.config_path else {}
    state: Dict[str, Any] = {}
    if args.state_path:
        state.update(_load_json_file(args.state_path))
    state.update(_parse_in_args(args.in_pairs))
    out_paths = _parse_out_args(args.out_pairs)

    run_step(
        fq_class=args.fq_class,
        config=config,
        state=state,
        out_paths=out_paths,
        update_json_path=args.update_json_path,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
