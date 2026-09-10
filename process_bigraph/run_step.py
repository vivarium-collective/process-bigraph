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
import os
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


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
    provision: Optional[list] = None,
    failure_out: Optional[str] = None,
    summary_out: Optional[str] = None,
    span_name: Optional[str] = None,
    export_context: bool = False,
) -> Dict[str, Any]:
    """Instantiate the Step, run ``update(state)``, write outputs.

    Returns the full update dict. Side effects: writes JSON to each
    path in ``out_paths`` and (if given) ``update_json_path``.
    """
    from bigraph_schema import allocate_core
    from process_bigraph import events as _events
    from process_bigraph.workflow.provision import provision_core

    core = allocate_core()
    core = provision_core(core, provision)

    cls = _resolve_class(fq_class)
    instance = cls(config or {}, core=core)

    # Observability: the Step is one task span; a failure writes
    # ``failure.json`` next to the first output and re-raises unchanged.
    em = _events.get_emitter()
    label = span_name or cls.__name__
    anchor = update_json_path or next(iter((out_paths or {}).values()), None)
    failure_path = failure_out or (str(Path(anchor).parent / 'failure.json') if anchor else None)
    span = em.start_span('task', name=label, step_class=fq_class)
    if export_context:
        # CLI only: let subprocesses inherit the task span. Library callers
        # never get their environment mutated.
        os.environ['PBG_TRACEPARENT'] = em.current_traceparent()
    em.event('task.start', name=label, step_class=fq_class)
    _t0 = _time.monotonic()
    try:
        update = instance.invoke(state or {}).update
    except BaseException as exc:
        record = _events.exception_record(exc, task=label, step_class=fq_class)
        if failure_path:
            try:
                _write_json(failure_path, record)
            except Exception:
                pass
        em.event('task.end', level='error', status='error', name=label,
                 exc_type=record['exc_type'], exc_msg=record['exc_msg'],
                 traceback_tail=record['traceback_tail'], failure_path=failure_path)
        span.end('error', f"{record['exc_type']}: {record['exc_msg']}")
        em.flush()
        raise
    if summary_out:
        _write_json(summary_out, {'task': label, 'step_class': fq_class, 'status': 'ok',
                                  'total': _time.monotonic() - _t0})
    em.event('task.end', status='ok', name=label)
    span.end('ok')
    em.flush()

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
    p.add_argument('--provision', dest='provision_specs', action='append', default=[],
                   metavar='PROVIDER_SPEC',
                   help='Provider spec for core provisioning (module:attr or JSON tuple); repeatable')
    p.add_argument('--update-json', dest='update_json_path',
                   help='Write the full update dict to this path')
    p.add_argument('--failure-out', dest='failure_out',
                   help='Where to write failure.json on error '
                        '(default: next to the first --out/--update-json)')
    p.add_argument('--summary-out', dest='summary_out',
                   help='Write a small run summary here on success')
    p.add_argument('--span-name', dest='span_name',
                   help='Name of the task span in the event stream (default: the Step class)')
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)

    # CLI default: events on stdout (PBG_EVENT_SINKS overrides); the
    # library default stays silent. See process_bigraph.events.
    from process_bigraph import events as _events
    _events.configure(default='stdout')

    config = _load_json_file(args.config_path) if args.config_path else {}
    state: Dict[str, Any] = {}
    if args.state_path:
        state.update(_load_json_file(args.state_path))
    state.update(_parse_in_args(args.in_pairs))
    out_paths = _parse_out_args(args.out_pairs)
    provision = args.provision_specs if args.provision_specs else None

    run_step(
        fq_class=args.fq_class,
        config=config,
        state=state,
        out_paths=out_paths,
        update_json_path=args.update_json_path,
        provision=provision,
        failure_out=args.failure_out,
        summary_out=args.summary_out,
        span_name=args.span_name,
        export_context=True,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
