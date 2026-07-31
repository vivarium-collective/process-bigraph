"""
Standalone stdio worker for the ``git:`` protocol.

This script runs **inside the fetched repo's own venv** (``<sha>/venv``), where
process-bigraph is deliberately *not* installed. It therefore imports only the
Python standard library, so it can be launched by any interpreter regardless of
what scientific stack the foreign repo pulls in. The host talks to it over a
small newline-delimited-JSON RPC on stdin/stdout — this is the isolation
boundary: foreign code (and foreign dependencies) live entirely in this
subprocess and never enter the host interpreter (spec contract 3, D1(a)).

Wire protocol (one JSON object per line):

    host -> worker : {"cmd": "init", "config": {...}}
    worker -> host : {"ok": true, "interface": {"inputs": {...}, "outputs": {...}}}

    host -> worker : {"cmd": "update", "state": {...}, "interval": 1.0}
    worker -> host : {"ok": true, "update": {...}}

    host -> worker : {"cmd": "interface"}
    worker -> host : {"ok": true, "interface": {...}}

    host -> worker : {"cmd": "end"}
    (worker exits)

Any failure resolving/instantiating the entrypoint, or in an update, is
reported as ``{"ok": false, "error": "..."}`` — a clear error the host raises,
never a host crash.

Invocation::

    <venv-python> _git_worker.py <module> <callable>

``<callable>`` is invoked with the init ``config`` if it accepts an argument,
else with no arguments. Its return value must be an edge-like object exposing
``inputs()``, ``outputs()`` and ``update(state, interval)``.
"""

import sys
import json
import inspect
import importlib
import traceback


def _load_entry(module_name, callable_name):
    module = importlib.import_module(module_name)
    entry = module
    for attr in callable_name.split('.'):
        entry = getattr(entry, attr)
    return entry


def _instantiate(entry, config):
    """Call the entrypoint. If it takes at least one parameter, pass the
    config; otherwise call it with no arguments. Keeps the entrypoint
    contract minimal (spec D4): ``module:callable`` returning an edge."""
    if not callable(entry):
        # The fragment named a value that is already an edge-like object.
        return entry
    try:
        sig = inspect.signature(entry)
        takes_arg = any(
            p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        takes_arg = True
    if takes_arg:
        return entry(config)
    return entry()


def _interface(edge):
    face = getattr(edge, 'interface', None)
    if callable(face):
        result = face() or {}
        return {
            'inputs': dict(result.get('inputs') or {}),
            'outputs': dict(result.get('outputs') or {}),
        }
    return {
        'inputs': dict(edge.inputs() or {}),
        'outputs': dict(edge.outputs() or {}),
    }


def _send(obj):
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()


def main(argv):
    if len(argv) != 3:
        _send({'ok': False,
               'error': f'usage: _git_worker.py <module> <callable> '
                        f'(got {argv!r})'})
        return 2

    module_name, callable_name = argv[1], argv[2]

    # --- init handshake -------------------------------------------------
    first = sys.stdin.readline()
    if not first:
        return 0
    try:
        request = json.loads(first)
    except json.JSONDecodeError as error:
        _send({'ok': False, 'error': f'bad init request: {error}'})
        return 2

    config = request.get('config') or {}
    try:
        entry = _load_entry(module_name, callable_name)
        edge = _instantiate(entry, config)
        interface = _interface(edge)
    except Exception:
        _send({'ok': False,
               'error': f'failed to load entrypoint '
                        f'{module_name}:{callable_name}: '
                        f'{traceback.format_exc()}'})
        return 1

    _send({'ok': True, 'interface': interface})

    # --- command loop ---------------------------------------------------
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as error:
            _send({'ok': False, 'error': f'bad request: {error}'})
            continue

        cmd = command.get('cmd')
        if cmd == 'end':
            break
        elif cmd == 'interface':
            try:
                _send({'ok': True, 'interface': _interface(edge)})
            except Exception:
                _send({'ok': False, 'error': traceback.format_exc()})
        elif cmd == 'update':
            try:
                update = edge.update(
                    command.get('state') or {},
                    command.get('interval'))
                _send({'ok': True, 'update': update})
            except Exception:
                _send({'ok': False, 'error': traceback.format_exc()})
        else:
            _send({'ok': False, 'error': f'unknown command: {cmd!r}'})

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
