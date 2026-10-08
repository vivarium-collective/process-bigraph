"""Opt-in debug checking of process and step update outputs (issue #99).

A Composite setting ``check_updates`` in {off, record, raise}; off by default
and zero-cost. When on, every update a Process or Step returns is checked at
the same seam strict-contract checking uses (``_project_process_update``),
before it is projected into global state. Three problems are reported:

``undeclared_port``
    The update writes a port the edge does not declare in ``outputs()``.
    Without this check the value is silently dropped during projection, so a
    typo in a port name looks like a process that never changes anything.
``non_finite``
    A numeric leaf (Python float, numpy scalar or numpy array element) is NaN
    or +/-inf. Without this check NaN propagates through every downstream
    store without an error.
``type_mismatch``
    The value does not fit the declared port schema according to
    ``core.check``. Without this check the failure, if any, surfaces later as
    an unrelated exception (e.g. ``TypeError`` inside ``apply``) that does not
    name the process responsible.

Structural deltas -- any dict carrying an underscore-prefixed key such as
``_add``, ``_remove`` or ``_divide`` -- are not type-checked, because they are
instructions to the type's ``apply`` rather than values of the port type. They
are still scanned for non-finite numbers.

In ``raise`` mode the first offending update raises ``UpdateViolation`` naming
the process path, its class and every problem found. In ``record`` mode the
problems are appended to ``Composite.update_violations`` and an
``update.violation`` event (level ``warning``) is emitted, and the run
continues.

This is a debugging aid: it checks the updates that were actually produced,
not every update a process could produce.
"""
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

CHECK_MODES = ('off', 'record', 'raise')

# Upper bound on ``Composite.update_violations`` in record mode, so a process
# that misbehaves on every tick of a long run cannot exhaust memory. Further
# violations are counted in ``update_violations_dropped``.
MAX_RECORDED = 10_000

_REPR_LIMIT = 80

Violation = Tuple[str, str]


class UpdateViolation(Exception):
    """Raised in ``raise`` mode when a process returns a malformed update."""


def validate_mode(mode: Optional[str]) -> str:
    """Normalise a ``check_updates`` setting, rejecting unknown values."""
    mode = mode or 'off'
    if mode not in CHECK_MODES:
        raise ValueError(
            f'check_updates must be one of {CHECK_MODES}, got {mode!r}')
    return mode


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _REPR_LIMIT else text[:_REPR_LIMIT - 3] + '...'


def _fmt_path(path: Tuple[Any, ...]) -> str:
    return '.'.join(str(p) for p in path)


def has_structural_keys(value: Any) -> bool:
    """True when ``value`` (or anything nested in it) is a structural delta:
    a dict with an underscore-prefixed key such as ``_add`` or ``_remove``."""
    if isinstance(value, dict):
        for key, sub in value.items():
            if isinstance(key, str) and key.startswith('_'):
                return True
            if has_structural_keys(sub):
                return True
    elif isinstance(value, (list, tuple)):
        return any(has_structural_keys(sub) for sub in value)
    return False


def iter_non_finite(value: Any, prefix: Tuple[Any, ...] = ()) -> Iterator[Tuple[Tuple[Any, ...], Any]]:
    """Yield ``(path, value)`` for every NaN or infinite number in ``value``.

    Walks dicts, lists and tuples; checks Python floats, numpy floating and
    complex scalars and numpy arrays of a numeric dtype. For an array only the
    first offending index is reported, to keep messages short."""
    if isinstance(value, bool):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            yield prefix, value
    elif isinstance(value, (complex, np.complexfloating)):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            yield prefix, value
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in 'fc' and value.size:
            bad = ~np.isfinite(value)
            if bad.any():
                index = tuple(int(i) for i in np.argwhere(bad)[0])
                yield prefix + (index,), value[index]
        elif value.dtype.kind == 'O':
            for index, item in np.ndenumerate(value):
                yield from iter_non_finite(item, prefix + (index,))
    elif isinstance(value, dict):
        for key, sub in value.items():
            yield from iter_non_finite(sub, prefix + (key,))
    elif isinstance(value, (list, tuple)):
        for index, sub in enumerate(value):
            yield from iter_non_finite(sub, prefix + (index,))


def check_update(update: Any, port_schemas: Dict[str, Any], core: Any) -> List[Violation]:
    """Return ``(code, reason)`` pairs describing what is wrong with ``update``.

    ``port_schemas`` maps each declared port to its schema (the edge's
    ``outputs()`` merged with any ``_outputs`` override on its state node).
    An empty list means the update is clean. ``None`` and empty updates are
    always clean."""
    if update is None:
        return []
    if not isinstance(update, dict):
        return [('not_a_dict',
                 f'update must be a dict keyed by port, got {type(update).__name__} {_short(update)}')]

    violations: List[Violation] = []
    declared = sorted(str(p) for p in port_schemas)
    for port, value in update.items():
        if port not in port_schemas:
            violations.append((
                'undeclared_port',
                f'wrote to port {port!r}, which is not declared in outputs() '
                f'(declared: {", ".join(declared) or "none"}); the value would be dropped'))
            continue

        for path, bad in iter_non_finite(value, (port,)):
            violations.append((
                'non_finite',
                f'non-finite value {bad!r} at {_fmt_path(path)!r}'))

        if value is None or has_structural_keys(value):
            continue
        schema = port_schemas[port]
        try:
            fits = core.check(schema, value)
        except Exception:  # noqa: BLE001 - a schema core.check cannot evaluate is left unchecked
            continue
        if not fits:
            violations.append((
                'type_mismatch',
                f'port {port!r} got {type(value).__name__} {_short(value)}, '
                f'which does not fit its schema {_render_schema(schema, core)}'))
    return violations


def _render_schema(schema: Any, core: Any) -> str:
    """Compact schema text for messages (``'float'`` rather than the resolved
    node's repr), falling back to ``repr`` if the core cannot render it."""
    try:
        rendered = core.render(schema)
    except Exception:  # noqa: BLE001 - rendering is cosmetic
        rendered = schema
    return _short(rendered) if not isinstance(rendered, str) else (
        rendered if len(rendered) <= _REPR_LIMIT else rendered[:_REPR_LIMIT - 3] + '...')


def handle_update_violations(
        violations: List[Violation], *, mode: str, path: Tuple[Any, ...], cls: str,
        composite: Any, global_time: Optional[float]) -> None:
    """Dispatch update violations: ``raise`` raises ``UpdateViolation``;
    ``record`` stores them on the composite, emits one ``update.violation``
    event and continues. No-op when ``violations`` is empty."""
    if not violations:
        return
    where = _fmt_path(tuple(path or ()))
    if mode == 'raise':
        summary = '; '.join(f'[{code}] {reason}' for code, reason in violations)
        raise UpdateViolation(f'{cls} at {where!r} returned a bad update: {summary}')
    if mode != 'record':
        return

    record = {
        'path': list(path or ()),
        'cls': cls,
        'global_time': global_time,
        'violations': [{'code': code, 'reason': reason} for code, reason in violations],
    }
    store = composite.update_violations
    if len(store) < MAX_RECORDED:
        store.append(record)
    else:
        composite.update_violations_dropped += 1

    emitter = getattr(composite, '_em', None)
    if emitter is not None:
        emitter.event('update.violation', level='warning', component='update_check',
                      path=record['path'], cls=cls, violations=record['violations'],
                      global_time=global_time)
