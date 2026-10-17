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
    The value is the wrong kind of thing for the port: text sent to a number,
    a float sent to an integer, a list sent to a single number, and so on.
    Without this check the failure, if any, surfaces later as an unrelated
    exception (e.g. ``TypeError`` inside ``apply``) that does not name the
    process responsible.

    An update is a change, not a new state, so this check is deliberately
    more lenient than ``core.check``: it accepts what ``apply`` accepts. Any
    real number (Python or numpy, any width) fits a float port, any integer
    fits an integer port, a numpy array fits a list port, an array port takes
    every update shape its ``apply`` understands (arrays, sparse
    ``[(index, delta), ...]`` lists, dicts, scalars), a partial dict fits a
    struct-like port, and bounded types (``range``, ``nonnegative``) are not
    bounds-checked, because a negative change to a non-negative store is
    ordinary. Bounds on the resulting value are ``contract_strict``'s job.

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
import numbers
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from bigraph_schema.schema import (
    Array, Boolean, Complex, Float, Integer, List as ListSchema, Map, Node,
    Set as SetSchema, String, Tree, Tuple as TupleSchema, Union, Wrap)

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


def _truncate(text: str) -> str:
    return text if len(text) <= _REPR_LIMIT else text[:_REPR_LIMIT - 3] + '...'


def _short(value: Any) -> str:
    return _truncate(repr(value))


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


def check_update(update: Any, port_schemas: Dict[str, Any], core: Any,
                 ports_key: str = 'outputs') -> List[Violation]:
    """Return ``(code, reason)`` pairs describing what is wrong with ``update``.

    ``port_schemas`` maps each declared port to its schema (the edge's
    ``outputs()`` merged with any ``_outputs`` override on its state node).
    ``ports_key`` names that port set in messages. An empty list means the
    update is clean. ``None`` and empty updates are always clean."""
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
                f'wrote to port {port!r}, which is not declared in {ports_key}() '
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
            # Fast path: almost every update already passes the strict
            # state check. Only a failure pays for the lenient walk below.
            fits = core.check(schema, value) or fits_update(core.access(schema), value, core)
        except Exception:  # noqa: BLE001 - a schema we cannot evaluate is left unchecked
            continue
        if not fits:
            violations.append((
                'type_mismatch',
                f'port {port!r} got {type(value).__name__} {_short(value)}, '
                f'which does not fit its schema {_render_schema(schema, core)}'))
    return violations


# ---------------------------------------------------------------------------
# Lenient update check: "is this the right kind of thing for the port?"
# ---------------------------------------------------------------------------
#
# ``core.check`` answers "is this a valid *state* for the schema", which is
# stricter than what ``apply`` accepts as an *update*: ``Float`` means
# ``isinstance(x, float)`` (so ``2`` and ``np.float32`` fail), bounded types
# check the delta against the bounds, struct-like nodes require every field,
# and ``Array`` requires an exact shape and dtype. ``fits_update`` widens
# only those cases and defers to ``core.check`` for everything else.

_DTYPE_KINDS = {
    'boolean': 'b',
    'integer': 'biu',
    'float': 'biuf',
    'complex': 'biufc',
}


def _number_kind(schema: Any) -> Optional[str]:
    """``'boolean' | 'integer' | 'float' | 'complex'`` for a numeric atom
    schema, else ``None``. ``Complex`` subclasses ``Float`` and ``Range``,
    ``Nonnegative`` and ``Delta`` are floats, so order matters."""
    if isinstance(schema, Boolean):
        return 'boolean'
    if isinstance(schema, Integer):
        return 'integer'
    if isinstance(schema, Complex):
        return 'complex'
    if isinstance(schema, Float):
        return 'float'
    return None


def _fits_number(kind: str, value: Any) -> bool:
    """Whether a scalar ``value`` is the right kind of number for ``kind``.

    Booleans are kept apart from numbers: ``True`` is not a float update and
    ``1`` is not a boolean update, even though Python would add them."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.dtype.kind in _DTYPE_KINDS[kind] and (
            kind == 'boolean' or value.dtype.kind != 'b')
    is_bool = isinstance(value, (bool, np.bool_))
    if kind == 'boolean':
        return is_bool
    if kind == 'integer':
        # bool is an int subclass and ``core.check`` already accepts it for
        # an integer port, so it is accepted here too (never narrow).
        return isinstance(value, numbers.Integral)
    if is_bool:
        return False
    if kind == 'float':
        return isinstance(value, numbers.Real)
    return isinstance(value, numbers.Number)  # complex


def _fits_array_dtype(element: Any, array: np.ndarray) -> Optional[bool]:
    """Fast answer for a numpy array against a numeric element schema, or
    ``None`` when the element schema is not a plain number."""
    kind = _number_kind(element)
    if kind is None:
        return None
    if array.dtype.kind == 'O':
        return all(_fits_number(kind, item) for item in array.ravel().tolist())
    allowed = _DTYPE_KINDS[kind]
    if kind != 'boolean':
        allowed = allowed.replace('b', '')
    return array.dtype.kind in allowed


def fits_update(schema: Any, value: Any, core: Any) -> bool:
    """Whether ``value`` is an acceptable *update* for a resolved ``schema``.

    Called only after ``core.check`` has rejected ``value``, so it may
    only widen what ``core.check`` accepts, never narrow it."""
    if value is None:
        return True

    kind = _number_kind(schema)
    if kind is not None:
        return _fits_number(kind, value)

    if isinstance(schema, Array):
        # Array.apply accepts arrays, sparse [(index, delta), ...] lists,
        # dicts (structured fields or per-index cells) and scalars that
        # broadcast. Flag only things it has no reading for.
        if isinstance(value, (np.ndarray, list, tuple, dict)):
            return True
        if isinstance(value, (bool, np.bool_)):
            return schema._data.kind == 'b'
        return isinstance(value, (numbers.Number, np.generic)) and not isinstance(value, (str, bytes))

    if isinstance(schema, Wrap):           # maybe, overwrite, const, quote, ...
        return core.check(schema, value) or fits_update(schema._value, value, core)

    if isinstance(schema, Union):
        return any(core.check(option, value) or fits_update(option, value, core)
                   for option in schema._options)

    if isinstance(schema, (ListSchema, SetSchema)):
        element = schema._element
        if isinstance(value, np.ndarray) and isinstance(schema, ListSchema):
            fast = _fits_array_dtype(element, value)
            if fast is not None:
                return fast
            items = value.tolist()
        elif isinstance(schema, SetSchema):
            if not isinstance(value, (set, frozenset)):
                return False
            items = value
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            return False
        return all(_fits_item(element, item, core) for item in items)

    if isinstance(schema, TupleSchema):
        if not isinstance(value, (list, tuple)) or len(value) != len(schema._values):
            return False
        return all(_fits_item(sub, item, core) for sub, item in zip(schema._values, value))

    if isinstance(schema, Map):
        if not isinstance(value, dict):
            return False
        if not isinstance(schema._key, String):
            if not all(_fits_item(schema._key, key, core) for key in value):
                return False
        return all(_fits_item(schema._value, item, core) for item in value.values())

    if isinstance(schema, Tree):
        if _fits_item(schema._leaf, value, core):
            return True
        return isinstance(value, dict) and all(
            isinstance(key, str) and _fits_item(schema, branch, core)
            for key, branch in value.items())

    if isinstance(schema, dict):
        # A dict schema: check only the keys the update carries.
        if not isinstance(value, dict):
            return False
        return all(_fits_item(schema[key], item, core)
                   for key, item in value.items() if key in schema)

    if isinstance(schema, Node):
        fields = [f for f in getattr(schema, '__dataclass_fields__', ()) if not f.startswith('_')]
        if fields and isinstance(value, dict):
            # Struct-like node: an update may carry only the fields it changes.
            return all(_fits_item(getattr(schema, key), value[key], core)
                       for key in fields if key in value)

    return False


def _fits_item(schema: Any, value: Any, core: Any) -> bool:
    try:
        return bool(core.check(schema, value)) or fits_update(schema, value, core)
    except Exception:  # noqa: BLE001 - an unevaluable sub-schema is left unchecked
        return True


def _render_schema(schema: Any, core: Any) -> str:
    """Compact schema text for messages (``'float'`` rather than the resolved
    node's repr), falling back to ``repr`` if the core cannot render it."""
    try:
        rendered = core.render(schema)
    except Exception:  # noqa: BLE001 - rendering is cosmetic
        rendered = schema
    return _truncate(rendered) if isinstance(rendered, str) else _short(rendered)


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
