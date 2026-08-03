"""Config-value normalizers for process ``initialize()`` / ``__init__`` bodies.

v2ecoli friction #3 (2026-05-19): a config value declared as a 2-element
range — e.g. ``band: [0.2, 0.5]`` in a composite/study yaml — does not always
arrive in the process as a list. ``bigraph-schema``'s ``node`` type (and JSON
round-trips through the dashboard's subprocess runner) rewrap it into a dict
with integer or string keys, and some authors hand-write the explicit-key
form. So ``initialize()`` ended up needing to tolerate three shapes:

1. ``[low, high]``                       — yaml-native list/tuple
2. ``{0: low, 1: high}`` / ``{"0": low, "1": high}`` — bigraph-schema rewrap
3. ``{"low": ..., "high": ...}``         — explicit-key form

Re-deriving that tolerance in every process is the boilerplate this helper
removes. Import it instead of hand-rolling ``isinstance(band, dict)`` ladders.
"""
from __future__ import annotations

from typing import Any


def normalize_config_list(
    value: Any,
    *,
    key_names: tuple[str, ...] = ("low", "high"),
    length: int | None = None,
) -> list:
    """Coerce a range-ish config value into a plain ordered ``list``.

    Accepts the three shapes a ``[low, high]``-style value can arrive in
    (see the module docstring) and returns a list in canonical order.

    - ``key_names`` names the explicit-key form's keys, in order. Defaults to
      ``("low", "high")``; pass e.g. ``("min", "max")`` or
      ``("x", "y", "z")`` to match your schema.
    - ``length`` (optional) asserts the result has exactly this many
      elements; raises ``ValueError`` otherwise. When omitted, the length
      is whatever the input implies.

    Ordering rules:
    - list / tuple → returned as a list, order preserved.
    - dict with the ``key_names`` keys → ordered by ``key_names``.
    - dict with integer-or-numeric-string keys (``0``/``"0"``/...) → ordered
      by numeric key value.
    - a scalar → wrapped as a single-element list (so callers can treat a
      bare ``0.3`` and ``[0.3]`` uniformly).

    Raises ``ValueError`` on a dict whose keys match neither convention, or
    when ``length`` is given and doesn't match.
    """
    if value is None:
        result: list = []
    elif isinstance(value, (list, tuple)):
        result = list(value)
    elif isinstance(value, dict):
        result = _from_dict(value, key_names)
    else:
        # Scalar — treat as a one-element range.
        result = [value]

    if length is not None and len(result) != length:
        raise ValueError(
            f"normalize_config_list: expected {length} element(s), "
            f"got {len(result)} from {value!r}"
        )
    return result


def _from_dict(value: dict, key_names: tuple[str, ...]) -> list:
    # Explicit-key form: every declared key present.
    if all(k in value for k in key_names):
        return [value[k] for k in key_names]

    # Numeric-key form: keys are ints or numeric strings ("0", "1", ...).
    numeric: list[tuple[int, Any]] = []
    for k, v in value.items():
        idx = _as_index(k)
        if idx is None:
            raise ValueError(
                f"normalize_config_list: dict key {k!r} is neither one of "
                f"{key_names} nor a numeric index; value={value!r}"
            )
        numeric.append((idx, v))
    numeric.sort(key=lambda pair: pair[0])
    return [v for _, v in numeric]


def _as_index(key: Any) -> int | None:
    if isinstance(key, bool):  # bool is an int subclass — exclude it
        return None
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return None
    return None
