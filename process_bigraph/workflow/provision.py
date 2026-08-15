from __future__ import annotations

import importlib
import sys
from typing import Any, Iterable


def _parse(p):
    """Parse a provider spec into (module, attr, args, kwargs).

    Accepts:
    - String: 'module:attr' format
    - Tuple: (module, attr) or (module, attr, args) or (module, attr, args, kwargs)
    """
    if isinstance(p, str):
        if ':' not in p:
            raise ValueError(f"provider must be 'module:attr', got {p!r}")
        m, a = p.split(':', 1)
        return m, a, (), {}
    return p[0], p[1], tuple(p[2]) if len(p) > 2 else (), dict(p[3]) if len(p) > 3 else {}


def provision_core(core: Any, providers: Iterable) -> Any:
    """Apply provider functions to a core object.

    Each provider is a callable that takes (core, *args, **kwargs) and
    returns either None (core is modified in-place) or a new core object
    (which is then used for the next provider).

    Providers can be specified as:
    - String: 'module:attr' (module and attr name)
    - Tuple: (module, attr, args, kwargs) with optional args/kwargs

    Args:
        core: The core object to provision.
        providers: Iterable of provider specs.

    Returns:
        The provisioned core object (possibly modified or replaced).
    """
    for prov in providers or []:
        mod, attr, args, kwargs = _parse(prov)
        try:
            fn = getattr(importlib.import_module(mod), attr)
            r = fn(core, *args, **kwargs)
            if r is not None:
                core = r
        except Exception as e:
            sys.stderr.write(f'[provision] {mod}:{attr} failed: {type(e).__name__}: {e}\n')
            raise
    return core
