"""Port-compatibility classifier built on the bigraph-schema Translator kernel.

process-bigraph composites wire a producer's output port onto a consumer's
input/store with **no** compatibility check: ``resolve`` silently widens
(e.g. ``integer -> float``) and an apply-relevant wrapper (``Overwrite`` /
``Delta``) declared on the producer side is silently dropped at the wiring
boundary, so a *replacement* update quietly becomes an *additive* one (the
``Float`` apply at ``methods/apply.py`` adds the update as a delta; the exact
failure ``resolve.py``'s ``promote`` docstring warns about).

This module is a **standalone, additive** demonstration that the typed
``Translator`` kernel (``bigraph_schema.translator`` +
``Core.register_translator`` / ``Core.cross``) catches those hazards. It does
NOT modify ``resolve`` / ``apply`` / ``Composite`` wiring. It classifies a
crossing from a producer output type ``P`` to a consumer store type ``C`` into
one of four outcomes, every one emitted through the kernel as ``Crossed`` or
``Refusal`` (never a silent ``None``/default, never a bare dict):

- **exact**       — ``render(access(P)) == render(access(C))``
                    -> ``Crossed`` carrying a ``PortCrossing`` tagged ``exact``.
- **widened**     — ``resolve(P, C)`` succeeds and widens the producer value
                    (``join != P``, e.g. ``integer`` into a ``float`` store)
                    -> ``Crossed`` tagged ``widened``, naming the join. This is
                    ``resolve`` finally *declaring* the join it silently takes.
- **semantics-shifted** — an apply-relevant wrapper (``Overwrite`` / ``Delta``)
                    or a ``_units`` annotation is present on exactly one side
                    -> ``Refusal(semantics_shifted, ...)`` (the dropped-Overwrite
                    -> additive-apply hazard).
- **irreconcilable** — ``resolve(P, C)`` raises "cannot resolve types"
                    -> ``Refusal(irreconcilable, ...)``.

Integration note (NOT done here, deliberately): this classifier would slot into
``Composite.initialize`` as a *report-only* wiring pass — after wiring is
resolved, walk every ``(producer_output_port_type -> target_store_type)`` pair
and run ``classify_port_crossing``; log ``widened`` crossings and surface
``Refusal``s as warnings/errors. It is kept OUT of the hot path here.
"""

from dataclasses import dataclass
from typing import Any, Optional

from bigraph_schema.translator import Crossed, Refusal, Translator
from bigraph_schema.methods.coerce import coerce


# Apply-relevant wrapper Node classes whose presence changes update semantics
# (replacement vs additive). Detected by class name so this module does not
# import the concrete schema classes.
_APPLY_WRAPPERS = {'Overwrite', 'Delta'}


@dataclass(frozen=True)
class PortCrossing:
    """The structured payload a successful crossing carries inside ``Crossed``.

    Not a bare dict — a typed report. ``tag`` is ``'exact'`` or ``'widened'``;
    ``join`` names the resolved join type; ``value`` is the genuinely-crossed
    value produced by ``Core.cross``.
    """

    tag: str
    source: str
    target: str
    join: str
    value: Any = None
    note: str = ''


def _semantics_signature(core, type_expr):
    """Return ``(wrapper_name_or_None, units_string)`` for ``type_expr``.

    ``wrapper_name`` is the apply-relevant wrapper class (``'Overwrite'`` /
    ``'Delta'``) at the top of the accessed Node, else ``None``. ``units`` is
    the ``_units`` annotation found on the node (or its wrapped ``_value``),
    else ``''``.
    """
    node = core.access(type_expr)
    wrapper = type(node).__name__ if type(node).__name__ in _APPLY_WRAPPERS else None
    units = getattr(node, '_units', '') or ''
    if not units:
        inner = getattr(node, '_value', None)
        if inner is not None:
            units = getattr(inner, '_units', '') or ''
    return wrapper, units


def _emit_crossing(core, source_type, target_type, cross_fn, translator_id):
    """Register a one-shot translator and drive it through ``Core.cross``.

    ``cross_fn(value) -> value | Refusal`` performs the actual crossing; the
    kernel's ``Core.cross`` enforces the source check, the anti-silent-None
    law, and (on success) the target check, returning ``Crossed`` or
    ``Refusal``. A valid sample of ``source_type`` is supplied so the source
    check passes; the classification is purely type-level.
    """
    core.register_translator(Translator(
        id=translator_id,
        source=source_type,
        target=target_type,
        mode='partial',
        cross_fn=cross_fn,
    ))
    sample = core.default(source_type)[1]
    return core.cross(translator_id, sample)


def classify_port_crossing(core, producer_type, consumer_type, translator_id=None):
    """Classify the crossing ``producer_type -> consumer_type`` via the kernel.

    Returns ``Crossed(PortCrossing(...))`` for ``exact`` / ``widened``
    crossings, or a ``Refusal`` for ``semantics_shifted`` / ``irreconcilable``.
    Always a kernel outcome — never a bare dict, never a silent default.

    ``core`` is a process-bigraph / bigraph-schema core exposing
    ``register_translator`` and ``cross`` (inherited from bigraph-schema's
    ``Core``).
    """
    rp = core.render(core.access(producer_type))
    rc = core.render(core.access(consumer_type))
    tid = translator_id or f'wiring::{rp}->{rc}'

    # 1. Semantics shift: an apply-relevant wrapper / units on exactly one side.
    #    Checked first: `overwrite[float] -> float` would otherwise look merely
    #    "widened", masking that the Overwrite (replacement) semantics are
    #    dropped and apply falls back to additive Float.
    wrap_p, units_p = _semantics_signature(core, producer_type)
    wrap_c, units_c = _semantics_signature(core, consumer_type)
    if wrap_p != wrap_c or units_p != units_c:
        detail = []
        if wrap_p != wrap_c:
            detail.append(
                f'apply-wrapper mismatch: producer={wrap_p!r} consumer={wrap_c!r} '
                f'(dropped {wrap_p or wrap_c} -> update semantics shift: '
                f'replacement silently becomes additive apply)')
        if units_p != units_c:
            detail.append(f'units mismatch: producer={units_p!r} consumer={units_c!r}')
        reason = 'semantics_shifted: ' + '; '.join(detail)

        def refuse_semantics(_value, _reason=reason, _tid=tid, _rp=rp, _rc=rc,
                             _off=(wrap_p, wrap_c, units_p, units_c)):
            return Refusal(translator_id=_tid, reason=_reason,
                           source=_rp, target=_rc, offending=_off)

        return _emit_crossing(core, producer_type, consumer_type,
                              refuse_semantics, tid)

    # 2. Exact: identical rendered normal forms.
    if rp == rc:
        join_node = core.access(consumer_type)

        def cross_exact(value, _node=join_node):
            return coerce(_node, value)

        result = _emit_crossing(core, producer_type, consumer_type, cross_exact, tid)
        if isinstance(result, Refusal):
            return result
        return Crossed(PortCrossing(
            tag='exact', source=rp, target=rc, join=rc, value=result.value,
            note='ports share an identical rendered type'))

    # 3. Try to resolve; a raise means irreconcilable.
    try:
        join_node = core.resolve(producer_type, consumer_type)
    except Exception as e:
        reason = f'irreconcilable: resolve({rp!r}, {rc!r}) raised: {e}'

        def refuse_irreconcilable(_value, _reason=reason, _tid=tid, _rp=rp, _rc=rc):
            return Refusal(translator_id=_tid, reason=_reason,
                           source=_rp, target=_rc, offending=(_rp, _rc))

        return _emit_crossing(core, producer_type, consumer_type,
                              refuse_irreconcilable, tid)

    # 4. Resolve succeeded -> widened (the join `resolve` silently takes today).
    join = core.render(join_node)

    def cross_widened(value, _node=join_node):
        return coerce(_node, value)

    result = _emit_crossing(core, producer_type, consumer_type, cross_widened, tid)
    if isinstance(result, Refusal):
        return result
    return Crossed(PortCrossing(
        tag='widened', source=rp, target=rc, join=join, value=result.value,
        note=(f'resolve widens {rp!r} into the join {join!r} '
              f'(lossy: producer value silently widened at the wiring boundary)')))


def iter_wired_output_crossings(composite, instances):
    """Yield ``(process_path, port, store_path, producer_type, store_type)`` for
    every wired output port of every process in ``composite``.

    A report-only helper mirroring what a ``Composite.initialize`` wiring pass
    would walk: for each edge, each output port's declared type (producer ``P``,
    from the process instance's ``outputs()``) paired with the target store's
    resolved schema type (consumer ``C``, from the composite schema at the wired
    path). The caller runs ``classify_port_crossing(core, P, C)`` on each pair.

    ``instances`` maps a process name in ``composite.state`` to its live
    instance (process instances are built lazily and are not guaranteed to sit
    in ``state`` at initialize time, so they are passed in explicitly). A real
    ``Composite.initialize`` pass would read the same ``outputs()`` off the
    instances it just built.
    """
    core = composite.core
    schema = composite.schema
    state = composite.state

    for name, node in state.items():
        if not isinstance(node, dict) or 'outputs' not in node:
            continue
        instance = instances.get(name)
        if instance is None or not hasattr(instance, 'outputs'):
            continue
        out_types = instance.outputs()
        for port, store_path in (node.get('outputs') or {}).items():
            producer_type = out_types.get(port)
            if producer_type is None:
                continue
            store_type = _schema_type_at(core, schema, store_path)
            if store_type is None:
                continue
            yield (name, port, tuple(store_path), producer_type, store_type)


def _schema_type_at(core, schema, path) -> Optional[str]:
    """Render the schema type at ``path`` within a composite ``schema`` dict."""
    node = schema
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return core.render(node)
    except Exception:
        return None
