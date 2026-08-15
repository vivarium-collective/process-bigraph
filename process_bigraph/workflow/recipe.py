"""Build a Composite from a *build recipe* document.

A build recipe names a registered ``@composite_generator`` (rather than
embedding a literal ``{schema, state}`` document) plus the modules to import
so that generator registers itself, the core extensions/provisioning needed
to run it, and the parameter overrides to apply. This is the declarative
counterpart to ``run_composite --document``: instead of a pre-built document
file, a recipe says *how to build one*.

Build-doc schema::

    {
      "build": {
        "generator": "<name>",         # matched against GeneratorEntry.name
        "import": ["<module>", ...],   # imported (for side effects) first
        "overrides": {...},            # generator kwarg overrides
        "provision": [...]             # provider specs, see workflow.provision
      },
      "artifacts": {"<port>": {"kind": ..., "map": "store"}},  # parsed by callers
      "run": {"steps": N}
    }

Order of operations (see module docstring / task brief for the rationale):

1. Import every module listed in ``build.import`` (fires the
   ``@composite_generator`` decorator so the entry lands in the registry).
2. Resolve the entry by ``name`` in ``composite_generator._REGISTRY``.
3. Allocate a fresh core.
4. ``apply_core_extensions(entry, core)`` FIRST — the generator's own
   declared extensions are the foundation.
5. ``provision_core(core, build.provision + provision)`` SECOND — CLI/caller
   supplied providers supplement (never replace) the generator's foundation.
6. Merge ``overrides`` with ``sets`` (``sets`` wins — it's the more specific,
   caller-supplied layer), then with the artifact-derived overrides (each
   build-doc ``artifacts`` entry with ``map == 'store'`` resolves its
   ``artifacts[port]`` ref file to ``overrides[port] = ArtifactRef.store``).
7. ``build_generator(entry, overrides, core)`` to get the document.
8. Return ``Composite(doc, core), core``.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

_log = logging.getLogger(__name__)


def _infer_parameters(func) -> Dict[str, dict]:
    """Infer a permissive ``{name: {type, default}}`` parameter declaration
    from a generator function's signature.

    A generator registered via ``@composite_generator`` without an explicit
    ``parameters=`` declaration has an empty ``entry.parameters`` — fine for
    the dashboard's parameter-form rendering (nothing to render), but
    ``build_generator``/``CompositeSpec.to_document`` rejects any override
    key not present in ``parameters`` (unknown-parameter guard). A recipe's
    ``overrides``/``sets`` are addressed at the generator's actual kwarg
    surface, not its (optional) UI metadata, so fall back to introspecting
    the function signature for any override key the declared parameters
    don't cover.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    out: Dict[str, dict] = {}
    for pname, p in sig.parameters.items():
        if pname == 'core':
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        default = None if p.default is inspect.Parameter.empty else p.default
        out[pname] = {'type': 'any', 'default': default}
    return out


def _accepts_core(func) -> bool:
    """Whether ``func(core=..., **kwargs)`` is a legal call for ``func``.

    The composite-generator convention documents the builder signature as
    ``(core=None, **kwargs) -> dict``, and ``CompositeSpec.to_document``
    always calls it that way. A generator that only cares about its own
    kwargs (no ``core`` param, no ``**kwargs`` catch-all) would otherwise
    raise ``TypeError`` on the ``core=`` keyword it never declared.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    params = sig.parameters
    return 'core' in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _core_optional(func):
    """Wrap ``func`` to swallow a ``core=`` kwarg it doesn't accept.

    No-op (returns ``func`` unchanged) when ``func`` already accepts
    ``core`` — see :func:`_accepts_core`.
    """
    if _accepts_core(func):
        return func

    def _wrapped(core=None, **kwargs):
        return func(**kwargs)

    return _wrapped


def _resolve_artifact_overrides(
    build_doc: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """``{port: ArtifactRef.store}`` for each build-doc artifact entry that
    was supplied a ``ref_path``.

    Only ``build_doc['artifacts']`` entries with ``map == 'store'`` are
    consumed — the doc schema reserves other ``map`` values for
    artifact-consumption shapes not yet implemented. A ``port`` declared in
    the build doc but absent from ``artifacts`` is left for
    ``overrides``/``sets`` (or the generator's own default) to supply.

    When the loaded ref carries a ``fingerprint``, it is checked — warned
    on mismatch, never raised. This is an attestation that the artifact's
    producer is deterministic given its inputs, not a cache gate: a stale or
    nondeterministic artifact is still usable, it is just flagged.
    """
    from process_bigraph.artifacts import (
        ArtifactRef, ArtifactResults, check_fingerprint, fingerprint_of,
    )

    declared = build_doc.get('artifacts', {}) or {}
    result: Dict[str, Any] = {}
    for port, spec in declared.items():
        if (spec or {}).get('map') != 'store':
            continue
        ref_path = (artifacts or {}).get(port)
        if ref_path is None:
            continue
        with open(ref_path) as fh:
            ref = ArtifactRef.coerce(json.load(fh))
        result[port] = ref.store

        if ref.fingerprint:
            try:
                results = ArtifactResults.from_ref(ref)
                observed = fingerprint_of(results.resolve())
                status = check_fingerprint(results, observed)
                if status != 'ok':
                    _log.warning(
                        "artifact %r (port %r): fingerprint mismatch "
                        "(stored=%r observed=%r) — producer may be "
                        "nondeterministic; proceeding anyway",
                        ref.hash, port, ref.fingerprint, observed)
            except Exception as exc:  # best-effort attestation, never fatal
                _log.warning(
                    "artifact %r (port %r): could not verify fingerprint "
                    "(%s) — proceeding anyway", ref.hash, port, exc)
    return result


def build_from_recipe(
    build_doc: Dict[str, Any],
    sets: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    provision: Optional[Iterable] = None,
) -> Tuple[Any, Any]:
    """Build a ``(Composite, core)`` pair from a build-recipe document.

    Args:
        build_doc: parsed recipe document, see module docstring for schema.
        sets: parameter overrides that take precedence over
            ``build_doc['build']['overrides']`` (e.g. from CLI ``--set``).
        artifacts: ``{port: ref_path}`` — for each ``build_doc['artifacts']``
            entry named ``port`` with ``map == 'store'``, the JSON file at
            ``ref_path`` is loaded as an :class:`ArtifactRef` and its
            ``store`` (a path — a file for a trajectory, a directory for a
            bundle like ``sim_data``) is injected into ``overrides[port]``
            unchanged, e.g. from CLI ``--artifact PORT=REF.json``. When the
            ref carries a ``fingerprint``, it is checked (warn-only, never
            raises) as an attestation that the artifact's producer is
            deterministic — not a cache gate.
        provision: additional provider specs appended AFTER
            ``build_doc['build']['provision']`` — CLI-supplied providers
            supplement the generator's declared core_extensions/provisioning,
            they don't replace them.

    Returns:
        ``(Composite, core)``.
    """
    from process_bigraph import Composite, allocate_core
    from process_bigraph.composite_generator import (
        _REGISTRY, apply_core_extensions, build_generator,
    )
    from process_bigraph.workflow.provision import provision_core

    build = build_doc.get('build', {}) or {}

    for mod_name in build.get('import', []) or []:
        importlib.import_module(mod_name)

    generator_name = build.get('generator')
    matches = [e for e in _REGISTRY.values() if e.name == generator_name]
    if not matches:
        raise ValueError(
            f"build_from_recipe: no registered generator named {generator_name!r}")
    entry = matches[0]

    core = allocate_core()

    # Generator's own declaration is the foundation.
    core = apply_core_extensions(entry, core)
    # CLI/caller-supplied providers supplement it.
    all_providers = list(build.get('provision', []) or []) + list(provision or [])
    core = provision_core(core, all_providers)

    overrides = dict(build.get('overrides', {}) or {})
    overrides.update(sets or {})
    overrides.update(_resolve_artifact_overrides(build_doc, artifacts))

    build_target: Any = entry
    declared = dict(entry.parameters or {})
    undeclared = set(overrides) - set(declared)
    func = entry.func
    needs_shim = bool(undeclared and func is not None) or (
        func is not None and not _accepts_core(func))
    if needs_shim:
        if undeclared:
            inferred = _infer_parameters(func)
            for key in undeclared:
                if key in inferred:
                    declared[key] = inferred[key]
        from process_bigraph import composite_spec as _cs
        build_target = _cs.CompositeSpec(
            id=entry.id, name=entry.name, description=entry.description,
            parameters=declared, builder=_core_optional(func), module=entry.module,
            default_n_steps=entry.default_n_steps,
            visualizations=list(entry.visualizations or []),
            emitters=list(entry.emitters or []),
            core_extensions=list(entry.core_extensions or []),
        )

    doc = build_generator(build_target, overrides, core)

    composite = Composite(doc, core=core)
    return composite, core
