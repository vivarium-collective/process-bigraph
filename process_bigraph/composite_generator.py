"""Composite generator convention — decorator + registry.

Sibling to the *.composite.{yaml,json} static-spec convention. A composite
generator is a Python function `(core=None, **kwargs) -> dict` that builds
a process-bigraph document; the decorator records it in a module-level
registry so discovery can enumerate generators without callers having to
maintain a separate list.

This module is now a thin shim over ``process_bigraph.composite_spec``, which
is the single source of truth for the registry.  The ``GeneratorEntry``
dataclass and helpers (``emitter_defaults``, ``install_default_emitters``,
``apply_core_extensions``) are preserved unchanged so the dashboard keeps
working on the same attribute surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from process_bigraph import composite_spec as _cs


@dataclass
class GeneratorEntry:
    """One registered composite-generator function."""

    id: str                           # "<dotted_module>.<name>"
    name: str
    description: str
    parameters: dict[str, dict]       # {name: {type, default, description?, choices?}}
    # Each parameter is a {type, default, description?} dict. A string-typed
    # parameter may additionally declare ``choices: [..]`` (a list of allowed
    # string values); the dashboard / bigraph-loom Configure form renders such
    # a parameter as a dropdown instead of a free-text input. ``choices`` is
    # passed through to the config_schema verbatim (see composite_discovery's
    # ``discover_all``), so no special handling is needed beyond declaring it.
    func: Callable[..., dict]
    module: str
    default_n_steps: int | None = None  # framework-owned runtime knob; UI pre-fill
    # Canonical visualizations that ship with this composite. Each entry is
    # a Study-spec visualization dict ({name, address, config, ...}). When
    # a Study is built on top of this composite the dashboard merges these
    # defaults into its visualizations list; Studies can still declare extras.
    visualizations: list[dict] = field(default_factory=list)
    # Emitter(s) this composite ships as its default observation sink. Each
    # entry is a lightweight ``{address, config, paths?}`` dict: ``address`` is
    # the registered emitter link (e.g. ``"local:ParquetEmitter"``), ``config``
    # is the base config the workspace merges into the emitter step, and the
    # optional ``paths`` lists dotted observable store-paths to wire. Unlike a
    # full process-node spec, the emit-schema + topology are left for the
    # generator/runner to compute — this declaration only selects which
    # emitter(s) to install and with what base config. Parallel to
    # ``visualizations``: when present, a workspace that builds this composite
    # standalone (no dashboard observable-injection) uses these as its default
    # emitter set, so e.g. a parquet sink travels with the generator instead of
    # being toggled by external override globals. See ``emitter_defaults``.
    emitters: list[dict] = field(default_factory=list)
    # Callables ``(core) -> core | None`` that register the custom types /
    # processes this generator's document references but that a bare
    # ``build_core()`` doesn't know about. v2ecoli friction #16 (2026-05-19):
    # the dashboard runs each composite in a subprocess that calls the
    # workspace's ``build_core()``; when a composite uses types registered by
    # a *different* package (e.g. ``map[pymunk_agent]`` from ``viva_munk``),
    # the subprocess core never gets those registrations and the Composite
    # build dies with "cannot resolve types … pymunk_agent". Declaring the
    # package's ``register_*`` functions here lets the runner apply them to
    # the right core. See ``apply_core_extensions``.
    core_extensions: list[Callable[[Any], Any]] = field(default_factory=list)


def _entry_for(spec) -> GeneratorEntry:
    """Adapt a process-bigraph CompositeSpec to the GeneratorEntry the dashboard reads."""
    return GeneratorEntry(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        func=(spec.builder if callable(spec.builder)
              else (_cs._resolve_builder(spec.builder, spec.module) if spec.builder else None)),
        module=spec.module,
        default_n_steps=spec.default_n_steps,
        visualizations=spec.visualizations,
        emitters=spec.emitters,
        core_extensions=spec.core_extensions,
    )


class _RegistryView:
    """Live view over the process-bigraph CompositeSpec registry.

    Returns ``GeneratorEntry`` objects so the dashboard's attribute surface
    (``.id``, ``.name``, ``.func``, …) is preserved.  Caches entries by
    spec-id so identity (``is``) comparisons survive across separate
    ``__getitem__`` calls — required by the ``_composite_generator_entry is
    entry`` sidecar test.
    """

    def __init__(self) -> None:
        self._cache: dict[str, GeneratorEntry] = {}

    def _entry(self, spec_id: str) -> GeneratorEntry | None:
        s = _cs.get(spec_id)
        if s is None:
            return None
        if spec_id not in self._cache:
            self._cache[spec_id] = _entry_for(s)
        return self._cache[spec_id]

    def get(self, k: str, default=None):
        e = self._entry(k)
        return e if e is not None else default

    def __getitem__(self, k: str) -> GeneratorEntry:
        e = self._entry(k)
        if e is None:
            raise KeyError(k)
        return e

    def __contains__(self, k: object) -> bool:
        return _cs.get(str(k)) is not None

    def values(self):
        return [self._entry(sid) for sid in _cs.all_specs()]

    def items(self):
        return [(sid, self._entry(sid)) for sid in _cs.all_specs()]

    def __iter__(self):
        return iter(_cs.all_specs())

    def keys(self):
        return list(_cs.all_specs().keys())

    def __bool__(self) -> bool:
        return bool(_cs.all_specs())

    def __setitem__(self, key, entry):
        # v2ecoli registers "clean alias" composites via
        #   _REGISTRY[clean_id] = dataclasses.replace(orig, id=clean_id)
        # where `orig` is a GeneratorEntry from this view. Translate the assigned
        # entry into a CompositeSpec registered under `key` in the unified registry.
        import dataclasses as _dc
        # invalidate any cached GeneratorEntry for this key before re-registering
        if hasattr(self, "_cache"):
            self._cache.pop(key, None)
        if isinstance(entry, _cs.CompositeSpec):
            spec = entry if entry.id == key else _dc.replace(entry, id=key)
        else:  # a GeneratorEntry (or replace()-d copy of one)
            spec = _cs.CompositeSpec(
                id=key,
                name=entry.name,
                description=entry.description,
                parameters=dict(entry.parameters or {}),
                builder=getattr(entry, "func", None),
                module=getattr(entry, "module", ""),
                default_n_steps=getattr(entry, "default_n_steps", None),
                visualizations=list(getattr(entry, "visualizations", []) or []),
                emitters=list(getattr(entry, "emitters", []) or []),
                core_extensions=list(getattr(entry, "core_extensions", []) or []),
            )
        _cs.register(spec)

    def __len__(self):
        return len(_cs.all_specs())

    def clear(self) -> None:
        """Clear the cache AND the backing process-bigraph registry."""
        self._cache.clear()
        _cs.clear_registry()


# Process-level registry.  Backed by process_bigraph.composite_spec._REGISTRY;
# populated by @composite_generator on import.
_REGISTRY: _RegistryView = _RegistryView()


def composite_generator(
    *,
    name: str,
    description: str = "",
    parameters: dict[str, dict] | None = None,
    visualizations: list[dict] | None = None,
    emitters: list[dict] | None = None,
    default_n_steps: int | None = None,
    core_extensions: list[Callable[[Any], Any]] | None = None,
    default_state_ref: str | None = None,
) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Decorator: register a doc-building function.

    The wrapped function must accept ``(core=None, **kwargs) -> dict`` and
    return a process-bigraph state document (or a {state, schema} envelope).

    Delegates registration to ``process_bigraph.composite_spec`` so that the
    process-bigraph registry is the single source of truth.  The
    ``GeneratorEntry`` view (read by the dashboard) is built lazily from the
    registered ``CompositeSpec`` on first access via ``_REGISTRY``.

    `parameters` declares each kwarg in the same shape that *.composite.yaml
    uses, so the dashboard's parameter-form code is shared across both
    conventions.  Parameter ``type`` values are normalised to canonical
    vocabulary (``int``→``integer``, ``float``→``float``, etc.) by
    ``CompositeSpec.__post_init__``.

    `visualizations` declares the canonical visualization set that ships with
    this composite. Each entry is a Study-spec visualization dict
    (``{name, address, config, ...}``). The dashboard merges these defaults
    into a Study's visualizations list when the Study is built on this
    composite, so callers get the v2ecoli simulation report (or whatever the
    composite author considers canonical) without having to hand-author them
    in every Study spec.

    `emitters` (optional) declares the default observation sink(s) this
    composite ships with. Each entry is a lightweight
    ``{"address": "local:ParquetEmitter", "config": {...}, "paths": [...]}``
    dict — ``address`` selects the registered emitter link, ``config`` is the
    base config merged into the emitter step, and the optional ``paths`` lists
    dotted observable store-paths to wire. The emit-schema and topology are
    NOT part of this declaration; the generator/runner computes them. This is
    the standalone analogue of the dashboard's run-time observable injection:
    when a workspace builds the composite outside the Investigations flow, it
    reads these defaults (via :func:`emitter_defaults`) so the composite still
    has a sink. External override mechanisms a workspace may keep (e.g.
    v2ecoli's ``set_parquet_emitter_override``) take precedence; the declared
    default fills in when none is set. Example::

        @composite_generator(
            name="baseline",
            emitters=[{
                "address": "local:ParquetEmitter",
                "config": {"out_dir": "out/parquet"},
            }],
        )
        def baseline(core=None): ...

    `default_n_steps` (optional) is a UI hint for the Composite Explorer's
    ``steps`` pre-fill. It is NOT a composite-builder kwarg — runtime knobs
    are framework-owned and live next to the generator entry.

    `default_state_ref` (optional) path to a pre-computed default-state
    artifact relative to the workspace root.  When present, ``CompositeSpec``
    can serve the state without re-running the builder.

    `core_extensions` (optional) is a list of callables ``(core) -> core | None``
    that register the custom types/processes this generator's document
    references but that a bare ``build_core()`` doesn't provide. Declare a
    package's ``register_*`` functions here so the dashboard's subprocess
    runner can apply them to the core it actually runs against — see
    ``apply_core_extensions`` and v2ecoli friction #16. Example::

        from viva_munk import register_pymunk_types, register_processes

        @composite_generator(
            name="attachment",
            core_extensions=[register_pymunk_types, register_processes],
        )
        def attachment(core=None): ...
    """
    # Validate emitters at decoration time (not first use) so malformed
    # declarations fail loudly on import — same guarantee as before.
    validated_emitters = _validate_emitters(emitters, name)

    def decorate(fn: Callable[..., dict]) -> Callable[..., dict]:
        # Register with process-bigraph as the single source of truth.
        _cs.composite_spec(
            name=name,
            description=description,
            parameters=parameters,
            visualizations=visualizations,
            emitters=validated_emitters,
            default_n_steps=default_n_steps,
            core_extensions=core_extensions,
            default_state_ref=default_state_ref,
        )(fn)
        # Build + cache the GeneratorEntry NOW so that identity comparisons
        # (``fn._composite_generator_entry is _REGISTRY[spec_id]``) hold.
        spec_id = f"{fn.__module__}.{name}"
        entry = _REGISTRY[spec_id]          # creates + caches the GeneratorEntry
        fn._composite_generator_entry = entry   # introspection sidecar
        return fn

    return decorate


def _validate_emitters(emitters: list[dict] | None, name: str) -> list[dict]:
    """Normalise + sanity-check the decorator's ``emitters`` declaration.

    Each entry must be a dict with a non-empty string ``address``. ``config``,
    when present, must be a dict; ``paths``, when present, must be a list of
    strings. We validate at decoration time (not first use) so a malformed
    declaration fails loudly on import — the same place a bad ``parameters``
    block would. Returns a fresh list of copied dicts so later mutation of the
    caller's literal can't leak into the registry.
    """
    out: list[dict] = []
    for i, em in enumerate(emitters or []):
        where = f"{name!r} emitters[{i}]"
        if not isinstance(em, dict):
            raise ValueError(f"{where}: each emitter must be a dict, got {type(em).__name__}")
        address = em.get("address")
        if not isinstance(address, str) or not address:
            raise ValueError(f"{where}: 'address' must be a non-empty string")
        config = em.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"{where}: 'config' must be a dict")
        paths = em.get("paths")
        if paths is not None and not (
            isinstance(paths, list) and all(isinstance(p, str) for p in paths)
        ):
            raise ValueError(f"{where}: 'paths' must be a list of strings")
        out.append(dict(em))
    return out


def emitter_defaults(fn_or_entry: Any) -> list[dict]:
    """Return the declared default emitter(s) for a generator OR static spec.

    Accepts a decorated generator function (reads its
    ``_composite_generator_entry`` sidecar), a :class:`GeneratorEntry`, or a
    parsed static composite-spec ``dict`` (reads its top-level ``emitters:``
    key — the static-spec analogue of the decorator's ``emitters=``). Returns
    the (possibly empty) ``emitters`` list — a workspace builds the composite's
    default sink from this when it isn't running under the dashboard's
    observable-injection flow. Returns ``[]`` for anything that declares none,
    so callers can use it unconditionally.
    """
    if isinstance(fn_or_entry, dict):  # parsed static composite spec
        return list(fn_or_entry.get("emitters") or [])
    entry = getattr(fn_or_entry, "_composite_generator_entry", fn_or_entry)
    return list(getattr(entry, "emitters", []) or [])


def _emitter_node_from_decl(decl: dict, *, run_id: str | None = None,
                            out_dir: Any = None, registered=None,
                            fallback: str = "local:RAMEmitter") -> dict:
    """Materialise one ``emitters=`` declaration into a process-bigraph step node.

    The declaration is the lightweight ``{address, config?, paths?}`` form; the
    emit-schema + topology are computed here (they depend on the composite's
    shape, so they aren't part of the declaration). Each ``paths`` entry (slash-
    or dot-joined) becomes one ``config.emit`` column wired to that store;
    ``global_time`` is always emitted so trajectories have a time axis and the
    Step re-fires every tick. When the declared ``address`` isn't in
    ``registered`` (the core's link registry), it degrades to ``fallback`` so
    the composite still builds — the convention's RAMEmitter safety net.
    """
    address = decl.get("address") or fallback
    name = address.split(":", 1)[-1]
    if registered is not None and name not in registered:
        address = fallback
        name = address.split(":", 1)[-1]

    emit_schema: dict = {}
    inputs: dict = {}
    for p in decl.get("paths") or []:
        parts = [seg for seg in str(p).replace(".", "/").split("/") if seg]
        if not parts:
            continue
        key = "_".join(parts)
        emit_schema[key] = "node"
        inputs[key] = parts
    if "global_time" not in inputs:
        inputs["global_time"] = ["global_time"]
        emit_schema["global_time"] = "node"

    config = dict(decl.get("config") or {})
    config["emit"] = emit_schema
    # Run-specific layering for hive-partitioned parquet sinks: a per-run out_dir
    # + experiment_id partition. Applied to any *ParquetEmitter (the generic
    # ParquetEmitter and workspace variants like DataFrameParquetEmitter), which
    # understand these keys; other sinks keep their declared base config.
    if name.endswith("ParquetEmitter"):
        if out_dir is not None:
            config["out_dir"] = str(out_dir)
        if run_id is not None:
            config.setdefault("partitioning_keys", ["experiment_id"])
            md = dict(config.get("metadata") or {})
            md.setdefault("experiment_id", run_id)
            config["metadata"] = md

    return {"_type": "step", "address": address, "config": config, "inputs": inputs}


def install_default_emitters(state: dict, source: Any, *, run_id: str | None = None,
                             out_dir: Any = None, core: Any = None) -> dict:
    """Return a copy of ``state`` with the composite's declared default
    emitter(s) installed as ``emitter`` / ``emitter_<i>`` step nodes.

    ``source`` is a generator fn/entry or a parsed static-spec dict — whatever
    :func:`emitter_defaults` understands. This is the convention's install step:
    a composite built outside the dashboard's observable-injection flow still
    gets its declared sink. Resolution order (declared → RAMEmitter fallback) is
    handled per-node by :func:`_emitter_node_from_decl`; external overrides are
    layered by the caller before this call.

    ``run_id`` / ``out_dir`` are run-specific parquet parameters (ignored by
    non-parquet sinks). ``core`` (when given) lets the installer degrade an
    unregistered declared address to RAMEmitter. Returns ``state`` unchanged
    when nothing is declared, so callers can invoke it unconditionally.
    """
    decls = emitter_defaults(source)
    if not decls:
        return dict(state)
    registered = getattr(core, "link_registry", None) if core is not None else None
    new_state = dict(state)
    for i, decl in enumerate(decls):
        if not isinstance(decl, dict):
            continue
        key = "emitter" if i == 0 else f"emitter_{i}"
        new_state[key] = _emitter_node_from_decl(
            decl, run_id=run_id, out_dir=out_dir, registered=registered)
    return new_state


def apply_core_extensions(entry: GeneratorEntry, core: Any) -> Any:
    """Run ``entry.core_extensions`` against ``core``; return the final core.

    Each extension is a callable ``(core) -> core | None`` that registers
    custom types/processes (e.g. ``viva_munk.register_pymunk_types``). By the
    ``register_types`` convention an extension may return a (possibly new)
    core; when it returns ``None`` we keep the one we passed in.

    Failures are **not** swallowed. A missing registration is exactly the
    kind of silent gap v2ecoli friction #16 is about — letting the exception
    propagate surfaces it (with the offending function name) instead of
    deferring to a cryptic "cannot resolve types" error at Composite-build
    time.
    """
    for ext in entry.core_extensions or []:
        result = ext(core)
        if result is not None:
            core = result
    return core


def build_generator(entry, overrides=None, core=None):
    """Delegate to the CompositeSpec document. ``entry`` may be a CompositeSpec,
    a GeneratorEntry (has ``.id``/``.func``), or a registered id's entry.

    Unknown override keys raise ``ValueError`` so dashboards / callers can't
    silently smuggle in parameters that the generator doesn't declare.
    (``CompositeSpec._merged_params`` raises ``KeyError`` for unknown keys; we
    intercept here so callers always get ``ValueError``.)
    """
    # Validate overrides before delegating so callers always get ValueError
    # (not KeyError from _merged_params) for unknown parameters.
    overrides = overrides or {}
    params = getattr(entry, "parameters", {}) or {}
    unknown = set(overrides) - set(params)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) for {getattr(entry, 'id', '?')}: {sorted(unknown)}"
        )
    if isinstance(entry, _cs.CompositeSpec):
        spec = entry
    else:
        spec = _cs.get(getattr(entry, "id", None))
        if spec is None and callable(getattr(entry, "func", None)):
            # out-of-registry GeneratorEntry (e.g. registry cleared after decoration):
            # rebuild a transient spec from it so we preserve "build from the entry".
            spec = _cs.CompositeSpec(
                id=entry.id, name=entry.name, builder=entry.func,
                parameters=entry.parameters, module=entry.module,
            )
    if spec is None:
        raise ValueError(
            f"build_generator: no registered composite for "
            f"{getattr(entry, 'id', entry)!r}")
    return spec.to_document(overrides, core=core)


def _import_bigraph_packages(extra_packages: list[str] | None = None) -> None:
    """Walk installed bigraph-schema-dependent packages and import them so
    that ``@composite_generator`` decorators fire and register their specs.

    This is the distribution-walking body extracted from ``discover_generators``
    so it can be called independently (e.g. from ``composite_spec.discover_specs``
    via the shim's ``discover_generators``).
    """
    import importlib
    import importlib.metadata as md

    extra_packages = extra_packages or []
    targets: set[str] = set(extra_packages)

    for dist in md.distributions():
        deps = dist.requires or []
        if not any("bigraph-schema" in (d or "") for d in deps):
            continue
        # Find the importable top-level packages for this distribution.
        # Prefer top_level.txt when present (wheel installs); fall back to
        # the normalised package name (hyphens → underscores) for editable
        # installs built with hatchling / PEP 660, which omit top_level.txt.
        top_level_txt = dist.read_text("top_level.txt") or ""
        mods_from_txt = [
            line.strip()
            for line in top_level_txt.splitlines()
            if line.strip() and not line.strip().startswith("_")
        ]
        if mods_from_txt:
            targets.update(mods_from_txt)
        else:
            dist_name = dist.metadata.get("Name") or ""
            fallback = dist_name.replace("-", "_")
            if fallback and not fallback.startswith("_"):
                targets.add(fallback)

    import pkgutil
    import warnings

    for mod_name in sorted(targets):
        try:
            top = importlib.import_module(mod_name)
        except Exception as e:  # noqa: BLE001 — skip any unimportable package
            warnings.warn(
                f"discover_generators: skipping {mod_name!r}: "
                f"{type(e).__name__}: {e}",
                stacklevel=2,
            )
            continue
        # mem3dg-readdy friction #22: @composite_generator decorators
        # only fire when their containing module is imported. Importing
        # the top-level package alone misses subpackages like
        # `pbg_<ws>/composites/__init__.py` unless the top-level
        # __init__.py eagerly does `from . import composites`. Walk the
        # subpackage tree so workspaces don't have to remember.
        pkg_path = getattr(top, "__path__", None)
        if not pkg_path:
            continue  # single-file module — nothing to walk

        # v2ecoli friction #4: skip subpaths that look like CLI scripts
        # (e.g. `<pkg>.scripts.compare_runs`). Discovery should walk the
        # library, not the CLI tool layer — and CLI scripts commonly have
        # module-level `sys.exit()` / argparse / etc. that crash under
        # import.
        #
        # Also skip `tests` subpackages. process-bigraph itself declares
        # `bigraph-schema` as a dependency, so its own distribution is
        # always one of `_import_bigraph_packages`'s walk targets — and in
        # this repo (unlike viva-superpowers, whose tests/ sits *outside*
        # the viva-superpowers/ package) tests live *inside* the package at
        # `process_bigraph/tests/`. Without this skip, every call to
        # `discover_generators()` — from any test, in any installed package
        # that colocates tests this way — would import the whole test suite
        # as a side effect: firing module-level decorators (registry
        # pollution) and running expensive/stateful test-only code (e.g.
        # subprocess-driven fixtures) purely as a byproduct of composite
        # discovery.
        #
        # This has to be a pre-descent filter, not a post-hoc skip of
        # pkgutil.walk_packages's yielded items: walk_packages auto-imports
        # (and recurses into) every package-like entry it finds *before*
        # a caller's loop body gets a chance to react, so a `continue` on
        # the yielded `<pkg>.tests` item does not stop it from already
        # having imported `<pkg>.tests` and descended into e.g.
        # `<pkg>.tests.fixtures.some_pkg`. Walking the tree ourselves lets
        # us drop an excluded subpackage before it (or anything beneath it)
        # is ever imported.
        _SKIP_SEGMENTS = ("scripts", "tests")

        def _walk(path, prefix):
            for _finder, name, is_pkg in pkgutil.iter_modules(path, prefix=prefix):
                if name.rsplit(".", 1)[-1] in _SKIP_SEGMENTS:
                    continue
                try:
                    mod = importlib.import_module(name)
                except SystemExit as e:  # noqa: BLE001
                    # `sys.exit(N)` at module level is NOT a subclass of
                    # Exception; without this branch it would propagate out
                    # of discover_generators and crash the dashboard.
                    # v2ecoli's `scripts/compare-runs.py` had a top-level
                    # sys.exit(0) that took the whole subprocess down before
                    # this catch.
                    warnings.warn(
                        f"discover_generators: subpackage {name!r} called "
                        f"sys.exit({e.code!r}) at import time; skipping. "
                        "Wrap top-level CLI logic in `if __name__ == \"__main__\":`.",
                        stacklevel=2,
                    )
                    continue
                except Exception as e:  # noqa: BLE001
                    warnings.warn(
                        f"discover_generators: skipping subpackage {name!r}: "
                        f"{type(e).__name__}: {e}",
                        stacklevel=2,
                    )
                    continue
                if is_pkg:
                    sub_path = getattr(mod, "__path__", None)
                    if sub_path:
                        _walk(sub_path, name + ".")

        _walk(pkg_path, mod_name + ".")


def discover_generators(
    extra_packages: list[str] | None = None,
) -> dict[str, GeneratorEntry]:
    """Discover composite generators from installed packages.

    Walks every installed distribution that depends on ``bigraph-schema``,
    imports each top-level package so its ``@composite_generator`` decorators
    fire, then returns whatever generator-kind specs ended up in the
    process-bigraph registry as ``GeneratorEntry`` views.

    Unlike ``discover_composites`` (file-glob; imports only to resolve
    package paths, not to run decorator side-effects), this MUST import
    the host packages so the decorators fire. Subsequent calls return the
    same registry; there is no automatic invalidation. Hot-reload callers
    can ``_REGISTRY.clear()`` before re-importing.
    """
    _import_bigraph_packages(extra_packages)
    return {
        sid: _entry_for(s)
        for sid, s in _cs.all_specs().items()
        if s.kind == "generator"
    }
