"""Content-addressed study artifacts.

A study node produces an **artifact**, and some artifacts are trajectories.
That is the generalization ``sim_data`` forces: a simulation's ``results`` is a
zarr trajectory, but a calibration object is not a trajectory at all, and both
are things a downstream study consumes through the same ``results`` port.

Three pieces:

- the **content address** — ported verbatim from the workbench's artifact store
  so both sides agree on where an artifact lives (see the lock-step note);
- :class:`ArtifactRef` — a typed reference: *what kind*, *which address*,
  *where on disk*;
- :class:`ArtifactResults` — the resolved handle a ``results`` site carries,
  dispatching ``resolve()`` on kind. ``EmitterResults`` is the ``trajectory``
  case and keeps its own API.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

# ── content address ─────────────────────────────────────────────────
#
# LOCK-STEP: this is a hand-port of
# ``vivarium_workbench/lib/artifacts/hashing.py``. process-bigraph cannot
# import the workbench (the dependency runs the other way), so the formula is
# duplicated deliberately — the same two-writers/one-store arrangement the
# reproducible-rerun spine already has. **Any change to either copy must be
# made to both**, or the workbench pipeline and the engine will disagree about
# where an artifact lives and every cache lookup will silently miss.

ARTIFACT_ROOT = '.pbg/artifacts'
"""Store convention: an artifact lives under ``.pbg/artifacts/<hash>/``."""


def _stable(o):
    if isinstance(o, float) and o.is_integer():
        return int(o)
    raise TypeError(type(o))


def canonical(config: dict) -> str:
    """The canonical JSON encoding a content address is computed over."""
    return json.dumps(
        config or {}, sort_keys=True, separators=(",", ":"), default=_stable)


def artifact_id(*, composite_id: str, config: dict,
                input_ids: list, commit: str) -> str:
    """The content address of an artifact.

    A function of *what produced it* and *what went into it* — never of a
    filesystem path, so two worktrees at the same commit resolve the same
    address and share a cache with no symlink between them.
    """
    payload = "\n".join([composite_id, canonical(config),
                         "\n".join(sorted(input_ids or [])), commit or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def artifact_store(address: str, root: str = ARTIFACT_ROOT) -> str:
    """Where the artifact at ``address`` lives."""
    return os.path.join(root, address)


def artifact_exists(address: str, root: str = ARTIFACT_ROOT) -> bool:
    """Is this artifact already in the store?"""
    return os.path.isdir(artifact_store(address, root))


FINGERPRINT_FILE = 'fingerprint'
"""Where a stored artifact records what it produced, next to the payload."""


def write_fingerprint(address: str, value: str,
                      root: str = ARTIFACT_ROOT) -> str:
    """Record an artifact's fingerprint in its store entry.

    Producers call this after writing the payload. Without it the address is
    the *only* thing a cache hit is checked against — and an address is a hash
    of inputs, which says what *should* come out, never what did.
    """
    store = artifact_store(address, root)
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, FINGERPRINT_FILE)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(value)
    return path


def read_fingerprint(address: str, root: str = ARTIFACT_ROOT) -> str:
    """The recorded fingerprint, or '' when the producer wrote none.

    Absent is not a failure: artifacts stored before fingerprinting, and
    producers that opt out, simply cannot be checked — and
    :func:`check_fingerprint` treats an empty stored value as "no claim".
    """
    try:
        with open(os.path.join(artifact_store(address, root),
                               FINGERPRINT_FILE), encoding='utf-8') as handle:
            return handle.read().strip()
    except OSError:
        return ''


# ── typed references ────────────────────────────────────────────────

TRAJECTORY = 'trajectory'
SIM_DATA = 'sim_data'
FIGURES = 'figures'


@dataclass
class ArtifactRef:
    """A typed, content-addressed reference to a study's output."""

    kind: str
    hash: str = ''
    store: str = ''
    context: dict = field(default_factory=dict)
    fingerprint: str = ''

    @classmethod
    def coerce(cls, ref) -> 'ArtifactRef':
        if isinstance(ref, ArtifactRef):
            return ref
        if isinstance(ref, dict):
            known = {'kind', 'hash', 'store', 'context', 'fingerprint'}
            return cls(**{k: v for k, v in ref.items() if k in known})
        raise TypeError(f'not an artifact reference: {ref!r}')

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'hash': self.hash, 'store': self.store,
                'context': dict(self.context), 'fingerprint': self.fingerprint}


# ── loaders: one per kind ───────────────────────────────────────────

_LOADERS: dict = {}


def register_artifact_loader(kind: str, loader: Callable) -> None:
    """Teach ``ArtifactResults`` how to resolve a kind.

    ``loader(ref) -> value``. Kinds are open on purpose: a workspace adds
    ``sim_data`` by registering the loader that knows its calibration format,
    without the engine knowing anything about it.
    """
    _LOADERS[kind] = loader


def artifact_loader(kind: str) -> Callable:
    loader = _LOADERS.get(kind)
    if loader is None:
        raise KeyError(
            f'no loader registered for artifact kind {kind!r} — '
            f'known kinds: {sorted(_LOADERS)}. Register one with '
            f'`register_artifact_loader({kind!r}, ...)`.')
    return loader


def _load_trajectory(ref: ArtifactRef):
    """Read a trajectory back from its store.

    The default understands a zarr directory via xarray, which is what the
    emitters write; a workspace with another format registers its own.
    """
    import xarray as xr
    return xr.open_datatree(ref.store, engine='zarr')


def _load_pickle(ref: ArtifactRef):
    """Load a calibration object (the shape ParCa writes).

    A stand-in: it reads the single pickle in the artifact's store. The real
    v2ecoli loader replaces it by registering over this kind.
    """
    import pickle
    from pathlib import Path

    store = Path(ref.store)
    candidate = store if store.is_file() else None
    if candidate is None:
        pickles = sorted(store.glob('*.pkl')) + sorted(store.glob('*.pickle'))
        if not pickles:
            raise FileNotFoundError(
                f'no calibration object under {store} for artifact '
                f'{ref.hash!r}')
        candidate = pickles[0]
    with open(candidate, 'rb') as handle:
        return pickle.load(handle)


register_artifact_loader(TRAJECTORY, _load_trajectory)
register_artifact_loader(SIM_DATA, _load_pickle)


# ── the resolved handle ─────────────────────────────────────────────

class ArtifactResults:
    """The handle a ``results`` site carries for a **stored** artifact.

    The counterpart of ``EmitterResults``: that one references a live
    emitter, this one references the store. Both answer ``kind``, ``context``
    and ``resolve()``, so a downstream consumer cannot tell whether the study
    ran or was pulled — which is the whole point of caching a study.
    """

    def __init__(self, ref, context=None):
        self.ref = ArtifactRef.coerce(ref)
        if context:
            self.ref.context = {**self.ref.context, **context}
        self._resolved = {}
        self.provenance_status = 'ok'

    @classmethod
    def from_ref(cls, ref) -> 'ArtifactResults':
        return cls(ref)

    @property
    def kind(self) -> str:
        return self.ref.kind

    @property
    def context(self) -> dict:
        return self.ref.context

    @property
    def hash(self) -> str:
        return self.ref.hash

    @property
    def store(self) -> str:
        return self.ref.store

    def resolve(self, paths=None):
        """Load the artifact. Memoized, like ``EmitterResults.resolve``: a
        stored artifact does not change under us, and several flush entities
        resolve the same handle."""
        key = tuple(paths) if isinstance(paths, list) else paths
        if key not in self._resolved:
            self._resolved[key] = artifact_loader(self.kind)(self.ref)
        return self._resolved[key]

    def to_dict(self) -> dict:
        return {**self.ref.to_dict(),
                'provenance_status': self.provenance_status}

    def __repr__(self) -> str:
        return (f'ArtifactResults(kind={self.kind!r}, hash={self.hash!r}, '
                f'status={self.provenance_status!r})')


# ── determinism ─────────────────────────────────────────────────────

def fingerprint_of(value) -> str:
    """A cheap, stable fingerprint of a produced result.

    Used to catch the unsound case: a content address is a hash of *inputs*,
    so it is only a valid cache key when the producer is deterministic given
    them. A recompute that disagrees with the stored fingerprint means it is
    not.
    """
    return hashlib.sha256(
        canonical(value if isinstance(value, dict) else {'value': value})
        .encode('utf-8')).hexdigest()[:16]


def check_fingerprint(results, observed: str) -> str:
    """Compare a recomputed fingerprint against the stored one.

    Returns (and records) ``'ok'`` or ``'nondeterministic'``. A divergence is
    *flagged*, never silently served — the address said these results should
    be identical and they were not.
    """
    stored = getattr(getattr(results, 'ref', None), 'fingerprint', '')
    status = 'ok' if (not stored or stored == observed) else 'nondeterministic'
    results.provenance_status = status
    return status


# ── admitting kinds at a `results` site ─────────────────────────────

def results_kind(filler) -> str:
    """The artifact kind a filler carries, or '' if it is not a handle."""
    return getattr(filler, 'kind', '') or ''


def register_artifact_sorts(core, kinds=(TRAJECTORY, SIM_DATA, FIGURES)):
    """Register the sorts a ``results`` site is constrained by.

    ``'results'`` admits any handle; ``'results_<kind>'`` admits only that
    kind — so a study declaring it needs ``sim_data`` cannot be wired to a
    trajectory.

    The names use an underscore, not a colon: ``_sort`` is a schema field, so
    ``access`` runs it through the type grammar, where ``a:b`` is the
    named-parameter form and would be parsed rather than kept as a name.
    """
    core.register_sort('results', lambda core, site, filler: bool(
        results_kind(filler)))
    for kind in kinds:
        core.register_sort(
            results_sort(kind),
            (lambda wanted: (lambda core, site, filler:
                             results_kind(filler) == wanted))(kind))
    return core


def results_sort(kind: str) -> str:
    """The sort name constraining a ``results`` site to one artifact kind."""
    return f'results_{kind}'
