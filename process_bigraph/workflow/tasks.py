"""``CompositeTask`` — scatter a whole-Composite build over a parameter axis.

The heart of the workflow engine: each match in an incoming ``per_match``
match-set becomes one ``python -m process_bigraph.run_composite --build``
subprocess, content-addressed by :func:`process_bigraph.artifacts.artifact_id`
so a rerun with identical inputs is a cache hit rather than a recompute.

Threads, not processes, drive the fan-out (:class:`ThreadPoolExecutor`):
each unit of work is a blocking ``subprocess.run`` call, so a Python thread
(which releases the GIL while blocked in the subprocess) is exactly the
right concurrency primitive — no pickling, no process-pool startup cost.

Cache correctness is the point of this Step, so the two rules that matter
most:

- The cache key (F1) is a function of *everything that can change the
  output*: the generator's overrides (with the scatter value merged in),
  ``steps``, ``provision``, the hashes of any input artifacts, and
  ``code_version``. Miss any of those and two runs that should be
  distinguished collide on one cache entry; over-include and a harmless
  rerun (e.g. touching an unrelated file) forces a wasted recompute — the
  formula here is deliberately exactly what :func:`build_from_recipe`
  consumes to produce the document, no more.
- The skip decision (F2) is **not** ``check_fingerprint`` — that function
  compares a *freshly computed* fingerprint against a *stored* one (an
  attestation of determinism), which requires the recompute it exists to
  avoid. A cache lookup is just: does the address exist, and is the
  payload it names still on disk.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from process_bigraph.artifacts import (
    ArtifactRef, artifact_exists, artifact_id, fingerprint_of,
    write_fingerprint,
)
from process_bigraph.composite import Step, SyncUpdate

_log_prefix = '[CompositeTask]'


# ── emitter guard (F4) ──────────────────────────────────────────────

_IN_MEMORY_EMITTER_DEFAULT = 'local:RAMEmitter'


def _resolve_emitter_address(entry) -> str:
    """The emitter address this generator resolves to, without building it.

    Reads the generator's declared ``emitters=[...]`` (see
    ``composite_generator``'s ``emitter_defaults``). A generator that
    declares none falls back to the framework's own silent default —
    ``local:RAMEmitter`` (see ``process_bigraph.emitter`` module docs) — so
    an undeclared generator is treated as in-memory, the conservative
    (safe) reading.
    """
    from process_bigraph.composite_generator import emitter_defaults

    declared = emitter_defaults(entry)
    if declared:
        return declared[0].get('address', '') or _IN_MEMORY_EMITTER_DEFAULT
    return _IN_MEMORY_EMITTER_DEFAULT


def _is_in_memory_emitter(address: str) -> bool:
    if not address:
        return True
    tail = address.rsplit(':', 1)[-1]
    return tail in ('RAMEmitter',) or address.startswith('memory:')


def _guard_emitter(entry, allow_in_memory_emitter: bool) -> None:
    address = _resolve_emitter_address(entry)
    if _is_in_memory_emitter(address) and not allow_in_memory_emitter:
        raise ValueError(
            f"{_log_prefix} generator {entry.name!r} resolves to an "
            f"in-memory emitter ({address!r}); its output would vanish "
            f"when the run_composite subprocess exits, so it cannot be "
            f"cached or read back by a downstream node. Declare a "
            f"file-backed emitter (e.g. `emitters=[{{'address': "
            f"'local:JSONEmitter', ...}}]` on @composite_generator) or "
            f"pass allow_in_memory_emitter=True to proceed anyway "
            f"(results are then only visible for the life of the "
            f"subprocess and every match is an unconditional cache miss "
            f"in practice)."
        )


# ── generator introspection (mirrors workflow.recipe's shim logic) ──

def _accepts_param(func, name: str) -> bool:
    if func is None:
        return False
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _generator_declares(entry, name: str) -> bool:
    """Whether ``entry``'s generator accepts a ``name`` kwarg (declared or inferred)."""
    if name in (entry.parameters or {}):
        return True
    return _accepts_param(entry.func, name)


def _resolve_entry(generator_name: str, import_modules):
    from process_bigraph.composite_generator import _REGISTRY

    for mod_name in import_modules or []:
        importlib.import_module(mod_name)

    matches = [e for e in _REGISTRY.values() if e.name == generator_name]
    if not matches:
        raise ValueError(
            f"{_log_prefix}: no registered generator named {generator_name!r} "
            f"(imported {list(import_modules or [])!r})")
    return matches[0]


# ── code_version default (part of the cache key, F1) ────────────────

def _default_code_version(import_modules) -> str:
    """``process_bigraph.__version__`` plus the first import module's
    installed-distribution version, when resolvable.

    Best-effort: a bare test module or a package with no installed
    distribution metadata simply contributes nothing beyond the
    process_bigraph version — never fatal, since ``code_version`` is
    diagnostic (bump it explicitly via config when precision matters).
    """
    import process_bigraph as _pbg

    parts = [str(getattr(_pbg, '__version__', '0'))]
    for mod_name in import_modules or []:
        top = mod_name.split('.', 1)[0]
        try:
            parts.append(importlib.metadata.version(top))
        except importlib.metadata.PackageNotFoundError:
            pass
        break
    return '+'.join(parts)


# ── the Step ──────────────────────────────────────────────────────────

class CompositeTask(Step):
    """Scatter a whole-Composite build over a parameter axis.

    Each match in the ``scatter_param``-named ``per_match`` input becomes
    one ``run_composite --build`` subprocess (unless the content-addressed
    cache already has it), fanned out across a bounded
    :class:`ThreadPoolExecutor`.
    """

    _cache = 'none'  # own content-hash cache (F1/F2) replaces the generic by-hash Step cache

    config_schema = {
        'generator': 'string',
        'import': 'list[string]',
        'overrides': 'node',
        'artifact_params': 'map[string]',
        'scatter_param': 'maybe[string]',
        'steps': 'float',
        'provision': 'list[string]',
        'code_version': 'maybe[string]',
        'max_workers': 'maybe[integer]',
        'artifact_root': 'maybe[string]',
        'allow_in_memory_emitter': 'boolean',
    }

    _SOLO_PORT = 'match'  # scatter-port name used only when scatter_param is unset

    # ── ports ────────────────────────────────────────────────────────

    def _scatter_port_name(self) -> str:
        return self.config.get('scatter_param') or self._SOLO_PORT

    def inputs(self):
        ports: Dict[str, Any] = {
            self._scatter_port_name(): {'_type': 'node', '_cardinality': 'per_match'},
        }
        for port in (self.config.get('artifact_params') or {}):
            ports[port] = {'_type': 'string', '_is_file': True}
        return ports

    def outputs(self):
        return {'results': 'node'}

    # ── paths ────────────────────────────────────────────────────────

    def _artifact_root(self) -> str:
        return self.config.get('artifact_root') or os.path.join(
            os.getcwd(), '.pbg', 'artifacts')

    def _workdir_root(self) -> str:
        """Scratch directory for seed outputs — sibling of the artifact store."""
        base = os.path.dirname(os.path.normpath(self._artifact_root())) or os.getcwd()
        return os.path.join(base, 'work')

    def _seed_dir(self, val) -> str:
        return os.path.join(self._workdir_root(), f'seed_{_safe_token(val)}')

    def _provenance_path(self) -> str:
        node = self.config['generator']
        return os.path.join(self._workdir_root(), node, 'provenance.json')

    def _code_version(self) -> str:
        return self.config.get('code_version') or _default_code_version(
            self.config.get('import') or [])

    # ── cache key (F1) ──────────────────────────────────────────────

    def _address(self, entry, val, ref_hashes: List[str]) -> str:
        overrides = dict(self.config.get('overrides') or {})
        scatter_param = self.config.get('scatter_param')
        if scatter_param:
            overrides[scatter_param] = val
        key_config = {
            **overrides,
            'steps': self.config['steps'],
            'provision': list(self.config.get('provision') or []),
        }
        return artifact_id(
            composite_id=self.config['generator'],
            config=key_config,
            input_ids=sorted(ref_hashes),
            commit=self._code_version(),
        )

    # ── one match ────────────────────────────────────────────────────

    def _build_doc(self, entry, val, result_dir_for_emitter: Optional[str]) -> Dict[str, Any]:
        overrides = dict(self.config.get('overrides') or {})
        scatter_param = self.config.get('scatter_param')
        if scatter_param:
            overrides[scatter_param] = val
        if result_dir_for_emitter is not None and _generator_declares(entry, 'emitter_out_dir'):
            overrides['emitter_out_dir'] = result_dir_for_emitter

        artifacts_decl = {
            port: {'kind': kind, 'map': 'store'}
            for port, kind in (self.config.get('artifact_params') or {}).items()
        }
        return {
            'build': {
                'generator': self.config['generator'],
                'import': list(self.config.get('import') or []),
                'overrides': overrides,
                'provision': list(self.config.get('provision') or []),
            },
            'artifacts': artifacts_decl,
            'run': {'steps': self.config['steps']},
        }

    def _run_match(self, key, val, entry, artifact_paths: Dict[str, str]
                   ) -> Tuple[str, str, bool, float, str]:
        t0 = time.monotonic()

        ref_hashes = []
        for path in artifact_paths.values():
            ref_hashes.append(_load_ref_hash(path))

        address = self._address(entry, val, ref_hashes)

        seed_dir = self._seed_dir(val)
        result_dir = os.path.join(seed_dir, 'results')
        state_out = os.path.join(seed_dir, 'state.json')

        artifact_root = self._artifact_root()
        cache_hit = artifact_exists(address, root=artifact_root) and os.path.isfile(state_out)

        if not cache_hit:
            os.makedirs(seed_dir, exist_ok=True)
            build_doc = self._build_doc(
                entry, val,
                result_dir_for_emitter=result_dir)
            build_path = os.path.join(seed_dir, 'build.json')
            with open(build_path, 'w') as fh:
                json.dump(build_doc, fh)

            cmd = [sys.executable, '-m', 'process_bigraph.run_composite',
                   '--build', build_path]
            scatter_param = self.config.get('scatter_param')
            if scatter_param:
                cmd += ['--set', f'{scatter_param}={json.dumps(val)}']
            for port, path in artifact_paths.items():
                cmd += ['--artifact', f'{port}={path}']
            cmd += ['--state-out', state_out]

            proc = subprocess.run(
                cmd, env=os.environ, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{_log_prefix} run_composite subprocess failed for match "
                    f"{key!r} (val={val!r}, address={address}):\n"
                    f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

            with open(state_out) as fh:
                produced = json.load(fh)
            write_fingerprint(address, fingerprint_of(produced), root=artifact_root)

        wall_s = time.monotonic() - t0
        result_path = result_dir if os.path.isdir(result_dir) else state_out
        return key, result_path, cache_hit, wall_s, address

    # ── invoke (R2: bounded ThreadPoolExecutor, not ProcessPool) ─────

    def invoke(self, state: Dict[str, Any], _: Optional[float] = None) -> SyncUpdate:
        entry = _resolve_entry(self.config['generator'], self.config.get('import') or [])
        _guard_emitter(entry, bool(self.config.get('allow_in_memory_emitter')))

        scatter_port = self._scatter_port_name()
        match_set = state.get(scatter_port) or {}
        if isinstance(match_set, list):
            items = [(str(i), v) for i, v in enumerate(match_set)]
        elif isinstance(match_set, dict):
            items = list(match_set.items())
        else:
            raise TypeError(
                f"{_log_prefix}: scatter port {scatter_port!r} expects a dict "
                f"or list match-set, got {type(match_set).__name__}")

        artifact_paths = {
            port: state[port] for port in (self.config.get('artifact_params') or {})
            if state.get(port)
        }

        n = len(items)
        if n == 0:
            self._write_provenance({})
            return SyncUpdate({'results': {}})

        cpu = os.cpu_count() or 2
        default_workers = max(1, min(n, cpu // 2))
        max_workers = int(self.config.get('max_workers') or default_workers)
        max_workers = max(1, max_workers)

        results: Dict[str, str] = {}
        provenance: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._run_match, key, val, entry, artifact_paths)
                for key, val in items
            ]
            for future in futures:
                key, result_path, cache_hit, wall_s, address = future.result()
                results[key] = result_path
                provenance[key] = {
                    'address': address, 'cache_hit': cache_hit, 'wall_s': wall_s}

        self._write_provenance(provenance)
        return SyncUpdate({'results': results})

    def _write_provenance(self, provenance: Dict[str, Dict[str, Any]]) -> None:
        path = self._provenance_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(provenance, fh, indent=2)


def _load_ref_hash(ref_path: str) -> str:
    with open(ref_path) as fh:
        ref = ArtifactRef.coerce(json.load(fh))
    return ref.hash


def _safe_token(val) -> str:
    token = str(val)
    return ''.join(c if (c.isalnum() or c in ('.', '-', '_')) else '_' for c in token)
