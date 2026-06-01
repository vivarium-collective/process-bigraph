"""
RayProcess — distributed transport backed by Ray actors.

Pair with the ``parallel_processes=True`` flag on Composite so the orchestrator
can dispatch per-step ``update()`` calls concurrently — that's what turns N
clients talking to a Ray actor pool into N parallel solves.

Install with the optional ray extra::

    pip install process-bigraph[ray]

Architecture: pooled actors
---------------------------
Each (process_class, process_config) pair backs a fixed pool of N Ray actors
(default N = ncpu). Every RayProcess client is round-robin assigned to one
pool actor; many "logical" processes share the same physical worker. This
bounds memory at O(ncpu) underlying-process instances instead of O(clients),
and bounds spawn cost at ncpu actors regardless of how many clients the
orchestrator wires up.

Why pooled, not actor-per-client:
  - One actor per cell at moderate grids (e.g. 256 cells with a 150 MB cobra
    Model each) trivially OOMs a typical workstation.
  - Per-actor spawn cost (process fork + module import + heavy state init)
    is 50-500 ms; paying that 256× per run is minutes of overhead.
  - Ray actor methods are serialized by default — concurrent calls to one
    actor are queued, so non-thread-safe state inside the underlying Process
    isn't a concern.

Pool lifecycle:
  - Pools live for the lifetime of the Python interpreter by default.
    Subsequent ``Composite`` runs re-use the same actors — no re-spawn,
    no model reload.
  - Call ``shutdown_pools()`` to tear them down explicitly (useful in tests).
  - ``RayProcess.end()`` is a no-op — clients come and go but actors persist.

Usage
-----
1. Register the underlying Process classes once at startup so each Ray
   worker can resolve them by name::

       from process_bigraph.protocols.ray import register_process_class
       from my_pkg.processes import MyProcess
       register_process_class("MyProcess", MyProcess)

2. Reference RayProcess in your composite spec::

       "worker_0": {
           "_type": "process",
           "address": "local:RayProcess",
           "config": {
               "process_class": "MyProcess",
               "process_config": { ... MyProcess's config ... },
               # optional: cap pool size (default = os.cpu_count())
               "pool_size": 8,
           },
           "inputs":  { ... },
           "outputs": { ... },
           "interval": 0.1,
       }

3. Pass ``parallel_processes=True`` to Composite so the orchestrator dispatches
   the per-step ``update()`` calls concurrently.
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List, Type, Optional

# Ray is optional. We let the module import even when ray isn't installed
# (so package scanners like discover_packages don't trip), and only raise
# a helpful error when something tries to actually use it.
try:
    import ray
    _RAY_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _e:  # pragma: no cover
    ray = None  # type: ignore[assignment]
    _RAY_IMPORT_ERROR = _e


def _require_ray() -> None:
    """Guard for code paths that need ray. Raises a clear install hint."""
    if ray is None:
        raise ImportError(
            "process_bigraph.protocols.ray requires the optional `ray` "
            "dependency. Install with: pip install process-bigraph[ray]"
        ) from _RAY_IMPORT_ERROR


from process_bigraph import Process
from bigraph_schema.methods import load_protocol


# ---------------------------------------------------------------------------
# Process class registry.
# Ray pickles this into each new actor at spawn so workers don't need to
# import the same modules in their startup script.
# ---------------------------------------------------------------------------
_PROCESS_REGISTRY: Dict[str, Type[Process]] = {}

# Type-providers are pickleable references (``(module_path, attr_name)``)
# to functions ``fn(core) → None`` that register custom types on a
# core. Driver registers them once; each Ray actor calls every
# provider against its freshly-allocated core before instantiating
# the Process. Lets cell-as-Composite-on-Ray work — without this the
# actor's allocate_core() doesn't have project-specific types like
# vEcoli's ``sim_data_object_store`` and the actor's Composite init
# crashes inside ``realize`` walking the cell tree.
_TYPE_PROVIDERS: list = []


def register_process_class(name: str, cls: Type[Process]) -> None:
    """Register a Process class so RayProcess can resolve it by name."""
    _PROCESS_REGISTRY[name] = cls


def register_type_provider(module: str, attr: str,
                            args: tuple = (), kwargs: dict = None) -> None:
    """Register a type-registration function for actors to call on
    their core. ``module`` is the importable Python module path,
    ``attr`` is a top-level attribute callable as ``fn(core, *args, **kwargs)``.
    Multiple providers may be registered; actors call them in order.

    Use ``args``/``kwargs`` to thread per-run data into the provider
    (e.g. a sim_data path the actor should load from disk). Both must
    be picklable so they cross the actor boundary intact."""
    if kwargs is None:
        kwargs = {}
    _TYPE_PROVIDERS.append((module, attr, tuple(args), dict(kwargs)))


def get_registry() -> Dict[str, Type[Process]]:
    return dict(_PROCESS_REGISTRY)


def get_type_providers() -> list:
    return list(_TYPE_PROVIDERS)


# ---------------------------------------------------------------------------
# Ray actor — one per pool slot. Holds a single Process instance.
#
# Declared as a plain class at module load time so this file imports cleanly
# without ray installed. ``ray.remote(...)`` is applied lazily on first use
# (cached) inside ``_remote_actor_class()``.
# ---------------------------------------------------------------------------
def _apply_type_providers(core, providers: list) -> None:
    """Import each (module, attr, args, kwargs) provider and call
    ``fn(core, *args, **kwargs)`` to register custom types. Imports
    are lazy so the provider's module doesn't have to be installed
    where the protocol module loads. Legacy 2-tuple form
    ``(module, attr)`` is also accepted for back-compat."""
    import importlib
    for entry in providers:
        if len(entry) == 2:
            module_name, attr_name = entry
            args, kwargs = (), {}
        else:
            module_name, attr_name, args, kwargs = entry
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, attr_name)
            fn(core, *args, **kwargs)
        except Exception as e:
            import sys
            sys.stderr.write(
                f'[ray-protocol] type provider {module_name}.{attr_name} '
                f'failed: {type(e).__name__}: {e}\n')
            sys.stderr.flush()
            raise


def _make_state_writeable(config: dict) -> None:
    """Walk ``config['state']`` and copy any read-only numpy arrays.
    Ray's plasma store hands actors arrays with the read-only flag for
    zero-copy semantics; Cython buffers (stochastic_arrow, fbax, ...)
    refuse to accept them. Done before Composite instantiation so the
    actor's self.state starts fully mutable.
    """
    state = config.get('state')
    if state is None:
        return
    try:
        from bigraph_schema import make_arrays_writeable
        make_arrays_writeable(state)
    except Exception:
        # Fallback walk in case make_arrays_writeable doesn't reach
        # structured-array subfields or nested lists.
        import numpy as _np
        def _walk(node):
            if isinstance(node, dict):
                for k, v in list(node.items()):
                    if isinstance(v, _np.ndarray) and not v.flags.writeable:
                        node[k] = v.copy()
                        node[k].flags.writeable = True
                    elif isinstance(v, (dict, list)):
                        _walk(v)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    if isinstance(v, _np.ndarray) and not v.flags.writeable:
                        node[i] = v.copy()
                        node[i].flags.writeable = True
                    elif isinstance(v, (dict, list)):
                        _walk(v)
        _walk(state)


class _ProcessActor:
    def __init__(self, registry: Dict[str, Type[Process]],
                 class_name: str, config: dict,
                 type_providers: list = None):
        for k, v in registry.items():
            _PROCESS_REGISTRY[k] = v
        cls = _PROCESS_REGISTRY[class_name]
        from process_bigraph import allocate_core
        core = allocate_core()
        if type_providers:
            _apply_type_providers(core, type_providers)
        _make_state_writeable(config)
        self.instance = cls(config, core=core)

    def inputs(self):
        return self.instance.inputs()

    def outputs(self):
        return self.instance.outputs()

    def update(self, state: dict, interval: float):
        return self.instance.update(state, interval)


_REMOTE_ACTOR_CLASS = None  # cached ray.remote(_ProcessActor)


def _remote_actor_class():
    """Return the ray-remote-wrapped _ProcessActor, building it on first call."""
    global _REMOTE_ACTOR_CLASS
    if _REMOTE_ACTOR_CLASS is None:
        _require_ray()
        _REMOTE_ACTOR_CLASS = ray.remote(_ProcessActor)
    return _REMOTE_ACTOR_CLASS


# ---------------------------------------------------------------------------
# Actor pool. One pool per (process_class, process_config). Persistent across
# RayProcess instances and across simulation runs.
# ---------------------------------------------------------------------------
class _ActorPool:
    def __init__(self, class_name: str, config: dict, n_workers: int):
        registry = get_registry()
        type_providers = get_type_providers()
        actor_cls = _remote_actor_class()
        # Spawn all actors concurrently — actor.remote() returns immediately;
        # we don't ray.get on the constructor. The first .inputs.remote() call
        # implicitly waits for the actor to be ready.
        self.actors = [
            actor_cls.remote(registry, class_name, config, type_providers)
            for _ in range(n_workers)
        ]
        self._next = 0

    def assign(self):
        actor = self.actors[self._next % len(self.actors)]
        self._next += 1
        return actor

    def shutdown(self):
        for a in self.actors:
            try:
                ray.kill(a)
            except Exception:
                pass
        self.actors = []


# Module-level pool registry, keyed by (class_name, config_hash).
_POOLS: Dict[str, _ActorPool] = {}


def _config_hash(config: Any) -> str:
    """Stable hash of a process_config dict for pool keying."""
    try:
        s = json.dumps(config, sort_keys=True, default=repr)
    except TypeError:
        s = repr(config)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _config_hash_pool_key(config: Any) -> str:
    """Hash for actor-pool keying. Excludes ``state`` so cells with
    different state (e.g. mother + her daughters after divide) share
    the same actor pool — each actor then holds many Composite
    instances keyed by proc_id. Compared to ``_config_hash``, this
    drops per-cell-only fields that don't affect actor lifecycle."""
    if isinstance(config, dict):
        pool_config = {k: v for k, v in config.items() if k != 'state'}
    else:
        pool_config = config
    try:
        s = json.dumps(pool_config, sort_keys=True, default=repr)
    except TypeError:
        s = repr(pool_config)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _pool_key(class_name: str, config: Any) -> str:
    return f"{class_name}:{_config_hash(config)}"


def _get_or_make_pool(class_name: str, config: dict,
                      n_workers: Optional[int]) -> _ActorPool:
    key = _pool_key(class_name, config)
    pool = _POOLS.get(key)
    if pool is None:
        if n_workers is None:
            n_workers = max(1, os.cpu_count() or 4)
        pool = _ActorPool(class_name, config, n_workers)
        _POOLS[key] = pool
    return pool


def shutdown_pools() -> None:
    """Tear down all actor pools. Call at program exit / between test runs."""
    for pool in list(_POOLS.values()):
        pool.shutdown()
    _POOLS.clear()


def pool_stats() -> List[dict]:
    """Diagnostic: list all live pools."""
    return [
        {"key": k, "n_actors": len(p.actors)}
        for k, p in _POOLS.items()
    ]


# ---------------------------------------------------------------------------
# Client — what the orchestrator sees as a Process.
# ---------------------------------------------------------------------------
class RayProcess(Process):
    """A Process whose update() runs on a pooled remote Ray actor.

    Config:
        process_class : str
            Name of a Process subclass registered via register_process_class().
        process_config : dict
            Config dict passed to the underlying Process subclass.
        pool_size : int (optional)
            Number of actors in the pool for this (class, config). Defaults
            to os.cpu_count(). The first RayProcess instantiation for a
            given (class, config) sizes the pool — subsequent instances
            reuse the existing pool and ignore this field.
    """

    config_schema = {
        "process_class": "string",
        "process_config": "node",
        "pool_size": "maybe[integer]",
    }

    def initialize(self, config):
        _require_ray()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        class_name = config["process_class"]
        if class_name not in _PROCESS_REGISTRY:
            raise KeyError(
                f"Process class {class_name!r} not in RayProcess registry. "
                f"Call register_process_class({class_name!r}, <cls>) first."
            )

        pool = _get_or_make_pool(
            class_name,
            config["process_config"],
            config.get("pool_size"),
        )
        self.actor = pool.assign()

        # Cache port schemas — one round-trip per client at init. (We could
        # cache per-pool to drop this, but it's a single call and the result
        # could in principle differ if the underlying Process introspects
        # config-specific port shapes.)
        self._inputs  = ray.get(self.actor.inputs.remote())
        self._outputs = ray.get(self.actor.outputs.remote())

    def inputs(self):
        return self._inputs

    def outputs(self):
        return self._outputs

    def update(self, state, interval):
        # Blocking get: releases the GIL while the actor runs. ParallelComposite
        # gives us N concurrent in-flight calls = N actors busy in parallel.
        return ray.get(self.actor.update.remote(state, float(interval)))

    def end(self):
        # Pool actors persist across RayProcess instances — DON'T kill them
        # here. Use shutdown_pools() to tear down explicitly.
        pass


# ===========================================================================
# Address-based protocol: ``address: "ray:Foo"``
# ---------------------------------------------------------------------------
# Lets a Composite document declare individual processes with
# ``"address": "ray:DynamicFBA"`` (instead of ``"local:RayProcess"``)
# and have the Ray protocol handle sharding + batched RPCs transparently.
#
# The user-facing graph stays faithful — every cell is still a real Process
# node (a ``RayShadowProcess``). The protocol intercepts ``invoke()`` to
# enqueue the per-cell call onto a shared runtime, and the Composite's
# ``_flush_protocol_runtimes`` hook (added between the per-tick invoke pass
# and ``apply_updates``) issues *one* batched RPC per shard. So 4096 cells
# with one shared config → 16 shard actors → 16 RPCs/tick, not 4096.
#
# Lifecycle: one ``RayProtocolRuntime`` per (Composite × core); created on
# first enqueue, closed via ``RayProtocolRuntime.close()`` (caller's
# responsibility for now — Composite shutdown integration is a follow-up).
# ===========================================================================
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

try:
    from plum import dispatch as _plum_dispatch  # noqa: F401
except ImportError:
    _plum_dispatch = None  # type: ignore[assignment]


# Lazy ray.remote wrapping for the shard actor — same pattern as _ProcessActor.
_BatchActorClass = None


def _batch_actor_class():
    global _BatchActorClass
    if _BatchActorClass is not None:
        return _BatchActorClass
    _require_ray()

    @ray.remote
    class _RayBatchActor:
        """Long-lived actor hosting MANY underlying Process instances,
        keyed by proc_id. Cells with the same shared config (schema/
        bridge/address — see ``_pool_config``) share one actor pool;
        each cell gets its own Composite inside the actor. This amortizes
        sim_data + Cython import overhead across all cells on a node.

        Lifecycle:
        - ``__init__``: set up registry + core + type providers ONCE.
            No Composite built yet. The optional ``warm_config`` lets
            ping include a smoke check; not required.
        - ``init_cell(proc_id, full_config)``: build the per-cell
            Composite on first encounter of that proc_id. Driver
            ensures init_cell is called before the first batch_update
            for that proc_id.
        - ``batch_update(batch, interval)``: route per-proc_id to the
            right Composite. Persistent state — Composites are kept
            across ticks so warm solver state survives.
        """

        def __init__(self, registry: Dict[str, Type[Process]],
                     class_name: str,
                     type_providers: list = None,
                     warm_config: dict = None):
            for k, v in registry.items():
                _PROCESS_REGISTRY[k] = v
            self.cls = _PROCESS_REGISTRY[class_name]
            from process_bigraph import allocate_core
            self.core = allocate_core()
            if type_providers:
                _apply_type_providers(self.core, type_providers)
            self.composites: Dict[int, Any] = {}

        def init_cell(self, proc_id: int, config: dict) -> str:
            """Build the per-cell Composite. Idempotent: a proc_id
            already initialized is a no-op (driver may retry on
            actor recovery)."""
            if proc_id in self.composites:
                return "exists"
            _make_state_writeable(config)
            self.composites[proc_id] = self.cls(config, core=self.core)
            return "built"

        def has_cell(self, proc_id: int) -> bool:
            return proc_id in self.composites

        def inputs(self):
            # Inputs schema is per-cell. With multi-cell actors,
            # the driver should be using RayShadowProcess._template_inputs
            # (the inputs/outputs cached at protocol-bind time, before
            # the actor was even spun up). This method preserved for
            # back-compat with any caller that still pings actors for
            # I/O shape. Returns first cell's inputs if available.
            if self.composites:
                return next(iter(self.composites.values())).inputs()
            return {}

        def outputs(self):
            if self.composites:
                return next(iter(self.composites.values())).outputs()
            return {}

        def batch_update(self, batch: list, interval: float) -> dict:
            out = {}
            for proc_id, inputs in batch:
                composite = self.composites.get(proc_id)
                if composite is None:
                    raise RuntimeError(
                        f"_RayBatchActor.batch_update: proc_id "
                        f"{proc_id} not initialized — driver must "
                        f"call init_cell before first batch_update.")
                out[proc_id] = composite.update(inputs, float(interval))
            return out

        def ping(self) -> str:
            return "ready"

    _BatchActorClass = _RayBatchActor
    return _BatchActorClass


def _stable_proc_id(shadow: "RayShadowProcess") -> int:
    """A stable integer id for routing and result lookup. Composite doesn't
    expose a per-Process unique id, so use Python ``id()`` of the shadow
    instance — stable for the shadow's lifetime, which is what we need."""
    return id(shadow)


@dataclass
class _ShardPool:
    """One pool of N batch actors keyed by (target_class, shared_config_hash).
    Each actor hosts MANY Composites keyed by proc_id (one per cell that
    routes to that shard). Process ids assigned to a shard on first
    enqueue stay sticky — keeps warm solver state aligned with the cells
    it's seen. ``proc_initialized`` tracks which proc_ids have had
    ``init_cell`` called on their actor."""
    actors: List[Any]
    proc_to_shard: Dict[int, int] = field(default_factory=dict)
    pending: Dict[int, list] = field(default_factory=lambda: defaultdict(list))
    proc_initialized: set = field(default_factory=set)


class RayProtocolRuntime:
    """Per-(core) runtime that batches updates from all ``ray:Foo`` shadow
    processes through fixed actor pools. Owns lifecycle of those actors —
    call ``close()`` to release them.

    ``n_shards_default`` determines pool size for newly-seen
    (target_class, config_hash) pairs; ``RAY_SHARDS_DEFAULT`` env var
    overrides at import time.
    """

    def __init__(self,
                 n_shards_default: Optional[int] = None,
                 ray_address: Optional[str] = None):
        _require_ray()
        if not ray.is_initialized():
            if ray_address:
                ray.init(address=ray_address, log_to_driver=False)
            else:
                ray.init(ignore_reinit_error=True, log_to_driver=False)

        if n_shards_default is None:
            env = os.environ.get("RAY_SHARDS_DEFAULT")
            if env:
                n_shards_default = int(env)
            else:
                n_shards_default = max(1, os.cpu_count() or 4)
        self.n_shards_default = int(n_shards_default)

        self._pools: Dict[str, _ShardPool] = {}
        self._results: Dict[int, dict] = {}
        # enqueue() may be called from threads if Composite uses
        # parallel_processes=True; per-pool dispatch and proc_to_shard
        # assignment must be threadsafe.
        self._lock = threading.Lock()

    # -- pool management ---------------------------------------------- #

    def _pool_for(self, class_name: str, config: dict) -> _ShardPool:
        """Resolve (or create) the actor pool for this class. Pool
        keying uses ONLY ``class_name`` (plus an optional explicit
        ``pool_key`` config field) — cells of the same target class
        share one pool of actors regardless of their per-cell config.
        Each actor holds its own ``{proc_id: Composite}`` dict; per-cell
        config is delivered to the actor via ``init_cell`` on first
        enqueue, so the actor still gets the per-cell setup it needs.

        Why class-only? Pickling schema dicts through actor↔driver
        round-trips produces new Schema Node instances whose ``repr()``
        is not byte-stable, so hashing schema as the pool key produces
        a NEW pool per daughter (one per divide) — defeating the actor
        pool optimization. Users who genuinely need pool isolation
        between distinct schemas of the same class can set an explicit
        ``pool_key`` string in config."""
        explicit_key = config.get('pool_key') if isinstance(config, dict) else None
        key_suffix = explicit_key if explicit_key else 'default'
        key = f"{class_name}:{key_suffix}"
        pool = self._pools.get(key)
        if pool is None:
            actor_cls = _batch_actor_class()
            registry = get_registry()
            type_providers = get_type_providers()
            actors = [
                actor_cls.remote(registry, class_name, type_providers)
                for _ in range(self.n_shards_default)
            ]
            # Race all __init__'s in parallel so cold-start doesn't
            # serialize on the first tick.
            ray.get([a.ping.remote() for a in actors])
            pool = _ShardPool(actors=actors)
            self._pools[key] = pool
        return pool

    def _shard_index_for(self, pool: _ShardPool, proc_id: int) -> int:
        idx = pool.proc_to_shard.get(proc_id)
        if idx is None:
            # Round-robin across shards by next-available-count. Sticky
            # after first assignment.
            counts = [0] * len(pool.actors)
            for s in pool.proc_to_shard.values():
                counts[s] += 1
            idx = counts.index(min(counts))
            pool.proc_to_shard[proc_id] = idx
        return idx

    # -- API used by RayShadowProcess --------------------------------- #

    def enqueue(self, proc_id: int, class_name: str, config: dict,
                inputs: dict, interval: float) -> None:
        """Add one process's update to its shard's pending batch.
        Threadsafe — Composite may call this from N parallel threads.

        On first enqueue for a proc_id, syncronously initialize the
        cell on its assigned actor (``init_cell.remote(proc_id, config)``).
        This may be slow on first call (loads sim_data etc.) — kept
        synchronous so batch_update later finds the Composite ready."""
        with self._lock:
            pool = self._pool_for(class_name, config)
            shard_idx = self._shard_index_for(pool, proc_id)
            if proc_id not in pool.proc_initialized:
                actor = pool.actors[shard_idx]
                ray.get(actor.init_cell.remote(proc_id, config))
                pool.proc_initialized.add(proc_id)
            pool.pending[shard_idx].append((proc_id, inputs, float(interval)))

    def collect(self, proc_id: int) -> dict:
        """Pull a process's resolved delta. Returns ``{}`` when the
        process didn't have a pending update this tick."""
        return self._results.pop(proc_id, {})

    def flush_pending(self) -> None:
        """Resolve all pending shard batches in parallel. Called by
        ``Composite._flush_protocol_runtimes`` after the invoke pass."""
        if not any(pool.pending for pool in self._pools.values()):
            return
        # Issue all batched RPCs concurrently — we ray.get the union.
        futures = []
        manifest = []  # parallel list of (intervals, batch) for result mapping
        for pool in self._pools.values():
            for shard_idx, batch in list(pool.pending.items()):
                if not batch:
                    continue
                # All cells in a shard share the same tick interval —
                # Composite calls invoke() with the same per-process
                # interval at any one tick. Use the first.
                interval = batch[0][2]
                payload = [(pid, inp) for (pid, inp, _) in batch]
                fut = pool.actors[shard_idx].batch_update.remote(
                    payload, float(interval))
                futures.append(fut)
                manifest.append(batch)
                pool.pending[shard_idx] = []
        # Wait on all in parallel.
        results_list = ray.get(futures)
        # Scatter into self._results keyed by proc_id.
        for batch, results in zip(manifest, results_list):
            for proc_id, _, _ in batch:
                self._results[proc_id] = results.get(proc_id, {})

    def close(self) -> None:
        for pool in self._pools.values():
            for a in pool.actors:
                try:
                    ray.kill(a)
                except Exception:
                    pass
        self._pools.clear()
        self._results.clear()


# Module-level cache of runtimes keyed by core id. One Composite per core
# is the common case, so this maps 1:1 in practice; multi-Composite-on-
# one-core workloads share the runtime, which is fine — actors are
# pool-global.
_RUNTIMES: Dict[int, RayProtocolRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_or_create_runtime(core: Any,
                          n_shards_default: Optional[int] = None,
                          ray_address: Optional[str] = None) -> RayProtocolRuntime:
    """Return the shared runtime for this core, creating it on first call.
    ``n_shards_default`` and ``ray_address`` are honored only on creation."""
    with _RUNTIMES_LOCK:
        rt = _RUNTIMES.get(id(core))
        if rt is None:
            rt = RayProtocolRuntime(
                n_shards_default=n_shards_default,
                ray_address=ray_address)
            _RUNTIMES[id(core)] = rt
        return rt


def shutdown_runtime(core: Any) -> None:
    """Tear down the runtime for one core (kills its actors)."""
    with _RUNTIMES_LOCK:
        rt = _RUNTIMES.pop(id(core), None)
    if rt is not None:
        rt.close()


def shutdown_all_runtimes() -> None:
    """Tear down every runtime in the process. Useful at end-of-test."""
    with _RUNTIMES_LOCK:
        rts = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for rt in rts:
        rt.close()


class _RayDefer:
    """Defer-shaped object returned by ``RayShadowProcess.invoke``.
    ``.get()`` blocks until ``RayProtocolRuntime.flush_pending`` has
    run. The Composite's ``_flush_protocol_runtimes`` hook ensures
    that's true before any ``apply_updates`` reads from us."""
    __slots__ = ("_runtime", "_proc_id")

    def __init__(self, runtime: RayProtocolRuntime, proc_id: int):
        self._runtime = runtime
        self._proc_id = proc_id

    def get(self):
        return self._runtime.collect(self._proc_id)


class RayShadowProcess(Process):
    """Local Process whose ``invoke()`` enqueues onto a RayProtocolRuntime
    instead of running locally. The wrapped process class lives on a
    Ray actor; this shadow is just a port-shape declaration + a queue tap.

    The ``load_protocol`` dispatch for ``RayProtocol`` returns dynamic
    subclasses with these class-level bindings populated:

        _target_class      : Type[Process]   the underlying class
        _target_class_name : str             registry key for the actor
        _runtime           : RayProtocolRuntime
        _template_inputs   : dict            cached inputs() schema
        _template_outputs  : dict            cached outputs() schema
    """

    _target_class: Any = None
    _target_class_name: str = ""
    _runtime: Any = None
    _template_inputs: Any = None
    _template_outputs: Any = None
    config_schema: Any = None  # set per-bound-subclass at load_protocol time

    def initialize(self, config):
        # Stash the resolved config so we can use the same shape on the
        # actor side. The runtime's pool key is computed from this dict.
        self._proc_config = config
        # Composite reads ``_protocol_runtime`` to build the deduped
        # active-runtime list for ``flush_pending``.
        self._protocol_runtime = self._runtime

    def inputs(self):
        return self._template_inputs

    def outputs(self):
        return self._template_outputs

    def invoke(self, state, interval):
        proc_id = _stable_proc_id(self)
        self._runtime.enqueue(
            proc_id,
            self._target_class_name,
            self._proc_config,
            state,
            float(interval),
        )
        return _RayDefer(self._runtime, proc_id)


# ---------------------------------------------------------------------------
# Protocol type registration: ``"address": "ray:Foo"`` parses to
# {protocol: "ray", data: "Foo"}. The dispatch resolves "Foo" against the
# RayProcess registry, builds a bound RayShadowProcess subclass, and the
# framework instantiates it like any other Process.
# ---------------------------------------------------------------------------
from bigraph_schema.schema import Protocol as _ProtocolNode
from bigraph_schema.schema import String


@dataclass(kw_only=True)
class RayProtocol(_ProtocolNode):
    data: String = field(default_factory=String)


def _build_shadow_class(target_name: str, target_cls: Any,
                       runtime: RayProtocolRuntime):
    """Construct a RayShadowProcess subclass bound to a specific
    underlying class. Schema queries (inputs/outputs) come from a
    one-time template instantiation of the underlying class — this
    pays the per-class init cost ONCE locally, then never again."""
    # Build a temporary local instance to read its port schemas. For
    # processes whose __init__ is expensive (e.g. cobra Model load),
    # this is paid once per address binding, regardless of how many
    # cells reference it.
    from process_bigraph import allocate_core
    tmpl = target_cls({}, core=allocate_core()) if False else None
    # Most processes need a real config to instantiate. Defer the
    # schema query to first use, where the shadow has its actual config.
    template_inputs: Any = None
    template_outputs: Any = None

    bound_name = f"RayShadow_{target_name}"
    config_schema = getattr(target_cls, "config_schema", None) or {}

    bound_attrs = {
        "_target_class": target_cls,
        "_target_class_name": target_name,
        "_runtime": runtime,
        "_template_inputs": template_inputs,
        "_template_outputs": template_outputs,
        "config_schema": config_schema,
        "__module__": __name__,
    }
    cls = type(bound_name, (RayShadowProcess,), bound_attrs)

    # Override initialize to lazily populate the template schema on the
    # first instance of this class. Fast path after first init.
    original_initialize = cls.initialize

    def initialize_with_schema_cache(self, config):
        original_initialize(self, config)
        if cls._template_inputs is None:
            # One-time per bound subclass: build a temp local instance
            # with this config to read its port shapes; cache on the
            # class. The temp instance is discarded — the actor holds
            # the long-lived Process state.
            from process_bigraph import allocate_core as _ac
            tmpl = target_cls(config, core=_ac())
            cls._template_inputs = tmpl.inputs()
            cls._template_outputs = tmpl.outputs()
        self._template_inputs = cls._template_inputs
        self._template_outputs = cls._template_outputs

    cls.initialize = initialize_with_schema_cache
    return cls


def _resolve_target(core, name: str):
    """Resolve a process class by name from the core's link_registry."""
    cls = core.link_registry.get(name)
    if cls is None:
        raise KeyError(
            f"ray:{name} — no Process class named {name!r} in the "
            f"link_registry. Make sure the package is discovered "
            f"(usually via discover_packages or register_link)."
        )
    return cls


@load_protocol.dispatch
def load_protocol(core, protocol: RayProtocol, data):
    target_cls = _resolve_target(core, data)
    runtime = get_or_create_runtime(core)

    # Register the underlying class once with the actor-side registry so
    # _RayBatchActor.__init__ on any spawned actor can resolve it by name.
    # Idempotent — register_process_class is just a dict assignment.
    register_process_class(data, target_cls)

    bound_cls = _build_shadow_class(data, target_cls, runtime)

    def instantiate(config, core=None):
        return bound_cls(config, core)

    instantiate.config_schema = bound_cls.config_schema
    return instantiate


def register_types(core):
    core.register_types({
        'ray': RayProtocol})
    return core


# ---------------------------------------------------------------------------
# Smoke test — wraps IncreaseProcess (a built-in toy Process) in a Ray pool
# and runs a few updates. Useful as both a sanity check and an example.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from process_bigraph import allocate_core
    from process_bigraph.processes.examples import IncreaseProcess

    register_process_class("IncreaseProcess", IncreaseProcess)

    proc_a = RayProcess(
        {"process_class": "IncreaseProcess",
         "process_config": {"rate": 0.5},
         "pool_size": 2},
        core=allocate_core(),
    )
    proc_b = RayProcess(
        {"process_class": "IncreaseProcess",
         "process_config": {"rate": 0.5},
         "pool_size": 2},  # ignored — pool already exists
        core=allocate_core(),
    )
    print("pool stats:", pool_stats())
    for proc, label in [(proc_a, "A"), (proc_b, "B")]:
        upd = proc.update({"level": 4.0}, interval=1.0)
        print(f"{label} update :", upd)
    shutdown_pools()
    print("after shutdown:", pool_stats())
