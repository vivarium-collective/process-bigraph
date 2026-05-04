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

try:
    import ray
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "process_bigraph.protocols.ray requires the optional `ray` "
        "dependency. Install with: pip install process-bigraph[ray]"
    ) from e

from process_bigraph import Process


# ---------------------------------------------------------------------------
# Process class registry.
# Ray pickles this into each new actor at spawn so workers don't need to
# import the same modules in their startup script.
# ---------------------------------------------------------------------------
_PROCESS_REGISTRY: Dict[str, Type[Process]] = {}


def register_process_class(name: str, cls: Type[Process]) -> None:
    """Register a Process class so RayProcess can resolve it by name."""
    _PROCESS_REGISTRY[name] = cls


def get_registry() -> Dict[str, Type[Process]]:
    return dict(_PROCESS_REGISTRY)


# ---------------------------------------------------------------------------
# Ray actor — one per pool slot. Holds a single Process instance.
# ---------------------------------------------------------------------------
@ray.remote
class _ProcessActor:
    def __init__(self, registry: Dict[str, Type[Process]],
                 class_name: str, config: dict):
        for k, v in registry.items():
            _PROCESS_REGISTRY[k] = v
        cls = _PROCESS_REGISTRY[class_name]
        from process_bigraph import allocate_core
        self.instance = cls(config, core=allocate_core())

    def inputs(self):
        return self.instance.inputs()

    def outputs(self):
        return self.instance.outputs()

    def update(self, state: dict, interval: float):
        return self.instance.update(state, interval)


# ---------------------------------------------------------------------------
# Actor pool. One pool per (process_class, process_config). Persistent across
# RayProcess instances and across simulation runs.
# ---------------------------------------------------------------------------
class _ActorPool:
    def __init__(self, class_name: str, config: dict, n_workers: int):
        registry = get_registry()
        # Spawn all actors concurrently — actor.remote() returns immediately;
        # we don't ray.get on the constructor. The first .inputs.remote() call
        # implicitly waits for the actor to be ready.
        self.actors = [
            _ProcessActor.remote(registry, class_name, config)
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
