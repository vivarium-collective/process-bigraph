"""ActorPool — long-lived collection of remote actors of one template
class, with their expensive state (loaded scientific models, JIT caches,
GPU contexts) paid once and reused across many Composite simulations.

Module-global registry of pools keyed by (actor_class, config_hash) so
multiple Composites with identical actor configs share one pool. This
is the layer that makes per-Composite cold-start an artifact rather
than a fundamental cost.

See ``doc/distributed_lifecycles.md`` for the layering this fits into:

    cluster (EC2/Ray) ⊃ pool (this module) ⊃ session ⊃ tick

A pool's lifetime is not tied to any single Composite. Use it like:

    pool = get_or_create_pool(MyActor, {"model_id": "ecoli core"}, size=72)
    pool.warm()                             # paid once

    # later, in any Composite:
    with Session(pool, n_actors=8, sim_config={"cell_keys": [...]}) as s:
        sim = Composite(...)
        sim.run(t)
    # session exits — actors back to pool, NOT killed.

    # at process / cluster teardown:
    shutdown_all_pools()
"""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

try:
    import ray
    _RAY_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _e:
    ray = None  # type: ignore[assignment]
    _RAY_IMPORT_ERROR = _e


def _require_ray() -> None:
    if ray is None:
        raise ImportError(
            "ActorPool requires the optional `ray` dependency. "
            "Install with: pip install process-bigraph[ray]"
        ) from _RAY_IMPORT_ERROR


class ActorPool:
    """Long-lived pool of Ray actors of one template class.

    Lifetime: the pool outlives any single Composite simulation. Actors
    are spawned once via ``warm()``, claimed cheaply by a Session for
    one sim, returned via release, and only torn down via ``shutdown()``
    at process / cluster teardown.

    Thread-safe: ``acquire`` and ``release`` use a lock so multiple
    Composites in the same process can share a pool concurrently.

    For sessions to repurpose pool actors without paying the actor's
    cold-start, the actor's underlying Process should override
    ``Process.reconfigure(sim_config)`` to rebind cheap per-sim params.
    """

    def __init__(
        self,
        actor_class: Any,
        actor_config: dict,
        size: int,
        cluster: Any = None,
    ):
        """
        Args:
            actor_class: A Ray @ray.remote actor class (or a wrapper
                returning one). Actor must accept ``actor_config`` in
                __init__ and expose a ``ping()`` method for warmup.
            actor_config: Constructor args (passed once at warm time).
            size: How many actors to keep in the pool. Sessions claim
                slices of this size.
            cluster: Optional cluster context (EC2SSMRayCluster, etc.) the
                pool belongs to. v1 is informational only; v2 may use
                it for placement hints.
        """
        _require_ray()
        self._actor_class = actor_class
        self._actor_config = dict(actor_config)
        self._size = int(size)
        self._cluster = cluster

        self._all: list = []          # every spawned actor
        self._available: list = []    # not currently claimed
        self._in_use: list = []       # currently claimed by a session

        # Lock guards _available / _in_use mutations and warmup state.
        self._lock = threading.Lock()
        self._warmed = False

    @property
    def size(self) -> int:
        return self._size

    def warm(self) -> None:
        """Spawn ``size`` actors and race their __init__ in parallel.
        Returns when all are responsive (``ping`` returns).

        Idempotent — second call is a no-op. Pays the cobra/model load
        cost (or whatever expensive thing the actor does in __init__)
        once for the lifetime of the pool, regardless of how many
        Composites end up running through it.
        """
        with self._lock:
            if self._warmed:
                return
            self._all = [
                self._actor_class.remote(self._actor_config)
                for _ in range(self._size)
            ]
            self._available = list(self._all)
            self._in_use = []
        # Race all __init__'s in parallel. ray.get on the list of pings
        # blocks until the slowest one returns.
        ray.get([a.ping.remote() for a in self._all])
        with self._lock:
            self._warmed = True

    def grow(self, new_size: int) -> None:
        """Increase pool capacity to ``new_size`` by spawning additional
        actors in parallel. No-op if ``new_size <= current size``.

        Called by ``acquire(n)`` when ``n`` exceeds current size, and by
        ``get_or_create_pool`` when a caller asks for a larger pool than
        the existing one. The expensive per-actor ``__init__`` (cobra
        load etc.) only fires for the *new* slots; existing pooled
        actors keep their state intact.
        """
        if new_size <= 0:
            return
        with self._lock:
            if new_size <= self._size:
                return
            if not self._warmed:
                # First spawn AND grow at once: behave like warm() for
                # the requested size.
                self._all = [
                    self._actor_class.remote(self._actor_config)
                    for _ in range(new_size)
                ]
                self._available = list(self._all)
                self._in_use = []
                self._size = new_size
                new_actors = list(self._all)
            else:
                delta = new_size - self._size
                new_actors = [
                    self._actor_class.remote(self._actor_config)
                    for _ in range(delta)
                ]
                self._all.extend(new_actors)
                self._available.extend(new_actors)
                self._size = new_size
        # Race new actors' __init__ in parallel — outside the lock so
        # other threads can still acquire / release while we wait.
        ray.get([a.ping.remote() for a in new_actors])
        with self._lock:
            self._warmed = True

    def acquire(self, n: int = 1) -> list:
        """Claim ``n`` actors for one Session.

        Returns a list of ``n`` actor handles, removed from the
        available set. Sessions must call ``release()`` on these
        handles at the end of their work (``Session.__exit__``).

        Grows the pool if ``n`` exceeds current size — useful when one
        sweep runs sims of varying scale (n_shards changing per grid).

        Raises ``ValueError`` only if not enough actors are currently
        available even after growing — i.e. another session has them
        in use. (v1 doesn't block; v2 may add a blocking variant.)
        """
        if not self._warmed:
            self.warm()
        # Grow first if needed. Done outside the acquire lock so the
        # ray.get(ping) for new actors doesn't block other operations.
        if n > self._size:
            self.grow(n)
        with self._lock:
            if n > len(self._available):
                raise ValueError(
                    f"acquire({n}) exceeds available {len(self._available)}; "
                    f"in_use={len(self._in_use)}, total={self._size}. "
                    f"Other sessions have actors checked out — wait for "
                    f"them to release.")
            actors = self._available[:n]
            self._available = self._available[n:]
            self._in_use.extend(actors)
            return actors

    def release(self, actors: list) -> None:
        """Return actors to the available set. Does NOT kill them.

        Idempotent for actors not in ``_in_use`` (e.g. double-release).
        """
        with self._lock:
            for a in actors:
                # Identity-match — actors are Ray ActorHandles which are
                # __eq__'d by underlying actor_id.
                for i, existing in enumerate(self._in_use):
                    if existing is a or existing == a:
                        self._in_use.pop(i)
                        self._available.append(a)
                        break

    def shutdown(self) -> None:
        """Kill every actor in the pool. Called on process exit or
        cluster teardown — NOT per-session. After shutdown, the pool
        must be re-warmed before further use.
        """
        with self._lock:
            for a in self._all:
                try:
                    ray.kill(a)
                except Exception:
                    pass
            self._all = []
            self._available = []
            self._in_use = []
            self._warmed = False

    def stats(self) -> dict:
        """Snapshot for diagnostics."""
        with self._lock:
            return {
                "size": self._size,
                "warmed": self._warmed,
                "available": len(self._available),
                "in_use": len(self._in_use),
            }


# ---------------------------------------------------------------------------
# Module-global registry. Two callers asking for a pool with the same
# (actor_class, config_hash) get the same pool — the typical case for two
# Composites with identical actor templates.
# ---------------------------------------------------------------------------

_POOLS: dict[tuple, ActorPool] = {}
_POOLS_LOCK = threading.Lock()


def _pool_key(actor_class: Any, actor_config: dict) -> tuple:
    cls_name = f"{actor_class.__module__}.{getattr(actor_class, '__name__', repr(actor_class))}"
    cfg_hash = _config_hash(actor_config)
    return (cls_name, cfg_hash)


def _config_hash(config: dict) -> str:
    """Stable string hash of config — JSON-sorted with str fallback for
    non-JSON types. Two configs producing the same canonical JSON share
    a pool, regardless of dict ordering or numeric vs string-numeric."""
    return json.dumps(config, sort_keys=True, default=str)


def get_or_create_pool(
    actor_class: Any,
    actor_config: dict,
    size: int,
    cluster: Any = None,
) -> ActorPool:
    """Return the pool for ``(actor_class, hash(actor_config))``,
    creating it if necessary. Subsequent calls with matching args
    return the same pool — actors are shared across Composites
    transparently.

    If the pool already exists with a smaller size than requested,
    it grows to ``size``. Existing pooled actors keep their state.
    A request for a SMALLER size than the existing pool is honored
    only as a minimum — the pool keeps its current larger size.
    """
    key = _pool_key(actor_class, actor_config)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = ActorPool(actor_class, actor_config, size, cluster)
            _POOLS[key] = pool
    # Grow outside the registry lock so concurrent get_or_create_pool
    # callers don't serialize on a long ray.get(ping) for new actors.
    if pool.size < size:
        pool.grow(size)
    return pool


def shutdown_all_pools() -> None:
    """Tear down every pool. Useful at end-of-test or process-exit.
    After this, any subsequent ``get_or_create_pool`` call creates a
    fresh pool (and pays warmup cost again).
    """
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for p in pools:
        p.shutdown()


def pool_count() -> int:
    """Number of distinct pools currently registered. Diagnostic."""
    with _POOLS_LOCK:
        return len(_POOLS)
