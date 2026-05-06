"""Session — a Composite's claim on a slice of an ActorPool for one
simulation. Replaces the per-Composite "spawn-then-kill" pattern that
ties actor lifetime to sim lifetime.

A Session enter is cheap (claim N actor handles from the pool, optionally
``reconfigure(sim_config)`` them); a Session exit is also cheap (return
handles to the pool — no kill). The actors' expensive state (loaded
scientific models, JIT caches, GPU contexts, persistent solver bases)
is preserved across sessions so subsequent Composites pay no
cold-start cost.

See ``doc/distributed_lifecycles.md`` for the layering this fits into:

    cluster ⊃ pool ⊃ session ⊃ tick

Typical usage with the rest of the lifecycle stack:

    with EC2SSMRayCluster(...) as cluster:
        pool = get_or_create_pool(MyActor, {...}, size=72, cluster=cluster)
        pool.warm()                                   # paid once
        for sim_params in many_sims:
            with Session(pool, n_actors=8,
                         sim_config={"cell_keys": ...}) as session:
                composite = Composite(...)
                composite.run(t)
            # session exits — actors return to pool with state intact
        # pool stays alive — many Composites just used it
    # cluster torn down here
"""
from __future__ import annotations

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
            "Session requires the optional `ray` dependency. "
            "Install with: pip install process-bigraph[ray]"
        ) from _RAY_IMPORT_ERROR


class Session:
    """One Composite's claim on N actors from an ActorPool.

    Use as a context manager. ``__enter__`` claims actors and optionally
    re-binds their per-sim config; ``__exit__`` returns them to the pool.
    No actor spawn or kill happens at session boundaries.

    For ``reconfigure`` to have any effect, the underlying Process /
    actor class must override ``Process.reconfigure(config)`` (default
    re-runs ``initialize``, which defeats the pool's whole point).
    Subclasses that override ``reconfigure`` to update only the cheap
    per-sim fields are what makes the pool actually amortize.
    """

    def __init__(
        self,
        pool: Any,
        n_actors: int,
        sim_config: Optional[dict] = None,
        reconfigure: bool = True,
    ):
        """
        Args:
            pool: An ActorPool (already warmed; if not, ``acquire``
                will warm it on demand).
            n_actors: How many actors this session claims.
            sim_config: Optional per-sim configuration passed to
                ``actor.reconfigure(config)`` on session enter.
            reconfigure: If True (default) and sim_config is non-empty,
                the session calls ``reconfigure`` on each actor. Set
                False to skip — useful when the pool is already
                configured for the sim or for raw actor handles that
                don't expose reconfigure.
        """
        _require_ray()
        self._pool = pool
        self._n_actors = int(n_actors)
        self._sim_config = dict(sim_config) if sim_config else {}
        self._reconfigure = bool(reconfigure)
        self._actors: Optional[list] = None

    def __enter__(self) -> "Session":
        self._actors = self._pool.acquire(self._n_actors)
        if self._reconfigure and self._sim_config:
            # All reconfigures fire in parallel; ray.get blocks until
            # the slowest one finishes. Fast in practice because
            # reconfigure is by design cheap (no model reload).
            ray.get([
                a.reconfigure.remote(self._sim_config)
                for a in self._actors
            ])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._actors is not None:
            self._pool.release(self._actors)
            self._actors = None
        # Don't suppress exceptions.
        return None

    @property
    def actors(self) -> list:
        """The actor handles claimed by this session. Raises if not in
        an ``__enter__``/``__exit__`` block."""
        if self._actors is None:
            raise RuntimeError(
                "Session.actors accessed outside of `with Session(...)`"
            )
        return list(self._actors)

    @property
    def n_actors(self) -> int:
        return self._n_actors

    @property
    def sim_config(self) -> dict:
        return dict(self._sim_config)

    @property
    def pool(self) -> Any:
        return self._pool
