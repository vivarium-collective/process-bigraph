# Distributed lifecycles in process_bigraph

## Premise

`process_bigraph` was designed around `Composite` as the unit of simulation:
construct a Composite from a document, `composite.run(t)`, throw it away.
For local single-machine work this is the right abstraction — every per-sim
cost (process construction, schema realization) is paid once per sim and
amortizes naturally.

For **distributed** work, this premise breaks down. A simulation backed by
Ray (or another remote dispatcher) has expensive resources whose cost is
roughly linear in the number of simulations *only because we couple their
lifetime to Composite's lifetime*:

- **Cluster** (EC2 instances + Ray daemon): minutes to bring up.
- **Pool** of remote actors (Ray handles + loaded scientific models like
  cobra): seconds to tens of seconds per actor cold-start.
- **Sim** (one `composite.run(t)`): the actual work, often shorter than
  the setup cost.
- **Tick** (one invoke+apply pass inside the sim): the unit of parallel
  dispatch.

Today, every Composite that wants distributed execution pays *all four*
costs. This makes:

- Repeated benchmark sweeps (the spatio-flux dFBA comparison: 5 grid sizes
  × ~10s actor warmup each = 50s of cobra cold-load wasted).
- Long-running scientific services that run many sims back-to-back
  (architecturally infeasible — every request costs minutes).
- Iteration during development (each `composite.run` rebuilds the world).

## The lifecycle layering

The fix is to **separate the four lifetimes** and let each be managed by
its own context, longest to shortest:

```
┌────────────────────────────────────────────────────────────┐
│  CLUSTER  (EC2 instances + Ray daemon)                     │
│  context: EC2SSMRayCluster (or LocalRayCluster, etc.)     │
│  cost: minutes; amortized over hours of work               │
│                                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  POOL  (long-lived Ray actors with loaded models)     │ │
│  │  context: ActorPool                                   │ │
│  │  cost: seconds-to-tens; amortized over many sims      │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────────┐  │ │
│  │  │  SESSION  (one Composite's claim on N actors)   │  │ │
│  │  │  context: Session (replaces ShardManager today) │  │ │
│  │  │  cost: ms; cheap claim/release of pool slots    │  │ │
│  │  │                                                 │  │ │
│  │  │  ┌───────────────────────────────────────────┐  │  │ │
│  │  │  │  TICK  (invoke + flush + apply_updates)   │  │  │ │
│  │  │  │  context: tick_lifecycle hook (#22)       │  │  │ │
│  │  │  │  cost: ms-to-seconds per tick             │  │  │ │
│  │  │  └───────────────────────────────────────────┘  │  │ │
│  │  └─────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

## Concrete contracts

### Cluster

A `RayCluster` is a context manager over the underlying compute resources
(EC2 instances, k8s pods, etc.). It does NOT manage actors. It just
ensures Ray is up and reachable.

```python
class RayCluster(Protocol):
    def __enter__(self) -> "RayCluster": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    @property
    def address(self) -> str: ...        # e.g. "ray://1.2.3.4:10001"
    @property
    def cluster_resources(self) -> dict: ... # {"CPU": 72, "memory": ...}


class EC2SSMRayCluster(RayCluster):
    """Provisions EC2 + Ray via SSM (today's scripts/ec2_cluster.py
    refactored as a context-manager class)."""
    def __init__(self, *, stack: str, image_uri: str, n_workers: int = 4): ...


class LocalRayCluster(RayCluster):
    """No-op for already-up Ray (e.g. dev work). Connects via ray.init()."""
```

### Pool

An `ActorPool` is a long-lived collection of remote actors of one
template type, with their expensive state (cobra-loaded models, JIT
caches, GPU contexts) paid once and reused. The pool is decoupled from
any specific Composite.

```python
class ActorPool:
    def __init__(self,
                 actor_class: type,
                 actor_config: dict,
                 size: int,
                 cluster: RayCluster | None = None):
        """size = how many actors to keep warm. cluster determines
        which Ray instance they live on."""

    def warm(self) -> None:
        """Spawn `size` actors and run their cold-start sequentially or
        in parallel. Returns when all are responsive."""

    def acquire(self, n: int = 1) -> list:
        """Return n actor handles for use by a Session. Blocks if
        fewer than n are available."""

    def release(self, actors: list) -> None:
        """Return actors to the available set. Does NOT kill them."""

    def shutdown(self) -> None:
        """Kill all actors. Called on cluster teardown, not per-sim."""
```

The pool is keyed by `(actor_class, hash(actor_config))` at module level
(similar to how `RayProtocolRuntime._RUNTIMES` works today, but a level
deeper). Multiple Composites share pools transparently.

### Session

A `Session` is the per-Composite claim on a slice of an ActorPool. It
replaces today's `ShardManager`. Session enter is cheap (claim N actors;
optionally call `actor.reconfigure(...)` to rebind per-sim parameters);
session exit returns actors to pool.

```python
class Session:
    def __init__(self, pool: ActorPool, n_actors: int, sim_config: dict):
        ...

    def __enter__(self) -> "Session":
        self.actors = pool.acquire(n_actors)
        # Optionally reconfigure per-sim: e.g., rebind cell_keys without
        # reloading cobra.
        ray.get([a.reconfigure.remote(self.sim_config) for a in self.actors])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pool.release(self.actors)

    # Process-side facades (e.g., _ShardFacade) look up their actor
    # via this session, not via a global registry.
```

### Process.reconfigure

For a Session to claim and re-purpose pool actors without paying their
cold-start cost, the actor's underlying Process needs to accept new
configuration without re-running `initialize`. This adds a single hook:

```python
class Process:
    def initialize(self, config: dict) -> None: ...      # already exists
    def reconfigure(self, config: dict) -> None: ...     # NEW

    # Default implementation: re-run initialize. Subclasses override
    # for cheap reconfiguration that preserves expensive state.
```

For `ShardedDFBA` specifically: `reconfigure(cell_keys=[...])` rebinds
the cell list without reloading cobra or rebuilding the LP — the
expensive-to-load cobra Model is preserved.

### Tick (recap from #22)

The tick lifecycle hook (`Composite._run_tick_lifecycle`) is unchanged
from the previous design doc. The Session passes its claimed actors to
the tick_lifecycle implementation, which dispatches in batched parallel.

## Migration path

This is invasive but additive. The old `ShardManager(...)` API can be
preserved as a thin shim over the new layers:

```python
# Old (still works for single-shot scripts):
with ShardManager(model_id="ecoli core", n=64, n_shards=8) as mgr:
    sim = Composite(...)
    sim.run(4.0)

# New (efficient for repeated runs):
with EC2SSMRayCluster(stack="prod") as cluster:
    pool = ActorPool(ShardActor, {"model_id": "ecoli core"}, size=72,
                     cluster=cluster)
    pool.warm()  # paid ONCE
    for grid_size in [8, 16, 32, 64, 128]:
        with Session(pool, n_actors=72,
                     sim_config={"cell_keys": gen_keys(grid_size)}) as session:
            sim = Composite(...)
            sim.run(4.0)
        # session exits cheaply; actors return to pool with cobra still loaded
    # pool exits; actors killed only here
# cluster exits; EC2 terminated only here
```

Old call sites continue to work because `ShardManager` becomes a
convenience wrapper that creates a single-use Pool + Session under the
hood. The cost is the same as today; the new path is opt-in for users
who want the speedup.

## Implementation order (concrete)

1. **Process.reconfigure hook** (process-bigraph upstream).
   Smallest piece. Default impl re-runs initialize; subclasses override.
   Validation: existing tests still pass.

2. **ActorPool** (process-bigraph upstream, in `protocols/pool.py`).
   Module-global registry of pools keyed by (actor_class, config_hash).
   `warm()`, `acquire()`, `release()`, `shutdown()`.
   Validation: unit test that two Composites share one pool.

3. **Session** (process-bigraph upstream, in `protocols/session.py`).
   Context manager that claims/releases pool actors.
   Validation: unit test that session enter/exit doesn't kill actors.

4. **Refactor ShardManager** (spatio-flux): ShardManager becomes a
   convenience wrapper over Pool + Session for single-shot use. Gain:
   when used in a loop (the report's grid sweep), the second iteration
   onward is fast.

5. **Refactor `EC2SSMRayCluster`** (process-bigraph upstream extras
   `process_bigraph[ec2-ssm]`, lifting today's `scripts/ec2_cluster.py`):
   becomes a `RayCluster` context manager. Validation: end-to-end
   spatio-flux comparison runs with cluster context preserved across
   grids.

6. **Re-enable tick_lifecycle** with a proper batched view: now that
   we have Session + Pool, the Session knows which actors it owns and
   their (cell_keys, config) — the tick_lifecycle implementation can do
   one batched read per tick. Closes task #24.

## What this gives the user

- **Reports / benchmarks**: actor cold-start paid once across all grid
  sizes. 5-grid sweep drops from ~3 min to ~30s of setup overhead.

- **Long-running services**: actor pools live across requests. New
  request = cheap session, no cobra reload. Sub-second cold path.

- **Development iteration**: connecting to a running cluster + warmed
  pool means re-running a sim is sub-second instead of minutes.

- **Framework users in general**: the lifecycle layering is the right
  mental model for any distributed workload, not just dFBA. Other
  protocols (REST batching, GPU-resident models, etc.) get the same
  pool-amortization for free.

## Out of scope for this round

- Cross-cluster pools (one pool spanning multiple clusters): solvable
  but not needed for v1.
- Pool autoscaling (resize pool based on demand): pools are fixed-size
  in v1; resize requires manually growing the pool.
- Heterogeneous actor classes in one pool: a pool is one actor class;
  multiple types = multiple pools.
