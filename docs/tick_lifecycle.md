# Protocol-batched tick lifecycle

## Problem

`Composite.run()` does per-Process work twice per tick:

1. **Invoke pass** — for each Process in `process_paths`, call `_cached_view(path)`
   to extract the sub-state visible to that Process, then call
   `process_update(...)` (which wraps `process.invoke(state, interval)` in a
   `Defer`).

2. **Apply pass** — for each `Defer` in `self.front`, call `apply_updates(...)`,
   which walks the schema for that single Process's update tree.

When many Processes share a single distributed runtime (the canonical case:
N `RayShadowProcess` or `_ShardFacade` instances all routed through one
`RayProtocolRuntime` / `ShardManager`), this per-Process cost is **deadweight
vs. the actual remote work**. We measured 72 `_ShardFacade` at a 64×64 grid:

- `ray.get` on the batched future list (the actual cluster-side work): **24.6 ms / tick**
- Composite Python overhead (`_cached_view` × 72 + `apply_updates` × 72 + bookkeeping): **196.3 ms / tick**

Composite is paying 8× more overhead than the remote dispatcher pays for compute.
The framework's per-Process cost dominates over the parallelism it's supposed
to coordinate.

## Existing partial fix: `flush_pending`

`Composite._flush_protocol_runtimes()` runs between the invoke pass and apply
pass. Runtimes use it to batch their pending RPCs into one round-trip
(`ray.get([fut1, fut2, ...])`). That eliminates **N round-trips → 1**.

But `flush_pending` does NOT eliminate:
- N state-view extractions in the invoke pass
- N schema walks in the apply pass

Both still scale linearly with the number of Processes, and at scale they
dominate the wall-clock.

## Proposal: `tick_lifecycle` hook

A runtime can opt into taking over the **entire invoke+apply lifecycle** for
the Processes it manages, by exposing a `tick_lifecycle()` method.

When Composite sees a Process with a `_protocol_runtime` that has
`tick_lifecycle`, it:

1. Groups all such Processes by their runtime (multiple Processes → one
   runtime; one tick_lifecycle call per runtime per tick).
2. Skips the per-Process loop for these (no `run_process`, no
   `_cached_view` per Process).
3. Calls the runtime's `tick_lifecycle(...)` once with the group.
4. The runtime returns ONE `Defer` carrying the combined update for all
   its Processes.
5. Composite places it in `self.front` under a common path; `apply_updates`
   walks the schema once for the combined update tree.

Default behavior unchanged for runtimes without `tick_lifecycle`, and for
plain Processes without any runtime — they go through the existing
per-Process path.

### Interface

```python
class TickLifecycleRuntime:
    """A runtime exposing this method opts into Composite-bypass:
    its managed Processes are NOT iterated per-Process by Composite;
    instead, this single method is called per tick per runtime.

    Implementations are responsible for:
      - extracting any state the dispatch needs from `store` (using
        `core.view(...)` or equivalent — schema-aware, single walk)
      - dispatching all per-Process work in parallel
      - building one combined update tree spanning all Process outputs
      - returning a single Defer + the future schedule time

    If unimplemented, Composite falls back to its per-Process loop and
    `flush_pending` (which is unchanged and still useful for batching
    transport-only)."""

    def tick_lifecycle(
        self,
        processes: list[ProcessTickRequest],
        store: dict,
        core: TypeSystem,           # for schema-aware views/apply
        global_time: float,
        end_time: float,
        force_complete: bool,
    ) -> CombinedTickResult: ...


@dataclass
class ProcessTickRequest:
    path: tuple                 # path in the state tree
    instance: Process            # the Process object
    interval: float              # configured tick interval

@dataclass
class CombinedTickResult:
    common_path: tuple           # parent path covering all Processes' outputs
    next_time: float             # global_time at which these Processes are next due
    process_paths: list[tuple]   # paths these results cover (for self.front bookkeeping)
    defer: Defer                 # .get() returns ONE combined update dict tree
                                 # rooted at common_path
```

### Composite changes

In `Composite.run()`:

```python
# Before the per-Process loop, partition by runtime.
runtime_groups: dict[int, tuple[Runtime, list[ProcessTickRequest]]] = {}
plain_paths: list[tuple] = []
for path in self.process_paths:
    process = get_path(self.state, path)
    rt = getattr(process['instance'], '_protocol_runtime', None)
    if rt is not None and hasattr(rt, 'tick_lifecycle'):
        runtime_groups.setdefault(id(rt), (rt, []))[1].append(
            ProcessTickRequest(
                path=path,
                instance=process['instance'],
                interval=process['interval'],
            )
        )
    else:
        plain_paths.append(path)

# Plain processes: existing per-process loop.
for path in plain_paths:
    process = get_path(self.state, path)
    full_step = self.run_process(path, process, end_time, full_step, force_complete)

# Batched groups: one tick_lifecycle call per runtime.
for rt, group in runtime_groups.values():
    result = rt.tick_lifecycle(
        processes=group,
        store=self.state,
        core=self.core,
        global_time=self.state['global_time'],
        end_time=end_time,
        force_complete=force_complete,
    )
    # next_time is the min of per-Process futures inside the runtime;
    # populate self.front so the rest of the run() loop sees these
    # Processes as scheduled together.
    interval = result.next_time - self.state['global_time']
    if interval < full_step:
        full_step = interval
    for p in result.process_paths:
        if p not in self.front:
            self.front[p] = empty_front(self.state['global_time'])
        self.front[p]['time'] = result.next_time
    # Stash the single combined Defer under the common path. apply_updates
    # walks the schema once for it (vs once per Process).
    if result.common_path not in self.front:
        self.front[result.common_path] = empty_front(self.state['global_time'])
    self.front[result.common_path]['time'] = result.next_time
    self.front[result.common_path]['update'] = result.defer
```

`flush_pending` stays exactly as today for runtimes that DON'T implement
`tick_lifecycle` — it batches transport without touching Composite's
per-Process loop.

`apply_updates` is unchanged. The combined Defer's `.get()` returns one big
dict; `apply_updates` walks the schema for it the same way it walks any
other update dict — but only ONCE for the whole batch.

### Why this is the right shape

- **Decision 1 (mutate vs return)**: Returns a Defer that produces ONE update
  dict. Composite still applies via `apply_updates` — same code path,
  same correctness guarantees, same schema integrity. The runtime never
  touches the store directly.

- **Decision 2 (state extraction)**: Runtime is responsible for state
  extraction, and the `core` (TypeSystem) is passed in so it can use
  schema-aware operations. The runtime can do one walk for the whole
  group instead of N walks.

- **Decision 3 (default)**: Runtimes without `tick_lifecycle` get the
  current `flush_pending` behavior. No regression for `RayShadowProcess`
  with a runtime that only implements `flush_pending`.

### Validation

For the `spatio-flux` 64×64 large-grid case (current data):

| Phase | Per-tick (now) | Per-tick (after) |
|---|---|---|
| State view × 72 processes | ~96 ms | one extraction → ~3 ms |
| `ray.get` on batch | ~25 ms | unchanged |
| Apply × 72 deltas | ~75 ms | one apply → ~3 ms |
| **Total per tick** | **~196 ms** | **~31 ms** |

39 ticks → expected wall: 8.6 s → ~1.2 s. Should flip the 64×64 result from
"ties cometspy" to "wins by 6×".

### Out of scope for this change

- Mixed-interval Processes within one runtime group. v1 assumes all
  Processes managed by one runtime share the same interval (true today
  for `_ShardFacade` and `RayShadowProcess`). Mixed-interval batching can
  be a follow-up if a use case emerges.
- Cross-runtime batching (multiple distinct runtimes coalescing into one
  call). v1 batches per-runtime; cross-runtime is a future generalization.
