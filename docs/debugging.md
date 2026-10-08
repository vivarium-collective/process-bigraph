# Debugging a composite

Three opt-in tools help when a simulation runs but gives the wrong answer, or
runs slowly. All are off by default and cost nothing when off.

| Question | Tool | How to turn it on |
| --- | --- | --- |
| Is a process returning a malformed update? | `check_updates` | Composite config `check_updates: raise` or `record` |
| Is a process breaking its declared contract? | `contract_strict` | Composite config `contract_strict: raise` or `record` |
| Which process is slow? | `timing_summary()` | `PROCESS_BIGRAPH_PROFILE_PROCESSES=1`, or `composite._profile_per_process = True` |

## Checking process updates (`check_updates`)

Every update a Process or Step returns is checked before it is applied to
the state. Three problems are reported:

- **`undeclared_port`**: the update writes a port that is not in `outputs()`.
  Without the check the value is silently dropped, so a misspelled port name
  looks like a process that never changes anything.
- **`non_finite`**: a NaN or infinite number appears anywhere in the update,
  including inside numpy arrays. Without the check NaN spreads through every
  downstream store with no error.
- **`type_mismatch`**: the value does not fit the port's schema (for example a
  string sent to a `float` port). Without the check the failure surfaces later
  as an unrelated exception that does not name the responsible process.

```python
sim = Composite({'state': doc, 'check_updates': 'raise'}, core=core)
sim.run(10.0)
# UpdateViolation: Decay at 'decay' returned a bad update:
#   [undeclared_port] wrote to port 'conc', which is not declared in outputs()
#   (declared: concentration); the value would be dropped
```

`raise` stops at the first bad update. `record` keeps running and collects
every problem in `sim.update_violations` (capped at 10,000 entries; extra ones
are counted in `sim.update_violations_dropped`), and emits an
`update.violation` event when an event sink is configured:

```python
sim = Composite({'state': doc, 'check_updates': 'record'}, core=core)
sim.run(10.0)
for v in sim.update_violations:
    print(v['global_time'], '/'.join(map(str, v['path'])), v['violations'])
```

To turn checking on for every composite built from one core, set
`core.check_updates = 'record'` (or `'raise'`).

Structural updates such as `{'_add': ...}` or `{'_remove': ...}` are not
type-checked, because they are instructions to the type rather than values of
it; they are still scanned for NaN and infinity. In `record` mode a
type-mismatched value is still applied afterwards, so it may also raise from
`apply`; the violation is recorded first, so `sim.update_violations` names the
process responsible.

## Checking contracts (`contract_strict`)

Processes that declare a contract (bounds on ports, plus `pre`, `post` and
`invariant` conditions) can have it checked on every invoke. See
`process_bigraph/contract_strict.py`.

## Finding slow processes (`timing_summary`)

`Composite.timing_summary()` splits the last `run()` into time spent inside
process `invoke()` calls versus framework overhead. Per-process times are
collected when per-process profiling is on:

```python
sim._profile_per_process = True      # or PROCESS_BIGRAPH_PROFILE_PROCESSES=1
sim.run(10.0)
print(sim.timing_summary().format())
```
