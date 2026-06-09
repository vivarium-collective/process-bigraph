"""
process.py

Schema type definitions and method specializations for process-bigraph.

This module extends bigraph_schema by defining new schema node classes used by process-bigraph.
"""

import importlib
import threading
import typing
import numpy as np
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema import capture_object_state, restore_object_value
from bigraph_schema.schema import Node, Empty, Float, Wires, Link, Schema, is_schema_field
from bigraph_schema.methods import resolve, realize, realize_link, default, default_link, render, wrap_default
from bigraph_schema.methods import serialize, divide, bundle, apply
from bigraph_schema.methods.bundle import BundleContext


def float_default(value):
    """Factory-of-a-factory: returns a default_factory that produces Float(_default=value)."""

    def float_factory():
        return Float(_default=value)

    return float_factory


@dataclass(kw_only=True)
class StepLink(Link):
    """A link type used for 'step'-level connectivity."""
    priority: Float = field(default_factory=float_default(0.0))
    _triggers: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class ProcessLink(Link):
    """
    A link type for temporal processes, extending the base Link schema with a time interval.
    """
    interval: Float = field(default_factory=float_default(1.0))


@dataclass(kw_only=True)
class Bridge(Node):
    """
    Structural node used by `CompositeLink` that declares wiring from inputs and outputs to internal structure.
    """
    inputs: Wires = field(default_factory=Wires)
    outputs: Wires = field(default_factory=Wires)


@dataclass(kw_only=True)
class Interface(Node):
    """
    Declares process I/O schemas: what a process expects (inputs) and produces (outputs).
    """
    inputs: Schema = field(default_factory=Schema)
    outputs: Schema = field(default_factory=Schema)


@dataclass(kw_only=True)
class CompositeLink(ProcessLink):
    """
    Link type for a composite, a process that can have internal structure and processes.

    In addition to being a ProcessLink (and thus having `interval`), composites carry:
      - schema: the schema for the composite state
      - state: a Node describing/holding the composite's state structure
      - interface: explicit I/O schema boundary for the composite
      - bridge: wiring info mapping internal structure to the interface
    """
    schema: Schema = field(default_factory=Schema)
    state: Node = field(default_factory=Node)
    interface: Interface = field(default_factory=Interface)
    bridge: Bridge = field(default_factory=Bridge)


# --- bigraph_schema method specializations -----------------------------------
# These decorators extend the imported multimethod objects (they do not replace them).

@default.dispatch
def default(schema: StepLink):
    """
    Produce a default value for a StepLink.
    """
    link = default_link(schema)

    link['priority'] = default(schema.priority)

    return link


@default.dispatch
def default(schema: ProcessLink):
    """
    Produce a default value for a ProcessLink.
    """
    link = default_link(schema)

    # Use the default of the Float schema node, not a hard-coded numeric literal.
    link['interval'] = default(schema.interval)

    return link


@realize.dispatch
def realize(core, schema: ProcessLink, state, path=()):
    """
    Realize a ProcessLink against a provided state value.
    """
    # Tree.realize speculatively probes leaf types; only an explicit
    # process declaration (dict with 'address') matches. Return
    # (schema, None, []) for anything else so Tree.realize recurses.
    if not isinstance(state, dict):
        return schema, None, []

    link_schema, link_state, merges = realize_link(core, schema, state, path=path)

    # Realize the interval field (schema is Float). We pass the incoming value if present.
    _, link_state['interval'], _ = realize(
        core,
        link_schema.interval,
        state.get('interval'),
        path + ('interval',))

    return link_schema, link_state, merges


@realize.dispatch
def realize(core, schema: StepLink, state, path=()):
    """
    Realize a StepLink against a provided state value.
    """
    if not isinstance(state, dict):
        return schema, None, []

    link_schema, link_state, merges = realize_link(core, schema, state, path=path)

    _, link_state['priority'], _ = realize(
        core,
        link_schema.priority,
        state.get('priority'),
        path + ('priority',))

    # Preserve _triggers from the declaration — these specify which
    # input ports trigger the step (vs silent inputs).
    if '_triggers' in state and state['_triggers']:
        link_state['_triggers'] = state['_triggers']

    return link_schema, link_state, merges


@render.dispatch
def render(schema: StepLink, defaults=False):
    result = {'_type': 'step'}
    for field_name in schema.__dataclass_fields__:
        if field_name == '_default':
            continue
        value = getattr(schema, field_name)
        result[field_name] = render(value, defaults=defaults)
    return wrap_default(schema, result) if defaults else result


@render.dispatch
def render(schema: ProcessLink, defaults=False):
    result = {'_type': 'process'}
    for field_name in schema.__dataclass_fields__:
        if field_name == '_default':
            continue
        value = getattr(schema, field_name)
        result[field_name] = render(value, defaults=defaults)
    return wrap_default(schema, result) if defaults else result


@apply.dispatch
def apply(schema: Link, state, update, path):
    """Update-driven apply for Link types (process, step, composite).

    The default apply(Node) walks every dataclass field of the schema and
    recurses with `update.get(key)` — passing None for fields the caller
    didn't include. Subschemas like Wires error or clobber on None, so a
    partial update like `{'interval': 0.5}` either crashes or wipes out
    inputs/outputs/address/config. Iterating the update keys instead lets
    callers update interval, inputs, outputs, etc. independently.
    """
    if update is None:
        return state, []
    if not isinstance(state, dict) or not isinstance(update, dict):
        return update, []

    result = dict(state)
    merges = []
    for key, update_value in update.items():
        if not is_schema_field(schema, key):
            result[key] = update_value
            continue
        subschema = getattr(schema, key, None)
        if subschema is None:
            sub_result, submerges = apply(
                Node(), state.get(key), update_value, path + (key,))
        else:
            sub_result, submerges = apply(
                subschema, state.get(key), update_value, path + (key,))
        result[key] = sub_result
        merges += submerges

    return result, merges


def register_types(core):
    """
    Register process-bigraph schema node classes
    """
    core.register_types({
        'step': StepLink,
        'process': ProcessLink,
        'interface': Interface,
        'bridge': Bridge,
        'composite': CompositeLink,
        'shared_process': SharedProcess,
        'shared_process_ref': SharedProcessRef})

    return core


# ============================================================================
# SharedProcess / SharedProcessRef — process instances stored in a global
# registry, referenceable by ID from elsewhere in the schema.
#
# Use case: the Requester/Evolver pattern wires the *same* process
# instance through two distinct steps. Each step's config carries a
# ``SharedProcessRef`` that resolves at realize time to the live
# instance registered under its ID by the corresponding
# ``SharedProcess`` entry in the document's ``process`` store.
# ============================================================================

# Per-``core`` shared-process registries are the primary store: each
# Composite owns its own ``core`` (``allocate_core`` returns a fresh core
# per call), so scoping the registry on the core isolates one simulation's
# shared processes from another's even when several Composites run in the
# same process or across the ThreadPoolExecutor used by ``parallel_steps`` /
# ``parallel_processes``. The module-level dict below is only a fallback for
# the (unusual) case where a dispatch is reached without a core, and is what
# ``clear_shared_processes`` / ``get_shared_process`` operate on for
# backwards compatibility.
_shared_processes: typing.Dict[str, object] = {}

# Guards lazy creation of per-core registries and writes into both the
# per-core and the module-level fallback dicts. Registry mutation happens at
# realize/divide time (composite init / load / division), not on the
# parallel update hot path, so contention is negligible.
_shared_processes_lock = threading.Lock()

# Attribute used to stash the per-core registry on a ``core`` instance.
_CORE_REGISTRY_ATTR = '_shared_process_registry'


def _core_shared_processes(core) -> typing.Dict[str, object]:
    """Return the shared-process registry scoped to ``core``.

    Created lazily on first use and stored on the core instance, so it is
    garbage-collected with the core (and therefore with its Composite) and
    never leaks results into a different simulation's core. Falls back to the
    module-level dict when ``core`` is ``None`` (no scope available).
    """
    if core is None:
        return _shared_processes
    registry = getattr(core, _CORE_REGISTRY_ATTR, None)
    if registry is None:
        with _shared_processes_lock:
            registry = getattr(core, _CORE_REGISTRY_ATTR, None)
            if registry is None:
                registry = {}
                setattr(core, _CORE_REGISTRY_ATTR, registry)
    return registry


def clear_shared_processes(core=None):
    """Clear a shared-process registry (e.g. between simulations).

    With ``core`` given, clears that core's scoped registry; otherwise
    clears the module-level fallback dict.
    """
    if core is None:
        with _shared_processes_lock:
            _shared_processes.clear()
    else:
        registry = getattr(core, _CORE_REGISTRY_ATTR, None)
        if registry is not None:
            with _shared_processes_lock:
                registry.clear()


def get_shared_process(process_id, core=None):
    """Look up a registered shared-process instance by id.

    Prefers ``core``'s scoped registry when provided; otherwise consults the
    module-level fallback dict.
    """
    if core is not None:
        instance = _core_shared_processes(core).get(process_id)
        if instance is not None:
            return instance
    return _shared_processes.get(process_id)


@dataclass(kw_only=True)
class SharedProcess(Node):
    """A shared process instance declared in the document's ``process`` store.

    Document form::

        {"_type": "shared_process",
         "address": "local:!my.module.MyProcess",
         "config": { ... }}

    On ``realize()``, the class is imported from ``address``, instantiated
    with ``config``, wrapped as ``(instance,)``, and registered in the
    global ``_shared_processes`` dict keyed by the last path segment.
    """
    pass


@dataclass(kw_only=True)
class SharedProcessRef(Node):
    """Reference to a shared process instance by ID.

    Used in step config_schema so realize() resolves the reference to the
    live instance before ``__init__`` is called.

    Document forms::

        {"process": {"_type": "shared_process_ref", "process_id": "my-proc"}}
        {"process": "my-proc"}             # bare string when type is declared
    """
    pass


def _class_address_from_instance(instance):
    """Generate a local:! address from an instance's class."""
    cls = type(instance)
    return f'local:!{cls.__module__}.{cls.__name__}'


@dispatch
def render(schema: SharedProcess, defaults=False):
    return 'shared_process'


@dispatch
def render(schema: SharedProcessRef, defaults=False):
    return 'shared_process_ref'


@realize.dispatch
def realize(core, schema: SharedProcess, state, path=()):
    # Already realized — (instance,) tuple
    if isinstance(state, tuple) and len(state) > 0:
        return schema, state, []
    if not isinstance(state, dict):
        return schema, state, []

    address = state.get('address')
    config = state.get('config', {})
    process_id = path[-1] if path else state.get('process_id')

    if address is None:
        return schema, state, []

    # Import the class from address (``local:!module.Class``)
    if isinstance(address, str) and address.startswith('local:!'):
        module_path, class_name = address[7:].rsplit('.', 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    else:
        return schema, state, []

    # Realize the config through the class's config_schema so typed
    # fields (Quantity, Function, custom types) get reconstructed.
    config_schema = getattr(cls, 'config_schema', None)
    if config_schema:
        _, config, _ = core.realize(config_schema, config)

    instance = cls(config)

    # Set core so serialize can access config_schema
    if not hasattr(instance, 'core') or instance.core is None:
        instance.core = core

    # Restore process-internal RandomState if a checkpoint captured
    # one. Cross-gen daughter loads should strip ``rng_state`` from
    # the saved bundle before construction (so the daughter starts
    # fresh from a freshly-seeded process); mid-tick checkpoints
    # leave it in for bit-for-bit continuation.
    rng_state = state.get('rng_state')
    if (rng_state and hasattr(instance, 'random_state')
            and isinstance(instance.random_state, np.random.RandomState)):
        try:
            instance.random_state.set_state((
                rng_state['alg'],
                np.asarray(rng_state['key'], dtype=np.uint32),
                int(rng_state['pos']),
                int(rng_state['has_gauss']),
                float(rng_state['cached_gauss']),
            ))
        except Exception as e:
            print(f"[SharedProcess] failed to restore rng_state for "
                  f"{process_id}: {e}", flush=True)

    # Restore mid-tick bookkeeping attrs that __init__ resets but the
    # running simulation mutates. Keeping them in sync with the saved
    # snapshot is what lets load+run match continue-in-place.
    internal = state.get('internal_state') or {}
    for attr_name, saved_value in internal.items():
        try:
            setattr(instance, attr_name, restore_object_value(saved_value))
        except Exception as e:
            print(f"[SharedProcess] failed to restore {attr_name} for "
                  f"{process_id}: {e}", flush=True)

    if process_id is not None:
        # ``_core_shared_processes`` does its own locked lazy-create; the
        # dict item assignment itself is atomic under CPython, so no extra
        # lock is needed here (and acquiring one would re-enter the
        # non-reentrant lock taken inside the helper → deadlock).
        _core_shared_processes(core)[process_id] = instance
        # Stamp the id on the instance so serialize/bundle of a
        # SharedProcessRef can reverse-resolve it WITHOUT consulting any
        # (potentially cross-simulation) global registry.
        instance._shared_process_id = process_id

    instance._shared_address = address if isinstance(address, str) else (
        f"local:!{cls.__module__}.{cls.__name__}")

    return schema, (instance,), []


@dispatch
def serialize(schema: SharedProcess, state):
    """Serialize a SharedProcess back to its declaration dict."""
    if isinstance(state, tuple) and len(state) > 0:
        instance = state[0]
        address = getattr(instance, '_shared_address',
                          _class_address_from_instance(instance))
        config = instance.parameters if hasattr(instance, 'parameters') else {}
        instance_core = getattr(instance, 'core', None)
        raw_schema = getattr(instance, 'config_schema', None)
        if instance_core and raw_schema:
            config_schema = instance_core.access(raw_schema)
            config = serialize(config_schema, config)
        result = {
            '_type': 'shared_process',
            'address': address,
            'config': config,
        }
        rng = getattr(instance, 'random_state', None)
        if isinstance(rng, np.random.RandomState):
            alg, key, pos, has_gauss, cached = rng.get_state()
            result['rng_state'] = {
                'alg': alg,
                'key': key.tolist(),
                'pos': int(pos),
                'has_gauss': int(has_gauss),
                'cached_gauss': float(cached),
            }
        import os
        internal = capture_object_state(
            instance, debug=bool(os.environ.get('INTERNAL_DEBUG')))
        if internal:
            result['internal_state'] = internal
        return result
    if isinstance(state, dict):
        return state
    return state


@divide.dispatch
def divide(schema: SharedProcess, state, context=None, path=(), rng=None):
    """Each daughter gets a FRESH process instance.

    Without this, ``divide(Node)`` would share the mother's instance
    by reference between both daughters. We serialize the mother's
    declaration (address + config), drop the mid-tick rng/internal
    snapshots (they belong to mother), and return that dict for each
    daughter. ``realize(SharedProcess)`` re-instantiates from
    address+config on the next pass.
    """
    if state is None:
        return None, None
    declaration = serialize(schema, state)
    if isinstance(declaration, dict):
        declaration = {k: v for k, v in declaration.items()
                       if k not in ('rng_state', 'internal_state')}
        return dict(declaration), dict(declaration)
    return state, state


@dispatch
def bundle(schema: SharedProcess, state, context: typing.Optional[BundleContext] = None):
    """Bundle a SharedProcess: produce the declaration dict, bundling
    arrays inside the config through the process's config_schema so
    save_bundle externalizes large arrays to Parquet."""
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) > 0:
        instance = state[0]
        address = getattr(instance, '_shared_address',
                          _class_address_from_instance(instance))
        config = instance.parameters if hasattr(instance, 'parameters') else {}
        instance_core = getattr(instance, 'core', None)
        raw_schema = getattr(instance, 'config_schema', None)
        if instance_core and raw_schema:
            config_schema = instance_core.access(raw_schema)
            config = bundle(config_schema, config, context)
        result = {
            '_type': 'shared_process',
            'address': address,
            'config': config,
        }
        rng = getattr(instance, 'random_state', None)
        if isinstance(rng, np.random.RandomState):
            alg, key, pos, has_gauss, cached = rng.get_state()
            result['rng_state'] = {
                'alg': alg,
                'key': key.tolist(),
                'pos': int(pos),
                'has_gauss': int(has_gauss),
                'cached_gauss': float(cached),
            }
        import os
        internal = capture_object_state(
            instance, debug=bool(os.environ.get('INTERNAL_DEBUG')))
        if internal:
            result['internal_state'] = internal
        return result
    if isinstance(state, dict):
        return state
    return serialize(schema, state)


def _lookup_shared_process_id(state):
    """Reverse-lookup a process instance's id in ``_shared_processes``.

    SharedProcess.realize registers each instance keyed by its path
    segment, so we can resolve the stable string id even when the
    instance itself doesn't carry one (or carries an ambiguous name).
    """
    if isinstance(state, tuple) and len(state) > 0:
        state = state[0]
    # Prefer the id stamped on the instance at realize time — this avoids
    # depending on any global/per-core registry from a context (serialize/
    # bundle) that has no ``core`` handle.
    stamped = getattr(state, '_shared_process_id', None)
    if isinstance(stamped, str):
        return stamped
    for pid, inst in _shared_processes.items():
        if inst is state:
            return pid
    return None


@dispatch
def serialize(schema: SharedProcessRef, state):
    """Serialize a SharedProcessRef: the process name (or bare id)."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return state
    pid = _lookup_shared_process_id(state)
    if pid is not None:
        return pid
    if hasattr(state, 'name') and isinstance(state.name, str):
        return state.name
    return str(state)


@bundle.dispatch
def bundle(schema: SharedProcessRef, state, context: typing.Optional[BundleContext] = None):
    """Bundle a SharedProcessRef as its string process_id.

    Without this dispatch the ``bundle(Node, ...)`` fallback writes
    ``str(instance)`` (the Python repr), which can't round-trip.
    """
    return serialize(schema, state)


@realize.dispatch
def realize(core, schema: SharedProcessRef, state, path=()):
    # Already resolved to an instance
    if not isinstance(state, (str, dict)):
        return schema, state, []

    if isinstance(state, dict):
        process_id = state.get('process_id')
    else:
        process_id = state

    if process_id is None:
        return schema, state, []

    registry = _core_shared_processes(core)
    instance = registry.get(process_id)
    if instance is None:
        # Fall back to the module-level registry for any path that realized
        # the SharedProcess without a core scope.
        instance = _shared_processes.get(process_id)
    if instance is None:
        available = sorted(set(registry.keys()) | set(_shared_processes.keys()))
        raise RuntimeError(
            f"SharedProcessRef at {path}: process_id '{process_id}' "
            f"not found. Available: {available}. "
            f"Ensure SharedProcess entries are realized before refs.")
    return schema, instance, []
