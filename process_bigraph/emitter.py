'''
===========================
Emitter Utilities & Classes
===========================

Emitters are steps that observe a composite simulation's state and emit data to an external source
(e.g., console, memory, or file). This module provides tools to:
- Define emitter steps programmatically
- Insert emitters into a running composite
- Collect data from emitter steps
- Implement concrete in-tree emitters (RAM, console, JSON)

``SQLiteEmitter`` (and ``ParquetEmitter``) now live in the focused
``pbg-emitters`` library (https://github.com/vivarium-collective/pbg-emitters)
and are re-exported from the bottom of this module for back-compat — see
the re-export shim at the end of the file.
'''

import copy
import json
import os
import uuid
from typing import Dict

import numpy as np

from bigraph_schema import Edge, get_path, is_schema_key, set_path
from process_bigraph.composite import Step, find_instance_paths


# ==========================
# Emitter Spec Construction
# ==========================

def anyize_paths(tree):
    '''Recursively convert all leaves of a nested path tree to "node".'''
    if isinstance(tree, dict):
        return {key: anyize_paths(value) for key, value in tree.items()}
    else:
        return 'node'

def emitter_from_wires(wires, address='local:RAMEmitter', subsample=1):
    '''Create an emitter step spec from wire mappings.

    ``subsample`` (RAMEmitter / SQLiteEmitter only): record every
    Nth composite tick. Default 1 records every tick.
    '''
    config = {'emit': anyize_paths(wires)}
    if subsample is not None and int(subsample) > 1:
        config['subsample'] = int(subsample)
    return {
        '_type': 'step',
        'address': address,
        'config': config,
        'inputs': wires}

def collect_input_ports(state, path=None):
    '''Recursively collect all valid input ports from state tree, skipping processes and schema keys.'''
    process_paths = find_instance_paths(state, 'process_bigraph.composite.Process')
    step_paths = find_instance_paths(state, 'process_bigraph.composite.Step')
    path = path or ()
    input_ports = {}
    for key, value in state.items():
        full_path = path + (key,) if path else (key,)
        full_key = '/'.join(full_path)

        if is_schema_key(key):
            continue
        if full_path in process_paths or full_path in step_paths:
            continue
        if isinstance(value, dict):
            input_ports.update(collect_input_ports(value, full_path))
        else:
            input_ports[full_key] = list(full_path)
    return input_ports

def generate_emitter_state(composite, emitter_mode='all', address='local:RAMEmitter'):
    '''
    Generate emitter state for a given composite and mode.
    Modes:
        - "all": observe all valid inputs
        - "none": observe nothing
        - {"paths": [...]}: custom paths to observe

    The node comes from :func:`emitter_from_wires` so a generated sink and a
    declared one are built by the same constructor.
    '''
    input_ports = {}

    if emitter_mode == 'all':
        input_ports = collect_input_ports(composite.state)
    elif emitter_mode == 'none':
        input_ports = {}
    elif isinstance(emitter_mode, dict) and 'paths' in emitter_mode:
        for path in emitter_mode['paths']:
            if isinstance(path, str):
                input_ports[path] = [path]
            elif isinstance(path, list):
                input_ports[path[0]] = path
    else:
        raise ValueError(f'Invalid mode: {emitter_mode}.')

    if 'global_time' not in input_ports:
        input_ports['global_time'] = ['global_time']

    return emitter_from_wires(input_ports, address=address)


def _known_emitter_addresses(registered):
    '''The registered link names that look like emitters (for error text).

    Falls back to *all* registered names when none end in ``Emitter`` so the
    message is never empty.
    '''
    emitters = sorted(
        key for key in registered
        if isinstance(key, str) and key.endswith('Emitter'))
    if emitters:
        return emitters
    return sorted(key for key in registered if isinstance(key, str))


def _resolve_declared_address(address, name, registered, fallback, on_unknown_address):
    '''Resolve a declared emitter address against the core's link registry.

    When ``name`` is registered, returns it unchanged. When it is *not*, policy
    decides: ``"raise"`` (default) fails loud, naming the bad address and the
    known registered emitter addresses; ``"ram"`` restores the historical
    silent degrade to ``fallback`` (``local:RAMEmitter``) so the composite
    still builds.
    '''
    if on_unknown_address not in ('raise', 'ram'):
        raise ValueError(
            "on_unknown_address must be 'raise' or 'ram', "
            f"got {on_unknown_address!r}")
    if registered is None or name in registered:
        return address, name
    if on_unknown_address == 'ram':
        return fallback, fallback.split(':', 1)[-1]
    raise ValueError(
        f"unknown emitter address {address!r}: {name!r} is not registered in "
        f"the core's link registry. Known emitter addresses: "
        f"{_known_emitter_addresses(registered)}. Pass on_unknown_address='ram' "
        f"to fall back to {fallback!r} instead.")


def emitter_node_from_declaration(
        decl,
        run_id=None,
        out_dir=None,
        core=None,
        fallback='local:RAMEmitter',
        on_unknown_address='raise'):
    '''Materialize one declared emitter (``{address, config?, paths?}``) into
    a step node.

    A declaration names *what to observe*; the emit schema and topology depend
    on the composite's shape, so they are computed here. Each ``paths`` entry
    (slash- or dot-joined) becomes one wired column; ``global_time`` is always
    emitted so trajectories have a time axis and the step re-fires every tick.

    The node itself comes from :func:`emitter_from_wires` — the one emitter
    constructor — so a declared sink and a generated one are the same shape.

    ``on_unknown_address`` sets the policy when the declared address is not in
    the core's link registry: ``"raise"`` (the default) fails loud at build
    time, naming the bad address and the known emitter addresses; ``"ram"`` is
    the explicit opt-in to the historical silent degrade to ``fallback``
    (``local:RAMEmitter``) so the composite still builds. With no ``core`` the
    registry is unknown and neither branch fires.
    '''
    address = decl.get('address') or fallback
    name = address.split(':', 1)[-1]
    registered = getattr(core, 'link_registry', None) if core is not None else None
    address, name = _resolve_declared_address(
        address, name, registered, fallback, on_unknown_address)

    wires = {}
    for path in decl.get('paths') or []:
        parts = [part for part in str(path).replace('.', '/').split('/') if part]
        if not parts:
            continue
        wires['_'.join(parts)] = parts
    wires.setdefault('global_time', ['global_time'])

    node = emitter_from_wires(wires, address=address)

    # Declared config layers *under* the computed emit schema.
    declared_config = dict(decl.get('config') or {})
    node['config'] = {**declared_config, **node['config']}

    # Run-specific partitioning for hive-partitioned parquet sinks; other
    # sinks keep their declared config untouched.
    if name.endswith('ParquetEmitter'):
        if out_dir is not None:
            node['config']['out_dir'] = str(out_dir)
        if run_id is not None:
            node['config'].setdefault('partitioning_keys', ['experiment_id'])
            metadata = dict(node['config'].get('metadata') or {})
            metadata.setdefault('experiment_id', run_id)
            node['config']['metadata'] = metadata

    return node


def _node_is_emitter(node, core=None):
    '''True when ``node`` is an edge node that resolves to an Emitter.

    Works on a *raw* document node (not yet realized): an already-built
    ``instance`` that is an Emitter, an ``address`` the core resolves to an
    Emitter subclass, or — as a last resort when no core is on hand — an
    address whose class name ends in ``Emitter``.
    '''
    if not isinstance(node, dict) or node.get('_type') not in ('step', 'process', 'edge'):
        return False
    if isinstance(node.get('instance'), Emitter):
        return True
    address = node.get('address')
    if not isinstance(address, str):
        return False
    name = address.split(':', 1)[-1]
    registry = getattr(core, 'link_registry', None) if core is not None else None
    if registry is not None:
        cls = registry.get(name) or registry.get(name.rsplit('.', 1)[-1])
        if isinstance(cls, type) and issubclass(cls, Emitter):
            return True
        if cls is not None:
            return False
    return name.rsplit('.', 1)[-1].endswith('Emitter')


def document_has_emitter(state, core=None):
    '''Recursively test whether a document ``state`` already contains an
    emitter node (at any depth — e.g. one a builder nested inside an agent
    sub-composite).'''
    if _node_is_emitter(state, core):
        return True
    if isinstance(state, dict):
        return any(document_has_emitter(v, core) for v in state.values())
    if isinstance(state, (list, tuple)):
        return any(document_has_emitter(v, core) for v in state)
    return False


def install_emitters(state, declarations, run_id=None, out_dir=None, core=None,
                     on_unknown_address='raise'):
    '''Return a copy of ``state`` with the declared emitter(s) installed.

    Emitters land at the conventional ``emitter`` / ``emitter_<i>`` keys.
    Because those keys are deterministic, installing twice rewrites the same
    slots rather than adding a second sink — so a caller may invoke this
    unconditionally without risking a composite that emits everything twice.

    ``on_unknown_address`` (``"raise"`` default, or ``"ram"`` to opt into the
    silent RAMEmitter fallback) is threaded to
    :func:`emitter_node_from_declaration`.

    Returns ``state`` unchanged when nothing is declared.
    '''
    declarations = [decl for decl in (declarations or []) if isinstance(decl, dict)]
    if not declarations:
        return dict(state)

    installed = dict(state)
    for index, decl in enumerate(declarations):
        key = 'emitter' if index == 0 else f'emitter_{index}'
        installed[key] = emitter_node_from_declaration(
            decl, run_id=run_id, out_dir=out_dir, core=core,
            on_unknown_address=on_unknown_address)

    return installed

def gather_emitter_results(composite, queries=None):
    '''Retrieve query results from all emitter steps in a composite.'''
    emitter_paths = find_instance_paths(composite.state, 'process_bigraph.emitter.Emitter')
    queries = queries or {path: None for path in emitter_paths}

    results = {}
    for path, query in queries.items():
        emitter = get_path(composite.state, path)
        results[path] = emitter['instance'].query(query)
    return results

def add_emitter_to_composite(composite, core, emitter_mode='all', address='local:RAMEmitter'):
    '''Insert an emitter into a composite and rebuild the step network.'''
    path = ('emitter',)
    emitter_state = generate_emitter_state(composite, emitter_mode=emitter_mode, address=address)
    composite.merge({}, set_path({}, path, emitter_state))

    # TODO -- this is a hack to get the emitter to show up in the state
    _, instance = core.traverse(composite.schema, composite.state, path)
    composite.step_paths[path] = instance
    composite.build_step_network()
    return composite


# =====================
# Emitter Base Classes
# =====================

class EmitterResults:
    """A durable **reference** to what an emitter accumulated.

    Carries no bulk data — only what is needed to find it again: the
    emitter's address, where it sits, how much it holds, and whatever
    context (``sim_data`` and friends) a consumer needs to interpret it.
    :meth:`resolve` pulls the data on demand, so handing a handle down a
    step network stays cheap no matter how large the run was.

    This is what an emitter's ``results`` port carries, so a downstream
    step can depend on the emitter as an ordinary producer rather than
    reaching for the imperative ``gather_emitter_results`` pull.
    """

    #: This handle's artifact kind. ``EmitterResults`` *is* the trajectory
    #: case of ``artifacts.ArtifactResults`` — it answers the same
    #: ``kind``/``context``/``resolve()`` protocol, so a consumer cannot tell
    #: a live emitter from a pulled artifact. The difference is only where
    #: the data comes from: an emitter still in memory, or a store on disk.
    kind = 'trajectory'

    #: Set when a recompute disagreed with a stored fingerprint.
    provenance_status = 'ok'

    __slots__ = ('emitter', 'address', 'path', 'context', '_resolved')

    def __init__(self, emitter, address=None, path=None, context=None):
        self.emitter = emitter
        self.address = address
        self.path = tuple(path) if path else ()
        self.context = context or {}
        self._resolved = {}

    @property
    def count(self):
        """How many records the emitter is holding, when it can say."""
        history = getattr(self.emitter, 'history', None)
        return len(history) if history is not None else None

    def resolve(self, paths=None):
        """Pull the accumulated data. The handle stays a reference; this is
        the only place the bulk is materialized.

        Memoized per ``paths``: a handle refers to a run that has *completed*,
        so resolving it twice must give the same answer. It also must not cost
        twice — several flush entities resolve the same handle, and a durable
        emitter's ``query()`` can be far from free (``XArrayEmitter`` re-reads
        its zarr store, and flushes buffered rows on the way, which without
        this appends them again on every read).
        """
        key = tuple(paths) if isinstance(paths, list) else paths
        if key not in self._resolved:
            self._resolved[key] = self.emitter.query(paths)
        return self._resolved[key]

    def to_dict(self):
        """A JSON-safe summary — the reference, never the data."""
        return {
            'address': self.address,
            'path': list(self.path),
            'count': self.count,
            'context': self.context}

    def __repr__(self):
        return (f'EmitterResults(address={self.address!r}, '
                f'path={"/".join(self.path)!r}, count={self.count})')


class Emitter(Step):
    '''Base emitter class: defines schema and stub methods.'''
    config_schema = {'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

    def outputs(self) -> Dict:
        '''Emitters produce a ``results`` handle.

        Declaring the port is what lets a flush step depend on the emitter
        as an ordinary producer/consumer edge. The handle is a reference,
        not the data — see :class:`EmitterResults`.

        Note the port is *not* written by :meth:`update`: results are a
        completion-time value, and writing them per tick would fire every
        downstream consumer on every tick. :meth:`finalize` produces them.
        '''
        return {'results': 'node'}

    def results(self, path=None, context=None) -> 'EmitterResults':
        '''The durable handle for what this emitter has accumulated.'''
        return EmitterResults(
            self,
            address=self.config.get('address'),
            path=path,
            context=context)

    def finalize(self, path=None, context=None) -> Dict:
        '''The update an emitter contributes at the end of a run.

        Separate from :meth:`update` because ``results`` is meaningful once,
        at completion — not once per tick.
        '''
        return {'results': self.results(path=path, context=context)}

    def query(self, paths=None, query=None):
        '''Return recorded history.

        :param paths: a list of paths to project from each recorded state.
            If None, the entire history is returned.
        :param query: deprecated alias for ``paths``.
        '''
        return {}

    def update(self, state) -> Dict:
        return {}


def _resolve_query_paths(paths, query):
    '''Accept either the new ``paths`` kwarg or the legacy ``query`` kwarg.'''
    if paths is None and query is not None:
        return query
    return paths


# ========================
# Emitter Implementations
# ========================

class ConsoleEmitter(Emitter):
    '''Print state to console each timestep.'''
    def update(self, state) -> Dict:
        print(state)
        return {}

def tree_copy(state):
    '''Deep copy utility for nested simulation state (excluding Edge instances).'''
    if isinstance(state, dict):
        return {k: v for k, v in ((k, tree_copy(v)) for k, v in state.items()) if v is not None}
    if isinstance(state, np.ndarray):
        return state.copy()
    if isinstance(state, Edge):
        return None
    return copy.deepcopy(state)


class RAMEmitter(Emitter):
    '''Store historical states in memory.

    ``subsample`` records only every Nth composite tick (default 1 =
    every tick). Use this for long runs or composites with heavy
    state (large fields, many agents) to keep RAM bounded — the
    saved time-series still reflects the simulation's true cadence
    via each row's ``global_time`` field.

    ``max_len`` optionally bounds ``history`` to the most recent N
    recorded rows (a ring buffer): once the cap is reached each new
    record drops the oldest. Default ``None`` = unbounded (the
    historical behaviour — consumers such as v2ecoli that read the
    full ``history`` are unaffected unless they opt in). ``max_len``
    counts *recorded* rows (i.e. post-``subsample``).
    '''
    config_schema = {
        **Emitter.config_schema,
        'subsample': {'_type': 'integer', '_default': 1},
        'max_len': {'_type': 'maybe[integer]', '_default': None},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        subsample = config.get('subsample')
        self.subsample = 1 if subsample is None else int(subsample)
        if self.subsample < 1:
            raise ValueError(
                f'RAMEmitter subsample must be >= 1, got {self.subsample}'
            )
        max_len = config.get('max_len')
        self.max_len = None if max_len is None else int(max_len)
        if self.max_len is not None and self.max_len < 1:
            raise ValueError(
                f'RAMEmitter max_len must be >= 1 (or None), got {self.max_len}'
            )
        self.history = []
        self._step = 0

    def update(self, state) -> Dict:
        step = self._step
        self._step += 1
        if step % self.subsample != 0:
            return {}
        self.history.append(tree_copy(state))
        # Ring-buffer: keep only the most recent ``max_len`` rows. ``history``
        # stays a plain list (slicing/indexing preserved for consumers) — the
        # trim only runs when a cap is configured.
        if self.max_len is not None and len(self.history) > self.max_len:
            del self.history[:len(self.history) - self.max_len]
        return {}

    def query(self, paths=None, schema=None, query=None):
        paths = _resolve_query_paths(paths, query)
        schema = schema or self.inputs()
        if isinstance(paths, list):
            results = []
            for t in self.history:
                result = {}
                for path in paths:
                    _, value = self.core.traverse(schema, t, path)
                    result = set_path(result, path, value)
                results.append(result)
            return results
        return self.history


class JSONEmitter(Emitter):
    '''Append simulation state to a persistent JSON Lines file each timestep.

    Each recorded tick is one JSON object on its own line (``.json`` file in
    JSON Lines / ``jsonl`` form). This is an append-only O(1)-per-tick write:
    the previous behaviour re-read and re-serialized the entire history on
    every tick (O(n) per tick -> O(n^2) per run), which dominated long runs.

    ``query`` transparently reads BOTH the current line-delimited format and
    the legacy single JSON-array format, so files written by older versions
    remain readable.
    '''
    config_schema = {
        **Emitter.config_schema,
        'file_path': {'_type': 'string', '_default': './out'},
        'simulation_id': {'_type': 'string', '_default': None}
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.simulation_id = config.get('simulation_id') or str(uuid.uuid4())
        self.file_path = config.get('file_path', './out')
        os.makedirs(self.file_path, exist_ok=True)
        self.filepath = os.path.join(self.file_path, f'history_{self.simulation_id}.json')

    def update(self, state) -> dict:
        # Append-only: one JSON record per line. No read-modify-write.
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(copy.deepcopy(state)))
            f.write('\n')
        return {}

    def _load_history(self):
        '''Read the history file, supporting the current JSON Lines format
        and the legacy single-JSON-array format.'''
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, 'r') as f:
            text = f.read()
        stripped = text.lstrip()
        if not stripped:
            return []
        if stripped[0] == '[':
            # Legacy single-array format written by older JSONEmitter versions.
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return []
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partial trailing write
        return rows

    def query(self, paths=None, query=None):
        paths = _resolve_query_paths(paths, query)
        data = self._load_history()
        if not data:
            return []

        if isinstance(paths, list):
            results = []
            for t in data:
                result = {}
                for path in paths:
                    element = get_path(t, path)
                    result = set_path(result, path, element)
                results.append(result)
            return results
        return data


# ====================
# Base Emitter Mapping
# ====================


# ------------------------------------------------------------
# Back-compat re-exports from pbg-emitters
# ------------------------------------------------------------
# SQLiteEmitter + ParquetEmitter were extracted to a focused
# emitter library (https://github.com/vivarium-collective/pbg-emitters)
# so each can iterate (and ship optional heavy deps) independently of
# the framework. Existing code that imports them from
# ``process_bigraph.emitter`` keeps working as long as ``pbg-emitters``
# is installed (``pip install pbg-emitters[sqlite]`` or
# ``pip install pbg-emitters[parquet]``). Install both via the
# ``process-bigraph[emitters]`` extra.
try:
    from pbg_emitters import SQLiteEmitter  # noqa: F401
    from pbg_emitters import (  # noqa: F401
        save_simulation_metadata,
        list_simulations,
        load_history,
        load_simulation_metadata,
        mark_simulation_finished,
    )
except ImportError:
    pass

try:
    from pbg_emitters import ParquetEmitter  # noqa: F401
except ImportError:
    pass
