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

def generate_emitter_state(composite, emitter_mode='all', address='local:ram-emitter'):
    '''
    Generate emitter state for a given composite and mode.
    Modes:
        - "all": observe all valid inputs
        - "none": observe nothing
        - {"paths": [...]}: custom paths to observe
    '''
    config = {}
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

    if 'emit' not in config:
        config['emit'] = {port: 'node' for port in input_ports}

    return {
        '_type': 'step',
        'address': address,
        'config': config,
        'inputs': input_ports
    }

def gather_emitter_results(composite, queries=None):
    '''Retrieve query results from all emitter steps in a composite.'''
    emitter_paths = find_instance_paths(composite.state, 'process_bigraph.emitter.Emitter')
    queries = queries or {path: None for path in emitter_paths}

    results = {}
    for path, query in queries.items():
        emitter = get_path(composite.state, path)
        results[path] = emitter['instance'].query(query)
    return results

def add_emitter_to_composite(composite, core, emitter_mode='all', address='local:ram-emitter'):
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

class Emitter(Step):
    '''Base emitter class: defines schema and stub methods.'''
    config_schema = {'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

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
    '''
    config_schema = {
        **Emitter.config_schema,
        'subsample': {'_type': 'integer', '_default': 1},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        subsample = config.get('subsample')
        self.subsample = 1 if subsample is None else int(subsample)
        if self.subsample < 1:
            raise ValueError(
                f'RAMEmitter subsample must be >= 1, got {self.subsample}'
            )
        self.history = []
        self._step = 0

    def update(self, state) -> Dict:
        step = self._step
        self._step += 1
        if step % self.subsample != 0:
            return {}
        self.history.append(tree_copy(state))
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
    '''Append simulation state to a persistent JSON file each timestep.'''
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
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def update(self, state) -> dict:
        with open(self.filepath, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(copy.deepcopy(state))
            f.seek(0)
            json.dump(data, f, indent=4)
        return {}

    def query(self, paths=None, query=None):
        paths = _resolve_query_paths(paths, query)
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
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
