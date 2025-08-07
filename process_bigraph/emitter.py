"""
===========================
Emitter Utilities & Classes
===========================

Emitters are steps that observe a composite simulation's state and emit data to an external source
(e.g., console, memory, or file). This module provides tools to:
- Define emitter steps programmatically
- Insert emitters into a running composite
- Collect data from emitter steps
- Implement concrete emitters (RAM, console, JSON)
"""

import os
import json
import copy
import uuid
import pytest
import numpy as np
from typing import Dict

from bigraph_schema import get_path, set_path, is_schema_key, Edge
from process_bigraph.composite import Composite, Step, find_instance_paths


# ==========================
# Emitter Spec Construction
# ==========================

def anyize_paths(tree):
    """Recursively convert all leaves of a nested path tree to 'any'."""
    if isinstance(tree, dict):
        return {key: anyize_paths(value) for key, value in tree.items()}
    return 'any'

def emitter_from_wires(wires, address='local:ram-emitter'):
    """Create an emitter step spec from wire mappings."""
    return {
        '_type': 'step',
        'address': address,
        'config': {
            'emit': anyize_paths(wires)},
        'inputs': wires}

def collect_input_ports(state, path=None):
    """Recursively collect all valid input ports from state tree, skipping processes and schema keys."""
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

def generate_emitter_state(composite, emitter_mode="all", address="local:ram-emitter"):
    """
    Generate emitter state for a given composite and mode.
    Modes:
        - "all": observe all valid inputs
        - "none": observe nothing
        - {"paths": [...]}: custom paths to observe
    """
    config = {}
    input_ports = {}

    if emitter_mode == "all":
        input_ports = collect_input_ports(composite.state)
    elif emitter_mode == "none":
        input_ports = {}
    elif isinstance(emitter_mode, dict) and "paths" in emitter_mode:
        for path in emitter_mode["paths"]:
            if isinstance(path, str):
                input_ports[path] = [path]
            elif isinstance(path, list):
                input_ports[path[0]] = path
    else:
        raise ValueError(f"Invalid mode: {emitter_mode}.")

    if "global_time" not in input_ports:
        input_ports["global_time"] = ["global_time"]

    if "emit" not in config:
        config["emit"] = {port: "any" for port in input_ports}

    return {
        "_type": "step",
        "address": address,
        "config": config,
        "inputs": input_ports
    }

def gather_emitter_results(composite, queries=None):
    """Retrieve query results from all emitter steps in a composite."""
    emitter_paths = find_instance_paths(composite.state, 'process_bigraph.emitter.Emitter')
    queries = queries or {path: None for path in emitter_paths}

    results = {}
    for path, query in queries.items():
        emitter = get_path(composite.state, path)
        results[path] = emitter['instance'].query(query)
    return results

def add_emitter_to_composite(composite, core, emitter_mode='all', address="local:ram-emitter"):
    """Insert an emitter into a composite and rebuild the step network."""
    path = ('emitter',)
    emitter_state = generate_emitter_state(composite, emitter_mode=emitter_mode, address=address)
    composite.merge({}, set_path({}, path, emitter_state))

    # TODO -- this is a hack to get the emitter to show up in the state
    _, instance = core.slice(composite.composition, composite.state, path)
    composite.step_paths[path] = instance
    composite.build_step_network()
    return composite


# =====================
# Emitter Base Classes
# =====================

class Emitter(Step):
    """Base emitter class: defines schema and stub methods."""
    config_schema = {'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

    def query(self, query=None):
        """
        Query the history of the emitter.
        :param query: a list of paths to query from the history. If None, the entire history is returned.
        :return: results of the query in a list
        """
        return {}

    def update(self, state) -> Dict:
        return {}


# ========================
# Emitter Implementations
# ========================

class ConsoleEmitter(Emitter):
    """Print state to console each timestep."""
    def update(self, state) -> Dict:
        print(state)
        return {}

def tree_copy(state):
    """Deep copy utility for nested simulation state (excluding Edge instances)."""
    if isinstance(state, dict):
        return {k: v for k, v in ((k, tree_copy(v)) for k, v in state.items()) if v is not None}
    if isinstance(state, np.ndarray):
        return state.copy()
    if isinstance(state, Edge):
        return None
    return copy.deepcopy(state)


class RAMEmitter(Emitter):
    """Store historical states in memory."""
    def __init__(self, config, core):
        super().__init__(config, core)
        self.history = []

    def update(self, state) -> Dict:
        self.history.append(tree_copy(state))
        return {}

    def query(self, query=None, schema=None):
        schema = schema or self.inputs()
        if isinstance(query, list):
            results = []
            for t in self.history:
                result = {}
                for path in query:
                    _, value = self.core.slice(schema, t, path)
                    result = set_path(result, path, value)
                results.append(result)
            return results
        return self.history


class JSONEmitter(Emitter):
    """Append simulation state to a persistent JSON file each timestep."""
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
        self.filepath = os.path.join(self.file_path, f"history_{self.simulation_id}.json")
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

    def query(self, query=None):
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return []

        if isinstance(query, list):
            results = []
            for t in data:
                result = {}
                for path in query:
                    element = get_path(t, path)
                    result = set_path(result, path, element)
                results.append(result)
            return results
        return data


# ====================
# Base Emitter Mapping
# ====================

BASE_EMITTERS = {
    'console-emitter': ConsoleEmitter,
    'ram-emitter': RAMEmitter,
    'json-emitter': JSONEmitter,
}


# ==========
# Unit Tests
# ==========

@pytest.fixture
def core():
    from process_bigraph import register_types, ProcessTypes
    core = ProcessTypes()
    return register_types(core)

def test_ram_emitter(core):
    composite_spec = {
        'increase': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.IncreaseProcess',
            'config': {'rate': 0.3},
            'inputs': {'level': ['valueA']},
            'outputs': {'level': ['valueA']}},
        'increase2': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.IncreaseProcess',
            'config': {'rate': 0.1},
            'inputs': {'level': ['valueB']},
            'outputs': {'level': ['valueB']}},
        'emitter': emitter_from_wires({
            'time': ['global_time'],
            'valueA': ['valueA'],
            'valueB': ['valueB']})}

    composite = Composite({'state': composite_spec}, core=core)
    composite.run(10)

    results = composite.state['emitter']['instance'].query()
    assert len(results) == 11
    assert results[-1]['time'] == 10
    assert 'valueA' in results[0] and 'valueB' in results[0]

    composite_spec['emitter'] = emitter_from_wires({
        'time': ['global_time'],
        'valueA': ['valueA']})
    composite2 = Composite({'state': composite_spec}, core=core)
    composite2.run(10)

    results2 = composite2.state['emitter']['instance'].query()
    assert 'valueA' in results2[0] and 'valueB' not in results2[0]
    print(results2)

def test_json_emitter(core):
    composite_spec = {
        'increase': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.IncreaseProcess',
            'config': {'rate': 0.3},
            'interval': 1.0,
            'inputs': {'level': ['value']},
            'outputs': {'level': ['value']}}}
    composite = Composite({'state': composite_spec}, core)
    composite = add_emitter_to_composite(composite, core, emitter_mode='all', address='local:json-emitter')
    composite.run(10)

    results = composite.state['emitter']['instance'].query()
    assert len(results) == 10
    assert results[-1]['global_time'] == 10
    print(results)


if __name__ == '__main__':
    from process_bigraph import register_types, ProcessTypes
    core = ProcessTypes()
    core = register_types(core)
    test_ram_emitter(core)
    test_json_emitter(core)
