"""
========
Emitters
========

Emitters are steps that observe the state of the system and emit it to an external source.
This could be to a database, to a file, or to the console.
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


def anyize_paths(tree):
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            result[key] = anyize_paths(value)
    else:
        result = 'any'

    return result


def emitter_from_wires(wires, address='local:ram-emitter'):
    return {
        '_type': 'step',
        'address': address,
        'config': {
            'emit': anyize_paths(wires)},
        'inputs': wires}


def collect_input_ports(state, path=None):
    process_paths = find_instance_paths(state, 'process_bigraph.composite.Process')
    step_paths = find_instance_paths(state, 'process_bigraph.composite.Step')
    path = path or ()
    input_ports = {}
    for key, value in state.items():

        # TODO -- make these full paths
        full_path = path + (key,) if path else (key,)
        full_key = '/'.join(full_path)

        if is_schema_key(key):  # skip schema keys
            continue
        # if composite.core.inherits_from(composite.composition.get(key, {}), "edge"):  # skip edges
        #     continue
        if full_path in process_paths.keys() or full_path in step_paths.keys():  # skip processes
            continue
        if isinstance(value, dict):  # recurse into nested dictionaries
            input_ports.update(collect_input_ports(value, full_path))
        else:
            input_ports[full_key] = list(full_path)
    return input_ports


def generate_emitter_state(
        composite,
        emitter_mode="all",
        address="local:ram-emitter"
):
    """
    Generate the state configuration for an emitter.

    This function creates the state configuration for an emitter based on the provided
    composite, mode, and configuration. The emitter observes the state of the system
    and emits it to an external source, such as a database, file, or console.

    Args:
        composite (Composite): The composite object representing the system state.
        emitter_mode (Optional[Union[str, Dict[str, Any]]]): The mode of the emitter.
            - "all": Emit all input ports (default).
            - "none": Emit no input ports.
            - "paths" as a key in a dict to a list of paths to emit.
        address (str): The default address of the emitter. Defaults to "local:ram-emitter".

    Returns:
        Dict[str, Any]: The state configuration for the emitter.
    """
    config = {}

    input_ports = {}
    if emitter_mode == "all":
        input_ports = collect_input_ports(composite.state)
    elif emitter_mode == "none":
        input_ports = {}
    elif isinstance(emitter_mode, dict):
        if "paths" in emitter_mode:
            assert isinstance(emitter_mode["paths"], list), "emitter_mode['paths'] must be a list"
            for path in emitter_mode["paths"]:
                if isinstance(path, str):
                    input_ports[path] = [path]
                elif isinstance(path, list):
                    input_ports[path[0]] = path
    else:
        raise ValueError(f"Invalid mode: {emitter_mode}. Expected one of ['all', 'none']")

    if "global_time" not in input_ports:
        input_ports["global_time"] = ["global_time"]

    if "emit" not in config:
        config["emit"] = {
            input_port: "any"
            for input_port in input_ports}

    return {
        "_type": "step",
        "address": address,
        "config": config,
        "inputs": input_ports}


def gather_emitter_results(composite, queries=None):
    """
    a map of paths to emitter --> queries for the emitter at that path
    """

    emitter_paths = find_instance_paths(
        composite.state,
        instance_type='process_bigraph.emitter.Emitter')

    if queries is None:
        queries = {
            path: None
            for path in emitter_paths.keys()}

    results = {}
    for path, query in queries.items():
        emitter = get_path(composite.state, path)
        results[path] = emitter['instance'].query(query)

        # TODO: unnest the results?
        # TODO: allow the results to be transposed

    return results


def add_emitter_to_composite(composite,
                             core,
                             emitter_mode='all',
                             address="local:ram-emitter"
                             ):
    # add an emitter
    path = ('emitter',)
    emitter_state = generate_emitter_state(composite,
                                           emitter_mode=emitter_mode,
                                           address=address)
    emitter_state = set_path({}, path, emitter_state)
    composite.merge({}, emitter_state)
    # TODO -- this is a hack to get the emitter to show up in the state
    _, instance = core.slice(
        composite.composition,
        composite.state,
        path)
    # add to steps and rebuild
    composite.step_paths[path] = instance
    composite.build_step_network()
    return composite


# =========
# Emitters
# =========


class Emitter(Step):
    """Base emitter class.

    An `Emitter` implementation instance diverts all querying of data to
    the primary historical collection whose type pertains to Emitter child, i.e:
     database-emitter=>`pymongo.Collection`, ram-emitter=>`.RamEmitter.history`(`List`)
    """
    config_schema = {
        'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

    def query(self, query=None):
        return {}

    def update(self, state) -> Dict:
        return {}


class ConsoleEmitter(Emitter):
    """Console emitter class.

    This emitter logs the state to the console.
    """

    def update(self, state) -> Dict:
        print(state)
        return {}


def tree_copy(state):
    if isinstance(state, dict):
        result = {}
        for key, value in state.items():
            tree = tree_copy(value)
            if tree is not None:
                result[key] = tree
        return result
    elif isinstance(state, np.ndarray):
        return state.copy()
    elif isinstance(state, Edge):
        return None
    else:
        return copy.deepcopy(state)


class RAMEmitter(Emitter):
    """RAM emitter class.

    This emitter logs the state to a list in memory.
    """

    def __init__(self, config, core):
        super().__init__(config, core)
        self.history = []

    def update(self, state) -> Dict:
        tree = tree_copy(state)
        self.history.append(tree)

        return {}

    def query(self, query=None, schema=None):
        """
        Query the history of the emitter.
        :param query: a list of paths to query from the history. If None, the entire history is returned.
        :return: results of the query in a list
        """
        schema = schema or self.inputs()
        if isinstance(query, list):
            results = []
            for t in self.history:
                result = {}
                for path in query:
                    _, state = self.core.slice(
                        schema=schema,
                        state=t,
                        path=path)
                
                    # element = get_path(t, path)
                    result = set_path(result, path, state)
                results.append(result)
                # element = get_path(self.history, path)
                # result = set_path(result, path, element)
        else:
            results = self.history

        return results


class JSONEmitter(Emitter):
    """JSON emitter class.

    This emitter logs the state to a JSON file efficiently by appending to it.
    """

    config_schema = {
        **Emitter.config_schema,
        'file_path': {
            '_type': 'string',
            '_default': './out'  # Changed to a writable directory
        },
        'simulation_id': {
            '_type': 'string',
            '_default': None
        }
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.simulation_id = config.get('simulation_id') or str(uuid.uuid4())
        self.file_path = config.get('file_path', './out')  # Changed default to a writable path
        os.makedirs(self.file_path, exist_ok=True)
        self.filepath = os.path.join(self.file_path, f"history_{self.simulation_id}.json")

        # Ensure the file exists and initialize properly
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)  # Initialize with an empty list

    def update(self, state) -> dict:
        """Appends the deep-copied state to the JSON file efficiently."""
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
        """Queries the JSON history by streaming the file to avoid memory overhead."""
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


BASE_EMITTERS = {
    'console-emitter': ConsoleEmitter,
    'ram-emitter': RAMEmitter,
    'json-emitter': JSONEmitter,
}


# ======
# Tests
# ======

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
            # 'interval': 1.0,
            'inputs': {'level': ['valueA']},
            'outputs': {'level': ['valueA']}},
        'increase2': {
            '_type': 'process',
            'address': 'local:!process_bigraph.tests.IncreaseProcess',
            'config': {'rate': 0.1},
            # 'interval': 1.0,
            'inputs': {'level': ['valueB']},
            'outputs': {'level': ['valueB']}},
    }
    composite = Composite({'state': composite_spec}, core)
    composite = add_emitter_to_composite(composite,
                                         core,
                                         emitter_mode='all',
                                         address='local:ram-emitter')

    # run the simulation
    composite.run(10)

    # query the emitter
    results = composite.state['emitter']['instance'].query()
    assert len(results) == 11
    assert results[-1]['global_time'] == 10
    assert 'valueA' in results[0] and 'valueB' in results[0]
    # print(results)

    composite2 = Composite({'state': composite_spec}, core)
    composite2 = add_emitter_to_composite(composite2,
                                         core,
                                         emitter_mode={'paths': ['valueA']},
                                         address='local:ram-emitter')

    # run the simulation
    composite2.run(10)

    # query the emitter
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
            'outputs': {'level': ['value']}},
    }
    composite = Composite({'state': composite_spec}, core)
    composite = add_emitter_to_composite(composite,
                                         core,
                                         emitter_mode='all',
                                         address='local:json-emitter')

    # run the simulation
    composite.run(10)

    # query the emitter
    results = composite.state['emitter']['instance'].query()
    assert len(results) == 11
    assert results[-1]['global_time'] == 10
    print(results)


if __name__ == '__main__':
    from process_bigraph import register_types, ProcessTypes
    core = ProcessTypes()
    core = register_types(core)

    test_ram_emitter(core)
    test_json_emitter(core)
