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
from typing import Dict

from bigraph_schema import get_path, set_path, is_schema_key

from process_bigraph.composite import Step, find_instance_paths


def emitter_config(composite, emitter_config):
    """Return the emitter configuration schema."""
    address = emitter_config.get("address", "local:ram-emitter")
    config = emitter_config.get("config", {})
    mode = emitter_config.get("mode", "all")

    valid_modes = {"all", "none"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Expected one of {valid_modes}.")

    inputs_config = emitter_config.get("inputs", {})
    process_paths = find_instance_paths(composite.state, 'process_bigraph.composite.Process')
    step_paths = find_instance_paths(composite.state, 'process_bigraph.composite.Step')

    def collect_input_ports(state, prefix=""):
        input_ports = {}
        for key, value in state.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if is_schema_key(key):  # skip schema keys
                continue
            if composite.core.inherits_from(composite.composition.get(key, {}), "edge"):  # skip edges
                continue
            if full_key in process_paths or full_key in step_paths:  # skip processes
                continue
            if isinstance(value, dict):  # recurse into nested dictionaries
                input_ports.update(collect_input_ports(value, full_key))
            else:
                input_ports[full_key] = [inputs_config.get(full_key, full_key)]
        return input_ports

    input_ports = collect_input_ports(composite.state) if mode == "all" else emitter_config.get("emit", {})

    if "emit" not in config:
        config["emit"] = {
            input_port: "any"
            for input_port in input_ports}

    return {
        "_type": "step",
        "address": address,
        "config": config,
        "inputs": input_ports}


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


class RAMEmitter(Emitter):
    """RAM emitter class.

    This emitter logs the state to a list in memory.
    """

    def __init__(self, config, core):
        super().__init__(config, core)
        self.history = []


    def update(self, state) -> Dict:
        self.history.append(copy.deepcopy(state))
        return {}


    def query(self, query=None):
        """
        Query the history of the emitter.
        :param query: a list of paths to query from the history. If None, the entire history is returned.
        :return: results of the query in a list
        """
        if isinstance(query, list):
            results = []
            for t in self.history:
                result = {}
                for path in query:
                    element = get_path(t, path)
                    result = set_path(result, path, element)
                results.append(result)
                # element = get_path(self.history, path)
                # result = set_path(result, path, element)
        else:
            results = self.history

        return results


def gather_results(composite, queries=None):
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


class JSONEmitter(Emitter):
    """JSON emitter class.

    This emitter logs the state to a JSON file efficiently by appending to it.
    """
    config_schema = {
        **Emitter.config_schema,
        'filepath': {
            '_type': 'string',
            '_default': '/out'
        },
        'simulation_id': {
            '_type': 'string',
            '_default': None
        }
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.simulation_id = config.get('simulation_id') or str(uuid.uuid4())
        self.file_path = config.get('file_path', '/out')
        os.makedirs(self.file_path, exist_ok=True)
        self.filepath = os.path.join(self.file_path, f"history_{self.simulation_id}.json")

        # Ensure the file exists
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                f.write('[')  # Start JSON array

    def update(self, state) -> dict:
        """Appends the deep-copied state to the JSON file efficiently."""
        with open(self.filepath, 'a') as f:
            if os.path.getsize(self.filepath) > 1:
                f.write(',\n')  # Add a comma separator
            json.dump(copy.deepcopy(state), f)
        return {}

    def finalize(self):
        """Closes the JSON array properly at the end of execution."""
        with open(self.filepath, 'a') as f:
            f.write(']')

    def query(self, query=None):
        """Queries the JSON history by streaming the file to avoid memory overhead."""
        results = []
        with open(self.filepath, 'r') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() <= 1:  # Empty or only the opening bracket
                return results

            f.seek(0)
            data = json.loads(f.read() + ']')  # Ensure JSON format

            if isinstance(query, list):
                for t in data:
                    result = {}
                    for path in query:
                        element = get_path(t, path)
                        result = set_path(result, path, element)
                    results.append(result)
            else:
                results = data

        return results


BASE_EMITTERS = {
    'console-emitter': ConsoleEmitter,
    'ram-emitter': RAMEmitter,
    'json-emitter': JSONEmitter,
}
