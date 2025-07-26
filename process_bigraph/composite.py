"""
composite.py

This module defines the core execution logic for compositional simulation workflows using the
Process Bigraph Protocol (PBP). It includes:

- `Composite`: A process orchestrator supporting nested processes, steps, and synchronization.
- `Step`: A process steps triggered by dependency updates.
- `Process`: A time-driven process unit.
- Utility functions for dependency tracking, merging, scheduling, and update application.

Used as part of the Vivarium 2.0 ecosystem for modular biological modeling.
"""

import os
import copy
import json
import math
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Mapping, MutableMapping, Sequence
)
import collections

from bigraph_schema import (
    Edge, Registry, 
    get_path, set_path, resolve_path, hierarchy_depth, deep_merge,
    is_schema_key, strip_schema_keys)

from process_bigraph.protocols import local_lookup, local_lookup_module


# =========================
# Process Utility Functions
# =========================

def assert_interface(interface: Dict[str, Any]) -> None:
    """
    Ensure that the interface dictionary contains both 'inputs' and 'outputs' keys.

    Args:
        interface: A dictionary describing a process interface.

    Raises:
        AssertionError: If required keys are missing or extra keys are present.
    """
    required_keys = {'inputs', 'outputs'}
    existing_keys = set(interface.keys())
    assert existing_keys == required_keys, (
        f"Every interface requires exactly the keys 'inputs' and 'outputs', "
        f"but found: {existing_keys}"
    )


def find_instances(
    state: Dict[str, Any],
    instance_type: str = 'process_bigraph.composite.Process'
) -> Dict[str, Any]:
    """
    Recursively find all dictionary entries that contain an 'instance' of the given type.

    Args:
        state: Nested state dictionary.
        instance_type: Fully qualified path to the target class (e.g., 'module.Class').

    Returns:
        A dictionary of matching subtrees keyed by their path segment.
    """
    process_class = local_lookup_module(instance_type)
    found: Dict[str, Any] = {}

    for key, inner in state.items():
        if isinstance(inner, dict):
            if isinstance(inner.get('instance'), process_class):
                found[key] = inner
            elif not is_schema_key(key):
                sub_instances = find_instances(inner, instance_type)
                if sub_instances:
                    found[key] = sub_instances

    return found


def find_instance_paths(
    state: Dict[str, Any],
    instance_type: str = 'process_bigraph.composite.Process'
) -> Dict[Tuple[str, ...], Any]:
    """
    Find all paths to instances of a given type in the state.

    Args:
        state: The full nested state dictionary.
        instance_type: Fully qualified class name to match.

    Returns:
        A dictionary mapping full paths (as tuples) to instance-containing subtrees.
    """
    instances = find_instances(state, instance_type)
    return hierarchy_depth(instances)


def find_step_triggers(
    path: Union[List[str], Tuple[str, ...]],
    step: Dict[str, Any]
) -> Dict[Tuple[str, ...], List[Union[List[str], Tuple[str, ...]]]]:
    """
    Identify which paths, when updated, should trigger the execution of a given step.

    Args:
        path: Path to the step in the composite model tree.
        step: Step object containing an 'inputs' field with wire mappings.

    Returns:
        Mapping from trigger paths to lists of step paths that they trigger.
    """
    prefix = tuple(path[:-1])
    triggers: Dict[Tuple[str, ...], List[Union[List[str], Tuple[str, ...]]]] = {}
    wire_paths = find_leaves(step['inputs'])

    for wire in wire_paths:
        trigger_path = resolve_path(prefix + tuple(wire))
        triggers.setdefault(trigger_path, []).append(path)

    return triggers


def explode_path(path: Union[List[str], Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    """
    Break a hierarchical path into all its prefix paths.

    Example:
        ('a', 'b', 'c') → [(), ('a',), ('a', 'b'), ('a', 'b', 'c')]

    Args:
        path: A tuple or list representing a path.

    Returns:
        A list of prefix paths.
    """
    explode: Tuple[str, ...] = ()
    paths = [explode]
    for node in path:
        explode = explode + (node,)
        paths.append(explode)
    return paths


def merge_collections(
    existing: Optional[MutableMapping[str, Any]],
    new: Optional[MutableMapping[str, Any]]
) -> MutableMapping[str, Any]:
    """
    Merge two nested structures (dicts or lists), combining compatible elements in-place.

    Args:
        existing: An existing collection to merge into.
        new: A new collection to merge.

    Returns:
        The merged structure.

    Raises:
        Exception: If types are incompatible or mergeable fields conflict.
    """
    existing = existing or {}
    new = new or {}

    for key, value in new.items():
        if key in existing:
            if isinstance(existing[key], dict) and isinstance(value, Mapping):
                merge_collections(existing[key], value)
            elif isinstance(existing[key], list) and isinstance(value, Sequence):
                existing[key].extend(value)
            else:
                raise Exception(
                    f"Cannot merge conflicting types or values for key '{key}':\n"
                    f"existing={existing[key]}\nnew={value}"
                )
        else:
            existing[key] = value

    return existing


def empty_front(time: float) -> Dict[str, Any]:
    """
    Generate a default front buffer for a process.

    Args:
        time: The current simulation time.

    Returns:
        A dictionary with time and an empty update field.
    """
    return {'time': time, 'update': {}}


def find_leaves(tree_structure, path=None):
    """
    Recursively find all leaf paths in a nested dictionary structure.

    Args:
        tree_structure (any): A nested structure of dicts/lists/tuples.
        path (tuple or None): Current traversal path (for recursion).

    Returns:
        list: List of leaf paths as tuples.
    """
    leaves = []
    path = ()

    if tree_structure is None:
        pass
    elif isinstance(tree_structure, list):
        leaves = tree_structure
    elif isinstance(tree_structure, tuple):
        leaves.append(tree_structure)
    else:
        for key, value in tree_structure.items():
            if isinstance(value, dict):
                subleaves = find_leaves(value, path + (key,))
                leaves.extend(subleaves)
            else:
                leaves.append(path + tuple(value))

    return leaves


def build_step_network(steps):
    """
    Build the data dependency graph among steps.

    Args:
        steps: A mapping of step identifiers to their instance/config.

    Returns:
        - ancestors: A mapping from step keys to their input/output paths.
        - nodes: A mapping from paths to sets of steps that are dependent on them.
    """
    ancestors = {
        step_key: {'input_paths': None, 'output_paths': None}
        for step_key in steps
    }
    nodes = {}

    for step_key, step in steps.items():
        schema = step['instance'].interface()
        assert_interface(schema)

        # Compute input paths once per step
        if ancestors[step_key]['input_paths'] is None:
            ancestors[step_key]['input_paths'] = find_leaves(step['inputs'])

        # Compute output paths once per step
        if ancestors[step_key]['output_paths'] is None:
            ancestors[step_key]['output_paths'] = find_leaves(step.get('outputs', {}))

        input_paths = ancestors[step_key]['input_paths'] or []
        output_paths = ancestors[step_key]['output_paths'] or []

        # Track which steps consume/produce each path
        for input_path in input_paths:
            path = tuple(input_path)
            nodes.setdefault(path, {'before': set(), 'after': set()})
            nodes[path]['after'].add(step_key)

        for output_path in output_paths:
            if output_path not in input_paths:
                path = tuple(output_path)
                nodes.setdefault(path, {'before': set(), 'after': set()})
                nodes[path]['before'].add(step_key)

    return ancestors, nodes


def build_trigger_state(nodes):
    """
    Initialize the trigger state from dependency nodes.

    Args:
        nodes: Dependency graph nodes with 'before' and 'after' sets.

    Returns:
        A mapping of paths to the set of steps waiting on those paths.
    """
    return {key: value['before'].copy() for key, value in nodes.items()}


def find_downstream(steps, nodes, upstream):
    """
    Given a set of updated steps, identify all downstream steps that depend on them.

    Args:
        steps: Step metadata with input/output info.
        nodes: Dependency graph.
        upstream: Initial set of triggered step paths.

    Returns:
        Set of all steps affected directly or transitively.
    """
    downstream = set(upstream)
    visited = set([])
    previous_len = -1

    while len(downstream) > len(visited) and len(visited) > previous_len:
        previous_len = len(visited)
        down = set([])
        for step_path in downstream:
            if step_path not in visited:
                step_outputs = steps[step_path]['output_paths']
                if step_outputs is None:
                    step_outputs = []  # Ensure step_outputs is always an iterable
                for output in step_outputs:
                    for subpath in explode_path(output):
                        if subpath in nodes:
                            for dependent in nodes[subpath]['after']:
                                down.add(dependent)
                visited.add(step_path)
        downstream |= down

    return downstream


def determine_steps(steps, remaining, fulfilled):
    """
    Determine which steps are eligible to run, based on current fulfilled triggers.

    Args:
        steps: Step metadata.
        remaining: Set of step paths not yet run.
        fulfilled: Map of data paths to steps waiting for fulfillment.

    Returns:
        - to_run: List of ready step paths.
        - remaining: Updated remaining steps.
        - fulfilled: Updated fulfilled structure.
    """
    to_run = []

    for step_path in list(remaining):
        step_inputs = steps[step_path].get('input_paths', []) or []
        if all(len(fulfilled[input]) == 0 for input in step_inputs):
            to_run.append(step_path)

    for step_path in to_run:
        remaining.remove(step_path)
        step_outputs = steps[step_path].get('output_paths', []) or []
        for output in step_outputs:
            if step_path in fulfilled.get(output, set()):
                fulfilled[output].remove(step_path)

    return to_run, remaining, fulfilled


def interval_time_precision(timestep: float) -> int:
    """
    Compute the number of decimal places required to represent the given timestep.

    Args:
        timestep: Time interval as float.

    Returns:
        The number of digits after the decimal point.
    """
    return len(str(timestep).split('.')[1]) if '.' in str(timestep) else 0


# ===============
# Process Classes
# ===============

class SyncUpdate:
    """
    Wrapper for synchronous process updates.

    This object encapsulates an update dictionary and provides a `.get()` method
    for compatibility with deferred or lazy update execution pipelines.
    """

    def __init__(self, update: Dict[str, Any]) -> None:
        """
        Args:
            update: The process update to wrap.
        """
        self.update = update

    def get(self) -> Dict[str, Any]:
        """
        Returns:
            The stored process update.
        """
        return self.update


class Step(Edge):
    """
    Step base class.

    A `Step` is a stateless, non-temporal computational unit within a composite process.
    It is triggered when its data dependencies are satisfied, functioning like a reaction
    or transformation rule.

    Override the `.update()` method to define custom behavior.
    """
    # TODO: support trigger every time as well as dependency trigger

    def invoke(self, state: Dict[str, Any], _: Optional[float] = None) -> SyncUpdate:
        """
        Run the step using the given state and return its update.

        Args:
            state: The input state to compute the update from.
            _: Ignored time interval placeholder (not used by steps).

        Returns:
            A SyncUpdate object containing the update dictionary.
        """
        update = self.update(state)
        return SyncUpdate(update)

    def register_shared(self, instance):
        """
        Register a reference to a shared instance, e.g., for access to core or context.

        Args:
            instance: A reference to the external object being shared with the step.
        """
        self.instance = instance

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute and return the update for the step.

        Override this method in subclasses to define the step's logic.

        Args:
            state: The current simulation state at the step's inputs.

        Returns:
            A dictionary representing the update to apply.
        """
        return {}


class Process(Edge):
    """
    Process base class.

    A `Process` is a temporal unit of computation that operates on state and advances in time.
    Each subclass must implement the `update()` method and optionally the `invoke()` method.

    Processes are stateful and typically used for simulations of continuous or discrete dynamics.
    """

    def invoke(self, state: Dict[str, Any], interval: float) -> SyncUpdate:
        """
        Execute the process update for a given state and time interval.

        Args:
            state: The current simulation state for this process.
            interval: The time step over which to apply the update.

        Returns:
            A SyncUpdate containing the update result.
        """
        update = self.update(state, interval)
        return SyncUpdate(update)

    def update(self, state: Dict[str, Any], interval: float) -> Dict[str, Any]:
        """
        Override this method to implement the process logic.

        Args:
            state: The current simulation state at the process ports.
            interval: The time step over which to simulate.

        Returns:
            A dictionary representing the update to apply to the state.
        """
        return {}


class ProcessEnsemble(Process):
    """
    ProcessEnsemble base class.

    A container for multiple sub-processes that exposes a combined interface by unifying
    their inputs and outputs. Useful when combining multiple related processes into a single one.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, core: Optional[Any] = None) -> None:
        """
        Args:
            config: Configuration dictionary for the ensemble process.
            core: Optional shared core/context for schema operations and initialization.
        """
        super().__init__(config=config, core=core)

    def union_interface(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate a unified interface by combining all inputs_*/outputs_* methods defined in the subclass.

        Returns:
            A dictionary with 'inputs' and 'outputs' schemas merged from all sub-process interfaces.
        """
        union_inputs: Dict[str, Any] = {}
        union_outputs: Dict[str, Any] = {}

        for attr_name in dir(self):
            if attr_name.startswith('inputs_'):
                inputs_func = getattr(self, attr_name)
                if callable(inputs_func):
                    inputs = inputs_func()
                    union_inputs = self.core.resolve_schemas(union_inputs, inputs)

            if attr_name.startswith('outputs_'):
                outputs_func = getattr(self, attr_name)
                if callable(outputs_func):
                    outputs = outputs_func()
                    union_outputs = self.core.resolve_schemas(union_outputs, outputs)

        return {
            'inputs': union_inputs,
            'outputs': union_outputs
        }


class Defer:
    """
    Defer a computation by holding a reference to a function and its arguments
    until a later time when `.get()` is called.

    This is used to delay the application of a function to a value until all
    required data is available, typically used for processing deferred updates
    in simulation pipelines.

    Attributes:
        defer (SupportsGet): An object that supports `.get()` and returns the input to the function.
        f (Callable[[Any, Any], Any]): A binary function to apply to the result of `defer.get()` and `args`.
        args (Any): Arguments passed to the function alongside the deferred value.
    """

    def __init__(
            self,
            defer,
            f,
            args
    ) -> None:
        """
        Args:
            defer: Any object that implements `.get()` and returns a value.
            f: A function that takes two arguments: the result of `defer.get()` and `args`.
            args: A secondary argument passed to `f`.
        """
        self.defer = defer
        self.f = f
        self.args = args

    def get(self) -> Any:
        """
        Perform the deferred computation by calling the stored function with the
        deferred result and provided arguments.

        Returns:
            The result of `f(defer.get(), args)`.
        """
        return self.f(self.defer.get(), self.args)


def match_star_path(
    path: Tuple[str, ...],
    star_path: Tuple[str, ...]
) -> bool:
    """
    Compare two paths where elements in `star_path` may contain wildcards (*).

    Args:
        path: A tuple representing the actual path (e.g., ('cells', 'A', 'growth')).
        star_path: A tuple that may contain '*' wildcards to match any segment.

    Returns:
        True if the paths match, treating '*' as a wildcard; False otherwise.

    Example:
        match_star_path(('cells', 'A', 'growth'), ('cells', '*', 'growth'))  # True
        match_star_path(('cells', 'A'), ('cells', '*', 'growth'))            # False
    """
    for element, star_element in zip(path, star_path):
        if star_element != "*" and element != star_element:
            return False
    return True





class Composite(Process):
    """
    A Composite process contains a dynamic network of child Processes and Steps
    connected via a schema and bridge. It manages time, state, dependencies, and
    update propagation during simulation.
    """

    config_schema = {
        'composition': 'schema',
        'state': 'tree[any]',
        'interface': {
            'inputs': 'schema',
            'outputs': 'schema'
        },
        'bridge': {
            'inputs': 'wires',
            'outputs': 'wires'
        },
        'global_time_precision': 'maybe[float]'
    }

    @classmethod
    def load(cls, path: str, core: Optional[Any] = None) -> "Composite":
        """
        Load a Composite from a saved JSON file.

        Args:
            path: Path to the saved composition file.
            core: Optional core context providing deserialization.

        Returns:
            A new Composite instance.
        """
        with open(path) as data:
            document = json.load(data)
            composition = document['composition']
            document['composition'] = core.deserialize('schema', composition)
            return cls(document, core=core)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the composite model from its config.

        This method:
        - Adds `global_time` to schema/state if missing
        - Generates the full composition/state tree
        - Finds all step/process instances
        - Resolves the schema bridge
        - Prepares the step execution network
        - Computes initial front (per-process timeline)

        Args:
            config: Optional override configuration (usually not needed).
        """

        # Get the initial composition schema from config.
        initial_composition = self.config.get('composition', {})

        # Ensure 'global_time' is explicitly declared in the schema.
        if 'global_time' not in initial_composition:
            initial_composition['global_time'] = 'float'

        # Get the initial state from config.
        initial_state = self.config.get('state', {})

        # Ensure the initial simulation state has a global_time initialized.
        if 'global_time' not in initial_state:
            initial_state['global_time'] = 0.0

        # Generate internal schema and state structures using the core engine.
        self.composition, self.state = self.core.generate(
            initial_composition,
            initial_state)

        # Load the bridge configuration, which defines how inputs/outputs connect to the world.
        self.bridge = self.config.get('bridge', {})

        # Identify all Process and Step instances in the state tree.
        self.find_instance_paths(self.state)

        # Merge both process and step paths into a single edge dictionary.
        self.edge_paths = {**self.process_paths, **self.step_paths}

        # Initialize each process/step's state and accumulate it into a unified state tree.
        edge_state: Dict[str, Any] = {}
        for path, edge in self.edge_paths.items():
            # Generate the initial state for this specific edge (process or step).
            initial = self.core.initialize_edge_state(
                self.composition,
                path,
                edge)

            # Merge the new edge state with the global state tree, checking for conflicts.
            try:
                edge_state = deep_merge(edge_state, initial)
            except Exception:
                raise Exception(
                    f'initial state from edge does not match initial state from other edges:\n'
                    f'{path}\n{edge}\n{edge_state}'
                )

        # Apply the merged edge_state into the global state and update instance paths.
        self.merge(self.composition, edge_state)

        # Wire the input/output schema for the Composite from the bridge config.
        self.process_schema = {
            port: self.core.wire_schema(self.composition, self.bridge[port])
            for port in ['inputs', 'outputs']
        }

        # Set the global time precision used to round step time advances.
        self.global_time_precision = self.config.get('global_time_precision')

        # Initialize a "front" dictionary tracking the next update time and update data per process.
        self.front = {
            path: empty_front(self.state['global_time'])
            for path in self.process_paths
        }

        # A buffer for updates to be emitted at the composite's output interface.
        self.bridge_updates: List[Any] = []

        # Build the dependency network between steps and determine which steps should run first.
        self.build_step_network()

        # Run all steps that are ready on the first cycle.
        self.run_steps(self.to_run)

    def find_instance_paths(self, state: Dict[str, Any]) -> None:
        """
        Identify all Step and Process instances in the current state.

        Populates:
            - self.process_paths
            - self.step_paths
        """
        self.process_paths = find_instance_paths(state, 'process_bigraph.composite.Process')
        self.step_paths = find_instance_paths(state, 'process_bigraph.composite.Step')

    def merge(self, schema: Dict[str, Any], state: Dict[str, Any], path: Optional[List[str]] = None) -> None:
        """
        Merge a new schema/state subtree into the Composite.

        Args:
            schema: Schema dictionary to merge.
            state: State dictionary to merge.
            path: Path where merge should occur (default: root).
        """
        path = path or []
        self.composition, self.state = self.core.merge(
            self.composition,
            self.state,
            path,
            schema,
            state)
        self.find_instance_paths(self.state)

    def apply(self, update: Dict[str, Any], path: Optional[List[str]] = None) -> None:
        """
        Apply an update to the current state.

        Args:
            update: A state update dictionary.
            path: Optional path to scope the update under.
        """
        path = path or []
        scoped_update = set_path({}, path, update)
        self.state = self.core.apply(self.composition, self.state, scoped_update)
        self.find_instance_paths(self.state)

    def inputs(self) -> Dict[str, Any]:
        """Return the composite's input schema (wired to the bridge)."""
        return self.process_schema.get('inputs', {})

    def outputs(self) -> Dict[str, Any]:
        """Return the composite's output schema (wired to the bridge)."""
        return self.process_schema.get('outputs', {})

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize the internal state using the core serializer.

        Returns:
            A serialized representation of the current state.
        """
        return self.core.serialize(self.composition, self.state)

    def serialize_schema(self) -> Dict[str, Any]:
        """
        Serialize the composition (schema) using the core serializer.

        Returns:
            A serialized schema representation.
        """
        return self.core.serialize('schema', self.composition)

    def save(
            self,
            filename: str = 'composite.json',
            outdir: str = 'out',
            schema: bool = False,
            state: bool = False
    ) -> None:
        """
        Save the composite to a JSON file.

        WARNING: This method is deprecated. Use Vivarium's native tools instead.

        Args:
            filename: Output filename.
            outdir: Output directory.
            schema: Whether to include the serialized schema.
            state: Whether to include the serialized state.
        """
        print("Warning: save() is deprecated and will be removed in a future version.")

        if not schema and not state:
            schema = state = True

        document = {}
        if state:
            document['state'] = self.serialize_state()
        if schema:
            document['composition'] = self.serialize_schema()

        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, filename)
        with open(filepath, 'w') as f:
            json.dump(document, f, indent=4)
            print(f"Saved composite to {filepath}")



    def build_step_network(self):
        self.step_triggers = {}
        self.star_triggers = {}
        for step_path, step in self.step_paths.items():
            step_triggers = find_step_triggers(
                step_path, step)
            self.step_triggers = merge_collections(
                self.step_triggers,
                step_triggers)
        for trigger in self.step_triggers:
            if "*" in trigger:
                self.star_triggers[trigger] = self.step_triggers[trigger]
        self.steps_run = set([])

        self.step_dependencies, self.node_dependencies = build_step_network(
            self.step_paths)

        self.reset_step_state(
            self.step_paths)

        self.to_run = self.cycle_step_state()

    def reset_step_state(self, step_paths):
        self.trigger_state = build_trigger_state(
            self.node_dependencies)
        self.steps_remaining = set(step_paths)

    def cycle_step_state(self):
        to_run, self.steps_remaining, self.trigger_state = determine_steps(
            self.step_dependencies,
            self.steps_remaining,
            self.trigger_state)
        return to_run

    def merge_schema(self, schema, path=None):
        path = path or []
        schema = set_path({}, path, schema)
        self.composition = self.core.merge_schemas(self.composition, schema)
        self.composition, self.state = self.core.generate(self.composition, self.state)
        self.find_instance_paths(self.state)

    def process_update(
            self,
            path,
            process,
            states,
            interval,
            ports_key='outputs'):

        """Start generating a process's update.

        This function is similar to :py:meth:`_invoke_process` except in
        addition to triggering the computation of the process's update
        (by calling ``_invoke_process``), it also generates a
        :py:class:`Defer` object to transform the update into absolute
        terms.

        Args:
            path: Path to process.
            process: The process.
            states: Simulation state to pass to process's
                ``next_update`` method.
            interval: Interval for which to compute the update.

        Returns:
            Tuple of the deferred update (in absolute terms) and
            ``store``.
        """

        states = strip_schema_keys(states)

        update = process['instance'].invoke(states, interval)

        def defer_project(update, args):
            schema, state, path = args
            return self.core.project_edge(
                schema,
                state,
                path,
                update,
                ports_key)

        absolute = Defer(
            update,
            defer_project, (
                self.composition,
                self.state,
                path))

        return absolute


    def run_process(self, path, process, end_time, full_step, force_complete):
        if path not in self.front:
            self.front[path] = empty_front(
                self.state['global_time'])

        process_time = self.front[path]['time']
        if process_time <= self.state['global_time']:
            if self.front[path].get('future'):
                future_front = self.front[path]['future']
                process_interval = future_front['interval']
                store = future_front['store']
                state = future_front['state']
                del self.front[path]['future']
            else:
                state = self.core.view_edge(
                    self.composition,
                    self.state,
                    path)

                process_interval = process['interval']

            if force_complete:
                # force the process to complete at end_time
                future = min(process_time + process_interval, end_time)
            else:
                future = process_time + process_interval

            if self.global_time_precision is not None:
                # set future time based on global_time_precision
                future = round(future, self.global_time_precision)

            # absolute interval
            interval = future - self.state['global_time']
            if interval < full_step:
                full_step = interval

            if future <= end_time:
                update = self.process_update(
                    path,
                    process,
                    state,
                    process_interval
                )

                # update front, to be applied at its projected time
                self.front[path]['time'] = future
                self.front[path]['update'] = update

        else:
            # don't shoot past processes that didn't run this time
            process_delay = process_time - self.state['global_time']
            if process_delay < full_step:
                full_step = process_delay

        return full_step


    def read_bridge(self, state=None):
        if state is None:
            state = self.state

        bridge_view = self.core.view(
            self.interface()['outputs'],
            self.bridge['outputs'],
            (),
            top_schema=self.composition,
            top_state=state)

        return bridge_view


    def apply_updates(self, updates):
        # view_expire = False
        update_paths = []

        for defer in updates:
            series = defer.get()
            if series is None:
                continue

            if not isinstance(series, list):
                series = [series]

            for update in series:
                paths = hierarchy_depth(update)
                update_paths.extend(paths.keys())

                self.state = self.core.apply_update(
                    self.composition,
                    self.state,
                    update)

                bridge_update = self.read_bridge(
                    update)

                if bridge_update:
                    self.bridge_updates.append(
                        bridge_update)

        self.find_instance_paths(
            self.state)

        return update_paths

                # view_expire_update = self.apply_update(up, store)
                # view_expire = view_expire or view_expire_update

        # if view_expire:
        #     self.state.build_topology_views()


    def expire_process_paths(self, update_paths):
        for update_path in update_paths:
            for process_path in self.process_paths.copy():
                updated = all([
                    update == process
                    for update, process in zip(update_path, process_path)])

                if updated:
                    self.find_instance_paths(
                        self.state)
                    return

                    # del self.process_paths[process_path]

                    # target_schema, target_state = self.core.slice(
                    #     self.composition,
                    #     self.state,
                    #     update_path)

                    # process_subpaths = find_instance_paths(
                    #     target_state,
                    #     'process_bigraph.composite.Process')

                    # for subpath, process in process_subpaths.items():
                    #     process_path = update_path + subpath
                    #     self.process_paths[process_path] = process


    def run(self, interval, force_complete=False):
        # Define the end time for the run
        end_time = self.state['global_time'] + interval

        # Run the processes and apply updates until the end time is reached
        while self.state['global_time'] < end_time or force_complete:
            full_step = math.inf

            for path in self.process_paths:
                process = get_path(self.state, path)
                full_step = self.run_process(
                    path,
                    process,
                    end_time,
                    full_step,
                    force_complete)

            # apply updates based on process times in self.front
            if full_step == math.inf:
                # no processes ran, jump to next process
                next_event = end_time
                for path in self.front.keys():
                    if self.front[path]['time'] < next_event:
                        next_event = self.front[path]['time']
                self.state['global_time'] = next_event

            elif self.state['global_time'] + full_step <= end_time:
                # at least one process ran within the interval
                # increase the time, apply updates, and continue
                self.state['global_time'] += full_step

                # apply updates that are behind global time
                updates = []
                paths = []
                for path, advance in self.front.items():
                    if advance['time'] <= self.state['global_time'] \
                            and advance['update']:
                        new_update = advance['update']
                        updates.append(new_update)
                        advance['update'] = {}
                        paths.append(path)

                # get all update paths, then trigger steps that
                # depend on those paths
                update_paths = self.apply_updates(updates)

                self.expire_process_paths(update_paths)
                self.trigger_steps(update_paths)

            else:
                # all processes have run past the interval
                self.state['global_time'] = end_time

            if force_complete and self.state['global_time'] == end_time:
                force_complete = False


    def run_steps(self, step_paths):
        if len(step_paths) > 0:
            updates = []
            for step_path in step_paths:
                step = get_path(
                    self.state,
                    step_path)

                state = self.core.view_edge(
                    self.composition,
                    self.state,
                    step_path,
                    'inputs')

                step_update = self.process_update(
                    step_path,
                    step,
                    state,
                    -1.0,
                    'outputs')

                updates.append(step_update)

            update_paths = self.apply_updates(updates)
            self.expire_process_paths(update_paths)
            to_run = self.cycle_step_state()

            if len(to_run) > 0:
                self.run_steps(to_run)
            else:
                self.steps_run = set([])

        else:
            self.steps_run = set([])


    def trigger_steps(self, update_paths):
        steps_to_run = []

        for update_path in update_paths:
            paths = explode_path(update_path)
            for path in paths:
                step_paths = self.step_triggers.get(path, [])
                if self.star_triggers:
                    for star_trigger, star_steps in self.star_triggers.items():
                        if match_star_path(path, star_trigger):
                            step_paths.extend(star_steps)
                for step_path in step_paths:
                    if step_path is not None and step_path not in self.steps_run:
                        steps_to_run.append(step_path)
                        self.steps_run.add(step_path)

        steps_to_run = find_downstream(
            self.step_dependencies,
            self.node_dependencies,
            steps_to_run)

        self.reset_step_state(steps_to_run)
        to_run = self.cycle_step_state()

        self.run_steps(to_run)


    def update(self, state, interval):
        # do everything

        projection = self.core.project(
            self.interface()['inputs'],
            self.bridge['inputs'],
            [],
            state)

        self.merge(
            {},
            projection)

        self.run(interval)

        updates = self.bridge_updates
        self.bridge_updates = []

        return updates