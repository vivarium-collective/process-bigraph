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
    Mapping, MutableMapping, Sequence,
    Callable, Type
)
import collections

from bigraph_schema import (
    Edge, Registry, TypeSystem, visit_method,
    get_path, set_path, resolve_path, hierarchy_depth, deep_merge,
    is_schema_key, strip_schema_keys)

from bigraph_schema.protocols import local_lookup_module


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


class Open(Edge):
    METHOD_COMMANDS = (
        'initial_state', 'inputs', 'outputs', 'update')

    ATTRIBUTE_READ_COMMANDS = (
        'config', 'composition', 'state')


    def __init__(self, config=None, core=None):
        self._command_result: Any = None
        self._pending_command: Optional[
            Tuple[str, Optional[tuple], Optional[dict]]] = None

        super().__init__(config, core=core)

    def pre_send_command(
            self, command: str, args: Optional[tuple], kwargs:
            Optional[dict]) -> None:
        '''Run pre-checks before starting a command.

        This method should be called at the start of every
        implementation of :py:meth:`send_command`.

        Args:
            command: The name of the command to run.
            args: A tuple of positional arguments for the command.
            kwargs: A dictionary of keyword arguments for the command.

        Raises:
            RuntimeError: Raised when a user tries to send a command
                while a previous command is still pending (i.e. the user
                hasn't called :py:meth:`get_command_result` yet for the
                previous command).
        '''
        if self._pending_command:
            raise RuntimeError(
                f'Trying to send command {(command, args, kwargs)} but '
                f'command {self._pending_command} is still pending.')
        self._pending_command = command, args, kwargs


    def send_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            run_pre_check: bool = True) -> None:
        '''Handle :term:`process commands`.

        This method handles the commands listed in
        :py:attr:`METHOD_COMMANDS` by passing ``args``
        and ``kwargs`` to the method of ``self`` with the name
        of the command and saving the return value as the result.

        This method handles the commands listed in
        :py:attr:`ATTRIBUTE_READ_COMMANDS` by returning the attribute of
        ``self`` with the name matching the command, and it handles the
        commands listed in :py:attr:`ATTRIBUTE_WRITE_COMMANDS` by
        setting the attribute in the command to the first argument in
        ``args``. The command must be named ``set_attr`` for attribute
        ``attr``.

        To add support for a custom command, override this function in
        your subclass. Each command is defined by a name (a string)
        and accepts both positional and keyword arguments. Any custom
        commands you add should have associated methods such that:

        * The command name matches the method name.
        * The command and method accept the same positional and keyword
          arguments.
        * The command and method return the same values.

        If all of the above are satisfied, you can use
        :py:meth:`Process.run_command_method` to handle the command.

        Your implementation of this function needs to handle all the
        commands you want to support.  When presented with an unknown
        command, you should call the superclass method, which will
        either handle the command or call its superclass method. At the
        top of this recursive chain, this ``Process.send_command()``
        method handles some built-in commands and will raise an error
        for unknown commands.

        Any overrides of this method must also call
        :py:meth:`pre_send_command` at the start of the method. This
        call will check that no command is currently pending to avoid
        confusing behavior when multiple commands are started without
        intervening retrievals of command results. Since your overriding
        method will have already performed the pre-check, it should pass
        ``run_pre_check=False`` when calling the superclass method.

        Args:
            command: The name of the command to run.
            args: A tuple of positional arguments for the command.
            kwargs: A dictionary of keyword arguments for the command.
            run_pre_check: Whether to run the pre-checks implemented in
                :py:meth:`pre_send_command`. This should be left at its
                default value unless the pre-checks have already been
                performed (e.g. if this method is being called by a
                subclass's overriding method.)

        Returns:
            None. This method just starts the command running.

        Raises:
            ValueError: For unknown commands.
        '''
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)
        args = args or tuple()
        kwargs = kwargs or {}
        if command in self.METHOD_COMMANDS:
            self._command_result = self.run_command_method(
                command, args, kwargs)
        elif command in self.ATTRIBUTE_READ_COMMANDS:
            self._command_result = getattr(self, command)
        # elif command in self.ATTRIBUTE_WRITE_COMMANDS:
        #     assert command.startswith('set_')
        #     assert args
        #     setattr(self, command[len('set_'):], args[0])
        else:
            raise ValueError(
                f'Process {self} does not understand the process '
                f'command {command}')

    def run_command_method(
            self, command: str, args: tuple, kwargs: dict) -> Any:
        '''Run a command whose name and interface match a method.

        Args:
            command: The command name, which must equal to a method of
                ``self``.
            args: The positional arguments to pass to the method.
            kwargs: The keywords arguments for the method.

        Returns:
            The result of calling ``self.command(*args, **kwargs)`` is
            returned for command ``command``.
        '''
        return getattr(self, command)(*args, **kwargs)

    def get_command_result(self) -> Any:
        '''Retrieve the result from the last-run command.

        Returns:
            The result of the last command run. Note that this method
            should only be called once immediately after each call to
            :py:meth:`send_command`.

        Raises:
            RuntimeError: When there is no command pending. This can
                happen when this method is called twice without an
                intervening call to :py:meth:`send_command`.
        '''
        if not self._pending_command:
            raise RuntimeError(
                'Trying to retrieve command result, but no command is '
                'pending.')
        self._pending_command = None
        result = self._command_result
        self._command_result = None
        return result

    def run_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None) -> Any:
        '''Helper function that sends a command and returns result.'''
        self.send_command(command, args, kwargs)
        return self.get_command_result()


class Step(Open):
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

    def update(self, state: Dict[str, Any], interval=None) -> Dict[str, Any]:
        """
        Compute and return the update for the step.

        Override this method in subclasses to define the step's logic.

        Args:
            state: The current simulation state at the step's inputs.

        Returns:
            A dictionary representing the update to apply.
        """
        return {}


class Process(Open):
    """
    Process base class.

    A `Process` is a temporal unit of computation that operates on state and advances in time.
    Each subclass must implement the `update()` method and optionally the `invoke()` method.

    Processes are stateful and typically used for simulations of continuous or discrete dynamics.
    """

    def invoke(self, state: Dict[str, Any], interval: float):
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


def as_step(inputs, outputs, core=None):
    """
    Decorator to create a Step from a function named update_*.
    If core is provided, registers under the name *.
    """
    def decorator(func):
        assert func.__name__.startswith('update_'), "Function name must be of the form update_*"
        step_name = func.__name__[len('update_'):]

        class FunctionStep(Step):
            def inputs(self):
                return inputs

            def outputs(self):
                return outputs

            def update(self, state):
                return func(state)

        FunctionStep.__name__ = step_name + 'Step'

        if core is not None:
            core.register_process(step_name, FunctionStep)

        return FunctionStep

    return decorator


def as_process(inputs, outputs, core=None):
    """
    Decorator to create a Process from a function named update_*.
    If core is provided, registers under the name *.
    """
    def decorator(func):
        assert func.__name__.startswith('update_'), "Function name must be of the form update_*"
        process_name = func.__name__[len('update_'):]

        class FunctionProcess(Process):
            def __init__(self, config=None, core=None):
                super().__init__(config=config, core=core)

            def inputs(self):
                return inputs

            def outputs(self):
                return outputs

            def update(self, state, interval):
                return func(state, interval)

        FunctionProcess.__name__ = process_name + 'Process'

        if core is not None:
            core.register_process(process_name, FunctionProcess)

        return FunctionProcess

    return decorator


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


    # ==============================
    # Initialization & Configuration
    # ==============================

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

        # initialize an empty front for finding the instance paths
        self.front = {}

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
          
    def clean_front(self, state):
        self.find_instance_paths(state)

    def find_instance_paths(self, state: Dict[str, Any]) -> None:
        """
        Identify all Step and Process instances in the current state.

        Populates:
            - self.process_paths
            - self.step_paths
        """
        self.process_paths = find_instance_paths(state, 'process_bigraph.composite.Process')
        self.step_paths = find_instance_paths(state, 'process_bigraph.composite.Step')

        all_paths = set(
            list(self.process_paths.keys()) +
            list(self.step_paths.keys()))

        front_paths = set(
            self.front.keys())

        for removed_key in front_paths.difference(all_paths):
            # do we want to do anything with these?
            removed_front = self.front.pop(removed_key)

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

    def merge_schema(
            self,
            schema: Dict[str, Any],
            path: Optional[List[str]] = None
    ) -> None:
        """
        Merge a new schema subtree into the current composite schema and regenerate state.

        Args:
            schema: The schema subtree to merge.
            path: Optional path at which to merge the schema (defaults to root).
        """
        path = path or []

        # Set the new schema subtree at the given path
        scoped_schema = set_path({}, path, schema)

        # Merge it into the existing schema
        self.composition = self.core.merge_schemas(self.composition, scoped_schema)

        # Re-generate state based on the new schema structure
        self.composition, self.state = self.core.generate(self.composition, self.state)

        # Re-scan the state tree for processes and steps
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


    # ===================
    # Serialization & I/O
    # ===================

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

        Args:
            filename: Output filename.
            outdir: Output directory.
            schema: Whether to include the serialized schema.
            state: Whether to include the serialized state.
        """
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


    # ==================
    # Interface & Wiring
    # ==================

    def inputs(self) -> Dict[str, Any]:
        """Return the composite's input schema (wired to the bridge)."""
        return self.process_schema.get('inputs', {})

    def outputs(self) -> Dict[str, Any]:
        """Return the composite's output schema (wired to the bridge)."""
        return self.process_schema.get('outputs', {})

    def read_bridge(self, state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        View the external bridge output ports using the current or provided state.

        This method uses the composite's interface and bridge configuration to extract
        the substate that corresponds to external output ports.

        Args:
            state: Optional state dictionary. If not provided, uses `self.state`.

        Returns:
            A dictionary of output values from the bridge view, or None if not found.
        """
        state = state or self.state

        bridge_view = self.core.view(
            self.interface()['outputs'],
            self.bridge['outputs'],
            (),
            top_schema=self.composition,
            top_state=state
        )

        return bridge_view


    # =======================
    # Step Network Management
    # =======================

    def build_step_network(self) -> None:
        """
        Construct the internal dependency network for all registered steps.

        This includes:
        - Finding trigger paths for each step based on their input wires
        - Registering wildcard triggers (e.g. `*`-based patterns)
        - Building a graph of data dependencies between steps
        - Initializing tracking structures for trigger state and pending steps
        - Populating the `self.to_run` queue with steps ready to execute
        """
        self.step_triggers: Dict[Tuple[str, ...], List[Union[str, Tuple[str, ...]]]] = {}
        self.star_triggers: Dict[Tuple[str, ...], List[Union[str, Tuple[str, ...]]]] = {}

        # Collect triggers for each step's input schema
        for step_path, step in self.step_paths.items():
            step_triggers = find_step_triggers(step_path, step)
            self.step_triggers = merge_collections(self.step_triggers, step_triggers)

        # Identify wildcard-based triggers (those containing '*')
        for trigger_path in self.step_triggers:
            if "*" in trigger_path:
                self.star_triggers[trigger_path] = self.step_triggers[trigger_path]

        # Track which steps have already executed in the current cycle
        self.steps_run: Set[Union[str, Tuple[str, ...]]] = set()

        # Build the step execution dependency graph
        self.step_dependencies, self.node_dependencies = build_step_network(self.step_paths)

        # Initialize trigger fulfillment state and steps remaining
        self.reset_step_state(self.step_paths)

        # Compute the initial set of runnable steps
        self.to_run = self.cycle_step_state()

        self.clean_front(self.state)

    def reset_step_state(
            self,
            step_paths: Dict[Union[str, Tuple[str, ...]], Any]
        ) -> None:
        """
        Reset the trigger tracking state for a given set of steps.

        Args:
            step_paths: A dictionary of step paths (as keys).
        """
        # Start with a fresh trigger state from the dependency graph
        self.trigger_state = build_trigger_state(self.node_dependencies)

        # Track steps still waiting to be executed in this cycle
        self.steps_remaining: Set[Union[str, Tuple[str, ...]]] = set(step_paths)

    def cycle_step_state(self) -> List[Union[str, Tuple[str, ...]]]:
        """
        Evaluate the current trigger state and determine which steps can run.

        Returns:
            A list of step paths that are ready to be invoked in this cycle.
        """
        to_run, self.steps_remaining, self.trigger_state = determine_steps(
            self.step_dependencies,
            self.steps_remaining,
            self.trigger_state
        )
        return to_run

    def trigger_steps(self, update_paths: List[Tuple[str, ...]]) -> None:
        """
        Determine and run step processes triggered by recent state updates.

        Args:
            update_paths: Paths in the state that were updated.
        """
        steps_to_run: List[Tuple[str, ...]] = []

        for update_path in update_paths:
            for path in explode_path(update_path):
                # Check direct trigger matches
                step_paths = self.step_triggers.get(path, [])

                # Also handle wildcard (*) path matches
                if self.star_triggers:
                    for star_trigger, star_steps in self.star_triggers.items():
                        if match_star_path(path, star_trigger):
                            step_paths.extend(star_steps)

                # Add unrun steps to the execution queue
                for step_path in step_paths:
                    if step_path is not None and step_path not in self.steps_run:
                        steps_to_run.append(step_path)
                        self.steps_run.add(step_path)

        # Identify downstream steps dependent on triggered ones
        steps_to_run = find_downstream(
            self.step_dependencies,
            self.node_dependencies,
            steps_to_run
        )

        self.reset_step_state(steps_to_run)
        to_run = self.cycle_step_state()
        self.run_steps(to_run)

    def run_steps(self, step_paths: List[Tuple[str, ...]]) -> None:
        """
        Execute a list of step processes, apply their updates, and handle cascading triggers.

        Args:
            step_paths: A list of step path tuples to run.
        """
        if step_paths:
            updates = []

            for step_path in step_paths:
                step = get_path(self.state, step_path)
                state = self.core.view_edge(
                    self.composition, self.state, step_path, 'inputs'
                )

                # Steps are always invoked with interval = -1.0
                step_update = self.process_update(
                    step_path, step, state, -1.0, 'outputs'
                )
                updates.append(step_update)

            update_paths = self.apply_updates(updates)
            self.expire_process_paths(update_paths)

            to_run = self.cycle_step_state()

            if to_run:
                self.run_steps(to_run)
            else:
                self.steps_run = set()
        else:
            self.steps_run = set()


    # ====================
    # Simulation Execution
    # ====================

    def run(self, interval: float, force_complete: bool = False) -> None:
        """
        Advance simulation by running processes until a target time is reached.

        The method loops through all registered processes and executes their updates
        incrementally based on their configured interval. Updates are applied and
        steps are triggered accordingly.

        Args:
            interval: Time interval to simulate.
            force_complete: If True, forces all processes to reach the end time.
        """
        end_time = self.state['global_time'] + interval

        while self.state['global_time'] < end_time or force_complete:
            full_step = math.inf

            # Run each process and compute the minimum time step that advances simulation
            for path in self.process_paths:
                process = get_path(self.state, path)
                full_step = self.run_process(
                    path, process, end_time, full_step, force_complete)

            if full_step == math.inf:
                # No process ran — jump to the next scheduled process time
                next_event = end_time
                for path in self.front.keys():
                    if self.front[path]['time'] < next_event:
                        next_event = self.front[path]['time']
                self.state['global_time'] = next_event

            elif self.state['global_time'] + full_step <= end_time:
                # At least one process ran — advance time and apply its update
                self.state['global_time'] += full_step
                updates = []
                paths = []

                for path, advance in self.front.items():
                    if advance['time'] <= self.state['global_time'] and advance['update']:
                        updates.append(advance['update'])
                        advance['update'] = {}
                        paths.append(path)

                update_paths = self.apply_updates(updates)
                self.expire_process_paths(update_paths)
                self.trigger_steps(update_paths)

            else:
                # All remaining process events are beyond end_time
                self.state['global_time'] = end_time

            if force_complete and self.state['global_time'] == end_time:
                force_complete = False

    def run_process(
            self,
            path: Union[str, Tuple[str, ...]],
            process: Dict[str, Any],
            end_time: float,
            full_step: float,
            force_complete: bool
    ) -> float:
        """
        Run a process at a given path and determine its next scheduled time.

        This updates the `self.front` to store when the process is due next,
        and captures its update as a deferred computation.

        Args:
            path: The path to the process in the state/composition tree.
            process: The dictionary representing the process (must contain 'interval').
            end_time: The simulation time to run up to.
            full_step: The current smallest time step among all processes.
            force_complete: If True, forces the process to reach `end_time` exactly.

        Returns:
            The updated `full_step`, i.e., the shortest remaining time across all processes.
        """
        # Initialize the front buffer for this process if missing
        if path not in self.front:
            self.front[path] = empty_front(self.state['global_time'])

        process_time = self.front[path]['time']

        if process_time <= self.state['global_time']:
            # Use future state if already scheduled and saved
            if 'future' in self.front[path]:
                future_front = self.front[path].pop('future')
                process_interval = future_front['interval']
                state = future_front['state']
            else:
                # Otherwise, slice the current state for the process
                state = self.core.view_edge(self.composition, self.state, path)
                process_interval = process['interval']

            # Determine the target time for the next update
            future = (
                min(process_time + process_interval, end_time)
                if force_complete
                else process_time + process_interval
            )

            # Apply rounding if global time precision is set
            if self.global_time_precision:
                future = round(future, self.global_time_precision)

            # Compute how long this process would advance
            interval = future - self.state['global_time']
            if interval < full_step:
                full_step = interval

            # Only proceed if the next step occurs within the target range
            if future <= end_time:
                update = self.process_update(path, process, state, process_interval)

                # Store the update to apply when simulation reaches `future` time
                self.front[path]['time'] = future
                self.front[path]['update'] = update

        else:
            # This process is scheduled in the future — ensure we don't skip ahead
            process_delay = process_time - self.state['global_time']
            if process_delay < full_step:
                full_step = process_delay

        return full_step

    def process_update(
            self,
            path: Union[str, Tuple[str, ...]],
            process: Dict[str, Any],
            states: Dict[str, Any],
            interval: float,
            ports_key: str = 'outputs'
    ) -> Defer:
        """
        Start generating a process's update and wrap it in a deferred transformation.

        This is similar to invoking a process directly, but it delays transformation
        into absolute state terms until `.get()` is called on the returned `Defer` object.

        Args:
            path: The path to the process in the state/composition tree.
            process: The dictionary representing the process instance (must include 'instance').
            states: The current state values at the process’s ports.
            interval: The time interval to simulate.
            ports_key: Which port ('inputs' or 'outputs') to use when projecting the update.

        Returns:
            A `Defer` object that, when resolved, transforms the update to absolute paths.
        """
        # Strip schema-specific metadata from the state
        clean_state = strip_schema_keys(states)

        # Invoke the process and retrieve a wrapped SyncUpdate object
        update = process['instance'].invoke(clean_state, interval)

        # This nested function projects the update into the global state at the given path
        def defer_project(update_result: Any, args: Tuple[Any, Any, Union[str, Tuple[str, ...]]]) -> Any:
            schema, state, process_path = args
            return self.core.project_edge(schema, state, process_path, update_result, ports_key)

        # Return a deferred object that will project the update when requested
        return Defer(update, defer_project, (self.composition, self.state, path))

    def apply_updates(self, updates: List["Defer"]) -> List[Union[str, Tuple[str, ...]]]:
        """
        Apply a series of deferred updates and record the resulting bridge outputs.

        For each update in the list, the deferred `.get()` method is called, which
        may return a single update or a list of updates. Each is then applied to the
        composite's state, and corresponding bridge outputs are captured.

        Args:
            updates: A list of `Defer` objects representing delayed update functions.

        Returns:
            A list of update paths (used to determine which processes to refresh).
        """
        update_paths = []

        for defer in updates:
            # Resolve deferred computation to get update(s)
            series = defer.get()
            if series is None:
                continue
            if not isinstance(series, list):
                series = [series]

            for update in series:
                # if update and isinstance(update, dict) and 'environment' in update and update['environment'] and isinstance(update['environment'], dict) and '_react' in update['environment']:
                #     import ipdb; ipdb.set_trace()

                # Extract all hierarchical paths touched by this update
                paths = hierarchy_depth(update)
                update_paths.extend(paths.keys())

                # Apply update directly to the internal state
                self.state = self.core.apply_update(self.composition, self.state, update)

                # Read updated bridge outputs, if available
                bridge_update = self.read_bridge(update)
                if bridge_update:
                    self.bridge_updates.append(bridge_update)

        # Refresh process and step instance paths
        self.find_instance_paths(self.state)

        return update_paths

    def expire_process_paths(self, update_paths: List[Union[str, Tuple[str, ...]]]) -> None:
        """
        Invalidate and refresh process paths if affected by recent updates.

        This is used to ensure that processes are rediscovered if a state update
        altered a region where a process instance may be added, removed, or replaced.

        Args:
            update_paths: A list of hierarchical paths that were modified.
        """
        for update_path in update_paths:
            for process_path in self.process_paths.copy():
                # Match if update path completely overlaps the process path prefix
                updated = all(update == process for update, process in zip(update_path, process_path))
                if updated:
                    self.find_instance_paths(self.state)
                    return  # Exit early after one match, as paths are re-evaluated


    # ====================
    # Update Integration
    # ====================
    # This section handles how updates from processes and steps are applied to
    # the global state, how downstream effects are triggered, and how bridge
    # outputs are collected.

    def update(self, state: Dict[str, Any], interval: float) -> List[Dict[str, Any]]:
        """
        Project input state, run the simulation interval, and return bridge updates.

        This is the main entry point for executing a time step of the composite.
        It performs:
        - Input projection using the bridge schema
        - Merging projected input into state
        - Executing processes for the given time interval
        - Returning updates for the bridge output

        Args:
            state: Input state to project into the composite.
            interval: Time interval to simulate.

        Returns:
            A list of updates generated for the bridge outputs.
        """
        projection = self.core.project(
            self.interface()['inputs'],
            self.bridge['inputs'],
            [],
            state
        )

        self.merge({}, projection)
        self.run(interval)

        updates = self.bridge_updates
        self.bridge_updates = []

        return updates
