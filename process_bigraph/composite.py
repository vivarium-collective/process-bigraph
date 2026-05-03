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
import time as _time
import numpy as np

from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Mapping, MutableMapping, Sequence,
    Callable, Type
)
import collections

from bigraph_schema import (
    Edge,
    get_path, set_path, resolve_path, hierarchy_depth,
    is_schema_key, strip_schema_keys)

from bigraph_schema.protocols import local_lookup_module
from bigraph_schema.methods.events import NodeAdded, NodeRemoved, Divided


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
            instance = inner.get('instance')

            if isinstance(instance, process_class):
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

    Uses the instance's ``triggers()`` method if available to determine
    which input ports trigger the step. Ports not in ``triggers()`` are
    still received as inputs but don't cause the step to re-run.

    Args:
        path: Path to the step in the composite model tree.
        step: Step object containing an 'inputs' field with wire mappings.

    Returns:
        Mapping from trigger paths to lists of step paths that they trigger.
    """
    prefix = tuple(path[:-1])
    triggers: Dict[Tuple[str, ...], List[Union[List[str], Tuple[str, ...]]]] = {}

    wires = step.get('inputs', {})
    instance = step.get('instance')

    # Check if the instance declares a triggers() subset
    trigger_port_names = None
    if instance is not None and hasattr(instance, 'triggers'):
        try:
            trigger_schema = instance.triggers()
            input_schema = instance.inputs()
            # Only filter if triggers() returns something different from inputs()
            if trigger_schema is not input_schema and trigger_schema != input_schema:
                trigger_port_names = set(trigger_schema.keys())
        except Exception:
            pass

    # Also check for _triggers key in the step state
    if trigger_port_names is None and '_triggers' in step:
        trigger_port_names = set(step['_triggers'].keys()) if isinstance(step['_triggers'], dict) else None

    if trigger_port_names is not None:
        trigger_wires = {k: v for k, v in wires.items() if k in trigger_port_names}
        wire_paths = find_leaves(trigger_wires, path=prefix)
    else:
        wire_paths = find_leaves(wires, path=prefix)

    for wire in wire_paths:
        trigger_path = resolve_path(tuple(wire))
        if isinstance(trigger_path, list):
            raise ValueError(f'resolve_path returned a list instead of a tuple: {trigger_path}')
        triggers.setdefault(trigger_path, []).append(path)

    return triggers


def wire_step_layers(
        state: Dict[str, Any],
        dep_graph: Dict[str, List[Any]],
        flow_path: str = 'step_flow',
        token_prefix: str = 'layer_',
        root_trigger: str = 'global_time',
) -> Dict[int, List[str]]:
    """Wire steps in `state` for layer-batched execution.

    Given a step dependency graph, compute each step's topological depth
    and wire them so that all steps in the same layer share an incoming
    and outgoing trigger token. The composite engine's run_steps() will
    then batch every step in a layer into one apply_updates call, yielding
    per-layer atomicity (each step in a layer sees the same starting
    state — the same semantics as vivarium's per-layer execution).

    The dependency graph is the source of truth for ordering. Steps in
    the same layer are independent of each other, so the composite's
    `cycle_step_state` returns the whole layer in one batch and
    `run_steps` collects all their updates and reconciles them via
    `apply_updates`.

    Args:
        state: The composite state dict containing step edges. Each step
            referenced in `dep_graph` is mutated in place: its `inputs`,
            `outputs`, `_dep_outputs`, and `_triggers` are populated to
            point at this layer's tokens.
        dep_graph: Mapping of step_name -> list of dependency identifiers.
            Each dependency is either a bare step name (str) or a path
            tuple/list whose last element is the step name.
        flow_path: State key under which layer tokens live. Defaults to
            '_flow'. The key is created in `state` if not already present.
        token_prefix: Token name prefix. Defaults to '_layer_'.
        root_trigger: Trigger path for layer-0 steps (those with no deps).
            Defaults to 'global_time', so the first layer fires whenever
            the global clock advances.

    Returns:
        A dict mapping layer index -> list of step names in that layer
        (useful for inspection / debugging the dep DAG).

    Notes:
        - Each step's `_triggers` is set to ONLY the layer-token input,
          so other inputs are received but don't re-trigger the step.
        - Steps in the same layer all write to the same outgoing token.
          `apply_updates` reconciles those writes (Integer apply is
          additive, but the precise value doesn't matter — the change
          fires the next layer exactly once per `run_steps` cycle).
        - Layer-0 steps trigger from `root_trigger` (typically the global
          clock). All subsequent layers trigger from `_layer_{N-1}`.
    """
    # 1. Topological depth: each step's level is max(deps' level) + 1.
    levels: Dict[str, int] = {}
    for step_name in dep_graph.keys():
        deps = dep_graph.get(step_name) or []
        if not deps:
            levels[step_name] = 0
            continue
        max_dep_level = -1
        for dep_path in deps:
            dep_name = dep_path[-1] if isinstance(dep_path, (list, tuple)) else dep_path
            if dep_name in levels:
                max_dep_level = max(max_dep_level, levels[dep_name])
        levels[step_name] = max_dep_level + 1

    # 2. Group steps by layer.
    layers: Dict[int, List[str]] = {}
    for step_name, level in levels.items():
        layers.setdefault(level, []).append(step_name)

    # 3. Initialize the flow tokens in state.
    state.setdefault(flow_path, {})
    for level in sorted(layers.keys()):
        state[flow_path][f'{token_prefix}{level}'] = 0

    # 4. Wire each step's trigger and outgoing token wire.
    # Flow ports are added to both the port schemas (_inputs/_outputs)
    # and the wire maps (inputs/outputs) so they survive realize().
    for level in sorted(layers.keys()):
        in_token = f'{token_prefix}{level - 1}' if level > 0 else None
        out_token = f'{token_prefix}{level}'
        for name in layers[level]:
            step = state.get(name)
            if not isinstance(step, dict):
                continue
            if 'instance' not in step and '_type' not in step:
                continue
            step.setdefault('inputs', {})
            step.setdefault('outputs', {})
            step.setdefault('_inputs', {})
            step.setdefault('_outputs', {})

            if in_token is not None:
                # Wire _flow_in as an input port
                step['inputs']['_flow_in'] = [flow_path, in_token]
                step['_inputs']['_flow_in'] = 'integer'
                step['_triggers'] = {'_flow_in': 'integer'}
            else:
                step['_triggers'] = {root_trigger: 'float'}

            # Wire _flow_out as an output port
            step['outputs']['_flow_out'] = [flow_path, out_token]
            step['_outputs']['_flow_out'] = 'integer'

    return layers


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
    path = path or ()

    if tree_structure is None:
        pass
    elif isinstance(tree_structure, list):
        leaves = tree_structure
    elif isinstance(tree_structure, tuple):
        leaves.append(tree_structure)
    else:
        for key, value in tree_structure.items():
            if isinstance(value, dict):
                subleaves = find_leaves(value, path=path)
                leaves.extend(subleaves)
            else:
                leaves.append(path + tuple(value))

    return leaves


def _pre_extract_edge_schemas(core, state, initial_schema, path=()):
    """Walk the state tree to find edges (processes/steps) and wire their
    port schemas into the schema tree so that realize uses correct types."""
    from bigraph_schema.schema import deep_merge

    schema = dict(initial_schema)

    def _walk(node, current_path):
        if not isinstance(node, dict):
            return
        # Check if this node is an edge (has instance + wires)
        instance = node.get('instance')
        if instance is not None and hasattr(instance, 'ports_schema'):
            wires = node.get('inputs', {})
            try:
                ports = instance.ports_schema()
            except Exception:
                return
            parent_path = current_path[:-1]
            for port_name, port_schema in ports.items():
                wire = wires.get(port_name)
                if wire is None:
                    wire = [port_name]
                if isinstance(wire, str):
                    wire = [wire]
                # Build absolute path from parent + wire
                abs_path = list(parent_path) + list(wire)
                # Set schema at this path
                target = schema
                for step in abs_path[:-1]:
                    if step not in target or not isinstance(target.get(step), dict):
                        target[step] = {}
                    target = target[step]
                last = abs_path[-1]
                if last not in target:
                    target[last] = port_schema
                # Don't overwrite existing schema — first one wins
        else:
            for key, value in node.items():
                if isinstance(key, str) and not key.startswith('_'):
                    _walk(value, current_path + (key,))

    _walk(state, ())
    return schema


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
        step_key: {'input_paths': None, 'output_paths': None, 'priority': None}
        for step_key in steps
    }
    nodes = {}

    for step_key, step in steps.items():
        schema = step['instance'].interface()
        assert_interface(schema)

        # Compute input paths for dependency graph.
        # Use triggers() if available to narrow which inputs create
        # dependency edges. Silent inputs are still received but don't
        # affect step scheduling.
        if ancestors[step_key]['input_paths'] is None:
            instance = step.get('instance')
            trigger_ports = None
            if instance is not None and hasattr(instance, 'triggers'):
                try:
                    trigger_schema = instance.triggers()
                    input_schema = instance.inputs()
                    if trigger_schema is not input_schema and trigger_schema != input_schema:
                        trigger_ports = set(trigger_schema.keys())
                except Exception:
                    pass
            if trigger_ports is None and '_triggers' in step:
                trigger_ports = set(step['_triggers'].keys()) if isinstance(step['_triggers'], dict) else None

            if trigger_ports is not None:
                trigger_wires = {k: v for k, v in step['inputs'].items() if k in trigger_ports}
                ancestors[step_key]['input_paths'] = find_leaves(trigger_wires, path=step_key[:-1])
            else:
                ancestors[step_key]['input_paths'] = find_leaves(step['inputs'], path=step_key[:-1])

        # Compute output paths once per step
        if ancestors[step_key]['output_paths'] is None:
            ancestors[step_key]['output_paths'] = find_leaves(
                step.get('_dep_outputs', step.get('outputs', {})),
                path=step_key[:-1])

        # Assign the priority
        if ancestors[step_key]['priority'] is None:
            ancestors[step_key]['priority'] = step.get('priority', 0.0)

        input_paths = ancestors[step_key]['input_paths'] or []
        output_paths = ancestors[step_key]['output_paths'] or []

        # Track which steps consume/produce each path
        for input_path in input_paths:
            path = tuple(input_path)
            nodes.setdefault(path, {'before': set(), 'after': set()})
            nodes[path]['after'].add(step_key)

        for output_path in output_paths:
            output_tuple = tuple(output_path)
            exploded_path = explode_path(output_path)[1:]
            for explode in exploded_path:
                # Skip self-loops: don't register this step as a
                # producer of paths it also consumes. Self-loops
                # can't trigger (a step runs at most once per cycle)
                # and they clutter the dependency graph.
                if explode not in input_paths:
                    path = tuple(explode)
                    nodes.setdefault(path, {'before': set(), 'after': set()})
                    nodes[path]['before'].add(step_key)


    # Second pass: propagate parent outputs to child input paths.
    # If step A produces path P and step B consumes P/X, then B depends on A.
    all_output_paths = {}
    for step_key in steps:
        for output_path in (ancestors[step_key]['output_paths'] or []):
            all_output_paths.setdefault(tuple(output_path), set()).add(step_key)

    # Collect all input paths for the reverse check
    all_input_paths = set()
    for step_key in steps:
        for input_path in (ancestors[step_key]['input_paths'] or []):
            all_input_paths.add(tuple(input_path))

    for node_path, deps in nodes.items():
        if deps['before']:
            continue  # already has producers
        # Check if any output is an ancestor of this input path
        for out_path, producers in all_output_paths.items():
            if (len(node_path) > len(out_path)
                    and node_path[:len(out_path)] == out_path):
                deps['before'].update(producers)

    # Also propagate child outputs to parent inputs.
    # If step A produces path P/X and step B consumes P, then B depends on A.
    for out_path, producers in all_output_paths.items():
        for input_path in all_input_paths:
            if (len(out_path) > len(input_path)
                    and out_path[:len(input_path)] == input_path):
                node = nodes.get(input_path)
                if node is not None:
                    # Add child producers, excluding self-loops
                    for producer in producers:
                        if producer not in node['after']:
                            node['before'].add(producer)

    return ancestors, nodes


def build_trigger_state(nodes, paths):
    """
    Initialize the trigger state from dependency nodes.

    Args:
        nodes: Dependency graph nodes with 'before' and 'after' sets.

    Returns:
        A mapping of paths to the set of steps waiting on those paths.
    """
    path_set = set(paths)

    return {
        key: set(value['before']).intersection(path_set)
        for key, value in nodes.items()}


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

    if not remaining:
        return to_run, remaining, fulfilled

    for step_path in list(remaining):
        step_inputs = steps[step_path].get('input_paths', []) or []
        if all(len(fulfilled[input]) == 0 for input in step_inputs):
            to_run.append(step_path)

    if not to_run:
        # All remaining steps are in cycles — pick the highest priority one.
        # This avoids expensive cycle detection for densely connected graphs.
        if remaining:
            priority = max(
                remaining,
                key=lambda path: steps[path]['priority'])
            to_run = [priority]
        else:
            return to_run, remaining, fulfilled

    for step_path in to_run:
        if step_path in remaining:
            remaining.remove(step_path)

        step_outputs = steps[step_path].get('output_paths', []) or []
        for output in step_outputs:
            exploded_path = explode_path(output)[1:]
            for explode in exploded_path:
                if step_path in fulfilled.get(explode, set()):
                    fulfilled[explode].remove(step_path)

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
        'config', 'schema', 'state')


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

    Override `.triggers()` to control which input ports trigger the step.
    By default all inputs trigger — the step runs whenever any input changes.
    To make some inputs "silent" (received but not triggering), return only
    the triggering subset from `triggers()`.

    Class attributes for memoization:
        ``_cache`` — ``'none'`` (default) or ``'by_hash'``. When
            ``'by_hash'``, ``invoke()`` hashes the Step's config and the
            incoming state, keys a class-level cache by
            ``(schema_version, hash)``, and reuses prior outputs on a
            hit. Requires ``_schema_version``.
        ``_schema_version`` — opaque string identifying the Step's
            output-relevant implementation version. Bumping invalidates
            all prior cache entries under the old version. Required when
            ``_cache == 'by_hash'``.
    """

    _cache: str = 'none'
    _schema_version: Optional[str] = None
    _by_hash_cache: Dict[Any, Any] = {}

    def triggers(self):
        """Return the subset of input ports that trigger this step.

        By default, all inputs trigger — the step runs whenever any
        input path is updated. Override to make some inputs "silent"
        (the step still receives them but they don't cause re-triggering).

        Returns:
            A dict of port names to type expressions, like inputs().
            Only these ports generate trigger edges in the step network.
        """
        return self.inputs()

    def scatter_port(self):
        """Return the name of this step's scatter input port, or ``None``.

        A port is a scatter port when its declared schema is a dict
        carrying ``_cardinality: 'per_match'``. At most one scatter port
        is permitted per Step; declaring more than one raises — multiple
        scatter axes must be materialised upstream by a ``Combine`` Step
        (see ``process_bigraph.plumbing``).

        The scatter annotation drives Nextflow channel semantics
        (queue-channel + per-item process invocation) and is available
        to any downstream interpreter that wants instance-reuse
        scattering at native runtime.
        """
        found = []
        for port_name, port_schema in self.inputs().items():
            if isinstance(port_schema, dict) and port_schema.get('_cardinality') == 'per_match':
                found.append(port_name)
        if len(found) > 1:
            raise ValueError(
                f"{type(self).__name__} declares _cardinality 'per_match' on "
                f"multiple input ports ({found!r}); at most one scatter axis "
                f"per Step is permitted. Upstream a Combine step to materialise "
                f"the product into a single match set."
            )
        return found[0] if found else None

    def invoke(self, state: Dict[str, Any], _: Optional[float] = None) -> SyncUpdate:
        """
        Run the step using the given state and return its update.

        When the Step declares a scatter port (``_cardinality: 'per_match'``
        on one of its inputs), this method loops ``update()`` once per
        match in the incoming match-set, reusing the single Step instance
        (Steps are stateless by design). Each per-output value is
        collected into a dict keyed by the match key, so the returned
        update has shape ``{port: {match_key: value}}``. Downstream
        consumers see the aggregated dict at the output wire's path.

        Non-scatter Steps take the fast path: a single ``update()`` call.

        When the Step declares ``_cache = 'by_hash'``, the (config, state)
        pair is hashed and the result looked up in a class-level cache
        keyed by ``(_schema_version, hash)``. Hits skip ``update()``
        entirely and return the prior output.
        """
        cache_key = self._cache_key(state) if self._cache == 'by_hash' else None
        if cache_key is not None:
            cached = type(self)._by_hash_cache.get(cache_key)
            if cached is not None:
                return SyncUpdate(cached)

        scatter_port = self.scatter_port()
        if scatter_port is None:
            result = self.update(state)
            if cache_key is not None:
                type(self)._by_hash_cache[cache_key] = result
            return SyncUpdate(result)

        match_set = state.get(scatter_port)
        if isinstance(match_set, dict):
            items = list(match_set.items())
        elif isinstance(match_set, list):
            items = list(enumerate(match_set))
        elif match_set is None:
            items = []
        else:
            raise TypeError(
                f"{type(self).__name__}: scatter port {scatter_port!r} expects "
                f"a dict or list match-set, got {type(match_set).__name__}"
            )

        per_port_entries: Dict[str, Dict[Any, Any]] = {}
        for key, item in items:
            per_state = dict(state)
            per_state[scatter_port] = item
            per_update = self.update(per_state) or {}
            for port, value in per_update.items():
                per_port_entries.setdefault(port, {})[key] = value

        # Wrap each port's per-match dict as an ``_add`` sentinel so the
        # update composes with a plain ``map[X]`` destination (no type
        # retyping required by the caller). ``_add`` on an existing key
        # overwrites, so re-runs with the same match set are idempotent.
        aggregated: Dict[str, Dict[str, Any]] = {
            port: {'_add': entries}
            for port, entries in per_port_entries.items()
        }

        if cache_key is not None:
            type(self)._by_hash_cache[cache_key] = aggregated
        return SyncUpdate(aggregated)

    def _cache_key(self, state: Dict[str, Any]) -> Tuple[str, str]:
        """Hash the Step's config + incoming state into a cache key.

        Raises if ``_schema_version`` is unset — the spec requires an
        explicit version when caching is enabled so old entries don't
        silently leak through implementation changes.
        """
        version = self._schema_version
        if version is None:
            raise ValueError(
                f"{type(self).__name__}: _cache = 'by_hash' requires "
                f"_schema_version to be set (a string identifying the "
                f"output-relevant implementation version). See the "
                f"Step docstring."
            )

        import hashlib
        import json
        payload = json.dumps(
            {'config': getattr(self, 'config', {}), 'state': state},
            sort_keys=True, default=repr,
        )
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        return (version, digest)

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

    def calculate_timestep(self, interval, state):
        return interval

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


def as_step(inputs, outputs, name=None, aliases=None):
    """
    Decorator: convert an `update_*` pure function into a Step subclass.

    - Does NOT register into any core.
    - Adds metadata so discover_packages can register nice aliases (e.g. "add").
    """
    def decorator(func):
        if not func.__name__.startswith("update_"):
            raise AssertionError("Function name must be of the form update_*")

        step_name = name or func.__name__[len("update_"):]
        step_aliases = list(aliases or [])
        # default alias: the function-derived name, e.g. update_add -> "add"
        if step_name not in step_aliases:
            step_aliases.insert(0, step_name)

        class FunctionStep(Step):
            def inputs(self): return inputs
            def outputs(self): return outputs
            def update(self, state): return func(state)

        FunctionStep.__name__ = f"{step_name}Step"

        # IMPORTANT: make this class look like it belongs to the user's module
        FunctionStep.__module__ = func.__module__

        # Discovery metadata
        FunctionStep.__pb_kind__ = "step"
        FunctionStep.__pb_aliases__ = step_aliases
        FunctionStep.__pb_wrapped__ = func

        return FunctionStep
    return decorator


def as_process(inputs, outputs, name=None, aliases=None):
    """
    Decorator: convert an `update_*` function into a Process subclass.

    - Does NOT register into any core.
    - Adds metadata so discover_packages can register nice aliases (e.g. "odeint").
    """
    def decorator(func):
        if not func.__name__.startswith("update_"):
            raise AssertionError("Function name must be of the form update_*")

        process_name = name or func.__name__[len("update_"):]
        process_aliases = list(aliases or [])
        if process_name not in process_aliases:
            process_aliases.insert(0, process_name)

        class FunctionProcess(Process):
            def inputs(self): return inputs
            def outputs(self): return outputs
            def update(self, state, interval): return func(state, interval)

        FunctionProcess.__name__ = f"{process_name}Process"

        # IMPORTANT: make this class look like it belongs to the user's module
        FunctionProcess.__module__ = func.__module__

        # Discovery metadata
        FunctionProcess.__pb_kind__ = "process"
        FunctionProcess.__pb_aliases__ = process_aliases
        FunctionProcess.__pb_wrapped__ = func

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
                    union_inputs = self.core.resolve(union_inputs, inputs)

            if attr_name.startswith('outputs_'):
                outputs_func = getattr(self, attr_name)
                if callable(outputs_func):
                    outputs = outputs_func()
                    union_outputs = self.core.resolve(union_outputs, outputs)

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
        'schema': 'schema',
        'state': 'tree[node]',
        'interface': {
            'inputs': 'schema',
            'outputs': 'schema'
        },
        'bridge': {
            'inputs': 'wires',
            'outputs': 'wires'
        },
        'global_time_precision': 'maybe[float]',
        'run_steps_on_init': 'boolean{false}',
        # Inner-layer parallelism: when True, run_steps fans out the
        # steps in each layer across a ThreadPoolExecutor. Threads
        # share memory and don't trigger the daemon-process restriction,
        # so this composes cleanly with outer multiprocess distribution
        # (Nextflow, joblib, ProcessPoolExecutor) — each sim still runs
        # in one Python process. The speedup depends on whether the
        # step hot path releases the GIL (numpy / scipy / numba / C
        # extensions: yes; pure-Python loops: no).
        'parallel_steps': 'boolean{false}',
        'parallel_workers': 'maybe[integer]',
        # When True, ``initialize`` trusts ``state`` as-is and skips the
        # per-process ``link_state`` + ``combine`` pass that normally
        # seeds initial values. Set this when loading from a saved
        # bundle (e.g. daughter_state from a workflow generation):
        # the saved state already contains everything processes would
        # have contributed, and re-combining their ``initial_state()``
        # overwrites correctly-divided values with mother-shaped ones.
        'skip_process_state': 'boolean{false}',
    }


    # ==============================
    # Initialization & Configuration
    # ==============================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the composite model from its config.

        This method:
        - Adds `global_time` to schema/state if missing
        - Generates the full schema/state tree
        - Finds all step/process instances
        - Resolves the schema bridge
        - Prepares the step execution network
        - Computes initial front (per-process timeline)

        Args:
            config: Optional override configuration (usually not needed).
        """

        # Get the initial schema schema from config.
        initial_schema = self.config.get('schema', {})

        # Ensure 'global_time' is explicitly declared in the schema.
        if 'global_time' not in initial_schema:
            initial_schema['global_time'] = 'float'

        # Get the initial state from config.
        initial_state = self.config.get('state', {})

        # Ensure the initial simulation state has a global_time initialized.
        if 'global_time' not in initial_state:
            initial_state['global_time'] = 0.0

        # Generate internal schema and state structures using the core engine.
        self.schema, self.state = self.core.realize(
            initial_schema,
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
        # ``skip_process_state`` short-circuits this when the caller supplied
        # state that already contains everything processes would have seeded
        # (e.g. a daughter_state bundle from a previous workflow generation).
        # Without the skip, ``combine`` overwrites correctly-divided values
        # with full mother-shaped ones from each process's ``initial_state()``.
        if not self.config.get('skip_process_state', False):
            edge_schema = {}
            edge_state = {}
            for path, edge in self.edge_paths.items():
                # Generate the initial state for this specific edge (process or step).
                initial_schema, initial_state = self.core.link_state(
                    edge,
                    path)

                # Merge the new edge state with the global state tree, checking for conflicts.
                try:
                    edge_schema, edge_state = self.core.combine(
                        edge_schema, edge_state,
                        initial_schema, initial_state)

                except Exception as e:
                    import sys as _sys
                    _sys.stderr.write(f'[INIT_COMBINE_FAIL] edge={path}\n')
                    _sys.stderr.write(f'[INIT_COMBINE_FAIL] new_schema={initial_schema}\n')
                    _sys.stderr.write(f'[INIT_COMBINE_FAIL] err={e}\n')
                    _sys.stderr.flush()
                    raise Exception(
                        f'initial state from edge does not match initial state from other edges:\n'
                        f'{path}\n{edge}\n{edge_state}\n'
                        f'{e}'
                    )

            # Apply the merged edge_state into the global state and update instance paths.
            if edge_state:
                self.schema, self.state = self.core.combine(
                    edge_schema, edge_state,
                    self.schema, self.state)

        # Wire the input/output schema for the Composite from the bridge config.
        self.process_schema = {
            port: self.core.wire_schema(
                self.schema,
                self.state,
                self.bridge.get(port, {}))
            for port in ['inputs', 'outputs']
        }

        # Set the global time precision used to round step time advances.
        self.global_time_precision = self.config.get('global_time_precision')

        # Inner-layer parallelism: optional ThreadPoolExecutor that
        # fans out the steps in each layer. See the parallel_steps
        # field in interface_schema for the rationale (threading vs.
        # multiprocessing for inner parallelism).
        self._parallel_steps = bool(self.config.get('parallel_steps', False))
        self._parallel_workers = self.config.get('parallel_workers')
        self._step_executor = None  # lazy: created on first run_steps need

        # Initialize a "front" dictionary tracking the next update time and update data per process.
        self.front = {
            path: empty_front(self.state['global_time'])
            for path in self.process_paths
        }

        # A buffer for updates to be emitted at the composite's output interface.
        self.bridge_updates: List[Any] = []

        # Timing accumulators for profiling (reset on each run() call)
        self.process_update_time: float = 0.0
        self.framework_time: float = 0.0

        # Precompile view/project operations for fast runtime access.
        self._compiled_links = {}
        self._build_view_project_cache()

        # Build the dependency network between steps and determine which steps should run first.
        self.build_step_network()

        # Optionally run all steps that are ready on the first cycle.
        if self._config.get('run_steps_on_init', False):
            self.run_steps(self.to_run)

    @classmethod
    def load(cls, path: str, core: Optional[Any] = None) -> "Composite":
        """
        Load a Composite from a saved JSON file.

        Args:
            path: Path to the saved schema file.
            core: Optional core context providing deserialization.

        Returns:
            A new Composite instance.
        """
        with open(path) as data:
            document = json.load(data)
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
        # Structural change incoming — drop the compiled-apply cache:
        # ``apply(dict)`` mutates schemas in place for ``_divide``
        # sentinels, so cached compiled functions may now reference
        # stale schema layouts.
        self.core.invalidate_compiled_apply()
        self.process_paths = find_instance_paths(state, 'process_bigraph.composite.Process')
        if hasattr(self, 'step_paths'):
            previous_step_paths = self.step_paths.keys()
            self.step_paths = find_instance_paths(state, 'process_bigraph.composite.Step')
            if previous_step_paths != self.step_paths.keys():
                self.build_step_network()
        else:
            self.step_paths = find_instance_paths(state, 'process_bigraph.composite.Step')

        all_paths = set(
            list(self.process_paths.keys()) +
            list(self.step_paths.keys()))

        front_paths = set(
            self.front.keys())

        for removed_key in front_paths.difference(all_paths):
            # do we want to do anything with these?
            removed_front = self.front.pop(removed_key)

    def _realize_merge_subtrees(self, paths: List[tuple]) -> None:
        """Realize only the subtrees touched by ``port_merges``.

        ``apply_updates`` collects every path that gained new schema
        info from a process's port defaults. The full-state realize
        that previously followed walked the whole tree on every tick
        with merges; on the brownian-particles test that meant ~2000
        full walks for 200s of sim. Incremental realize over just the
        affected paths is bounded by the merge count (~ports per tick).
        """
        for path in paths:
            try:
                sub_schema, sub_state = self.core.traverse(
                    self.schema, self.state, list(path))
            except Exception:
                continue
            if sub_schema is None:
                continue
            new_sub_schema, new_sub_state = self.core.realize(
                sub_schema, sub_state, path=tuple(path))
            if not path:
                self.state = new_sub_state
                continue
            parent_state = self.state
            for key in path[:-1]:
                parent_state = parent_state[key]
            parent_state[path[-1]] = new_sub_state

    def _realize_structural_subtrees(self, events: List[Any]) -> None:
        """Realize only the subtrees that structural events touched.

        After ``apply`` emits ``NodeAdded`` / ``Divided`` events, the new
        subtrees contain raw process declarations (config dicts) that
        need realizing into Process instances. The rest of the tree is
        already realized — re-walking it is wasted work that scales
        O(N) per division and so O(N²) over the simulation.

        This walks each affected subtree via ``core.traverse`` to fetch
        its (sub_schema, sub_state), realizes that pair, and splices the
        realized state back into ``self.state``. ``NodeRemoved`` is a
        no-op here — there's nothing to realize at a removed path.
        ``self.schema`` is left untouched: container schemas (Map,
        Tree) don't change shape on add/divide; dict containers were
        already mutated in place by ``apply``'s divide handler.
        """
        affected_paths: List[tuple] = []
        for event in events:
            if isinstance(event, NodeAdded):
                affected_paths.append(event.path + (event.key,))
            elif isinstance(event, Divided):
                for d_key in event.daughter_keys:
                    affected_paths.append(event.path + (d_key,))

        for path in affected_paths:
            try:
                sub_schema, sub_state = self.core.traverse(
                    self.schema, self.state, list(path))
            except Exception:
                continue
            if sub_schema is None:
                continue
            new_sub_schema, new_sub_state = self.core.realize(
                sub_schema, sub_state, path=tuple(path))
            # Splice the realized state back at its path. The parent
            # container is a mutable dict (Map keys are dict-keyed at
            # state level), so we rewrite the leaf entry. Schema dicts
            # may have been mutated by apply's divide handler already.
            parent_state = self.state
            for key in path[:-1]:
                parent_state = parent_state[key]
            parent_state[path[-1]] = new_sub_state

    def _apply_structural_events(self, events: List[Any]) -> None:
        """Update process/step indexes from apply-emitted structural events.

        Replaces the full-state ``find_instance_paths`` rescan when apply
        tells us exactly which subtrees changed. For each:
        - ``NodeAdded``: scan the new subtree only and merge any
          discovered Process/Step instances into the existing index.
        - ``NodeRemoved``: drop entries at-or-under the removed path.
        - ``Divided``: drop the mother's entries, scan each daughter
          subtree.

        The compiled-apply cache is invalidated unconditionally because
        ``apply(dict)``'s ``_divide`` branch mutates schemas in place.

        Step network rebuild is conditional on step_paths actually
        changing — value-only structural events that don't add/remove
        any Step instances skip the rebuild.
        """
        self.core.invalidate_compiled_apply()

        previous_step_paths = (set(self.step_paths.keys())
                               if hasattr(self, 'step_paths') else set())
        if not hasattr(self, 'step_paths'):
            self.step_paths = {}

        def _index_subtree(root_path: tuple, subtree_state: Any) -> None:
            """Walk subtree, merge Process/Step instances into indexes."""
            # find_instances takes a dict; if subtree is a dict, walk;
            # if it's a leaf or wrapped instance, check at this level.
            if not isinstance(subtree_state, dict):
                return
            instance = subtree_state.get('instance')
            from process_bigraph.composite import Process as _Proc
            from process_bigraph.composite import Step as _Step
            if isinstance(instance, _Proc):
                self.process_paths[root_path] = subtree_state
                return  # Don't recurse into a process's internal state
            if isinstance(instance, _Step):
                self.step_paths[root_path] = subtree_state
                return
            # Recurse into non-instance dict
            for key, child in subtree_state.items():
                if is_schema_key(key):
                    continue
                _index_subtree(root_path + (key,), child)

        def _drop_under(root_path: tuple) -> None:
            """Remove process/step/front entries at-or-under root_path."""
            for path in list(self.process_paths.keys()):
                if path == root_path or path[:len(root_path)] == root_path:
                    del self.process_paths[path]
            for path in list(self.step_paths.keys()):
                if path == root_path or path[:len(root_path)] == root_path:
                    del self.step_paths[path]
            for path in list(self.front.keys()):
                if path == root_path or path[:len(root_path)] == root_path:
                    del self.front[path]

        for event in events:
            if isinstance(event, NodeAdded):
                added_path = event.path + (event.key,)
                # Read fresh subtree from realized state — event.state is
                # pre-realize (no instances yet).
                subtree = get_path(self.state, list(added_path))
                if subtree is not None:
                    _index_subtree(added_path, subtree)
            elif isinstance(event, NodeRemoved):
                removed_path = event.path + (event.key,)
                _drop_under(removed_path)
            elif isinstance(event, Divided):
                mother_path = event.path + (event.mother_key,)
                _drop_under(mother_path)
                for d_key in event.daughter_keys:
                    d_path = event.path + (d_key,)
                    subtree = get_path(self.state, list(d_path))
                    if subtree is not None:
                        _index_subtree(d_path, subtree)

        # Step network only needs rebuild if step_paths changed
        current_step_paths = set(self.step_paths.keys())
        if current_step_paths != previous_step_paths:
            self.build_step_network()

        # Sync front buffer: drop entries for paths no longer present
        all_paths = set(self.process_paths.keys()) | current_step_paths
        for removed_key in set(self.front.keys()).difference(all_paths):
            self.front.pop(removed_key)

    def _build_view_project_cache(self) -> None:
        """Precompile view/project operations for each process path.

        Delegates to core.precompile_link() which pre-resolves wire paths
        and precomputes projection schemas so that runtime view/project
        calls bypass schema traversal entirely.

        Currently disabled for debugging — all views/projects go through
        the slow but correct core.view / core.project path.
        """
        self._compiled_links = {}

        for path in list(self.process_paths) + list(self.step_paths):
            compiled = self.core.precompile_link(
                self.schema, self.state, path)
            if compiled is not None:
                self._compiled_links[path] = compiled

    def _invalidate_caches(self) -> None:
        """Invalidate precompiled link caches, forcing rebuild on next use."""
        self._compiled_links = {}

    def _cached_view(self, path: Tuple[str, ...]) -> Dict[str, Any]:
        """View using precompiled link cache when available, falling back
        to the slow path otherwise.

        precompile_link resolves wire paths and precomputes projection
        schemas at build_step_network time so per-tick view calls
        bypass schema traversal. The fallback to core.view covers
        paths that weren't pre-compiled (e.g. freshly added processes
        between cache rebuilds)."""
        compiled = self._compiled_links.get(path)
        if compiled is not None:
            view_compiled = compiled.get('view')
            if view_compiled is not None:
                return self.core.view_fast(view_compiled, self.state)
        return self.core.view(self.schema, self.state, path)

    def _cached_project(self, path: Tuple[str, ...], view: Any,
                        ports_key: str = 'outputs') -> Any:
        """Project using precompiled link cache when available, falling
        back to the slow path otherwise. Only outputs are precompiled
        (inputs never get projected back into state); other ports fall
        through to the slow path."""
        if ports_key == 'outputs':
            compiled = self._compiled_links.get(path)
            if compiled is not None and compiled.get('project') is not None:
                return self.core.project_ports_fast(compiled['project'], view)
        return self.core.project(
            self.schema, self.state, path, view, ports_key)

    def merge(self, schema: Dict[str, Any], state: Dict[str, Any], path: Optional[List[str]] = None) -> None:
        """
        Merge a new schema/state subtree into the Composite.

        Args:
            schema: Schema dictionary to merge.
            state: State dictionary to merge.
            path: Path where merge should occur (default: root).
        """
        path = path or []
        # self.schema, self.state = self.core.merge(
        self.schema, self.state = self.core.combine(
            self.schema,
            self.state,
            schema,
            state)

        self.find_instance_paths(self.state)
        self._build_view_project_cache()

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
        self.schema = self.core.merge(self.schema, scoped_schema)

        # Re-generate state based on the new schema structure
        self.schema, self.state = self.core.generate(self.schema, self.state)

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
        self.state, merges = self.core.apply(self.schema, self.state, scoped_update)
        if merges:
            self.schema = self.core.resolve_merges(self.schema, merges)
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
        return self.core.serialize(self.schema, self.state)

    def serialize_schema(self) -> Dict[str, Any]:
        """
        Serialize the schema (schema) using the core serializer.

        Returns:
            A serialized schema representation.
        """
        return self.core.render(self.schema)
        # return self.core.serialize('schema', self.schema)

    def nextflow(self, options: Optional[Dict[str, Any]] = None) -> str:
        """Render this composite as a Nextflow DSL2 workflow document.

        Delegates to ``process_bigraph.nextflow.render_composite``. See
        that module for the full renderer description and the schema
        annotations (``_cardinality``, ``_nextflow``,
        ``_nextflow_directives``) it consumes.

        Args:
            options: optional renderer options; currently
                ``workflow_name`` (default ``'main'``) and ``header``
                (default ``nextflow.enable.dsl=2``) are recognized.

        Returns:
            The rendered Nextflow document as a string.
        """
        from process_bigraph.nextflow import render_composite
        return render_composite(self, options)

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
            document['schema'] = self.serialize_schema()

        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, filename)
        # outjson = json.dumps(
        #     document,
        #     default=encode_key)

        with open(filepath, 'w') as outfile:
            json.dump(document, outfile, separators=(',', ':'))
            print(f"Saved composite to {filepath}")

    def save_bundle(
            self,
            outdir: str = 'out/bundle',
            schema: bool = False,
            state: bool = False,
            min_bytes: int = 10_000,
    ) -> Dict[str, Any]:
        """Save the composite as a directory bundle with externalized arrays.

        Uses the dispatched ``bundle`` method to serialize state, which
        writes large arrays directly to Parquet files (skipping the
        intermediate Python-list representation). The remaining document
        (with ``$bundle_ref`` markers) is written to ``outdir/document.json``.

        Args:
            outdir: Bundle directory path (will be created).
            schema: Whether to include the serialized schema.
            state: Whether to include the serialized state.
            min_bytes: Minimum raw byte size to externalize an array.

        Returns:
            Summary dict with file counts and sizes.
        """
        from bigraph_schema.methods.bundle import BundleContext
        from process_bigraph.bundle import save_bundle

        if not schema and not state:
            schema = state = True

        arrays_dir = os.path.join(outdir, 'arrays')
        context = BundleContext(
            arrays_dir=arrays_dir, min_bytes=min_bytes)

        document = {}
        if state:
            # Use bundle dispatch — goes directly from numpy to Parquet,
            # no intermediate Python lists for large arrays.
            document['state'] = self.core.bundle(
                self.schema, self.state, context)
        if schema:
            document['schema'] = self.serialize_schema()

        # Write the document JSON (arrays already saved by bundle dispatch)
        os.makedirs(outdir, exist_ok=True)
        doc_path = os.path.join(outdir, 'document.json')

        # Sanity-check for ndarrays that bundle() should have externalized.
        import numpy as _np
        def _find_np(obj, path=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _find_np(v, f'{path}.{k}')
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _find_np(v, f'{path}[{i}]')
            elif isinstance(obj, _np.ndarray):
                raise TypeError(
                    f'ndarray at {path} was not bundled (shape={obj.shape}, '
                    f'dtype={obj.dtype}) — schema probably mismatches the '
                    f'actual runtime type')
        _find_np(document)
        with open(doc_path, 'w') as f:
            json.dump(document, f, separators=(',', ':'))

        # Summary
        doc_size = os.path.getsize(doc_path)
        array_sizes = {}
        if os.path.isdir(arrays_dir):
            for fname in os.listdir(arrays_dir):
                fpath = os.path.join(arrays_dir, fname)
                array_sizes[fname] = os.path.getsize(fpath)
        total_array_bytes = sum(array_sizes.values())

        summary = {
            'document_size': doc_size,
            'num_arrays': len(context.refs),
            'total_array_bytes': total_array_bytes,
            'total_bytes': doc_size + total_array_bytes,
            'array_files': array_sizes,
        }

        print(f"Saved bundle to {outdir}/")
        print(f"  document.json: {doc_size / 1e6:.1f} MB")
        print(f"  arrays: {len(context.refs)} files, "
              f"{total_array_bytes / 1e6:.1f} MB")
        print(f"  total: {(doc_size + total_array_bytes) / 1e6:.1f} MB")

        return summary

    @classmethod
    def load_bundle(
            cls,
            bundle_dir: str,
            core: Optional[Any] = None,
            **kwargs,
    ) -> 'Composite':
        """Load a composite from a bundle directory.

        Reads ``document.json``, resolves ``$bundle_ref`` markers by
        loading the corresponding Parquet files, and constructs a new
        Composite.

        Args:
            bundle_dir: Path to the bundle directory.
            core: The bigraph-schema core (with registered types).
            **kwargs: Extra arguments passed to the Composite constructor
                (e.g. ``parallel_steps=True``).

        Returns:
            A new Composite instance.
        """
        from process_bigraph.bundle import load_bundle

        document = load_bundle(bundle_dir, as_numpy=True)
        # The bundle's state is authoritative — it was saved post-init
        # so it already contains every contribution the per-process
        # ``link_state`` pass would re-add. Combining ``initial_state()``
        # on top here would clobber correctly-divided values (e.g. a
        # daughter's halved bulk array) with mother-shaped originals.
        #
        # ``run_steps_on_init`` fires derivers (mass listeners,
        # post-division-mass-listener) at startup, mirroring v1's
        # ``Engine.run_steps()`` call in ``__init__``. The bundle was
        # saved with mother's pre-divide derived state (the post-divide
        # cascade halts at the structural change to match v1's
        # DivisionDetected exception). Running derivers on load
        # refreshes those values from the daughter's halved bulk before
        # any process reads them.
        config = {
            'skip_process_state': True,
            'run_steps_on_init': True,
            **document,
            **kwargs,
        }
        return cls(config, core=core)


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
        # Fast path: composites without an output bridge (like vEcoli)
        # call this on every apply_updates. Skip the view_ports call
        # entirely when there's nothing to read.
        bridge_outputs = self.bridge.get('outputs') if self.bridge else None
        if not bridge_outputs:
            return None

        state = state or self.state

        bridge_view = self.core.view_ports(
            self.schema,
            state,
            (),
            self.interface()['outputs'],
            bridge_outputs)

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


    def initialize_steps(self, steps_to_run):
        # Identify downstream steps dependent on triggered ones
        steps_to_run = find_downstream(
            self.step_dependencies,
            self.node_dependencies,
            steps_to_run,
        )

        # Exclude steps already triggered in this timestep
        steps_to_run = {s for s in steps_to_run if s not in self.steps_run}
        # Mark all as triggered
        self.steps_run.update(steps_to_run)

        # Layer-walk cache: the SEQUENCE of layers produced by the trigger
        # graph walk is deterministic given the initial set of triggered
        # steps (and the static step network). Cache the sequence keyed
        # by frozenset(steps_to_run); replay it on subsequent ticks
        # without re-running determine_steps. Invalidated by
        # _last_apply_structural via expire_layer_walk_cache().
        cache = getattr(self, '_layer_walk_cache', None)
        if cache is None:
            cache = {}
            self._layer_walk_cache = cache
        cache_key = frozenset(steps_to_run)
        cached = cache.get(cache_key)
        if cached is not None:
            # Cache hit — replay the sequence. Set up an iterator and
            # mark recording=False so cycle_step_state pulls from it.
            self._layer_walk_replay = iter(cached)
            self._layer_walk_recording = None
            self.to_run = next(self._layer_walk_replay, [])
            return

        # Cache miss — record the walk as it runs.
        self._layer_walk_replay = None
        self._layer_walk_recording = []
        self._layer_walk_cache_key = cache_key

        # Initialize trigger fulfillment state and steps remaining
        self.reset_step_state(steps_to_run)

        # Compute the initial set of runnable steps
        self.to_run = self.cycle_step_state()


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
        self.trigger_state = build_trigger_state(self.node_dependencies, step_paths)

        # Track steps still waiting to be executed in this cycle
        self.steps_remaining: Set[Union[str, Tuple[str, ...]]] = set(step_paths)

    def cycle_step_state(self) -> List[Union[str, Tuple[str, ...]]]:
        """
        Evaluate the current trigger state and determine which steps can run.

        Uses the layer-walk replay cache when available, falling back to
        determine_steps when recording (cache miss) or when no cache is
        active (e.g. invoked from build_step_network).

        Returns:
            A list of step paths that are ready to be invoked in this cycle.
        """
        replay = getattr(self, '_layer_walk_replay', None)
        if replay is not None:
            return next(replay, [])

        to_run, self.steps_remaining, self.trigger_state = determine_steps(
            self.step_dependencies,
            self.steps_remaining,
            self.trigger_state
        )

        recording = getattr(self, '_layer_walk_recording', None)
        if recording is not None:
            recording.append(list(to_run))
            if not to_run:
                # Walk complete — commit the recording to the cache.
                cache = self._layer_walk_cache
                cache[self._layer_walk_cache_key] = recording
                self._layer_walk_recording = None

        return to_run

    def expire_layer_walk_cache(self) -> None:
        """Drop the layer-walk replay cache. Called when structural changes
        invalidate the step network."""
        self._layer_walk_cache = {}
        self._layer_walk_replay = None
        self._layer_walk_recording = None

    def _get_step_executor(self, n_needed: int):
        """Lazily build (or grow) the inner thread pool for layer parallelism.

        Sized to max(n_needed) we've seen so far, bounded by
        `parallel_workers` if set. Workers are daemon threads so the
        process can exit cleanly without explicit shutdown.
        """
        from concurrent.futures import ThreadPoolExecutor
        bound = self._parallel_workers
        target = n_needed if bound is None else min(n_needed, bound)
        # Recreate if existing pool is smaller than what we need.
        if self._step_executor is None or getattr(
                self._step_executor, '_max_workers', 0) < target:
            if self._step_executor is not None:
                self._step_executor.shutdown(wait=False)
            self._step_executor = ThreadPoolExecutor(
                max_workers=target,
                thread_name_prefix='pb-step')
        return self._step_executor

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

                # Add steps to the execution queue
                for step_path in step_paths:
                    if step_path is not None:
                        steps_to_run.append(step_path)

        self.initialize_steps(steps_to_run)
        self.run_steps(self.to_run)

    def run_steps(self, step_paths: List[Tuple[str, ...]]) -> None:
        """
        Execute a list of step processes, apply their updates, and handle cascading triggers.

        When `parallel_steps` is enabled and the layer has multiple steps,
        the per-step view + invoke phase runs concurrently across a
        ThreadPoolExecutor. apply_updates remains single-threaded so
        update reconciliation is atomic.

        Args:
            step_paths: A list of step path tuples to run.
        """
        if step_paths:
            n_steps = len(step_paths)
            use_pool = self._parallel_steps and n_steps > 1

            if use_pool:
                # Lazy-init the thread pool, sized to the largest layer
                # we've seen (bounded by parallel_workers if set).
                pool = self._get_step_executor(n_steps)
                # Each task: build view, invoke, return Defer.
                # _cached_view and process_update only read self.state /
                # self.schema and write per-step Defer wrappers — no
                # shared mutable state across threads. apply_updates
                # (called below, single-threaded) reconciles the Defers.
                def _run_one(step_path):
                    step = get_path(self.state, step_path)
                    state = self._cached_view(step_path)
                    instance = step.get('instance')
                    # _view_resolved emits only port-key entries; no schema
                    # keys to strip. perform_update sees the same dict.
                    if instance is not None and hasattr(instance, 'perform_update'):
                        if not instance.perform_update(state):
                            return None
                    return self.process_update(
                        step_path, step, state, -1.0, 'outputs',
                        already_clean=True)
                # list() forces all futures to resolve before continuing
                updates = [u for u in pool.map(_run_one, step_paths) if u is not None]
            else:
                updates = []
                for step_path in step_paths:
                    step = get_path(self.state, step_path)
                    # Use the precompiled view cache when possible. The
                    # direct core.view call falls back to view_ports which
                    # walks the wires every tick (~104 calls/sim_sec for
                    # vEcoli).
                    state = self._cached_view(step_path)

                    instance = step.get('instance')
                    # _view_resolved emits only port-key entries; no schema
                    # keys to strip. perform_update sees the same dict.
                    if instance is not None and hasattr(instance, 'perform_update'):
                        if not instance.perform_update(state):
                            continue

                    # Steps are always invoked with interval = -1.0
                    step_update = self.process_update(
                        step_path, step, state, -1.0, 'outputs',
                        already_clean=True)

                    updates.append(step_update)

            update_paths = self.apply_updates(updates)
            self.expire_process_paths(update_paths)

            # Opt-in halt: caller (e.g. EcoliSim) sets
            # ``_halt_after_structural`` to abort the post-divide step
            # cascade so daughters save with mother's pre-divide derived
            # state — mirroring v1 vivarium's ``DivisionDetected``
            # behavior. Default behavior continues cascading for
            # dynamic-structure tests that rely on spawn chains.
            if (getattr(self, '_halt_after_structural', False)
                    and getattr(self, '_last_apply_structural', False)):
                return

            to_run = self.cycle_step_state()

            if to_run:
                self.run_steps(to_run)


    # ====================
    # Simulation Execution
    # ====================

    def run(self, interval: float, force_complete: bool = False) -> None:
        """
        Advance simulation by running processes until a target time is reached.

        The method loops through all registered processes and executes their updates
        incrementally based on their configured interval. Updates are applied and
        steps are triggered accordingly.

        After completion, ``self.process_update_time`` holds the cumulative time
        spent inside process ``invoke()`` calls, and ``self.framework_time`` holds
        the time spent in framework operations (view, project, apply, realize).

        Args:
            interval: Time interval to simulate.
            force_complete: If True, forces all processes to reach the end time.
        """
        self.process_update_time = 0.0
        self.framework_time = 0.0
        run_start = _time.monotonic()

        end_time = self.state['global_time'] + interval

        # Run any steps that are ready (from init or previous triggers)
        if self.to_run:
            self.run_steps(self.to_run)
            self.to_run = []

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

                fw_start = _time.monotonic()
                update_paths = self.apply_updates(updates)
                update_paths.append(('global_time',)) # updated global time can trigger steps
                self.expire_process_paths(update_paths)
                self.steps_run = set()  # Reset for new timestep
                # Opt-in halt after structural change. EcoliSim sets
                # ``_halt_after_structural=True`` to mirror v1's halt
                # via DivisionDetected. Dynamic-structure tests leave
                # the flag unset for natural cascading.
                if (getattr(self, '_halt_after_structural', False)
                        and getattr(self, '_last_apply_structural', False)):
                    self.framework_time += _time.monotonic() - fw_start
                    return
                self.trigger_steps(update_paths)
                self.framework_time += _time.monotonic() - fw_start

            else:
                # All remaining process events are beyond end_time
                self.state['global_time'] = end_time

            if force_complete and self.state['global_time'] == end_time:
                force_complete = False

        total = _time.monotonic() - run_start
        # Framework time = total minus process update time
        # (the process_update_time was accumulated in process_update)
        self.framework_time = total - self.process_update_time

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
            path: The path to the process in the state/schema tree.
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
                state = self._cached_view(path)
                state_interval = process['interval']
                process_interval = process['instance'].calculate_timestep(state_interval, state)
                process['interval'] = process_interval

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
                # state came from _cached_view (or future_front whose source
                # is also _cached_view) — _view_resolved only emits port-key
                # entries, so the dict has no schema keys to strip.
                update = self.process_update(
                    path, process, state, process_interval,
                    already_clean=True)

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
            ports_key: str = 'outputs',
            already_clean: bool = False,
    ) -> Defer:
        """
        Start generating a process's update and wrap it in a deferred transformation.

        This is similar to invoking a process directly, but it delays transformation
        into absolute state terms until `.get()` is called on the returned `Defer` object.

        Args:
            path: The path to the process in the state/schema tree.
            process: The dictionary representing the process instance (must include 'instance').
            states: The current state values at the process’s ports.
            interval: The time interval to simulate.
            ports_key: Which port ('inputs' or 'outputs') to use when projecting the update.
            already_clean: Set True when ``states`` was already passed through
                ``strip_schema_keys`` by the caller (e.g. ``run_steps``)
                so we skip the redundant strip.

        Returns:
            A `Defer` object that, when resolved, transforms the update to absolute paths.
        """
        # Strip schema-specific metadata from the state (once, unless caller did)
        clean_state = states if already_clean else strip_schema_keys(states)

        # Invoke the process and retrieve a wrapped SyncUpdate object
        t0 = _time.monotonic()
        update = process['instance'].invoke(clean_state, interval)
        self.process_update_time += _time.monotonic() - t0
        # This nested function projects the update into the global state at the given path
        def defer_project(update_results: Any, args: Tuple[Any, Any, Union[str, Tuple[str, ...]]]) -> Any:
            schema, state, process_path = args

            if not isinstance(update_results, list):
                update_results = [update_results]

            return [self._cached_project(
                process_path,
                update_result,
                ports_key) for update_result in update_results]

        # Return a deferred object that will project the update when requested
        return Defer(update, defer_project, (self.schema, self.state, path))

    @staticmethod
    def _has_structural_keys(state: Any) -> bool:
        """Check if a state dict contains keys that signal structural changes.

        Structural changes (_add, _remove, _type, _divide) require
        re-running realize() and find_instance_paths(). Plain value
        updates do not. _divide replaces a mother with two daughters
        — the daughter agents have new process instances that the
        framework needs to discover.
        """
        if not isinstance(state, dict):
            return False
        for key, value in state.items():
            if key in ('_add', '_remove', '_divide'):
                return True
            if key == '_type':
                return True
            if isinstance(value, dict) and Composite._has_structural_keys(value):
                return True
        return False

    def _update_touches_process_path(
            self, update_paths: List[Union[str, Tuple[str, ...]]]) -> bool:
        """Return True if any update path goes into an existing process.

        Mirrors the overlap check in `expire_process_paths` — process
        paths and update paths are already on hand from `_walk_update`,
        so detecting "this update changed a process's metadata" doesn't
        need a second walk.
        """
        if not self.process_paths:
            return False
        process_path_tuples = [tuple(p) for p in self.process_paths]
        process_roots = {p[0] for p in process_path_tuples if p}
        for update_path in update_paths:
            if not update_path or update_path[0] not in process_roots:
                continue
            update_tuple = tuple(update_path)
            for proc_path in process_path_tuples:
                plen = len(proc_path)
                if len(update_tuple) >= plen and update_tuple[:plen] == proc_path:
                    return True
        return False

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
        had_structural_changes = False
        had_structural_sentinels = False  # _add/_remove/_divide detected

        # Phase 1: Resolve all deferred updates and collect them.
        # Path discovery + structural detection happen during reconcile
        # (Phase 2) via a ReconcileSummary sink — no separate walk pass.
        resolved_updates = []
        for defer in updates:
            series = defer.get()
            if series is None:
                continue
            if not isinstance(series, list):
                series = [series]

            for update_schema, update_state in series:
                # read_bridge fast-paths to None when no bridge outputs
                # are configured (vEcoli's case) — no walk happens.
                bridge_update = self.read_bridge(update_state)
                if bridge_update:
                    self.bridge_updates.append(bridge_update)

                resolved_updates.append((update_schema, update_state))

        # Phase 2: Reconcile all updates into a single combined update,
        # then apply once. This ensures atomic application of related
        # changes (e.g. unique molecule adds across linked types).
        if resolved_updates:
            # Combined-schema cache: most layers combine the same set of
            # update schemas every tick. Key by the tuple of input schema
            # ids; verify with a witness check (id may be reused after
            # GC, so we keep references to the originals alive).
            schema_ids = tuple(id(us) for us, _ in resolved_updates)
            cache = getattr(self, '_combined_schema_cache', None)
            if cache is None:
                cache = {}
                self._combined_schema_cache = cache
            entry = cache.get(schema_ids)
            combined_schema = None
            if entry is not None:
                witnesses, cached_schema = entry
                if all(w is r[0] for w, r in zip(witnesses, resolved_updates)):
                    combined_schema = cached_schema

            if combined_schema is None:
                # Cache miss — resolve once, then store with witnesses
                combined_schema = resolved_updates[0][0]
                for update_schema, _ in resolved_updates[1:]:
                    combined_schema = self.core.resolve(combined_schema, update_schema)
                cache[schema_ids] = (
                    tuple(us for us, _ in resolved_updates),
                    combined_schema,
                )

            # Reconcile all update states using the combined schema.
            # Install a summary sink so reconcile populates leaf paths
            # and the structural-sentinel flag during its existing walk
            # — eliminates the redundant per-defer _walk_update pass.
            from bigraph_schema.methods.events import (
                ReconcileSummary, install_reconcile_sink, uninstall_reconcile_sink)
            summary = ReconcileSummary(paths=[])
            prev_sink = install_reconcile_sink(summary)
            try:
                all_states = [state for _, state in resolved_updates]
                combined_update = self.core.reconcile(combined_schema, all_states)
            finally:
                uninstall_reconcile_sink(prev_sink)
            update_paths = summary.paths
            if summary.has_structural:
                had_structural_sentinels = True

            if combined_update:
                # NOTE: previously this path also forced
                # ``had_structural_sentinels = True`` whenever an
                # update touched a process subtree, because the older
                # code applied with the stripped ``combined_schema``
                # and would clobber link state. With ``promote``
                # supplying typed nodes (Array, Map, ProcessLink, ...)
                # at each apply path, value-only updates into a
                # process's path apply correctly without a full
                # realize. Only true sentinels (_add/_remove/_divide,
                # detected by the reconcile sink above) flip the
                # structural flag.
                pass

                # Apply needs the live state schema so dispatch sees
                # the underlying types (Array, Map, ProcessLink, etc.)
                # at each path. ``combined_schema`` alone reflects the
                # update's wire shape (often dict-of-int-keys for
                # array-cell projections) and loses the typed
                # information present in ``self.schema``. Without
                # promoting, an update of shape ``{i: {j: delta}}``
                # would land at apply with a dict schema but an
                # ndarray state — bypassing the Array-aware overload.
                # ``promote`` walks only the paths combined_schema
                # touches (vs ``resolve`` which walks all of
                # self.schema each tick); structural sentinels still
                # use the full resolve so divide/add/remove see the
                # mother's existing schema.
                if had_structural_sentinels:
                    apply_schema = self.core.resolve(
                        self.schema, combined_schema)
                else:
                    apply_schema = self.core.promote(
                        self.schema, combined_schema)
                # Collect structural events (NodeAdded/NodeRemoved/
                # Divided) emitted by sentinel handlers during apply.
                # On structural ticks, used to update process_paths/
                # step_paths indexes incrementally instead of re-scanning
                # the full state via find_instance_paths.
                structural_events = [] if had_structural_sentinels else None
                self.state, merges = self.core.apply(
                    apply_schema,
                    self.state,
                    combined_update,
                    update_has_structural=had_structural_sentinels,
                    events=structural_events)
                # For structural sentinels, apply may have mutated
                # apply_schema in place (e.g. _divide pops/inserts
                # keys in a dict schema). Propagate back to self.schema
                # so downstream realize sees the split.
                if had_structural_sentinels:
                    self.schema = apply_schema

                if merges:
                    had_structural_changes = True
                    self.schema = self.core.resolve_merges(
                        self.schema,
                        merges)
                    # Track exactly which paths gained new schema info so
                    # we can realize only those subtrees instead of the
                    # whole tree (full realize was the dominant per-tick
                    # framework cost on tests with port_merges every tick,
                    # e.g. brownian_particles' 2000 ticks each triggering
                    # a full-state walk).
                    if not hasattr(self, '_merge_paths_pending'):
                        self._merge_paths_pending = []
                    for entry in merges:
                        # merges are (path, subschema, link_path); we
                        # only need the path prefix.
                        if entry and entry[0] is not None:
                            self._merge_paths_pending.append(tuple(entry[0]))

        # Schema merges (from process outputs introducing new fields)
        # require realize to fill defaults but do NOT add/remove
        # processes — no need to rebuild the step network.
        # Only actual structural sentinels (_add/_remove/_divide)
        # require find_instance_paths + build_step_network.
        if had_structural_changes:
            merge_paths = getattr(self, '_merge_paths_pending', None)
            if merge_paths:
                # Realize only the affected subtrees. Each merge path is
                # already covered by the incremental walk; deduplicate
                # under shortest-prefix containment so we don't visit
                # ancestor and descendant separately.
                unique = []
                for p in sorted(set(merge_paths), key=len):
                    if any(p[:len(u)] == u for u in unique):
                        continue
                    unique.append(p)
                self._realize_merge_subtrees(unique)
                self._merge_paths_pending = []
            else:
                # Fallback: no path info available — full realize.
                self.schema, self.state = self.core.realize(
                    self.schema, self.state)
            self._build_view_project_cache()

        if had_structural_sentinels:
            # Real structural change: processes may have been
            # added/removed/replaced. Realize the affected subtrees
            # only (instantiating new daughter processes) instead of
            # re-walking the entire schema — full realize scales O(N)
            # per division and so O(N²) over a sim with N divisions.
            if structural_events:
                self._realize_structural_subtrees(structural_events)
                self._apply_structural_events(structural_events)
            else:
                # Fallback: had_structural_sentinels was set but no
                # events emitted (e.g. _update_touches_process_path
                # detected an in-process update with no _add/_remove/
                # _divide sentinel). Full realize + rescan is correct
                # here — we don't know which subtrees changed.
                self.schema, self.state = self.core.realize(
                    self.schema, self.state)
                self.find_instance_paths(self.state)
            self._build_view_project_cache()
            if hasattr(self, 'expire_layer_walk_cache'):
                self.expire_layer_walk_cache()

        # Expose structural-change flag so the run loop can skip the
        # redundant expire_process_paths walk on value-only ticks.
        self._last_apply_structural = had_structural_sentinels
        return update_paths

    def expire_process_paths(self, update_paths: List[Union[str, Tuple[str, ...]]]) -> None:
        """
        Invalidate and refresh process paths if affected by recent updates.

        Now a no-op: ``apply_updates`` already maintains ``process_paths``,
        ``step_paths``, and the view-project cache via either
        ``_apply_structural_events`` (incremental, when structural
        sentinels emit events) or its fallback ``find_instance_paths``
        + ``_build_view_project_cache`` (when the sentinel was detected
        but no events surfaced). On value-only ticks no structural
        change occurred, so there is nothing to re-discover. Repeating
        the work here was the dominant framework cost on
        particle/division-heavy tests — ``find_instance_paths`` is a
        full-state walk that scaled with population, turning each
        division into O(N) work and the simulation into O(N²) overall.

        Args:
            update_paths: A list of hierarchical paths that were
                modified. Retained for caller compatibility; unused.
        """
        return


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
        project_schema, project_state = self.core.project_ports(
            self.interface()['inputs'],
            self.bridge.get('inputs', {}),
            [],
            state)
        self.merge({}, project_state)

        # first_update = self.read_bridge(
        #     self.state)
        # self.bridge_updates = [first_update]

        self.bridge_updates = []

        self.run(interval)

        return self.bridge_updates


def encode_key(o):
        if isinstance(o, np.ndarray):
            o.tolist()

        elif isinstance(o, dict):
            return {
                str(k): encode_key(v)
                for k, v in o.items()}

        else:
            return o
