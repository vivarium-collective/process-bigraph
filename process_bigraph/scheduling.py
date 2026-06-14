"""
scheduling.py — Step-scheduling graph for composite execution.

Pure functions that compute, from step metadata and wire paths, which steps
fire, in what layered order, and how their dependency network resolves. They
operate only on dicts / paths / step-metadata plus bigraph-schema's
``resolve_path`` — no Composite/Process/Step class coupling — so they live
apart from the engine in composite.py, which imports them back and re-exports
them for back-compat (external callers use e.g.
``from process_bigraph.composite import empty_front`` or
``from process_bigraph import wire_step_layers``).
"""
from typing import (
    Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union)

from bigraph_schema import resolve_path


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

    # Precompute each step's direct dependents (= the set of steps
    # that consume something this step writes). ``find_downstream``
    # was reconstructing this every tick by walking
    # output_paths × explode_path × nodes lookup; doing it once at
    # graph-build time makes ``find_downstream`` a plain BFS over a
    # cached adjacency list.
    for step_key, ancestor in ancestors.items():
        direct = set()
        for output_path in ancestor['output_paths'] or []:
            for subpath in explode_path(output_path):
                node = nodes.get(subpath)
                if node is not None:
                    direct.update(node['after'])
        ancestor['_direct_dependents'] = frozenset(direct)

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
    Given a set of updated steps, identify all downstream steps that
    depend on them — directly or transitively.

    BFS over the cached ``_direct_dependents`` adjacency that
    ``build_step_network`` precomputes per step. ``nodes`` is no
    longer consulted on the hot path (kept in the signature for
    backward compatibility with callers that still pass it).
    """
    downstream = set(upstream)
    queue = list(upstream)
    while queue:
        step_path = queue.pop()
        meta = steps.get(step_path)
        if meta is None:
            continue
        for dep in meta.get('_direct_dependents', ()):
            if dep not in downstream:
                downstream.add(dep)
                queue.append(dep)
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
