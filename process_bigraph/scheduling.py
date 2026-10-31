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
import warnings
from typing import (
    Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union)

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
                # A trigger only fires a step if it matches one of the
                # step's input ports, so wire the root-trigger port itself
                # when the step does not already carry it.
                if root_trigger not in step['inputs']:
                    step['inputs'][root_trigger] = [root_trigger]
                if root_trigger not in step['_inputs']:
                    step['_inputs'][root_trigger] = 'float'

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
                # Normalize `..` ascents. A wire is relative to its edge's
                # parent, so a nested step reaching a shared store writes
                # `['..', 'results']`. Runtime views resolve that, but the
                # dependency graph compared the literal path — which never
                # matches the producer's output path, so the edge was
                # silently dropped and the consumer ran unordered.
                leaves.append(resolve_path(path + tuple(value)))

    return leaves


def _is_structural_output(port_type) -> bool:
    """True when an output port's type is a structural (``_add``/``_remove``)
    restructure of a subtree rather than a data write to leaves.

    A ``node`` port, or a ``map``/``array``/``tree`` of ``node``, carries
    structural updates (``{'_add': ..., '_remove': ...}``) — e.g. the
    division shape ``{'agents': {'_add': [...], '_remove': [...]}}``. The step
    replaces whole members; it never writes the internal leaves of whatever
    already lives under that port. The dependency graph must not treat it as a
    producer of those leaves (process-bigraph #248).
    """
    if port_type == 'node':
        return True
    if isinstance(port_type, dict):
        if port_type.get('_type') == 'node':
            return True
        if (port_type.get('_type') in ('map', 'array', 'tree')
                and port_type.get('_value') == 'node'):
            return True
    return False


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
    # path (tuple) -> set of step_keys that write it via a STRUCTURAL (node /
    # map-of-node) output port. Such a write restructures the container at that
    # path; it does not produce the data leaves beneath it (#248).
    structural_outputs = {}
    # step_key -> frozenset of the step's FULL inputs() leaves. Used by the
    # container-propagation pass to apply A2's read-modify-write rule: a step
    # that READS a container it would otherwise propagate down to child leaves
    # is an accumulator, not a fresh producer (A3).
    full_inputs_by_step = {}
    # leaf path (tuple) -> set of step_keys that READ-MODIFY-WRITE it in place
    # (A2 suppressed them as producers of their own accessed store). The
    # pure-reader ordering pass below uses this to order a reader after the
    # accumulators without ordering the accumulators among themselves.
    inout_writers = {}

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

        # Compute output paths once per step. A `_dep_outputs` override is
        # trusted verbatim; otherwise narrow the wired outputs to the step's
        # declared outputs() ports (the output-side analogue of the triggers()
        # input narrowing above) — a port wired as an output but not declared
        # in outputs() is inout-inflation, not a real write. `_layer_*`
        # framework ports are always kept.
        dep_outputs = step.get('_dep_outputs')
        if dep_outputs is not None:
            out_wiring = dep_outputs
        else:
            wiring = step.get('outputs', {})
            # Distinguish an ABSENT outputs schema (key missing -> `None`) from
            # an explicitly EMPTY one (`{}`). Absent = can't narrow safely, fall
            # back to the raw wiring; empty = the step declares no outputs, so
            # narrow to nothing (only framework ports survive).
            out_schema_raw = (schema.get('outputs')
                              if isinstance(schema, dict) else None)
            out_ports = (set(out_schema_raw.keys())
                         if out_schema_raw is not None else None)
            if isinstance(wiring, dict) and out_ports is not None:
                # Framework-injected output ports (wire_step_layers' `_flow_out`
                # layer token, and any `_layer*` token) are not part of the
                # step's outputs() interface but carry real ordering edges, so
                # they must survive narrowing. They follow the underscore-
                # prefixed framework-port convention; genuine data ports never do.
                out_wiring = {k: v for k, v in wiring.items()
                              if k in out_ports
                              or str(k).startswith('_flow')
                              or str(k).startswith('_layer')}
            else:
                out_wiring = wiring
        if ancestors[step_key]['output_paths'] is None:
            ancestors[step_key]['output_paths'] = find_leaves(
                out_wiring, path=step_key[:-1])

        # Record which of this step's output paths are STRUCTURAL (a node /
        # map-of-node port). These restructure the container, so later passes
        # must not propagate them as producers of the data leaves beneath them.
        out_schema = schema.get('outputs', {}) if isinstance(schema, dict) else {}
        if isinstance(out_wiring, dict):
            for port, port_type in out_schema.items():
                if port in out_wiring and _is_structural_output(port_type):
                    for p in find_leaves({port: out_wiring[port]}, path=step_key[:-1]):
                        structural_outputs.setdefault(tuple(p), set()).add(step_key)

        # Assign the priority
        if ancestors[step_key]['priority'] is None:
            ancestors[step_key]['priority'] = step.get('priority', 0.0)

        input_paths = ancestors[step_key]['input_paths'] or []
        output_paths = ancestors[step_key]['output_paths'] or []

        # Full input leaves (the complete inputs() wiring, independent of any
        # triggers() narrowing of input_paths) — the read side of the
        # read-modify-write test below. input_paths may be a trigger-narrowed
        # SUBSET of these, so testing inout against the full set is what makes
        # the skip hold for triggered steps.
        full_input_leaves = {
            tuple(p) for p in find_leaves(step.get('inputs', {}), path=step_key[:-1])}
        full_inputs_by_step[step_key] = full_input_leaves

        # Track which steps consume/produce each path
        for input_path in input_paths:
            path = tuple(input_path)
            nodes.setdefault(path, {'before': set(), 'after': set()})
            nodes[path]['after'].add(step_key)

        for output_path in output_paths:
            exploded_path = explode_path(output_path)[1:]
            for explode in exploded_path:
                # Read-modify-write: a store this step both reads and writes is
                # never a cross-step producer (it mutates its own accessed store
                # in place, like a shared accumulator). Test against the FULL
                # inputs() leaves, not the trigger-narrowed input_paths.
                if tuple(explode) in full_input_leaves or explode in input_paths:
                    inout_writers.setdefault(tuple(explode), set()).add(step_key)
                    continue
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
        consumers = deps['after']
        # Check if any output is an ancestor of this input path
        for out_path, producers in all_output_paths.items():
            if (len(node_path) > len(out_path)
                    and node_path[:len(out_path)] == out_path):
                # Structural producers (a node / map-of-node _add/_remove at
                # out_path) restructure the container; they do NOT write the
                # data leaves beneath it, so they must not become producers of
                # this descendant leaf. Dropping them is the exact-leaf case
                # that the container-restructure-peer skip below (#243) did not
                # cover: there the consumer also writes the container, here the
                # consumer reads/writes a deeper leaf the structural step never
                # touches (process-bigraph #248 — the Division vs Allocator
                # false cycle). The structural step still orders against steps
                # that read/write the container membership itself (an exact or
                # shallower match, handled in the first pass).
                data_producers = producers - structural_outputs.get(out_path, set())
                if not data_producers:
                    continue
                # Container-restructure peers: when every consumer of this
                # member path also WRITES the same ancestor container
                # (``outputs: {..: ['..']}`` — the map _add/_remove / division
                # shape), the container write is a shared structural edit, not
                # a producer→consumer ordering. Each peer reads only its own
                # member and never consumes what the others write, so ordering
                # them by the container would be a false cycle (process-bigraph
                # 1.9.0 regression on the v2ecoli division shape). An exact or
                # deeper data match is handled by the first pass above (which
                # leaves ``before`` non-empty, so we never reach here for it),
                # so skipping only affects pure container-ancestor edges.
                if consumers and consumers <= data_producers:
                    continue
                # A3 (A2 extended to this pass): an inout producer — one whose
                # full inputs() include the container ``out_path`` it would
                # propagate down — reads and writes that container in place, so
                # it is a read-modify-write accumulator, not a fresh producer of
                # the child leaves beneath it. A2 already suppressed its DIRECT
                # producer edge on the container, but ``output_paths`` still
                # lists the container, so without this it re-leaks here as a
                # false producer of every child leaf a peer reads (the v2ecoli
                # `division` ↔ `unique_update_*` false cycle). A genuine,
                # non-inout whole-container writer is unaffected and still orders
                # its child readers. This narrowing runs AFTER the peer-skip
                # above so it can only SHRINK the recorded producer set, never
                # flip the `consumers <= data_producers` test and add an edge —
                # preserving the edge-removal-only invariant.
                data_producers = {
                    producer for producer in data_producers
                    if out_path not in full_inputs_by_step.get(producer, ())}
                if not data_producers:
                    continue
                deps['before'].update(data_producers)

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

    # Pure-reader ordering (the A2/A3 counterpart). A2/A3 drop an inout
    # accumulator (a step that read-modify-writes a store) from the ``before``
    # producer set of that store's child leaves, which correctly breaks the
    # accumulator-PEER cycle (two accumulators of one store do not produce what
    # each other consumes — v2ecoli ``division`` ↔ ``unique_update_*``). But a
    # PURE READER of that store — a step that reads the settled value and never
    # writes it (a contract/listener bridge reading ``dry_mass``, a tank reading
    # an exchange) — genuinely needs the accumulators to have run, and the
    # dropped ``before`` edge left it unordered (process-bigraph #259 regression:
    # ``contract_out`` read ``dry_mass`` mid-accumulation). We cannot restore the
    # edge through ``before`` — that is a per-PATH set shared by every consumer,
    # so it would also re-order the accumulators against each other and rebuild
    # the very cycle A2/A3 removed. Instead record a per-STEP predecessor edge
    # accumulator → pure-reader: it orders the reader after the accumulators
    # WITHOUT ordering the accumulators among themselves.
    #
    # A pure reader of leaf ``L``: reads ``L`` (exactly, or via an ancestor
    # container of ``L``) and is not itself any writer of ``L`` (neither a
    # recorded producer nor an inout writer). ``determine_steps`` and
    # ``find_step_cycle`` both honour ``_extra_predecessors``.
    writers_of_leaf = {}
    for p, node in nodes.items():
        if node['before']:
            writers_of_leaf.setdefault(p, set()).update(node['before'])
    for p, w in inout_writers.items():
        writers_of_leaf.setdefault(p, set()).update(w)
    reads_by_step = {
        s: {tuple(x) for x in (ancestors[s]['input_paths'] or [])}
        for s in ancestors}

    # Base producer→consumer adjacency (the ordering edges already implied by
    # ``before``/``after``), used to add a pure-reader edge ONLY where it stays
    # acyclic: if the reader can already reach the accumulator, the two are in a
    # genuine zero-time feedback loop (the reader also feeds the accumulator) and
    # a hard ``accumulator → reader`` edge would close a cycle. Those are left to
    # the scheduler's existing priority-based cycle resolution; we only hard-order
    # the acyclic readers that A2/A3 wrongly left dangling.
    base_adj = {s: set() for s in ancestors}
    for node in nodes.values():
        for producer in (node.get('before') or ()):
            if producer not in base_adj:
                continue
            for consumer in (node.get('after') or ()):
                if consumer != producer:
                    base_adj[producer].add(consumer)

    def _reaches(src, dst):
        if src == dst:
            return True
        seen = {src}
        stack = [src]
        while stack:
            cur = stack.pop()
            for nxt in base_adj.get(cur, ()):
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    extra_predecessors = {s: set() for s in ancestors}
    for leaf, accumulators in inout_writers.items():
        leaf_writers = writers_of_leaf.get(leaf, set())
        for step_key, reads in reads_by_step.items():
            if step_key in accumulators or step_key in leaf_writers:
                continue  # a writer of this leaf, not a pure reader
            reads_leaf = leaf in reads or any(
                len(i) <= len(leaf) and tuple(leaf[:len(i)]) == i for i in reads)
            if not reads_leaf:
                continue
            for acc in accumulators:
                if acc == step_key:
                    continue
                # Skip the edge if the reader already reaches the accumulator
                # (acc → reader would close a feedback cycle); priority resolves
                # those. Otherwise hard-order reader after acc.
                if not _reaches(step_key, acc):
                    extra_predecessors[step_key].add(acc)

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
        ancestor['_extra_predecessors'] = frozenset(extra_predecessors[step_key])
        ancestor['_direct_dependents'] = frozenset(direct)
    # A pure reader is a dependent of each accumulator that precedes it, so the
    # hot-path downstream BFS (``find_downstream``) re-triggers it when the
    # accumulator updates.
    for reader, preds in extra_predecessors.items():
        for pred in preds:
            if pred in ancestors:
                ancestors[pred]['_direct_dependents'] = frozenset(
                    ancestors[pred]['_direct_dependents'] | {reader})

    return ancestors, nodes


def find_step_cycle(ancestors, nodes):
    """Find a dependency cycle among zero-time steps, if one exists.

    The process-bigraph paper (Supplement 1, 3.8.4) guarantees that the
    step network is a DAG: "Cycles are disallowed, ensuring deterministic
    zero-time DAG evaluation." A cycle (step A consumes what step B produces
    AND step B consumes what step A produces) has no topological order, so
    ``determine_steps`` resolves it by firing the highest-priority runnable
    step first. That resolution is DETERMINISTIC as long as the cyclic steps
    carry **distinct priorities** — priority alone fixes their order. It
    becomes non-deterministic only when two steps *in the same cycle* share a
    priority: then the tie falls through to set-iteration order, i.e. the
    Python hash seed (issue #233), and the same document gives different
    results in different processes.

    So this function is priority-aware: a cycle whose members all have
    distinct priorities is deterministically orderable and is ALLOWED (with a
    warning); only a cycle containing a priority tie is reported as fatal.
    This matches the behaviour real composites (e.g. v2ecoli, whose steps
    carry distinct priorities and whose feedback among listener/deriver steps
    is intentional) relied on before the #233 hard check — verified by
    bit-identical state across differing hash seeds.

    The edges here are exactly the ordering constraints ``determine_steps``
    walks: for each wired path, every producer (a step in the path's
    ``before`` set) must run before every consumer (a step in ``after``).
    Self-loops — a step that reads and writes the same path, e.g. a shared
    accumulator — are already excluded from ``before`` by
    ``build_step_network`` and run together in one layer, so they do not form
    a cycle here.

    Args:
        ancestors: The per-step metadata from ``build_step_network`` (its
            keys are the full set of step identifiers); each value's
            ``priority`` is read to decide whether a cycle is orderable.
        nodes: The path dependency map from ``build_step_network`` — each
            value has ``before`` (producers) and ``after`` (consumers) sets.

    Returns:
        A list of step identifiers describing one NON-DETERMINISTIC cycle —
        a closed loop (first element repeated at the end, e.g. ``[a, b, a]``)
        in which at least two members share a priority — or ``None`` if every
        cycle, if any, is deterministically orderable by distinct priorities
        (those are allowed, with a warning).
    """
    # Build the step -> step adjacency: an edge producer -> consumer means
    # the producer must run strictly before the consumer.
    graph = {step_key: set() for step_key in ancestors}
    for node in nodes.values():
        before = node.get('before') or ()
        after = node.get('after') or ()
        for producer in before:
            if producer not in graph:
                continue
            for consumer in after:
                if consumer != producer:
                    graph[producer].add(consumer)
    # Per-step pure-reader edges (accumulator -> reader) carry the same ordering
    # constraint as a ``before`` -> ``after`` edge and must be visible to cycle
    # detection, so a genuine reader↔accumulator feedback is still caught.
    for step_key, ancestor in ancestors.items():
        for pred in ancestor.get('_extra_predecessors', ()):
            if pred in graph and pred != step_key:
                graph[pred].add(step_key)

    # Iterative DFS with three-colour marking (white/grey/black). A grey
    # successor is a back-edge, i.e. a cycle; parent pointers reconstruct it.
    # Successors are visited in a stable (repr-sorted) order so the reported
    # cycle is itself deterministic.
    WHITE, GREY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    parent = {}
    allowed_cycles = []

    def _priority(step_key):
        pr = ancestors.get(step_key, {}).get('priority')
        return 0.0 if pr is None else pr

    def _agent_subtree(step_key):
        # The ('agents', <id>) prefix a step lives under, or None if the step is
        # not inside a per-agent subtree. Used to tell an inter-agent tie (two
        # independent agents' same-typed steps) from a genuine within-agent one.
        k = step_key if isinstance(step_key, (tuple, list)) else (step_key,)
        try:
            i = list(k).index('agents')
        except ValueError:
            return None
        return tuple(k[:i + 2])

    for root in graph:
        if color[root] != WHITE:
            continue
        color[root] = GREY
        stack = [(root, iter(sorted(graph[root], key=repr)))]
        while stack:
            node, successors = stack[-1]
            descended = False
            for succ in successors:
                if color[succ] == WHITE:
                    color[succ] = GREY
                    parent[succ] = node
                    stack.append((succ, iter(sorted(graph[succ], key=repr))))
                    descended = True
                    break
                if color[succ] == GREY:
                    # Back-edge node -> succ closes a cycle. Walk parents
                    # from node back up to succ, then close the loop.
                    cycle = [node]
                    cursor = node
                    while cursor != succ:
                        cursor = parent[cursor]
                        cycle.append(cursor)
                    cycle.reverse()
                    cycle.append(succ)
                    # Priority-guard: if every step in the cycle has a
                    # distinct priority, ``determine_steps`` orders them by
                    # priority alone — deterministic — so the cycle is
                    # allowed (recorded for one end-of-scan warning). A
                    # repeated priority among cycle members is the real
                    # hash-seed hazard (#233): report that cycle as fatal.
                    members = cycle[:-1]
                    priorities = [_priority(m) for m in members]
                    if len(set(priorities)) == len(priorities):
                        allowed_cycles.append(cycle)
                        continue
                    # Tied priorities among cycle members. If the cycle spans
                    # MORE THAN ONE agent subtree (agents/<id>/...), the tie is
                    # between corresponding steps of INDEPENDENT agents — the
                    # same step type carries the same priority in every agent,
                    # and the agents couple only through the shared environment
                    # and the `agents` container (a structural map). Their
                    # inter-agent evaluation order does not change any agent's
                    # own result, so this is a multi-agent wiring artifact, not
                    # the hash-seed hazard #233 targets. Allow it; a tie confined
                    # to a SINGLE agent subtree is still reported as fatal.
                    agent_subtrees = {
                        s for s in (_agent_subtree(m) for m in members)
                        if s is not None}
                    if len(agent_subtrees) > 1:
                        allowed_cycles.append(cycle)
                        continue
                    return cycle
                # BLACK successors are fully explored; skip them.
            if not descended:
                color[node] = BLACK
                stack.pop()

    if allowed_cycles:
        def _fmt(step_key):
            if isinstance(step_key, (tuple, list)):
                return '/'.join(str(part) for part in step_key)
            return str(step_key)
        shown = '; '.join(
            ' -> '.join(_fmt(k) for k in cyc) for cyc in allowed_cycles[:5])
        more = '' if len(allowed_cycles) <= 5 else f' (+{len(allowed_cycles) - 5} more)'
        warnings.warn(
            f'{len(allowed_cycles)} step dependency cycle(s) resolved by '
            f'distinct-priority ordering (deterministic, allowed): '
            f'{shown}{more}. These are feedback loops among zero-time steps '
            f'that priority orders unambiguously; a cycle whose members share '
            f'a priority would instead be rejected (issue #233).',
            stacklevel=2)
    return None


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
        if not all(len(fulfilled[input]) == 0 for input in step_inputs):
            continue
        # A pure reader waits on its inout accumulators too (see
        # ``_extra_predecessors`` in build_step_network): it is ready only once
        # every accumulator it reads from has already run this cycle.
        extra = steps[step_path].get('_extra_predecessors', ())
        if any(pred in remaining for pred in extra):
            continue
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
