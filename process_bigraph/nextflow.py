"""
Render a Composite as a Nextflow DSL2 workflow document.

The renderer is a two-pass interpreter over the step graph:

1. **Contribution pass** — ask each Step for the fragments it owns
   (process block, optional script, directives, port annotations).
2. **Linking pass** — walk the graph topologically, name one channel
   per distinct global-state path, and assemble a ``workflow { }``
   block whose invocation order respects the producer/consumer
   relationships carried by the step wires.

Plumbing Steps (``process_bigraph.plumbing``) carry a
``nextflow_operator`` class attribute and render as channel operator
calls (``.mix()``, ``.combine()``, ``.groupTuple(by: ...)``, ...)
instead of as Nextflow processes. Everything else becomes a
``process { ... }`` block.

This is the only place in process-bigraph that knows the shape of a
Nextflow document. Everything else is declarative on the Steps —
``inputs()`` / ``outputs()`` for port types, ``_cardinality`` /
``_nextflow`` / ``_nextflow_directives`` annotations for
renderer-specific semantics, and optional ``nextflow_script()`` /
``nextflow_directives`` overrides on individual Step classes for
custom emission.

See vEcoli's ``doc/nextflow_composite_spec.md`` for the spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


Path = Tuple[str, ...]


def _path_to_channel_name(path: Path) -> str:
    """Stable, valid Groovy-identifier name for a channel."""
    if not path:
        return 'ch_root'
    return 'ch_' + '_'.join(str(p).replace('-', '_') for p in path)


def _path_to_step_name(path: Path) -> str:
    """Stable, valid Nextflow process name from a step path."""
    if not path:
        return 'step_root'
    return '_'.join(str(p).replace('-', '_') for p in path)


def _port_schema(port_decl: Any) -> Dict[str, Any]:
    """Normalize a port declaration (string type or dict) to a dict."""
    if isinstance(port_decl, dict):
        return port_decl
    return {'_type': port_decl}


def _port_annotation(port_decl: Any, key: str, default: Any = None) -> Any:
    return _port_schema(port_decl).get(key, default)


def _class_annotation(instance: Any, key: str, default: Any = None) -> Any:
    return getattr(type(instance), key, default)


def _is_plumbing(instance: Any) -> bool:
    return _class_annotation(instance, 'nextflow_operator') is not None


def _topological_order(step_paths: Dict[Path, Dict],
                       step_dependencies: Dict[Path, Dict]) -> List[Path]:
    """Kahn's algorithm over the step graph.

    Dependency: step A precedes step B if any of B's input_paths is
    produced by A (appears in A's output_paths). Two-pointer reverse
    lookup keeps this O(V+E).
    """
    producers: Dict[Path, Path] = {}
    for step_path, info in step_dependencies.items():
        for out_path in info.get('output_paths', []):
            producers[tuple(out_path)] = step_path

    incoming: Dict[Path, List[Path]] = {sp: [] for sp in step_paths}
    outgoing: Dict[Path, List[Path]] = {sp: [] for sp in step_paths}
    for step_path, info in step_dependencies.items():
        for in_path in info.get('input_paths', []):
            producer = producers.get(tuple(in_path))
            if producer is not None and producer != step_path:
                incoming[step_path].append(producer)
                outgoing[producer].append(step_path)

    ordered: List[Path] = []
    ready = [sp for sp, preds in incoming.items() if not preds]
    ready.sort()
    remaining = dict(incoming)

    while ready:
        step = ready.pop(0)
        ordered.append(step)
        for consumer in outgoing[step]:
            remaining[consumer] = [p for p in remaining[consumer] if p != step]
            if not remaining[consumer] and consumer not in ordered:
                if consumer not in ready:
                    ready.append(consumer)
        ready.sort()

    if len(ordered) != len(step_paths):
        missing = set(step_paths) - set(ordered)
        raise ValueError(
            f"step graph contains a cycle; could not order: {sorted(missing)!r}"
        )
    return ordered


def _script_body(instance: Any,
                 step_name: str,
                 inputs_wires: Dict[str, List],
                 outputs_wires: Dict[str, List]) -> str:
    """Return the script block for a process.

    Defaults to a Python stub invoking the Step class by fully-qualified
    name; Steps override ``nextflow_script()`` to emit anything else.
    """
    if hasattr(instance, 'nextflow_script'):
        return instance.nextflow_script()

    cls = type(instance)
    fq = f"{cls.__module__}.{cls.__name__}"
    in_names = ', '.join(inputs_wires.keys()) or '(none)'
    out_names = ', '.join(outputs_wires.keys()) or '(none)'
    return (
        f'"""\n'
        f'# Stub for {step_name} ({fq}).\n'
        f'# Inputs: {in_names}\n'
        f'# Outputs: {out_names}\n'
        f'# Replace with the invocation appropriate to your runtime.\n'
        f'echo "{step_name} invoked"\n'
        f'"""'
    )


def _directive_lines(directives: Dict[str, Any]) -> List[str]:
    """Render directive key/values as one-line Nextflow directives."""
    lines = []
    for key, value in directives.items():
        if isinstance(value, str):
            lines.append(f'    {key} {value!r}')
        elif isinstance(value, bool):
            lines.append(f'    {key} {str(value).lower()}')
        else:
            lines.append(f'    {key} {value}')
    return lines


def _process_block(step_name: str,
                   instance: Any,
                   inputs_wires: Dict[str, List],
                   outputs_wires: Dict[str, List]) -> str:
    """Emit a ``process { ... }`` block for a non-plumbing Step."""
    lines = [f'process {step_name} {{']

    directives = dict(_class_annotation(instance, 'nextflow_directives', {}) or {})
    if _is_plumbing(instance):
        directives.setdefault('executor', 'local')
    lines.extend(_directive_lines(directives))

    if inputs_wires:
        lines.append('    input:')
        for port in inputs_wires:
            lines.append(f'    val {port}')

    if outputs_wires:
        lines.append('    output:')
        for port in outputs_wires:
            lines.append(f'    val {port}')

    lines.append('    script:')
    lines.append(_script_body(instance, step_name, inputs_wires, outputs_wires))

    lines.append('}')
    return '\n'.join(lines)


def _channel_expr_for_input(port_name: str,
                            wire: List,
                            path_to_channel: Dict[Path, str],
                            port_cardinality: Optional[str],
                            bridge_inputs: Optional[Dict[Path, str]] = None) -> str:
    """Build the channel expression that feeds one input port.

    Resolution order:
      1. A step produces this exact path → use its channel.
      2. A step produces the path with a trailing ``*`` stripped →
         use that channel (the star is the consumer's concern).
      3. The composite bridge declares an input for this path →
         emit ``params.<bridge_name>``, which in Nextflow DSL2 can be
         passed directly into a ``val`` input without explicit wrapping.
      4. Fallback: emit ``params.<joined_path>`` so the user can wire
         the parameter from the Nextflow command line.
    """
    bridge_inputs = bridge_inputs or {}
    path = tuple(wire)
    if path in path_to_channel:
        return path_to_channel[path]

    if path and path[-1] == '*':
        head = path[:-1]
        if head in path_to_channel:
            return path_to_channel[head]

    if path in bridge_inputs:
        return f'params.{bridge_inputs[path]}'

    if path:
        fallback = '_'.join(str(p).replace('-', '_') for p in path)
        return f'params.{fallback}'

    return f'channel.empty() /* TODO: {port_name} wire is empty */'


def _emit_plumbing_call(step_name: str,
                        instance: Any,
                        inputs_wires: Dict[str, List],
                        outputs_wires: Dict[str, List],
                        path_to_channel: Dict[Path, str],
                        bridge_inputs: Dict[Path, str]) -> str:
    """Emit a channel-operator call for a plumbing Step."""
    op = _class_annotation(instance, 'nextflow_operator')

    def resolve(port, wire):
        return _channel_expr_for_input(
            port, wire, path_to_channel, None, bridge_inputs)

    if op == 'mix':
        streams = inputs_wires.get('streams', [])
        channels = [path_to_channel.get(tuple(w), f'ch_{step_name}_in')
                    for w in streams] if isinstance(streams, list) and streams and isinstance(streams[0], list) else []
        if len(channels) >= 2:
            call = f'{channels[0]}.mix({", ".join(channels[1:])})'
        elif len(channels) == 1:
            call = channels[0]
        else:
            call = 'channel.empty() /* TODO: Mix streams unresolved */'
    elif op == 'collect':
        call = f'{resolve("stream", inputs_wires.get("stream", []))}.collect()'
    elif op == 'combine':
        a = resolve('a', inputs_wires.get('a', []))
        b = resolve('b', inputs_wires.get('b', []))
        call = f'{a}.combine({b})'
    elif op == 'groupTuple':
        src = resolve('stream', inputs_wires.get('stream', []))
        key_field = getattr(instance, 'config', {}).get('key_field') if hasattr(instance, 'config') else None
        if key_field:
            call = f'{src}.groupTuple(by: {key_field!r})'
        else:
            call = f'{src}.groupTuple()'
    elif op == 'join':
        left = resolve('left', inputs_wires.get('left', []))
        right = resolve('right', inputs_wires.get('right', []))
        call = f'{left}.join({right})'
    else:
        call = f'channel.empty() /* TODO: unknown plumbing operator {op!r} */'

    out_port, out_wire = next(iter(outputs_wires.items()), (None, None))
    if out_wire is None:
        return f'    // {step_name}: {call}  (no output wire)'
    out_channel = _path_to_channel_name(tuple(out_wire))
    return f'    {out_channel} = {call}'


def render_composite(composite: Any, options: Optional[Dict[str, Any]] = None) -> str:
    """Render a realized ``Composite`` as a Nextflow DSL2 workflow string.

    Args:
        composite: an initialized ``process_bigraph.Composite``.
        options: optional dict; recognized keys:
            ``workflow_name`` (default ``'main'``) — entry workflow name.
            ``header`` (default DSL2 declaration) — leading text.

    Returns:
        The rendered workflow document, ready to save as ``.nf``.
    """
    options = options or {}
    workflow_name = options.get('workflow_name', 'main')
    header = options.get('header', 'nextflow.enable.dsl=2\n')

    step_paths = composite.step_paths
    step_dependencies = getattr(composite, 'step_dependencies', {}) or {}

    order = _topological_order(step_paths, step_dependencies)

    # Assign one channel per producer output_path.
    path_to_channel: Dict[Path, str] = {}
    for step_path, info in step_dependencies.items():
        for out_path in info.get('output_paths', []):
            path_to_channel[tuple(out_path)] = _path_to_channel_name(tuple(out_path))

    # Composite-level inputs declared on the bridge become ``params.<name>``
    # references. The bridge input map is keyed by wire path so the consumer
    # lookup in _channel_expr_for_input is O(1).
    bridge = getattr(composite, 'bridge', None) or {}
    bridge_inputs_decl = bridge.get('inputs', {}) if isinstance(bridge, dict) else {}
    bridge_inputs: Dict[Path, str] = {
        tuple(wire): name
        for name, wire in bridge_inputs_decl.items()
    }

    # Pass 1: collect process blocks for non-plumbing Steps.
    process_blocks: List[str] = []
    workflow_lines: List[str] = [f'workflow {workflow_name} {{']

    for step_path in order:
        step = step_paths[step_path]
        instance = step['instance']
        inputs_wires = step.get('inputs') or {}
        outputs_wires = step.get('outputs') or {}
        name = _path_to_step_name(step_path)

        if _is_plumbing(instance):
            workflow_lines.append(
                _emit_plumbing_call(name, instance, inputs_wires,
                                    outputs_wires, path_to_channel,
                                    bridge_inputs))
        else:
            process_blocks.append(
                _process_block(name, instance, inputs_wires, outputs_wires))

            # Emit a call with positional channel args in input-port order.
            call_args = []
            for port_name, wire in inputs_wires.items():
                cardinality = _port_annotation(
                    instance.inputs().get(port_name, {}), '_cardinality')
                call_args.append(_channel_expr_for_input(
                    port_name, wire, path_to_channel, cardinality,
                    bridge_inputs))

            # The process's outputs become channels named after their wire path.
            out_port, out_wire = next(iter(outputs_wires.items()), (None, None))
            if out_wire is not None:
                out_channel = _path_to_channel_name(tuple(out_wire))
                call = f'{out_channel} = {name}({", ".join(call_args)})'
            else:
                call = f'{name}({", ".join(call_args)})'
            workflow_lines.append(f'    {call}')

    workflow_lines.append('}')

    parts = [header.rstrip(), '']
    parts.extend(process_blocks)
    parts.append('')
    parts.append('\n'.join(workflow_lines))
    return '\n'.join(parts) + '\n'
