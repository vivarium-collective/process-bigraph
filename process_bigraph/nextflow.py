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


def _port_type_name(port_decl: Any) -> str:
    """Return the bare type name from a port declaration.

    ``'list[float]'`` → ``'list'``; ``{'_type': 'string'}`` → ``'string'``.
    Used to decide the Nextflow input/output keyword.
    """
    raw = _port_schema(port_decl).get('_type', '')
    if not isinstance(raw, str):
        return ''
    bracket = raw.find('[')
    return raw[:bracket] if bracket != -1 else raw


def _port_to_nextflow_decl(port_name: str,
                           port_decl: Any,
                           class_overrides: Optional[Dict[str, str]] = None) -> str:
    """Derive a Nextflow input/output declaration line from a port schema.

    Conventions (in priority order):

    1. A class-level ``nextflow_port_decls: {port: decl}`` override on
       the Step wins — this is the escape hatch for Groovy constructs
       that don't fit the structural model (``tuple val, env(...), ...,
       emit: name``). Overrides live on the class, not the port schema,
       so they bypass bigraph-schema's reify_schema parameter walk.
    2. Ports whose type is ``'path'`` or carry ``_is_file: True`` emit
       ``path <name>`` — Nextflow stages them as files.
    3. ``'tuple[A, B, C]'`` emits ``tuple val(name_0), val(name_1), ...``;
       individual tuple elements can be promoted to ``path()`` via
       ``_tuple_paths: [index, ...]`` on the port schema.
    4. Everything else (scalars, lists, maps, nested containers) emits
       ``val <name>``. Collections travel as Groovy values; JSON
       marshalling (if needed) is the runner's concern, not Nextflow's.
    """
    if class_overrides and port_name in class_overrides:
        return class_overrides[port_name]

    schema = _port_schema(port_decl)

    type_name = _port_type_name(port_decl)
    if type_name == 'path' or schema.get('_is_file') is True:
        return f'path {port_name}'

    if type_name == 'tuple':
        raw = schema.get('_type', '')
        inner = raw[raw.find('[') + 1:-1] if '[' in raw else ''
        elements = [e.strip() for e in inner.split(',') if e.strip()]
        if not elements:
            return f'val {port_name}'
        path_indices = set(schema.get('_tuple_paths', ()))
        parts = []
        for i, _elem_type in enumerate(elements):
            kind = 'path' if i in path_indices else 'val'
            parts.append(f'{kind}({port_name}_{i})')
        return 'tuple ' + ', '.join(parts)

    return f'val {port_name}'


def _class_annotation(instance: Any, key: str, default: Any = None) -> Any:
    return getattr(type(instance), key, default)


def _is_plumbing(instance: Any) -> bool:
    return _class_annotation(instance, 'nextflow_operator') is not None


def _topological_order(step_paths: Dict[Path, Dict],
                       step_dependencies: Dict[Path, Dict],
                       node_dependencies: Optional[Dict[Path, Dict]] = None) -> List[Path]:
    """Kahn's algorithm over the step graph.

    Edge model: prefer ``node_dependencies`` (authoritative, prefix-aware).
    For each shared store path, every writer in ``before`` precedes every
    reader in ``after``. Falls back to exact ``input_path == output_path``
    matching when ``node_dependencies`` is absent (back-compat).
    """
    incoming = {sp: set() for sp in step_paths}
    outgoing = {sp: set() for sp in step_paths}

    if node_dependencies:
        for deps in node_dependencies.values():
            writers = [w for w in deps.get('before', ()) if w in step_paths]
            readers = [r for r in deps.get('after', ()) if r in step_paths]
            for w in writers:
                for r in readers:
                    if w != r:
                        outgoing[w].add(r)
                        incoming[r].add(w)
    else:
        producers = {}
        for step_path, info in step_dependencies.items():
            for out_path in info.get('output_paths', []):
                producers[tuple(out_path)] = step_path
        for step_path, info in step_dependencies.items():
            for in_path in info.get('input_paths', []):
                producer = producers.get(tuple(in_path))
                if producer is not None and producer != step_path:
                    incoming[step_path].add(producer)
                    outgoing[producer].add(step_path)

    ordered = []
    remaining = {sp: set(preds) for sp, preds in incoming.items()}
    ready = sorted(sp for sp, preds in remaining.items() if not preds)
    while ready:
        step = ready.pop(0)
        ordered.append(step)
        for consumer in outgoing[step]:
            remaining[consumer].discard(step)
            if not remaining[consumer] and consumer not in ordered and consumer not in ready:
                ready.append(consumer)
        ready.sort()

    if len(ordered) != len(step_paths):
        missing = set(step_paths) - set(ordered)
        raise ValueError(
            f"step graph contains a cycle; could not order: {sorted(missing)!r}")
    return ordered


def _script_body(instance: Any,
                 step_name: str,
                 inputs_wires: Dict[str, List],
                 outputs_wires: Dict[str, List],
                 python: str = 'python',
                 config_ref: Optional[str] = None) -> str:
    """Return the script block for a process.

    Priority order:
      1. ``nextflow_script()`` override on the Step — escape hatch for
         legacy CLI wrappers.
      2. Auto-generated body that invokes ``process_bigraph.run_step``,
         dispatching to the Step class by fully-qualified name. The
         same ``update()`` runs natively and under Nextflow.
    """
    if hasattr(instance, 'nextflow_script'):
        return instance.nextflow_script()

    cls = type(instance)
    fq = f"{cls.__module__}.{cls.__name__}"
    in_flags = ' '.join(
        f'--in {port}="${{{port}}}"' for port in inputs_wires
    )
    out_flags = ' '.join(
        f'--out {port}={port}.json' for port in outputs_wires
    )
    parts = [
        f'{python} -m process_bigraph.run_step',
        f'--class {fq}',
    ]
    if config_ref:
        parts.append(f'--config {config_ref}')
    if in_flags:
        parts.append(in_flags)
    if out_flags:
        parts.append(out_flags)
    cmd = ' \\\n    '.join(parts)
    return f'"""\n{cmd}\n"""'


def _composite_node_script(instance: Any,
                           doc_ref: str,
                           steps: int,
                           inputs_wires: Dict[str, List],
                           outputs_wires: Dict[str, List],
                           python: str = 'python') -> str:
    """Emit the ``script:`` block for a whole-Composite node.

    Runs the entire nested simulation via ``run_composite``: the first input
    port (if any) is staged as the initial-state document; the first output
    port (if any) receives the final-state document.

    EXPERIMENTAL: the composite-node → run_composite rendering is
    scaffolding and not yet runnable end-to-end — the composite document is
    not auto-staged and composite nodes are not yet integrated into the
    topological ordering. The plain Step-network path IS fully supported.
    See docs/superpowers/specs/2026-08-13-nextflow-step-network-deploy-design.md.
    """
    parts = [
        f'{python} -m process_bigraph.run_composite',
        f'--document {doc_ref}',
        f'--steps {steps}',
    ]
    in_iter = iter(inputs_wires)
    first_in = next(in_iter, None)
    if first_in is not None:
        parts.append(f'--initial-state ${{{first_in}}}')
    out_iter = iter(outputs_wires)
    first_out = next(out_iter, None)
    if first_out is not None:
        parts.append(f'--state-out {first_out}.json')
    cmd = ' \\\n    '.join(parts)
    return f'"""\n{cmd}\n"""'


def _directive_lines(directives: Dict[str, Any]) -> List[str]:
    """Render directive key/values as one-line Nextflow directives.

    Strings that start with ``{`` are emitted raw — they are Groovy
    closures (e.g. ``publishDir { "..." }, mode: "copy"``) and
    wrapping them in quotes would turn the closure into a string
    literal. Everything else is repr-quoted.
    """
    lines = []
    for key, value in directives.items():
        if isinstance(value, bool):
            lines.append(f'    {key} {str(value).lower()}')
        elif isinstance(value, str):
            if value.lstrip().startswith('{'):
                lines.append(f'    {key} {value}')
            else:
                lines.append(f'    {key} {value!r}')
        else:
            lines.append(f'    {key} {value}')
    return lines


def _process_block(step_name: str,
                   instance: Any,
                   inputs_wires: Dict[str, List],
                   outputs_wires: Dict[str, List],
                   python: str = 'python',
                   config_ref: Optional[str] = None) -> str:
    """Emit a ``process { ... }`` block for a non-plumbing Step."""
    lines = [f'process {step_name} {{']

    directives = dict(_class_annotation(instance, 'nextflow_directives', {}) or {})
    if _is_plumbing(instance):
        directives.setdefault('executor', 'local')
    lines.extend(_directive_lines(directives))

    step_inputs_schema = instance.inputs() if hasattr(instance, 'inputs') else {}
    step_outputs_schema = instance.outputs() if hasattr(instance, 'outputs') else {}
    class_overrides = _class_annotation(instance, 'nextflow_port_decls', {}) or {}

    if inputs_wires or config_ref:
        lines.append('    input:')
        for port in inputs_wires:
            decl = _port_to_nextflow_decl(
                port, step_inputs_schema.get(port, {}), class_overrides)
            lines.append(f'    {decl}')
        if config_ref:
            # Declared, not merely referenced: writing the file beside main.nf
            # and passing --config is NOT enough -- Nextflow stages only
            # declared inputs, so the task would open a path that does not
            # exist in its work dir. stageAs pins the name the script uses.
            lines.append(f"    path config_json, stageAs: '{config_ref}'")

    uses_run_step = not hasattr(instance, 'nextflow_script')

    if outputs_wires:
        lines.append('    output:')
        for port in outputs_wires:
            if port not in class_overrides and uses_run_step:
                decl = f'path "{port}.json"'
            else:
                decl = _port_to_nextflow_decl(
                    port, step_outputs_schema.get(port, {}), class_overrides)
            lines.append(f'    {decl}')

    lines.append('    script:')
    lines.append(_script_body(instance, step_name, inputs_wires, outputs_wires,
                              python, config_ref))

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


def _unified_order(step_paths: Dict[Path, Dict],
                   composite_nodes: Dict[Path, Dict],
                   seed_order: List[Path]) -> List[Path]:
    """Topologically order Steps AND nested Composites as one graph.

    A Steps-only sort cannot see a dependency that runs *through* a composite
    node (``parca -> cache -> runs -> results -> analysis``), so the two ends
    come out in arbitrary relative order and the emitted workflow references an
    unassigned channel. Sorting both kinds together fixes it in one pass.
    """
    units: Dict[Path, Dict] = {}
    units.update(step_paths)
    units.update(composite_nodes)

    def wires(node, key):
        got = set()
        for w in (node.get(key) or {}).values():
            if isinstance(w, list) and w and not isinstance(w[0], list):
                got.add(tuple(w))
        return got

    producer: Dict[Path, Path] = {}
    for up, node in units.items():
        for out in wires(node, 'outputs'):
            producer.setdefault(out, up)

    incoming = {u: set() for u in units}
    outgoing: Dict[Path, set] = {u: set() for u in units}
    for up, node in units.items():
        for inp in wires(node, 'inputs'):
            src = producer.get(inp)
            if src is not None and src != up:
                incoming[up].add(src)
                outgoing[src].add(up)

    # Stable: prefer the Steps-only order for units with no constraint.
    rank = {u: i for i, u in enumerate(seed_order)}
    ready = sorted((u for u in units if not incoming[u]),
                   key=lambda u: rank.get(u, len(rank)))
    ordered: List[Path] = []
    while ready:
        u = ready.pop(0)
        ordered.append(u)
        for v in sorted(outgoing[u], key=lambda x: rank.get(x, len(rank))):
            incoming[v].discard(u)
            if not incoming[v] and v not in ordered and v not in ready:
                ready.append(v)
        ready.sort(key=lambda x: rank.get(x, len(rank)))
    for u in units:                      # cycles / leftovers: keep them
        if u not in ordered:
            ordered.append(u)
    return ordered


def _insert_position(workflow_lines: List[str],
                     input_paths: List[Path],
                     path_to_channel: Dict[Path, str]) -> int:
    """Index at which to splice a call so it follows its own producers.

    The composite-node loop runs after the Step loop, so a naive append puts a
    producer AFTER its consumer. Insert instead just past the last line that
    assigns one of this node's input channels.
    """
    wanted = {path_to_channel[p] for p in input_paths if p in path_to_channel}
    pos = 1
    for i, line in enumerate(workflow_lines):
        lhs = line.strip().split('=')[0].strip()
        if lhs in wanted:
            pos = i + 1
    return pos


def _terminal_channels(step_paths: Dict[Path, Dict],
                       step_dependencies: Dict[Path, Dict],
                       path_to_channel: Dict[Path, str]) -> List[str]:
    """Channels produced in this scope that nothing in this scope consumes.

    These are the scope's results -- what a sub-workflow should ``emit:``.
    """
    consumed = set()
    for sp, node in step_paths.items():
        for wire in (node.get('inputs') or {}).values():
            if isinstance(wire, list) and wire and not isinstance(wire[0], list):
                consumed.add(tuple(wire))
    out: List[str] = []
    for sp, node in step_paths.items():
        for wire in (node.get('outputs') or {}).values():
            t = tuple(wire) if isinstance(wire, list) else None
            if t and t not in consumed and t in path_to_channel:
                ch = path_to_channel[t]
                if ch not in out:
                    out.append(ch)
    return out


def _homogeneous_group(step_paths: Dict[Path, Dict],
                       order: List[Path]) -> Dict[str, List[Path]]:
    """Group step paths by (class, rendered script) -- the "re-roll" key.

    An unrolled document expresses N identical units as N distinct nodes.
    Nextflow's natural form is ONE process invoked over a channel of N items,
    which is also the only form whose outputs arrive as a single channel (and
    therefore the only one that can be gathered without an N-way merge).
    Nodes that share a class AND render an identical script body are exactly
    the nodes that can be re-rolled that way.
    """
    groups: Dict[str, List[Path]] = {}
    for sp in order:
        inst = step_paths[sp]['instance']
        key = type(inst).__module__ + '.' + type(inst).__qualname__
        groups.setdefault(key, []).append(sp)
    return groups


def render_composite(composite: Any, options: Optional[Dict[str, Any]] = None) -> str:
    """Render a realized ``Composite`` as a Nextflow DSL2 workflow string.

    Args:
        composite: an initialized ``process_bigraph.Composite``.
        options: optional dict; recognized keys:
            ``workflow_name`` (default ``'main'``) — entry workflow name.
                ``deploy()`` passes ``''`` (an unnamed entry workflow),
                because ``main`` is reserved in Nextflow.
            ``header`` (default DSL2 declaration) — leading text.
            ``python`` (default ``'python'``) — interpreter used in emitted
                task scripts; ``deploy()`` pins this to ``sys.executable``.
            ``composite_steps`` (default ``1000``) — steps to advance a
                whole-Composite node (experimental path).
            ``composite_documents`` (default ``{}``) — ``{step_name:
                document_path}`` for whole-Composite nodes (experimental path).

    Returns:
        The rendered workflow document, ready to save as ``.nf``.

    Composite nodes (nested ``Composite`` instances rendered as a whole-task
    ``run_composite`` process, see ``_composite_node_script``): EXPERIMENTAL.
    This rendering path is scaffolding and not yet runnable end-to-end — the
    composite document is not auto-staged and composite nodes are not yet
    integrated into the topological ordering. The plain Step-network path IS
    fully supported. See
    docs/superpowers/specs/2026-08-13-nextflow-step-network-deploy-design.md.
    """
    options = options or {}
    workflow_name = options.get('workflow_name', 'main')
    header = options.get('header', 'nextflow.enable.dsl=2\n')
    python = options.get('python', 'python')

    step_paths = composite.step_paths
    step_dependencies = getattr(composite, 'step_dependencies', {}) or {}

    node_dependencies = getattr(composite, 'node_dependencies', None)
    order = _topological_order(step_paths, step_dependencies, node_dependencies)

    # Nested Composites are units of the SAME graph as the Steps. Ordering them
    # in a second pass (the original structure) emits a producer after its
    # consumer; interleave them here instead.
    from process_bigraph.composite import Composite as _CNode
    _composite_nodes = {
        np: nd for np, nd in (getattr(composite, 'process_paths', {}) or {}).items()
        if isinstance(nd.get('instance'), _CNode)}
    order = _unified_order(step_paths, _composite_nodes, order)

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

    # Nested Composites are producers too: register their output paths before the
    # Step loop runs, or a consuming Step resolves to params.<path> and the run
    # dies with "A process input channel evaluates to null".
    from process_bigraph.composite import Composite as _C0
    for _np, _nd in (getattr(composite, 'process_paths', {}) or {}).items():
        if isinstance(_nd.get('instance'), _C0):
            for _w in (_nd.get('outputs') or {}).values():
                if isinstance(_w, list) and _w and not isinstance(_w[0], list):
                    path_to_channel.setdefault(tuple(_w), _path_to_channel_name(tuple(_w)))

    # A sub-workflow's `take:` ports are in scope as bare identifiers. Seed them
    # so _channel_expr_for_input resolves to `cache` rather than `params.cache`.
    for tp in (options.get('_take_ports') or []):
        path_to_channel.setdefault((tp,), tp)

    # Pass 1: collect process blocks for non-plumbing Steps.
    process_blocks: List[str] = []
    subworkflow_blocks: List[str] = []
    staged_configs: Dict[str, Dict] = options.setdefault('_staged_configs', {})
    take_ports = options.get('_take_ports') or []
    emit_ports = options.get('_emit_ports') or []
    workflow_lines: List[str] = [f'workflow {workflow_name} {{']
    if take_ports:
        # DSL2 sub-workflow inputs. These shadow the parent's channels by name,
        # which is how the composite's own bridge inputs cross the boundary.
        workflow_lines.append('    take:')
        for tp in take_ports:
            workflow_lines.append(f'    {tp}')
        workflow_lines.append('    main:')

    for step_path in order:
        if step_path not in step_paths:
            # A composite node: reserve its ordered slot; the loop below fills it.
            workflow_lines.append(f'@@SUBWF:{_path_to_step_name(step_path)}@@')
            continue
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
            node_config = step.get('config') or getattr(instance, 'config', None) or {}
            cfg_ref = None
            if node_config:
                cfg_ref = f'{name}.config.json'
                staged_configs[cfg_ref] = dict(node_config)
            process_blocks.append(
                _process_block(name, instance, inputs_wires, outputs_wires,
                               python, cfg_ref))

            # Emit a call with positional channel args in input-port order.
            call_args = []
            for port_name, wire in inputs_wires.items():
                cardinality = _port_annotation(
                    instance.inputs().get(port_name, {}), '_cardinality')
                call_args.append(_channel_expr_for_input(
                    port_name, wire, path_to_channel, cardinality,
                    bridge_inputs))
            if cfg_ref:
                # A value channel over the staged config file, so the task
                # receives it as a real input rather than a dangling filename.
                # DOUBLE quotes: Groovy does not interpolate single-quoted
                # strings, so '${projectDir}/x' would be a literal dollar sign.
                call_args.append(f'file("${{projectDir}}/{cfg_ref}")')

            # The process's outputs become channels named after their wire path.
            out_port, out_wire = next(iter(outputs_wires.items()), (None, None))
            if out_wire is not None:
                out_channel = _path_to_channel_name(tuple(out_wire))
                call = f'{out_channel} = {name}({", ".join(call_args)})'
            else:
                call = f'{name}({", ".join(call_args)})'
            workflow_lines.append(f'    {call}')

    if emit_ports:
        # Gather every terminal producer in this scope into ONE emitted channel.
        # `_terminal_channels` is what makes a nested scope's N units arrive at
        # the parent as a single channel instead of N separate ones.
        terminal = _terminal_channels(step_paths, step_dependencies, path_to_channel)
        if len(terminal) > 1:
            # Chain BINARY mixes. A single `a.mix(b, c, ...)` with N-1
            # arguments is a Java method call and fails at 255 parameters
            # ("bad parameter count"); N statements of arity 1 do not.
            acc = '_merged'
            workflow_lines.append(f'    {acc} = {terminal[0]}')
            for ch in terminal[1:]:
                workflow_lines.append(f'    {acc} = {acc}.mix({ch})')
            workflow_lines.append('    emit:')
            workflow_lines.append(f'    {acc}.collect()')
        elif terminal:
            workflow_lines.append('    emit:')
            workflow_lines.append(f'    {terminal[0]}.collect()')
        else:
            workflow_lines.append('    emit:')
            workflow_lines.append('    channel.empty()')
    workflow_lines.append('}')

    from process_bigraph.composite import Composite as _Composite
    default_steps = options.get('composite_steps', 1000)
    doc_map = options.get('composite_documents', {})
    nest = options.get('nest_composites', True)

    for node_path, node in (getattr(composite, 'process_paths', {}) or {}).items():
        instance = node.get('instance')
        if not isinstance(instance, _Composite):
            continue
        name = _path_to_step_name(node_path)
        inputs_wires = node.get('inputs') or {}
        outputs_wires = node.get('outputs') or {}
        inner_steps = getattr(instance, 'step_paths', {}) or {}

        # NEW: a nested Composite with renderable content becomes a Nextflow
        # SUB-WORKFLOW, preserving the hierarchy the document already carries.
        # Collapsing to one ``run_composite`` task (the original behaviour) is
        # kept for an empty/opaque composite, and via nest_composites=False.
        if nest and inner_steps:
            sub_opts = dict(options)
            sub_opts['workflow_name'] = name
            sub_opts['_emit_ports'] = list(outputs_wires)
            sub_opts['_take_ports'] = list(inputs_wires)
            sub_text = render_composite(instance, {**sub_opts, 'header': ''})
            subworkflow_blocks.append(sub_text.strip())

            call_args = [
                _channel_expr_for_input(port, wire, path_to_channel, None, bridge_inputs)
                for port, wire in inputs_wires.items()]
            out_port, out_wire = next(iter(outputs_wires.items()), (None, None))
            if out_wire := next(iter(outputs_wires.values()), None):
                out_channel = _path_to_channel_name(tuple(out_wire))
                call_line = f'    {out_channel} = {name}({", ".join(call_args)})'
            else:
                call_line = f'    {name}({", ".join(call_args)})'
            marker = f'@@SUBWF:{name}@@'
            if marker in workflow_lines:
                workflow_lines[workflow_lines.index(marker)] = call_line
            else:
                workflow_lines.insert(-1, call_line)
            continue

        doc_ref = doc_map.get(name, f'{name}_document.json')

        block_lines = [f'process {name} {{']
        if inputs_wires:
            block_lines.append('    input:')
            for port in inputs_wires:
                block_lines.append(f'    path {port}')
        if outputs_wires:
            first_out_port = next(iter(outputs_wires), None)
            if first_out_port is not None:
                block_lines.append('    output:')
                block_lines.append(f'    path "{first_out_port}.json"')
        block_lines.append('    script:')
        block_lines.append(_composite_node_script(
            instance, doc_ref, default_steps, inputs_wires, outputs_wires, python))
        block_lines.append('}')
        process_blocks.append('\n'.join(block_lines))

        call_args = [
            _channel_expr_for_input(port, wire, path_to_channel, None, bridge_inputs)
            for port, wire in inputs_wires.items()]
        out_port, out_wire = next(iter(outputs_wires.items()), (None, None))
        if out_wire is not None:
            out_channel = _path_to_channel_name(tuple(out_wire))
            workflow_lines.insert(-1, f'    {out_channel} = {name}({", ".join(call_args)})')
        else:
            workflow_lines.insert(-1, f'    {name}({", ".join(call_args)})')

    parts = [header.rstrip(), '']
    parts.extend(process_blocks)
    parts.extend(subworkflow_blocks)
    parts.append('')
    workflow_lines = [l for l in workflow_lines if not l.startswith('@@SUBWF:')]
    parts.append('\n'.join(workflow_lines))
    return '\n'.join(parts) + '\n'
