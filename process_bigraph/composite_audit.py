"""Composite internal-consistency audit (spec §7, non-tautological slice).

A built Composite is already type-consistent by construction (combine raises
on conflicting member port types; the boundary face is wire-derived). This
module checks only what construction does NOT guarantee: on each shared store,
what a member writes must fit the bounds/units each reading member requires;
and a declared boundary-output override must subsume what members produce.
Advisory only (warning/info, never error). Purely additive.
"""
from dataclasses import dataclass

from bigraph_schema.subsumption import range_subsumes, units_compatible
from bigraph_schema.contract_audit import Finding, AuditReport


@dataclass
class MemberWiring:
    address: str
    parent_path: tuple
    inputs_face: dict          # raw declared input ports {port: schema}
    outputs_face: dict         # raw declared output ports {port: schema}
    input_wires: dict          # {port: wire}  (wire is a path list, or nested)
    output_wires: dict


def _member_wirings(composite):
    """Enumerate a built composite's member edges as MemberWiring records.

    Reads composite.edge_paths ({path: edge_dict}); each edge_dict carries the
    built 'instance' (for the RAW declared face via inputs()/outputs()) and the
    'inputs'/'outputs' wires. Members that cannot be read are skipped.
    """
    wirings = []
    edge_paths = getattr(composite, 'edge_paths', None) or {}
    for path, edge in edge_paths.items():
        instance = edge.get('instance')
        if instance is None:
            continue
        try:
            inputs_face = dict(instance.inputs() or {})
            outputs_face = dict(instance.outputs() or {})
        except Exception:  # noqa: BLE001 - unreadable face: skip this member
            continue

        # Format address: if dict with 'protocol' and 'data', reconstruct as 'protocol:data'
        addr = edge.get('address', path)
        if isinstance(addr, dict):
            protocol = addr.get('protocol', '')
            data = addr.get('data', '')
            address_str = f'{protocol}:{data}' if protocol else str(data)
        else:
            address_str = str(addr)

        wirings.append(MemberWiring(
            address=address_str,
            parent_path=tuple(path[:-1]),
            inputs_face=inputs_face,
            outputs_face=outputs_face,
            input_wires=dict(edge.get('inputs') or {}),
            output_wires=dict(edge.get('outputs') or {}),
        ))
    return wirings


def _resolve_wire(parent_path, wire):
    """Absolute store path for a FLAT wire (a list of plain string segments),
    resolved against the member's parent path. Returns None for a nested dict
    wire or a relative ('..') / non-string segment — those are unanalyzable.
    """
    if not isinstance(wire, (list, tuple)):
        return None
    if not all(isinstance(segment, str) for segment in wire):
        return None
    if any(segment == '..' for segment in wire):
        return None
    return tuple(parent_path) + tuple(wire)


def _store_graph(wirings):
    """Map each resolvable store path to its writers and readers.

    writers/readers are (address, port, raw_port_schema). A wire that does not
    resolve to a flat path is recorded in the returned unanalyzable list.
    """
    graph = {}
    unanalyzable = []

    def _slot(store):
        return graph.setdefault(store, {'writers': [], 'readers': []})

    for wiring in wirings:
        for port, wire in wiring.output_wires.items():
            store = _resolve_wire(wiring.parent_path, wire)
            if store is None:
                unanalyzable.append(f'{wiring.address} output {port!r} wire is not a flat path')
                continue
            _slot(store)['writers'].append((wiring.address, port, wiring.outputs_face.get(port)))
        for port, wire in wiring.input_wires.items():
            store = _resolve_wire(wiring.parent_path, wire)
            if store is None:
                unanalyzable.append(f'{wiring.address} input {port!r} wire is not a flat path')
                continue
            _slot(store)['readers'].append((wiring.address, port, wiring.inputs_face.get(port)))

    return graph, unanalyzable


def _check_store_compat(graph):
    """For each shared store, each writer's output must fit every reader's
    required bounds/units (writer-produced ⊆ reader-accepted). Returns warnings.
    """
    findings = []
    for store, slot in graph.items():
        writers = slot.get('writers', [])
        readers = slot.get('readers', [])
        if not writers or not readers:
            continue
        where = '.'.join(store)
        for w_addr, w_port, w_schema in writers:
            for r_addr, r_port, r_schema in readers:
                if w_schema is None or r_schema is None:
                    continue
                ok, reason = range_subsumes(r_schema, w_schema)   # required=reader, candidate=writer
                if not ok:
                    findings.append(Finding('warning', 'store_bounds_mismatch', where,
                        f'store {where!r}: {w_addr}.{w_port} produces a range not accepted by '
                        f'{r_addr}.{r_port} ({reason})'))
                ok, reason = units_compatible(r_schema, w_schema)
                if not ok:
                    findings.append(Finding('warning', 'store_units_mismatch', where,
                        f'store {where!r}: {w_addr}.{w_port} units not convertible to '
                        f'{r_addr}.{r_port} ({reason})'))
    return findings


def _bridge_outputs(composite):
    """The composite's boundary output wires {port: wire}, best-effort."""
    reader = getattr(composite, 'read_bridge_outputs', None)
    if callable(reader):
        try:
            return dict(reader() or {})
        except Exception:  # noqa: BLE001
            return {}
    bridge = (getattr(composite, 'config', None) or {}).get('bridge') or {}
    return dict(bridge.get('outputs') or {})


def _check_boundary_override(composite, graph):
    """A declared interface.outputs override must subsume what the internal
    writers of the store it wires to actually produce. No override → no check.
    """
    findings = []
    config = getattr(composite, 'config', None) or {}
    declared = (config.get('interface') or {}).get('outputs') or {}
    if not declared:
        return findings
    bridge = _bridge_outputs(composite)
    parent = ()
    for port, declared_schema in declared.items():
        wire = bridge.get(port)
        store = _resolve_wire(parent, wire) if wire is not None else None
        if store is None or store not in graph:
            continue
        for _w_addr, _w_port, w_schema in graph[store]['writers']:
            if w_schema is None:
                continue
            ok, reason = range_subsumes(declared_schema, w_schema)  # declared boundary must accept producer
            if not ok:
                findings.append(Finding('warning', 'boundary_override_mismatch', f'outputs.{port}',
                    f'declared boundary output {port!r} does not accept what the internal '
                    f'producer yields ({reason})'))
            ok, reason = units_compatible(declared_schema, w_schema)
            if not ok:
                findings.append(Finding('warning', 'boundary_override_units', f'outputs.{port}',
                    f'declared boundary output {port!r} units mismatch ({reason})'))
    return findings
