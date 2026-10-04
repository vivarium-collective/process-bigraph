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
