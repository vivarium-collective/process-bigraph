from bigraph_schema import allocate_core

from process_bigraph.composite import (
    Process, Step, Composite, TimingSummary, interval_time_precision, wire_step_layers,
)
from process_bigraph.bundle import save_bundle, load_bundle
from process_bigraph.emitter import Emitter, gather_emitter_results, generate_emitter_state
from process_bigraph.types import StepLink, ProcessLink, CompositeLink


import pprint
pretty = pprint.PrettyPrinter(indent=2)

def pp(x):
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)

def pf(x):
    """Format ``x`` for display."""
    return pretty.pformat(x)


def register_types(core):
    core.register_type('interval', {
        '_inherit': 'float'})

    core.register_type('default 1', {
        '_inherit': 'float',
        '_default': 1.0})

    core.register_type('ode_config', {
        'stoichiometry': {
            '_type': 'array',
            '_data': 'int64'},
        'rates': 'map[float]',
        'species': 'map[float]'})

    return core

