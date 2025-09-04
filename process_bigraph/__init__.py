import pprint
from bigraph_schema.registry import deep_merge, default
from process_bigraph.processes import register_processes
from process_bigraph.composite import Process, Step, Composite, interval_time_precision
from process_bigraph.emitter import Emitter, gather_emitter_results, generate_emitter_state, BASE_EMITTERS
from process_bigraph.process_types import ProcessTypes
from process_bigraph.package.discover import discover_packages

pretty = pprint.PrettyPrinter(indent=2)


def pp(x):
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x):
    """Format ``x`` for display."""
    return pretty.pformat(x)


def register_types(core):
    core.register('default 1', {
        '_inherit': 'float',
        '_default': 1.0})

    core.register('species_dependent_process', {
        '_inherit': ['process'],
        '_inputs': {
            'species': {
                '_type': 'array',
                '_data': 'float'}},
        '_outputs': {
            'species': {
                '_type': 'array',
                '_data': 'float'}}})

    core.register('ode_config', {
        'stoichiometry': {
            '_type': 'array',
            '_data': 'integer'},
        'rates': 'map[float]',
        'species': 'map[float]'})

    register_processes(
        core)

    return core


def allocate_core():
    core = ProcessTypes()
    return register_types(core)
