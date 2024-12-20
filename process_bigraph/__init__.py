import pprint
from process_bigraph.processes import register_processes
from process_bigraph.composite import Process, Step, Composite, ProcessTypes, interval_time_precision
from process_bigraph.process_types import process_types_registry


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

    core = register_processes(core)

    return core


# Make the core process types
process_types_core = ProcessTypes()
process_types_core = register_types(process_types_core)
process_types_registry.register('core', process_types_core)
