from bigraph_schema import allocate_core  # noqa: F401

# The installed distribution's version, exposed so a run manifest can record
# *which* engine produced an artifact. Read from installed metadata rather
# than hardcoded, so the two can never disagree — and note the trap that
# makes it worth a test: an editable install freezes this at install time, so
# every later commit reports the version that was in `pyproject.toml` when
# `pip install -e` ran. `tests.py::test_version_matches_pyproject` fails
# loudly when they drift instead of letting a manifest lie.
try:
    from importlib.metadata import PackageNotFoundError, version as _version
    try:
        __version__ = _version('process-bigraph')
    except PackageNotFoundError:      # not installed (running from a tree)
        __version__ = '0+unknown'
except ImportError:                   # pragma: no cover — py<3.8
    __version__ = '0+unknown'

from process_bigraph.composite import (  # noqa: F401
    Process, Step, Composite, TimingSummary, interval_time_precision, wire_step_layers,
)
from process_bigraph.composite_spec import (  # noqa: F401
    CompositeSpec, discover_specs, regenerate_default_state,
    normalize_type, CANONICAL_TYPES,
)
from process_bigraph.bundle import load_bundle  # noqa: F401
from process_bigraph.draft_process import DraftProcess, draft_process  # noqa: F401
from process_bigraph.emitter import Emitter, gather_emitter_results, generate_emitter_state  # noqa: F401
from process_bigraph.types import StepLink, ProcessLink, CompositeLink  # noqa: F401
from process_bigraph import core_introspection  # noqa: F401
# NOTE: do NOT re-export the `composite_generator` decorator here — that name
# is the submodule `process_bigraph.composite_generator`, and exporting the
# same-named function shadows it (breaks `import process_bigraph.composite_generator`).
# Get the decorator via `from process_bigraph.composite_generator import composite_generator`.
from process_bigraph.composite_generator import (  # noqa: F401
    discover_generators, build_generator,
    install_default_emitters, emitter_defaults,
)
from process_bigraph.composite_discovery import discover_all  # noqa: F401
from process_bigraph.config_helpers import normalize_config_list  # noqa: F401
from process_bigraph.visualization import Visualization, as_visualization, render_results  # noqa: F401


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

