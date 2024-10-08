import pprint
from bigraph_schema.registry import deep_merge, default
from process_bigraph.composite import Process, Step, Composite, ProcessTypes, interval_time_precision


pretty = pprint.PrettyPrinter(indent=2)


def pp(x):
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x):
    """Format ``x`` for display."""
    return pretty.pformat(x)
