import pprint
from process_bigraph.composite import Process, Step, Composite, ProcessTypes


__all__ = [
    'Process',
    'Step',
    'Composite',
    'pp',
    'pf',
]


pretty = pprint.PrettyPrinter(indent=2)


def pp(x):
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x):
    """Format ``x`` for display."""
    return pretty.pformat(x)
