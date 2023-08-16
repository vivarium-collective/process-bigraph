"""
================================
Registry of Processes, Protocols
================================

---------
Processes
---------
TODO

---------
Protocols
---------
TODO
"""
from bigraph_schema.registry import Registry

# Initialize registries
# These are imported into module __init__.py files,
# where the functions and classes are registered upon import

#: Maps process names to :term:`process classes`
process_registry = Registry()

#: Maps process names to :term:`protocol methods`
protocol_registry = Registry()
