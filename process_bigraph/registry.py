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

# decorator to register processes
def register_process(name, registry=process_registry):
    """Register a process with the process registry.
    
    :param name: The name of the process.
    
    :Example:
        @register_process('my_process')
        class MyProcess(Process):
            ...
    """
    def decorator(func):
        registry.register(name, func)
        return func

    return decorator
