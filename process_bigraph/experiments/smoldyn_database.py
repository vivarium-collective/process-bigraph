"""A test of the `..emitter.DatabaseEmitter` class with Smoldyn to emit and query the formatted results of
    a Smoldyn simulation. The db workflow could be described as:

        0. A local MongoDB server is spun up.  # TODO: Automatically check for this and kill/start as needed.
        1. A Smoldyn simulation object is created by reading in a Smoldyn model.txt file.
        2. Smoldyn "datasets" (output commands) are added using the Python API despite what `cmd` lines are
            declared in the Smoldyn model file. (`addOutputData` && `addCommand`)
        3. The Smoldyn simulation runs for the specified time params dependent on which simulation method is used.
        4. The raw data is retrieved and formatted (`getOutputData` for time and listmols)
        5. An instance of DatabaseEmitter is created.
        6. The simulation data from #4's output is used as input for the database emitter and emitted to memory.
        7. This process is repeated `N` times.
        8. The user queries the vast database based on simulation run id.

    Currently, this emitter works with MongoDB. TODO: more include dbTypes
"""


