from process_bigraph.types.process import Process, Step, deserialize


def register_types(core):
    process_types = {
        'process': Process,
        'step': Step}

    core.register_types(process_types)

    return core
