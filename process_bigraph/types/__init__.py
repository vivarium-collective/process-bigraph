from process_bigraph.types.process import ProcessLink, StepLink, deserialize


def register_types(core):
    process_types = {
        'process': ProcessLink,
        'step': StepLink}

    core.register_types(process_types)

    return core
