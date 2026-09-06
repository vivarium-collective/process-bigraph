from process_bigraph.composite_generator import composite_generator, _REGISTRY


def test_decorator_carries_analyses():
    @composite_generator(
        name="probe_gen",
        analyses=[{"name": "ptools_rna_multigeneration"}],
        visualizations=[{"name": "v", "address": "local:X"}],
    )
    def build(core=None, **kw):
        return {}
    entry = _REGISTRY[f"{build.__module__}.probe_gen"]
    assert entry.analyses == [{"name": "ptools_rna_multigeneration"}]
    assert entry.visualizations  # unchanged path still works
