from process_bigraph.composite_generator import composite_generator


@composite_generator(
    name="demo",
    description="A fake generator used only by tests.",
    parameters={"x": {"type": "int", "default": 7}},
)
def demo(core=None, *, x=7):
    return {"x": x}
