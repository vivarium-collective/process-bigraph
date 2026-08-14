from process_bigraph.nextflow import _topological_order


def test_topological_order_respects_nested_store_edges():
    # 'zebra' writes ('shared',); 'alpha' reads the nested ('shared','x').
    # Names chosen so alphabetical (Kahn tie-break) order CONTRADICTS the
    # dependency — only real edge inference puts zebra before alpha.
    step_paths = {('zebra',): {}, ('alpha',): {}}
    step_dependencies = {
        ('zebra',): {'input_paths': [], 'output_paths': [['shared']]},
        ('alpha',): {'input_paths': [['shared', 'x']], 'output_paths': []},
    }
    node_dependencies = {
        ('shared',): {'before': {('zebra',)}, 'after': set()},
        # build_step_network's prefix propagation puts zebra in before(shared/x)
        ('shared', 'x'): {'before': {('zebra',)}, 'after': {('alpha',)}},
    }
    order = _topological_order(step_paths, step_dependencies, node_dependencies)
    assert order.index(('zebra',)) < order.index(('alpha',))


from types import SimpleNamespace
from process_bigraph import Composite, allocate_core
from process_bigraph.nextflow import _composite_node_script, render_composite


def test_composite_node_script_emits_run_composite():
    script = _composite_node_script(
        instance=None, doc_ref='sim_document.json', steps=1000,
        inputs_wires={'init': ['init_store']},
        outputs_wires={'results': ['results_store']},
        python='python')
    assert 'run_composite' in script
    assert '--document sim_document.json' in script
    assert '--steps 1000' in script


def test_render_composite_emits_composite_node():
    # An outer network with one Composite node (no steps). Fake the outer
    # composite's attributes the renderer reads; the node instance is a real
    # (empty) Composite so the isinstance check fires.
    core = allocate_core()
    inner = Composite({'state': {}}, core=core)
    outer = SimpleNamespace(
        step_paths={},
        step_dependencies={},
        node_dependencies={},
        process_paths={('sim',): {
            'instance': inner,
            'inputs': {'init': ['init_store']},
            'outputs': {'results': ['results_store']},
        }},
        bridge={},
    )
    nf = render_composite(outer)
    assert 'process sim {' in nf
    assert 'run_composite' in nf
