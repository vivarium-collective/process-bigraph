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
