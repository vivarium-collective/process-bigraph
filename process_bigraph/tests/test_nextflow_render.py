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


from process_bigraph.nextflow import _process_block
from process_bigraph.composite import Step


class _OutStep(Step):
    """A plain Step with a scalar output port and no nextflow_script/port_decls."""
    def inputs(self):
        return {'seed': 'integer'}

    def outputs(self):
        return {'value': 'integer'}

    def update(self, state):
        return {'value': int(state.get('seed', 0)) + 1}


def test_plain_step_output_decl_matches_run_step_file():
    from process_bigraph import allocate_core
    inst = _OutStep({}, core=allocate_core())
    block = _process_block('outstep', inst,
                           inputs_wires={'seed': ['seed']},
                           outputs_wires={'value': ['value']})
    # run_step writes --out value=value.json, so the nextflow output decl
    # MUST capture that exact file, not an unbound `val value`.
    assert 'path "value.json"' in block
    assert 'val value' not in block
    # sanity: the script body still writes value.json
    assert '--out value=value.json' in block


# --- nested Composite -> Nextflow sub-workflow -------------------------------

class _Unit(Step):
    """A trivial unit; N of these stand in for an unrolled fan-out."""
    nextflow_port_decls = {'cache': 'path cache', 'o': 'path "out"'}

    def inputs(self):
        return {'cache': {'_type': 'string', '_is_file': True}}

    def outputs(self):
        return {'o': {'_type': 'string', '_is_file': True}}

    def nextflow_script(self):
        return '"""\nmkdir -p out\n"""'

    def update(self, state):
        return {'o': 'out'}


def _nested(n):
    """An outer scope holding one nested Composite of n units."""
    core = allocate_core()
    core.register_link('_Unit', _Unit)
    inner_state = {'cache': ''}
    for i in range(n):
        inner_state[f'sweep_{i}'] = ''
        inner_state[f'unit_{i}'] = {
            '_type': 'step', 'address': 'local:_Unit', 'config': {},
            'inputs': {'cache': ['cache']}, 'outputs': {'o': [f'sweep_{i}']}}
    inner = Composite({'state': inner_state}, core=core)
    return SimpleNamespace(
        step_paths={}, step_dependencies={}, node_dependencies={},
        process_paths={('runs',): {
            'instance': inner,
            'inputs': {'cache': ['cache_store']},
            'outputs': {'results': ['results_store']}}},
        bridge={})


def test_nested_composite_renders_as_subworkflow():
    nf = render_composite(_nested(3))
    assert 'workflow runs {' in nf
    assert 'take:' in nf and 'emit:' in nf
    # the hierarchy is preserved: inner units are their own processes
    assert 'process unit_0 {' in nf and 'process unit_2 {' in nf
    # ...and the parent sees ONE channel, not three
    assert 'runs(' in nf


def test_subworkflow_take_port_is_a_bare_identifier():
    """Inside a sub-workflow a take: port is in scope by name, not params.<x>."""
    nf = render_composite(_nested(2))
    start = nf.index('workflow runs {')
    body = nf[start:nf.index('workflow', start + 1)]   # sub-workflow block ONLY
    assert 'unit_0(cache)' in body
    assert 'params.' not in body


def test_gather_uses_chained_binary_mix_not_nary_call():
    """`a.mix(b, c, ... )` is a Java method call and dies at 255 parameters;
    chained binary mixes are N statements of arity 1 and do not."""
    nf = render_composite(_nested(300))
    body = nf[nf.index('workflow runs {'):]
    assert '_merged.mix(' in body
    # no single mix() call with more than one argument
    for line in body.splitlines():
        if '.mix(' in line:
            assert line.count(',') == 0, f'n-ary mix would hit the 255 limit: {line[:80]}'
    assert '.collect()' in body


def test_composite_node_ordered_between_its_producer_and_consumer():
    """The dependency runs THROUGH the composite node, so a Steps-only sort
    cannot order the two ends; _unified_order must."""
    core = allocate_core()
    core.register_link('_Unit', _Unit)
    inner = Composite({'state': {
        'cache': '', 'sweep_0': '',
        'unit_0': {'_type': 'step', 'address': 'local:_Unit', 'config': {},
                   'inputs': {'cache': ['cache']}, 'outputs': {'o': ['sweep_0']}}}},
        core=core)

    class _Src(Step):
        nextflow_port_decls = {'c': 'path "cache"'}
        def inputs(self): return {}
        def outputs(self): return {'c': {'_type': 'string', '_is_file': True}}
        def nextflow_script(self): return '"""\nmkdir -p cache\n"""'
        def update(self, s): return {'c': 'cache'}

    class _Sink(Step):
        nextflow_port_decls = {'r': 'path r', 'out': 'path "done.txt"'}
        def inputs(self): return {'r': {'_type': 'list', '_is_file': True}}
        def outputs(self): return {'out': {'_type': 'string', '_is_file': True}}
        def nextflow_script(self): return '"""\ntouch done.txt\n"""'
        def update(self, s): return {'out': 'done.txt'}

    outer = SimpleNamespace(
        step_paths={
            ('src',): {'instance': _Src({}, core=core), 'inputs': {},
                       'outputs': {'c': ['cache_store']}},
            ('sink',): {'instance': _Sink({}, core=core),
                        'inputs': {'r': ['results_store']},
                        'outputs': {'out': ['done']}}},
        step_dependencies={('src',): {'output_paths': [['cache_store']]},
                           ('sink',): {'output_paths': [['done']]}},
        node_dependencies={},
        process_paths={('runs',): {'instance': inner,
                                   'inputs': {'cache': ['cache_store']},
                                   'outputs': {'results': ['results_store']}}},
        bridge={})
    nf = render_composite(outer, {'workflow_name': ''})
    body = nf[nf.rindex('workflow'):]
    lines = [l.strip() for l in body.splitlines() if '(' in l]
    pos = {k: i for i, l in enumerate(lines)
           for k in ('src(', 'runs(', 'sink(') if k in l}
    assert pos['src('] < pos['runs('] < pos['sink('], body


def test_per_node_config_is_threaded_into_run_step():
    """An unrolled sweep differs only by config. If it is not passed to
    run_step, every node runs with DEFAULTS -- N copies of the same run, with
    N tasks, N work dirs and N outputs all looking correct."""
    class _Cfg(Step):
        config_schema = {'seed': {'_type': 'integer', '_default': 0}}
        def inputs(self): return {}
        def outputs(self): return {'o': {'_type': 'string', '_is_file': True}}
        def update(self, state): return {'o': 'out'}

    core = allocate_core()
    core.register_link('_Cfg', _Cfg)
    state = {}
    for i in range(3):
        state[f'out_{i}'] = ''
        state[f'node_{i}'] = {'_type': 'step', 'address': 'local:_Cfg',
                              'config': {'seed': i},
                              'inputs': {}, 'outputs': {'o': [f'out_{i}']}}
    comp = Composite({'state': state}, core=core)

    opts = {'workflow_name': 'w'}
    nf = render_composite(comp, opts)
    assert '--config node_0.config.json' in nf

    staged = opts['_staged_configs']
    assert len(staged) == 3
    seeds = sorted(c['seed'] for c in staged.values())
    assert seeds == [0, 1, 2], f'seeds collapsed: {seeds}'
