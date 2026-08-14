import json
from process_bigraph.composite import Process
from process_bigraph.composite_generator import composite_generator


class _Ramp(Process):
    config_schema = {'rate': 'float'}

    def inputs(self):
        return {'level': 'float'}

    def outputs(self):
        return {'level': 'float'}

    def update(self, state, interval):
        return {'level': self.config['rate'] * interval}


def provision_ramp(core):
    core.register_link('_Ramp', _Ramp)
    return core


@composite_generator(name='ramp_toy', core_extensions=[provision_ramp])
def ramp_toy(rate=2.0, start=1.0, cache_dir=''):
    return {'state': {'level': start,
        'ramp': {'_type': 'process', 'address': 'local:_Ramp', 'config': {'rate': rate},
                 'inputs': {'level': ['level']}, 'outputs': {'level': ['level']}}}}


_IMP = ['process_bigraph.tests.test_run_composite_build']


def test_build_via_generator_and_extensions(tmp_path):
    b = tmp_path / 'b.json'
    b.write_text(json.dumps(
        {'build': {'generator': 'ramp_toy', 'import': _IMP, 'overrides': {'rate': 3.0}, 'provision': []},
         'run': {'steps': 4}}))
    out = tmp_path / 'f.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(b), steps=4.0, state_out_path=str(out))
    assert float(json.loads(out.read_text())['state']['level']) > 1.0


def test_build_set_override(tmp_path):
    b = tmp_path / 'b.json'
    b.write_text(json.dumps(
        {'build': {'generator': 'ramp_toy', 'import': _IMP, 'overrides': {}, 'provision': []},
         'run': {'steps': 0}}))
    out = tmp_path / 'f.json'
    from process_bigraph.run_composite import run_composite
    run_composite(build_path=str(b), steps=0.0, sets={'start': 41.0}, state_out_path=str(out))
    assert float(json.loads(out.read_text())['state']['level']) == 41.0
