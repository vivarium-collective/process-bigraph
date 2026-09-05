from process_bigraph.nextflow_deploy import generate_nextflow_config


def test_config_has_requested_profiles_and_resources():
    cfg = generate_nextflow_config(
        executor='slurm',
        resources={'sim': {'cpus': 4, 'memory': '8 GB', 'time': '2h'}},
        params={'publishDir': 'results'})
    assert 'profiles {' in cfg
    assert 'local {' in cfg
    assert 'slurm {' in cfg
    assert "executor = 'slurm'" in cfg
    assert 'withLabel: sim' in cfg
    assert 'cpus = 4' in cfg
    assert "publishDir = 'results'" in cfg


def test_config_default_executor_local():
    cfg = generate_nextflow_config()
    assert 'local {' in cfg
    assert "executor = 'local'" in cfg


def test_config_params_render_valid_groovy_scalars():
    cfg = generate_nextflow_config(params={'spot': True, 'off': False, 'x': None, 'n': 3, 's': 'hi'})
    assert 'spot = true' in cfg
    assert 'off = false' in cfg
    assert 'x = null' in cfg
    assert 'n = 3' in cfg
    assert "s = 'hi'" in cfg
    assert 'True' not in cfg and 'None' not in cfg


import shutil
import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Step
from process_bigraph.nextflow_deploy import deploy


class _EmitStep(Step):
    """Writes a constant to its output store when it fires.

    The output port is declared ``integer`` at the process-bigraph level
    (the real, semantic type of the value ``update()`` returns) but
    carries a ``nextflow_port_decls`` override so the *rendered* Nextflow
    process declares a quoted, literal ``path "value.json"`` output —
    matching the fixed ``<port>.json`` filename that the auto-generated
    ``run_step --out value=value.json`` script actually writes. Without
    the override, the renderer's default declaration for an output-only
    port (``val value`` for a scalar, or bare ``path value`` for a path
    type) refers to a Groovy variable/glob named exactly ``value`` that
    is never bound anywhere in the process scope — Nextflow fails at
    task time with "Missing value declared as output parameter" (val) or
    "Missing output file(s) `value`" (bare path, which literally globs
    for a file named ``value`` with no extension). The override is the
    escape hatch nextflow.py's ``_port_to_nextflow_decl`` documents for
    exactly this situation.
    """
    nextflow_port_decls = {'value': 'path "value.json"'}

    def inputs(self):
        return {'seed': 'integer'}

    def outputs(self):
        return {'value': 'integer'}

    def update(self, state):
        return {'value': int(state.get('seed', 0)) + 1}


def _emit_core():
    core = allocate_core()
    core.register_link('_EmitStep', _EmitStep)
    return core


def _emit_composite():
    state = {
        'seed': 3,
        'emit': {
            '_type': 'step',
            'address': 'local:_EmitStep',
            'config': {},
            'inputs': {'seed': ['seed']},
            'outputs': {'value': ['value']},
        },
        'value': 0,
    }
    return Composite({'state': state}, core=_emit_core())


def test_deploy_writes_files(tmp_path):
    composite = _emit_composite()
    result = deploy(composite, outdir=str(tmp_path), executor='local', launch=False)
    assert (tmp_path / 'main.nf').exists()
    assert (tmp_path / 'nextflow.config').exists()
    assert result['returncode'] is None
    # main.nf pins this interpreter for task subprocesses.
    import sys
    assert sys.executable in (tmp_path / 'main.nf').read_text()


@pytest.mark.skipif(shutil.which('nextflow') is None,
                    reason='nextflow binary not on PATH')
def test_deploy_launch_local_end_to_end(tmp_path):
    composite = _emit_composite()
    # No `options` override here on purpose: this proves the *shipped
    # default* `deploy(..., launch=True)` path works end-to-end. deploy()
    # itself defaults render_options['workflow_name'] to '' (an
    # unnamed/implicit entry workflow), sidestepping the fact that
    # render_composite's own default ('main') is a reserved identifier in
    # real Nextflow (naming an explicit workflow block `main` is a
    # compile error). A caller who never heard of that trap must still
    # get a working deploy.
    result = deploy(composite, outdir=str(tmp_path), executor='local',
                    launch=True, params={'seed': 3},
                    work_dir=str(tmp_path / 'work'))
    assert result['returncode'] == 0


# --- launch flags: -resume is the reason to run a campaign under a DAG engine ---


def _captured_launch(tmp_path, monkeypatch, **kwargs):
    """Run deploy(launch=True) with nextflow stubbed, and return the argv it built."""
    import subprocess as _sp
    from process_bigraph import nextflow_deploy as nd

    seen = {}

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        return _sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(nd.shutil, 'which', lambda _: '/usr/bin/nextflow')
    monkeypatch.setattr(nd.subprocess, 'run', fake_run)
    deploy(_emit_composite(), outdir=str(tmp_path), executor='local',
           launch=True, **kwargs)
    return seen['cmd']


def test_resume_is_emitted_only_when_asked(tmp_path, monkeypatch):
    """go/no-go 3 of the Nextflow dispatch plan is '-resume re-runs only the failed
    lineage'. deploy() could not pass -resume at all, so that gate was untestable."""
    assert '-resume' not in _captured_launch(tmp_path, monkeypatch)
    assert '-resume' in _captured_launch(tmp_path, monkeypatch, resume=True)


def test_report_trace_and_weblog_are_passed_through(tmp_path, monkeypatch):
    """The trace CSV is how a resumed run is told apart from a repeated one: a reused
    task reports CACHED there and nowhere else."""
    cmd = _captured_launch(
        tmp_path, monkeypatch,
        report=tmp_path / 'r.html', trace=tmp_path / 't.csv',
        weblog_url='http://receiver/events')
    assert cmd[cmd.index('-with-report') + 1] == str(tmp_path / 'r.html')
    assert cmd[cmd.index('-with-trace') + 1] == str(tmp_path / 't.csv')
    assert cmd[cmd.index('-with-weblog') + 1] == 'http://receiver/events'


def test_nextflow_args_appended_verbatim_and_never_shell_split(tmp_path, monkeypatch):
    cmd = _captured_launch(tmp_path, monkeypatch, nextflow_args=['-queue-size', '50'])
    assert cmd[-2:] == ['-queue-size', '50']
    with pytest.raises(TypeError):
        _captured_launch(tmp_path, monkeypatch, nextflow_args='-queue-size 50')


# --- the interpreter default is executor-scoped (a latent AWS Batch bug) ---


def test_python_pinned_for_local_but_not_for_other_executors(tmp_path):
    """sys.executable is the HEAD's interpreter path. On the local executor that is
    also the tasks' interpreter. On awsbatch the task runs in a container where that
    path need not exist, and baking it in emits a command that cannot run."""
    import sys
    composite = _emit_composite()

    local_dir = tmp_path / 'local'
    deploy(composite, outdir=str(local_dir), executor='local', launch=False)
    assert sys.executable in (local_dir / 'main.nf').read_text()

    batch_dir = tmp_path / 'batch'
    deploy(_emit_composite(), outdir=str(batch_dir), executor='awsbatch', launch=False)
    assert sys.executable not in (batch_dir / 'main.nf').read_text()


def test_explicit_python_option_still_wins_on_any_executor(tmp_path):
    out = tmp_path / 'explicit'
    deploy(_emit_composite(), outdir=str(out), executor='awsbatch', launch=False,
           options={'python': '/opt/venv/bin/python'})
    assert '/opt/venv/bin/python' in (out / 'main.nf').read_text()
