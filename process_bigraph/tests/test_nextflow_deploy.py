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
import subprocess
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


_AWS_PARAMS = {
    'container_image': '000.dkr.ecr.us-gov-west-1.amazonaws.com/img:abc123',
    'queue': 'a-spot-first-queue',
    'aws_region': 'us-gov-west-1',
    'project_root': '/app/v2ecoli',
}


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
    deploy(_emit_composite(), outdir=str(batch_dir), executor='awsbatch', launch=False,
           params=_AWS_PARAMS)
    assert sys.executable not in (batch_dir / 'main.nf').read_text()


def test_explicit_python_option_still_wins_on_any_executor(tmp_path):
    out = tmp_path / 'explicit'
    deploy(_emit_composite(), outdir=str(out), executor='awsbatch', launch=False,
           options={'python': '/opt/venv/bin/python'}, params=_AWS_PARAMS)
    assert '/opt/venv/bin/python' in (out / 'main.nf').read_text()


# --- the awsbatch profile (was `// STUB (untested in v1)`) ---


def test_awsbatch_profile_emits_queue_container_region_and_retry():
    cfg = generate_nextflow_config(
        executor='awsbatch',
        resources={'lineage': {'cpus': 2, 'memory': '8 GB', 'time': '4h'}},
        params=_AWS_PARAMS)
    assert '// STUB' not in cfg.split("'google-batch'")[0]
    assert 'container = params.container_image' in cfg
    assert 'queue = params.queue' in cfg
    assert 'region = params.aws_region' in cfg
    # Both retry mechanisms, which are independent and both default to "none".
    assert 'maxSpotAttempts = 10' in cfg
    assert 'maxTransferAttempts = 10' in cfg
    assert 'maxRetries = 3' in cfg
    assert "'retry' : 'finish'" in cfg
    # Per-label `time` is the only bound on a runaway task; it reaches Batch as
    # attemptDurationSeconds.
    assert "time = '4h'" in cfg
    # jobRole and cliPath are deliberately unset -- see _awsbatch_profile.
    assert 'jobRole' not in cfg
    assert 'cliPath' not in cfg


def test_awsbatch_requires_the_params_it_cannot_invent():
    """Rendering with a null queue/container is the silent failure this guards.

    Nextflow reads a missing param as null and writes a perfectly valid config
    around it; the mistake surfaces only at submission, on real infrastructure.
    """
    import pytest
    with pytest.raises(ValueError) as exc:
        generate_nextflow_config(executor='awsbatch', params={'aws_region': 'us-gov-west-1'})
    assert 'container_image' in str(exc.value)
    assert 'queue' in str(exc.value)
    assert 'aws_region' not in str(exc.value)


def test_local_executor_does_not_require_aws_params():
    """Every profile is always emitted, so the local path must not depend on AWS
    settings it never uses."""
    cfg = generate_nextflow_config(executor='local')
    assert "executor = 'local'" in cfg
    assert 'awsbatch {' in cfg


def test_pythonpath_is_emitted_only_when_a_project_root_is_declared():
    """A null PYTHONPATH is worse than none -- it shadows the container's own."""
    with_root = generate_nextflow_config(executor='awsbatch', params=_AWS_PARAMS)
    assert '--env PYTHONPATH=${params.project_root}' in with_root

    without = dict(_AWS_PARAMS)
    without.pop('project_root')
    cfg = generate_nextflow_config(executor='awsbatch', params=without)
    assert 'PYTHONPATH' not in cfg
    assert '--env AWS_DEFAULT_REGION=${params.aws_region}' in cfg


def test_s3_endpoint_and_work_dir_are_optional():
    plain = generate_nextflow_config(executor='awsbatch', params=_AWS_PARAMS)
    assert 'endpoint' not in plain
    assert 'workDir' not in plain

    full = generate_nextflow_config(executor='awsbatch', params=dict(
        _AWS_PARAMS, s3_endpoint='https://s3.us-gov-west-1.amazonaws.com',
        work_dir='s3://bucket/nf/eid/work'))
    assert 'client { endpoint = params.s3_endpoint }' in full
    assert 'workDir = params.work_dir' in full


def test_awsbatch_retry_knobs_are_overridable():
    cfg = generate_nextflow_config(executor='awsbatch', params=dict(
        _AWS_PARAMS, max_spot_attempts=3, max_transfer_attempts=1, max_retries=7))
    assert 'maxSpotAttempts = 3' in cfg
    assert 'maxTransferAttempts = 1' in cfg
    assert 'maxRetries = 7' in cfg


@pytest.mark.skipif(shutil.which('nextflow') is None, reason='nextflow binary not on PATH')
def test_awsbatch_profile_parses_and_resolves_under_real_nextflow(tmp_path):
    """The only check that catches a Groovy-level mistake.

    Asserting that the config *text* contains `queue = params.queue` proves
    nothing about whether Nextflow resolves it -- an unparseable block, or a
    param that never interpolates, passes every string assertion. This runs
    `nextflow config -profile awsbatch` and asserts the RESOLVED values.
    """
    (tmp_path / 'main.nf').write_text('workflow { }\n')
    (tmp_path / 'nextflow.config').write_text(generate_nextflow_config(
        executor='awsbatch',
        resources={'lineage': {'cpus': 2, 'time': '4h'}},
        params=dict(_AWS_PARAMS,
                    s3_endpoint='https://s3.us-gov-west-1.amazonaws.com',
                    work_dir='s3://bucket/nf/eid/work')))

    proc = subprocess.run(['nextflow', 'config', '-profile', 'awsbatch', '.'],
                          cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    resolved = proc.stdout

    assert "executor = 'awsbatch'" in resolved
    assert "queue = 'a-spot-first-queue'" in resolved
    assert "container = '000.dkr.ecr.us-gov-west-1.amazonaws.com/img:abc123'" in resolved
    # Interpolated, not left as a literal ${...}.
    assert 'AWS_DEFAULT_REGION=us-gov-west-1' in resolved
    assert 'PYTHONPATH=/app/v2ecoli' in resolved
    assert '${' not in resolved
    assert "endpoint = 'https://s3.us-gov-west-1.amazonaws.com'" in resolved
    assert "workDir = 's3://bucket/nf/eid/work'" in resolved
    assert 'maxSpotAttempts = 10' in resolved
