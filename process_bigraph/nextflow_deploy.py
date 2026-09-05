"""Generate nextflow.config + deploy a Composite's Step network to a backend.

Wraps process_bigraph.nextflow.render_composite (which emits main.nf) with a
nextflow.config profile block and an optional `nextflow run` launch. The
executor abstraction mirrors vEcoli's runscripts/nextflow/config.template:
one `profiles { }` block, backend selected by name.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _resource_lines(resources: Optional[Dict[str, Dict[str, Any]]]) -> str:
    if not resources:
        return ''
    blocks = []
    for label, res in resources.items():
        lines = [f'            withLabel: {label} {{']
        if 'cpus' in res:
            lines.append(f'                cpus = {res["cpus"]}')
        if 'memory' in res:
            lines.append(f'                memory = {res["memory"]!r}')
        if 'time' in res:
            lines.append(f'                time = {res["time"]!r}')
        lines.append('            }')
        blocks.append('\n'.join(lines))
    return '\n'.join(blocks)


def _params_block(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ''
    lines = ['params {']
    for key, value in params.items():
        # Dispatch on type to render valid Groovy (check bool before int, since bool is int subclass)
        if isinstance(value, bool):
            groovy_value = 'true' if value else 'false'
        elif value is None:
            groovy_value = 'null'
        else:
            # str, int, float, etc. — repr() is correct
            groovy_value = repr(value)
        lines.append(f'    {key} = {groovy_value}')
    lines.append('}')
    return '\n'.join(lines) + '\n\n'


# The AWS Batch profile is params-driven, exactly like vEcoli's proven
# `config.template`: this module must not know about any one deployment's
# queue names, image registry or region. Callers supply them via `params`.
AWSBATCH_REQUIRED_PARAMS = ('container_image', 'queue', 'aws_region')

# Defaults for the two independent retry mechanisms. Both matter, and both are
# absent from a bare `executor = 'awsbatch'`:
#
#   aws.batch.maxSpotAttempts   Batch retries the SAME job on a new instance,
#                               and ONLY for a spot reclaim -- nf-amazon pins
#                               EvaluateOnExit to RETRY on `Host EC2*`, EXIT on
#                               `*`. Its default is 0, i.e. no retry at all, and
#                               a spot-first queue will eventually reclaim a
#                               long-running task.
#   errorStrategy + maxRetries  Nextflow resubmits as a NEW Batch job. This is
#                               the one that survives an OOM or a code fault,
#                               and the only reason "the outer DAG is the
#                               resubmitter" is true rather than aspirational.
#
# Nextflow's own default for errorStrategy is 'terminate', so without the second
# row a single failed task takes the whole campaign with it.
AWSBATCH_DEFAULTS = {
    'max_spot_attempts': 10,
    'max_transfer_attempts': 10,
    'max_retries': 3,
}


def _awsbatch_profile(res_block: str, params: Optional[Dict[str, Any]]) -> str:
    """Render the `awsbatch` profile body, modelled on vEcoli's production one.

    Deliberately NOT emitted here:

    * ``aws.batch.jobRole`` -- the submitting role's ``iam:PassRole`` is usually
      scoped to a handful of named roles, so any value we invented would fail at
      submission. Unset means the task runs as the compute environment's own
      instance profile, which is what already has the work-bucket grant.
    * ``aws.batch.cliPath`` -- unset relies on ``aws`` being on ``PATH`` inside
      the task container, which is how vEcoli runs and what the science image
      already provides.
    """
    params = params or {}
    missing = [k for k in AWSBATCH_REQUIRED_PARAMS if params.get(k) in (None, '')]
    if missing:
        raise ValueError(
            "the awsbatch profile needs params " + ', '.join(missing) + " -- without them the "
            "profile still renders, and Nextflow reads them as null. Pass them in `params`.")

    opts = dict(AWSBATCH_DEFAULTS)
    for key in opts:
        if params.get(key) is not None:
            opts[key] = params[key]

    # PYTHONPATH is separate from the region env because it is not an AWS
    # concern: `scratch true` (and Batch's own working directory) moves the
    # task's cwd off the image's project root, so a repo that relies on
    # cwd-relative imports stops importing. Emitted only when the caller
    # declares a project root -- a null PYTHONPATH would be worse than none.
    env_opts = '--env AWS_DEFAULT_REGION=${params.aws_region}'
    if params.get('project_root'):
        env_opts += ' --env PYTHONPATH=${params.project_root}'

    # A GovCloud (or any non-standard-partition) S3 endpoint. Emitted only when
    # given, so the common commercial-partition case stays on the SDK default.
    endpoint_line = ''
    if params.get('s3_endpoint'):
        endpoint_line = '\n            client { endpoint = params.s3_endpoint }'

    work_dir_line = ''
    if params.get('work_dir'):
        work_dir_line = '\n        workDir = params.work_dir'

    return f"""    awsbatch {{
        process {{
            executor = 'awsbatch'
            container = params.container_image
            queue = params.queue
            // AWS_DEFAULT_REGION is required by the AWS CLI *inside* the task
            // container, which is what stages the S3 work dir in and out.
            containerOptions = "{env_opts}"
            // Retry, then stop scheduling new work and let running tasks drain.
            // 'finish' rather than vEcoli's 'ignore': ignore is safe there only
            // because it also sets workflow.failOnIgnore, and an ignored failed
            // task is a campaign that goes green having produced no science.
            errorStrategy = {{ task.attempt <= task.maxRetries ? 'retry' : 'finish' }}
            maxRetries = {opts['max_retries']}{res_block}
        }}
        aws {{
            region = params.aws_region{endpoint_line}
            batch {{
                // Spot preemption is retried by Batch itself; default is 0.
                maxSpotAttempts = {opts['max_spot_attempts']}
                maxTransferAttempts = {opts['max_transfer_attempts']}
            }}
        }}
        docker.enabled = true{work_dir_line}
    }}
"""


def generate_nextflow_config(executor: str = 'local',
                             resources: Optional[Dict[str, Dict[str, Any]]] = None,
                             params: Optional[Dict[str, Any]] = None) -> str:
    res = _resource_lines(resources)
    res_block = ('\n' + res) if res else ''
    # Only validate the profile actually being deployed: this function always
    # emits every profile, so requiring awsbatch params from a local run would
    # make the local path depend on AWS settings it never uses.
    awsbatch = (_awsbatch_profile(res_block, params) if executor == 'awsbatch'
                else "    awsbatch {\n        // not configured; pass awsbatch params to render it\n"
                     "        process { executor = 'awsbatch' }\n    }\n")
    return f"""{_params_block(params)}profiles {{
    local {{
        process {{
            executor = 'local'{res_block}
        }}
    }}
    slurm {{
        process {{
            executor = 'slurm'
            errorStrategy = {{ task.attempt <= 3 ? 'retry' : 'finish' }}{res_block}
        }}
        executor.queueSize = 100
        executor.submitRateLimit = '20/min'
    }}
{awsbatch}    'google-batch' {{
        // STUB (untested in v1)
        process {{ executor = 'google-batch' }}
    }}
}}
"""


def deploy(composite, *, outdir: str, executor: str = 'local',
           launch: bool = False, resources=None, params=None,
           options=None, work_dir=None, resume: bool = False,
           report=None, trace=None, weblog_url=None,
           nextflow_args=None) -> Dict[str, Optional[str]]:
    """Write ``main.nf`` + ``nextflow.config`` for a Composite, optionally launch it.

    Renders the Step network via ``render_composite`` and writes a matching
    ``nextflow.config`` via ``generate_nextflow_config``. When ``launch=True``,
    shells out to ``nextflow -C <config> run <main.nf> -profile <executor>``
    and raises ``subprocess.CalledProcessError`` on a non-zero exit.

    ``resume`` adds ``-resume``, which is the entire point of running a campaign
    under a DAG engine: on a re-invocation Nextflow reuses cached *successful*
    tasks and re-runs only what changed or failed. Without it a re-run repeats
    every ParCa and every completed lineage. Note this is distinct from in-run
    retry, which is ``errorStrategy``/``maxRetries`` in the profile.

    ``report`` / ``trace`` add ``-with-report`` / ``-with-trace``. The trace CSV
    is how you tell a resumed run apart from a repeated one -- a reused task
    shows CACHED there and nowhere else.

    ``weblog_url`` adds ``-with-weblog``, which POSTs task-level events to a
    receiver, so a long campaign is observable without tailing a head log.

    ``nextflow_args`` is an escape hatch appended verbatim; it is a list, never
    a string, so nothing is shell-split or shell-interpreted.

    **The interpreter default is executor-scoped, deliberately.**
    ``sys.executable`` is the path of the *head's* Python. On the local executor
    that is also the tasks' Python, so pinning it is right. On any other
    executor tasks run in a container or on another host where that path need
    not exist, and baking it in produces a command that cannot run -- a failure
    that surfaces only once real infrastructure is involved. Callers that do
    want an explicit interpreter pass ``options={'python': ...}``, which always
    wins.
    """
    from process_bigraph.nextflow import render_composite

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    render_options = dict(options or {})
    if executor == 'local':
        render_options.setdefault('python', sys.executable)
    # render_composite's own default ('main') is a reserved identifier in
    # real Nextflow — naming the entry workflow block `main` is a compile
    # error. Default to an unnamed/implicit entry workflow instead, which
    # Nextflow runs without needing `-entry`. A caller-supplied
    # `workflow_name` (including '') in `options` always wins.
    render_options.setdefault('workflow_name', '')

    main_nf = out / 'main.nf'
    main_nf.write_text(render_composite(composite, render_options))

    # Write each Step's own config beside main.nf. Without these the emitted
    # `run_step --config <name>.config.json` has nothing to read, and every
    # unrolled node silently runs with DEFAULTS -- an N-way parameter sweep
    # collapsing into N copies of the same run, with every structural check
    # (N tasks, N work dirs, N outputs) still passing.
    import json as _json
    for _name, _cfg in (render_options.get('_staged_configs') or {}).items():
        (out / _name).write_text(_json.dumps(_cfg, indent=2, default=str))

    config = out / 'nextflow.config'
    config.write_text(generate_nextflow_config(
        executor=executor, resources=resources, params=params))

    returncode: Optional[int] = None
    if launch:
        if shutil.which('nextflow') is None:
            raise RuntimeError('nextflow binary not found on PATH')
        cmd = ['nextflow', '-C', str(config), 'run', str(main_nf),
               '-profile', executor]
        if work_dir is not None:
            cmd += ['-work-dir', str(work_dir)]
        if resume:
            cmd += ['-resume']
        if report is not None:
            cmd += ['-with-report', str(report)]
        if trace is not None:
            cmd += ['-with-trace', str(trace)]
        if weblog_url is not None:
            cmd += ['-with-weblog', str(weblog_url)]
        if nextflow_args:
            if isinstance(nextflow_args, str):
                raise TypeError(
                    'nextflow_args must be a sequence of arguments, not a string -- '
                    'a string would have to be shell-split, and quoting is exactly '
                    'where that goes wrong silently')
            cmd += [str(a) for a in nextflow_args]
        proc = subprocess.run(cmd, cwd=str(out))
        returncode = proc.returncode
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    return {'main_nf': str(main_nf), 'config': str(config),
            'returncode': returncode}
