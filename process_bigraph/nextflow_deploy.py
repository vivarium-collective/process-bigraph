"""Generate nextflow.config + deploy a Composite's Step network to a backend.

Wraps process_bigraph.nextflow.render_composite (which emits main.nf) with a
nextflow.config profile block and an optional `nextflow run` launch. The
executor abstraction mirrors vEcoli's runscripts/nextflow/config.template:
one `profiles { }` block, backend selected by name.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _resource_value(value: Any) -> str:
    """Render one directive value: a Groovy CLOSURE raw, anything else quoted.

    A string starting with ``{`` is a closure and must be emitted verbatim --
    repr-quoting it turns `{ task.exitStatus == 137 ? 32.GB * task.attempt :
    32.GB }` into a string literal, and the process then asks Batch for a
    quantity of memory named "{ task.exitStatus ... }".

    This is what makes retry-with-more-memory expressible at all. Retrying an
    OOM with the SAME memory is three identical failures: measured, exit 137
    three times on one ParCa. vEcoli scales via its `scaledMemory` closure for
    exactly this reason.

    Same convention as ``_directive_lines`` in nextflow.py, deliberately -- two
    places that emit Groovy should not disagree about what a leading ``{`` means.
    """
    if isinstance(value, str) and value.lstrip().startswith('{'):
        return value
    return repr(value) if isinstance(value, str) else str(value)


def retry_error_strategy(exit_codes) -> str:
    """The Groovy ``errorStrategy`` closure that retries only ``exit_codes``
    (up to ``task.maxRetries``) and otherwise lets the campaign ``finish``.
    Shared by the awsbatch profile and by callers that want the same policy
    on another executor via a per-label ``resources`` entry."""
    codes = '[' + ', '.join(str(int(c)) for c in exit_codes) + ']'
    return ("{ (task.exitStatus in " + codes + ") && task.attempt <= task.maxRetries"
            " ? 'retry' : 'finish' }")


def _resource_lines(resources: Optional[Dict[str, Dict[str, Any]]]) -> str:
    if not resources:
        return ''
    blocks = []
    for label, res in resources.items():
        lines = [f'            withLabel: {label} {{']
        for key in ('cpus', 'memory', 'time'):
            if key in res:
                lines.append(f'                {key} = {_resource_value(res[key])}')
        # Per-label retry policy. ``maxRetries`` is an int; ``errorStrategy``
        # is rendered RAW when it is a Groovy closure (starts with ``{``) and
        # quoted otherwise ('ignore', 'terminate', ...).
        if 'maxRetries' in res:
            lines.append(f'                maxRetries = {int(res["maxRetries"])}')
        if 'errorStrategy' in res:
            strategy = str(res['errorStrategy']).strip()
            rendered = strategy if strategy.startswith('{') else _resource_value(strategy)
            lines.append(f'                errorStrategy = {rendered}')
        lines.append('            }')
        blocks.append('\n'.join(lines))
    return '\n'.join(blocks)


def _params_block(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ''
    lines = ['params {']
    for key, value in params.items():
        # Dispatch on type to render valid Groovy (check bool before int, since bool is int subclass)
        if isinstance(value, dict):
            # A Python dict repr is a Groovy CLOSURE, not a map -- `{'a': 1}`
            # fails to compile and takes the whole config with it, reported as a
            # column number in a generated script. Name the parameter instead.
            # (A list is fine: Python and Groovy list literals coincide.)
            raise ValueError(
                f'param {key!r} is a dict, which has no Groovy literal here -- Python renders it '
                f'as a closure and Nextflow fails to parse the config. Flatten it, or pass it '
                f'through a directive that takes structured values (e.g. `container_env`).')
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


_ENV_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
# Characters that would end the Groovy string, be interpolated by it, or split
# the docker argument. A value carrying one of these silently produces a broken
# `--env`, which is the failure this module keeps trying not to ship.
_ENV_VALUE_FORBIDDEN = set(' \t\n"\'$\\')


def _container_env_opts(container_env: Optional[Dict[str, Any]]) -> str:
    """Render caller-supplied container env vars as ``--env K=V`` fragments.

    This exists so the profile carries no opinion about what any one image
    needs. The motivating case is ``PYTHONPATH``: Nextflow moves a task's cwd
    off the image's project root, so an application whose imports resolve on cwd
    stops importing -- but that is a fact about *that image*, not about AWS
    Batch, and encoding it here would make a general profile carry one
    consumer's layout.

    Values are rendered into a Groovy double-quoted string that becomes a docker
    argument, so anything that would end the string, be interpolated by it, or
    split the argument is rejected rather than emitted broken.
    """
    if not container_env:
        return ''
    parts = []
    for key, value in container_env.items():
        if not _ENV_NAME.match(str(key)):
            raise ValueError(f'container_env key {key!r} is not a valid environment variable name')
        text = str(value)
        bad = sorted(_ENV_VALUE_FORBIDDEN.intersection(text))
        if bad:
            raise ValueError(
                f'container_env[{key!r}] contains {"".join(bad)!r}, which cannot be rendered into '
                f'a `--env` argument inside a Groovy string. Pass a value with no whitespace, '
                f'quotes, `$` or backslashes.')
        parts.append(f' --env {key}={text}')
    return ''.join(parts)


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
#
# ``retry_exit_codes`` narrows the Nextflow-level retry to exits that a fresh
# attempt can plausibly fix: 137 (SIGKILL / OOM -- the memory closures scale
# on it), 143 (SIGTERM, an instance draining), and the nf-core set 104/134/139
# (I/O, abort, segfault under memory pressure). A Python exception exits 1 and
# is NOT in the list: retrying a deterministic fault re-runs the whole task N
# times and reports "running" the whole while (measured on a 30-minute
# generation that died 7 s after division, three times over). Spot reclaim is
# Batch's own retry (``maxSpotAttempts``) and never reaches errorStrategy
# unless those attempts are exhausted.
AWSBATCH_DEFAULTS = {
    'max_spot_attempts': 10,
    'max_transfer_attempts': 10,
    'max_retries': 3,
    'retry_exit_codes': [137, 143, 104, 134, 139],
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
    retry_strategy = retry_error_strategy(opts['retry_exit_codes'])
    # Optional identity in the Batch job name: nf-amazon derives the job name
    # from the task name, which includes the ``tag`` directive.
    tag_line = ''
    if params.get('sim_tag'):
        tag_line = f"\n            tag = {_resource_value(str(params['sim_tag']))}"

    # AWS_DEFAULT_REGION is the executor's own requirement, so it is always
    # emitted. Anything else the image needs is the CALLER's business, not this
    # library's -- see _container_env_opts.
    env_opts = '--env AWS_DEFAULT_REGION=${params.aws_region}' + _container_env_opts(
        params.get('container_env'))

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
            // ...and only for exit classes a fresh attempt can fix (see
            // AWSBATCH_DEFAULTS['retry_exit_codes']); a code fault is not retried.
            errorStrategy = {retry_strategy}
            maxRetries = {opts['max_retries']}{tag_line}
            // Hash inputs by name+size, NOT last-modified. The default mode
            // includes the timestamp, and a re-render rewrites each task's
            // staged config with identical content and a new mtime -- so every
            // task hash moves and `-resume` matches nothing, having restored a
            // perfectly good session. Measured: the same ParCa hashed 40/466afa
            // then ab/95dc07 across two identical dispatches.
            // Trade-off, stated: a content change that preserves name AND size
            // is not detected. `deep` would hash content instead, at the cost of
            // reading every staged input -- here that includes a multi-hundred-MB
            // cache directory on S3.
            cache = 'lenient'{res_block}
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


# Params the generator CONSUMES rather than echoes. They configure a profile
# directive (not a Nextflow `params.<name>` lookup), so rendering them into the
# params block would be noise at best -- and `container_env`, being a dict, has
# no Groovy literal at all.
PROFILE_ONLY_PARAMS = ('container_env', 'retry_exit_codes')


def generate_nextflow_config(executor: str = 'local',
                             resources: Optional[Dict[str, Dict[str, Any]]] = None,
                             params: Optional[Dict[str, Any]] = None) -> str:
    block_params = {k: v for k, v in (params or {}).items() if k not in PROFILE_ONLY_PARAMS}
    res = _resource_lines(resources)
    res_block = ('\n' + res) if res else ''
    # Only validate the profile actually being deployed: this function always
    # emits every profile, so requiring awsbatch params from a local run would
    # make the local path depend on AWS settings it never uses.
    awsbatch = (_awsbatch_profile(res_block, params) if executor == 'awsbatch'
                else "    awsbatch {\n        // not configured; pass awsbatch params to render it\n"
                     "        process { executor = 'awsbatch' }\n    }\n")
    return f"""{_params_block(block_params)}profiles {{
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
           nextflow_args=None, config=None) -> Dict[str, Optional[str]]:
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

    **``config`` replaces the generated ``nextflow.config`` outright**, and is
    the escape hatch that matters. ``generate_nextflow_config`` emits ONE fixed
    shape with params substituted into it; a real deployment routinely needs
    more than that shape can express -- per-label *executor* switching (ParCa on
    SLURM, sims on HyperQueue), memory that scales on the previous attempt's exit
    status, ``workflow.failOnIgnore``, Fusion, accelerators. Without ``config``
    the only way to have those is to stop using ``deploy()``, which turns a
    disagreement about one directive into a fork. With it, the built-in profiles
    are a default rather than the only path.

    Pass a ``Path`` to read a file, or a ``str`` used verbatim as the config
    text. ``executor`` still selects ``-profile``, so a supplied config must
    define a profile by that name. Supplying one bypasses
    ``generate_nextflow_config`` entirely, including its required-param checks.

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

    config_path = out / 'nextflow.config'
    if config is None:
        config_text = generate_nextflow_config(
            executor=executor, resources=resources, params=params)
    elif isinstance(config, Path):
        config_text = config.read_text()
    else:
        config_text = str(config)
        # A one-line str that names a real file is almost certainly a path the
        # caller forgot to wrap. Left alone it writes that path INTO the config
        # and fails as a Groovy parse error pointing at nothing useful.
        if '\n' not in config_text and Path(config_text).is_file():
            raise TypeError(
                f'config={config_text!r} looks like a file path, but a str is used as the config '
                f'TEXT. Wrap it in Path(...) to read the file.')
    config_path.write_text(config_text)

    returncode: Optional[int] = None
    if launch:
        if shutil.which('nextflow') is None:
            raise RuntimeError('nextflow binary not found on PATH')
        cmd = ['nextflow', '-C', str(config_path), 'run', str(main_nf),
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

    return {'main_nf': str(main_nf), 'config': str(config_path),
            'returncode': returncode}
