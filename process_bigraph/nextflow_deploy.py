"""Generate nextflow.config + deploy a Composite's Step network to a backend.

Wraps process_bigraph.nextflow.render_composite (which emits main.nf) with a
nextflow.config profile block and an optional `nextflow run` launch. The
executor abstraction mirrors vEcoli's runscripts/nextflow/config.template:
one `profiles { }` block, backend selected by name.
"""

from __future__ import annotations

import json
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


def generate_nextflow_config(executor: str = 'local',
                             resources: Optional[Dict[str, Dict[str, Any]]] = None,
                             params: Optional[Dict[str, Any]] = None) -> str:
    res = _resource_lines(resources)
    res_block = ('\n' + res) if res else ''
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
    awsbatch {{
        // STUB (untested in v1)
        process {{ executor = 'awsbatch' }}
    }}
    'google-batch' {{
        // STUB (untested in v1)
        process {{ executor = 'google-batch' }}
    }}
}}
"""


def deploy(composite, *, outdir: str, executor: str = 'local',
           launch: bool = False, resources=None, params=None,
           options=None, work_dir=None) -> Dict[str, Optional[str]]:
    """Write ``main.nf`` + ``nextflow.config`` for a Composite, optionally launch it.

    Renders the Step network via ``render_composite`` (pinning
    ``sys.executable`` so Nextflow's subprocess tasks use this interpreter)
    and writes a matching ``nextflow.config`` via ``generate_nextflow_config``.
    When ``launch=True``, shells out to ``nextflow -C <config> run <main.nf>
    -profile <executor>`` and raises ``subprocess.CalledProcessError`` on a
    non-zero exit.
    """
    from process_bigraph.nextflow import render_composite

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    render_options = dict(options or {})
    render_options.setdefault('python', sys.executable)
    # render_composite's own default ('main') is a reserved identifier in
    # real Nextflow — naming the entry workflow block `main` is a compile
    # error. Default to an unnamed/implicit entry workflow instead, which
    # Nextflow runs without needing `-entry`. A caller-supplied
    # `workflow_name` (including '') in `options` always wins.
    render_options.setdefault('workflow_name', '')

    main_nf = out / 'main.nf'
    main_nf.write_text(render_composite(composite, render_options))

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
        proc = subprocess.run(cmd, cwd=str(out))
        returncode = proc.returncode
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    return {'main_nf': str(main_nf), 'config': str(config),
            'returncode': returncode}
