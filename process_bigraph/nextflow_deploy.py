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
