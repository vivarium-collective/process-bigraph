"""The base-library boundary, enforced.

process-bigraph knows composites, processes, steps, runs, ticks, structural
changes, protocol runtimes, exceptions and entrypoint tasks. It knows nothing
about any particular domain that runs on it, and nothing about any cloud.
Domain identifiers belong in ``baggage``; infrastructure identifiers belong in
``tags``. This test greps the observability code this repo ships for a
deny-list of domain and infrastructure words so the boundary survives future
changes. A hit names the file, the line and the word.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / 'process_bigraph'

# Whole file: the observability module is new and entirely ours.
WHOLE_FILES = ('events.py',)
# Hook regions: only the lines the observability branch added to files that
# predate it (their inherited prose is a separate, general vocabulary cleanup).
ADDED_LINES_FILES = ('composite.py', 'protocols/ray.py', 'run_composite.py',
                     'run_step.py', 'nextflow_deploy.py')
BASE_COMMIT = '78d1488'   # the merge-base of the observability branch

DENY = (
    'lineage', 'generation', 'variant', 'seed', 'campaign', 'dispatcher',
    'runner', 'parca', 'cell', 'sim_id', 'experiment_id', 'viva', 'vecoli',
    'cloudwatch', 's3://',
)
# word-ish: not preceded/followed by an identifier character, so ``init_cell``
# (a pre-existing API name) and ``generic`` do not match.
_PATTERNS = [(word, re.compile(r'(?<![A-Za-z0-9_])' + re.escape(word) + r'(?![A-Za-z0-9_])', re.I))
             for word in DENY]


def _added_lines(relpath: str):
    """``(lineno, text)`` for lines this branch added to ``relpath``; the whole
    file when git cannot answer (a tarball checkout)."""
    try:
        out = subprocess.run(
            ['git', 'diff', '--unified=0', BASE_COMMIT, '--', f'process_bigraph/{relpath}'],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return list(enumerate((PKG / relpath).read_text().splitlines(), 1))
    added, lineno = [], 0
    for line in out.splitlines():
        if line.startswith('@@'):
            m = re.search(r'\+(\d+)', line)
            lineno = int(m.group(1)) if m else 0
            continue
        if line.startswith('+') and not line.startswith('+++'):
            added.append((lineno, line[1:]))
            lineno += 1
        elif not line.startswith('-'):
            lineno += 1
    return added


def _scan(relpath, lines):
    hits = []
    for lineno, text in lines:
        for word, pattern in _PATTERNS:
            if pattern.search(text):
                hits.append(f'{relpath}:{lineno}: {word!r} in {text.strip()[:80]!r}')
    return hits


@pytest.mark.parametrize('relpath', WHOLE_FILES)
def test_events_vocabulary_is_domain_free_in_shipped_modules(relpath):
    lines = list(enumerate((PKG / relpath).read_text().splitlines(), 1))
    hits = _scan(relpath, lines)
    assert not hits, 'domain/infrastructure words in the engine:\n' + '\n'.join(hits)


@pytest.mark.parametrize('relpath', ADDED_LINES_FILES)
def test_events_vocabulary_is_domain_free_in_hook_regions(relpath):
    hits = _scan(relpath, _added_lines(relpath))
    assert not hits, 'domain/infrastructure words in the engine hooks:\n' + '\n'.join(hits)
