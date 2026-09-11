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
# The hook-region scan diffs against the merge-base with ``main`` (so a future
# branch is checked on ITS added lines); ``FALLBACK_BASE`` is the merge-base of
# the branch that introduced this test, for checkouts without ``origin/main``.
FALLBACK_BASE = '78d1488'

DENY = (
    'lineage', 'generation', 'variant', 'seed', 'campaign', 'dispatcher',
    'runner', 'parca', 'cell', 'sim_id', 'experiment_id', 'viva', 'vecoli',
    'cloudwatch', 's3://',
)
# word-ish: not preceded/followed by an identifier character, so ``init_cell``
# (a pre-existing API name) and ``generic`` do not match.
_PATTERNS = [(word, re.compile(r'(?<![A-Za-z0-9_])' + re.escape(word) + r'(?![A-Za-z0-9_])', re.I))
             for word in DENY]


def _base_commit():
    """The commit to diff against: ``merge-base origin/main HEAD`` when the
    checkout has it, else ``FALLBACK_BASE`` when that object exists, else
    ``None`` (a shallow or tarball checkout -- the caller skips)."""
    def _git(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    for attempt in (lambda: _git('merge-base', 'origin/main', 'HEAD'),
                    lambda: _git('rev-parse', '--verify', '--quiet', FALLBACK_BASE + '^{commit}')):
        try:
            base = attempt()
            if base:
                return base
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def _added_lines(relpath: str, base: str):
    """``(lineno, text)`` for lines added to ``relpath`` since ``base``."""
    out = subprocess.run(
        ['git', 'diff', '--unified=0', base, '--', f'process_bigraph/{relpath}'],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
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
    base = _base_commit()
    if base is None:
        pytest.skip('no git history to diff against (shallow or tarball checkout); '
                    'CI checks out with fetch-depth 0 so this runs there')
    hits = _scan(relpath, _added_lines(relpath, base))
    assert not hits, 'domain/infrastructure words in the engine hooks:\n' + '\n'.join(hits)
