"""The base-library boundary, enforced.

process-bigraph knows composites, processes, steps, runs, ticks, structural
changes, protocol runtimes, exceptions and entrypoint tasks. It knows nothing
about any particular domain that runs on it, and nothing about any cloud.
Domain identifiers belong in ``baggage``; infrastructure identifiers belong in
``tags``.

This test scans the observability code this repo ships for a deny-list of
domain and infrastructure words, and it scans **constructs only**: names the
code defines or uses (functions, classes, variables, arguments, attributes,
keyword arguments, imports) and string literals (event names, keys, CLI
flags, templates). Comments and docstrings are prose and are deliberately
exempt -- explaining a design choice by reference to the application that
motivated it is fine; building the application's vocabulary into the engine
is not. Comment lines inside an embedded-language template string (``#``,
``//``) are prose too. A hit names the file, the line and the word.
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / 'process_bigraph'

# The observability module plus every file the engine hooks live in.
FILES = ('events.py', 'composite.py', 'protocols/ray.py', 'run_composite.py',
         'run_step.py', 'nextflow_deploy.py')

DENY = (
    'lineage', 'generation', 'variant', 'seed', 'campaign', 'dispatcher',
    'runner', 'parca', 'cell', 'sim_id', 'experiment_id', 'viva', 'vecoli',
    'cloudwatch', 's3://',
)
# word-ish: not preceded/followed by an identifier character, so ``generic``
# does not match ``generation``. Constructs are also matched per ``_``
# segment, so ``lineage_id`` is a hit.
_PATTERNS = [(word, re.compile(r'(?<![A-Za-z0-9_])' + re.escape(word) + r'(?![A-Za-z0-9_])', re.I))
             for word in DENY]
# Pre-existing names the observability branch inherited and did not add. They
# are internal to the Ray batch actor (nothing downstream calls them) and are
# listed here so the boundary is enforced on everything else; renaming them is
# a separate, mechanical change.
INHERITED = {'protocols/ray.py': {'init_cell', 'has_cell'}}

_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_EMBEDDED_COMMENT = re.compile(r'^\s*(#|//)')


def _docstring_nodes(tree):
    """The ``Constant`` nodes that are docstrings (first statement of a module,
    class or function body)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_OWNERS) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.add(first.value)
    return found


def _literal_code(text: str) -> str:
    """A string literal with its embedded-language comment lines removed."""
    return '\n'.join(line for line in text.splitlines() if not _EMBEDDED_COMMENT.match(line))


def constructs(source: str):
    """``(lineno, text, is_identifier)`` for every construct in ``source``:
    identifiers the code defines or uses, and string literals other than
    docstrings."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.lineno, node.id, True
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr, True
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg, True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.lineno, node.name, True
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.lineno, node.arg, True
        elif isinstance(node, ast.alias):
            yield node.lineno, node.asname or node.name, True
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module, True
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node not in docstrings):
            yield node.lineno, _literal_code(node.value), False


def scan(relpath: str, source: str):
    inherited = INHERITED.get(relpath, ())
    hits = []
    for lineno, text, is_identifier in constructs(source):
        if is_identifier and text in inherited:
            continue
        for name in inherited:  # a message that names an inherited method
            text = text.replace(name, '')
        # also checked segment-wise: ``lineage_id`` and ``"cell_queue"`` are hits
        candidates = (text, text.replace('_', ' '))
        for word, pattern in _PATTERNS:
            if any(pattern.search(c) for c in candidates):
                hits.append(f'{relpath}:{lineno}: {word!r} in {text.strip()[:80]!r}')
                break
    return hits


@pytest.mark.parametrize('relpath', FILES)
def test_events_vocabulary_is_domain_free(relpath):
    hits = scan(relpath, (PKG / relpath).read_text())
    assert not hits, 'domain/infrastructure words in engine constructs:\n' + '\n'.join(hits)


def test_scanner_flags_constructs_not_prose():
    """The boundary the scan enforces, stated as examples."""
    prose = '''
# a cell-as-Composite divides here (vEcoli's case)
def run(interval):
    """One ParCa per lineage; see the campaign notes."""
    template = """
        // 'finish' rather than vEcoli's 'ignore'
        errorStrategy = 'finish'
    """
    return interval
'''
    assert scan('prose.py', prose) == []

    for construct in ('lineage_id = 1',
                      'def parca(): pass',
                      'class Cell: pass',
                      'emit("run.start", generation=3)',
                      'x = obj.experiment_id',
                      'name = "sim_id"',
                      'template = """\n    queue = params.cell_queue\n"""',
                      'from viva import x'):
        assert scan('c.py', construct), construct
