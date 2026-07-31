"""Byte-identity check for the CompositeSpec template lowering.

`CompositeSpec` resolves a spec's ``${name}`` parameters two ways during the
transition: the legacy ``substitute_parameters`` regex walk, and the lowering
onto bigraph-schema's site/fill primitive. This script asserts they produce
**identical** documents for every static composite spec on this machine.

The corpus lives in sibling repos (pbg-lammps, pbg-membrane, pbg-copasi,
v2ecoli, …), so this cannot run in CI. `tests.py` carries self-contained
fixtures covering the same substitution semantics; this is the cross-repo
check, run locally before deleting the legacy path.

    PYTHONPATH=<this worktree> python scripts/verify_template_lowering.py
    PYTHONPATH=<this worktree> python scripts/verify_template_lowering.py --freeze

`--freeze` records today's documents to scripts/corpus/legacy_documents.json
so the comparison still has a baseline once the legacy path is deleted.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

SKIP = ('/.git/', '/node_modules/', '/.pbg/worktrees/', '/.claude/worktrees/')
BASELINE = Path(__file__).parent / 'corpus' / 'legacy_documents.json'


def discover(root):
    """Every static composite spec under `root`, de-duplicated."""
    found = {}
    for path in Path(root).rglob('*.composite.*'):
        if path.suffix not in ('.yaml', '.yml', '.json'):
            continue
        if any(part in str(path) for part in SKIP):
            continue
        try:
            found.setdefault((path.name, path.stat().st_size), path)
        except OSError:
            continue
    return {name: path for (name, _size), path in sorted(found.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path.home() / 'code'))
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    import process_bigraph
    from process_bigraph.composite_spec import CompositeSpec
    print(f'process_bigraph: {process_bigraph.__file__}')

    specs = discover(args.root)
    print(f'corpus: {len(specs)} static composite specs under {args.root}\n')

    documents, mismatches, errors = {}, [], []
    for name, path in specs.items():
        try:
            spec = CompositeSpec.from_file(path)
            current = spec.to_document(emit=False)
            documents[name] = current
        except Exception as error:
            errors.append((name, f'{type(error).__name__}: {error}'))
            continue

        legacy_path = getattr(spec, '_to_document_legacy', None)
        if legacy_path is None:
            continue  # legacy path already deleted
        legacy = legacy_path(None)
        if current != legacy:
            mismatches.append((name, legacy, current))

    print(f'built: {len(documents)}   errors: {len(errors)}   '
          f'mismatches: {len(mismatches)}')

    for name, message in errors:
        print(f'  ERROR    {name}: {message}')

    for name, legacy, current in mismatches:
        print(f'\n  MISMATCH {name}')
        print(f'    legacy : {json.dumps(legacy, sort_keys=True, default=str)[:400]}')
        print(f'    lowered: {json.dumps(current, sort_keys=True, default=str)[:400]}')

    if args.freeze:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(
            documents, indent=1, sort_keys=True, default=str))
        print(f'\nfroze {len(documents)} documents -> {BASELINE}')
    elif BASELINE.is_file():
        baseline = json.loads(BASELINE.read_text())
        drifted = [
            name for name in sorted(set(baseline) & set(documents))
            if json.dumps(baseline[name], sort_keys=True, default=str)
            != json.dumps(documents[name], sort_keys=True, default=str)]
        print(f'\nbaseline: {len(baseline)} documents, '
              f'{len(drifted)} drifted from the frozen record')
        for name in drifted:
            print(f'  DRIFT    {name}')
        if drifted:
            return 1

    return 1 if (mismatches or errors) else 0


if __name__ == '__main__':
    sys.exit(main())
