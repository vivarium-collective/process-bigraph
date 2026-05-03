#!/usr/bin/env bash
# Local-development helper: install the canonical, frozen dependency
# tree (so versions match CI), then swap in the editable sibling
# checkout of bigraph-schema.
#
# Run after `git pull` whenever the lock or pyproject changed, or
# whenever you want to re-sync after editing across packages.

set -e

# Sync to the frozen lock first — gives us the same versions CI
# resolves to.
uv sync --frozen

# Swap in editables for sibling repos. --no-deps means the editable
# install does not pull in its own resolution; we keep the locked
# transitive closure intact.
for sibling in ../bigraph-schema; do
    if [ -d "$sibling" ]; then
        echo "Linking editable: $sibling"
        uv pip install -e "$sibling" --no-deps --reinstall
    else
        echo "Skipping $sibling (not present)"
    fi
done

echo
echo "Active versions:"
uv pip list 2>/dev/null | grep -E '^(bigraph-schema|process-bigraph) '
