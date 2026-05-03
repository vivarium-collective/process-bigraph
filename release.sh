#!/usr/bin/env bash
# Bump version, commit, tag, and push. CI publishes to PyPI via OIDC
# trusted publisher (.github/workflows/release.yml).
#
# Usage:
#   ./release.sh           # bumps patch
#   ./release.sh minor     # bumps minor
#   ./release.sh major     # bumps major

set -e

bump_kind="${1:-patch}"

if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have uncommitted changes. Aborting."
    exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
    echo "You are on $branch but releases must be cut from main."
    exit 1
fi
git fetch origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "main is not in sync with origin/main. Pull or push first."
    exit 1
fi

uv version --bump "$bump_kind"
version=$(uv version --short)
uv lock

git add pyproject.toml uv.lock
git commit -m "version $version"

git push origin main

git tag -m "version v$version" "v$version"
git push origin "v$version"

echo "Tagged v$version. Watch the release workflow:"
echo "  gh run watch"
