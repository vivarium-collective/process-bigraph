# Create and publish a new version by creating and pushing a git tag for
# the version and publishing the version to PyPI. Also perform some
# basic checks to avoid mistakes in releases, for example tags not
# matching PyPI.
# Usage: ./release.sh

set -e

# check working directory is clean
if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have changes that have yet to be committed."
    echo "Aborting."
    exit 1
fi

# check that we are on main
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
    echo "You are on $branch but should be on main for releases."
    echo "Aborting."
    exit 1
fi

# bump the version
uv version --bump patch
version=$(uv version --short)

# create and push git tag
git add pyproject.toml
git commit -m "version $version"
git tag -m "version v$version" "v$version"
git push --tags

# perform the build
rm -rf build/ dist/
uv build

# publish to pypi
uv publish --token $(cat ~/.pypi-token)

echo "version v$version has been published on PyPI and has a git tag."
