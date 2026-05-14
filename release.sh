#!/usr/bin/env bash
# Tag and push a release for a package in this repo.
#
# Usage: ./release.sh <package> <version> [<commit>]
#   <package>  Package name as it appears under packages/ (e.g. cheese3d-annotator)
#   <version>  Version string without leading 'v' (e.g. 0.2.0)
#   <commit>   Optional commit to tag (defaults to HEAD)

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <package> <version> [<commit>]" >&2
    exit 2
fi

package="$1"
version="$2"
commit="${3:-HEAD}"

repo_root="$(git rev-parse --show-toplevel)"

if [[ ! -d "$repo_root/packages/$package" ]]; then
    echo "error: no package directory at packages/$package" >&2
    exit 1
fi

tag="$package/v$version"
message="Release $package v$version"

if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    echo "error: tag $tag already exists" >&2
    exit 1
fi

read -r -p "Did you update the version in pyproject.toml and commit? [y/N] " reply
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "aborted" >&2
    exit 1
fi

git tag -a "$tag" -m "$message" "$commit"
git push --tags
