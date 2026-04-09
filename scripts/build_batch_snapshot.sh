#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
VERSION_FILE="$REPO_ROOT/src/findspingroup/version.py"
PACKAGE_VERSION="$(sed -nE 's/^__version__ = "([^"]+)"$/\1/p' "$VERSION_FILE" | head -n 1)"
PACKAGE_VERSION="${PACKAGE_VERSION:-unknown}"
ROOT_NAME="${ROOT_NAME:-FindSpinGroup-batch-snapshot-v${PACKAGE_VERSION}-${STAMP}}"
OUTPUT_PATH="${OUTPUT_PATH:-$DIST_DIR/${ROOT_NAME}-rooted.tar.gz}"
INCLUDE_PATHS=(
  "src"
  "scripts"
  "tests/testset/mcif_241130_no2186"
  "pyproject.toml"
)

usage() {
  cat <<EOF
Usage: $(basename "$0")

Create a rooted snapshot tarball for remote batch execution.

Defaults:
  DIST_DIR      $DIST_DIR
  ROOT_NAME     $ROOT_NAME
  OUTPUT_PATH   $OUTPUT_PATH

The snapshot intentionally includes only the runtime core paths:
  ${INCLUDE_PATHS[*]}
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "$DIST_DIR"
TMPDIR_SNAPSHOT="$(mktemp -d "${TMPDIR:-/tmp}/fsg-batch-snapshot.XXXXXX")"
STAGING_DIR="$TMPDIR_SNAPSHOT/$ROOT_NAME"
mkdir -p "$STAGING_DIR"

cleanup() {
  rm -rf "$TMPDIR_SNAPSHOT"
}
trap cleanup EXIT

for rel_path in "${INCLUDE_PATHS[@]}"; do
  if [[ ! -e "$REPO_ROOT/$rel_path" ]]; then
    echo "Required snapshot path does not exist: $REPO_ROOT/$rel_path" >&2
    exit 2
  fi
done

COPYFILE_DISABLE=1 LC_ALL=C LANG=C tar \
  --exclude='._*' \
  -C "$REPO_ROOT" \
  -cf - \
  "${INCLUDE_PATHS[@]}" | COPYFILE_DISABLE=1 LC_ALL=C LANG=C tar -C "$STAGING_DIR" -xf -

find "$STAGING_DIR" -name '._*' -delete

COPYFILE_DISABLE=1 LC_ALL=C LANG=C tar -C "$TMPDIR_SNAPSHOT" -cf - "$ROOT_NAME" | gzip -n > "$OUTPUT_PATH"

echo "Snapshot root : $ROOT_NAME"
echo "Snapshot file : $OUTPUT_PATH"
