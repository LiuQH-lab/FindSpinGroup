#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_SCRIPT="${BUILD_SCRIPT:-$REPO_ROOT/scripts/build_batch_snapshot.sh}"

SERVER_ALIAS="${SERVER_ALIAS:-yuyt-26-130}"
REMOTE_BASE="${REMOTE_BASE:-/share/home/yuyt/project/fsg-2026}"
REMOTE_SHARED_REPO="${REMOTE_SHARED_REPO:-$REMOTE_BASE/FindSpinGroup}"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-$REMOTE_BASE/findspingroup_snapshots/batch_mcif}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-$REMOTE_BASE/output/mcif_241130_no2186_run}"
INPUT_SUBDIR="${INPUT_SUBDIR:-tests/testset/mcif_241130_no2186}"
BASELINE_SUITE="${BASELINE_SUITE:-mcif_241130_no2186}"

SPACE_TOL="${SPACE_TOL:-0.02}"
MTOL="${MTOL:-0.02}"
MEIGTOL="${MEIGTOL:-0.00002}"
MATRIX_TOL="${MATRIX_TOL:-0.01}"
EXPORT_TXT="${EXPORT_TXT:-selected.txt}"
EXPORT_FIELDS="${EXPORT_FIELDS:-index,phase,properties.ss_w_soc}"
INCLUDE_G0_SELF_AUDIT="${INCLUDE_G0_SELF_AUDIT:-0}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Build a local batch snapshot, upload it, unpack it on the server, relink the
shared runtime dependencies, and submit the standard .mcif Slurm batch.

Defaults:
  SERVER_ALIAS         $SERVER_ALIAS
  REMOTE_BASE          $REMOTE_BASE
  REMOTE_SHARED_REPO   $REMOTE_SHARED_REPO
  REMOTE_SNAPSHOT_ROOT $REMOTE_SNAPSHOT_ROOT
  REMOTE_OUTPUT_ROOT   $REMOTE_OUTPUT_ROOT
  INPUT_SUBDIR         $INPUT_SUBDIR
  BASELINE_SUITE       $BASELINE_SUITE
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$BUILD_SCRIPT" ]]; then
  echo "Build script is not executable: $BUILD_SCRIPT" >&2
  exit 2
fi

BUILD_OUTPUT="$("$BUILD_SCRIPT")"
SNAPSHOT_PATH="$(printf '%s\n' "$BUILD_OUTPUT" | sed -n 's/^Snapshot file : //p' | tail -n 1)"
SNAPSHOT_ROOT_NAME="$(printf '%s\n' "$BUILD_OUTPUT" | sed -n 's/^Snapshot root : //p' | tail -n 1)"

if [[ -z "$SNAPSHOT_PATH" || -z "$SNAPSHOT_ROOT_NAME" ]]; then
  echo "Failed to parse snapshot build output:" >&2
  printf '%s\n' "$BUILD_OUTPUT" >&2
  exit 2
fi

SNAPSHOT_BASENAME="$(basename "$SNAPSHOT_PATH")"
REMOTE_TARBALL="$REMOTE_SNAPSHOT_ROOT/$SNAPSHOT_BASENAME"
REMOTE_REPO_ROOT="$REMOTE_SNAPSHOT_ROOT/$SNAPSHOT_ROOT_NAME"
REMOTE_INPUT_DIR="$REMOTE_REPO_ROOT/$INPUT_SUBDIR"
REMOTE_BASELINE_ROOT="$REMOTE_REPO_ROOT/batch_baselines"

echo "$BUILD_OUTPUT"
echo "Upload target  : $SERVER_ALIAS:$REMOTE_TARBALL"
echo "Remote root    : $REMOTE_REPO_ROOT"

ssh "$SERVER_ALIAS" "mkdir -p '$REMOTE_SNAPSHOT_ROOT' '$REMOTE_OUTPUT_ROOT'"
scp "$SNAPSHOT_PATH" "$SERVER_ALIAS:$REMOTE_TARBALL"

ssh "$SERVER_ALIAS" "
  set -euo pipefail
  cd '$REMOTE_SNAPSHOT_ROOT'
  rm -rf '$REMOTE_REPO_ROOT'
  tar -xzf '$REMOTE_TARBALL'
  cd '$REMOTE_REPO_ROOT'
  ln -sfn '$REMOTE_SHARED_REPO/.venv' .venv
  ln -sfn '$REMOTE_SHARED_REPO/batch_baselines' batch_baselines
  test -x ./.venv/bin/python
  test -d '$REMOTE_INPUT_DIR'
  PYTHONPATH=\"\$PWD/src\" ./.venv/bin/python - <<'PY'
import spintensor
from findspingroup.version import __version__
print('snapshot_version', __version__)
print('spintensor_ok')
PY
  SPACE_TOL='$SPACE_TOL' \
  MTOL='$MTOL' \
  MEIGTOL='$MEIGTOL' \
  MATRIX_TOL='$MATRIX_TOL' \
  EXPORT_TXT='$EXPORT_TXT' \
  EXPORT_FIELDS='$EXPORT_FIELDS' \
  INCLUDE_G0_SELF_AUDIT='$INCLUDE_G0_SELF_AUDIT' \
  SBATCH_BIN='$SBATCH_BIN' \
  bash scripts/submit_batch_mcif.sh \
    '$REMOTE_INPUT_DIR' \
    '$REMOTE_OUTPUT_ROOT' \
    '$BASELINE_SUITE' \
    '$REMOTE_BASELINE_ROOT'
"
