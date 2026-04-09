#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_SERVER="yuyt@10.20.26.130"
DEFAULT_REMOTE_REPO_ROOT="/share/home/yuyt/project/fsg-2026/FindSpinGroup"
DEFAULT_REMOTE_OUTPUT_ROOT="$DEFAULT_REMOTE_REPO_ROOT/output/mcif_241130_no2186_run"
DEFAULT_LOCAL_DEST="$REPO_ROOT/tests/error_info"

SERVER="$DEFAULT_SERVER"
REMOTE_OUTPUT_ROOT="$DEFAULT_REMOTE_OUTPUT_ROOT"
LOCAL_DEST="$DEFAULT_LOCAL_DEST"
RUN_DIR=""
DRY_RUN=0
SSH_BIN="${SSH_BIN:-ssh}"
SCP_BIN="${SCP_BIN:-scp}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Download a batch-test run directory from the server to the local workspace.

Defaults:
  server            $DEFAULT_SERVER
  remote output     $DEFAULT_REMOTE_OUTPUT_ROOT
  local destination $DEFAULT_LOCAL_DEST

Options:
  --server USER@HOST         Remote login target
  --remote-output-root PATH  Remote batch output root containing run_* dirs
  --run-dir NAME_OR_PATH     Specific run dir name or absolute remote path
  --local-dest PATH          Local directory to store the downloaded run dir
  --dry-run                  Print commands only
  -h, --help                 Show this help

Examples:
  bash scripts/fetch_batch_results.sh
  bash scripts/fetch_batch_results.sh --run-dir run_v0.13.1_20260312_153000
  bash scripts/fetch_batch_results.sh --local-dest /tmp/fsg_runs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)
      SERVER="$2"
      shift 2
      ;;
    --remote-output-root)
      REMOTE_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --local-dest)
      LOCAL_DEST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$RUN_DIR" ]]; then
  if [[ "$RUN_DIR" == /* ]]; then
    REMOTE_RUN_DIR="$RUN_DIR"
  else
    REMOTE_RUN_DIR="$REMOTE_OUTPUT_ROOT/$RUN_DIR"
  fi
elif [[ $DRY_RUN -eq 1 ]]; then
  REMOTE_RUN_DIR="$REMOTE_OUTPUT_ROOT/<latest run_*>"
else
  REMOTE_RUN_DIR="$("$SSH_BIN" "$SERVER" "ls -dt \"$REMOTE_OUTPUT_ROOT\"/run_* 2>/dev/null | head -n 1")"
  if [[ -z "$REMOTE_RUN_DIR" ]]; then
    echo "Cannot find any run_* directory under: $REMOTE_OUTPUT_ROOT" >&2
    exit 1
  fi
fi

RUN_BASENAME="$(basename "$REMOTE_RUN_DIR")"
LOCAL_DEST="$(cd "$(dirname "$LOCAL_DEST")" 2>/dev/null && pwd)/$(basename "$LOCAL_DEST")"
LOCAL_RUN_DIR="$LOCAL_DEST/$RUN_BASENAME"

echo "Server         : $SERVER"
echo "Remote run dir : $REMOTE_RUN_DIR"
echo "Local dest     : $LOCAL_DEST"
echo "Local run dir  : $LOCAL_RUN_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "mkdir -p \"$LOCAL_DEST\""
  echo "$SCP_BIN -r \"$SERVER:$REMOTE_RUN_DIR\" \"$LOCAL_DEST/\""
  exit 0
fi

mkdir -p "$LOCAL_DEST"

if [[ -e "$LOCAL_RUN_DIR" ]]; then
  echo "Local run dir already exists: $LOCAL_RUN_DIR" >&2
  exit 2
fi

"$SCP_BIN" -r "$SERVER:$REMOTE_RUN_DIR" "$LOCAL_DEST/"

echo "Downloaded to  : $LOCAL_RUN_DIR"
