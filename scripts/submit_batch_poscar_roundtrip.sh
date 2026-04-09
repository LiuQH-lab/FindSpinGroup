#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [INPUT_DIR] [OUTPUT_ROOT]

Submit the POSCAR full-batch roundtrip test through Slurm using defaults rooted
at this repo.

Defaults:
  INPUT_DIR    $INPUT_DIR
  OUTPUT_ROOT  $OUTPUT_ROOT

Environment overrides:
  SPACE_TOL    current: ${SPACE_TOL:-0.02}
  MTOL         current: ${MTOL:-0.02}
  MEIGTOL      current: ${MEIGTOL:-0.00002}
  MATRIX_TOL   current: ${MATRIX_TOL:-0.01}
  SOURCE_MODE  current: ${SOURCE_MODE:-acc_primitive}
  COMPARE_CONF current: ${COMPARE_CONF:-1}
  SAVE_POSCAR  current: ${SAVE_POSCAR:-1}
  QUIET        current: ${QUIET:-1}
  SBATCH_BIN   current: ${SBATCH_BIN:-sbatch}

Examples:
  $(basename "$0")
  SBATCH_BIN=echo $(basename "$0")
EOF
}

INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/tests/testset/mcif_241130_no2186}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/poscar_roundtrip_run}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/scripts/run_batch_poscar_roundtrip.slurm}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

SPACE_TOL="${SPACE_TOL:-0.02}"
MTOL="${MTOL:-0.02}"
MEIGTOL="${MEIGTOL:-0.00002}"
MATRIX_TOL="${MATRIX_TOL:-0.01}"
SOURCE_MODE="${SOURCE_MODE:-acc_primitive}"
COMPARE_CONF="${COMPARE_CONF:-1}"
SAVE_POSCAR="${SAVE_POSCAR:-1}"
QUIET="${QUIET:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  INPUT_DIR="$1"
fi

if [[ $# -gt 1 ]]; then
  OUTPUT_ROOT="$2"
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Slurm script does not exist: $SLURM_SCRIPT" >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

echo "Repo root     : $REPO_ROOT"
echo "Slurm script  : $SLURM_SCRIPT"
echo "Input dir     : $INPUT_DIR"
echo "Output root   : $OUTPUT_ROOT"
echo "Tolerances    : space=$SPACE_TOL mtol=$MTOL meigtol=$MEIGTOL matrix=$MATRIX_TOL"
echo "Source mode   : $SOURCE_MODE"
echo "Compare conf  : $COMPARE_CONF"
echo "Save poscar   : $SAVE_POSCAR"
echo "Quiet         : $QUIET"

CMD=(
  "$SBATCH_BIN"
  "$SLURM_SCRIPT"
  "$INPUT_DIR"
  "$OUTPUT_ROOT"
)

echo "Submit command :"
printf '  %q' "${CMD[@]}"
echo

SPACE_TOL="$SPACE_TOL" \
MTOL="$MTOL" \
MEIGTOL="$MEIGTOL" \
MATRIX_TOL="$MATRIX_TOL" \
SOURCE_MODE="$SOURCE_MODE" \
COMPARE_CONF="$COMPARE_CONF" \
SAVE_POSCAR="$SAVE_POSCAR" \
QUIET="$QUIET" \
FSG_REPO_ROOT="$REPO_ROOT" \
"${CMD[@]}"
