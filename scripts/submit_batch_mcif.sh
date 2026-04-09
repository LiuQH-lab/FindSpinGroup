#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [INPUT_DIR] [OUTPUT_ROOT] [BASELINE_SUITE] [BASELINE_ROOT]

Submit the mcif batch test through Slurm using defaults rooted at this repo.

Defaults:
  INPUT_DIR      $INPUT_DIR
  OUTPUT_ROOT    $OUTPUT_ROOT
  BASELINE_SUITE $BASELINE_SUITE
  BASELINE_ROOT  $BASELINE_ROOT

Environment overrides:
  SPACE_TOL      current: ${SPACE_TOL:-0.02}
  MTOL           current: ${MTOL:-0.02}
  MEIGTOL        current: ${MEIGTOL:-0.00002}
  MATRIX_TOL     current: ${MATRIX_TOL:-0.01}
  EXPORT_TXT     current: ${EXPORT_TXT:-selected.txt}
  EXPORT_FIELDS  current: ${EXPORT_FIELDS:-index,phase,properties.ss_w_soc}
  INCLUDE_G0_SELF_AUDIT current: ${INCLUDE_G0_SELF_AUDIT:-0}
  SBATCH_BIN     current: ${SBATCH_BIN:-sbatch}

Examples:
  $(basename "$0")
  SPACE_TOL=0.01 MTOL=0.01 MATRIX_TOL=0.005 $(basename "$0")
  $(basename "$0") /path/to/input /path/to/output suite_name /path/to/batch_baselines
  SBATCH_BIN=echo $(basename "$0")
EOF
}

INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/tests/testset/mcif_241130_no2186}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/mcif_241130_no2186_run}"
BASELINE_SUITE="${BASELINE_SUITE:-mcif_241130_no2186}"
BASELINE_ROOT="${BASELINE_ROOT:-$REPO_ROOT/batch_baselines}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/scripts/run_batch_mcif.slurm}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

SPACE_TOL="${SPACE_TOL:-0.02}"
MTOL="${MTOL:-0.02}"
MEIGTOL="${MEIGTOL:-0.00002}"
MATRIX_TOL="${MATRIX_TOL:-0.01}"

EXPORT_TXT="${EXPORT_TXT:-selected.txt}"
EXPORT_FIELDS="${EXPORT_FIELDS:-index,phase,properties.ss_w_soc}"

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

if [[ $# -gt 2 ]]; then
  BASELINE_SUITE="$3"
fi

if [[ $# -gt 3 ]]; then
  BASELINE_ROOT="$4"
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Slurm script does not exist: $SLURM_SCRIPT" >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$BASELINE_ROOT"

echo "Repo root      : $REPO_ROOT"
echo "Slurm script   : $SLURM_SCRIPT"
echo "Input dir      : $INPUT_DIR"
echo "Output root    : $OUTPUT_ROOT"
echo "Baseline suite : $BASELINE_SUITE"
echo "Baseline root  : $BASELINE_ROOT"
echo "Tolerances     : space=$SPACE_TOL mtol=$MTOL meigtol=$MEIGTOL matrix=$MATRIX_TOL"
echo "Export txt     : ${EXPORT_TXT:-<none>}"
echo "Export fields  : ${EXPORT_FIELDS:-<none>}"
echo "G0 self audit  : ${INCLUDE_G0_SELF_AUDIT:-0}"

CMD=(
  "$SBATCH_BIN"
  "$SLURM_SCRIPT"
  "$INPUT_DIR"
  "$OUTPUT_ROOT"
  "$BASELINE_SUITE"
  "$BASELINE_ROOT"
)

echo "Submit command :"
printf '  %q' "${CMD[@]}"
echo

SPACE_TOL="$SPACE_TOL" \
MTOL="$MTOL" \
MEIGTOL="$MEIGTOL" \
MATRIX_TOL="$MATRIX_TOL" \
EXPORT_TXT="$EXPORT_TXT" \
EXPORT_FIELDS="$EXPORT_FIELDS" \
INCLUDE_G0_SELF_AUDIT="${INCLUDE_G0_SELF_AUDIT:-0}" \
FSG_REPO_ROOT="$REPO_ROOT" \
"${CMD[@]}"
