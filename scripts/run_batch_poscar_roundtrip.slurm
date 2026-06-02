#!/bin/bash
#SBATCH --job-name=fsg_poscar
#SBATCH --partition=256G56c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -euo pipefail

ulimit -s unlimited
ulimit -l unlimited

if command -v module >/dev/null 2>&1; then
  module load anaconda || true
fi
if command -v conda >/dev/null 2>&1; then
  set +u
  source activate base || true
  set -u
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "Usage: sbatch $0 <mcif_dir> <output_root>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${FSG_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
INPUT_DIR="$1"
OUTPUT_ROOT="$2"
SPACE_TOL="${SPACE_TOL:-0.02}"
MTOL="${MTOL:-0.02}"
MEIGTOL="${MEIGTOL:-0.00002}"
MATRIX_TOL="${MATRIX_TOL:-0.01}"
SOURCE_MODE="${SOURCE_MODE:-acc_primitive}"
COMPARE_MODE="${COMPARE_MODE:-basic}"
COMPARE_CONF="${COMPARE_CONF:-1}"
COMPARE_ATOL="${COMPARE_ATOL:-1e-6}"
COMPARE_RTOL="${COMPARE_RTOL:-1e-8}"
MAX_DIFFERENCES_PER_CASE="${MAX_DIFFERENCES_PER_CASE:-200}"
SAVE_POSCAR="${SAVE_POSCAR:-1}"
QUIET="${QUIET:-1}"
WORKERS="${BATCH_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
VERSION_FILE="$REPO_ROOT/src/findspingroup/version.py"
if [[ -f "$VERSION_FILE" ]]; then
  PACKAGE_VERSION="$(sed -nE 's/^__version__ = \"([^\"]+)\"$/\1/p' "$VERSION_FILE" | head -n 1)"
fi
PACKAGE_VERSION="${PACKAGE_VERSION:-unknown}"
OUTPUT_DIR="${OUTPUT_ROOT}/run_v${PACKAGE_VERSION}_${RUN_STAMP}_poscar"
mkdir -p "$OUTPUT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ ! -d "$REPO_ROOT/src/findspingroup" ]]; then
  echo "Repository root does not look valid: $REPO_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$PYTHON_BIN"
  -m
  findspingroup.batch_poscar_roundtrip
  "$INPUT_DIR"
  --output-dir
  "$OUTPUT_DIR"
  --space-tol
  "$SPACE_TOL"
  --mtol
  "$MTOL"
  --meigtol
  "$MEIGTOL"
  --matrix-tol
  "$MATRIX_TOL"
  --workers
  "$WORKERS"
  --source-mode
  "$SOURCE_MODE"
  --compare-mode
  "$COMPARE_MODE"
  --compare-atol
  "$COMPARE_ATOL"
  --compare-rtol
  "$COMPARE_RTOL"
  --max-differences-per-case
  "$MAX_DIFFERENCES_PER_CASE"
)

if [[ "$COMPARE_CONF" == "0" ]]; then
  CMD+=(--no-compare-conf)
fi

if [[ "$SAVE_POSCAR" == "0" ]]; then
  CMD+=(--no-save-poscar)
fi

if [[ "$QUIET" != "0" ]]; then
  CMD+=(--quiet)
fi

echo "Repo root   : $REPO_ROOT"
echo "Input dir   : $INPUT_DIR"
echo "Output root : $OUTPUT_ROOT"
echo "Run dir     : $OUTPUT_DIR"
echo "Tolerances  : space=$SPACE_TOL mtol=$MTOL meigtol=$MEIGTOL matrix=$MATRIX_TOL"
echo "Source mode : $SOURCE_MODE"
echo "Compare mode: $COMPARE_MODE"
echo "Compare conf: $COMPARE_CONF"
echo "Compare atol: $COMPARE_ATOL"
echo "Compare rtol: $COMPARE_RTOL"
echo "Max diffs   : $MAX_DIFFERENCES_PER_CASE"
echo "Save poscar : $SAVE_POSCAR"
echo "Quiet       : $QUIET"
echo "Workers     : $WORKERS"
echo "Python      : $PYTHON_BIN"
echo "Start time  : $(date '+%Y-%m-%d %H:%M:%S')"

"${CMD[@]}"

echo "Finished at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Summary     : $OUTPUT_DIR/summary.json"
