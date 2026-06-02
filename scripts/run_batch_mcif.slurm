#!/bin/bash
#SBATCH --job-name=fsg_batch
#SBATCH --partition=128G24c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: sbatch $0 <mcif_dir> <output_root> [baseline_suite] [baseline_root]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${FSG_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
INPUT_DIR="$1"
OUTPUT_ROOT="$2"
BASELINE_SUITE="${3:-}"
BASELINE_ROOT="${4:-$REPO_ROOT/batch_baselines}"
EXPORT_TXT="${EXPORT_TXT:-}"
EXPORT_FIELDS="${EXPORT_FIELDS:-}"
SPACE_TOL="${SPACE_TOL:-0.02}"
MTOL="${MTOL:-0.02}"
MEIGTOL="${MEIGTOL:-0.00002}"
MATRIX_TOL="${MATRIX_TOL:-0.01}"
BATCH_ROUTE="${BATCH_ROUTE:-full}"
CALCULATION_MODE="${CALCULATION_MODE:-auto}"
VACUUM_AXIS="${VACUUM_AXIS:-}"
EXPORT_RUNTIME_ROWS="${EXPORT_RUNTIME_ROWS:-1}"
EXPORT_ROWS_JSONL_NAME="${EXPORT_ROWS_JSONL_NAME:-export_rows.jsonl}"
EXPORT_ROWS_XLSX_NAME="${EXPORT_ROWS_XLSX_NAME:-export_rows.xlsx}"
EXPORT_FERROELECTRIC_SWITCHING_OPS="${EXPORT_FERROELECTRIC_SWITCHING_OPS:-0}"
FERROELECTRIC_SWITCHING_OPS_JSONL_NAME="${FERROELECTRIC_SWITCHING_OPS_JSONL_NAME:-ferroelectric_switching_ops.jsonl}"
FERROELECTRIC_SWITCHING_OPS_CSV_NAME="${FERROELECTRIC_SWITCHING_OPS_CSV_NAME:-ferroelectric_switching_ops.csv}"
FERROELECTRIC_SWITCHING_OPS_XLSX_NAME="${FERROELECTRIC_SWITCHING_OPS_XLSX_NAME:-ferroelectric_switching_ops.xlsx}"
FERROELECTRIC_SWITCHING_OPS_SUMMARY_NAME="${FERROELECTRIC_SWITCHING_OPS_SUMMARY_NAME:-ferroelectric_switching_ops_summary.json}"
INCLUDE_G0_SELF_AUDIT="${INCLUDE_G0_SELF_AUDIT:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
BATCH_WORKERS="${BATCH_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
LIMIT="${LIMIT:-}"

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
OUTPUT_DIR="${OUTPUT_ROOT}/run_v${PACKAGE_VERSION}_${RUN_STAMP}"
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
  findspingroup.batch_mcif
  "$INPUT_DIR"
  --output-dir
  "$OUTPUT_DIR"
  --auto-baseline
  --baseline-root
  "$BASELINE_ROOT"
  --space-tol
  "$SPACE_TOL"
  --mtol
  "$MTOL"
  --meigtol
  "$MEIGTOL"
  --matrix-tol
  "$MATRIX_TOL"
  --route
  "$BATCH_ROUTE"
  --calculation-mode
  "$CALCULATION_MODE"
  --shard-index
  "$SHARD_INDEX"
  --shard-count
  "$SHARD_COUNT"
  --workers
  "$BATCH_WORKERS"
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(
    --limit
    "$LIMIT"
  )
fi

if [[ -n "$VACUUM_AXIS" ]]; then
  CMD+=(
    --vacuum-axis
    "$VACUUM_AXIS"
  )
fi

if [[ -n "$BASELINE_SUITE" ]]; then
  CMD+=(
    --baseline-suite
    "$BASELINE_SUITE"
  )
fi

if [[ -n "$EXPORT_TXT" ]]; then
  CMD+=(
    --export-txt
    "$EXPORT_TXT"
  )
fi

if [[ -n "$EXPORT_FIELDS" ]]; then
  IFS=',' read -r -a EXPORT_FIELD_ARRAY <<< "$EXPORT_FIELDS"
  for raw_field in "${EXPORT_FIELD_ARRAY[@]}"; do
    field="${raw_field#"${raw_field%%[![:space:]]*}"}"
    field="${field%"${field##*[![:space:]]}"}"
    if [[ -n "$field" ]]; then
      CMD+=(
        --export-field
        "$field"
      )
    fi
  done
fi

if [[ "$INCLUDE_G0_SELF_AUDIT" == "1" ]]; then
  CMD+=(
    --include-g0-self-audit
  )
fi

echo "Repo root   : $REPO_ROOT"
echo "Input dir   : $INPUT_DIR"
echo "Output root : $OUTPUT_ROOT"
echo "Run dir     : $OUTPUT_DIR"
echo "Baseline root: $BASELINE_ROOT"
if [[ -n "$BASELINE_SUITE" ]]; then
  echo "Baseline suite: $BASELINE_SUITE"
else
  echo "Baseline suite: <auto from input dir>"
fi
echo "Tolerances  : space=$SPACE_TOL mtol=$MTOL meigtol=$MEIGTOL matrix=$MATRIX_TOL"
echo "Batch route : $BATCH_ROUTE"
echo "Calculation mode: $CALCULATION_MODE"
if [[ -n "$VACUUM_AXIS" ]]; then
  echo "Vacuum axis : $VACUUM_AXIS"
else
  echo "Vacuum axis : <auto>"
fi
echo "Shard       : $SHARD_INDEX / $SHARD_COUNT"
echo "Workers     : $BATCH_WORKERS"
if [[ -n "$LIMIT" ]]; then
  echo "Limit       : $LIMIT"
else
  echo "Limit       : <none>"
fi
if [[ -n "$EXPORT_TXT" ]]; then
  echo "Export txt  : $EXPORT_TXT"
else
  echo "Export txt  : <none>"
fi
if [[ -n "$EXPORT_FIELDS" ]]; then
  echo "Export fields: $EXPORT_FIELDS"
else
  echo "Export fields: <none>"
fi
echo "G0 self audit: $INCLUDE_G0_SELF_AUDIT"
echo "Python      : $PYTHON_BIN"
echo "Start time  : $(date '+%Y-%m-%d %H:%M:%S')"

"${CMD[@]}"

RUNTIME_JSONL="$OUTPUT_DIR/full_results.jsonl"
EXPORT_ROWS_JSONL="$OUTPUT_DIR/$EXPORT_ROWS_JSONL_NAME"
EXPORT_ROWS_XLSX="$OUTPUT_DIR/$EXPORT_ROWS_XLSX_NAME"

if [[ "$EXPORT_RUNTIME_ROWS" != "0" ]]; then
  if [[ -f "$RUNTIME_JSONL" ]]; then
    EXPORT_CMD=(
      "$PYTHON_BIN"
      "$REPO_ROOT/scripts/export_mcif_results_to_excel.py"
      --runtime-jsonl
      "$RUNTIME_JSONL"
      --output-jsonl
      "$EXPORT_ROWS_JSONL"
      --output-xlsx
      "$EXPORT_ROWS_XLSX"
    )
    echo "Runtime export:"
    printf '  %q' "${EXPORT_CMD[@]}"
    echo
    if ! "${EXPORT_CMD[@]}"; then
      echo "WARNING: runtime Excel/JSONL export failed; batch results remain in $OUTPUT_DIR" >&2
    fi
  else
    echo "WARNING: runtime export skipped because $RUNTIME_JSONL was not found" >&2
  fi
fi

if [[ "$EXPORT_FERROELECTRIC_SWITCHING_OPS" != "0" ]]; then
  if [[ -f "$RUNTIME_JSONL" ]]; then
    SWITCHING_EXPORT_CMD=(
      "$PYTHON_BIN"
      "$REPO_ROOT/scripts/export_ferroelectric_switching_ops.py"
      --runtime-jsonl
      "$RUNTIME_JSONL"
      --output-jsonl
      "$OUTPUT_DIR/$FERROELECTRIC_SWITCHING_OPS_JSONL_NAME"
      --output-csv
      "$OUTPUT_DIR/$FERROELECTRIC_SWITCHING_OPS_CSV_NAME"
      --output-xlsx
      "$OUTPUT_DIR/$FERROELECTRIC_SWITCHING_OPS_XLSX_NAME"
      --output-summary
      "$OUTPUT_DIR/$FERROELECTRIC_SWITCHING_OPS_SUMMARY_NAME"
    )
    echo "Ferroelectric switching operation export:"
    printf '  %q' "${SWITCHING_EXPORT_CMD[@]}"
    echo
    if ! "${SWITCHING_EXPORT_CMD[@]}"; then
      echo "WARNING: ferroelectric switching operation export failed; batch results remain in $OUTPUT_DIR" >&2
    fi
  else
    echo "WARNING: ferroelectric switching export skipped because $RUNTIME_JSONL was not found" >&2
  fi
fi

echo "Finished at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Summary     : $OUTPUT_DIR/summary.json"
if [[ -f "$EXPORT_ROWS_JSONL" ]]; then
  echo "Export rows : $EXPORT_ROWS_JSONL"
fi
if [[ -f "$EXPORT_ROWS_XLSX" ]]; then
  echo "Export xlsx : $EXPORT_ROWS_XLSX"
fi
if [[ "$EXPORT_FERROELECTRIC_SWITCHING_OPS" != "0" ]]; then
  echo "Switching ops: $OUTPUT_DIR/$FERROELECTRIC_SWITCHING_OPS_JSONL_NAME"
fi
