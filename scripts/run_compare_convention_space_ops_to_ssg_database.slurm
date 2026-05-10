#!/bin/bash
#SBATCH --job-name=fsg_ssgdb_cmp
#SBATCH --partition=128G24c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

set -euo pipefail

FULL_RESULTS_JSONL="${FULL_RESULTS_JSONL:?FULL_RESULTS_JSONL is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:?COMPARE_SCRIPT is required}"
SSG_DATABASE_ROOT="${SSG_DATABASE_ROOT:?SSG_DATABASE_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATABASE_COORDINATE_MODE="${DATABASE_COORDINATE_MODE:-raw}"
COMPARE_TOL="${COMPARE_TOL:-1e-6}"
MAX_DENOMINATOR="${MAX_DENOMINATOR:-48}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$COMPARE_SCRIPT" \
  --full-results-jsonl "$FULL_RESULTS_JSONL" \
  --ssg-database-root "$SSG_DATABASE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --database-coordinate-mode "$DATABASE_COORDINATE_MODE" \
  --tol "$COMPARE_TOL" \
  --max-denominator "$MAX_DENOMINATOR"
