#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from findspingroup.batch_mcif import _compare_cases
from findspingroup.find_spin_group import NumpyEncoder
from findspingroup.version import __version__


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n"
            )


def _record_key(record: dict) -> tuple[str, str]:
    return (str(record.get("case_id") or ""), str(record.get("file_name") or ""))


_CASE_ID_RE = re.compile(r'"case_id"\s*:\s*("(?:\\.|[^"\\])*")')
_FILE_NAME_RE = re.compile(r'"file_name"\s*:\s*("(?:\\.|[^"\\])*")')


def _line_record_key(line: str) -> tuple[str, str]:
    """Extract JSONL sort key without materializing the full result payload."""

    case_match = _CASE_ID_RE.search(line)
    file_match = _FILE_NAME_RE.search(line)
    if case_match is None or file_match is None:
        record = json.loads(line)
        return _record_key(record)
    return (json.loads(case_match.group(1)), json.loads(file_match.group(1)))


def _iter_jsonl_lines(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def _write_full_results_stream(output_path: Path, shard_dirs: list[Path]) -> bool:
    keyed_lines: list[tuple[tuple[str, str], str]] = []
    for shard_dir in shard_dirs:
        path = shard_dir / "full_results.jsonl"
        if not path.exists():
            continue
        for line in _iter_jsonl_lines(path):
            keyed_lines.append((_line_record_key(line), line))
    if not keyed_lines:
        return False
    keyed_lines.sort(key=lambda item: item[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _key, line in keyed_lines:
            handle.write(line if line.endswith("\n") else f"{line}\n")
    return True


def _discover_shard_dirs(root: Path, shard_glob: str) -> list[Path]:
    shard_dirs = [
        path
        for path in root.glob(shard_glob)
        if path.is_dir() and (path / "summary.json").exists()
    ]
    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.name.rsplit("_", 1)[1]), path.name)
        except (IndexError, ValueError):
            return (10**9, path.name)

    return sorted(shard_dirs, key=sort_key)


def _comparison_key(original_key: str, record: dict, key_mode: str) -> str:
    if key_mode == "case_id":
        return original_key
    if key_mode == "file_name":
        return str(record.get("file_name") or Path(original_key).name)
    raise ValueError(f"Unsupported comparison key mode: {key_mode}")


def _rekey_cases(cases: dict[str, dict], key_mode: str) -> dict[str, dict]:
    if key_mode == "case_id":
        return cases
    rekeyed: dict[str, dict] = {}
    duplicates: set[str] = set()
    for original_key, record in cases.items():
        key = _comparison_key(original_key, record, key_mode)
        if key in rekeyed:
            duplicates.add(key)
        normalized_record = dict(record)
        normalized_record["case_id"] = key
        rekeyed[key] = normalized_record
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates)[:8])
        raise ValueError(f"Duplicate comparison keys for {key_mode}: {duplicate_text}")
    return rekeyed


def _aggregate_existing_comparisons(shard_dirs: list[Path]) -> dict | None:
    comparisons = []
    for shard_dir in shard_dirs:
        path = shard_dir / "comparison.json"
        if path.exists():
            comparisons.append(_read_json(path))
    if not comparisons:
        return None
    mismatches = []
    missing_in_baseline = []
    tensor_summary_backfills = []
    protected_ok_mismatches = []
    error_to_ok_updates = []
    new_cases = []
    for comparison in comparisons:
        mismatches.extend(comparison.get("mismatches", []))
        missing_in_baseline.extend(comparison.get("missing_in_baseline", []))
        tensor_summary_backfills.extend(comparison.get("tensor_summary_backfills", []))
        protected_ok_mismatches.extend(comparison.get("protected_ok_mismatches", []))
        error_to_ok_updates.extend(comparison.get("error_to_ok_updates", []))
        new_cases.extend(comparison.get("new_cases", []))
    payload = {
        "checked_case_count": sum(int(c.get("checked_case_count", 0) or 0) for c in comparisons),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "missing_in_baseline_count": len(missing_in_baseline),
        "missing_in_baseline": missing_in_baseline,
        "tensor_summary_backfill_count": len(tensor_summary_backfills),
        "tensor_summary_backfills": tensor_summary_backfills,
    }
    if protected_ok_mismatches:
        payload["protected_ok_mismatch_count"] = len(protected_ok_mismatches)
        payload["protected_ok_mismatches"] = protected_ok_mismatches
    if error_to_ok_updates:
        payload["error_to_ok_update_count"] = len(error_to_ok_updates)
        payload["error_to_ok_updates"] = error_to_ok_updates
    if new_cases:
        payload["new_case_count"] = len(new_cases)
        payload["new_cases"] = new_cases
    return payload


def merge_batch_shards(
    root: Path,
    output_dir: Path,
    *,
    shard_glob: str = "shard_*",
    baseline_path: Path | None = None,
    full_results_mode: str = "stream",
    comparison_key: str = "case_id",
) -> dict:
    started = time.perf_counter()
    root = root.resolve()
    output_dir = output_dir.resolve()
    shard_dirs = _discover_shard_dirs(root, shard_glob)
    if not shard_dirs:
        raise ValueError(f"No shard directories with summary.json matched {root / shard_glob}")

    summaries = [_read_json(shard_dir / "summary.json") for shard_dir in shard_dirs]
    records: list[dict] = []
    baseline_cases: dict[str, dict] = {}
    errors_by_file: dict[str, dict] = {}

    for shard_dir in shard_dirs:
        records.extend(_iter_jsonl(shard_dir / "records.jsonl"))
        baseline_file = shard_dir / "baseline.json"
        if baseline_file.exists():
            baseline_cases.update(_read_json(baseline_file))
        errors_file = shard_dir / "errors_by_file.json"
        if errors_file.exists():
            errors_by_file.update(_read_json(errors_file))

    records.sort(key=_record_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_json(output_dir / "baseline.json", baseline_cases)
    _write_json(output_dir / "errors_by_file.json", errors_by_file)

    comparison = None
    if baseline_path is not None:
        comparison = _compare_cases(
            _rekey_cases(baseline_cases, comparison_key),
            _rekey_cases(_read_json(baseline_path), comparison_key),
        )
    else:
        comparison = _aggregate_existing_comparisons(shard_dirs)
    if comparison is not None:
        _write_json(output_dir / "comparison.json", comparison)

    full_results_written = False
    if full_results_mode == "stream":
        full_results_written = _write_full_results_stream(
            output_dir / "full_results.jsonl",
            shard_dirs,
        )
    elif full_results_mode == "skip":
        full_results_written = False
    else:
        raise ValueError(f"Unsupported full_results_mode: {full_results_mode}")

    processed_cases = len(records)
    success_count = sum(1 for record in records if record.get("status") == "ok")
    error_count = processed_cases - success_count
    summary = {
        "package_version": __version__,
        "source_root": root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "shard_glob": shard_glob,
        "shard_count": len(shard_dirs),
        "shards": [path.as_posix() for path in shard_dirs],
        "processed_cases": processed_cases,
        "success_count": success_count,
        "error_count": error_count,
        "records_jsonl": (output_dir / "records.jsonl").as_posix(),
        "full_results_jsonl": (output_dir / "full_results.jsonl").as_posix()
        if full_results_written
        else None,
        "baseline_path": baseline_path.resolve().as_posix() if baseline_path else None,
        "comparison_key": comparison_key,
        "full_results_mode": full_results_mode,
        "comparison": comparison,
        "source_summary_duration_seconds_sum": round(
            sum(float(summary.get("duration_seconds", 0) or 0) for summary in summaries),
            6,
        ),
        "source_summary_duration_seconds_max": round(
            max(float(summary.get("duration_seconds", 0) or 0) for summary in summaries),
            6,
        ),
        "duration_seconds": round(time.perf_counter() - started, 6),
    }
    exit_code = 0
    if error_count:
        exit_code = 1
    if comparison and (
        comparison.get("missing_in_baseline_count", 0)
        or comparison.get("mismatch_count", 0)
        or comparison.get("protected_ok_mismatch_count", 0)
    ):
        exit_code = 1
    summary["exit_code"] = exit_code
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FindSpinGroup shard_* batch outputs into one compact run directory."
    )
    parser.add_argument("root", type=Path, help="Root directory containing shard_* subdirectories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Merged output directory. Defaults to <root>/merged.",
    )
    parser.add_argument(
        "--shard-glob",
        default="shard_*",
        help="Glob used under root to discover shard directories.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline.json used to recompute comparison after merging.",
    )
    parser.add_argument(
        "--full-results",
        choices=("stream", "skip"),
        default="stream",
        help=(
            "How to handle shard full_results.jsonl. 'stream' preserves a merged sorted "
            "full_results file without materializing JSON objects; 'skip' writes only "
            "compact artifacts."
        ),
    )
    parser.add_argument(
        "--comparison-key",
        choices=("case_id", "file_name"),
        default="case_id",
        help=(
            "Key used for recomputing --baseline comparison. Use file_name when comparing "
            "equivalent runs from different absolute path roots."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or (args.root / "merged")
    summary = merge_batch_shards(
        args.root,
        output_dir,
        shard_glob=args.shard_glob,
        baseline_path=args.baseline,
        full_results_mode=args.full_results,
        comparison_key=args.comparison_key,
    )
    print(f"Merged {summary['processed_cases']} records from {summary['shard_count']} shards")
    print(f"Output: {summary['output_dir']}")
    raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
