from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from .batch_mcif import (
    _append_jsonl,
    _dedupe_sorted,
    _discover_mcif_files,
    _normalize_case_id,
    _read_manifest,
    _source_fractional_occupancy_annotation,
    _write_json,
)
from .find_spin_group import NumpyEncoder, find_spin_group, find_spin_group_from_data
from .io import parse_scif_text
from .version import __version__


def _isoformat_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _run_tag_from_isoformat(timestamp: str) -> str:
    return f"scif_roundtrip_v{__version__}_{dt.datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')}"


def _roundtrip_from_scif_text(
    *,
    source_name: str,
    scif_text: str,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
):
    lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_text(scif_text)
    return find_spin_group_from_data(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )


def _profile_result_payload(original, roundtrip, *, compare_conf: bool) -> tuple[dict, list[dict]]:
    differences = []
    if roundtrip.index != original.index:
        differences.append(
            {
                "field": "index",
                "expected": original.index,
                "actual": roundtrip.index,
            }
        )
    if compare_conf and roundtrip.conf != original.conf:
        differences.append(
            {
                "field": "conf",
                "expected": original.conf,
                "actual": roundtrip.conf,
            }
        )
    payload = {
        "original_index": original.index,
        "roundtrip_index": roundtrip.index,
        "index_match": roundtrip.index == original.index,
        "original_conf": original.conf,
        "roundtrip_conf": roundtrip.conf,
        "conf_match": roundtrip.conf == original.conf,
        "differences": differences,
    }
    return payload, differences


def _run_scif_roundtrip_case(
    index: int,
    total_cases: int,
    source_path: Path,
    *,
    output_root: Path,
    compare_conf: bool,
    save_scif: bool,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
) -> dict:
    case_start = time.perf_counter()
    original_duration = None
    roundtrip_duration = None
    case_id = _normalize_case_id(source_path)
    file_name = source_path.name
    occupancy_annotation = {
        "source_has_fractional_occupancy": None,
        "source_occupancy_values": None,
        "source_fractional_occupancy_values": None,
        "source_fractional_occupancy_site_count": None,
    }
    try:
        original_start = time.perf_counter()
        original = find_spin_group(
            str(source_path),
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
        )
        original_duration = round(time.perf_counter() - original_start, 6)
        occupancy_annotation = _source_fractional_occupancy_annotation(original)
        scif_text = original.scif
        if save_scif:
            scif_dir = output_root / "scif"
            scif_dir.mkdir(parents=True, exist_ok=True)
            scif_path = scif_dir / f"{source_path.stem}.scif"
            scif_path.write_text(scif_text, encoding="utf-8")

        roundtrip_start = time.perf_counter()
        roundtrip = _roundtrip_from_scif_text(
            source_name=f"{case_id}::scif",
            scif_text=scif_text,
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
        )
        roundtrip_duration = round(time.perf_counter() - roundtrip_start, 6)
        payload, differences = _profile_result_payload(
            original,
            roundtrip,
            compare_conf=compare_conf,
        )

        record = {
            "case_id": case_id,
            "file_name": file_name,
            "status": "ok" if not differences else "mismatch",
            "duration_seconds": round(time.perf_counter() - case_start, 6),
            "original_duration_seconds": original_duration,
            "roundtrip_duration_seconds": roundtrip_duration,
            "original": {
                "index": original.index,
                "conf": original.conf,
            },
            "roundtrip": payload,
        }
        record.update(occupancy_annotation)
        mismatch = None
        if differences:
            mismatch = {
                "case_id": case_id,
                "file_name": file_name,
                "original": record["original"],
                "roundtrip": {
                    "differences": differences,
                    **payload,
                },
                **occupancy_annotation,
            }
        return {
            "index": index,
            "case_id": case_id,
            "file_name": file_name,
            "status": record["status"],
            "record": record,
            "mismatch": mismatch,
            "error_by_file": None,
            "source_has_fractional_occupancy": bool(
                occupancy_annotation["source_has_fractional_occupancy"]
            ),
            "progress_message": f"[{index}/{total_cases}] {case_id} {record['status']}",
        }
    except Exception as exc:  # pragma: no cover - exercised in batch mode
        error_by_file = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            **occupancy_annotation,
        }
        record = {
            "case_id": case_id,
            "file_name": file_name,
            "status": "error",
            "duration_seconds": round(time.perf_counter() - case_start, 6),
            "original_duration_seconds": original_duration,
            "roundtrip_duration_seconds": roundtrip_duration,
            **occupancy_annotation,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        return {
            "index": index,
            "case_id": case_id,
            "file_name": file_name,
            "status": "error",
            "record": record,
            "mismatch": None,
            "error_by_file": error_by_file,
            "source_has_fractional_occupancy": bool(
                occupancy_annotation["source_has_fractional_occupancy"]
            ),
            "progress_message": f"[{index}/{total_cases}] {case_id} error",
        }


def _iter_scif_roundtrip_case_results(
    files: list[Path],
    *,
    workers: int,
    output_root: Path,
    compare_conf: bool,
    save_scif: bool,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
):
    total_cases = len(files)
    case_kwargs = {
        "output_root": output_root,
        "compare_conf": compare_conf,
        "save_scif": save_scif,
        "space_tol": space_tol,
        "mtol": mtol,
        "meigtol": meigtol,
        "matrix_tol": matrix_tol,
    }
    if workers == 1:
        for case_index, file_path in enumerate(files, start=1):
            yield _run_scif_roundtrip_case(case_index, total_cases, file_path, **case_kwargs)
        return

    next_index = 1
    next_yield_index = 1
    pending = set()
    completed: dict[int, dict] = {}
    max_pending = max(workers, workers * 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while next_index <= total_cases or pending:
            while next_index <= total_cases and len(pending) < max_pending:
                file_path = files[next_index - 1]
                pending.add(
                    executor.submit(
                        _run_scif_roundtrip_case,
                        next_index,
                        total_cases,
                        file_path,
                        **case_kwargs,
                    )
                )
                next_index += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                completed[int(result["index"])] = result
            while next_yield_index in completed:
                yield completed.pop(next_yield_index)
                next_yield_index += 1


def run_scif_roundtrip_batch(
    files: list[Path],
    output_dir: Path | str,
    *,
    compare_conf: bool = True,
    save_scif: bool = True,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    workers: int = 1,
    quiet: bool = False,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = _isoformat_now()
    run_tag = _run_tag_from_isoformat(started_at)
    records_path = output_root / "records.jsonl"
    mismatches_path = output_root / "mismatches.json"
    errors_path = output_root / "errors_by_file.json"
    summary_path = output_root / "summary.json"
    records_path.write_text("", encoding="utf-8")

    processed_cases = 0
    success_count = 0
    mismatch_count = 0
    error_count = 0
    fractional_occupancy_case_count = 0
    fractional_occupancy_mismatch_count = 0
    fractional_occupancy_error_count = 0
    mismatches = []
    errors_by_file = {}

    for case_result in _iter_scif_roundtrip_case_results(
        files,
        workers=workers,
        output_root=output_root,
        compare_conf=compare_conf,
        save_scif=save_scif,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    ):
        processed_cases += 1
        if not quiet:
            print(case_result["progress_message"])
        if case_result["source_has_fractional_occupancy"]:
            fractional_occupancy_case_count += 1
        _append_jsonl(records_path, case_result["record"])

        if case_result["status"] == "ok":
            success_count += 1
        elif case_result["status"] == "mismatch":
            mismatch_count += 1
            if case_result["source_has_fractional_occupancy"]:
                fractional_occupancy_mismatch_count += 1
            mismatches.append(case_result["mismatch"])
        else:
            error_count += 1
            errors_by_file[case_result["case_id"]] = case_result["error_by_file"]
            if case_result["source_has_fractional_occupancy"]:
                fractional_occupancy_error_count += 1

    finished_at = _isoformat_now()
    summary = {
        "schema_version": 1,
        "package_version": __version__,
        "run_tag": run_tag,
        "started_at": started_at,
        "finished_at": finished_at,
        "output_dir": str(output_root),
        "compare_conf": compare_conf,
        "save_scif": save_scif,
        "scif_output_dir": str(output_root / "scif") if save_scif else None,
        "tolerances": {
            "space_tol": space_tol,
            "mtol": mtol,
            "meigtol": meigtol,
            "matrix_tol": matrix_tol,
        },
        "workers": workers,
        "total_cases_requested": len(files),
        "processed_cases": processed_cases,
        "success_count": success_count,
        "mismatch_count": mismatch_count,
        "error_count": error_count,
        "fractional_occupancy_case_count": fractional_occupancy_case_count,
        "fractional_occupancy_mismatch_count": fractional_occupancy_mismatch_count,
        "fractional_occupancy_error_count": fractional_occupancy_error_count,
        "exit_code": 0 if mismatch_count == 0 and error_count == 0 else 1,
    }

    _write_json(mismatches_path, mismatches)
    _write_json(errors_path, errors_by_file)
    _write_json(summary_path, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-test MCIF -> SCIF -> parser -> re-identify roundtrip invariants."
    )
    parser.add_argument("inputs", nargs="*", help="Input .mcif files or directories.")
    parser.add_argument("--manifest", type=Path, help="Manifest file listing .mcif inputs.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for .mcif files.")
    parser.add_argument("--limit", type=int, help="Only process the first N resolved files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary and artifacts.")
    parser.add_argument("--no-compare-conf", action="store_true", help="Only require index equality.")
    parser.set_defaults(save_scif=True)
    parser.add_argument(
        "--save-scif",
        dest="save_scif",
        action="store_true",
        help="Save generated SCIF files under the output directory (default).",
    )
    parser.add_argument(
        "--no-save-scif",
        dest="save_scif",
        action="store_false",
        help="Do not save generated SCIF files.",
    )
    parser.add_argument(
        "--space-tol",
        "--space_tol",
        dest="space_tol",
        type=float,
        default=0.02,
    )
    parser.add_argument("--mtol", type=float, default=0.02)
    parser.add_argument("--meigtol", type=float, default=0.00002)
    parser.add_argument(
        "--matrix-tol",
        "--matrix_tol",
        dest="matrix_tol",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--workers",
        "--parallel-cores",
        dest="workers",
        type=int,
        default=1,
        help="Number of independent roundtrip cases to process concurrently. Default is 1.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _resolve_input_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    if args.manifest is not None:
        files.extend(_read_manifest(args.manifest))
    if args.inputs:
        files.extend(_discover_mcif_files(args.inputs, recursive=args.recursive))
    resolved = _dedupe_sorted(files)
    if args.limit is not None:
        resolved = resolved[: args.limit]
    if not resolved:
        raise ValueError("No .mcif inputs resolved for SCIF roundtrip batch.")
    return resolved


def main() -> None:
    args = _parse_args()
    files = _resolve_input_files(args)
    summary = run_scif_roundtrip_batch(
        files,
        args.output_dir,
        compare_conf=not args.no_compare_conf,
        save_scif=args.save_scif,
        space_tol=args.space_tol,
        mtol=args.mtol,
        meigtol=args.meigtol,
        matrix_tol=args.matrix_tol,
        workers=args.workers,
        quiet=args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, cls=NumpyEncoder))
    raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
