#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import traceback
from pathlib import Path

from findspingroup import find_spin_group
from findspingroup.find_spin_group import _ossg_oriented_spin_frame_ssg
from findspingroup.structure import SpinSpaceGroup
from findspingroup.structure.cell import CrystalCell
from findspingroup.version import __version__


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_case_id(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_supported_mcif_path(path: Path) -> bool:
    return path.suffix.lower() == ".mcif" and not path.name.startswith("._")


def _discover_mcif_files(inputs: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw_input in inputs:
        candidate = Path(raw_input)
        if candidate.is_file():
            if _is_supported_mcif_path(candidate):
                files.append(candidate.resolve())
            continue
        if candidate.is_dir():
            iterator = candidate.rglob("*.mcif") if recursive else candidate.glob("*.mcif")
            files.extend(
                path.resolve()
                for path in iterator
                if path.is_file() and _is_supported_mcif_path(path)
            )
            continue
        raise FileNotFoundError(f"Input path not found: {raw_input}")
    unique = {path.as_posix(): path for path in files}
    return [unique[key] for key in sorted(unique)]


def _isoformat_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _build_acc_primitive_ossg(result):
    cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    return _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), cell)


def _build_record(source_path: Path, result) -> dict:
    acc_primitive_ossg = _build_acc_primitive_ossg(result)
    derived = acc_primitive_ossg.msg_info
    spglib_msg_num = result.msg_num
    spglib_msg_type = result.msg_type
    reconstructed_msg_num = None if derived is None else derived.get("msg_int_num")
    reconstructed_msg_type = None if derived is None else derived.get("msg_type")
    return {
        "case_id": _normalize_case_id(source_path),
        "file_name": source_path.name,
        "source_path": str(source_path),
        "index": result.index,
        "conf": result.conf,
        "spglib_msg_num": spglib_msg_num,
        "spglib_msg_type": spglib_msg_type,
        "reconstructed_msg_num": reconstructed_msg_num,
        "reconstructed_msg_type": reconstructed_msg_type,
        "reconstructed_msg_bns_num": None if derived is None else derived.get("msg_bns_num"),
        "reconstructed_msg_bns_symbol": None if derived is None else derived.get("msg_bns_symbol"),
        "reconstructed_mpg_symbol": None if derived is None else derived.get("mpg_symbol"),
        "acc_primitive_msg_ops_count": len(result.acc_primitive_msg_ops or []),
        "msg_spin_frame_setting": result.acc_primitive_msg_ops_spin_frame_setting,
        "msg_num_match": spglib_msg_num == reconstructed_msg_num,
        "msg_type_match": spglib_msg_type == reconstructed_msg_type,
        "status": "ok",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-check spglib MSG numbers against current OSSG-derived MSG numbers.",
    )
    parser.add_argument("inputs", nargs="+", help="Input .mcif file(s) or directory(ies).")
    parser.add_argument("--output-dir", required=True, help="Directory for records and summaries.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input directories.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of cases.")
    parser.add_argument("--space-tol", type=float, default=0.02)
    parser.add_argument("--mtol", type=float, default=0.02)
    parser.add_argument("--meigtol", type=float, default=0.00002)
    parser.add_argument("--matrix-tol", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _discover_mcif_files(args.inputs, recursive=args.recursive)
    if args.limit is not None:
        files = files[: args.limit]

    started_at = _isoformat_now()
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    mismatches_path = output_dir / "mismatches.json"
    errors_path = output_dir / "errors.json"

    if records_path.exists():
        records_path.unlink()

    processed_cases = 0
    success_count = 0
    mismatch_count = 0
    error_count = 0
    mismatches = []
    errors = []

    for source_path in files:
        processed_cases += 1
        case_started = time.time()
        try:
            result = find_spin_group(
                str(source_path),
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
            )
            record = _build_record(source_path, result)
            record["duration_seconds"] = round(time.time() - case_started, 6)
            _append_jsonl(records_path, record)
            success_count += 1
            if not record["msg_num_match"] or not record["msg_type_match"]:
                mismatch_count += 1
                mismatches.append(record)
        except Exception as exc:
            error_count += 1
            error_record = {
                "case_id": _normalize_case_id(source_path),
                "file_name": source_path.name,
                "source_path": str(source_path),
                "status": "error",
                "duration_seconds": round(time.time() - case_started, 6),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                },
            }
            _append_jsonl(records_path, error_record)
            errors.append(error_record)

    summary = {
        "schema_version": 1,
        "package_version": __version__,
        "started_at": started_at,
        "finished_at": _isoformat_now(),
        "processed_cases": processed_cases,
        "success_count": success_count,
        "mismatch_count": mismatch_count,
        "error_count": error_count,
        "input_paths": [str(Path(raw).resolve()) for raw in args.inputs],
        "tolerances": {
            "space_tol": args.space_tol,
            "mtol": args.mtol,
            "meigtol": args.meigtol,
            "matrix_tol": args.matrix_tol,
        },
        "comparison_contract": {
            "spglib_field": "result.msg_num",
            "derived_field": "acc_primitive_ossg.msg_info.msg_int_num",
            "type_field": "msg_type",
            "spin_frame_setting": "ossg_oriented_spin_frame",
        },
        "output_dir": str(output_dir),
    }

    _write_json(summary_path, summary)
    _write_json(mismatches_path, mismatches)
    _write_json(errors_path, errors)


if __name__ == "__main__":
    main()
