from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np

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
from .io import parse_poscar_file
from .structure.cell import CrystalCell
from .version import __version__


POSCAR_SOURCE_ACC_PRIMITIVE = "acc_primitive"
POSCAR_SOURCE_G0_CPTRANS_CANDIDATE = "g0_cptrans_candidate"
POSCAR_SOURCE_CHOICES = (
    POSCAR_SOURCE_ACC_PRIMITIVE,
    POSCAR_SOURCE_G0_CPTRANS_CANDIDATE,
)


def _crystal_cell_from_snapshot(snapshot: dict) -> CrystalCell:
    return CrystalCell(
        snapshot["lattice"],
        snapshot["positions"],
        snapshot["occupancies"],
        snapshot["elements"],
        snapshot["moments"],
        spin_setting="in_lattice",
    )


def _source_poscar_payload(original, *, case_id: str, source_mode: str) -> tuple[str, str, str]:
    if source_mode == POSCAR_SOURCE_ACC_PRIMITIVE:
        poscar_text = original.acc_primitive_magnetic_cell_poscar
        if poscar_text is None:
            raise ValueError("Missing acc_primitive_magnetic_cell_poscar output.")
        return (
            f"{case_id}::acc_primitive_magnetic.POSCAR",
            poscar_text,
            "repo_generated_acc_primitive_magnetic_poscar",
        )

    if source_mode == POSCAR_SOURCE_G0_CPTRANS_CANDIDATE:
        if original.g0_standard_cell is None:
            raise ValueError("Missing g0_standard_cell output.")
        if original.T_convention_to_acc_primitive is None:
            raise ValueError("Missing T_convention_to_acc_primitive output.")
        candidate_cell = _crystal_cell_from_snapshot(original.g0_standard_cell).transform(
            np.asarray(original.T_convention_to_acc_primitive[0], dtype=float),
            np.asarray(original.T_convention_to_acc_primitive[1], dtype=float),
        )
        source_name = f"{case_id}::g0_cptrans_candidate_magnetic.POSCAR"
        return (
            source_name,
            candidate_cell.to_poscar(Path(source_name).name),
            "repo_generated_g0_cptrans_candidate_magnetic_poscar",
        )

    raise ValueError(f"Unsupported POSCAR source_mode: {source_mode}")


def _isoformat_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _run_tag_from_isoformat(timestamp: str) -> str:
    return f"poscar_roundtrip_v{__version__}_{dt.datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')}"


def _roundtrip_from_poscar_text(
    *,
    source_name: str,
    poscar_text: str,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
    output_dir: Path,
    save_poscar: bool,
):
    if save_poscar:
        poscar_path = output_dir / "poscar" / source_name
        poscar_path.parent.mkdir(parents=True, exist_ok=True)
        poscar_path.write_text(poscar_text, encoding="utf-8")
        parse_target = poscar_path
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".POSCAR", delete=False, encoding="utf-8") as handle:
            handle.write(poscar_text)
            parse_target = Path(handle.name)
        try:
            lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(parse_target)
        finally:
            parse_target.unlink(missing_ok=True)
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

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(parse_target)
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


def _result_payload(original, roundtrip, *, compare_conf: bool) -> tuple[dict, list[dict]]:
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


def run_poscar_roundtrip_batch(
    files: list[Path],
    output_dir: Path | str,
    *,
    source_mode: str = POSCAR_SOURCE_ACC_PRIMITIVE,
    compare_conf: bool = True,
    save_poscar: bool = True,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    quiet: bool = False,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = _isoformat_now()
    run_tag = _run_tag_from_isoformat(started_at)

    records_path = output_root / "records.jsonl"
    mismatches_path = output_root / "mismatches.json"
    errors_path = output_root / "errors_by_file.json"
    summary_path = output_root / "summary.json"

    processed_cases = 0
    success_count = 0
    mismatch_count = 0
    error_count = 0
    fractional_occupancy_case_count = 0
    fractional_occupancy_mismatch_count = 0
    fractional_occupancy_error_count = 0
    mismatches = []
    errors_by_file = {}
    input_format = None

    for source_path in files:
        processed_cases += 1
        case_id = _normalize_case_id(source_path)
        file_name = source_path.name
        occupancy_annotation = {
            "source_has_fractional_occupancy": None,
            "source_occupancy_values": None,
            "source_fractional_occupancy_values": None,
            "source_fractional_occupancy_site_count": None,
        }
        if not quiet:
            print(f"[{processed_cases}/{len(files)}] {case_id}")
        try:
            original = find_spin_group(
                str(source_path),
                space_tol=space_tol,
                mtol=mtol,
                meigtol=meigtol,
                matrix_tol=matrix_tol,
            )
            occupancy_annotation = _source_fractional_occupancy_annotation(original)
            if occupancy_annotation["source_has_fractional_occupancy"]:
                fractional_occupancy_case_count += 1
            source_name, poscar_text, record_input_format = _source_poscar_payload(
                original,
                case_id=case_id,
                source_mode=source_mode,
            )
            if input_format is None:
                input_format = record_input_format

            roundtrip = _roundtrip_from_poscar_text(
                source_name=source_name,
                poscar_text=poscar_text,
                space_tol=space_tol,
                mtol=mtol,
                meigtol=meigtol,
                matrix_tol=matrix_tol,
                output_dir=output_root,
                save_poscar=save_poscar,
            )
            payload, differences = _result_payload(original, roundtrip, compare_conf=compare_conf)

            record = {
                "case_id": case_id,
                "file_name": file_name,
                "status": "ok" if not differences else "mismatch",
                "original": {
                    "index": original.index,
                    "conf": original.conf,
                },
                "roundtrip": payload,
            }
            record.update(occupancy_annotation)
            _append_jsonl(records_path, record)

            if differences:
                mismatch_count += 1
                if occupancy_annotation["source_has_fractional_occupancy"]:
                    fractional_occupancy_mismatch_count += 1
                mismatches.append(record)
            else:
                success_count += 1
        except Exception as exc:  # pragma: no cover - exercised in batch mode
            error_count += 1
            errors_by_file[case_id] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                **occupancy_annotation,
            }
            if occupancy_annotation["source_has_fractional_occupancy"]:
                fractional_occupancy_error_count += 1
            _append_jsonl(
                records_path,
                {
                    "case_id": case_id,
                    "file_name": file_name,
                    "status": "error",
                    **occupancy_annotation,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )

    finished_at = _isoformat_now()
    summary = {
        "schema_version": 1,
        "package_version": __version__,
        "run_tag": run_tag,
        "started_at": started_at,
        "finished_at": finished_at,
        "output_dir": str(output_root),
        "input_format": input_format or "repo_generated_acc_primitive_magnetic_poscar",
        "source_mode": source_mode,
        "compare_conf": compare_conf,
        "save_poscar": save_poscar,
        "poscar_output_dir": str(output_root / "poscar") if save_poscar else None,
        "tolerances": {
            "space_tol": space_tol,
            "mtol": mtol,
            "meigtol": meigtol,
            "matrix_tol": matrix_tol,
        },
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
        description="Batch-test MCIF -> repo-generated magnetic POSCAR -> parser -> re-identify roundtrip invariants."
    )
    parser.add_argument("inputs", nargs="*", help="Input .mcif files or directories.")
    parser.add_argument("--manifest", type=Path, help="Manifest file listing .mcif inputs.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for .mcif files.")
    parser.add_argument("--limit", type=int, help="Only process the first N resolved files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary and artifacts.")
    parser.add_argument("--no-compare-conf", action="store_true", help="Only require index equality.")
    parser.add_argument(
        "--source-mode",
        choices=POSCAR_SOURCE_CHOICES,
        default=POSCAR_SOURCE_ACC_PRIMITIVE,
        help="Choose which repo-generated magnetic POSCAR path to roundtrip.",
    )
    parser.set_defaults(save_poscar=True)
    parser.add_argument(
        "--save-poscar",
        dest="save_poscar",
        action="store_true",
        help="Save generated POSCAR files under the output directory (default).",
    )
    parser.add_argument(
        "--no-save-poscar",
        dest="save_poscar",
        action="store_false",
        help="Do not save generated POSCAR files.",
    )
    parser.add_argument("--space-tol", type=float, default=0.02)
    parser.add_argument("--mtol", type=float, default=0.02)
    parser.add_argument("--meigtol", type=float, default=0.00002)
    parser.add_argument("--matrix-tol", type=float, default=0.01)
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
        raise ValueError("No .mcif inputs resolved for POSCAR roundtrip batch.")
    return resolved


def main() -> None:
    args = _parse_args()
    files = _resolve_input_files(args)
    summary = run_poscar_roundtrip_batch(
        files,
        args.output_dir,
        source_mode=args.source_mode,
        compare_conf=not args.no_compare_conf,
        save_poscar=args.save_poscar,
        space_tol=args.space_tol,
        mtol=args.mtol,
        meigtol=args.meigtol,
        matrix_tol=args.matrix_tol,
        quiet=args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, cls=NumpyEncoder))
    raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
