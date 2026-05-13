from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from .batch_mcif import (
    _append_jsonl,
    _dedupe_sorted,
    _discover_mcif_files,
    _normalize_case_id,
    _normalize_jsonable,
    _read_manifest,
    _source_fractional_occupancy_annotation,
    _write_json,
)
from .find_spin_group import (
    NumpyEncoder,
    find_spin_group,
    find_spin_group_from_data,
)
from .io import parse_poscar_file
from .structure.cell import CrystalCell
from .version import __version__


POSCAR_SOURCE_ACC_PRIMITIVE = "acc_primitive"
POSCAR_SOURCE_G0_CPTRANS_CANDIDATE = "g0_cptrans_candidate"
POSCAR_SOURCE_CHOICES = (
    POSCAR_SOURCE_ACC_PRIMITIVE,
    POSCAR_SOURCE_G0_CPTRANS_CANDIDATE,
)
COMPARE_MODE_BASIC = "basic"
COMPARE_MODE_FULL = "full"
COMPARE_MODE_CHOICES = (
    COMPARE_MODE_BASIC,
    COMPARE_MODE_FULL,
)

BASIC_COMPARE_ATTRS = (
    ("index", "index"),
    ("g0_symbol", "G0_symbol"),
    ("g0_number", "G0_num"),
    ("l0_symbol", "L0_symbol"),
    ("l0_number", "L0_num"),
    ("it", "it"),
    ("ik", "ik"),
    ("sspg", "SSPG_symbol_hm"),
    ("acc_symbol", "acc"),
    ("space_group_symbol", "input_space_group_symbol"),
    ("space_group_number", "input_space_group_number"),
    ("msg_symbol", "msg_symbol"),
    ("msg_bns_number", "msg_bns_number"),
    ("msg_og_number", "msg_og_number"),
    ("empg", "gspg_effective_mpg_symbol"),
    ("conf", "conf"),
    ("magnetic_phase", "magnetic_phase"),
    ("is_alter", "is_alter"),
    ("is_som", "is_spin_orbit_magnet"),
    ("sg_is_polar", "sg_is_polar"),
    ("sg_is_chiral", "sg_is_chiral"),
    ("ssg_is_polar", "ossg_is_polar"),
    ("ssg_is_chiral", "ossg_is_chiral"),
    ("msg_is_polar", "msg_is_polar"),
    ("msg_is_chiral", "msg_is_chiral"),
)

FULL_COMPARE_SOURCE_DEPENDENT_FIELDS = frozenset(
    {
        "input_ssg_ops",
        "T_input_to_ssg_std",
        "T_input_to_mag_primitive",
        "T_input_to_input_magnetic_primitive",
        "T_input_to_acc_primitive",
        "input_space_group_number",
        "input_space_group_symbol",
        "input_space_group_basis_or_setting",
        "sg_has_real_space_inversion",
        "sg_is_polar",
        "sg_is_chiral",
        "source_structure_metadata",
        "source_parent_space_group",
        "source_cell_parameter_strings",
        "input_magnetic_primitive_cell",
        "input_magnetic_primitive_cell_setting",
        "input_magnetic_primitive_cell_poscar",
        "input_magnetic_primitive_cell_detail",
        "input_magnetic_primitive_ssg_ops",
        "input_magnetic_primitive_ssg_setting",
        "input_magnetic_primitive_ssg_seitz",
        "input_magnetic_primitive_ssg_seitz_latex",
        "input_magnetic_primitive_ssg_seitz_descriptions",
        "input_magnetic_primitive_ssg_international_linear",
        "input_magnetic_primitive_ssg_international_latex",
        "input_magnetic_primitive_ssg_symbol_calibration_tol",
        "input_magnetic_primitive_ssg_type",
        "raw_T_input_to_G0std",
        "raw_T_input_to_L0std",
        "T_input_to_G0std",
        "T_input_to_L0std",
        "T_input_to_convention",
    }
)
_MISSING = object()


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
    compare_mode: str,
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
        return _identify_roundtrip_from_poscar_data(
            source_name=source_name,
            lattice_factors=lattice_factors,
            positions=positions,
            elements=elements,
            occupancies=occupancies,
            moments=moments,
            compare_mode=compare_mode,
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
        )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(parse_target)
    return _identify_roundtrip_from_poscar_data(
        source_name=source_name,
        lattice_factors=lattice_factors,
        positions=positions,
        elements=elements,
        occupancies=occupancies,
        moments=moments,
        compare_mode=compare_mode,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )


def _identify_roundtrip_from_poscar_data(
    *,
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    compare_mode: str,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
):
    if compare_mode not in COMPARE_MODE_CHOICES:
        raise ValueError(f"Unsupported compare_mode: {compare_mode}")
    return find_spin_group_from_data(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        input_spin_setting="cartesian",
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )


def _normalize_poscar_text(value: str) -> str:
    lines = value.splitlines()
    if not lines:
        return value
    return "\n".join(lines[1:])


def _normalize_compare_value(value, *, field_name: str | None = None, top_level: bool = False):
    value = _normalize_jsonable(value)
    if top_level and field_name in FULL_COMPARE_SOURCE_DEPENDENT_FIELDS:
        return _MISSING
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            child = _normalize_compare_value(item, field_name=str(key), top_level=False)
            if child is not _MISSING:
                normalized[str(key)] = child
        return normalized
    if isinstance(value, list):
        return [
            _normalize_compare_value(item, field_name=field_name, top_level=False)
            for item in value
        ]
    if isinstance(value, str) and field_name is not None and field_name.endswith("_poscar"):
        return _normalize_poscar_text(value)
    return value


def _basic_compare_payload(result, *, compare_conf: bool) -> dict:
    payload = {
        key: _normalize_compare_value(_result_value(result, attr, key=key), field_name=key)
        for key, attr in BASIC_COMPARE_ATTRS
    }
    if not compare_conf:
        payload.pop("conf", None)
    return payload


def _result_value(result, attr: str, *, key: str | None = None):
    if isinstance(result, dict):
        return result.get(key or attr)
    return getattr(result, attr, None)


def _full_compare_payload(result) -> dict:
    raw_payload = dict(result.to_dict())
    normalized = {}
    for key, value in raw_payload.items():
        child = _normalize_compare_value(value, field_name=str(key), top_level=True)
        if child is not _MISSING:
            normalized[str(key)] = child
    return normalized


def _compare_preview(value):
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, dict):
        return {"type": "dict", "size": len(value)}
    if isinstance(value, list):
        return {"type": "list", "size": len(value)}
    return value


def _append_difference(
    differences: list[dict],
    *,
    field: str,
    expected,
    actual,
    max_differences: int,
) -> int:
    if len(differences) < max_differences:
        differences.append(
            {
                "field": field,
                "expected": _compare_preview(expected),
                "actual": _compare_preview(actual),
            }
        )
        return 0
    return 1


def _numbers_match(expected, actual, *, float_atol: float, float_rtol: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
        return False
    if isinstance(expected, float) or isinstance(actual, float):
        expected_float = float(expected)
        actual_float = float(actual)
        if math.isnan(expected_float) or math.isnan(actual_float):
            return math.isnan(expected_float) and math.isnan(actual_float)
        return math.isclose(expected_float, actual_float, rel_tol=float_rtol, abs_tol=float_atol)
    return expected == actual


def _compare_values(
    expected,
    actual,
    *,
    field: str,
    differences: list[dict],
    float_atol: float,
    float_rtol: float,
    max_differences: int,
) -> int:
    if expected is _MISSING or actual is _MISSING:
        if expected is actual:
            return 0
        return _append_difference(
            differences,
            field=field,
            expected=expected,
            actual=actual,
            max_differences=max_differences,
        )
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if _numbers_match(expected, actual, float_atol=float_atol, float_rtol=float_rtol):
            return 0
        return _append_difference(
            differences,
            field=field,
            expected=expected,
            actual=actual,
            max_differences=max_differences,
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        truncated = 0
        for key in sorted(set(expected) | set(actual)):
            child_field = f"{field}.{key}" if field else str(key)
            truncated += _compare_values(
                expected.get(key, _MISSING),
                actual.get(key, _MISSING),
                field=child_field,
                differences=differences,
                float_atol=float_atol,
                float_rtol=float_rtol,
                max_differences=max_differences,
            )
        return truncated
    if isinstance(expected, list) and isinstance(actual, list):
        truncated = 0
        if len(expected) != len(actual):
            truncated += _append_difference(
                differences,
                field=f"{field}.length" if field else "length",
                expected=len(expected),
                actual=len(actual),
                max_differences=max_differences,
            )
        for index in range(min(len(expected), len(actual))):
            truncated += _compare_values(
                expected[index],
                actual[index],
                field=f"{field}[{index}]",
                differences=differences,
                float_atol=float_atol,
                float_rtol=float_rtol,
                max_differences=max_differences,
            )
        return truncated
    if expected == actual:
        return 0
    return _append_difference(
        differences,
        field=field,
        expected=expected,
        actual=actual,
        max_differences=max_differences,
    )


def _compare_payloads(
    expected: dict,
    actual: dict,
    *,
    float_atol: float,
    float_rtol: float,
    max_differences: int,
) -> tuple[list[dict], int]:
    differences: list[dict] = []
    truncated = _compare_values(
        expected,
        actual,
        field="",
        differences=differences,
        float_atol=float_atol,
        float_rtol=float_rtol,
        max_differences=max_differences,
    )
    return differences, truncated


def _comparison_payload(
    original,
    roundtrip,
    *,
    compare_mode: str,
    compare_conf: bool,
) -> tuple[dict, dict]:
    if compare_mode == COMPARE_MODE_BASIC:
        return (
            _basic_compare_payload(original, compare_conf=compare_conf),
            _basic_compare_payload(roundtrip, compare_conf=compare_conf),
        )
    if compare_mode == COMPARE_MODE_FULL:
        return _full_compare_payload(original), _full_compare_payload(roundtrip)
    raise ValueError(f"Unsupported compare_mode: {compare_mode}")


def _result_payload(
    original,
    roundtrip,
    *,
    compare_mode: str,
    compare_conf: bool,
    compare_float_atol: float,
    compare_float_rtol: float,
    max_differences_per_case: int,
) -> tuple[dict, list[dict]]:
    expected, actual = _comparison_payload(
        original,
        roundtrip,
        compare_mode=compare_mode,
        compare_conf=compare_conf,
    )
    differences, truncated_count = _compare_payloads(
        expected,
        actual,
        float_atol=compare_float_atol,
        float_rtol=compare_float_rtol,
        max_differences=max_differences_per_case,
    )
    payload = {
        "compare_mode": compare_mode,
        "original_index": _result_value(original, "index"),
        "roundtrip_index": _result_value(roundtrip, "index"),
        "index_match": _result_value(roundtrip, "index") == _result_value(original, "index"),
        "original_conf": _result_value(original, "conf"),
        "roundtrip_conf": _result_value(roundtrip, "conf"),
        "conf_match": _result_value(roundtrip, "conf") == _result_value(original, "conf"),
        "compared_field_count": len(expected),
        "difference_count": len(differences) + truncated_count,
        "differences_truncated_count": truncated_count,
        "differences": differences,
    }
    return payload, differences


def run_poscar_roundtrip_batch(
    files: list[Path],
    output_dir: Path | str,
    *,
    source_mode: str = POSCAR_SOURCE_ACC_PRIMITIVE,
    compare_mode: str = COMPARE_MODE_BASIC,
    compare_conf: bool = True,
    compare_float_atol: float = 1e-6,
    compare_float_rtol: float = 1e-8,
    max_differences_per_case: int = 200,
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
        case_start = time.perf_counter()
        original_duration = None
        roundtrip_duration = None
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
            if occupancy_annotation["source_has_fractional_occupancy"]:
                fractional_occupancy_case_count += 1
            source_name, poscar_text, record_input_format = _source_poscar_payload(
                original,
                case_id=case_id,
                source_mode=source_mode,
            )
            if input_format is None:
                input_format = record_input_format

            roundtrip_start = time.perf_counter()
            roundtrip = _roundtrip_from_poscar_text(
                source_name=source_name,
                poscar_text=poscar_text,
                compare_mode=compare_mode,
                space_tol=space_tol,
                mtol=mtol,
                meigtol=meigtol,
                matrix_tol=matrix_tol,
                output_dir=output_root,
                save_poscar=save_poscar,
            )
            roundtrip_duration = round(time.perf_counter() - roundtrip_start, 6)
            payload, differences = _result_payload(
                original,
                roundtrip,
                compare_mode=compare_mode,
                compare_conf=compare_conf,
                compare_float_atol=compare_float_atol,
                compare_float_rtol=compare_float_rtol,
                max_differences_per_case=max_differences_per_case,
            )

            record = {
                "case_id": case_id,
                "file_name": file_name,
                "status": "ok" if not differences else "mismatch",
                "duration_seconds": round(time.perf_counter() - case_start, 6),
                "original_duration_seconds": original_duration,
                "roundtrip_duration_seconds": roundtrip_duration,
                "original": {
                    "index": _result_value(original, "index"),
                    "conf": _result_value(original, "conf"),
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
                    "duration_seconds": round(time.perf_counter() - case_start, 6),
                    "original_duration_seconds": original_duration,
                    "roundtrip_duration_seconds": roundtrip_duration,
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
        "compare_mode": compare_mode,
        "compare_conf": compare_conf,
        "compare_float_atol": compare_float_atol,
        "compare_float_rtol": compare_float_rtol,
        "max_differences_per_case": max_differences_per_case,
        "full_compare_excluded_fields": sorted(FULL_COMPARE_SOURCE_DEPENDENT_FIELDS)
        if compare_mode == COMPARE_MODE_FULL
        else [],
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
        "--compare-mode",
        choices=COMPARE_MODE_CHOICES,
        default=COMPARE_MODE_BASIC,
        help=(
            "Comparison scope. basic compares final group identifiers and physical summary "
            "fields. full compares normalized source-independent MagSymmetryResult fields."
        ),
    )
    parser.add_argument(
        "--compare-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numeric comparison in roundtrip payloads.",
    )
    parser.add_argument(
        "--compare-rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for numeric comparison in roundtrip payloads.",
    )
    parser.add_argument(
        "--max-differences-per-case",
        type=int,
        default=200,
        help="Maximum number of field differences stored for one mismatching case.",
    )
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
        compare_mode=args.compare_mode,
        compare_conf=not args.no_compare_conf,
        compare_float_atol=args.compare_atol,
        compare_float_rtol=args.compare_rtol,
        max_differences_per_case=args.max_differences_per_case,
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
