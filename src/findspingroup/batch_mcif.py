from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from .find_spin_group import NumpyEncoder, audit_spatial_transform_effect, find_spin_group
from .structure.group import SpinSpaceGroup
from .version import __version__


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n")


def _source_fractional_occupancy_annotation(result) -> dict:
    detail = getattr(result, "acc_primitive_magnetic_cell_detail", None)
    if detail is None:
        detail = getattr(result, "primitive_magnetic_cell_detail", None)
    occupancies = detail.get("occupancies") if isinstance(detail, dict) else None
    if occupancies is None:
        return {
            "source_has_fractional_occupancy": None,
            "source_occupancy_values": None,
            "source_fractional_occupancy_values": None,
            "source_fractional_occupancy_site_count": None,
        }

    occupancy_values = sorted({float(value) for value in occupancies})
    fractional_values = sorted(
        {
            float(value)
            for value in occupancies
            if not math.isclose(float(value), 1.0, abs_tol=1e-9)
        }
    )
    fractional_site_count = sum(
        1 for value in occupancies if not math.isclose(float(value), 1.0, abs_tol=1e-9)
    )
    return {
        "source_has_fractional_occupancy": bool(fractional_values),
        "source_occupancy_values": occupancy_values,
        "source_fractional_occupancy_values": fractional_values,
        "source_fractional_occupancy_site_count": fractional_site_count,
    }


def _stringify_export_value(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder)
    except TypeError:
        return str(value)


def _normalize_case_id(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def _tagged_artifact_name(file_name: str, run_tag: str) -> str:
    path = Path(file_name)
    return f"{path.stem}__{run_tag}{path.suffix}"


def _error_json_path(root: Path, file_name: str, run_tag: str) -> Path:
    tagged_name = _tagged_artifact_name(file_name, run_tag)
    return root / "error_json" / f"{tagged_name}.json"


def _error_set_path(root: Path, file_name: str, run_tag: str) -> Path:
    return root / "error_set" / _tagged_artifact_name(file_name, run_tag)


def _read_manifest(manifest_path: Path) -> list[Path]:
    base_dir = manifest_path.resolve().parent
    files: list[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        if _is_supported_mcif_path(candidate):
            files.append(candidate)
    return files


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
    return files


def _dedupe_sorted(paths: list[Path]) -> list[Path]:
    unique = {path.resolve().as_posix(): path.resolve() for path in paths}
    return [unique[key] for key in sorted(unique)]


def _stable_record(case_id: str, file_name: str, runtime_record: dict) -> dict:
    record = {
        "case_id": case_id,
        "file_name": file_name,
        "status": runtime_record["status"],
    }
    if runtime_record["status"] == "ok":
        record["result"] = runtime_record["result"]
        if "tensor_summary" in runtime_record:
            record["tensor_summary"] = runtime_record["tensor_summary"]
    else:
        record["error"] = {
            "type": runtime_record["error"]["type"],
            "message": runtime_record["error"]["message"],
        }
    return record


def _summarize_tensor_payload(payload) -> dict:
    if not isinstance(payload, dict):
        return {"error": str(payload)}
    if "error" in payload:
        return {"error": str(payload["error"])}
    summary = {}
    if "free_parameters" in payload:
        summary["free_parameters"] = payload["free_parameters"]
    if "is_zero" in payload:
        summary["is_zero"] = payload["is_zero"]
    if "relations" in payload:
        summary["relations_count"] = len(payload.get("relations", []))
    return summary


def _build_tensor_summary(result) -> dict:
    tensor_outputs = getattr(result, "tensor_outputs", None) or {}
    ordered_keys = [
        "AHE_woSOC",
        "AHE_wSOC",
        "BCDTensor",
        "MSGBCDTensor",
        "QMDTensor",
        "MSGQMDTensor",
        "IMDTensor",
        "MSGIMDTensor",
    ]
    summary = {}
    for key in ordered_keys:
        if key not in tensor_outputs:
            continue
        summary[key] = _summarize_tensor_payload(tensor_outputs[key])
    return summary


def _build_g0_self_audit(result, *, tol: float = 1e-6, det_tol: float = 1e-2) -> dict | None:
    ops = getattr(result, "g0_standard_ssg_ops", None)
    raw_transform = getattr(result, "raw_T_input_to_G0std", None)
    if ops is None or raw_transform is None:
        return None
    try:
        matrix = np.asarray(raw_transform[0], dtype=float)
        shift = np.asarray(raw_transform[1], dtype=float)
        g0_ssg = SpinSpaceGroup(ops)
        audit = audit_spatial_transform_effect(
            g0_ssg,
            matrix,
            shift,
            tol=tol,
            det_tol=det_tol,
            use_nssg=False,
        )
        return {
            "transform_matrix": np.asarray(matrix, dtype=float).tolist(),
            "origin_shift": np.asarray(shift, dtype=float).tolist(),
            "determinant": audit.get("determinant"),
            "volume_preserving": audit.get("volume_preserving"),
            "real_ops_exact_same": audit.get("real_ops_exact_same"),
            "real_ops_same_mod_integer": audit.get("real_ops_same_mod_integer"),
            "real_ops_same_mod_pure_translations": audit.get("real_ops_same_mod_pure_translations"),
            "paired_spin_changed_count": audit.get("paired_spin_changed_count"),
            "source_real_op_count": audit.get("source_real_op_count"),
            "transformed_real_op_count": audit.get("transformed_real_op_count"),
            "unmatched_source_indices": audit.get("unmatched_source_indices"),
        }
    except Exception as exc:
        return {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        }


def _flatten_record(value, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_record(value[key], next_prefix))
        return flat
    if isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            flat.update(_flatten_record(item, next_prefix))
        return flat
    flat[prefix] = value
    return flat


def _record_differences(current: dict, expected: dict) -> list[dict]:
    current_flat = _flatten_record(current)
    expected_flat = _flatten_record(expected)
    differences = []
    for field in sorted(set(current_flat) | set(expected_flat)):
        current_value = current_flat.get(field)
        expected_value = expected_flat.get(field)
        if current_value == expected_value:
            continue
        differences.append(
            {
                "field": field,
                "expected": expected_value,
                "actual": current_value,
            }
        )
    return differences


def _records_match(current: dict, expected: dict) -> tuple[bool, list[dict], bool]:
    if current == expected:
        return True, [], False

    if current.get("status") != expected.get("status"):
        return False, _record_differences(current, expected), False

    if current.get("status") == "ok":
        current_without_tensor = dict(current)
        expected_without_tensor = dict(expected)
        current_without_tensor.pop("tensor_summary", None)
        expected_has_tensor = "tensor_summary" in expected_without_tensor
        expected_without_tensor.pop("tensor_summary", None)
        if current_without_tensor == expected_without_tensor and not expected_has_tensor:
            return True, [], True

    return False, _record_differences(current, expected), False


def _resolve_selector(root, selector: str):
    current = root
    for part in selector.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
            continue
        current = getattr(current, part)
    return current


def _normalize_jsonable(value):
    if isinstance(value, dict):
        return {key: _normalize_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _normalize_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _normalize_jsonable(value.to_dict())
        except Exception:
            pass
    return value


def _build_export_root(result) -> dict:
    payload = _normalize_jsonable(dict(result.to_dict()))
    payload["phase"] = result.magnetic_phase
    payload["properties"] = result.properties_summary()
    payload["summary"] = result.to_summary_dict()
    return payload


def _build_group_identifier_payload(result) -> dict:
    return {
        "index": getattr(result, "index", None),
        "conf": getattr(result, "conf", None),
        "sg_num": getattr(result, "input_space_group_number", None),
        "sg_symbol": getattr(result, "input_space_group_symbol", None),
        "sg_has_real_space_inversion": getattr(result, "sg_has_real_space_inversion", None),
        "sg_is_polar": getattr(result, "sg_is_polar", None),
        "sg_is_chiral": getattr(result, "sg_is_chiral", None),
        "ossg_space_group_number": getattr(result, "ossg_space_group_number", None),
        "ossg_has_real_space_inversion": getattr(result, "ossg_has_real_space_inversion", None),
        "ossg_is_polar": getattr(result, "ossg_is_polar", None),
        "ossg_is_chiral": getattr(result, "ossg_is_chiral", None),
        "msg_num": getattr(result, "msg_num", None),
        "msg_type": getattr(result, "msg_type", None),
        "msg_symbol": getattr(result, "msg_symbol", None),
        "msg_bns_number": getattr(result, "msg_bns_number", None),
        "msg_og_number": getattr(result, "msg_og_number", None),
        "msg_parent_space_group_number": getattr(result, "msg_parent_space_group_number", None),
        "msg_has_real_space_inversion": getattr(result, "msg_has_real_space_inversion", None),
        "msg_is_polar": getattr(result, "msg_is_polar", None),
        "msg_is_chiral": getattr(result, "msg_is_chiral", None),
    }


def _build_export_content(result, selectors: list[str]) -> str:
    root = _build_export_root(result)
    if len(selectors) == 1:
        return _stringify_export_value(_resolve_selector(root, selectors[0]))
    content = {
        selector: _resolve_selector(root, selector)
        for selector in selectors
    }
    return _stringify_export_value(content)


def _append_export_line(path: Path, file_name: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{file_name}: {content}\n")


def _baseline_meta_payload(
    *,
    files: list[Path],
    suite_name: str | None,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
    processed_cases: int,
    success_count: int,
    error_count: int,
    baseline_cases: dict[str, dict] | None = None,
    created_at: str | None = None,
    created_at_epoch: float | None = None,
    updated_at: str | None = None,
    updated_at_epoch: float | None = None,
) -> dict:
    stored_cases = baseline_cases or {}
    created_at_value = created_at or _isoformat_now()
    created_at_epoch_value = created_at_epoch if created_at_epoch is not None else time.time()
    return {
        "schema_version": 1,
        "created_at": created_at_value,
        "created_at_epoch": created_at_epoch_value,
        "updated_at": updated_at,
        "updated_at_epoch": updated_at_epoch,
        "package_version": __version__,
        "suite_name": suite_name,
        "tolerances": {
            "space_tol": space_tol,
            "mtol": mtol,
            "meigtol": meigtol,
            "matrix_tol": matrix_tol,
        },
        "total_cases_requested": len(files),
        "processed_cases": processed_cases,
        "success_count": success_count,
        "error_count": error_count,
        "resolved_inputs": [_normalize_case_id(path) for path in files],
        "baseline_case_count": len(stored_cases),
        "baseline_success_count": sum(
            1 for record in stored_cases.values() if record.get("status") == "ok"
        ),
        "baseline_error_count": sum(
            1 for record in stored_cases.values() if record.get("status") == "error"
        ),
    }


def _tolerances_payload(
    *,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
) -> dict:
    return {
        "space_tol": space_tol,
        "mtol": mtol,
        "meigtol": meigtol,
        "matrix_tol": matrix_tol,
    }


def _format_float_key(value: float) -> str:
    return format(value, ".12g")


def _isoformat_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _run_tag_from_isoformat(timestamp: str) -> str:
    return f"run_v{__version__}_{dt.datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')}"


def _slugify_suite_name(raw: str) -> str:
    slug = []
    for char in raw.strip():
        if char.isalnum() or char in {".", "_", "-"}:
            slug.append(char)
        else:
            slug.append("_")
    collapsed = "".join(slug).strip("._-")
    if not collapsed:
        raise ValueError("Baseline suite name becomes empty after sanitization.")
    return collapsed


def _tolerance_key(
    *,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
) -> str:
    return "__".join(
        [
            f"space_{_format_float_key(space_tol)}",
            f"mtol_{_format_float_key(mtol)}",
            f"meigtol_{_format_float_key(meigtol)}",
            f"matrix_{_format_float_key(matrix_tol)}",
        ]
    )


def _resolve_auto_baseline_paths(
    *,
    baseline_root: Path,
    suite_name: str,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
) -> dict:
    suite_slug = _slugify_suite_name(suite_name)
    tolerance_dir = _tolerance_key(
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )
    baseline_dir = baseline_root / suite_slug / tolerance_dir
    return {
        "suite_name": suite_slug,
        "baseline_dir": baseline_dir,
        "baseline_json": baseline_dir / "baseline.json",
        "baseline_meta": baseline_dir / "baseline.meta.json",
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_baseline_meta_timestamps(existing_meta: dict | None, *, is_update: bool) -> dict:
    current_at = _isoformat_now()
    current_at_epoch = time.time()
    if not existing_meta:
        return {
            "created_at": current_at,
            "created_at_epoch": current_at_epoch,
            "updated_at": None,
            "updated_at_epoch": None,
        }

    created_at = existing_meta.get("created_at") or current_at
    created_at_epoch = existing_meta.get("created_at_epoch")
    if created_at_epoch is None:
        created_at_epoch = current_at_epoch

    if not is_update:
        return {
            "created_at": created_at,
            "created_at_epoch": created_at_epoch,
            "updated_at": existing_meta.get("updated_at"),
            "updated_at_epoch": existing_meta.get("updated_at_epoch"),
        }

    return {
        "created_at": created_at,
        "created_at_epoch": created_at_epoch,
        "updated_at": current_at,
        "updated_at_epoch": current_at_epoch,
    }


def _validate_auto_baseline_meta(
    *,
    meta_path: Path,
    suite_name: str,
    space_tol: float,
    mtol: float,
    meigtol: float,
    matrix_tol: float,
) -> None:
    meta = _load_json(meta_path)
    expected_tolerances = _tolerances_payload(
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )
    if meta.get("suite_name") != suite_name:
        raise ValueError(
            f"Auto baseline suite mismatch: expected {suite_name}, got {meta.get('suite_name')!r}"
        )
    if meta.get("tolerances") != expected_tolerances:
        raise ValueError(
            f"Auto baseline tolerances mismatch in {meta_path}: "
            f"expected {expected_tolerances}, got {meta.get('tolerances')}"
        )


def _compare_cases(current_cases: dict[str, dict], baseline_cases: dict[str, dict]) -> dict:
    missing_in_baseline = []
    mismatches = []
    tensor_summary_backfills = []

    for case_id, current in current_cases.items():
        expected = baseline_cases.get(case_id)
        if expected is None:
            missing_in_baseline.append(case_id)
            continue
        matches, differences, missing_tensor_summary = _records_match(current, expected)
        if missing_tensor_summary:
            tensor_summary_backfills.append(case_id)
        if not matches:
            mismatches.append(
                {
                    "case_id": case_id,
                    "expected": expected,
                    "actual": current,
                    "differences": differences,
                }
            )

    return {
        "checked_case_count": len(current_cases),
        "missing_in_baseline": missing_in_baseline,
        "mismatches": mismatches,
        "tensor_summary_backfills": tensor_summary_backfills,
        "missing_in_baseline_count": len(missing_in_baseline),
        "mismatch_count": len(mismatches),
        "tensor_summary_backfill_count": len(tensor_summary_backfills),
    }


def _compare_auto_baseline_cases(current_cases: dict[str, dict], baseline_cases: dict[str, dict]) -> dict:
    protected_ok_mismatches = []
    error_to_ok_updates = []
    error_still_error = []
    ignored_error_mismatches = []
    new_cases = []
    baseline_only_cases = []
    tensor_summary_backfills = []

    for case_id, expected in baseline_cases.items():
        current = current_cases.get(case_id)
        if current is None:
            baseline_only_cases.append(case_id)
            continue
        if expected["status"] == "ok":
            matches, differences, missing_tensor_summary = _records_match(current, expected)
            if missing_tensor_summary:
                tensor_summary_backfills.append(case_id)
            if not matches:
                protected_ok_mismatches.append(
                    {
                        "case_id": case_id,
                        "expected": expected,
                        "actual": current,
                        "differences": differences,
                    }
                )
            continue
        if current["status"] == "ok":
            error_to_ok_updates.append(case_id)
            continue
        error_still_error.append(case_id)
        matches, differences, _ = _records_match(current, expected)
        if not matches:
            ignored_error_mismatches.append(
                {
                    "case_id": case_id,
                    "expected": expected,
                    "actual": current,
                    "differences": differences,
                }
            )

    for case_id in current_cases:
        if case_id not in baseline_cases:
            new_cases.append(case_id)

    protected_ok_mismatches.sort(key=lambda item: item["case_id"])
    ignored_error_mismatches.sort(key=lambda item: item["case_id"])
    error_to_ok_updates.sort()
    error_still_error.sort()
    new_cases.sort()
    baseline_only_cases.sort()

    return {
        "checked_case_count": len(current_cases),
        "baseline_case_count": len(baseline_cases),
        "protected_ok_mismatches": protected_ok_mismatches,
        "protected_ok_mismatch_count": len(protected_ok_mismatches),
        "error_to_ok_updates": error_to_ok_updates,
        "error_to_ok_update_count": len(error_to_ok_updates),
        "error_still_error": error_still_error,
        "error_still_error_count": len(error_still_error),
        "ignored_error_mismatches": ignored_error_mismatches,
        "ignored_error_mismatch_count": len(ignored_error_mismatches),
        "new_cases": new_cases,
        "new_case_count": len(new_cases),
        "baseline_only_cases": baseline_only_cases,
        "baseline_only_case_count": len(baseline_only_cases),
        "tensor_summary_backfills": tensor_summary_backfills,
        "tensor_summary_backfill_count": len(tensor_summary_backfills),
        "missing_in_baseline": [],
        "missing_in_baseline_count": 0,
        "mismatches": protected_ok_mismatches,
        "mismatch_count": len(protected_ok_mismatches),
    }


def _merge_auto_baseline_cases(
    baseline_cases: dict[str, dict],
    current_cases: dict[str, dict],
) -> dict[str, dict]:
    merged_cases = dict(baseline_cases)
    for case_id, current in current_cases.items():
        expected = baseline_cases.get(case_id)
        if expected is None:
            merged_cases[case_id] = current
            continue
        if (
            expected["status"] == "ok"
            and current["status"] == "ok"
            and "tensor_summary" not in expected
            and "tensor_summary" in current
        ):
            merged_cases[case_id] = current
            continue
        if expected["status"] == "error" and current["status"] == "ok":
            merged_cases[case_id] = current
    return merged_cases


def run_mcif_batch(
    files: list[Path],
    output_dir: Path,
    *,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    baseline_path: Path | None = None,
    stop_on_error: bool = False,
    fail_on_error: bool = False,
    fail_on_mismatch: bool = False,
    export_fields: list[str] | None = None,
    export_txt_path: Path | None = None,
    quiet: bool = False,
    include_g0_self_audit: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    full_results_path = output_dir / "full_results.jsonl"
    records_path.write_text("", encoding="utf-8")
    full_results_path.write_text("", encoding="utf-8")
    if export_txt_path is not None:
        export_txt_path.parent.mkdir(parents=True, exist_ok=True)
        export_txt_path.write_text("", encoding="utf-8")
    stable_cases: dict[str, dict] = {}
    errors_by_file: dict[str, dict] = {}

    success_count = 0
    error_count = 0
    stop_reason = None
    started_at = _isoformat_now()
    run_tag = _run_tag_from_isoformat(started_at)
    total_start = time.perf_counter()

    for index, file_path in enumerate(files, start=1):
        case_id = _normalize_case_id(file_path)
        case_start = time.perf_counter()
        try:
            result = find_spin_group(
                str(file_path),
                space_tol=space_tol,
                mtol=mtol,
                meigtol=meigtol,
                matrix_tol=matrix_tol,
            )
            snapshot = result.to_summary_dict()
            duration = round(time.perf_counter() - case_start, 6)
            runtime_record = {
                "case_id": case_id,
                "file_name": file_path.name,
                "source_path": file_path.resolve().as_posix(),
                "status": "ok",
                "duration_seconds": duration,
                "result": snapshot,
                "group_identifiers": _build_group_identifier_payload(result),
                "tensor_summary": _build_tensor_summary(result),
            }
            full_runtime_record = {
                "case_id": case_id,
                "file_name": file_path.name,
                "source_path": file_path.resolve().as_posix(),
                "status": "ok",
                "duration_seconds": duration,
                "result": _build_export_root(result),
                "tensor_summary": _build_tensor_summary(result),
            }
            if include_g0_self_audit:
                g0_self_audit = _build_g0_self_audit(result)
                runtime_record["g0_self_audit"] = g0_self_audit
                full_runtime_record["g0_self_audit"] = g0_self_audit
            if export_fields and export_txt_path is not None:
                export_content = _build_export_content(result, export_fields)
                _append_export_line(export_txt_path, file_path.name, export_content)
            success_count += 1
            if not quiet:
                print(
                    f"[{index}/{len(files)}] OK    {case_id} -> "
                    f"{snapshot['index']} ({duration:.3f}s)"
                )
        except Exception as exc:
            duration = round(time.perf_counter() - case_start, 6)
            runtime_record = {
                "case_id": case_id,
                "file_name": file_path.name,
                "source_path": file_path.resolve().as_posix(),
                "status": "error",
                "duration_seconds": duration,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                },
            }
            full_runtime_record = dict(runtime_record)
            error_count += 1
            errors_by_file[case_id] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if export_fields and export_txt_path is not None:
                _append_export_line(
                    export_txt_path,
                    file_path.name,
                    f"ERROR[{type(exc).__name__}] {exc}",
                )
            if not quiet:
                print(
                    f"[{index}/{len(files)}] ERROR {case_id} -> "
                    f"{type(exc).__name__}: {exc}"
                )

        stable_record = _stable_record(case_id, file_path.name, runtime_record)
        stable_cases[case_id] = stable_record
        _append_jsonl(records_path, runtime_record)
        _append_jsonl(full_results_path, full_runtime_record)
        if runtime_record["status"] == "error":
            _write_json(_error_json_path(output_dir, file_path.name, run_tag), runtime_record)
            error_set_path = _error_set_path(output_dir, file_path.name, run_tag)
            error_set_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, error_set_path)

        if runtime_record["status"] == "error" and stop_on_error:
            stop_reason = f"Stopped after first error at {case_id}"
            break

    baseline_compare = None
    if baseline_path is not None:
        baseline_cases = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_compare = _compare_cases(stable_cases, baseline_cases)
        _write_json(output_dir / "comparison.json", baseline_compare)

    _write_json(output_dir / "baseline.json", stable_cases)
    _write_json(output_dir / "errors_by_file.json", errors_by_file)

    summary = {
        "output_dir": output_dir.resolve().as_posix(),
        "full_results_jsonl": full_results_path.resolve().as_posix(),
        "package_version": __version__,
        "run_tag": run_tag,
        "started_at": started_at,
        "finished_at": _isoformat_now(),
        "tolerances": _tolerances_payload(
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
        ),
        "total_cases_requested": len(files),
        "processed_cases": len(stable_cases),
        "success_count": success_count,
        "error_count": error_count,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "duration_seconds": round(time.perf_counter() - total_start, 6),
        "baseline_path": baseline_path.resolve().as_posix() if baseline_path else None,
        "comparison": baseline_compare,
    }
    _write_json(output_dir / "summary.json", summary)

    exit_code = 0
    if fail_on_error and error_count:
        exit_code = 1
    if fail_on_mismatch and baseline_compare:
        if baseline_compare["missing_in_baseline_count"] or baseline_compare["mismatch_count"]:
            exit_code = 1
    summary["exit_code"] = exit_code
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_mcif_batch_with_auto_baseline(
    files: list[Path],
    output_dir: Path,
    *,
    baseline_root: Path,
    suite_name: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    stop_on_error: bool = False,
    export_fields: list[str] | None = None,
    export_txt_path: Path | None = None,
    quiet: bool = False,
    include_g0_self_audit: bool = False,
) -> dict:
    auto_paths = _resolve_auto_baseline_paths(
        baseline_root=baseline_root,
        suite_name=suite_name,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )
    baseline_exists = auto_paths["baseline_json"].exists()

    if baseline_exists:
        if not auto_paths["baseline_meta"].exists():
            raise FileNotFoundError(
                f"Auto baseline metadata is missing: {auto_paths['baseline_meta']}"
            )
        _validate_auto_baseline_meta(
            meta_path=auto_paths["baseline_meta"],
            suite_name=auto_paths["suite_name"],
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
        )

    summary = run_mcif_batch(
        files,
        output_dir,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
        baseline_path=None,
        stop_on_error=stop_on_error,
        fail_on_error=False,
        fail_on_mismatch=False,
        export_fields=export_fields,
        export_txt_path=export_txt_path,
        quiet=quiet,
        include_g0_self_audit=include_g0_self_audit,
    )

    run_baseline = _load_json(output_dir / "baseline.json")
    baseline_compare = None
    action = "created"
    stored_baseline = run_baseline
    existing_meta = None

    if baseline_exists:
        baseline_cases = _load_json(auto_paths["baseline_json"])
        existing_meta = _load_json(auto_paths["baseline_meta"])
        baseline_compare = _compare_auto_baseline_cases(run_baseline, baseline_cases)
        _write_json(output_dir / "comparison.json", baseline_compare)
        if baseline_compare["protected_ok_mismatch_count"]:
            action = "blocked_by_ok_mismatch"
            stored_baseline = baseline_cases
        else:
            stored_baseline = _merge_auto_baseline_cases(baseline_cases, run_baseline)
            if (
                baseline_compare["error_to_ok_update_count"]
                or baseline_compare["tensor_summary_backfill_count"]
                or baseline_compare["new_case_count"]
            ):
                action = "updated"
            else:
                action = "used_existing"

    if action in {"created", "updated"}:
        meta_timestamps = _resolve_baseline_meta_timestamps(
            existing_meta,
            is_update=(action == "updated"),
        )
        run_meta = _baseline_meta_payload(
            files=files,
            suite_name=auto_paths["suite_name"],
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
            processed_cases=summary["processed_cases"],
            success_count=summary["success_count"],
            error_count=summary["error_count"],
            baseline_cases=stored_baseline,
            created_at=meta_timestamps["created_at"],
            created_at_epoch=meta_timestamps["created_at_epoch"],
            updated_at=meta_timestamps["updated_at"],
            updated_at_epoch=meta_timestamps["updated_at_epoch"],
        )
        auto_paths["baseline_dir"].mkdir(parents=True, exist_ok=True)
        _write_json(auto_paths["baseline_json"], stored_baseline)
        _write_json(auto_paths["baseline_meta"], run_meta)

    summary["auto_baseline"] = {
        "action": action,
        "suite_name": auto_paths["suite_name"],
        "baseline_root": baseline_root.resolve().as_posix(),
        "baseline_dir": auto_paths["baseline_dir"].resolve().as_posix(),
        "baseline_json": auto_paths["baseline_json"].resolve().as_posix(),
        "baseline_meta": auto_paths["baseline_meta"].resolve().as_posix(),
    }
    summary["baseline_path"] = auto_paths["baseline_json"].resolve().as_posix()
    summary["comparison"] = baseline_compare
    summary["exit_code"] = 1 if action == "blocked_by_ok_mismatch" else 0

    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run find_spin_group on .mcif files and record stable summaries, "
            "per-case artifacts, and optional baseline comparisons."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input .mcif files or directories. Directories are scanned for .mcif files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Text file with one .mcif path per line. Relative paths are resolved from the manifest directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for batch artifacts such as baseline.json and records.jsonl.",
    )
    parser.add_argument("--space-tol", type=float, default=0.02)
    parser.add_argument("--mtol", type=float, default=0.02)
    parser.add_argument("--meigtol", type=float, default=0.00002)
    parser.add_argument("--matrix-tol", type=float, default=0.01)
    parser.add_argument("--limit", type=int, help="Only process the first N resolved .mcif files.")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index. Use with --shard-count for cluster array jobs.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count. Use 1 for a normal local run.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top level when an input path is a directory.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare the current run against an existing baseline.json file.",
    )
    parser.add_argument(
        "--auto-baseline",
        action="store_true",
        help=(
            "Automatically locate the baseline by dataset name and tolerance parameters. "
            "If no matching baseline exists, create one from the current run. Existing "
            "baseline entries with status=ok are protected from mismatches; error->ok "
            "repairs and new files are merged back into the stored baseline."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Root directory that stores auto-managed baselines.",
    )
    parser.add_argument(
        "--baseline-suite",
        help="Dataset name used under the auto baseline root. Defaults to the input directory name when unambiguous.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first error after writing the current case artifact.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any case errors.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if baseline comparison finds mismatches.",
    )
    parser.add_argument(
        "--export-field",
        action="append",
        default=[],
        help=(
            "Field to export from MagSymmetryResult, supports dot paths such as "
            "index, phase, properties.ss_w_soc, identify_index_details.G0_id."
        ),
    )
    parser.add_argument(
        "--export-txt",
        type=Path,
        help="Write per-file export lines as 'file_name: content'. Relative paths are resolved under output-dir.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress logs.",
    )
    parser.add_argument(
        "--include-g0-self-audit",
        action="store_true",
        help=(
            "Also record the G0std self-transform audit "
            "(raw input->G0std transform acting on current g0_standard_ssg_ops) "
            "into runtime records. Does not affect baseline comparison."
        ),
    )
    return parser.parse_args()


def _derive_baseline_suite(args: argparse.Namespace) -> str:
    if args.baseline_suite:
        return _slugify_suite_name(args.baseline_suite)
    if args.manifest is not None and not args.inputs:
        return _slugify_suite_name(args.manifest.stem)
    if len(args.inputs) == 1:
        candidate = Path(args.inputs[0])
        if candidate.exists():
            if candidate.is_dir():
                return _slugify_suite_name(candidate.resolve().name)
            if candidate.is_file():
                return _slugify_suite_name(candidate.stem)
    raise ValueError(
        "Cannot derive an auto baseline suite name from the current inputs. "
        "Pass --baseline-suite explicitly."
    )


def _resolve_input_files(args: argparse.Namespace) -> list[Path]:
    manifest_files: list[Path] = []
    if args.manifest is not None:
        manifest_files = _read_manifest(args.manifest)
    discovered_files = _discover_mcif_files(args.inputs, recursive=not args.non_recursive)
    files = _dedupe_sorted([*manifest_files, *discovered_files])

    if not files:
        raise ValueError("No .mcif files were resolved. Provide inputs or a manifest.")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count.")

    sharded = files[args.shard_index :: args.shard_count]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        sharded = sharded[: args.limit]
    if not sharded:
        raise ValueError("Resolved shard is empty. Adjust --limit or sharding parameters.")
    return sharded


def main() -> None:
    args = _parse_args()
    if args.auto_baseline and args.baseline is not None:
        raise ValueError("--auto-baseline and --baseline cannot be used together.")
    files = _resolve_input_files(args)
    export_txt_path = None
    if args.export_txt is not None:
        export_txt_path = args.export_txt
        if not export_txt_path.is_absolute():
            export_txt_path = args.output_dir / export_txt_path
    baseline_root = None
    baseline_suite = None
    if args.auto_baseline:
        baseline_root = args.baseline_root or (Path.cwd() / "batch_baselines")
        baseline_suite = _derive_baseline_suite(args)

    config = {
        "inputs": args.inputs,
        "manifest": args.manifest.resolve().as_posix() if args.manifest else None,
        "output_dir": args.output_dir.resolve().as_posix(),
        "resolved_files": [_normalize_case_id(path) for path in files],
        "space_tol": args.space_tol,
        "mtol": args.mtol,
        "meigtol": args.meigtol,
        "matrix_tol": args.matrix_tol,
        "limit": args.limit,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "non_recursive": args.non_recursive,
        "auto_baseline": args.auto_baseline,
        "baseline": args.baseline.resolve().as_posix() if args.baseline else None,
        "baseline_root": baseline_root.resolve().as_posix() if baseline_root else None,
        "baseline_suite": baseline_suite,
        "stop_on_error": args.stop_on_error,
        "fail_on_error": args.fail_on_error,
        "fail_on_mismatch": args.fail_on_mismatch,
        "export_fields": args.export_field,
        "export_txt": export_txt_path.resolve().as_posix() if export_txt_path else None,
        "include_g0_self_audit": args.include_g0_self_audit,
    }
    _write_json(args.output_dir / "run_config.json", config)

    if args.auto_baseline:
        summary = run_mcif_batch_with_auto_baseline(
            files,
            args.output_dir,
            baseline_root=baseline_root.resolve(),
            suite_name=baseline_suite,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            stop_on_error=args.stop_on_error,
            export_fields=args.export_field,
            export_txt_path=export_txt_path,
            quiet=args.quiet,
            include_g0_self_audit=args.include_g0_self_audit,
        )
    else:
        summary = run_mcif_batch(
            files,
            args.output_dir,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            baseline_path=args.baseline.resolve() if args.baseline else None,
            stop_on_error=args.stop_on_error,
            fail_on_error=args.fail_on_error,
            fail_on_mismatch=args.fail_on_mismatch,
            export_fields=args.export_field,
            export_txt_path=export_txt_path,
            quiet=args.quiet,
            include_g0_self_audit=args.include_g0_self_audit,
        )
    raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
