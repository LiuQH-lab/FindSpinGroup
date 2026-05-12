from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from findspingroup.core.identify_symmetry_from_ops import deduplicate_matrix_pairs
from findspingroup.structure.group import combine_parametric_solutions
from findspingroup.utils.matrix_utils import getNormInf, rref_with_tolerance


AXIS_LABELS = ("a", "b", "c")


@dataclass(frozen=True)
class VacuumAxisCandidate:
    axis: str
    axis_index: int
    axis_length: float
    axis_length_ratio: float
    occupied_span_fraction: float
    vacuum_gap_fraction: float
    candidate_margin: float


def _json_float(value: float, *, zero_tol: float = 1e-12) -> float:
    numeric = float(value)
    if abs(numeric) < zero_tol:
        return 0.0
    return numeric


def _json_vector(values) -> list[float]:
    return [_json_float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _json_mod1_vector(values, *, boundary_tol: float = 1e-8) -> list[float]:
    array = np.mod(np.asarray(values, dtype=float).reshape(-1), 1.0)
    array[np.abs(array - 1.0) <= boundary_tol] = 0.0
    array[np.abs(array) <= boundary_tol] = 0.0
    return _json_vector(array)


def _json_matrix(values) -> list[list[float]]:
    array = np.asarray(values, dtype=float)
    return [[_json_float(value) for value in row] for row in array]


def _normalize_scalar(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text or None


def _normalize_dimension(value) -> str | None:
    text = _normalize_scalar(value)
    if text is None:
        return None
    normalized = text.lower().replace("_", "-").replace(" ", "")
    if normalized in {"2d", "quasi2d", "quasi-2d", "two-dimensional"}:
        return "2d"
    if normalized in {"3d", "bulk"}:
        return "3d"
    return normalized


def _normalize_calculation_mode(value) -> str | None:
    text = _normalize_scalar(value)
    if text is None:
        return "3d"
    normalized = text.lower().replace("_", "-").replace(" ", "")
    if normalized in {"auto", "default"}:
        return "auto"
    if normalized in {"2d", "quasi2d", "quasi-2d", "slab", "layer", "two-dimensional"}:
        return "quasi2d"
    if normalized in {"3d", "bulk", "three-dimensional"}:
        return "3d"
    return normalized


def _normalize_axis(value) -> tuple[str | None, int | None]:
    text = _normalize_scalar(value)
    if text is None:
        return None, None
    normalized = text.lower()
    aliases = {
        "a": 0,
        "x": 0,
        "0": 0,
        "b": 1,
        "y": 1,
        "1": 1,
        "c": 2,
        "z": 2,
        "2": 2,
    }
    axis_index = aliases.get(normalized)
    if axis_index is None:
        return None, None
    return AXIS_LABELS[axis_index], axis_index


def _cell_geometry_candidates(input_cell_detail: dict | None) -> list[VacuumAxisCandidate]:
    if not isinstance(input_cell_detail, dict):
        return []
    lattice = np.asarray(input_cell_detail.get("lattice"), dtype=float)
    positions = np.asarray(input_cell_detail.get("positions"), dtype=float)
    if lattice.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
        return []
    lengths = np.linalg.norm(lattice, axis=1)
    candidates: list[VacuumAxisCandidate] = []
    for axis_index, axis in enumerate(AXIS_LABELS):
        other_lengths = np.delete(lengths, axis_index)
        denominator = max(float(np.max(other_lengths)), 1e-12)
        length_ratio = float(lengths[axis_index] / denominator)
        coords = np.sort(np.mod(positions[:, axis_index], 1.0))
        if len(coords) <= 1:
            largest_gap = 1.0
        else:
            gaps = np.diff(coords)
            wrap_gap = coords[0] + 1.0 - coords[-1]
            largest_gap = float(max(np.max(gaps), wrap_gap))
        occupied_span = max(0.0, min(1.0, 1.0 - largest_gap))
        vacuum_gap = max(0.0, min(1.0, largest_gap))
        length_margin = length_ratio / 1.8
        gap_margin = vacuum_gap / 0.45
        span_margin = 0.55 / max(occupied_span, 1e-12)
        candidate_margin = min(length_margin, gap_margin, span_margin)
        candidates.append(
            VacuumAxisCandidate(
                axis=axis,
                axis_index=axis_index,
                axis_length=float(lengths[axis_index]),
                axis_length_ratio=length_ratio,
                occupied_span_fraction=occupied_span,
                vacuum_gap_fraction=vacuum_gap,
                candidate_margin=float(candidate_margin),
            )
        )
    return candidates


def _candidate_to_dict(candidate: VacuumAxisCandidate) -> dict:
    return {
        "axis": candidate.axis,
        "axis_index": candidate.axis_index,
        "axis_length": _json_float(candidate.axis_length),
        "axis_length_ratio": _json_float(candidate.axis_length_ratio),
        "occupied_span_fraction": _json_float(candidate.occupied_span_fraction),
        "vacuum_gap_fraction": _json_float(candidate.vacuum_gap_fraction),
        "candidate_margin": _json_float(candidate.candidate_margin),
    }


def _select_heuristic_vacuum_axis(
    input_cell_detail: dict | None,
) -> tuple[str, int | None, list[VacuumAxisCandidate], str | None]:
    candidates = _cell_geometry_candidates(input_cell_detail)
    if not candidates:
        return "none", None, [], None
    viable = [candidate for candidate in candidates if candidate.candidate_margin >= 1.0]
    if not viable:
        return "none", None, candidates, None
    viable.sort(key=lambda item: item.candidate_margin, reverse=True)
    if len(viable) > 1 and viable[1].candidate_margin >= 0.85 * viable[0].candidate_margin:
        return "ambiguous", None, candidates, None
    return "heuristic", viable[0].axis_index, candidates, viable[0].axis


def _extract_transform_matrix(transform) -> np.ndarray | None:
    if transform is None:
        return None
    if isinstance(transform, (list, tuple)) and len(transform) == 2:
        matrix = transform[0]
    else:
        matrix = transform
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        return None
    return array


def _distance_to_reciprocal_plane(k_input: np.ndarray, vacuum_axis_index: int) -> float:
    component = float(k_input[vacuum_axis_index])
    return float(abs(component - round(component)))


def _classify_kpoint_plane(
    k_acc_primitive,
    input_to_acc_matrix: np.ndarray,
    vacuum_axis_index: int,
    *,
    tol: float,
) -> tuple[str, np.ndarray, float]:
    k_acc = np.asarray(k_acc_primitive, dtype=float).reshape(3)
    k_input = input_to_acc_matrix.T @ k_acc
    component_distances = np.abs(k_input - np.round(k_input))
    vacuum_distance = _distance_to_reciprocal_plane(k_input, vacuum_axis_index)
    in_plane_axes = [axis for axis in range(3) if axis != vacuum_axis_index]
    in_plane_distances = component_distances[in_plane_axes]
    if vacuum_distance <= tol:
        classification = "in_plane"
    elif bool(np.all(in_plane_distances <= tol)):
        classification = "out_of_plane"
    else:
        classification = "mixed"
    return classification, k_input, vacuum_distance


def _little_group_for_primitive_kpoint(ssg, k_point, *, tol: float) -> list:
    k_array = np.asarray(k_point, dtype=float).reshape(3)
    effective_ops = [
        np.linalg.det(op[0]) * np.array(np.linalg.inv(op[1]).T)
        for op in ssg.gspg_ops_raw
    ]
    little_group = []
    if ssg.cptrans is None or np.allclose(ssg.cptrans, np.eye(3)):
        for op, effective_op in zip(ssg.gspg_ops_raw, effective_ops):
            target = effective_op @ k_array % 1
            if getNormInf(k_array % 1, target) < tol:
                little_group.append(op)
        return little_group

    cptrans = np.asarray(ssg.cptrans, dtype=float)
    cptrans_inv = np.linalg.inv(cptrans)
    conjugated_effective_ops = [
        cptrans_inv @ effective_op @ cptrans for effective_op in effective_ops
    ]
    primitive_kpoint = cptrans.T @ k_array % 1
    for op, conjugated_effective_op in zip(ssg.gspg_ops_raw, conjugated_effective_ops):
        transformed = conjugated_effective_op @ primitive_kpoint % 1
        if getNormInf(primitive_kpoint, transformed) < tol:
            little_group.append(op)
    return little_group


def _spin_splitting_for_little_group(little_group: list, *, tol: float) -> tuple[str, list[str]]:
    if not little_group:
        return "unknown", []
    spin_matrices = deduplicate_matrix_pairs(
        [np.asarray(op[0], dtype=float) - np.eye(3) for op in little_group],
        tol=tol,
    )
    stacked = np.vstack(spin_matrices)
    singular_values = np.linalg.svd(stacked.astype(np.float32))[1]
    spin_splitting = (
        "no spin splitting"
        if all(abs(value) > 1e-3 for value in singular_values)
        else "spin splitting"
    )
    return spin_splitting, combine_parametric_solutions(rref_with_tolerance(stacked))


def _generic_in_plane_input_kpoint(vacuum_axis_index: int) -> np.ndarray:
    values = np.zeros(3)
    in_plane_axes = [axis for axis in range(3) if axis != vacuum_axis_index]
    values[in_plane_axes[0]] = 0.237
    values[in_plane_axes[1]] = 0.371
    return values


def _generic_in_plane_acc_kpoint(
    input_to_acc_matrix: np.ndarray,
    vacuum_axis_index: int,
) -> np.ndarray:
    k_input = _generic_in_plane_input_kpoint(vacuum_axis_index)
    return np.linalg.solve(input_to_acc_matrix.T, k_input)


def _serialize_kpoint_analysis(
    *,
    label: str,
    k_acc,
    input_to_acc_matrix: np.ndarray,
    vacuum_axis_index: int,
    spin_splitting: str,
    spin_polarizations,
    tol: float,
    kind: str,
) -> dict:
    plane_class, k_input, vacuum_distance = _classify_kpoint_plane(
        k_acc,
        input_to_acc_matrix,
        vacuum_axis_index,
        tol=tol,
    )
    return {
        "label": label,
        "kind": kind,
        "k_acc_primitive": _json_vector(k_acc),
        "k_input_reciprocal": _json_mod1_vector(k_input),
        "plane_classification": plane_class,
        "vacuum_component_distance_to_integer": _json_float(vacuum_distance),
        "spin_splitting": spin_splitting,
        "spin_polarizations": list(spin_polarizations),
    }


def _kpoint_projection_summary(kpoint_rows: list[dict]) -> dict:
    acc_rows = [row for row in kpoint_rows if row.get("kind") == "acc_table"]
    labels_by_plane = {"in_plane": [], "out_of_plane": [], "mixed": [], "unknown": []}
    for row in acc_rows:
        plane = row.get("plane_classification") or "unknown"
        labels_by_plane.setdefault(plane, []).append(row.get("label"))
    return {
        "source": "acc_table",
        "total": len(acc_rows),
        "by_plane_count": {plane: len(labels) for plane, labels in labels_by_plane.items()},
        "labels_by_plane": labels_by_plane,
        "non_in_plane_labels": (
            labels_by_plane.get("out_of_plane", []) + labels_by_plane.get("mixed", [])
        ),
    }


def _wrapped_k_delta(target, source) -> np.ndarray:
    target_array = np.asarray(target, dtype=float).reshape(3)
    source_array = np.asarray(source, dtype=float).reshape(3)
    return (target_array - source_array + 0.5) % 1.0 - 0.5


def _rows_equal_field(left: dict | None, right: dict | None, field: str) -> bool | None:
    if left is None or right is None:
        return None
    return left.get(field) == right.get(field)


def _generic_point_comparison(kpoint_rows: list[dict], gp2d_row: dict | None, *, tol: float) -> dict:
    gp3d_row = next(
        (row for row in kpoint_rows if row.get("kind") == "acc_table" and row.get("label") == "GP"),
        None,
    )
    if gp3d_row is None or gp2d_row is None:
        return {
            "status": "missing_generic_point",
            "gp_3d": gp3d_row,
            "gp_2d": gp2d_row,
            "summary": "unknown",
            "k_input_changed": None,
            "k_acc_changed": None,
            "spin_splitting_changed": None,
            "spin_polarization_changed": None,
        }

    k_input_delta = _wrapped_k_delta(
        gp2d_row["k_input_reciprocal"],
        gp3d_row["k_input_reciprocal"],
    )
    k_acc_delta = _wrapped_k_delta(
        gp2d_row["k_acc_primitive"],
        gp3d_row["k_acc_primitive"],
    )
    k_input_changed = bool(np.max(np.abs(k_input_delta)) > tol)
    k_acc_changed = bool(np.max(np.abs(k_acc_delta)) > tol)
    spin_splitting_changed = not _rows_equal_field(gp3d_row, gp2d_row, "spin_splitting")
    spin_polarization_changed = not _rows_equal_field(gp3d_row, gp2d_row, "spin_polarizations")
    if spin_splitting_changed:
        summary = "k_changed_spin_splitting_changed" if k_input_changed else "spin_splitting_changed"
    elif k_input_changed:
        summary = "k_changed_spin_splitting_same"
    else:
        summary = "same_k_and_spin_splitting"
    return {
        "status": "compared",
        "gp_3d": gp3d_row,
        "gp_2d": gp2d_row,
        "k_input_delta_wrapped": _json_vector(k_input_delta),
        "k_acc_delta_wrapped": _json_vector(k_acc_delta),
        "k_input_changed": k_input_changed,
        "k_acc_changed": k_acc_changed,
        "spin_splitting_changed": spin_splitting_changed,
        "spin_polarization_changed": spin_polarization_changed,
        "summary": summary,
    }


def _interpret_2d_spin_splitting(kpoint_rows: list[dict]) -> tuple[str, str]:
    in_plane_rows = [row for row in kpoint_rows if row["plane_classification"] == "in_plane"]
    if any(row["spin_splitting"] == "spin splitting" for row in in_plane_rows):
        return "in_plane_k_dependent", "spin splitting"
    if in_plane_rows and all(row["spin_splitting"] != "unknown" for row in in_plane_rows):
        return "in_plane_no_spin_splitting", "no spin splitting"
    non_in_plane_rows = [
        row for row in kpoint_rows if row["plane_classification"] in {"out_of_plane", "mixed"}
    ]
    if any(row["spin_splitting"] == "spin splitting" for row in non_in_plane_rows):
        return "out_of_plane_only", "not_applicable"
    return "unknown", "unknown"


def build_quasi2d_diagnostics(
    *,
    input_cell_detail: dict | None,
    transformation_input_to_acc_primitive,
    acc_primitive_ssg,
    base_is_alter: str,
    tol: float,
    calculation_mode: str | None = "3d",
    vacuum_axis: str | None = "c",
) -> dict | None:
    calculation_mode = _normalize_calculation_mode(calculation_mode)
    if calculation_mode == "3d":
        return None
    dimension = None
    if calculation_mode == "quasi2d":
        dimension = "2d"
    explicit_axis, explicit_axis_index = _normalize_axis(vacuum_axis)
    heuristic_status, heuristic_axis_index, candidates, heuristic_axis = _select_heuristic_vacuum_axis(
        input_cell_detail
    )
    geometry = {
        "candidate_axes": [_candidate_to_dict(candidate) for candidate in candidates],
    }

    source = "none"
    vacuum_axis = None
    vacuum_axis_index = None
    status = "not_applicable"
    if dimension == "2d":
        source = "runtime_parameter"
        if explicit_axis_index is not None:
            status = "explicit"
            vacuum_axis = explicit_axis
            vacuum_axis_index = explicit_axis_index
        elif heuristic_axis_index is not None:
            status = "runtime_mode_heuristic_axis"
            vacuum_axis = heuristic_axis
            vacuum_axis_index = heuristic_axis_index
        else:
            status = "ambiguous"
    elif dimension == "3d":
        source = "runtime_parameter"
    elif heuristic_status == "heuristic" and heuristic_axis_index is not None:
        source = "heuristic"
        status = "heuristic"
        dimension = "2d"
        vacuum_axis = heuristic_axis
        vacuum_axis_index = heuristic_axis_index
    elif heuristic_status == "ambiguous":
        source = "heuristic"
        status = "ambiguous"

    base_payload = {
        "status": status,
        "source": source,
        "calculation_mode": calculation_mode or "auto",
        "dimension": dimension or "3d_or_unknown",
        "vacuum_axis_input": vacuum_axis,
        "vacuum_axis_index": vacuum_axis_index,
        "geometry": geometry,
        "kpoints": [],
        "kpoint_projection_summary": {
            "source": "acc_table",
            "total": 0,
            "by_plane_count": {"in_plane": 0, "out_of_plane": 0, "mixed": 0, "unknown": 0},
            "labels_by_plane": {"in_plane": [], "out_of_plane": [], "mixed": [], "unknown": []},
            "non_in_plane_labels": [],
        },
        "diagnostic_points": [],
        "generic_point_comparison": {
            "status": "not_applicable",
            "gp_3d": None,
            "gp_2d": None,
            "summary": "not_applicable",
            "k_input_changed": None,
            "k_acc_changed": None,
            "spin_splitting_changed": None,
            "spin_polarization_changed": None,
        },
        "spin_splitting_2d": "not_applicable",
        "interpretation": "not_applicable",
        "is_alter_2d": "not_applicable",
    }
    if status == "ambiguous":
        base_payload["spin_splitting_2d"] = "ambiguous"
        base_payload["interpretation"] = "ambiguous"
        base_payload["is_alter_2d"] = "ambiguous"
        return base_payload
    if vacuum_axis_index is None:
        return base_payload

    input_to_acc_matrix = _extract_transform_matrix(transformation_input_to_acc_primitive)
    if input_to_acc_matrix is None:
        base_payload["status"] = "unknown"
        base_payload["spin_splitting_2d"] = "unknown"
        base_payload["interpretation"] = "unknown"
        base_payload["is_alter_2d"] = "unknown"
        return base_payload

    rows = []
    diagnostic_row = None
    kpoints = list(acc_primitive_ssg.kpoints_primitive)
    labels = list(acc_primitive_ssg.kpoints_label)
    polarizations = list(acc_primitive_ssg.spin_polarizations)
    spin_splitting_flags = list(acc_primitive_ssg.is_spinsplitting)
    for label, k_acc, spin_splitting, spin_polarization in zip(
        labels,
        kpoints,
        spin_splitting_flags,
        polarizations,
    ):
        rows.append(
            _serialize_kpoint_analysis(
                label=label,
                k_acc=k_acc,
                input_to_acc_matrix=input_to_acc_matrix,
                vacuum_axis_index=vacuum_axis_index,
                spin_splitting=spin_splitting,
                spin_polarizations=spin_polarization,
                tol=tol,
                kind="acc_table",
            )
        )

    try:
        diagnostic_k_acc = _generic_in_plane_acc_kpoint(input_to_acc_matrix, vacuum_axis_index)
        diagnostic_little_group = _little_group_for_primitive_kpoint(
            acc_primitive_ssg,
            diagnostic_k_acc,
            tol=tol,
        )
        diagnostic_spin_splitting, diagnostic_spin_polarizations = _spin_splitting_for_little_group(
            diagnostic_little_group,
            tol=tol,
        )
        diagnostic_row = _serialize_kpoint_analysis(
            label="GP2D",
            k_acc=diagnostic_k_acc,
            input_to_acc_matrix=input_to_acc_matrix,
            vacuum_axis_index=vacuum_axis_index,
            spin_splitting=diagnostic_spin_splitting,
            spin_polarizations=diagnostic_spin_polarizations,
            tol=tol,
            kind="generated_in_plane_generic",
        )
        diagnostic_row["little_group_order"] = len(diagnostic_little_group)
        rows.append(diagnostic_row)
        base_payload["diagnostic_points"] = [diagnostic_row]
    except np.linalg.LinAlgError:
        base_payload["diagnostic_points"] = [
            {
                "label": "GP2D",
                "kind": "generated_in_plane_generic",
                "error": "singular_input_to_acc_reciprocal_transform",
            }
        ]

    interpretation, spin_splitting_2d = _interpret_2d_spin_splitting(rows)
    projection_summary = _kpoint_projection_summary(rows)
    generic_point_comparison = _generic_point_comparison(rows, diagnostic_row, tol=tol)
    base_payload.update(
        {
            "reciprocal_transform": {
                "input_to_acc_real_space_matrix": _json_matrix(input_to_acc_matrix),
                "acc_k_to_input_k_matrix": _json_matrix(input_to_acc_matrix.T),
            },
            "kpoints": rows,
            "kpoint_projection_summary": projection_summary,
            "generic_point_comparison": generic_point_comparison,
            "spin_splitting_2d": spin_splitting_2d,
            "interpretation": interpretation,
            "is_alter_2d": (
                "(Altermagnet)"
                if base_is_alter == "(Altermagnet)" and interpretation == "in_plane_k_dependent"
                else ""
            ),
        }
    )
    return base_payload
