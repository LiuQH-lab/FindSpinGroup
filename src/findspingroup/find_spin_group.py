import json
import re
import warnings
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from spintensor import solve_ahe, solve_bcd, solve_imd, solve_qmd
from spglib import get_symmetry_dataset,get_magnetic_symmetry_dataset
from findspingroup.core.identify_symmetry_from_ops import (
    deduplicate_matrix_pairs,
    get_magnetic_space_group_from_operations,
    identify_point_group,
)
from findspingroup.core.identify_spin_space_group import (
    _candidate_audit_failure,
    identify_spin_space_group_result,
)
from findspingroup.core.identify_index.contract_222 import (
    get_coplanar_222_lookup_entry,
    has_coplanar_222_lookup_group,
)
from findspingroup.core.tolerances import DEFAULT_TOL, Tolerances
from findspingroup.data import MSGMPG_DB
from findspingroup.data.acc_aligned_p_index_loader import (
    get_acc_aligned_conventional_to_primitive_p,
    get_spin_texture_config_for_ssg_label,
)
from findspingroup.ferroelectric import (
    build_ferroelectric_switching_payload,
    build_polar_axes_by_symmetry_payload,
    build_parent_standard_supercell_domain_coset_analysis,
)
from findspingroup.io import parse_poscar_file, parse_structure_file
from findspingroup.io.scif_generator import (
    _build_chen_linear_name,
    _format_scif_symbolic_scalar,
    affine_matrix_to_xyz_expression,
    generate_scif,
)
from findspingroup.quasi2d import build_quasi2d_diagnostics, prepare_quasi2d_input_cell
from findspingroup.spin_splitting import (
    classify_public_spin_texture_config,
    collinear_axis_constraint_operation,
)
from findspingroup.structure import SpinSpaceGroup,SpinSpaceGroupOperation
from findspingroup.structure.group import integer_points_in_new_cell, op_key, _resolve_point_group_info
from findspingroup.structure.cell import (
    CrystalCell,
    SpaceToleranceDegeneracyError,
    calculate_vector_coordinates_from_latticefactors,
)
from findspingroup.data.PG_SYMBOL import PG_IF_HEX_MAPPING, SG_HALL_MAPPING
from findspingroup.utils.matrix_utils import (
    general_positions_to_matrix,
    rref_with_tolerance,
    normalize_vector_to_zero,
)
from findspingroup.utils.symbolic_format import (
    format_symbolic_scalar,
    symbolize_numeric_tokens_in_string,
)
from findspingroup.utils.space_group_flags import (
    format_polar_axis_vector,
    msg_parent_space_group_info,
    space_group_is_centrosymmetric,
    space_group_is_chiral,
    space_group_is_polar,
    space_group_polar_axis_basis,
)


from findspingroup.utils.seitz_symbol import (
    calibrated_symbol_tol,
    canonicalize_group_seitz_descriptions,
    describe_spin_space_operation,
)
from findspingroup.version import __version__


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def _spin_texture_config_for_public_output(index: str) -> dict | None:
    try:
        return deepcopy(get_spin_texture_config_for_ssg_label(index))
    except KeyError:
        return None


_SPIN_TEXTURE_CONFIG_CLASSIFICATION_FIELDS = (
    "spin_texture_type",
    "order",
    "nullity",
    "spin_rank",
    "momentum_space_spin_configuration",
)


def _spin_texture_config_classification_key(payload: dict | None) -> tuple | None:
    if not isinstance(payload, dict):
        return None
    return tuple(payload.get(field) for field in _SPIN_TEXTURE_CONFIG_CLASSIFICATION_FIELDS)


def _safe_classify_spin_texture_config(
    operations,
    *,
    source: str,
    include_diagnostics: bool = False,
    k_dimension: int | None = None,
    k_names: tuple[str, ...] | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    zero_tol: float = 1e-8,
) -> dict | None:
    try:
        return classify_public_spin_texture_config(
            operations,
            source=source,
            include_diagnostics=include_diagnostics,
            k_dimension=k_dimension,
            k_names=k_names,
            atol=atol,
            rtol=rtol,
            zero_tol=zero_tol,
        )
    except Exception as exc:
        warnings.warn(
            f"Unable to classify spin texture config from {source}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _ossg_unit_cartesian_frame(cell: CrystalCell) -> np.ndarray:
    lattice_col = _lattice_column_matrix(cell)
    a_vec = lattice_col[:, 0]
    b_vec = lattice_col[:, 1]
    x_axis = a_vec / np.linalg.norm(a_vec)
    y_axis = b_vec - x_axis * float(np.dot(x_axis, b_vec))
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-12:
        raise ValueError("Cannot build OSSG unit Cartesian frame from collinear a/b lattice vectors")
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _ossg_setting_to_unit_cartesian_transform(cell: CrystalCell) -> np.ndarray:
    lattice_col = _lattice_column_matrix(cell)
    unit_frame = _ossg_unit_cartesian_frame(cell)
    return unit_frame.T @ lattice_col


def _similarity_to_unit_cartesian(matrix, transform: np.ndarray, transform_inv: np.ndarray) -> np.ndarray:
    return transform @ np.asarray(matrix, dtype=float) @ transform_inv


def _ssg_operation_pairs_in_ossg_unit_cartesian(
    ops,
    cell: CrystalCell,
    *,
    collinear_axis=None,
) -> list[dict]:
    transform = _ossg_setting_to_unit_cartesian_transform(cell)
    transform_inv = np.linalg.inv(transform)
    pairs = [
        {
            "spin_rotation": _similarity_to_unit_cartesian(
                op.spin_rotation,
                transform,
                transform_inv,
            ),
            "real_rotation": _similarity_to_unit_cartesian(
                op.rotation,
                transform,
                transform_inv,
            ),
        }
        for op in ops
    ]
    if collinear_axis is not None:
        pairs.append(collinear_axis_constraint_operation(transform @ np.asarray(collinear_axis, dtype=float)))
    return pairs


def _msg_operation_pairs_in_ossg_unit_cartesian(ops, cell: CrystalCell) -> list[dict]:
    transform = _ossg_setting_to_unit_cartesian_transform(cell)
    transform_inv = np.linalg.inv(transform)
    pairs = []
    for op in ops:
        time_reversal = int(op[0])
        real_rotation = _similarity_to_unit_cartesian(op[1], transform, transform_inv)
        spin_rotation = time_reversal * np.linalg.det(real_rotation) * real_rotation
        pairs.append(
            {
                "spin_rotation": spin_rotation,
                "real_rotation": real_rotation,
            }
        )
    return pairs


def _quasi2d_input_in_plane_axes(vacuum_axis_index: int) -> list[int]:
    return [axis for axis in range(3) if axis != int(vacuum_axis_index)]


def _quasi2d_axis_labels(axis_indices: list[int]) -> list[str]:
    labels = ["a", "b", "c"]
    return [labels[index] for index in axis_indices]


def _quasi2d_k_names(axis_indices: list[int]) -> tuple[str, ...]:
    names = ("kx", "ky", "kz")
    return tuple(names[index] for index in axis_indices)


def _input_in_plane_reciprocal_basis_in_ossg_unit_cartesian(
    *,
    convention_cell: CrystalCell,
    transformation_input_to_convention,
    vacuum_axis_index: int,
) -> tuple[np.ndarray, list[str]]:
    input_to_convention = np.asarray(transformation_input_to_convention[0], dtype=float)
    input_axes = _quasi2d_input_in_plane_axes(vacuum_axis_index)
    input_basis = np.eye(3)[:, input_axes]
    convention_reciprocal_basis = np.linalg.solve(input_to_convention.T, input_basis)
    direct_transform = _ossg_setting_to_unit_cartesian_transform(convention_cell)
    unit_reciprocal_basis = np.linalg.solve(direct_transform.T, convention_reciprocal_basis)
    return unit_reciprocal_basis, _quasi2d_axis_labels(input_axes)


def _reciprocal_action_from_unit_cartesian_pair(spin_rotation, real_rotation) -> np.ndarray:
    spin_rotation = np.asarray(spin_rotation, dtype=float)
    real_rotation = np.asarray(real_rotation, dtype=float)
    det_factor = 1.0 if np.linalg.det(spin_rotation) >= 0.0 else -1.0
    return det_factor * np.linalg.inv(real_rotation).T


def _reduce_unit_cartesian_pair_to_in_plane(pair: dict, in_plane_basis: np.ndarray, *, tol: float):
    q_unit = _reciprocal_action_from_unit_cartesian_pair(
        pair["spin_rotation"],
        pair["real_rotation"],
    )
    target = q_unit @ in_plane_basis
    q_2d, *_ = np.linalg.lstsq(in_plane_basis, target, rcond=None)
    residual = float(np.max(np.abs(in_plane_basis @ q_2d - target)))
    if residual > max(tol, 1e-8):
        return None, residual
    return {
        "Q": q_2d,
        "S": pair["spin_rotation"],
    }, residual


def _quasi2d_operation_pairs_in_plane(
    unit_cartesian_pairs: list[dict],
    *,
    in_plane_basis: np.ndarray,
    tol: float,
) -> tuple[list[dict], dict]:
    pairs = []
    skipped_residuals = []
    kept_residuals = []
    for pair in unit_cartesian_pairs:
        reduced, residual = _reduce_unit_cartesian_pair_to_in_plane(
            pair,
            in_plane_basis,
            tol=tol,
        )
        if reduced is None:
            skipped_residuals.append(residual)
            continue
        kept_residuals.append(residual)
        pairs.append(reduced)
    return pairs, {
        "input_operation_count": len(unit_cartesian_pairs),
        "plane_preserving_operation_count": len(pairs),
        "non_plane_preserving_operation_count": len(skipped_residuals),
        "skipped_operation_count": len(skipped_residuals),
        "max_kept_plane_residual": max(kept_residuals) if kept_residuals else None,
        "max_skipped_plane_residual": max(skipped_residuals) if skipped_residuals else None,
        "plane_residual_tol": max(tol, 1e-8),
    }


def _classify_quasi2d_spin_texture_config(
    pairs: list[dict],
    *,
    source: str,
    operation_audit: dict,
    in_plane_axes: list[str],
    k_names: tuple[str, ...],
    calibration_atol_limit: float,
    relax_without_reference: bool,
) -> dict | None:
    k_variable_labels = {
        k_name: f"input reciprocal {axis}*"
        for k_name, axis in zip(k_names, in_plane_axes)
    }
    if int(operation_audit.get("non_plane_preserving_operation_count") or 0) > 0:
        return {
            "status": "not_evaluated_non_plane_preserving_operations",
            "source": source,
            "basis_setting": "quasi2d_ossg_unit_cartesian_in_plane",
            "in_plane_k_axes": list(in_plane_axes),
            "k_variable_labels": k_variable_labels,
            "operation_audit": operation_audit,
        }
    if not pairs:
        return {
            "status": "not_evaluated_no_plane_preserving_operations",
            "source": source,
            "basis_setting": "quasi2d_ossg_unit_cartesian_in_plane",
            "in_plane_k_axes": list(in_plane_axes),
            "k_variable_labels": k_variable_labels,
            "operation_audit": operation_audit,
        }
    payload = _safe_classify_spin_texture_config(
        pairs,
        source=source,
        k_dimension=2,
        k_names=k_names,
    )
    if payload is None:
        return None
    if (
        relax_without_reference
        and payload.get("spin_texture_type") == "forbidden"
        and calibration_atol_limit > 1e-10
    ):
        relaxed = _safe_classify_spin_texture_config(
            pairs,
            source=source,
            k_dimension=2,
            k_names=k_names,
            atol=float(calibration_atol_limit),
            zero_tol=max(1e-8, min(float(calibration_atol_limit), 1e-4)),
        )
        if relaxed is not None and relaxed.get("spin_texture_type") != "forbidden":
            relaxed["calibration"] = {
                "status": "tolerance_relaxed_without_reference",
                "strict_key": list(_spin_texture_config_classification_key(payload) or []),
                "atol": float(calibration_atol_limit),
                "boundary_atol": float(calibration_atol_limit),
            }
            payload = relaxed
    payload["basis_setting"] = "quasi2d_ossg_unit_cartesian_in_plane"
    payload["in_plane_k_axes"] = list(in_plane_axes)
    payload["k_variable_labels"] = k_variable_labels
    payload["operation_audit"] = operation_audit
    return payload


def _quasi2d_spin_texture_config_from_ossg_convention(
    *,
    quasi_2d: dict | None,
    convention_ossg: SpinSpaceGroup,
    convention_cell: CrystalCell,
    transformation_input_to_convention,
    tol: float,
    calibration_atol_limit: float,
) -> dict:
    if not isinstance(quasi_2d, dict):
        return {}
    vacuum_axis_index = quasi_2d.get("vacuum_axis_index")
    if vacuum_axis_index is None or quasi_2d.get("dimension") != "2d":
        return {}

    in_plane_basis, in_plane_axes = _input_in_plane_reciprocal_basis_in_ossg_unit_cartesian(
        convention_cell=convention_cell,
        transformation_input_to_convention=transformation_input_to_convention,
        vacuum_axis_index=int(vacuum_axis_index),
    )
    in_plane_k_names = _quasi2d_k_names(_quasi2d_input_in_plane_axes(int(vacuum_axis_index)))
    collinear_axis = convention_ossg.sog_direction if convention_ossg.conf == "Collinear" else None
    no_soc_unit_pairs = _ssg_operation_pairs_in_ossg_unit_cartesian(
        convention_ossg.ops,
        convention_cell,
        collinear_axis=collinear_axis,
    )
    no_soc_pairs, no_soc_audit = _quasi2d_operation_pairs_in_plane(
        no_soc_unit_pairs,
        in_plane_basis=in_plane_basis,
        tol=tol,
    )
    msg_ops = _primitive_msg_ops_from_ssg(
        convention_ossg.msg_ops,
        tol=tol,
        time_reversal_resolver=convention_ossg.classify_magnetic_operation,
    )
    soc_unit_pairs = _msg_operation_pairs_in_ossg_unit_cartesian(msg_ops, convention_cell)
    soc_pairs, soc_audit = _quasi2d_operation_pairs_in_plane(
        soc_unit_pairs,
        in_plane_basis=in_plane_basis,
        tol=tol,
    )
    no_soc = _classify_quasi2d_spin_texture_config(
        no_soc_pairs,
        source="quasi2d_ossg_unit_cartesian_in_plane_ops",
        operation_audit=no_soc_audit,
        in_plane_axes=in_plane_axes,
        k_names=in_plane_k_names,
        calibration_atol_limit=calibration_atol_limit,
        relax_without_reference=convention_ossg.conf == "Collinear",
    )
    soc = _classify_quasi2d_spin_texture_config(
        soc_pairs,
        source="quasi2d_ossg_unit_cartesian_in_plane_msg_ops",
        operation_audit=soc_audit,
        in_plane_axes=in_plane_axes,
        k_names=in_plane_k_names,
        calibration_atol_limit=calibration_atol_limit,
        relax_without_reference=False,
    )
    return {
        "spin_texture_config_no_soc": no_soc,
        "spin_texture_config_soc": soc,
        "spin_texture_config_basis": {
            "setting": "quasi2d_ossg_unit_cartesian_in_plane",
            "in_plane_k_axes": in_plane_axes,
            "k_variable_labels": {
                k_name: f"input reciprocal {axis}*"
                for k_name, axis in zip(in_plane_k_names, in_plane_axes)
            },
            "input_in_plane_reciprocal_basis_in_ossg_unit_cartesian": in_plane_basis.tolist(),
        },
    }


def _reference_calibration_atol(reference: dict | None, diagnostics: dict | None, *, limit: float) -> float | None:
    if not isinstance(reference, dict) or not isinstance(diagnostics, dict):
        return None
    target_order = reference.get("order")
    if target_order is None:
        return None
    for order_payload in diagnostics.get("allowed_orders") or []:
        if order_payload.get("order") != target_order:
            continue
        singular = order_payload.get("min_nonzero_singular")
        if singular is None:
            return None
        candidate = float(singular) * 1.25
        if candidate <= 0:
            return None
        return min(candidate, float(limit))
    return None


def _classify_spin_texture_config_with_reference(
    primary_operations,
    *,
    primary_source: str,
    reference: dict | None,
    calibration_atol_limit: float,
    fallback_operations=None,
    fallback_source: str | None = None,
) -> dict | None:
    reference_key = _spin_texture_config_classification_key(reference)
    primary = _safe_classify_spin_texture_config(primary_operations, source=primary_source)
    if reference_key is None:
        if primary is not None:
            primary["basis_setting"] = "ossg_unit_cartesian"
        return primary
    if _spin_texture_config_classification_key(primary) == reference_key:
        primary["basis_setting"] = "ossg_unit_cartesian"
        return primary

    strict = primary
    strict_source = primary_source
    calibration_operations = primary_operations
    if fallback_operations is not None and fallback_source is not None:
        fallback = _safe_classify_spin_texture_config(fallback_operations, source=fallback_source)
        if _spin_texture_config_classification_key(fallback) == reference_key:
            fallback["basis_setting"] = "ossg_unit_cartesian"
            fallback["calibration"] = {
                "status": "matched_reference_with_full_operations",
                "reference_key": list(reference_key),
                "strict_key": list(_spin_texture_config_classification_key(primary) or []),
            }
            return fallback
        strict = fallback if fallback is not None else strict
        strict_source = fallback_source
        calibration_operations = fallback_operations

    diagnostics = _safe_classify_spin_texture_config(
        calibration_operations,
        source=strict_source,
        include_diagnostics=True,
    )
    candidate_atols = []
    candidate = _reference_calibration_atol(
        reference,
        diagnostics,
        limit=calibration_atol_limit,
    )
    if candidate is not None:
        candidate_atols.append(candidate)
    candidate_atols.append(float(calibration_atol_limit))

    attempts = []
    seen_atols: set[float] = set()
    for atol in candidate_atols:
        rounded_atol = float(round(float(atol), 12))
        if rounded_atol in seen_atols:
            continue
        seen_atols.add(rounded_atol)
        calibrated = _safe_classify_spin_texture_config(
            calibration_operations,
            source=strict_source,
            atol=float(atol),
            zero_tol=max(1e-8, min(float(atol), 1e-4)),
        )
        calibrated_key = _spin_texture_config_classification_key(calibrated)
        attempts.append({"atol": float(atol), "key": list(calibrated_key or [])})
        if calibrated_key == reference_key:
            calibrated["basis_setting"] = "ossg_unit_cartesian"
            calibrated["calibration"] = {
                "status": "calibrated_to_reference",
                "reference_key": list(reference_key),
                "strict_key": list(_spin_texture_config_classification_key(strict) or []),
                "atol": float(atol),
                "boundary_atol": float(calibration_atol_limit),
                "attempts": attempts,
            }
            return calibrated

    if strict is not None:
        strict["basis_setting"] = "ossg_unit_cartesian"
        strict["calibration"] = {
            "status": "reference_mismatch",
            "reference_key": list(reference_key),
            "strict_key": list(_spin_texture_config_classification_key(strict) or []),
            "boundary_atol": float(calibration_atol_limit),
            "attempts": attempts,
        }
    return strict


def _spin_texture_config_from_ossg_convention(
    convention_ossg: SpinSpaceGroup,
    convention_cell: CrystalCell,
    *,
    tol: float,
    calibration_atol_limit: float | None = None,
    reference: dict | None = None,
    generator_ops: list[SpinSpaceGroupOperation] | None = None,
) -> tuple[dict | None, dict | None]:
    if generator_ops is None:
        generator_ops = _symbol_generator_ops_for_current_basis(convention_ossg)
    collinear_axis = convention_ossg.sog_direction if convention_ossg.conf == "Collinear" else None
    generator_pairs = _ssg_operation_pairs_in_ossg_unit_cartesian(
        generator_ops or convention_ossg.ops,
        convention_cell,
        collinear_axis=collinear_axis,
    )
    full_pairs = _ssg_operation_pairs_in_ossg_unit_cartesian(
        convention_ossg.ops,
        convention_cell,
        collinear_axis=collinear_axis,
    )
    no_soc = _classify_spin_texture_config_with_reference(
        generator_pairs,
        primary_source="ossg_unit_cartesian_generators",
        reference=reference,
        calibration_atol_limit=tol if calibration_atol_limit is None else calibration_atol_limit,
        fallback_operations=full_pairs,
        fallback_source="ossg_unit_cartesian_ops",
    )

    msg_ops = _primitive_msg_ops_from_ssg(
        convention_ossg.msg_ops,
        tol=tol,
        time_reversal_resolver=convention_ossg.classify_magnetic_operation,
    )
    soc = _safe_classify_spin_texture_config(
        _msg_operation_pairs_in_ossg_unit_cartesian(msg_ops, convention_cell),
        source="ossg_unit_cartesian_msg_ops",
    )
    if soc is not None:
        soc["basis_setting"] = "ossg_unit_cartesian"
    return no_soc, soc


def _format_spin_only_direction(direction) -> str:
    if direction is None:
        return ""
    values = []
    for value in np.asarray(direction, dtype=float).reshape(-1):
        if abs(value) < 1e-4:
            value = 0.0
        values.append(_format_scif_symbolic_scalar(float(value), decimal_precision=6))
    return ",".join(values)


def _normalize_spin_only_direction(direction) -> np.ndarray | None:
    if direction is None:
        return None
    array = np.asarray(direction, dtype=float)
    if array.size == 0:
        return array
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm < 1e-12:
            return array
        return array / norm

    normalized = np.array(array, dtype=float, copy=True)
    if normalized.shape[0] == 3:
        for column_index in range(normalized.shape[1]):
            norm = np.linalg.norm(normalized[:, column_index])
            if norm >= 1e-12:
                normalized[:, column_index] /= norm
        return normalized

    for row_index in range(normalized.shape[0]):
        norm = np.linalg.norm(normalized[row_index])
        if norm >= 1e-12:
            normalized[row_index] /= norm
    return normalized


ACC_PRIMITIVE_SETTING = "acc_primitive"
ACC_CONVENTIONAL_SETTING = "acc_conventional"
INPUT_MAGNETIC_PRIMITIVE_SETTING = "input_magnetic_primitive"
INPUT_POSCAR_SETTING = "input_poscar"
ACC_PRIMITIVE_CARTESIAN_SETTING = "acc_primitive_cartesian"
ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING = "acc_primitive_poscar_spin_frame"
OSSG_ORIENTED_SPIN_FRAME_SETTING = "ossg_oriented_spin_frame"
G0_STANDARD_SETTING = "G0std"
L0_STANDARD_SETTING = "L0std"
SCIF_SPIN_FRAME_CARTESIAN = "cartesian"
SCIF_SPIN_FRAME_ORIENTED = "oriented"
SCIF_CELL_MODE_SSG_CONVENTION_CARTESIAN = "ssg_convention_cartesian"
SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED = "ssg_convention_oriented"
SCIF_CELL_MODE_DATABASE_STANDARD_CARTESIAN = "database_standard_cartesian"
SCIF_CELL_MODE_DATABASE_STANDARD_ORIENTED = "database_standard_oriented"
SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_CARTESIAN = "magnetic_primitive_cartesian"
SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_ORIENTED = "magnetic_primitive_oriented"
SCIF_CELL_MODE_INPUT_CARTESIAN = "input_cartesian"
SCIF_CELL_MODE_INPUT_ORIENTED = "input_oriented"

# Legacy public string modes. Keep them accepted by MagSymmetryResult.to_scif(...)
# while exposing the explicit setting × spin-frame modes in result.scif_cell_modes.
SCIF_CELL_MODE_INPUT_IDENTIFIED = "input_identified"
SCIF_CELL_MODE_MAGNETIC_PRIMITIVE = "magnetic_primitive"
SCIF_CELL_MODE_DATABASE_STANDARD = "database_standard"
_SCIF_CELL_MODE_ALIASES = {
    SCIF_CELL_MODE_INPUT_IDENTIFIED: SCIF_CELL_MODE_INPUT_ORIENTED,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE: SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_ORIENTED,
    SCIF_CELL_MODE_DATABASE_STANDARD: SCIF_CELL_MODE_DATABASE_STANDARD_ORIENTED,
}


def _resolve_scif_cell_mode(cell_mode: str) -> str:
    return _SCIF_CELL_MODE_ALIASES.get(cell_mode, cell_mode)


def _is_identify_index_database_missing_error(error: Exception) -> bool:
    message = str(error)
    return message.startswith("No identify-index reduction record for ")


def _identify_index_database_missing_label(error: Exception) -> str:
    return f"not in identify-index database: {error}"


def _should_degrade_identify_index_error(error: Exception) -> bool:
    return _is_identify_index_database_missing_error(error)


def _handle_missing_identify_index(source_name: str, error: Exception) -> str:
    label = _identify_index_database_missing_label(error)
    warnings.warn(
        f"Identify-index database entry unavailable for {source_name}: {error}. "
        f"Continuing with index set to {label!r}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return label


def _assert_ssg_ops_consistency(
    label: str,
    ssg: SpinSpaceGroup,
    *,
    tol: Tolerances = DEFAULT_TOL,
    identify_index_details: dict | None = None,
) -> None:
    failure = _candidate_audit_failure(list(ssg.ops), group_tol=tol)
    if failure is not None:
        raise ValueError(f"Inconsistent {label} SSG operations: {failure}")

    ssg.validate_nsspg_invariants()

    if identify_index_details is None:
        return

    expected = {
        "G0_id": int(ssg.G0_num),
        "L0_id": int(ssg.L0_num),
        "t_index": int(ssg.it),
        "k_index": int(ssg.ik),
        "configuration": ssg.conf,
    }
    for key, expected_value in expected.items():
        actual_value = identify_index_details.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Inconsistent identify-index details for {label}: "
                f"{key}={actual_value!r} does not match SSG ops value {expected_value!r}."
            )


def _exact_translation_distance(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _matrix_close_atol(left, right, *, tol: float, rtol: float = 1e-05) -> bool:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    return bool(np.all(np.abs(left_array - right_array) <= tol + rtol * np.abs(right_array)))


def _ops_match_with_exact_translation(
    left: SpinSpaceGroupOperation,
    right: SpinSpaceGroupOperation,
    tol: float,
) -> bool:
    return (
        _matrix_close_atol(left[0], right[0], tol=tol)
        and _matrix_close_atol(left[1], right[1], tol=tol)
        and _exact_translation_distance(left[2], right[2]) < tol
    )


def _deduplicate_ops_with_exact_translation(
    ops: list[SpinSpaceGroupOperation],
    tol: float,
) -> list[SpinSpaceGroupOperation]:
    ordered_ops = sorted(ops, key=op_key)
    unique_ops: list[SpinSpaceGroupOperation] = []
    unique_spin_rotations = np.empty((0, 3, 3), dtype=float)
    unique_real_rotations = np.empty((0, 3, 3), dtype=float)
    unique_translations = np.empty((0, 3), dtype=float)
    for op in ordered_ops:
        spin_rotation = np.asarray(op[0], dtype=float)
        real_rotation = np.asarray(op[1], dtype=float)
        translation = np.asarray(op[2], dtype=float)
        if unique_ops:
            spin_close = np.all(np.isclose(spin_rotation, unique_spin_rotations, atol=tol), axis=(1, 2))
            if np.any(spin_close):
                spin_indices = np.flatnonzero(spin_close)
                real_close = np.all(
                    np.isclose(real_rotation, unique_real_rotations[spin_indices], atol=tol),
                    axis=(1, 2),
                )
                if np.any(real_close):
                    candidate_indices = spin_indices[real_close]
                    translation_close = (
                        np.max(np.abs(translation - unique_translations[candidate_indices]), axis=1) < tol
                    )
                    if np.any(translation_close):
                        continue
        unique_ops.append(op)
        unique_spin_rotations = np.concatenate((unique_spin_rotations, spin_rotation[None, :, :]), axis=0)
        unique_real_rotations = np.concatenate((unique_real_rotations, real_rotation[None, :, :]), axis=0)
        unique_translations = np.concatenate((unique_translations, translation[None, :]), axis=0)
    return unique_ops


def _spin_space_group_operation_sets_match(left_ops, right_ops, *, tol: float) -> bool:
    left = sorted(list(left_ops), key=op_key)
    right = sorted(list(right_ops), key=op_key)
    if len(left) != len(right):
        return False

    unmatched = list(right)
    for left_op in left:
        for index, right_op in enumerate(unmatched):
            if left_op.is_same_with(right_op, atol=tol):
                unmatched.pop(index)
                break
        else:
            return False
    return True


def _cell_position_bucket_params(tol: float) -> tuple[int, int]:
    bins = max(1, int(np.ceil(1.0 / max(float(tol), 1e-12))))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(float(tol) / bucket_width)))
    return bins, neighbor_radius


def _cell_position_bucket_key(position, bins: int) -> tuple[int, int, int]:
    wrapped = np.mod(np.asarray(position, dtype=float), 1.0)
    indices = np.floor(wrapped * bins).astype(int) % bins
    return tuple(int(value) for value in indices)


def _cell_position_neighbor_keys(
    bucket_key: tuple[int, int, int],
    bins: int,
    neighbor_radius: int,
):
    for dx in range(-neighbor_radius, neighbor_radius + 1):
        for dy in range(-neighbor_radius, neighbor_radius + 1):
            for dz in range(-neighbor_radius, neighbor_radius + 1):
                yield (
                    (bucket_key[0] + dx) % bins,
                    (bucket_key[1] + dy) % bins,
                    (bucket_key[2] + dz) % bins,
                )


def _build_cell_preservation_checker(
    cell: CrystalCell,
    *,
    tol: Tolerances,
):
    positions = np.asarray(cell.positions, dtype=float)
    moments = np.asarray(cell.moments, dtype=float)
    elements = list(cell.elements)
    occupancies = [float(value) for value in cell.occupancies]
    bins, neighbor_radius = _cell_position_bucket_params(tol.space)
    buckets: dict[tuple[str, int, int, int], list[int]] = {}
    for index, position in enumerate(positions):
        key = (elements[index], *_cell_position_bucket_key(position, bins))
        buckets.setdefault(key, []).append(index)

    spatial_cache: dict[tuple[int, ...], list[list[int]] | None] = {}

    def spatial_key(rotation: np.ndarray, translation: np.ndarray) -> tuple[int, ...]:
        key_tol = 1e-8
        normalized_translation = normalize_vector_to_zero(translation, atol=key_tol)
        return (
            *np.rint(np.asarray(rotation, dtype=float).ravel() / key_tol).astype(np.int64),
            *np.rint(normalized_translation.ravel() / key_tol).astype(np.int64),
        )

    def spatial_candidates_for_op(
        real_rotation: np.ndarray,
        translation: np.ndarray,
    ) -> list[list[int]] | None:
        key = spatial_key(real_rotation, translation)
        if key in spatial_cache:
            return spatial_cache[key]

        candidates_by_atom: list[list[int]] = []
        for atom_index, position in enumerate(positions):
            transformed_position = normalize_vector_to_zero(
                real_rotation @ position + translation,
                atol=1e-8,
            )
            bucket_key = _cell_position_bucket_key(transformed_position, bins)
            candidates: list[int] = []
            for neighbor_key in _cell_position_neighbor_keys(bucket_key, bins, neighbor_radius):
                for candidate_index in buckets.get((elements[atom_index], *neighbor_key), ()):
                    if abs(occupancies[atom_index] - occupancies[candidate_index]) >= tol.occupancy:
                        continue
                    if getNormInf(transformed_position, positions[candidate_index]) >= tol.space:
                        continue
                    candidates.append(candidate_index)
            if not candidates:
                spatial_cache[key] = None
                return None
            candidates_by_atom.append(candidates)

        spatial_cache[key] = candidates_by_atom
        return candidates_by_atom

    def op_preserves_cell(op: SpinSpaceGroupOperation) -> bool:
        real_rotation = np.asarray(op.rotation, dtype=float)
        spin_rotation = np.asarray(op.spin_rotation, dtype=float)
        translation = np.asarray(op.translation, dtype=float)
        candidates_by_atom = spatial_candidates_for_op(real_rotation, translation)
        if candidates_by_atom is None:
            return False

        transformed_moments = moments @ spin_rotation.T
        for atom_index, candidates in enumerate(candidates_by_atom):
            if not any(
                np.linalg.norm(transformed_moments[atom_index] - moments[candidate_index]) < tol.moment
                for candidate_index in candidates
            ):
                return False
        return True

    return op_preserves_cell


def _ssg_ops_preserve_cell(
    cell: CrystalCell,
    ssg: SpinSpaceGroup,
    *,
    tol: Tolerances,
) -> bool:
    if cell.moments is None:
        return False

    op_preserves_cell = _build_cell_preservation_checker(cell, tol=tol)
    for op in ssg.ops:
        if not op_preserves_cell(op):
            return False
    return True


def _ssg_real_op_is_lattice_compatible(op: SpinSpaceGroupOperation, *, tol: float) -> bool:
    rotation = np.asarray(op.rotation, dtype=float)
    return bool(
        np.allclose(rotation, np.rint(rotation), atol=tol)
        and np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=tol)
    )


def _input_compatible_ssg_from_transformed_primitive(
    input_cell: CrystalCell,
    transformed_ssg: SpinSpaceGroup,
    *,
    tol: Tolerances,
) -> SpinSpaceGroup | None:
    if input_cell.moments is None:
        return None

    op_preserves_cell = _build_cell_preservation_checker(input_cell, tol=tol)
    compatible_ops = [
        op
        for op in transformed_ssg.ops
        if _ssg_real_op_is_lattice_compatible(op, tol=tol.m_matrix_tol)
        and op_preserves_cell(op)
    ]
    if not compatible_ops:
        return None
    if len(compatible_ops) != len(transformed_ssg.ops) and _candidate_audit_failure(
        compatible_ops,
        group_tol=tol,
    ) is not None:
        return None
    return SpinSpaceGroup(
        compatible_ops,
        tol=transformed_ssg.tol,
        real_space_metric=transformed_ssg.real_space_metric,
        identify_source_name=transformed_ssg.identify_source_name,
        identify_tol=transformed_ssg.identify_tol,
    )


def _can_reuse_transformed_input_ssg(
    input_cell: CrystalCell,
    transformed_ssg: SpinSpaceGroup,
    *,
    tol: Tolerances,
) -> bool:
    input_compatible_ssg = _input_compatible_ssg_from_transformed_primitive(
        input_cell,
        transformed_ssg,
        tol=tol,
    )
    return bool(
        input_compatible_ssg is not None
        and len(input_compatible_ssg.ops) == len(transformed_ssg.ops)
    )


def _diagnostic_ssg_index(file_name: str, ssg: SpinSpaceGroup, *, tol: float) -> str:
    try:
        return ssg.identify_index(file_name, tol=tol)
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        return _identify_index_database_missing_label(exc)


def _is_identity_setting_transform(transform: tuple[np.ndarray, np.ndarray], *, tol: float) -> bool:
    matrix = np.asarray(transform[0], dtype=float)
    shift = np.asarray(transform[1], dtype=float)
    if not np.allclose(matrix, np.eye(3), atol=tol):
        return False
    wrapped_shift = shift - np.round(shift)
    return bool(np.allclose(wrapped_shift, np.zeros(3), atol=tol))


def _deduplicate_translation_vectors_exact(translations, tol: float) -> list[np.ndarray]:
    unique_translations: list[np.ndarray] = []
    for translation in translations:
        vector = np.asarray(translation, dtype=float)
        if any(_exact_translation_distance(vector, existing) < tol for existing in unique_translations):
            continue
        unique_translations.append(vector)
    return unique_translations


def _translations_equivalent_mod_pure_translations(
    left,
    right,
    pure_translations,
    tol: float,
) -> bool:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    if _exact_translation_distance(left_vector, right_vector) < tol:
        return True

    difference = left_vector - right_vector
    for pure_translation in pure_translations:
        if getNormInf(difference, np.asarray(pure_translation, dtype=float), mode=True) < tol:
            return True
    return False


def _translation_equivalent_mod_integer(left, right, tol: float) -> bool:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    if _exact_translation_distance(left_vector, right_vector) < tol:
        return True
    difference = left_vector - right_vector
    nearest_integer = np.rint(difference)
    return bool(np.max(np.abs(difference - nearest_integer)) < tol)


def _real_op_bucket_decimals(tol: float) -> int:
    tol = float(max(tol, 1e-12))
    return max(0, int(np.ceil(-np.log10(tol))) - 1)


def _real_rotation_bucket_key(rotation, tol: float):
    decimals = _real_op_bucket_decimals(tol)
    arr = np.asarray(rotation, dtype=float).reshape(-1)
    return tuple(np.round(arr, decimals))


def _fractional_translation_bucket_params(tol: float):
    tol = float(max(tol, 1e-12))
    bins = max(1, int(np.ceil(1.0 / tol)))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(tol / bucket_width)))
    return bins, neighbor_radius


def _fractional_translation_bucket_key(translation, bins: int):
    wrapped = np.mod(np.asarray(translation, dtype=float), 1.0)
    indices = np.floor(wrapped * bins).astype(int) % bins
    return tuple(int(value) for value in indices)


def _fractional_translation_neighbor_keys(bucket_key, bins: int, neighbor_radius: int):
    for dx in range(-neighbor_radius, neighbor_radius + 1):
        for dy in range(-neighbor_radius, neighbor_radius + 1):
            for dz in range(-neighbor_radius, neighbor_radius + 1):
                yield (
                    (bucket_key[0] + dx) % bins,
                    (bucket_key[1] + dy) % bins,
                    (bucket_key[2] + dz) % bins,
                )


def _collect_unique_real_ops_with_spin_sets(
    ops: list[SpinSpaceGroupOperation],
    *,
    tol: float,
) -> list[dict]:
    records: list[dict] = []
    bins, neighbor_radius = _fractional_translation_bucket_params(tol)
    exact_buckets: dict[tuple, list[dict]] = {}
    for op in ops:
        spin_rotation = np.asarray(op[0], dtype=float)
        real_rotation = np.asarray(op[1], dtype=float)
        translation = np.asarray(op[2], dtype=float)

        matched = None
        rotation_key = _real_rotation_bucket_key(real_rotation, tol)
        translation_key = _fractional_translation_bucket_key(translation, bins)
        for neighbor_key in _fractional_translation_neighbor_keys(translation_key, bins, neighbor_radius):
            for record in exact_buckets.get((rotation_key, neighbor_key), ()):
                if not _matrix_close_atol(real_rotation, record["rotation"], tol=tol, rtol=0):
                    continue
                if _exact_translation_distance(translation, record["translation"]) >= tol:
                    continue
                matched = record
                break
            if matched is not None:
                break

        if matched is None:
            matched = {
                "rotation": real_rotation,
                "translation": translation,
                "spin_rotations": [],
            }
            records.append(matched)
            exact_buckets.setdefault((rotation_key, translation_key), []).append(matched)

        if not any(
            _matrix_close_atol(spin_rotation, existing, tol=tol, rtol=0)
            for existing in matched["spin_rotations"]
        ):
            matched["spin_rotations"].append(spin_rotation)

    for record in records:
        record["spin_rotations"] = sorted(
            record["spin_rotations"],
            key=lambda matrix: tuple(np.round(np.asarray(matrix, dtype=float).flatten(), 6)),
        )
    return records


def _match_real_op_record(
    candidate: dict,
    records: list[dict],
    *,
    tol: float,
    pure_translation_vectors: list[np.ndarray],
    record_index: dict[tuple, list[dict]] | None = None,
) -> tuple[dict | None, str]:
    rotation_key = _real_rotation_bucket_key(candidate["rotation"], tol)
    if record_index is None:
        rotation_candidates = [
            record
            for record in records
            if _real_rotation_bucket_key(record["rotation"], tol) == rotation_key
        ]
    else:
        rotation_candidates = record_index.get(rotation_key, [])

    for record in rotation_candidates:
        if _matrix_close_atol(candidate["rotation"], record["rotation"], tol=tol, rtol=0) and _exact_translation_distance(
            candidate["translation"], record["translation"]
        ) < tol:
            return record, "exact"

    for record in rotation_candidates:
        if _matrix_close_atol(candidate["rotation"], record["rotation"], tol=tol, rtol=0) and _translation_equivalent_mod_integer(
            candidate["translation"], record["translation"], tol
        ):
            return record, "mod_integer"

    for record in rotation_candidates:
        if not _matrix_close_atol(candidate["rotation"], record["rotation"], tol=tol, rtol=0):
            continue
        if _translations_equivalent_mod_pure_translations(
            candidate["translation"],
            record["translation"],
            pure_translation_vectors,
            tol,
        ):
            return record, "mod_pure_translation"

    return None, "none"


def _build_real_op_record_match_index(
    records: list[dict],
    *,
    tol: float,
) -> dict[tuple, list[dict]]:
    index: dict[tuple, list[dict]] = {}
    for record in records:
        rotation_key = _real_rotation_bucket_key(record["rotation"], tol)
        index.setdefault(rotation_key, []).append(record)
    return index


def audit_spatial_transform_effect(
    ssg: SpinSpaceGroup,
    transformation_matrix: np.ndarray,
    origin_shift: np.ndarray,
    *,
    tol: float = 1e-6,
    det_tol: float = 1e-2,
    use_nssg: bool = True,
) -> dict:
    """
    Audit how a spatial setting transform affects the real-space part of an SSG.

    The helper distinguishes:
    - exact real-op preservation
    - preservation mod integer lattice vectors
    - preservation mod the group's pure real-space translations
    - whether the associated spin-op set attached to a matched real op changes
    """
    source_ops = list(ssg.nssg if use_nssg else ssg.ops)
    transformation_matrix = np.asarray(transformation_matrix, dtype=float)
    origin_shift = np.asarray(origin_shift, dtype=float)
    determinant = float(np.linalg.det(transformation_matrix))
    volume_preserving = abs(abs(determinant) - 1.0) <= det_tol

    if not volume_preserving:
        return {
            "tol": float(tol),
            "det_tol": float(det_tol),
            "use_nssg": bool(use_nssg),
            "transform_matrix": transformation_matrix.tolist(),
            "origin_shift": origin_shift.tolist(),
            "determinant": determinant,
            "volume_preserving": False,
            "can_be_affine_normalizer_equivalent": False,
            "source_real_op_count": None,
            "transformed_real_op_count": None,
            "real_ops_exact_same": False,
            "real_ops_same_mod_integer": False,
            "real_ops_same_mod_pure_translations": False,
            "paired_spin_changed_count": None,
            "unmatched_source_indices": None,
            "transformed_to_source": [],
        }

    transformed_ssg = ssg.transform(transformation_matrix, origin_shift)
    transformed_ops = list(transformed_ssg.nssg if use_nssg else transformed_ssg.ops)

    source_records = _collect_unique_real_ops_with_spin_sets(source_ops, tol=tol)
    transformed_records = _collect_unique_real_ops_with_spin_sets(transformed_ops, tol=tol)
    source_record_index = _build_real_op_record_match_index(source_records, tol=tol)
    transformed_record_index = _build_real_op_record_match_index(transformed_records, tol=tol)
    source_record_index_by_id = {
        id(record): index
        for index, record in enumerate(source_records)
    }
    pure_translation_vectors = [np.asarray(item[1], dtype=float) for item in ssg.pure_t_group]

    transformed_to_source = []
    exact_preserved = True
    mod_integer_preserved = True
    mod_pure_preserved = True
    paired_spin_changed_count = 0

    for transformed_index, transformed_record in enumerate(transformed_records):
        matched_record, match_kind = _match_real_op_record(
            transformed_record,
            source_records,
            tol=tol,
            pure_translation_vectors=pure_translation_vectors,
            record_index=source_record_index,
        )
        if match_kind != "exact":
            exact_preserved = False
        if match_kind not in {"exact", "mod_integer"}:
            mod_integer_preserved = False
        if match_kind not in {"exact", "mod_integer", "mod_pure_translation"}:
            mod_pure_preserved = False

        spin_set_same = None
        matched_index = None
        if matched_record is not None:
            matched_index = source_record_index_by_id.get(id(matched_record))
            source_spin_rotations = matched_record["spin_rotations"]
            transformed_spin_rotations = transformed_record["spin_rotations"]
            spin_set_same = (
                len(source_spin_rotations) == len(transformed_spin_rotations)
                and all(
                    _matrix_close_atol(left, right, tol=tol, rtol=0)
                    for left, right in zip(source_spin_rotations, transformed_spin_rotations)
                )
            )
            if not spin_set_same:
                paired_spin_changed_count += 1

        transformed_to_source.append(
            {
                "transformed_index": transformed_index,
                "source_index": matched_index,
                "match_kind": match_kind,
                "rotation": np.asarray(transformed_record["rotation"], dtype=float).tolist(),
                "translation": np.asarray(transformed_record["translation"], dtype=float).tolist(),
                "spin_set_same": spin_set_same,
                "source_spin_count": None if matched_record is None else len(matched_record["spin_rotations"]),
                "transformed_spin_count": len(transformed_record["spin_rotations"]),
            }
        )

    unmatched_source_indices = []
    for source_index, source_record in enumerate(source_records):
        _, match_kind = _match_real_op_record(
            source_record,
            transformed_records,
            tol=tol,
            pure_translation_vectors=pure_translation_vectors,
            record_index=transformed_record_index,
        )
        if match_kind == "none":
            unmatched_source_indices.append(source_index)

    return {
        "tol": float(tol),
        "det_tol": float(det_tol),
        "use_nssg": bool(use_nssg),
        "transform_matrix": transformation_matrix.tolist(),
        "origin_shift": origin_shift.tolist(),
        "determinant": determinant,
        "volume_preserving": True,
        "can_be_affine_normalizer_equivalent": mod_pure_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "source_real_op_count": len(source_records),
        "transformed_real_op_count": len(transformed_records),
        "real_ops_exact_same": exact_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "real_ops_same_mod_integer": mod_integer_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "real_ops_same_mod_pure_translations": mod_pure_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "paired_spin_changed_count": paired_spin_changed_count,
        "unmatched_source_indices": unmatched_source_indices,
        "transformed_to_source": transformed_to_source,
    }


def _identify_nssg_ops(conf: str, ops: list[SpinSpaceGroupOperation], tol: float) -> list[SpinSpaceGroupOperation]:
    if conf == 'Collinear':
        return [
            op for op in ops
            if np.allclose(op[0], -np.eye(3), atol=tol) or np.allclose(op[0], np.eye(3), atol=tol)
        ]
    if conf == 'Coplanar':
        return [op for op in ops if np.linalg.det(op[0]) > 0]
    return list(ops)


@dataclass
class IdentifyNoFracGroup:
    input_ops: list[SpinSpaceGroupOperation]
    conf: str
    tol: float
    ops: list[SpinSpaceGroupOperation] = field(init=False)
    spin_translation_group: list[SpinSpaceGroupOperation] = field(init=False)
    pure_translations: list[np.ndarray] = field(init=False)
    sog: list[SpinSpaceGroupOperation] = field(init=False)
    nssg: list[SpinSpaceGroupOperation] = field(init=False)
    n_spin_part_point_ops: list[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.ops = _deduplicate_ops_with_exact_translation(list(self.input_ops), self.tol)
        self.spin_translation_group = [
            op for op in self.ops if np.allclose(op[1], np.eye(3), atol=self.tol)
        ]
        self.pure_translations = _deduplicate_translation_vectors_exact(
            [
                np.asarray(op[2], dtype=float)
                for op in self.spin_translation_group
                if np.allclose(op[0], np.eye(3), atol=self.tol)
            ],
            self.tol,
        )
        if not self.pure_translations:
            self.pure_translations = [np.zeros(3)]
        self.sog = _deduplicate_ops_with_exact_translation(
            [
                op
                for op in self.spin_translation_group
                if _exact_translation_distance(op[2], np.zeros(3)) < self.tol
            ],
            self.tol,
        )
        self.nssg = _identify_nssg_ops(self.conf, self.ops, self.tol)
        self.n_spin_part_point_ops = deduplicate_matrix_pairs([op[0] for op in self.nssg], tol=0.1)


class MagSymmetryResult:
    def __init__(self, cell, symmetry, properties):
        self.input_ssg_ops = symmetry.get('input_ssg_ops', None)
        self.spin_only = symmetry.get('spin_only', None)
        self.ssg_std_cell = symmetry.get('ssg_std_cell', None)
        self.T_input_to_ssg_std = symmetry.get('T_input_to_ssg_std', None)
        self.T_input_to_mag_primitive = symmetry.get('T_input_to_mag_primitive', None)
        self.T_input_to_input_magnetic_primitive = symmetry.get(
            'T_input_to_input_magnetic_primitive',
            self.T_input_to_mag_primitive,
        )
        self.T_input_to_acc_primitive = symmetry.get('T_input_to_acc_primitive', None)
        self.G0_symbol = symmetry.get('G0_symbol', None)
        self.G0_num = symmetry.get('G0_num', None)
        self.L0_symbol = symmetry.get('L0_symbol', None)
        self.L0_num = symmetry.get('L0_num', None)
        self.it = symmetry.get('it', None)
        self.ik = symmetry.get('ik', None)
        self.SSPG_symbol_hm = symmetry.get('SSPG_symbol_hm', None)
        self.SSPG_symbol_s = symmetry.get('SSPG_symbol_s', None)
        self.input_space_group_number = symmetry.get('input_space_group_number', None)
        self.input_space_group_symbol = symmetry.get('input_space_group_symbol', None)
        self.sg_is_centrosymmetric = symmetry.get('sg_is_centrosymmetric', None)
        self.sg_is_polar = symmetry.get('sg_is_polar', None)
        self.sg_is_chiral = symmetry.get('sg_is_chiral', None)
        self.input_space_group_basis_or_setting = symmetry.get(
            'input_space_group_basis_or_setting',
            None,
        )
        self.source_structure_metadata = symmetry.get('source_structure_metadata', None)
        self.source_parent_space_group = symmetry.get('source_parent_space_group', None)
        self.source_cell_parameter_strings = symmetry.get('source_cell_parameter_strings', None)
        self.magnetic_site_summary = symmetry.get('magnetic_site_summary', None)
        self.input_cell_detail = cell.get('input_cell_detail', None)

        self.input_magnetic_primitive_cell = cell.get('input_magnetic_primitive_cell', None)
        self.input_magnetic_primitive_cell_setting = cell.get(
            'input_magnetic_primitive_cell_setting',
            INPUT_MAGNETIC_PRIMITIVE_SETTING,
        )
        self.input_magnetic_primitive_cell_poscar = cell.get(
            'input_magnetic_primitive_cell_poscar',
            None,
        )
        self.input_magnetic_primitive_cell_detail = cell.get(
            'input_magnetic_primitive_cell_detail',
            None,
        )

        self.magnetic_primitive_cell = cell.get(
            'magnetic_primitive_cell',
            cell['primitive_magnetic_cell'],
        )
        self.magnetic_primitive_cell_setting = cell.get(
            'magnetic_primitive_cell_setting',
            ACC_PRIMITIVE_SETTING,
        )
        self.magnetic_primitive_cell_poscar = cell.get(
            'magnetic_primitive_cell_poscar',
            cell['primitive_magnetic_cell_poscar'],
        )
        self.magnetic_primitive_cell_detail = cell.get(
            'magnetic_primitive_cell_detail',
            cell.get('primitive_magnetic_cell_detail', None),
        )

        self.primitive_magnetic_cell = cell['primitive_magnetic_cell']
        self.primitive_magnetic_cell_setting = cell.get(
            'primitive_magnetic_cell_setting',
            self.magnetic_primitive_cell_setting,
        )
        self.primitive_magnetic_cell_poscar = cell['primitive_magnetic_cell_poscar']
        self.scif = cell['scif']
        self.scif_outputs = cell.get(
            'scif_outputs',
            {
                SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED: self.scif,
            },
        )
        self.scif_cell_modes = cell.get(
            'scif_cell_modes',
            [SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED],
        )
        self.primitive_magnetic_cell_detail = cell.get(
            'primitive_magnetic_cell_detail',
            self.magnetic_primitive_cell_detail,
        )
        self.acc_primitive_magnetic_cell = cell.get(
            'acc_primitive_magnetic_cell',
            self.magnetic_primitive_cell,
        )
        self.acc_primitive_magnetic_cell_setting = cell.get(
            'acc_primitive_magnetic_cell_setting',
            self.primitive_magnetic_cell_setting,
        )
        self.acc_primitive_magnetic_cell_poscar = cell.get(
            'acc_primitive_magnetic_cell_poscar',
            self.primitive_magnetic_cell_poscar,
        )
        self.acc_primitive_magnetic_cell_detail = cell.get(
            'acc_primitive_magnetic_cell_detail',
            self.magnetic_primitive_cell_detail,
        )
        self.acc_conventional_cell = cell.get('acc_conventional_cell', None)
        self.acc_conventional_cell_setting = cell.get('acc_conventional_cell_setting', None)
        self.acc_conventional_cell_detail = cell.get('acc_conventional_cell_detail', None)
        self.g0_standard_cell = cell.get('g0_standard_cell', None)
        self.l0_standard_cell = cell.get('l0_standard_cell', None)
        self.convention_cell = cell.get('convention_cell', None)
        self.convention_cell_setting = cell.get('convention_cell_setting', None)
        self.convention_cell_detail = cell.get('convention_cell_detail', None)
        self.wp_chain = cell.get('wp_chain', None)
        self.acc_primitive_wp_chain = cell.get('acc_primitive_wp_chain', None)


        self.index = symmetry['index']
        self.conf = symmetry['configuration']
        self.magnetic_phase = symmetry['magnetic_phase']
        self.magnetic_phase_base = symmetry.get('magnetic_phase_base', self.magnetic_phase)
        self.magnetic_phase_modifier = symmetry.get('magnetic_phase_modifier', '')
        self.magnetic_phase_spin_orbit_magnet = symmetry.get('magnetic_phase_spin_orbit_magnet', '')
        self.magnetic_phase_details = symmetry.get('magnetic_phase_details', None)
        self.spin_texture_config_no_soc = symmetry.get(
            'spin_texture_config_no_soc',
            symmetry.get('spin_texture_config', None),
        )
        self.spin_texture_config_soc = symmetry.get('spin_texture_config_soc', None)
        self.spin_texture_config = symmetry.get('spin_texture_config', self.spin_texture_config_no_soc)
        self.acc = symmetry['acc']
        self.msg_acc = symmetry.get('msg_acc', None)
        self.KPOINTS = symmetry['KPOINTS']
        self.KPOINTS_setting = symmetry.get('KPOINTS_setting', ACC_PRIMITIVE_SETTING)
        self.KPOINTS_real_space_setting = symmetry.get(
            'KPOINTS_real_space_setting',
            self.KPOINTS_setting,
        )
        self.spin_polarizations = symmetry['spin_polarizations']
        self.spin_polarizations_setting = symmetry.get(
            'spin_polarizations_setting',
            ACC_PRIMITIVE_CARTESIAN_SETTING,
        )
        self.spin_polarizations_real_space_setting = symmetry.get(
            'spin_polarizations_real_space_setting',
            self.KPOINTS_real_space_setting,
        )
        self.spin_polarizations_spin_frame = symmetry.get(
            'spin_polarizations_spin_frame',
            self.spin_polarizations_setting,
        )
        self.spin_polarizations_acc_cartesian = symmetry.get(
            'spin_polarizations_acc_cartesian',
            self.spin_polarizations,
        )
        self.spin_polarizations_acc_cartesian_setting = symmetry.get(
            'spin_polarizations_acc_cartesian_setting',
            self.spin_polarizations_setting,
        )
        self.acc_primitive_real_cartesian_to_poscar_spin_frame = symmetry.get(
            'acc_primitive_real_cartesian_to_poscar_spin_frame',
            None,
        )
        self.poscar_spin_frame_to_acc_primitive_real_cartesian = symmetry.get(
            'poscar_spin_frame_to_acc_primitive_real_cartesian',
            None,
        )
        self.real_cartesian_to_spin_frame = symmetry.get(
            'real_cartesian_to_spin_frame',
            self.acc_primitive_real_cartesian_to_poscar_spin_frame,
        )
        self.spin_frame_to_real_cartesian = symmetry.get(
            'spin_frame_to_real_cartesian',
            self.poscar_spin_frame_to_acc_primitive_real_cartesian,
        )
        self.spin_polarizations_acc_poscar_spin_frame = symmetry.get(
            'spin_polarizations_acc_poscar_spin_frame',
            None,
        )
        self.spin_polarizations_acc_poscar_spin_frame_setting = symmetry.get(
            'spin_polarizations_acc_poscar_spin_frame_setting',
            ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
        )
        self.quasi_2d = symmetry.get('quasi_2d', None)
        self.ferroelectric_switching = symmetry.get('ferroelectric_switching', None)
        self.polar_axes_by_symmetry = symmetry.get('polar_axes_by_symmetry', None)


        self.input_magnetic_primitive_ssg_ops = symmetry.get('input_magnetic_primitive_ssg_ops', None)
        self.input_magnetic_primitive_ssg_setting = symmetry.get(
            'input_magnetic_primitive_ssg_setting',
            INPUT_MAGNETIC_PRIMITIVE_SETTING,
        )
        self.input_magnetic_primitive_ssg_seitz = symmetry.get(
            'input_magnetic_primitive_ssg_seitz',
            None,
        )
        self.input_magnetic_primitive_ssg_seitz_latex = symmetry.get(
            'input_magnetic_primitive_ssg_seitz_latex',
            None,
        )
        self.input_magnetic_primitive_ssg_seitz_descriptions = symmetry.get(
            'input_magnetic_primitive_ssg_seitz_descriptions',
            None,
        )
        self.input_magnetic_primitive_ssg_international_linear = symmetry.get(
            'input_magnetic_primitive_ssg_international_linear',
            None,
        )
        self.input_magnetic_primitive_ssg_international_latex = symmetry.get(
            'input_magnetic_primitive_ssg_international_latex',
            None,
        )
        self.input_magnetic_primitive_ssg_symbol_calibration_tol = symmetry.get(
            'input_magnetic_primitive_ssg_symbol_calibration_tol',
            None,
        )
        self.input_magnetic_primitive_ssg_type = symmetry.get(
            'input_magnetic_primitive_ssg_type',
            None,
        )

        self.magnetic_primitive_ssg_ops = symmetry.get(
            'magnetic_primitive_ssg_ops',
            symmetry['primitive_magnetic_cell_ssg_ops'],
        )
        self.magnetic_primitive_ssg_setting = symmetry.get(
            'magnetic_primitive_ssg_setting',
            ACC_PRIMITIVE_SETTING,
        )
        self.magnetic_primitive_ssg_seitz = symmetry.get(
            'magnetic_primitive_ssg_seitz',
            symmetry.get('primitive_magnetic_cell_ssg_seitz', None),
        )
        self.magnetic_primitive_ssg_seitz_latex = symmetry.get(
            'magnetic_primitive_ssg_seitz_latex',
            symmetry.get('primitive_magnetic_cell_ssg_seitz_latex', None),
        )
        self.magnetic_primitive_ssg_seitz_descriptions = symmetry.get(
            'magnetic_primitive_ssg_seitz_descriptions',
            symmetry.get('primitive_magnetic_cell_ssg_seitz_descriptions', None),
        )
        self.magnetic_primitive_ssg_international_linear = symmetry.get(
            'magnetic_primitive_ssg_international_linear',
            symmetry.get('primitive_magnetic_cell_ssg_international_linear', None),
        )
        self.magnetic_primitive_ssg_international_latex = symmetry.get(
            'magnetic_primitive_ssg_international_latex',
            symmetry.get('primitive_magnetic_cell_ssg_international_latex', None),
        )
        self.magnetic_primitive_ssg_symbol_calibration_tol = symmetry.get(
            'magnetic_primitive_ssg_symbol_calibration_tol',
            symmetry.get('primitive_magnetic_cell_ssg_symbol_calibration_tol', None),
        )
        self.magnetic_primitive_ssg_type = symmetry.get(
            'magnetic_primitive_ssg_type',
            symmetry.get('primitive_magnetic_cell_ssg_type', None),
        )

        self.primitive_magnetic_cell_ssg_ops = symmetry['primitive_magnetic_cell_ssg_ops']
        self.primitive_magnetic_cell_ssg_setting = symmetry.get(
            'primitive_magnetic_cell_ssg_setting',
            self.magnetic_primitive_ssg_setting,
        )
        self.primitive_magnetic_cell_ssg_seitz = symmetry.get('primitive_magnetic_cell_ssg_seitz', None)
        self.primitive_magnetic_cell_ssg_seitz_latex = symmetry.get(
            'primitive_magnetic_cell_ssg_seitz_latex',
            None,
        )
        self.primitive_magnetic_cell_ssg_seitz_descriptions = symmetry.get(
            'primitive_magnetic_cell_ssg_seitz_descriptions',
            None,
        )
        self.primitive_magnetic_cell_ssg_international_linear = symmetry.get(
            'primitive_magnetic_cell_ssg_international_linear', None
        )
        self.primitive_magnetic_cell_ssg_international_latex = symmetry.get(
            'primitive_magnetic_cell_ssg_international_latex', None
        )
        self.primitive_magnetic_cell_ssg_symbol_calibration_tol = symmetry.get(
            'primitive_magnetic_cell_ssg_symbol_calibration_tol',
            self.magnetic_primitive_ssg_symbol_calibration_tol,
        )
        self.acc_primitive_ssg_ops = symmetry.get(
            'acc_primitive_ssg_ops',
            self.primitive_magnetic_cell_ssg_ops,
        )
        self.acc_primitive_ssg_setting = symmetry.get(
            'acc_primitive_ssg_setting',
            self.primitive_magnetic_cell_ssg_setting,
        )
        self.acc_primitive_ssg_seitz = symmetry.get(
            'acc_primitive_ssg_seitz',
            self.primitive_magnetic_cell_ssg_seitz,
        )
        self.acc_primitive_ssg_seitz_latex = symmetry.get(
            'acc_primitive_ssg_seitz_latex',
            self.primitive_magnetic_cell_ssg_seitz_latex,
        )
        self.acc_primitive_ssg_seitz_descriptions = symmetry.get(
            'acc_primitive_ssg_seitz_descriptions',
            self.primitive_magnetic_cell_ssg_seitz_descriptions,
        )
        self.acc_primitive_ssg_international_linear = symmetry.get(
            'acc_primitive_ssg_international_linear',
            self.primitive_magnetic_cell_ssg_international_linear,
        )
        self.acc_primitive_ssg_international_latex = symmetry.get(
            'acc_primitive_ssg_international_latex',
            self.primitive_magnetic_cell_ssg_international_latex,
        )
        self.acc_primitive_ssg_symbol_calibration_tol = symmetry.get(
            'acc_primitive_ssg_symbol_calibration_tol',
            self.magnetic_primitive_ssg_symbol_calibration_tol,
        )
        self.acc_primitive_ssg_ops_cartesian = symmetry.get(
            'acc_primitive_ssg_ops_cartesian',
            None,
        )
        self.acc_primitive_ssg_seitz_cartesian = symmetry.get(
            'acc_primitive_ssg_seitz_cartesian',
            None,
        )
        self.acc_primitive_ssg_seitz_latex_cartesian = symmetry.get(
            'acc_primitive_ssg_seitz_latex_cartesian',
            None,
        )
        self.acc_primitive_ssg_ops_oriented = symmetry.get(
            'acc_primitive_ssg_ops_oriented',
            None,
        )
        self.acc_primitive_ssg_seitz_oriented = symmetry.get(
            'acc_primitive_ssg_seitz_oriented',
            None,
        )
        self.acc_primitive_ssg_seitz_latex_oriented = symmetry.get(
            'acc_primitive_ssg_seitz_latex_oriented',
            None,
        )
        self.acc_primitive_spin_only_direction_cartesian = symmetry.get(
            'acc_primitive_spin_only_direction_cartesian',
            "",
        )
        self.acc_primitive_spin_only_direction_poscar_spin_frame = symmetry.get(
            'acc_primitive_spin_only_direction_poscar_spin_frame',
            "",
        )
        self.input_ssg_ops_spin_cartesian = symmetry.get(
            'input_ssg_ops_spin_cartesian',
            None,
        )
        self.input_ssg_seitz_latex_spin_cartesian = symmetry.get(
            'input_ssg_seitz_latex_spin_cartesian',
            None,
        )
        self.input_ssg_ops_spin_oriented = symmetry.get(
            'input_ssg_ops_spin_oriented',
            None,
        )
        self.input_ssg_seitz_latex_spin_oriented = symmetry.get(
            'input_ssg_seitz_latex_spin_oriented',
            None,
        )
        self.input_wp_chain = cell.get('input_wp_chain', None)
        self.input_spin_only_direction_spin_cartesian = symmetry.get(
            'input_spin_only_direction_spin_cartesian',
            "",
        )
        self.input_spin_only_direction_spin_oriented = symmetry.get(
            'input_spin_only_direction_spin_oriented',
            "",
        )
        self.input_ssg_may_be_incomplete = symmetry.get(
            'input_ssg_may_be_incomplete',
            None,
        )
        self.input_setting_warning = symmetry.get(
            'input_setting_warning',
            None,
        )
        self.acc_conventional_ssg_ops = symmetry.get('acc_conventional_ssg_ops', None)
        self.acc_conventional_ssg_setting = symmetry.get('acc_conventional_ssg_setting', None)
        self.acc_conventional_ssg_seitz = symmetry.get('acc_conventional_ssg_seitz', None)
        self.acc_conventional_ssg_seitz_latex = symmetry.get('acc_conventional_ssg_seitz_latex', None)
        self.acc_conventional_ssg_seitz_descriptions = symmetry.get(
            'acc_conventional_ssg_seitz_descriptions',
            None,
        )
        self.acc_conventional_ssg_international_linear = symmetry.get(
            'acc_conventional_ssg_international_linear',
            None,
        )
        self.acc_conventional_ssg_international_latex = symmetry.get(
            'acc_conventional_ssg_international_latex',
            None,
        )
        self.acc_conventional_ssg_symbol_calibration_tol = symmetry.get(
            'acc_conventional_ssg_symbol_calibration_tol',
            None,
        )
        self.primitive_magnetic_cell_ssg_type = symmetry.get('primitive_magnetic_cell_ssg_type', None)
        self.spin_part_point_group = symmetry['full_spin_part_point_group']
        self.identify_index_details = symmetry.get('identify_index_details', None)
        self.acc_primitive_resolution_audit = symmetry.get('acc_primitive_resolution_audit', None)
        self.g0std_axis_collapse_audit = symmetry.get('g0std_axis_collapse_audit', None)
        self.msg_num = symmetry.get('msg_num', None)
        self.msg_type = symmetry.get('msg_type', None)
        self.msg_symbol = symmetry.get('msg_symbol', None)
        self.msg_bns_number = symmetry.get('msg_bns_number', None)
        self.msg_og_number = symmetry.get('msg_og_number', None)
        self.msg_parent_space_group_number = symmetry.get('msg_parent_space_group_number', None)
        self.msg_is_centrosymmetric = symmetry.get('msg_is_centrosymmetric', None)
        self.msg_is_polar = symmetry.get('msg_is_polar', None)
        self.msg_is_chiral = symmetry.get('msg_is_chiral', None)
        self.tolerances = symmetry.get('tolerances', None)
        self.symbol_calibration_tol = symmetry.get(
            'symbol_calibration_tol',
            self.acc_primitive_ssg_symbol_calibration_tol,
        )
        self.gspg_ops = symmetry.get('gspg_ops', None)
        self.gspg_raw_ops = symmetry.get('gspg_raw_ops', None)
        self.gspg_ops_xyz_uvw = symmetry.get('gspg_ops_xyz_uvw', None)
        self.gspg_raw_ops_xyz_uvw = symmetry.get('gspg_raw_ops_xyz_uvw', None)
        self.gspg_generator_indices = symmetry.get('gspg_generator_indices', None)
        self.gspg_generator_ops = symmetry.get('gspg_generator_ops', None)
        self.gspg_generator_ops_xyz_uvw = symmetry.get('gspg_generator_ops_xyz_uvw', None)
        self.gspg_spin_only_ops = symmetry.get('gspg_spin_only_ops', None)
        self.gspg_spin_only_ops_xyz_uvw = symmetry.get('gspg_spin_only_ops_xyz_uvw', None)
        self.gspg_text = symmetry.get('gspg_text', None)
        self.gspg_collinear_axis = symmetry.get('gspg_collinear_axis', None)
        self.gspg_symbol_linear = symmetry.get('gspg_symbol_linear', None)
        self.gspg_symbol_latex = symmetry.get('gspg_symbol_latex', None)
        self.gspg_effective_mpg_symbol = symmetry.get('gspg_effective_mpg_symbol', None)
        self.gspg_npg_symbol_s = symmetry.get('gspg_npg_symbol_s', None)
        self.gspg_output_mode = symmetry.get('gspg_output_mode', None)
        self.gspg_point_part_linear = symmetry.get('gspg_point_part_linear', None)
        self.gspg_real_space_setting = symmetry.get('gspg_real_space_setting', None)
        self.gspg_spin_frame_setting = symmetry.get('gspg_spin_frame_setting', None)
        self.gspg_spin_only_component_symbol_s = symmetry.get(
            'gspg_spin_only_component_symbol_s',
            None,
        )
        self.gspg_spin_only_part_linear = symmetry.get('gspg_spin_only_part_linear', None)
        self.gspg_symbol_mode = symmetry.get('gspg_symbol_mode', None)
        self.gspg_tentative_symbol_s = symmetry.get('gspg_tentative_symbol_s', None)
        self.g0_standard_ssg_ops = symmetry.get('g0_standard_ssg_ops', None)
        self.g0_standard_ssg_seitz = symmetry.get('g0_standard_ssg_seitz', None)
        self.g0_standard_ssg_seitz_latex = symmetry.get('g0_standard_ssg_seitz_latex', None)
        self.g0_standard_ssg_seitz_descriptions = symmetry.get(
            'g0_standard_ssg_seitz_descriptions',
            None,
        )
        self.l0_standard_ssg_ops = symmetry.get('l0_standard_ssg_ops', None)
        self.l0_standard_ssg_seitz = symmetry.get('l0_standard_ssg_seitz', None)
        self.l0_standard_ssg_seitz_latex = symmetry.get('l0_standard_ssg_seitz_latex', None)
        self.l0_standard_ssg_seitz_descriptions = symmetry.get(
            'l0_standard_ssg_seitz_descriptions',
            None,
        )
        self.convention_ssg_ops = symmetry.get('convention_ssg_ops', None)
        self.convention_ssg_setting = symmetry.get('convention_ssg_setting', None)
        self.convention_ssg_spin_frame_setting = symmetry.get(
            'convention_ssg_spin_frame_setting',
            None,
        )
        self.ossg_space_group_number = symmetry.get('ossg_space_group_number', None)
        self.ossg_is_centrosymmetric = symmetry.get('ossg_is_centrosymmetric', None)
        self.ossg_is_polar = symmetry.get('ossg_is_polar', None)
        self.ossg_is_chiral = symmetry.get('ossg_is_chiral', None)
        self.convention_spin_only_direction = symmetry.get('convention_spin_only_direction', "")
        self.convention_spin_only_direction_cartesian = symmetry.get(
            'convention_spin_only_direction_cartesian',
            "",
        )
        self.convention_ssg_seitz = symmetry.get('convention_ssg_seitz', None)
        self.convention_ssg_seitz_latex = symmetry.get('convention_ssg_seitz_latex', None)
        self.convention_ssg_seitz_descriptions = symmetry.get(
            'convention_ssg_seitz_descriptions',
            None,
        )
        self.convention_nssg_ops = symmetry.get('convention_nssg_ops', None)
        self.convention_nssg_seitz = symmetry.get('convention_nssg_seitz', None)
        self.convention_nssg_seitz_latex = symmetry.get('convention_nssg_seitz_latex', None)
        self.operation_views = symmetry.get('operation_views', None)
        self.convention_ssg_international_linear = symmetry.get(
            'convention_ssg_international_linear',
            None,
        )
        self.convention_ssg_international_latex = symmetry.get(
            'convention_ssg_international_latex',
            None,
        )
        self.convention_ssg_symbol_calibration_tol = symmetry.get(
            'convention_ssg_symbol_calibration_tol',
            None,
        )
        self.raw_T_input_to_G0std = symmetry.get('raw_T_input_to_G0std', None)
        self.raw_T_input_to_L0std = symmetry.get('raw_T_input_to_L0std', None)
        self.magnetic_primitive_msg_ops = symmetry.get(
            'magnetic_primitive_msg_ops',
            symmetry.get('primitive_msg_ops', None),
        )
        self.magnetic_primitive_msg_ops_setting = symmetry.get(
            'magnetic_primitive_msg_ops_setting',
            ACC_PRIMITIVE_SETTING,
        )
        self.magnetic_primitive_msg_ops_spin_frame_setting = symmetry.get(
            'magnetic_primitive_msg_ops_spin_frame_setting',
            None,
        )
        self.primitive_msg_ops = symmetry.get('primitive_msg_ops', self.magnetic_primitive_msg_ops)
        self.primitive_msg_ops_setting = symmetry.get(
            'primitive_msg_ops_setting',
            self.magnetic_primitive_msg_ops_setting,
        )
        self.acc_primitive_msg_ops = symmetry.get(
            'acc_primitive_msg_ops',
            self.primitive_msg_ops,
        )
        self.acc_primitive_msg_ops_setting = symmetry.get(
            'acc_primitive_msg_ops_setting',
            self.primitive_msg_ops_setting,
        )
        self.primitive_msg_ops_spin_frame_setting = symmetry.get(
            'primitive_msg_ops_spin_frame_setting',
            self.magnetic_primitive_msg_ops_spin_frame_setting,
        )
        self.acc_primitive_msg_ops_spin_frame_setting = symmetry.get(
            'acc_primitive_msg_ops_spin_frame_setting',
            self.primitive_msg_ops_spin_frame_setting,
        )
        self.ssg_little_group_ops = symmetry.get('ssg_little_group_ops', None)
        self.ssg_little_group_seitz_latex = symmetry.get('ssg_little_group_seitz_latex', None)
        self.msg_little_group_ops = symmetry.get('msg_little_group_ops', None)
        self.msg_little_group_seitz_latex = symmetry.get('msg_little_group_seitz_latex', None)
        self.msg_little_group_symbols = symmetry.get('msg_little_group_symbols', None)
        self.msg_spin_polarizations = symmetry.get('msg_spin_polarizations', None)
        self.msg_spin_polarizations_setting = symmetry.get(
            'msg_spin_polarizations_setting',
            ACC_PRIMITIVE_CARTESIAN_SETTING,
        )
        self.msg_spin_polarizations_real_space_setting = symmetry.get(
            'msg_spin_polarizations_real_space_setting',
            self.KPOINTS_real_space_setting,
        )
        self.msg_spin_polarizations_spin_frame = symmetry.get(
            'msg_spin_polarizations_spin_frame',
            self.msg_spin_polarizations_setting,
        )
        self.msg_spin_polarizations_acc_cartesian = symmetry.get(
            'msg_spin_polarizations_acc_cartesian',
            self.msg_spin_polarizations,
        )
        self.msg_spin_polarizations_acc_cartesian_setting = symmetry.get(
            'msg_spin_polarizations_acc_cartesian_setting',
            self.msg_spin_polarizations_setting,
        )
        self.msg_spin_polarizations_acc_poscar_spin_frame = symmetry.get(
            'msg_spin_polarizations_acc_poscar_spin_frame',
            None,
        )
        self.msg_spin_polarizations_acc_poscar_spin_frame_setting = symmetry.get(
            'msg_spin_polarizations_acc_poscar_spin_frame_setting',
            ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
        )
        self.T_input_to_G0std = symmetry.get('T_input_to_G0std', None)
        self.T_input_to_G0std_ops_nofrac = symmetry.get('T_input_to_G0std_ops_nofrac', None)
        self.T_G0std_to_primitive = symmetry.get('T_G0std_to_primitive', None)
        self.T_G0std_to_acc_primitive = symmetry.get(
            'T_G0std_to_acc_primitive',
            self.T_G0std_to_primitive,
        )
        self.T_acc_primitive_to_G0std = symmetry.get(
            'T_acc_primitive_to_G0std',
            None,
        )
        self.T_input_to_L0std = symmetry.get('T_input_to_L0std', None)
        self.T_L0std_to_primitive = symmetry.get('T_L0std_to_primitive', None)
        self.T_L0std_to_acc_primitive = symmetry.get(
            'T_L0std_to_acc_primitive',
            self.T_L0std_to_primitive,
        )
        self.T_acc_primitive_to_L0std = symmetry.get(
            'T_acc_primitive_to_L0std',
            None,
        )
        self.T_input_to_convention = symmetry.get('T_input_to_convention', None)
        self.T_G0std_to_input = symmetry.get('T_G0std_to_input', None)
        self.T_L0std_to_input = symmetry.get('T_L0std_to_input', None)
        self.T_acc_primitive_to_input = symmetry.get('T_acc_primitive_to_input', None)
        self.T_convention_to_input = symmetry.get('T_convention_to_input', None)
        self.T_convention_to_acc_primitive = symmetry.get(
            'T_convention_to_acc_primitive',
            None,
        )
        self.T_convention_to_acc_conventional = symmetry.get(
            'T_convention_to_acc_conventional',
            None,
        )
        self.T_convention_to_acc_conventional_is_convention_self_automorphism = symmetry.get(
            'T_convention_to_acc_conventional_is_convention_self_automorphism',
            None,
        )
        self.T_convention_to_acc_conventional_label = symmetry.get(
            'T_convention_to_acc_conventional_label',
            None,
        )
        self.T_convention_to_acc_conventional_audit = symmetry.get(
            'T_convention_to_acc_conventional_audit',
            None,
        )
        self.selected_standard_setting = symmetry.get('selected_standard_setting', None)
        self.T_selected_standard_to_acc_conventional = symmetry.get(
            'T_selected_standard_to_acc_conventional',
            None,
        )
        self.T_selected_standard_to_acc_conventional_is_self_automorphism = symmetry.get(
            'T_selected_standard_to_acc_conventional_is_self_automorphism',
            None,
        )
        self.T_selected_standard_to_acc_conventional_label = symmetry.get(
            'T_selected_standard_to_acc_conventional_label',
            None,
        )
        self.T_selected_standard_to_acc_conventional_audit = symmetry.get(
            'T_selected_standard_to_acc_conventional_audit',
            None,
        )


        self.spinsplitting_w_soc = properties['ss_w_soc']
        self.spinsplitting_wo_soc = properties['ss_wo_soc']
        self.ahc_w_soc = properties['ahc_w_soc']
        self.ahc_wo_soc = properties['ahc_wo_soc']
        self.is_alter = properties['is_alter']
        self.is_spin_orbit_magnet = properties.get('is_spin_orbit_magnet', '')
        self.tensor_outputs = properties.get('tensor_outputs', {})
        self.AHE_woSOC = self.tensor_outputs.get('AHE_woSOC')
        self.AHE_wSOC = self.tensor_outputs.get('AHE_wSOC')
        self.BCDTensor = self.tensor_outputs.get('BCDTensor')
        self.MSGBCDTensor = self.tensor_outputs.get('MSGBCDTensor')
        self.QMDTensor = self.tensor_outputs.get('QMDTensor')
        self.MSGQMDTensor = self.tensor_outputs.get('MSGQMDTensor')
        self.IMDTensor = self.tensor_outputs.get('IMDTensor')
        self.MSGIMDTensor = self.tensor_outputs.get('MSGIMDTensor')

    def __repr__(self):
        props = self.properties_summary()
        display_symbol = (
            self.primitive_magnetic_cell_ssg_international_linear
            or self.primitive_magnetic_cell_ssg_international_latex
            or self.primitive_magnetic_cell_ssg_seitz
            or "Unknown"
        )
        return (f"<{display_symbol}>\n"
                f"  index: {self.index}\n"
                f"  conf : {self.conf}\n"
                f"  phase: {self.magnetic_phase}\n"
                f"  acc  : {self.acc}\n"
                f"  properties: {{\n"
                f"      ss_w_soc : {props['ss_w_soc']},\n"
                f"      ss_wo_soc: {props['ss_wo_soc']},\n"
                f"      ahc_w_soc: {props['ahc_w_soc']},\n"
                f"      ahc_wo_soc: {props['ahc_wo_soc']},\n"
                f"      is_alter : {props['is_alter']},\n"
                f"      is_spin_orbit_magnet : {props['is_spin_orbit_magnet']}\n"
                f"  }}")

    def properties_summary(self):
        return {
            'ss_w_soc': self.spinsplitting_w_soc,
            'ss_wo_soc': self.spinsplitting_wo_soc,
            'ahc_w_soc': self.ahc_w_soc,
            'ahc_wo_soc': self.ahc_wo_soc,
            'is_alter': self.is_alter,
            'is_spin_orbit_magnet': self.is_spin_orbit_magnet,
        }

    def gspg_summary(self):
        return {
            'effective_mpg_symbol': self.gspg_effective_mpg_symbol,
            'npg_symbol_s': self.gspg_npg_symbol_s,
            'output_mode': self.gspg_output_mode,
            'point_part_linear': self.gspg_point_part_linear,
            'real_space_setting': self.gspg_real_space_setting,
            'spin_frame_setting': self.gspg_spin_frame_setting,
            'spin_only_component_symbol_s': self.gspg_spin_only_component_symbol_s,
            'spin_only_part_linear': self.gspg_spin_only_part_linear,
            'spin_space_point_group_symbol_hm': self.SSPG_symbol_hm,
            'spin_space_point_group_symbol_s': self.SSPG_symbol_s,
            'symbol_linear': self.gspg_symbol_linear,
            'symbol_mode': self.gspg_symbol_mode,
            'tentative_symbol_s': self.gspg_tentative_symbol_s,
            'text': self.gspg_text,
        }

    def to_summary_dict(self):
        return {
            'index': self.index,
            'conf': self.conf,
            'phase': self.magnetic_phase,
            'acc': self.acc,
            'properties': self.properties_summary(),
            'gspg': self.gspg_summary(),
            'spin_texture_config': self.spin_texture_config,
            'spin_texture_config_no_soc': self.spin_texture_config_no_soc,
            'spin_texture_config_soc': self.spin_texture_config_soc,
            'polar_axes_by_symmetry': self.polar_axes_by_symmetry,
            'ferroelectric_switching': self.ferroelectric_switching,
        }

    def _structured_ssg_payload(
        self,
        *,
        setting,
        spin_frame_setting=None,
        ops=None,
        seitz=None,
        seitz_latex=None,
        seitz_descriptions=None,
        international_linear=None,
        international_latex=None,
        symbol_calibration_tol=None,
        ssg_type=None,
    ):
        return {
            'setting': setting,
            'spin_frame_setting': spin_frame_setting,
            'ops': ops,
            'seitz': seitz,
            'seitz_latex': seitz_latex,
            'seitz_descriptions': seitz_descriptions,
            'international_linear': international_linear,
            'international_latex': international_latex,
            'symbol_calibration_tol': symbol_calibration_tol,
            'type': ssg_type,
        }

    def to_structured_dict(self):
        """Return a structured view of the full result without recomputation.

        ``to_dict()`` remains the compatibility surface for existing callers.
        This method groups the same data by semantic layer so newer consumers do
        not need to reverse-engineer meaning from flat field prefixes.
        """
        legacy = dict(self.__dict__)
        selected_database_standard_cell = (
            self.g0_standard_cell
            if self.selected_standard_setting == G0_STANDARD_SETTING
            else self.l0_standard_cell
        )

        return {
            'summary': {
                'index': self.index,
                'conf': self.conf,
                'phase': self.magnetic_phase,
                'phase_base': self.magnetic_phase_base,
                'phase_modifier': self.magnetic_phase_modifier,
                'acc': self.acc,
                'msg_acc': self.msg_acc,
                'spin_texture_config': self.spin_texture_config,
                'spin_texture_config_no_soc': self.spin_texture_config_no_soc,
                'spin_texture_config_soc': self.spin_texture_config_soc,
                'is_alter': self.is_alter,
                'is_spin_orbit_magnet': self.is_spin_orbit_magnet,
                'tolerances': self.tolerances,
                'source': {
                    'metadata': self.source_structure_metadata,
                    'parent_space_group': self.source_parent_space_group,
                    'cell_parameter_strings': self.source_cell_parameter_strings,
                },
            },
            'groups': {
                'input_space_group': {
                    'number': self.input_space_group_number,
                    'symbol': self.input_space_group_symbol,
                    'basis_or_setting': self.input_space_group_basis_or_setting,
                    'is_centrosymmetric': self.sg_is_centrosymmetric,
                    'is_polar': self.sg_is_polar,
                    'is_chiral': self.sg_is_chiral,
                },
                'G0': {
                    'number': self.G0_num,
                    'symbol': self.G0_symbol,
                },
                'L0': {
                    'number': self.L0_num,
                    'symbol': self.L0_symbol,
                    't_index': self.it,
                    'k_index': self.ik,
                },
                'spin_point_group': {
                    'hm': self.SSPG_symbol_hm,
                    'symbol': self.SSPG_symbol_s,
                    'full_hm': self.spin_part_point_group,
                },
                'gspg': self.gspg_summary(),
                'ossg': {
                    'space_group_number': self.ossg_space_group_number,
                    'symbol_linear': self.convention_ssg_international_linear,
                    'symbol_latex': self.convention_ssg_international_latex,
                    'is_centrosymmetric': self.ossg_is_centrosymmetric,
                    'is_polar': self.ossg_is_polar,
                    'is_chiral': self.ossg_is_chiral,
                    'real_space_setting': self.convention_ssg_setting,
                    'spin_frame_setting': self.convention_ssg_spin_frame_setting,
                    'spin_only_direction': self.convention_spin_only_direction,
                    'spin_only_direction_cartesian': self.convention_spin_only_direction_cartesian,
                },
                'msg': {
                    'num': self.msg_num,
                    'type': self.msg_type,
                    'symbol': self.msg_symbol,
                    'bns_number': self.msg_bns_number,
                    'og_number': self.msg_og_number,
                    'parent_space_group_number': self.msg_parent_space_group_number,
                    'is_centrosymmetric': self.msg_is_centrosymmetric,
                    'is_polar': self.msg_is_polar,
                    'is_chiral': self.msg_is_chiral,
                },
                'ssg_by_cell': {
                    'input_magnetic_primitive': self._structured_ssg_payload(
                        setting=self.input_magnetic_primitive_ssg_setting,
                        ops=self.input_magnetic_primitive_ssg_ops,
                        seitz=self.input_magnetic_primitive_ssg_seitz,
                        seitz_latex=self.input_magnetic_primitive_ssg_seitz_latex,
                        seitz_descriptions=self.input_magnetic_primitive_ssg_seitz_descriptions,
                        international_linear=self.input_magnetic_primitive_ssg_international_linear,
                        international_latex=self.input_magnetic_primitive_ssg_international_latex,
                        symbol_calibration_tol=self.input_magnetic_primitive_ssg_symbol_calibration_tol,
                        ssg_type=self.input_magnetic_primitive_ssg_type,
                    ),
                    'acc_primitive': self._structured_ssg_payload(
                        setting=self.acc_primitive_ssg_setting,
                        ops=self.acc_primitive_ssg_ops,
                        seitz=self.acc_primitive_ssg_seitz,
                        seitz_latex=self.acc_primitive_ssg_seitz_latex,
                        seitz_descriptions=self.acc_primitive_ssg_seitz_descriptions,
                        international_linear=self.acc_primitive_ssg_international_linear,
                        international_latex=self.acc_primitive_ssg_international_latex,
                        symbol_calibration_tol=self.acc_primitive_ssg_symbol_calibration_tol,
                        ssg_type=self.primitive_magnetic_cell_ssg_type,
                    ),
                    'acc_conventional': self._structured_ssg_payload(
                        setting=self.acc_conventional_ssg_setting,
                        ops=self.acc_conventional_ssg_ops,
                        seitz=self.acc_conventional_ssg_seitz,
                        seitz_latex=self.acc_conventional_ssg_seitz_latex,
                        seitz_descriptions=self.acc_conventional_ssg_seitz_descriptions,
                        international_linear=self.acc_conventional_ssg_international_linear,
                        international_latex=self.acc_conventional_ssg_international_latex,
                        symbol_calibration_tol=self.acc_conventional_ssg_symbol_calibration_tol,
                    ),
                    'convention': self._structured_ssg_payload(
                        setting=self.convention_ssg_setting,
                        spin_frame_setting=self.convention_ssg_spin_frame_setting,
                        ops=self.convention_ssg_ops,
                        seitz=self.convention_ssg_seitz,
                        seitz_latex=self.convention_ssg_seitz_latex,
                        seitz_descriptions=self.convention_ssg_seitz_descriptions,
                        international_linear=self.convention_ssg_international_linear,
                        international_latex=self.convention_ssg_international_latex,
                        symbol_calibration_tol=self.convention_ssg_symbol_calibration_tol,
                    ),
                    'g0_standard': self._structured_ssg_payload(
                        setting=G0_STANDARD_SETTING,
                        ops=self.g0_standard_ssg_ops,
                        seitz=self.g0_standard_ssg_seitz,
                        seitz_latex=self.g0_standard_ssg_seitz_latex,
                        seitz_descriptions=self.g0_standard_ssg_seitz_descriptions,
                    ),
                    'l0_standard': self._structured_ssg_payload(
                        setting=L0_STANDARD_SETTING,
                        ops=self.l0_standard_ssg_ops,
                        seitz=self.l0_standard_ssg_seitz,
                        seitz_latex=self.l0_standard_ssg_seitz_latex,
                        seitz_descriptions=self.l0_standard_ssg_seitz_descriptions,
                    ),
                },
                'msg_by_cell': {
                    'acc_primitive': {
                        'setting': self.acc_primitive_msg_ops_setting,
                        'spin_frame_setting': self.acc_primitive_msg_ops_spin_frame_setting,
                        'ops': self.acc_primitive_msg_ops,
                    },
                    'magnetic_primitive': {
                        'setting': self.magnetic_primitive_msg_ops_setting,
                        'spin_frame_setting': self.magnetic_primitive_msg_ops_spin_frame_setting,
                        'ops': self.magnetic_primitive_msg_ops,
                    },
                },
                'little_groups': {
                    'ssg_ops': self.ssg_little_group_ops,
                    'ssg_seitz_latex': self.ssg_little_group_seitz_latex,
                    'msg_ops': self.msg_little_group_ops,
                    'msg_seitz_latex': self.msg_little_group_seitz_latex,
                    'msg_symbols': self.msg_little_group_symbols,
                    'msg_spin_polarizations': {
                        'values': self.msg_spin_polarizations,
                        'setting': self.msg_spin_polarizations_setting,
                        'real_space_setting': self.msg_spin_polarizations_real_space_setting,
                        'spin_frame': self.msg_spin_polarizations_spin_frame,
                        'acc_cartesian': self.msg_spin_polarizations_acc_cartesian,
                        'acc_poscar_spin_frame': self.msg_spin_polarizations_acc_poscar_spin_frame,
                    },
                },
            },
            'cells': {
                'input': {
                    'detail': self.input_cell_detail,
                    'wp_chain': self.input_wp_chain,
                },
                'input_magnetic_primitive': {
                    'setting': self.input_magnetic_primitive_cell_setting,
                    'cell': self.input_magnetic_primitive_cell,
                    'detail': self.input_magnetic_primitive_cell_detail,
                },
                'database_standard': {
                    'selected': self.selected_standard_setting,
                    'cell': selected_database_standard_cell,
                    'g0_standard': self.g0_standard_cell,
                    'l0_standard': self.l0_standard_cell,
                    'wp_chain': self.wp_chain,
                },
                'convention': {
                    'setting': self.convention_cell_setting,
                    'cell': self.convention_cell,
                    'detail': self.convention_cell_detail,
                },
                'acc_primitive': {
                    'setting': self.acc_primitive_magnetic_cell_setting,
                    'cell': self.acc_primitive_magnetic_cell,
                    'detail': self.acc_primitive_magnetic_cell_detail,
                    'wp_chain': self.acc_primitive_wp_chain,
                },
                'acc_conventional': {
                    'setting': self.acc_conventional_cell_setting,
                    'cell': self.acc_conventional_cell,
                    'detail': self.acc_conventional_cell_detail,
                },
            },
            'transforms': {
                'input_to_input_magnetic_primitive': self.T_input_to_input_magnetic_primitive,
                'input_to_acc_primitive': self.T_input_to_acc_primitive,
                'input_to_G0std': self.T_input_to_G0std,
                'input_to_L0std': self.T_input_to_L0std,
                'input_to_convention': self.T_input_to_convention,
                'G0std_to_acc_primitive': self.T_G0std_to_acc_primitive,
                'L0std_to_acc_primitive': self.T_L0std_to_acc_primitive,
                'acc_primitive_to_G0std': self.T_acc_primitive_to_G0std,
                'acc_primitive_to_L0std': self.T_acc_primitive_to_L0std,
                'acc_primitive_to_input': self.T_acc_primitive_to_input,
                'convention_to_input': self.T_convention_to_input,
                'convention_to_acc_primitive': self.T_convention_to_acc_primitive,
                'convention_to_acc_conventional': self.T_convention_to_acc_conventional,
                'selected_standard_to_acc_conventional': self.T_selected_standard_to_acc_conventional,
                'raw': {
                    'input_to_G0std': self.raw_T_input_to_G0std,
                    'input_to_L0std': self.raw_T_input_to_L0std,
                },
                'audit': {
                    'acc_primitive_resolution': self.acc_primitive_resolution_audit,
                    'g0std_axis_collapse': self.g0std_axis_collapse_audit,
                    'convention_to_acc_conventional': self.T_convention_to_acc_conventional_audit,
                    'selected_standard_to_acc_conventional': (
                        self.T_selected_standard_to_acc_conventional_audit
                    ),
                },
            },
            'properties': {
                'magnetic_phase': {
                    'phase': self.magnetic_phase,
                    'base': self.magnetic_phase_base,
                    'modifier': self.magnetic_phase_modifier,
                    'spin_orbit_magnet_tag': self.magnetic_phase_spin_orbit_magnet,
                    'details': self.magnetic_phase_details,
                },
                'spin_splitting': {
                    'with_soc': self.spinsplitting_w_soc,
                    'without_soc': self.spinsplitting_wo_soc,
                    'is_alter': self.is_alter,
                    'is_spin_orbit_magnet': self.is_spin_orbit_magnet,
                    'spin_polarizations': {
                        'values': self.spin_polarizations,
                        'setting': self.spin_polarizations_setting,
                        'real_space_setting': self.spin_polarizations_real_space_setting,
                        'spin_frame': self.spin_polarizations_spin_frame,
                        'acc_cartesian': self.spin_polarizations_acc_cartesian,
                        'acc_poscar_spin_frame': self.spin_polarizations_acc_poscar_spin_frame,
                    },
                },
                'ahc': {
                    'with_soc': self.ahc_w_soc,
                    'without_soc': self.ahc_wo_soc,
                },
                'tensors': self.tensor_outputs,
                'magnetic_site': self.magnetic_site_summary,
                'quasi_2d': self.quasi_2d,
                'polar_axes_by_symmetry': self.polar_axes_by_symmetry,
                'ferroelectric_switching': self.ferroelectric_switching,
            },
            'artifacts': {
                'poscar': {
                    'input_magnetic_primitive': self.input_magnetic_primitive_cell_poscar,
                    'acc_primitive': self.acc_primitive_magnetic_cell_poscar,
                },
                'scif': {
                    'default': self.scif,
                    'by_mode': self.scif_outputs,
                    'modes': self.scif_cell_modes,
                },
                'kpoints': {
                    'acc_primitive': {
                        'text': self.KPOINTS,
                        'setting': self.KPOINTS_setting,
                        'real_space_setting': self.KPOINTS_real_space_setting,
                    },
                },
            },
            'legacy': legacy,
        }

    def to_dict(self):
        return self.__dict__

    def save_json(self):
        return json.dumps(self.__dict__, indent=4,cls=NumpyEncoder)

    def to_scif(
        self,
        *,
        cell_mode: str = SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED,
    ) -> str:
        resolved_cell_mode = _resolve_scif_cell_mode(cell_mode)
        try:
            return self.scif_outputs[resolved_cell_mode]
        except KeyError as exc:
            available = sorted(self.scif_outputs.keys())
            raise ValueError(
                f"Unsupported scif output cell_mode: {cell_mode}. "
                f"Available: {available}"
            ) from exc






AFM_LIKE_BASE_PHASES = {"AFM"}
FM_LIKE_BASE_PHASES = {"FM/FiM", "Compensated FiM"}


def is_alter(condition, magnetic_phase_base, spinsplitting):
    if (
        condition == 'Collinear'
        and magnetic_phase_base in AFM_LIKE_BASE_PHASES
        and spinsplitting == 'k-dependent'
    ):
        return '(Altermagnet)'
    return ''


def spin_splitting_wo_soc(magnetic_phase_base, is_ss_gp):
    if magnetic_phase_base in AFM_LIKE_BASE_PHASES:
        if is_ss_gp == "no spin splitting":
            return 'No'
        return 'k-dependent'
    return 'Zeeman'


def _spin_splitting_wo_soc_quasi2d(magnetic_phase_base: str, spin_splitting_2d: str | None) -> str:
    if magnetic_phase_base in FM_LIKE_BASE_PHASES:
        return 'Zeeman'
    if spin_splitting_2d == "spin splitting":
        return 'k-dependent'
    if spin_splitting_2d == "no spin splitting":
        return 'No'
    if spin_splitting_2d in {"ambiguous", "unknown", "not_applicable"}:
        return str(spin_splitting_2d)
    return 'unknown'


def _build_quasi2d_magnetic_phase(
    *,
    parent_magnetic_phase_payload: dict,
    quasi_2d: dict | None,
) -> str | None:
    if not isinstance(quasi_2d, dict):
        return None
    base_phase = parent_magnetic_phase_payload['base_phase']
    spin_splitting_2d = quasi_2d.get('spin_splitting_2d')
    ss_wo_soc_2d = _spin_splitting_wo_soc_quasi2d(base_phase, spin_splitting_2d)
    alter_tag_2d = is_alter(
        parent_magnetic_phase_payload['details'].get('conf'),
        base_phase,
        ss_wo_soc_2d,
    )
    som_tag = (
        parent_magnetic_phase_payload['spin_orbit_magnet_tag']
        if base_phase == 'AFM'
        else ''
    )
    phase = base_phase + alter_tag_2d
    if som_tag:
        phase += '\n' + som_tag
    return phase

def spin_splitting_w_soc(ssg:SpinSpaceGroup):
    if ssg.is_PT:
        return 'No'
    else:
        return 'Yes'


def is_ahc(mpg):
    if mpg == None:
        return 'Error, cannot determine MSG.'
    if mpg in MSGMPG_DB.FMMPG_INTlist:
        wSOC = 'Yes'
    else:
        wSOC = 'No'
    return wSOC


def _serialize_tensor_solution(solution, operations_count):
    constraint_matrix, nullspace_basis, relations, components = solution
    free_parameters = int(nullspace_basis.shape[1]) if nullspace_basis.ndim == 2 else 0

    def _symbolize_display(value):
        if isinstance(value, str):
            return symbolize_numeric_tokens_in_string(value, sqrt_tol=1e-4)
        if isinstance(value, list):
            return [_symbolize_display(item) for item in value]
        if isinstance(value, tuple):
            return [_symbolize_display(item) for item in value]
        if isinstance(value, dict):
            return {key: _symbolize_display(item) for key, item in value.items()}
        return value

    return {
        'operations_count': operations_count,
        'constraint_shape': list(constraint_matrix.shape),
        'nullspace_shape': list(nullspace_basis.shape),
        'free_parameters': free_parameters,
        'is_zero': free_parameters == 0,
        'relations': _symbolize_display(relations),
        'components': _symbolize_display(components),
        'nullspace_basis': nullspace_basis.tolist(),
    }


def _lattice_column_matrix(cell: CrystalCell) -> np.ndarray:
    return np.asarray(cell.lattice_matrix, dtype=float).T


def _cartesian_spin_only_direction_from_oriented(direction, cell: CrystalCell):
    if direction is None:
        return None
    direction_array = np.asarray(direction, dtype=float)
    if direction_array.size == 0:
        return direction_array
    lattice_col = _lattice_column_matrix(cell)
    if direction_array.ndim == 1:
        cartesian = lattice_col @ direction_array.reshape(3)
    elif direction_array.shape[0] == 3:
        cartesian = lattice_col @ direction_array
    else:
        cartesian = direction_array @ lattice_col.T
    return _normalize_spin_only_direction(cartesian)


def _cartesianize_similarity(matrix: np.ndarray, lattice_col: np.ndarray) -> np.ndarray:
    return lattice_col @ np.asarray(matrix, dtype=float) @ np.linalg.inv(lattice_col)


def _poscar_spin_frame_rotation(cell: CrystalCell) -> np.ndarray:
    # POSCAR export preserves the acc-primitive Cartesian spin frame.
    return np.eye(3)


def _ossg_oriented_spin_frame_ssg(ssg: SpinSpaceGroup, cell: CrystalCell) -> SpinSpaceGroup:
    lattice_col = _lattice_column_matrix(cell)
    oriented_ssg = ssg.transform_spin(np.linalg.inv(lattice_col))
    if oriented_ssg.real_space_metric is not None:
        return oriented_ssg
    return SpinSpaceGroup(
        oriented_ssg.ops,
        tol=ssg.tol,
        real_space_metric=np.asarray(cell.lattice_matrix, dtype=float) @ np.asarray(cell.lattice_matrix, dtype=float).T,
    )


def _tensor_ops_wo_soc(ssg: SpinSpaceGroup, cell: CrystalCell):
    lattice_col = _lattice_column_matrix(cell)
    oriented_ssg = _ossg_oriented_spin_frame_ssg(ssg, cell)
    return [
        [
            _cartesianize_similarity(Rs, lattice_col),
            _cartesianize_similarity(Rr, lattice_col),
        ]
        for Rs, Rr in oriented_ssg.gspg_ops_raw
    ]


def _tensor_ops_w_soc(ssg: SpinSpaceGroup, cell: CrystalCell, tol: float):
    lattice_col = _lattice_column_matrix(cell)
    oriented_ssg = _ossg_oriented_spin_frame_ssg(ssg, cell)
    magnetic_point_group = oriented_ssg.msg_ops
    return [
        [
            _cartesianize_similarity(Rs, lattice_col),
            _cartesianize_similarity(Rr, lattice_col),
        ]
        for Rs, Rr, _ in magnetic_point_group
    ]


def _compute_tensor_outputs(ssg: SpinSpaceGroup, cell: CrystalCell, tol: float):
    ops_wo_soc = _tensor_ops_wo_soc(ssg, cell)
    ops_w_soc = _tensor_ops_w_soc(ssg, cell, tol=tol)
    tensor_specs = {
        'AHE_woSOC': (solve_ahe, ops_wo_soc, {'symbol': r'\sigma', 'use_antisymmetry': True}),
        'AHE_wSOC': (solve_ahe, ops_w_soc, {'symbol': r'\sigma', 'use_antisymmetry': True}),
        'BCDTensor': (solve_bcd, ops_wo_soc, {'symbol': 'D'}),
        'MSGBCDTensor': (solve_bcd, ops_w_soc, {'symbol': 'D'}),
        'QMDTensor': (solve_qmd, ops_wo_soc, {'symbol': 'Q'}),
        'MSGQMDTensor': (solve_qmd, ops_w_soc, {'symbol': 'Q'}),
        'IMDTensor': (solve_imd, ops_wo_soc, {'symbol': 'I'}),
        'MSGIMDTensor': (solve_imd, ops_w_soc, {'symbol': 'I'}),
    }
    tensor_outputs = {}
    for key, (solver, operations, kwargs) in tensor_specs.items():
        try:
            tensor_outputs[key] = _serialize_tensor_solution(
                solver(operations, **kwargs),
                operations_count=len(operations),
            )
        except Exception as error:
            tensor_outputs[key] = {'error': str(error)}
    return tensor_outputs


def _serialize_cell_snapshot(cell: CrystalCell, site_order=None) -> dict:
    lattice, positions, type_ids, moments = cell.to_spglib(mag=True)
    positions = np.asarray(positions, dtype=float)
    type_ids = np.asarray(type_ids)
    moments = np.asarray(moments, dtype=float)
    if site_order is not None:
        order = np.asarray(site_order, dtype=int)
        if order.shape != (len(type_ids),):
            raise ValueError(
                f"site_order length {len(order)} does not match cell site count {len(type_ids)}"
            )
        positions = positions[order]
        type_ids = type_ids[order]
        moments = moments[order]
    return {
        'lattice': np.asarray(lattice, dtype=float).tolist(),
        'positions': positions.tolist(),
        'type_ids': [int(type_id) for type_id in type_ids],
        'moments': moments.tolist(),
        'elements': [cell.atom_types_to_symbol[type_id] for type_id in type_ids],
        'occupancies': [float(cell.atom_types_to_occupancies[type_id]) for type_id in type_ids],
    }


def _cell_to_spglib_in_snapshot_order(cell: CrystalCell, site_order=None):
    if site_order is None:
        return cell.to_spglib(mag=True)
    lattice, positions, type_ids, moments = cell.to_spglib(mag=True)
    order = np.asarray(site_order, dtype=int)
    if order.shape != (len(type_ids),):
        raise ValueError(
            f"site_order length {len(order)} does not match cell site count {len(type_ids)}"
        )
    return (
        lattice,
        [positions[index] for index in order],
        [type_ids[index] for index in order],
        [moments[index] for index in order],
    )


def _cell_to_poscar_in_snapshot_order(
    cell: CrystalCell,
    filename: str,
    site_order=None,
) -> str:
    if site_order is None:
        return cell.to_poscar(filename)
    snapshot = _serialize_cell_snapshot(cell, site_order=site_order)
    species = []
    counts = []
    for element in snapshot["elements"]:
        if species and species[-1] == element:
            counts[-1] += 1
        else:
            species.append(element)
            counts.append(1)

    information = filename + f'#FINDSPINGROUP(version{__version__})'
    scale = '1'
    lattice = '\n'.join(
        ' '.join(map(str, np.asarray(row, dtype=float).round(6)))
        for row in snapshot["lattice"]
    )
    positions = '\n'.join(
        ' '.join(f'{value:.8f}' for value in position)
        for position in snapshot["positions"]
    )
    magmom = '# MAGMOM=' + ' '.join(
        ' '.join(f'{value:.8f}' for value in moment)
        for moment in snapshot["moments"]
    )
    return '\n'.join(
        [
            information,
            scale,
            lattice,
            ' '.join(species),
            ' '.join(map(str, counts)),
            'direct',
            positions,
            magmom,
        ]
    )


def _normalise_polar_axis_vector(vector) -> tuple[float, float, float]:
    vector = np.asarray(vector, dtype=float)
    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs < 1e-12:
        return (0.0, 0.0, 0.0)
    vector = vector / max_abs
    vector[np.abs(vector) < 1e-10] = 0.0
    nonzero_indices = np.where(np.abs(vector) >= 1e-10)[0]
    if nonzero_indices.size and vector[int(nonzero_indices[0])] < 0:
        vector = -vector
    vector[np.abs(vector) < 1e-10] = 0.0
    return tuple(float(round(component, 12)) for component in vector)


def _polar_axis_payload_from_basis(
    basis,
    *,
    setting: str,
) -> list[dict]:
    if basis is None:
        return None
    return [
        {
            "label": format_polar_axis_vector(vector),
            "components": [float(component) for component in vector],
            "setting": setting,
        }
        for vector in basis
    ]


def _nonmagnetic_space_group_polar_symmetry_in_cell_basis(
    cell: CrystalCell,
    *,
    setting: str,
    tol_cfg: Tolerances,
) -> dict | None:
    dataset = get_symmetry_dataset(
        cell.to_spglib(mag=False),
        symprec=tol_cfg.space,
    )
    if dataset is None:
        return None

    space_group_number = int(dataset.number)
    axes_standard = space_group_polar_axis_basis(space_group_number)
    if axes_standard is None:
        allowed_polar_axes = None
    else:
        standard_to_current = np.linalg.inv(
            np.asarray(dataset.transformation_matrix, dtype=float)
        )
        axes_current = tuple(
            _normalise_polar_axis_vector(standard_to_current @ np.asarray(axis, dtype=float))
            for axis in axes_standard
        )
        allowed_polar_axes = _polar_axis_payload_from_basis(
            axes_current,
            setting=setting,
        )

    return {
        "source": "nonmagnetic_space_group_standard_transformed_to_ssg_convention",
        "msg_num": None,
        "msg_symbol": None,
        "space_group_number": space_group_number,
        "space_group_symbol": str(dataset.international),
        "is_polar": space_group_is_polar(space_group_number),
        "is_centrosymmetric": space_group_is_centrosymmetric(space_group_number),
        "allowed_polar_axes": allowed_polar_axes,
        "allowed_polar_axes_setting": setting,
        "allowed_polar_axes_source": (
            "space_group_standard_basis_transformed_to_ssg_convention"
        ),
        "standard_basis_transform": {
            "current_to_standard": np.asarray(
                dataset.transformation_matrix,
                dtype=float,
            ).tolist(),
            "standard_to_current": standard_to_current.tolist()
            if axes_standard is not None
            else None,
        },
    }


def _build_g0std_parent_coset_analysis(
    *,
    g0std_cell: CrystalCell,
    g0std_ssg: SpinSpaceGroup,
    ordered_magnetic_ops: list[tuple[np.ndarray, np.ndarray, int]],
    ordered_space_group_number: int | None,
    ordered_subgroup_source: str,
    relation_layer: str,
    subgroup_time_branch_scope: str,
    tol_cfg: Tolerances,
) -> dict:
    dataset = get_symmetry_dataset(
        g0std_cell.to_spglib(mag=False),
        symprec=tol_cfg.space,
    )
    if dataset is None:
        return {
            "status": "not_evaluated_parent_space_group_detection_failed",
            "basis_setting": G0_STANDARD_SETTING,
            "candidate_reversal_domains": [],
        }

    # spglib's transformation_matrix is the direct coordinate transform from
    # the current G0std supercell to the detected parent standard setting.  Do
    # not rebuild this from Cartesian lattice matrices: for rotated standard
    # cells that loses the integer supercell relation and makes the ordered
    # OSSG projection fail the parent-subgroup check.
    child_basis_in_parent = np.asarray(dataset.transformation_matrix, dtype=float)
    return build_parent_standard_supercell_domain_coset_analysis(
        parent_space_group_number=int(dataset.number),
        parent_space_group_symbol=str(dataset.international),
        parent_hall_number=int(dataset.hall_number),
        child_basis_in_parent=child_basis_in_parent,
        child_origin_in_parent=np.asarray(dataset.origin_shift, dtype=float),
        ordered_magnetic_ops=ordered_magnetic_ops,
        ordered_space_group_number=ordered_space_group_number,
        basis_setting=G0_STANDARD_SETTING,
        ordered_subgroup_source=ordered_subgroup_source,
        relation_layer=relation_layer,
        subgroup_time_branch_scope=subgroup_time_branch_scope,
        ordered_cell=g0std_cell,
        collinear_axis=g0std_ssg.collinear_axis,
        tol=tol_cfg.m_matrix_tol,
    )


def _build_g0std_domain_reversal_coset_analysis(
    *,
    g0std_cell: CrystalCell,
    g0std_ssg: SpinSpaceGroup,
    ordered_space_group_number: int | None,
    tol_cfg: Tolerances,
) -> dict:
    # The exchange-limit domain quotient is SG(parent supercell) / OSSG_real,
    # not SG(parent supercell) / MSG. MSG compatibility is a later
    # classification layer for a representative, not the ordered-domain
    # stabilizer.
    ordered_magnetic_ops = [
        (
            np.asarray(rotation, dtype=float),
            np.asarray(translation, dtype=float),
            1,
        )
        for rotation, translation in g0std_ssg.G0_ops
    ]
    return _build_g0std_parent_coset_analysis(
        g0std_cell=g0std_cell,
        g0std_ssg=g0std_ssg,
        ordered_magnetic_ops=ordered_magnetic_ops,
        ordered_space_group_number=ordered_space_group_number,
        ordered_subgroup_source="ordered_spin_space_real_space_projection",
        relation_layer="exchange_spin_space",
        subgroup_time_branch_scope="unit",
        tol_cfg=tol_cfg,
    )


def _build_g0std_soc_domain_reversal_coset_analysis(
    *,
    g0std_cell: CrystalCell,
    g0std_ssg: SpinSpaceGroup,
    msg_parent_space_group_number: int | None,
    tol_cfg: Tolerances,
) -> dict:
    ordered_magnetic_ops = []
    for op in g0std_ssg.msg_ops:
        time_reversal = g0std_ssg.classify_magnetic_operation(op)
        if time_reversal is None:
            continue
        ordered_magnetic_ops.append(
            (
                np.asarray(op[1], dtype=float),
                np.asarray(op[2], dtype=float),
                int(time_reversal),
            )
        )
    return _build_g0std_parent_coset_analysis(
        g0std_cell=g0std_cell,
        g0std_ssg=g0std_ssg,
        ordered_magnetic_ops=ordered_magnetic_ops,
        ordered_space_group_number=msg_parent_space_group_number,
        ordered_subgroup_source="soc_magnetic_space_group",
        relation_layer="soc_magnetic",
        subgroup_time_branch_scope="full",
        tol_cfg=tol_cfg,
    )


def _build_domain_reversal_coset_analysis(
    *,
    source_metadata: dict | None,
    g0std_cell: CrystalCell,
    g0std_ssg: SpinSpaceGroup,
    ordered_space_group_number: int | None,
    tol_cfg: Tolerances,
) -> dict:
    return _build_g0std_domain_reversal_coset_analysis(
        g0std_cell=g0std_cell,
        g0std_ssg=g0std_ssg,
        ordered_space_group_number=ordered_space_group_number,
        tol_cfg=tol_cfg,
    )


def _serialize_gspg_ops(ops) -> list[list[list[list[float]]]]:
    return [
        [
            np.asarray(spin_rotation, dtype=float).tolist(),
            np.asarray(space_rotation, dtype=float).tolist(),
        ]
        for spin_rotation, space_rotation in ops
    ]


def _gspg_time_reversal_from_spin_rotation(spin_rotation: np.ndarray, *, tol: float = 1e-6) -> int | None:
    det = float(np.linalg.det(np.asarray(spin_rotation, dtype=float)))
    if abs(det - 1.0) < tol:
        return 1
    if abs(det + 1.0) < tol:
        return -1
    return None


def _serialize_gspg_xyz_uvw_ops(
    ops,
    *,
    tol: float = 1e-6,
    translation: np.ndarray | None = None,
) -> list[dict]:
    zero_translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    payload = []
    for idx, (spin_rotation, real_rotation) in enumerate(ops):
        spin_rotation = np.asarray(spin_rotation, dtype=float)
        real_rotation = np.asarray(real_rotation, dtype=float)
        time_reversal = _gspg_time_reversal_from_spin_rotation(spin_rotation, tol=tol)
        xyzt = affine_matrix_to_xyz_expression(real_rotation, zero_translation)
        if time_reversal is not None:
            xyzt = f"{xyzt},{time_reversal:+d}"
        payload.append(
            {
                "index": idx + 1,
                "xyzt": xyzt,
                "uvw": affine_matrix_to_xyz_expression(spin_rotation),
                "time_reversal": time_reversal,
                "spin_rotation": spin_rotation.tolist(),
                "real_rotation": real_rotation.tolist(),
                "translation": zero_translation.tolist(),
            }
        )
    return payload


def _format_gspg_xyz_uvw_text(rows: list[dict]) -> list[str]:
    return [
        f"{row['index']} {row['xyzt']} {row['uvw']}"
        for row in rows
    ]


def _gspg_pair_key(pair, *, decimals: int = 8) -> tuple:
    spin_rotation, real_rotation = pair
    return (
        tuple(np.round(np.asarray(spin_rotation, dtype=float).flatten(), decimals)),
        tuple(np.round(np.asarray(real_rotation, dtype=float).flatten(), decimals)),
    )


def _gspg_pair_multiply(left, right) -> list[np.ndarray]:
    return [
        np.asarray(left[0], dtype=float) @ np.asarray(right[0], dtype=float),
        np.asarray(left[1], dtype=float) @ np.asarray(right[1], dtype=float),
    ]


def _gspg_pair_inverse(pair) -> list[np.ndarray]:
    return [
        np.linalg.inv(np.asarray(pair[0], dtype=float)),
        np.linalg.inv(np.asarray(pair[1], dtype=float)),
    ]


def _gspg_pair_closure(generators, *, tol: float, limit: int) -> set[tuple]:
    identity = [np.eye(3), np.eye(3)]
    generator_words = deduplicate_matrix_pairs(
        [
            [np.asarray(item[0], dtype=float), np.asarray(item[1], dtype=float)]
            for generator in generators
            for item in (generator, _gspg_pair_inverse(generator))
        ],
        tol=tol,
    )
    seen = {_gspg_pair_key(identity)}
    queue = [identity]
    while queue:
        current = queue.pop(0)
        for generator in generator_words:
            next_pair = _gspg_pair_multiply(current, generator)
            next_key = _gspg_pair_key(next_pair)
            if next_key in seen:
                continue
            seen.add(next_key)
            if len(seen) > limit:
                raise RuntimeError("GSPG generator closure exceeded limit")
            queue.append(next_pair)
    return seen


def _select_gspg_generator_ops(ops, *, tol: float) -> list:
    operations = deduplicate_matrix_pairs(
        [[np.asarray(op[0], dtype=float), np.asarray(op[1], dtype=float)] for op in ops],
        tol=tol,
    )
    if not operations:
        return []

    identity_key = _gspg_pair_key([np.eye(3), np.eye(3)])
    target_keys = {_gspg_pair_key(op) for op in operations}
    if target_keys == {identity_key}:
        return [operations[0]]

    candidates = [op for op in operations if _gspg_pair_key(op) != identity_key]
    selected = []
    closure = {identity_key}
    limit = max(4096, len(target_keys) * 8)
    for candidate in candidates:
        if _gspg_pair_key(candidate) in closure:
            continue
        selected.append(candidate)
        try:
            closure = _gspg_pair_closure(selected, tol=tol, limit=limit)
        except RuntimeError:
            return candidates
        if target_keys.issubset(closure):
            return selected
    return selected if target_keys.issubset(closure) else candidates


def _gspg_pair_is_spin_only(pair, *, tol: float) -> bool:
    return np.allclose(np.asarray(pair[1], dtype=float), np.eye(3), atol=tol, rtol=0)


def _gspg_operation_indices_from_pairs(all_ops, selected_ops, *, tol: float) -> list[int]:
    indices = []
    for selected in selected_ops:
        for idx, candidate in enumerate(all_ops, start=1):
            if (
                np.allclose(candidate[0], selected[0], atol=tol, rtol=0)
                and np.allclose(candidate[1], selected[1], atol=tol, rtol=0)
            ):
                indices.append(idx)
                break
    return _deduplicate_operation_view_indices(indices)


def _gspg_text_rows_for_indices(rows: list[dict], indices: list[int]) -> list[str]:
    selected_rows = [
        rows[index - 1]
        for index in indices
        if 1 <= index <= len(rows)
    ]
    return _format_gspg_xyz_uvw_text(selected_rows)


def _format_gspg_spin_only_text(
    *,
    conf: str,
    spin_only_xyz_uvw: list[dict],
    collinear_axis,
    tol: float,
) -> list[str]:
    if conf == "Collinear":
        direction = _format_spin_only_direction(collinear_axis)
        if direction:
            return [f"Collinear direction: {direction}"]
        return ["Collinear direction:"]
    if conf == "Noncoplanar":
        identity_row = _serialize_gspg_xyz_uvw_ops(
            [[np.eye(3), np.eye(3)]],
            tol=tol,
        )
        return _format_gspg_xyz_uvw_text(identity_row)
    return _format_gspg_xyz_uvw_text(spin_only_xyz_uvw)


def _build_gspg_text(
    *,
    symbol_linear: str,
    spin_space_point_group_symbol_hm: str | None,
    spin_space_point_group_symbol_s: str | None,
    effective_mpg_symbol: str,
    real_space_setting: str,
    spin_frame_setting: str,
    generator_rows: list[str],
    operation_rows: list[str],
    spin_only_rows: list[str],
) -> str:
    spin_space_point_group_symbol = spin_space_point_group_symbol_hm or ""
    if spin_space_point_group_symbol_s:
        spin_space_point_group_symbol = (
            f"{spin_space_point_group_symbol} ({spin_space_point_group_symbol_s})"
            if spin_space_point_group_symbol
            else spin_space_point_group_symbol_s
        )
    lines = [
        f"GSPG linear symbol: {symbol_linear}",
        f"Spin-space point group symbol: {spin_space_point_group_symbol}",
        f"Effective MPG: {effective_mpg_symbol}",
        f"Real-space setting: {real_space_setting}",
        f"Spin-frame setting: {spin_frame_setting}",
        "",
        "generators (excluding spin-only):",
        *generator_rows,
        "",
        "operations:",
        *operation_rows,
        "",
        "spin only:",
        *spin_only_rows,
    ]
    return "\n".join(lines)


def _gspg_public_operation_sets(ssg: SpinSpaceGroup):
    raw_ops = deduplicate_matrix_pairs([[i[0], i[1]] for i in ssg.ops], tol=ssg.tol)
    if ssg.conf == "Collinear":
        presented_ops = deduplicate_matrix_pairs(
            [[np.asarray(op[0], dtype=float), np.asarray(op[1], dtype=float)] for op in ssg.nssg],
            tol=ssg.tol,
        )
    else:
        presented_ops = raw_ops
    return presented_ops, raw_ops


def _serialize_ssg_operation_matrices(
    ops: list[SpinSpaceGroupOperation],
) -> list[dict]:
    return [
        {
            "index": idx + 1,
            "spin_rotation": np.asarray(op.spin_rotation, dtype=float).tolist(),
            "real_rotation": np.asarray(op.rotation, dtype=float).tolist(),
            "translation": np.asarray(op.translation, dtype=float).tolist(),
        }
        for idx, op in enumerate(ops)
    ]


def _operation_view_index_rows(indices: list[int], *, label: str, note=None) -> dict:
    return {
        "label": label,
        "indices": [int(index) for index in indices],
        "operation_count": len(indices),
        "note": note,
    }


def _operation_view_all_row(
    ops_payload: list[dict],
    seitz_latex: list[str],
    *,
    label: str = "All operations",
) -> dict:
    operation_count = len(ops_payload)
    return {
        "label": label,
        "indices": list(range(1, operation_count + 1)),
        "ops": ops_payload,
        "seitz_latex": list(seitz_latex),
        "operation_count": operation_count,
        "note": None,
    }


def _operation_view_indices_from_ops(
    all_ops: list[SpinSpaceGroupOperation],
    selected_ops,
    *,
    tol: float,
    view_key: str,
    strict: bool = True,
) -> list[int]:
    index_by_id = {id(op): idx + 1 for idx, op in enumerate(all_ops)}
    indices: list[int] = []
    for selected_op in selected_ops:
        selected_index = index_by_id.get(id(selected_op))
        if selected_index is None:
            for candidate_index, candidate_op in enumerate(all_ops, start=1):
                if candidate_op.is_same_with(selected_op, atol=tol):
                    selected_index = candidate_index
                    break
        if selected_index is None and strict:
            raise ValueError(
                f"operation_views.{view_key}: selected operation is not present in all view"
            )
        if selected_index is not None:
            indices.append(int(selected_index))
    return indices


def _operation_view_indices_from_predicate(all_ops, predicate) -> list[int]:
    return [
        idx + 1
        for idx, op in enumerate(all_ops)
        if predicate(op)
    ]


def _deduplicate_operation_view_indices(indices: list[int]) -> list[int]:
    deduplicated: list[int] = []
    seen: set[int] = set()
    for index in indices:
        index = int(index)
        if index in seen:
            continue
        seen.add(index)
        deduplicated.append(index)
    return deduplicated


def _symbol_generator_ops_for_current_basis(ssg: SpinSpaceGroup) -> list[SpinSpaceGroupOperation]:
    symbol_payload = ssg.get_international_symbol(
        tol=ssg.symbol_calibration_tol,
        basis_mode="current",
    )
    generator_payloads = symbol_payload.get("generator_operations") or []
    return [
        SpinSpaceGroupOperation(
            payload["spin_rotation"],
            payload["real_rotation"],
            payload["translation"],
        )
        for payload in generator_payloads
        if isinstance(payload, dict)
    ]


def _transform_operation_generators(
    generator_ops: list[SpinSpaceGroupOperation],
    transform: np.ndarray,
    shift: np.ndarray,
    *,
    tol: float,
    real_space_metric=None,
) -> list[SpinSpaceGroupOperation]:
    if not generator_ops:
        return []
    return list(
        SpinSpaceGroup(
            generator_ops,
            tol=tol,
            real_space_metric=real_space_metric,
        ).transform(transform, shift, frac=True).ops
    )


def _transform_spin_generators(
    generator_ops: list[SpinSpaceGroupOperation],
    spin_transform: np.ndarray,
) -> list[SpinSpaceGroupOperation]:
    if not generator_ops:
        return []
    spin_transform = np.asarray(spin_transform, dtype=float)
    spin_transform_inv = np.linalg.inv(spin_transform)
    return [
        SpinSpaceGroupOperation(
            spin_transform @ op.spin_rotation @ spin_transform_inv,
            op.rotation,
            op.translation,
        )
        for op in generator_ops
    ]


def _operation_view_collinear_note(ssg: SpinSpaceGroup, *, spin_frame: str) -> dict:
    public_spin_frame = "oriented" if spin_frame == OSSG_ORIENTED_SPIN_FRAME_SETTING else spin_frame
    return {
        "type": "collinear",
        "text": "This operation list shows the convention nSSG for the collinear case.",
        "nssg_point_part_hm": ssg.n_spin_part_point_group_symbol_hm,
        "nssg_point_part_s": ssg.n_spin_part_point_group_symbol_s,
        "spin_only_symbol_hm": "∞m",
        "spin_only_symbol_s": "C∞v",
        "spin_only_direction": _format_spin_only_direction(ssg.sog_direction),
        "spin_frame": public_spin_frame,
    }


def _build_operation_view_set(
    ssg: SpinSpaceGroup,
    *,
    ops_payload: list[dict],
    seitz_latex: list[str],
    setting_label: str,
    spin_frame: str,
    generator_ops: list[SpinSpaceGroupOperation] | None = None,
) -> dict:
    all_ops = list(ssg.ops)
    if len(ops_payload) != len(all_ops):
        raise ValueError(
            f"operation_views.{setting_label}: ops payload length does not match SSG ops"
        )
    if len(seitz_latex) != len(all_ops):
        raise ValueError(
            f"operation_views.{setting_label}: Seitz list length does not match SSG ops"
        )

    is_collinear = ssg.conf == "Collinear"
    collinear_note = None
    if is_collinear:
        nssg_indices_in_source = _deduplicate_operation_view_indices(
            _operation_view_indices_from_ops(
                all_ops,
                ssg.nssg,
                tol=ssg.tol,
                view_key="nssg",
            )
        )
        all_ops = [all_ops[index - 1] for index in nssg_indices_in_source]
        ops_payload = _serialize_ssg_operation_matrices(all_ops)
        seitz_latex = [seitz_latex[index - 1] for index in nssg_indices_in_source]
        if generator_ops is None:
            view_ssg = SpinSpaceGroup(
                all_ops,
                tol=ssg.tol,
                real_space_metric=ssg.real_space_metric,
            )
            generator_ops = _symbol_generator_ops_for_current_basis(view_ssg)
        else:
            generator_ops = list(generator_ops)
        collinear_note = _operation_view_collinear_note(ssg, spin_frame=spin_frame)

    identity = np.eye(3)
    views = {
        "all": _operation_view_all_row(ops_payload, seitz_latex),
    }
    if is_collinear:
        views["nssg"] = _operation_view_all_row(
            ops_payload,
            seitz_latex,
            label="nSSG operations",
        )
        views["nssg"]["note"] = collinear_note

    if generator_ops is None:
        generator_ops = _symbol_generator_ops_for_current_basis(ssg)
    elif not is_collinear:
        generator_ops = list(generator_ops) + _symbol_generator_ops_for_current_basis(ssg)
    if generator_ops:
        generator_indices = _deduplicate_operation_view_indices(
            _operation_view_indices_from_ops(
                all_ops,
                generator_ops,
                tol=ssg.tol,
                view_key="generators",
                strict=False,
            )
        )
        if generator_indices:
            views["generators"] = _operation_view_index_rows(
                generator_indices,
                label="Symbol generators",
            )

    pure_translation_indices = _operation_view_indices_from_predicate(
        all_ops,
        lambda op: (
            np.allclose(op.spin_rotation, identity, atol=ssg.tol, rtol=0)
            and np.allclose(op.rotation, identity, atol=ssg.tol, rtol=0)
        ),
    )
    if pure_translation_indices:
        views["pure_translations"] = _operation_view_index_rows(
            pure_translation_indices,
            label="Pure translations",
        )

    spin_translation_indices = _operation_view_indices_from_predicate(
        all_ops,
        lambda op: np.allclose(op.rotation, identity, atol=ssg.tol, rtol=0),
    )
    if spin_translation_indices:
        views["spin_translations"] = _operation_view_index_rows(
            spin_translation_indices,
            label="Spin translations",
        )

    if not is_collinear:
        nssg_indices = _deduplicate_operation_view_indices(
            _operation_view_indices_from_ops(
                all_ops,
                ssg.nssg,
                tol=ssg.tol,
                view_key="nssg",
            )
        )
        if nssg_indices:
            views["nssg"] = _operation_view_index_rows(
                nssg_indices,
                label="nSSG operations",
            )

    l0_ops = [
        op
        for op in (all_ops if is_collinear else ssg.nssg)
        if np.allclose(op.spin_rotation, identity, atol=ssg.tol, rtol=0)
    ]
    if l0_ops:
        views["l0_operations"] = _operation_view_index_rows(
            _operation_view_indices_from_ops(
                all_ops,
                l0_ops,
                tol=ssg.tol,
                view_key="l0_operations",
            ),
            label="L0 operations",
        )

    return {
        "default_view": "nssg" if is_collinear else "all",
        "setting_label": setting_label,
        "spin_frame": spin_frame,
        "view_contract": "views with ops store serialized operations; index-only views store 1-based indices into all",
        "views": views,
    }


def _build_operation_views(operation_sources: dict[str, dict]) -> dict:
    operation_views = {}
    for setting_key, source in operation_sources.items():
        ssg = source.get("ssg")
        if ssg is None:
            continue
        ops_payload = source.get("ops_payload")
        if ops_payload is None:
            ops_payload = _serialize_ssg_operation_matrices(list(ssg.ops))
        seitz_latex = source.get("seitz_latex")
        if seitz_latex is None:
            seitz_latex = ssg.seitz_symbols_latex
        operation_views[setting_key] = _build_operation_view_set(
            ssg,
            ops_payload=ops_payload,
            seitz_latex=seitz_latex,
            setting_label=source.get("setting_label", setting_key),
            spin_frame=source.get("spin_frame", "cartesian"),
            generator_ops=source.get("generator_ops"),
        )
    return operation_views


def _serialize_msg_operation_matrices(
    ops: list[SpinSpaceGroupOperation],
    *,
    tol: float,
) -> list[dict]:
    return [
        {
            "index": idx + 1,
            "time_reversal": int(op.magnetic_time_reversal(atol=tol)),
            "real_rotation": np.asarray(op[1], dtype=float).tolist(),
            "translation": np.asarray(op[2], dtype=float).tolist(),
        }
        for idx, op in enumerate(ops)
    ]


def _serialize_msg_operation_rows(ops: list[list]) -> list[dict]:
    return [
        {
            "index": idx + 1,
            "time_reversal": int(time_reversal),
            "real_rotation": np.asarray(rotation, dtype=float).tolist(),
            "translation": np.asarray(translation, dtype=float).tolist(),
        }
        for idx, (time_reversal, rotation, translation) in enumerate(ops)
    ]


def _serialize_effective_mpg_ops(ops) -> list[list]:
    return [
        [
            int(time_reversal),
            np.asarray(rotation, dtype=float).tolist(),
        ]
        for time_reversal, rotation in ops
    ]


def _serialize_rotation_ops(ops) -> list[list[list[float]]]:
    return [np.asarray(rotation, dtype=float).tolist() for rotation in ops]


def _serialize_seitz_descriptions(descriptions) -> list[dict]:
    return json.loads(json.dumps(descriptions, cls=NumpyEncoder))


def _serialize_op_list_seitz_symbols(
    ops: list[SpinSpaceGroupOperation],
    *,
    tol: float,
) -> tuple[list[str], list[str]]:
    descriptions = [
        op.seitz_description(
            tol=tol,
            max_order=120,
            max_axis_denom=12,
        )
        for op in ops
    ]
    canonicalized = canonicalize_group_seitz_descriptions(
        descriptions,
        tol=tol,
        max_axis_denom=12,
    )
    return (
        [item["symbol"] for item in canonicalized],
        [item["symbol_latex"] for item in canonicalized],
    )


def _serialize_ssg_little_group_ops(
    little_groups: list[list[SpinSpaceGroupOperation]],
) -> list[list[dict]]:
    return [_serialize_ssg_operation_matrices(list(group)) for group in little_groups]


def _serialize_ssg_little_group_seitz_latex(
    little_groups: list[list[SpinSpaceGroupOperation]],
    *,
    tol: float,
) -> list[list[str]]:
    return [
        _serialize_op_list_seitz_symbols(list(group), tol=tol)[1]
        for group in little_groups
    ]


def _serialize_msg_little_group_ops(little_groups: list[list[list]]) -> list[list[dict]]:
    return [_serialize_msg_operation_rows(list(group)) for group in little_groups]


def _serialize_msg_little_group_seitz_latex(
    little_groups: list[list[list]],
    *,
    tol: float,
) -> list[list[str]]:
    output: list[list[str]] = []
    symbol_tol = calibrated_symbol_tol(tol)
    for group in little_groups:
        descriptions = []
        for time_reversal, rotation, translation in group:
            description = describe_spin_space_operation(
                np.eye(3),
                np.asarray(rotation, dtype=float),
                np.asarray(translation, dtype=float),
                tol=symbol_tol,
                max_order=120,
                max_axis_denom=12,
            )
            if int(time_reversal) < 0:
                description["spin"]["symbol"] = "1'"
                description["spin"]["symbol_latex"] = r"1^{\prime}"
            descriptions.append(description)
        canonicalized = canonicalize_group_seitz_descriptions(
            descriptions,
            tol=symbol_tol,
            max_axis_denom=12,
        )
        output.append([item["symbol_latex"] for item in canonicalized])
    return output


def _seitz_descriptions_with_cartesian_spin_symbols(
    ssg: SpinSpaceGroup,
    *,
    spin_to_cartesian: np.ndarray,
    tol: float,
    max_order: int = 120,
    max_axis_denom: int = 12,
) -> list[dict]:
    """Describe ops whose spin matrices are expressed in a non-orthonormal frame.

    Operation matrices are kept in their original frame for output.  For the
    spin part of the Seitz symbol, finite-order recognition uses the Euclidean
    Cartesian representation; the recognized spin axis is then converted back to
    the original spin frame before canonicalization and formatting.
    """
    symbol_tol = calibrated_symbol_tol(tol)
    spin_to_cartesian = np.asarray(spin_to_cartesian, dtype=float)
    cartesian_to_spin = np.linalg.inv(spin_to_cartesian)
    descriptions = []
    for op in ssg.ops:
        description = describe_spin_space_operation(
            spin_to_cartesian @ np.asarray(op.spin_rotation, dtype=float) @ cartesian_to_spin,
            op.rotation,
            op.translation,
            tol=symbol_tol,
            max_order=max_order,
            max_axis_denom=max_axis_denom,
        )
        spin_axis = description["spin"].get("axis_vector")
        if spin_axis is not None:
            spin_axis_in_frame = cartesian_to_spin @ np.asarray(spin_axis, dtype=float)
            description["spin"]["axis_vector"] = tuple(float(value) for value in spin_axis_in_frame)
        descriptions.append(description)
    return canonicalize_group_seitz_descriptions(
        descriptions,
        tol=symbol_tol,
        max_axis_denom=max_axis_denom,
    )


def _seitz_symbols_from_descriptions(descriptions: list[dict]) -> tuple[list[str], list[str]]:
    return (
        [item["symbol"] for item in descriptions],
        [item["symbol_latex"] for item in descriptions],
    )


def _build_gspg_payload(
    ssg: SpinSpaceGroup,
    *,
    real_space_setting: str,
    spin_frame_setting: str,
    spin_analysis_transform: np.ndarray | None = None,
) -> dict:
    presented_ops, raw_ops = _gspg_public_operation_sets(ssg)
    output_mode = (
        "reduced_point_part_with_spin_only_annotation"
        if len(presented_ops) != len(raw_ops)
        else "explicit_ops"
    )

    spin_only_ops = [
        [np.asarray(rotation, dtype=float), np.eye(3)]
        for rotation in ssg.gspg_spin_only_ops
    ]
    public_gspg = None
    try:
        public_gspg = ssg.gspg
        spin_only_symbol = ssg.gspg_spin_only_symbol
        collinear_axis = public_gspg.collinear_axis
        empg_symbol = public_gspg.empg_symbol
    except ValueError as exc:
        if spin_analysis_transform is None or "closure exceeded limit" not in str(exc):
            raise
        analysis_transform = np.asarray(spin_analysis_transform, dtype=float)
        analysis_ssg = ssg.transform_spin(analysis_transform)
        spin_only_symbol = analysis_ssg.gspg_spin_only_symbol
        analysis_gspg = analysis_ssg.gspg
        empg_symbol = analysis_gspg.empg_symbol
        if analysis_gspg.collinear_axis is None:
            collinear_axis = None
        else:
            analysis_to_public = np.linalg.inv(analysis_transform)
            collinear_axis = _normalize_spin_only_direction(
                analysis_to_public @ np.asarray(analysis_gspg.collinear_axis, dtype=float)
            )

    point_part_linear = ssg.international_symbol.get("point_part_linear", "")
    point_part_latex = ssg.international_symbol.get("point_part_latex", "")
    symbol_linear = (
        f"{point_part_linear} {spin_only_symbol['linear']}".strip()
        if point_part_linear
        else spin_only_symbol["linear"]
    )
    symbol_latex = (
        f"{point_part_latex}{spin_only_symbol['latex']}"
        if point_part_latex
        else spin_only_symbol["latex"]
    )
    npg_symbol_s = ssg.n_spin_part_point_group_symbol_s
    spin_only_component_symbol_s = spin_only_symbol["s"]
    if ssg.conf == "Noncoplanar" and spin_only_component_symbol_s != "C1":
        symbol_mode = "point_part_and_spin_only"
        tentative_symbol_s = None
    else:
        symbol_mode = "npg_x_spin_only"
        tentative_symbol_s = (
            npg_symbol_s
            if spin_only_component_symbol_s in {"", "C1"}
            else f"{npg_symbol_s} x {spin_only_component_symbol_s}"
        )

    presented_xyz_uvw = _serialize_gspg_xyz_uvw_ops(presented_ops, tol=ssg.tol)
    raw_xyz_uvw = _serialize_gspg_xyz_uvw_ops(raw_ops, tol=ssg.tol)
    generator_source_ops = [
        op for op in presented_ops
        if not _gspg_pair_is_spin_only(op, tol=ssg.tol)
    ]
    generator_ops = _select_gspg_generator_ops(generator_source_ops, tol=ssg.tol)
    generator_indices = _gspg_operation_indices_from_pairs(
        presented_ops,
        generator_ops,
        tol=ssg.tol,
    )
    generator_xyz_uvw = _serialize_gspg_xyz_uvw_ops(generator_ops, tol=ssg.tol)
    spin_only_xyz_uvw = _serialize_gspg_xyz_uvw_ops(spin_only_ops, tol=ssg.tol)
    text_payload = _build_gspg_text(
        symbol_linear=symbol_linear,
        spin_space_point_group_symbol_hm=ssg.spin_part_point_group_symbol_hm,
        spin_space_point_group_symbol_s=ssg.spin_part_point_group_symbol_s,
        effective_mpg_symbol=empg_symbol,
        real_space_setting=real_space_setting,
        spin_frame_setting=spin_frame_setting,
        generator_rows=_gspg_text_rows_for_indices(presented_xyz_uvw, generator_indices),
        operation_rows=_format_gspg_xyz_uvw_text(presented_xyz_uvw),
        spin_only_rows=_format_gspg_spin_only_text(
            conf=ssg.conf,
            spin_only_xyz_uvw=spin_only_xyz_uvw,
            collinear_axis=collinear_axis,
            tol=ssg.tol,
        ),
    )

    return {
        "gspg_ops": _serialize_gspg_ops(presented_ops),
        "gspg_raw_ops": _serialize_gspg_ops(raw_ops),
        "gspg_ops_xyz_uvw": presented_xyz_uvw,
        "gspg_raw_ops_xyz_uvw": raw_xyz_uvw,
        "gspg_generator_indices": generator_indices,
        "gspg_generator_ops": _serialize_gspg_ops(generator_ops),
        "gspg_generator_ops_xyz_uvw": generator_xyz_uvw,
        "gspg_spin_only_ops": _serialize_gspg_ops(spin_only_ops),
        "gspg_spin_only_ops_xyz_uvw": spin_only_xyz_uvw,
        "gspg_text": text_payload,
        "gspg_collinear_axis": (
            None if collinear_axis is None else np.asarray(collinear_axis, dtype=float).tolist()
        ),
        "gspg_symbol_linear": symbol_linear,
        "gspg_symbol_latex": symbol_latex,
        "gspg_effective_mpg_symbol": empg_symbol,
        "gspg_npg_symbol_s": npg_symbol_s,
        "gspg_output_mode": output_mode,
        "gspg_point_part_linear": point_part_linear,
        "gspg_real_space_setting": real_space_setting,
        "gspg_spin_frame_setting": spin_frame_setting,
        "gspg_spin_only_component_symbol_s": spin_only_component_symbol_s,
        "gspg_spin_only_part_linear": spin_only_symbol["linear"],
        "gspg_symbol_mode": symbol_mode,
        "gspg_tentative_symbol_s": tentative_symbol_s,
    }


def _spin_only_component_symbols(ssg: SpinSpaceGroup) -> tuple[str, str]:
    if ssg.conf == "Collinear":
        if len(ssg.sog) == 4:
            return "∞m", "C∞v"
        if len(ssg.sog) == 8:
            return "∞/mm", "D∞h"
        raise ValueError("Collinear spin-only symbol identification error")

    info = _resolve_point_group_info(
        [np.asarray(op[0], dtype=float) for op in ssg.sog],
        tol=max(float(ssg.tol), 1e-6),
        label="spin-only component point group",
    )
    return info[0], info[4]


def _compose_setting_transform(
    source_matrix: np.ndarray,
    source_shift: np.ndarray,
    target_matrix: np.ndarray,
    target_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = target_matrix @ np.linalg.inv(source_matrix)
    shift = normalize_vector_to_zero(
        target_shift - target_matrix @ np.linalg.inv(source_matrix) @ source_shift,
        atol=1e-10,
    )
    return transform, shift


def _chain_setting_transform(
    first_matrix: np.ndarray,
    first_shift: np.ndarray,
    second_matrix: np.ndarray,
    second_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(second_matrix, dtype=float) @ np.asarray(first_matrix, dtype=float)
    shift = normalize_vector_to_zero(
        np.asarray(second_matrix, dtype=float) @ np.asarray(first_shift, dtype=float)
        + np.asarray(second_shift, dtype=float),
        atol=1e-10,
    )
    return transform, shift


def _invert_setting_transform(
    transform: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform_inv = np.linalg.inv(transform)
    shift_inv = normalize_vector_to_zero(-transform_inv @ shift, atol=1e-10)
    return transform_inv, shift_inv


def _space_op_mod_integer_key(rotation, translation, *, tol: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    decimals = _real_op_bucket_decimals(tol)
    rotation_key = tuple(np.round(np.asarray(rotation, dtype=float).reshape(-1), decimals))
    wrapped_translation = normalize_vector_to_zero(
        np.mod(np.asarray(translation, dtype=float), 1.0),
        atol=max(tol, 1e-8),
    )
    translation_key = tuple(np.round(wrapped_translation.reshape(-1), decimals))
    return rotation_key, translation_key


def _unique_space_op_mod_integer_count(ops, *, tol: float) -> int:
    return len(
        {
            _space_op_mod_integer_key(op[1], op[2], tol=tol)
            for op in ops
        }
    )


def _unique_G0_space_op_mod_integer_count(ssg: SpinSpaceGroup, *, tol: float) -> int:
    return len(
        {
            _space_op_mod_integer_key(rotation, translation, tol=tol)
            for rotation, translation in ssg.G0_ops
        }
    )


def _select_G0std_axis_collapse(
    ssg_primitive: SpinSpaceGroup,
    G0std_ssg: SpinSpaceGroup,
    *,
    identify_index_details: dict | None,
    tol: float,
) -> tuple[np.ndarray, dict | None]:
    if not identify_index_details:
        return np.eye(3), None

    cell_size = identify_index_details.get("identify_cell_size")
    if cell_size is None:
        return np.eye(3), None
    try:
        cell_size = int(cell_size)
    except (TypeError, ValueError):
        return np.eye(3), None
    if cell_size <= 0:
        return np.eye(3), None
    if int(ssg_primitive.G0_num) > 15:
        return np.eye(3), None

    expected_unique_space_count = (
        _unique_G0_space_op_mod_integer_count(ssg_primitive, tol=tol) * cell_size
    )
    current_unique_space_count = _unique_space_op_mod_integer_count(G0std_ssg.ops, tol=tol)
    basis_fix = (
        np.asarray(ssg_primitive.transformation_to_G0std_id, dtype=float)
        @ np.linalg.inv(np.asarray(ssg_primitive.transformation_to_G0std, dtype=float))
    )
    basis_fix_is_diagonal = np.allclose(
        basis_fix,
        np.diag(np.diag(basis_fix)),
        atol=max(tol, 1e-8),
    )
    basis_fix_diag = np.rint(np.diag(basis_fix)).astype(int) if basis_fix_is_diagonal else None
    basis_fix_requests_x_collapse = (
        basis_fix_diag is not None
        and basis_fix_diag[0] == 2
        and basis_fix_diag[2] == 2
        and current_unique_space_count % 2 == 0
    )
    if (
        current_unique_space_count <= expected_unique_space_count
        and not basis_fix_requests_x_collapse
    ):
        return np.eye(3), None

    axis_candidates = [
        ("x", np.diag([2.0, 1.0, 1.0])),
        ("y", np.diag([1.0, 2.0, 1.0])),
        ("z", np.diag([1.0, 1.0, 2.0])),
    ]
    for axis, collapse_matrix in axis_candidates:
        collapsed_ssg = G0std_ssg.transform(collapse_matrix, np.zeros(3))
        collapsed_unique_space_count = _unique_space_op_mod_integer_count(
            collapsed_ssg.ops,
            tol=tol,
        )
        if collapsed_unique_space_count == expected_unique_space_count:
            return collapse_matrix, {
                "strategy": "axis_collapse",
                "axis": axis,
                "cell_size": cell_size,
                "expected_unique_space_op_count": expected_unique_space_count,
                "current_unique_space_op_count": current_unique_space_count,
                "collapsed_unique_space_op_count": collapsed_unique_space_count,
                "collapse_matrix": collapse_matrix.tolist(),
            }
        if (
            axis == "x"
            and basis_fix_requests_x_collapse
            and collapsed_unique_space_count * 2 == current_unique_space_count
            and collapsed_unique_space_count
            >= _unique_G0_space_op_mod_integer_count(ssg_primitive, tol=tol)
        ):
            return collapse_matrix, {
                "strategy": "axis_collapse",
                "axis": axis,
                "cell_size": cell_size,
                "expected_unique_space_op_count": expected_unique_space_count,
                "current_unique_space_op_count": current_unique_space_count,
                "collapsed_unique_space_op_count": collapsed_unique_space_count,
                "collapse_matrix": collapse_matrix.tolist(),
                "basis_fix_before_collapse": basis_fix.tolist(),
                "basis_fix_rule": "x_and_z_doubled",
            }

    return np.eye(3), None


def _acc_aligned_convention_to_primitive_transform(index: str) -> tuple[np.ndarray, np.ndarray]:
    basis_p = np.asarray(
        [
            [float(value) for value in row]
            for row in get_acc_aligned_conventional_to_primitive_p(index)
        ],
        dtype=float,
    )
    # The generated index stores the direct-space basis relation
    # (a_acc,b_acc,c_acc)=(a_conv,b_conv,c_conv)P.  Internal setting
    # transforms are coordinate transforms x_target=A x_source + o, so
    # basis_source=basis_target A and A=P^{-1}.
    return np.linalg.inv(basis_p), np.zeros(3)


def _setting_transform_signature(
    transform: tuple[np.ndarray, np.ndarray],
    *,
    tol: float = 1e-10,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    matrix = np.asarray(transform[0], dtype=float)
    shift = normalize_vector_to_zero(np.asarray(transform[1], dtype=float), atol=tol)
    scale = 1.0 / max(tol, 1e-12)
    return (
        tuple(np.rint(matrix.reshape(-1) * scale).astype(np.int64)),
        tuple(np.rint(shift.reshape(-1) * scale).astype(np.int64)),
    )


def _append_unique_setting_transform_candidate(
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]],
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    name: str,
    transform: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> None:
    normalized = (
        np.asarray(transform[0], dtype=float),
        normalize_vector_to_zero(np.asarray(transform[1], dtype=float), atol=1e-10),
    )
    signature = _setting_transform_signature(normalized, tol=tol)
    if signature in seen:
        return
    seen.add(signature)
    candidates.append((name, normalized))


def _integer_row_gcd(row: np.ndarray, *, tol: float) -> int | None:
    rounded = np.rint(np.asarray(row, dtype=float)).astype(int)
    if not np.allclose(row, rounded, atol=tol):
        return None
    gcd_value = 0
    for value in rounded:
        gcd_value = int(np.gcd(gcd_value, abs(int(value))))
    return gcd_value


def _append_nofrac_lattice_shear_candidates(
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]],
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    ssg_primitive: SpinSpaceGroup,
    raw_transformation_primitive_to_G0std: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> None:
    """Generate no-fraction candidates inside the current integerized lattice.

    ``integerize_matrix`` fixes a no-fraction sublattice by clearing the
    denominators of the spglib standard transform.  The old greedy choice also
    fixed a particular basis of that sublattice, which can point the doubled
    conventional axis at the wrong primitive representative.  Here we keep the
    same sublattice and only shear rows whose integerized generator has a common
    factor, e.g. ``c' = c - 2a`` for the monoclinic C cases.
    """
    transformation_to_G0std_id = np.asarray(
        ssg_primitive.transformation_to_G0std_id,
        dtype=float,
    )
    origin_shift_to_G0std_id = np.asarray(
        ssg_primitive.origin_shift_to_G0std_id,
        dtype=float,
    )
    raw_matrix = np.asarray(raw_transformation_primitive_to_G0std[0], dtype=float)
    try:
        raw_basis_fix = transformation_to_G0std_id @ np.linalg.inv(raw_matrix)
    except np.linalg.LinAlgError:
        return

    raw_basis_fix_integer = np.rint(raw_basis_fix).astype(int)
    if not np.allclose(raw_basis_fix, raw_basis_fix_integer, atol=tol):
        return

    transformation_to_G0std_id_inv = np.linalg.inv(transformation_to_G0std_id)
    origin_seed = transformation_to_G0std_id_inv @ origin_shift_to_G0std_id
    multiplier_order = (-1, 1, -2, 2)
    for target_row in range(3):
        row_gcd = _integer_row_gcd(raw_basis_fix_integer[target_row], tol=tol)
        if row_gcd is None or row_gcd <= 1:
            continue
        for source_row in range(3):
            if source_row == target_row:
                continue
            for multiplier_factor in multiplier_order:
                multiplier = int(multiplier_factor * row_gcd)
                shear = np.eye(3, dtype=int)
                shear[target_row, source_row] = multiplier
                sheared_basis_fix = shear @ raw_basis_fix_integer
                candidate_integer_basis = (
                    transformation_to_G0std_id_inv @ sheared_basis_fix
                )
                candidate_integer_basis_rounded = np.rint(candidate_integer_basis).astype(int)
                if not np.allclose(
                    candidate_integer_basis,
                    candidate_integer_basis_rounded,
                    atol=tol,
                ):
                    continue
                try:
                    candidate_matrix = np.linalg.inv(
                        candidate_integer_basis_rounded.astype(float)
                    )
                except np.linalg.LinAlgError:
                    continue
                candidate_origin_shift = normalize_vector_to_zero(
                    candidate_matrix @ origin_seed,
                    atol=1e-10,
                )
                _append_unique_setting_transform_candidate(
                    candidates,
                    seen,
                    (
                        "nofrac_lattice_shear:"
                        f"r{target_row}+=({multiplier})r{source_row}"
                    ),
                    (candidate_matrix, candidate_origin_shift),
                    tol=tol,
                )


def _identify_space_group_setting_transform(
    identify_index_details: dict | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not identify_index_details:
        return None
    transform = identify_index_details.get("space_group_transformation")
    if transform is None:
        return None
    if len(transform) != 2:
        return None
    return np.asarray(transform[0], dtype=float), np.asarray(transform[1], dtype=float)


def _identify_index_setting_transform(
    identify_index_details: dict | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not identify_index_details:
        return None
    transform = identify_index_details.get("transformation_matrix")
    if transform is None:
        return None
    if len(transform) != 2:
        return None
    return np.asarray(transform[0], dtype=float), np.asarray(transform[1], dtype=float)


def _selected_standard_setting_from_identify_index_details(
    identify_index_details: dict | None,
) -> str:
    if identify_index_details is None:
        raise ValueError(
            "Cannot select G0std/L0std without identify-index details. "
            "The standard setting is determined from t_index/k_index, not "
            "from the printable index suffix."
        )
    try:
        it = int(identify_index_details["t_index"])
        ik = int(identify_index_details["k_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot select G0std/L0std because identify-index details do not "
            "contain integer t_index/k_index."
        ) from exc

    # k-type SSGs (it=1, ik>1) are naturally represented in L0std.
    # t-type (it>1, ik=1), g-type (it>1, ik>1), and trivial cases use G0std.
    # The suffix in the public index, e.g. `.L`, describes the spin
    # configuration branch and is not the real-space standard-cell type.
    if it == 1 and ik > 1:
        return L0_STANDARD_SETTING
    return G0_STANDARD_SETTING


def _append_identify_setting_transform_candidates(
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]],
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    base_transform: tuple[np.ndarray, np.ndarray],
    identify_index_details: dict | None,
    *,
    tol: float,
) -> None:
    transform = _identify_space_group_setting_transform(identify_index_details)
    if transform is None:
        return

    # `space_group_transformation` is the identify-index database-gauge
    # equivalence written from the database representative into the current
    # no-fraction generator gauge.  The ACC P table is defined in the database
    # convention, so the cell/SSG chain must first move current -> database.
    transform_current_to_database = _invert_setting_transform(transform[0], transform[1])
    _append_unique_setting_transform_candidate(
        candidates,
        seen,
        "identify_space_transform_current_to_database_after_current",
        _chain_setting_transform(
            base_transform[0],
            base_transform[1],
            transform_current_to_database[0],
            transform_current_to_database[1],
        ),
        tol=tol,
    )


def _reciprocal_integer_determinant_factor(
    matrix: np.ndarray,
    *,
    tol: float,
) -> int | None:
    determinant = abs(float(np.linalg.det(np.asarray(matrix, dtype=float))))
    determinant_tol = max(tol, 1e-8)
    if np.isclose(determinant, 1.0, atol=determinant_tol, rtol=0):
        return None
    if determinant <= determinant_tol or determinant > 1.0:
        return None
    reciprocal = 1.0 / determinant
    factor = int(round(reciprocal))
    if factor <= 1:
        return None
    if not np.isclose(determinant, 1.0 / factor, atol=determinant_tol, rtol=0):
        return None
    return factor


@lru_cache(maxsize=8)
def _fixed_b_ac_unimodular_column_transforms(
    max_entry: int,
) -> tuple[tuple[tuple[int, int, int, int], tuple[tuple[int, ...], ...]], ...]:
    transforms: list[
        tuple[tuple[int, int, int, int], tuple[tuple[int, ...], ...]]
    ] = []
    for p, q, r, s in product(range(-max_entry, max_entry + 1), repeat=4):
        if p * s - q * r != 1:
            continue
        matrix = (
            (p, 0, q),
            (0, 1, 0),
            (r, 0, s),
        )
        values = (p, q, r, s)
        transforms.append((values, matrix))

    def sort_key(
        item: tuple[tuple[int, int, int, int], tuple[tuple[int, ...], ...]]
    ) -> tuple[int, int, int, tuple[int, int, int, int]]:
        values, matrix = item
        identity_rank = 0 if matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) else 1
        return (
            identity_rank,
            max(abs(value) for value in values),
            sum(abs(value) for value in values),
            values,
        )

    return tuple(sorted(transforms, key=sort_key))


@lru_cache(maxsize=4)
def _unrestricted_unimodular_column_transforms(
    max_entry: int,
) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
    transforms: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for values in product(range(-max_entry, max_entry + 1), repeat=9):
        matrix = np.asarray(values, dtype=int).reshape(3, 3)
        determinant = round(float(np.linalg.det(matrix)))
        if determinant != 1 or not np.isclose(np.linalg.det(matrix), 1.0):
            continue
        transforms.append((values, tuple(tuple(int(v) for v in row) for row in matrix)))

    def sort_key(
        item: tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]
    ) -> tuple[int, int, int, tuple[int, ...]]:
        values, matrix = item
        identity_rank = 0 if matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) else 1
        return (
            identity_rank,
            max(abs(value) for value in values),
            sum(abs(value) for value in values),
            values,
        )

    return tuple(sorted(transforms, key=sort_key))


def _append_monoclinic_ac_column_reduction_candidates(
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]],
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    standard_setting: str,
    ssg_primitive: SpinSpaceGroup,
    base_transform: tuple[np.ndarray, np.ndarray],
    identify_info: str,
    identify_index_details: dict | None,
    *,
    tol: float,
) -> None:
    """Reduce a doubled monoclinic a/c column after identify setting alignment.

    For monoclinic G0 (ITA 3..15), the database convention fixes the unique
    b axis but leaves a/c representatives related by integer unimodular column
    changes.  If the selected convention-to-ACC P matrix and the current
    primitive-to-convention transform compose to an index-n primitive cell,
    search only those fixed-b a/c changes for a column divisible by n.  The
    resulting candidates still go through the normal paired cell+SSG+P
    validation; this is not a legacy fallback source.
    """
    try:
        G0_num = int(ssg_primitive.G0_num)
    except (TypeError, ValueError):
        return
    if not 3 <= G0_num <= 15:
        return

    transform_database_to_current = _identify_space_group_setting_transform(
        identify_index_details
    )
    if transform_database_to_current is None:
        return

    try:
        transform_current_to_database = _invert_setting_transform(
            transform_database_to_current[0],
            transform_database_to_current[1],
        )
        transform_primitive_to_database = _chain_setting_transform(
            base_transform[0],
            base_transform[1],
            transform_current_to_database[0],
            transform_current_to_database[1],
        )
        transform_database_to_acc = _acc_aligned_convention_to_primitive_transform(
            identify_info
        )
    except (KeyError, np.linalg.LinAlgError, ValueError):
        return

    determinant_factor = _reciprocal_integer_determinant_factor(
        transform_database_to_acc[0] @ transform_primitive_to_database[0],
        tol=tol,
    )
    if determinant_factor is None:
        return

    try:
        basis_primitive_to_database = np.linalg.inv(
            np.asarray(transform_primitive_to_database[0], dtype=float)
        )
    except np.linalg.LinAlgError:
        return
    basis_integer = np.rint(basis_primitive_to_database).astype(int)
    if not np.allclose(basis_primitive_to_database, basis_integer, atol=max(tol, 1e-8)):
        return

    origin_database = np.asarray(transform_primitive_to_database[1], dtype=float)
    generated_count = 0
    max_generated_candidates = 64
    for values, matrix_tuple in _fixed_b_ac_unimodular_column_transforms(8):
        ac_change = np.asarray(matrix_tuple, dtype=int)
        changed_basis = basis_integer @ ac_change
        for column_index in (0, 2):
            if np.any(changed_basis[:, column_index] % determinant_factor != 0):
                continue
            reduced_basis = changed_basis.astype(float)
            reduced_basis[:, column_index] /= float(determinant_factor)
            try:
                candidate_matrix = np.linalg.inv(reduced_basis)
            except np.linalg.LinAlgError:
                continue
            if not np.isclose(
                abs(float(np.linalg.det(transform_database_to_acc[0] @ candidate_matrix))),
                1.0,
                atol=max(tol, 1e-8),
                rtol=0,
            ):
                continue
            candidate_origin_shift = normalize_vector_to_zero(
                candidate_matrix @ basis_integer.astype(float) @ origin_database,
                atol=1e-10,
            )
            before_count = len(candidates)
            _append_unique_setting_transform_candidate(
                candidates,
                seen,
                (
                    "monoclinic_ac_column_reduce:"
                    f"setting={standard_setting};"
                    f"det_factor={determinant_factor};"
                    f"col={column_index};"
                    f"U_ac={values}"
                ),
                (candidate_matrix, candidate_origin_shift),
                tol=tol,
            )
            if len(candidates) > before_count:
                generated_count += 1
                if generated_count >= max_generated_candidates:
                    return


def _append_triclinic_column_reduction_candidates(
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]],
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    standard_setting: str,
    ssg_primitive: SpinSpaceGroup,
    base_transform: tuple[np.ndarray, np.ndarray],
    identify_info: str,
    identify_index_details: dict | None,
    *,
    tol: float,
) -> None:
    """Reduce a triclinic no-fraction supercell with unrestricted GL(3,Z).

    Triclinic G0 has no unique axis convention to preserve.  When the selected
    convention-to-ACC P matrix and the primitive-to-convention transform compose
    to an index-n supercell, search full orientation-preserving unimodular
    column changes for a reducible column.  The resulting candidates are still
    validated later as paired cell+SSG transforms.
    """
    try:
        G0_num = int(ssg_primitive.G0_num)
    except (TypeError, ValueError):
        return
    if G0_num not in {1, 2}:
        return

    transform_database_to_current = _identify_space_group_setting_transform(
        identify_index_details
    )
    if transform_database_to_current is None:
        return

    try:
        transform_current_to_database = _invert_setting_transform(
            transform_database_to_current[0],
            transform_database_to_current[1],
        )
        transform_primitive_to_database = _chain_setting_transform(
            base_transform[0],
            base_transform[1],
            transform_current_to_database[0],
            transform_current_to_database[1],
        )
        transform_database_to_acc = _acc_aligned_convention_to_primitive_transform(
            identify_info
        )
    except (KeyError, np.linalg.LinAlgError, ValueError):
        return

    determinant_factor = _reciprocal_integer_determinant_factor(
        transform_database_to_acc[0] @ transform_primitive_to_database[0],
        tol=tol,
    )
    if determinant_factor is None:
        return

    try:
        basis_primitive_to_database = np.linalg.inv(
            np.asarray(transform_primitive_to_database[0], dtype=float)
        )
    except np.linalg.LinAlgError:
        return
    basis_integer = np.rint(basis_primitive_to_database).astype(int)
    if not np.allclose(basis_primitive_to_database, basis_integer, atol=max(tol, 1e-8)):
        return

    origin_database = np.asarray(transform_primitive_to_database[1], dtype=float)
    raw_candidates: list[
        tuple[
            tuple[int, int, int, int, tuple[int, ...], int],
            tuple[str, tuple[np.ndarray, np.ndarray]],
        ]
    ] = []
    for values, matrix_tuple in _unrestricted_unimodular_column_transforms(2):
        column_change = np.asarray(matrix_tuple, dtype=int)
        changed_basis = basis_integer @ column_change
        for column_index in range(3):
            if np.any(changed_basis[:, column_index] % determinant_factor != 0):
                continue
            reduced_basis = changed_basis.astype(float)
            reduced_basis[:, column_index] /= float(determinant_factor)
            determinant = abs(float(np.linalg.det(reduced_basis)))
            if not np.isclose(determinant, 1.0, atol=max(tol, 1e-8), rtol=0):
                continue
            try:
                candidate_matrix = np.linalg.inv(reduced_basis)
            except np.linalg.LinAlgError:
                continue
            if not np.isclose(
                abs(float(np.linalg.det(transform_database_to_acc[0] @ candidate_matrix))),
                1.0,
                atol=max(tol, 1e-8),
                rtol=0,
            ):
                continue
            candidate_origin_shift = normalize_vector_to_zero(
                candidate_matrix @ basis_integer.astype(float) @ origin_database,
                atol=1e-10,
            )
            reduced_basis_integer = np.rint(reduced_basis).astype(int)
            candidate_matrix_integer = np.rint(candidate_matrix).astype(int)
            raw_candidates.append(
                (
                    (
                        int(np.max(np.abs(reduced_basis_integer))),
                        int(np.sum(np.abs(reduced_basis_integer))),
                        int(np.max(np.abs(candidate_matrix_integer))),
                        int(np.sum(np.abs(candidate_matrix_integer))),
                        values,
                        column_index,
                    ),
                    (
                        (
                            "triclinic_column_reduce:"
                            f"setting={standard_setting};"
                            f"det_factor={determinant_factor};"
                            f"col={column_index};"
                            f"U={values}"
                        ),
                        (candidate_matrix, candidate_origin_shift),
                    ),
                )
            )

    generated_count = 0
    max_generated_candidates = 64
    for _, (name, transform) in sorted(raw_candidates, key=lambda item: item[0]):
        before_count = len(candidates)
        _append_unique_setting_transform_candidate(
            candidates,
            seen,
            name,
            transform,
            tol=tol,
        )
        if len(candidates) > before_count:
            generated_count += 1
            if generated_count >= max_generated_candidates:
                return


def _signed_permutation_matrices() -> list[tuple[str, np.ndarray]]:
    matrices: list[tuple[str, np.ndarray]] = []
    identity = np.eye(3)
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for row, source_row in enumerate(permutation):
                matrix[row, source_row] = float(signs[row])
            if np.allclose(matrix, identity):
                continue
            if np.linalg.det(matrix) < 0:
                continue
            name = (
                "signed_permutation:"
                f"rows={','.join(str(item) for item in permutation)};"
                f"signs={','.join(str(item) for item in signs)}"
            )
            matrices.append((name, matrix))
    return matrices


def _append_signed_permutation_setting_candidates(
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]],
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    base_transform: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> None:
    base_matrix = np.asarray(base_transform[0], dtype=float)
    base_shift = np.asarray(base_transform[1], dtype=float)
    for name, permutation_matrix in _signed_permutation_matrices():
        _append_unique_setting_transform_candidate(
            candidates,
            seen,
            name,
            (
                permutation_matrix @ base_matrix,
                normalize_vector_to_zero(permutation_matrix @ base_shift, atol=1e-10),
            ),
            tol=tol,
        )


def _build_standard_transform_candidates(
    standard_setting: str,
    ssg_primitive: SpinSpaceGroup,
    raw_transform: tuple[np.ndarray, np.ndarray],
    identify_info: str,
    identify_index_details: dict | None,
    *,
    tol: float,
) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    _append_unique_setting_transform_candidate(
        candidates,
        seen,
        "current_integerized",
        raw_transform,
        tol=tol,
    )
    if standard_setting == G0_STANDARD_SETTING:
        _append_nofrac_lattice_shear_candidates(
            candidates,
            seen,
            ssg_primitive,
            raw_transform,
            tol=tol,
        )
        _append_unique_setting_transform_candidate(
            candidates,
            seen,
            "spglib_id",
            (
                np.asarray(ssg_primitive.transformation_to_G0std_id, dtype=float),
                np.asarray(ssg_primitive.origin_shift_to_G0std_id, dtype=float),
            ),
            tol=tol,
        )
    _append_identify_setting_transform_candidates(
        candidates,
        seen,
        raw_transform,
        identify_index_details,
        tol=tol,
    )
    _append_monoclinic_ac_column_reduction_candidates(
        candidates,
        seen,
        standard_setting,
        ssg_primitive,
        raw_transform,
        identify_info,
        identify_index_details,
        tol=tol,
    )
    _append_triclinic_column_reduction_candidates(
        candidates,
        seen,
        standard_setting,
        ssg_primitive,
        raw_transform,
        identify_info,
        identify_index_details,
        tol=tol,
    )
    _append_signed_permutation_setting_candidates(
        candidates,
        seen,
        raw_transform,
        tol=tol,
    )
    return candidates


def _try_select_one_standard_transform_for_acc_alignment(
    standard_setting: str,
    ssg_primitive: SpinSpaceGroup,
    magnetic_primitive_cell: CrystalCell,
    raw_transformation_primitive_to_standard: tuple[np.ndarray, np.ndarray],
    legacy_transformation_primitive_to_acc_primitive: tuple[np.ndarray, np.ndarray],
    legacy_acc_primitive_cell: CrystalCell,
    *,
    identify_info: str,
    identify_index_details: dict | None,
    tol: Tolerances,
) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """
    Choose a standard transform as a cell/SSG setting transform, not only as an
    operation-list basis change.

    The identify-index P matrix is defined between the database conventional
    setting and its ACC primitive setting.  Therefore a valid standard candidate
    must compose with that same P to reproduce the ACC primitive cell directly
    from the magnetic primitive SSG/cell pair.
    """
    transformation_standard_to_acc_primitive = (
        _acc_aligned_convention_to_primitive_transform(identify_info)
    )

    candidate_tol = max(tol.m_matrix_tol, 1e-10)
    candidates = _build_standard_transform_candidates(
        standard_setting,
        ssg_primitive,
        raw_transformation_primitive_to_standard,
        identify_info,
        identify_index_details,
        tol=candidate_tol,
    )

    rejected: list[dict] = []
    for name, candidate in candidates:
        try:
            transformation_primitive_to_acc_primitive = _chain_setting_transform(
                candidate[0],
                candidate[1],
                transformation_standard_to_acc_primitive[0],
                transformation_standard_to_acc_primitive[1],
            )
            candidate_acc_primitive_lattice = _setting_transform_lattice_matrix(
                magnetic_primitive_cell.lattice_matrix,
                transformation_primitive_to_acc_primitive[0],
            )
            _assert_acc_primitive_lattice_matrix_matches_magnetic_primitive(
                magnetic_primitive_cell.lattice_matrix,
                candidate_acc_primitive_lattice,
                tol=tol,
                label=f"{identify_info}:{name}",
            )
            candidate_acc_primitive_cell = magnetic_primitive_cell.transform(
                *transformation_primitive_to_acc_primitive
            )
            transformation_legacy_acc_to_candidate_acc = _compose_setting_transform(
                legacy_transformation_primitive_to_acc_primitive[0],
                legacy_transformation_primitive_to_acc_primitive[1],
                transformation_primitive_to_acc_primitive[0],
                transformation_primitive_to_acc_primitive[1],
            )
            _assert_acc_primitive_cells_equivalent(
                legacy_acc_primitive_cell,
                candidate_acc_primitive_cell,
                transformation_legacy_acc_to_candidate_acc,
                tol=tol,
                label=f"{identify_info}:{name}",
            )
            candidate_standard_ssg = ssg_primitive.transform(*candidate)
            _ = candidate_standard_ssg.international_symbol_type
            candidate_acc_primitive_ssg = ssg_primitive.transform(
                *transformation_primitive_to_acc_primitive
            )
            _ = candidate_acc_primitive_ssg.international_symbol_type
        except Exception as exc:
            rejected.append(
                {
                    "strategy": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "matrix": np.asarray(candidate[0], dtype=float).tolist(),
                    "origin_shift": np.asarray(candidate[1], dtype=float).tolist(),
                }
            )
            continue

        return candidate, {
            "strategy": "acc_aligned_p_candidate_selection",
            "standard_setting": standard_setting,
            "selected_strategy": name,
            "selected_matrix": np.asarray(candidate[0], dtype=float).tolist(),
            "selected_origin_shift": np.asarray(candidate[1], dtype=float).tolist(),
            "rejected_candidates": rejected,
        }

    raise ValueError(
        f"No non-fallback {standard_setting} transform candidate composes with the "
        "identify-index P matrix for "
        f"{identify_info}. Legacy ACC-primitive-derived fallback candidates "
        "are intentionally disabled so the wrong standard-cell matrix is "
        "exposed instead of hidden. rejected_candidates="
        f"{json.dumps(rejected, cls=NumpyEncoder, sort_keys=True)}"
    )


def _select_standard_transform_for_acc_alignment(
    ssg_primitive: SpinSpaceGroup,
    magnetic_primitive_cell: CrystalCell,
    raw_transform_by_standard: dict[str, tuple[np.ndarray, np.ndarray]],
    legacy_transformation_primitive_to_acc_primitive: tuple[np.ndarray, np.ndarray],
    legacy_acc_primitive_cell: CrystalCell,
    *,
    identify_info: str,
    identify_index_details: dict | None,
    tol: Tolerances,
) -> tuple[str, tuple[np.ndarray, np.ndarray], dict]:
    if identify_index_details is None and identify_info is not None:
        raise KeyError(identify_info)
    standard_setting = _selected_standard_setting_from_identify_index_details(
        identify_index_details
    )
    try:
        candidate, audit = _try_select_one_standard_transform_for_acc_alignment(
            standard_setting,
            ssg_primitive,
            magnetic_primitive_cell,
            raw_transform_by_standard[standard_setting],
            legacy_transformation_primitive_to_acc_primitive,
            legacy_acc_primitive_cell,
            identify_info=identify_info,
            identify_index_details=identify_index_details,
            tol=tol,
        )
    except ValueError as exc:
        raise ValueError(
            f"No non-fallback {standard_setting} transform candidate composes "
            "with the identify-index P matrix for "
            f"{identify_info}. The standard setting was selected from "
            f"t_index={identify_index_details.get('t_index') if identify_index_details else None}, "
            f"k_index={identify_index_details.get('k_index') if identify_index_details else None}; "
            "cross-setting fallback is intentionally disabled."
        ) from exc

    audit["preferred_standard_setting"] = standard_setting
    audit["standard_setting_rule"] = "t_index/k_index"
    audit["rejected_standard_settings"] = []
    return standard_setting, candidate, audit


def _fractional_position_distance(left, right) -> float:
    diff = np.abs(np.mod(np.asarray(left, dtype=float) - np.asarray(right, dtype=float), 1.0))
    wrapped = np.minimum(diff, 1.0 - diff)
    return float(np.max(wrapped))


def _magnetic_cells_equivalent(
    left: CrystalCell,
    right: CrystalCell,
    *,
    space_tol: float,
    moment_tol: float,
    occupancy_tol: float,
    lattice_tol: float,
) -> bool:
    if len(left.positions) != len(right.positions):
        return False
    if not np.allclose(left.lattice_matrix, right.lattice_matrix, atol=lattice_tol, rtol=0):
        return False

    unmatched = set(range(len(right.positions)))
    for left_index, left_position in enumerate(left.positions):
        left_element = left.elements[left_index]
        left_occupancy = float(left.occupancies[left_index])
        left_moment = np.asarray(left.moments[left_index], dtype=float)
        matched_index = None
        for right_index in list(unmatched):
            if left_element != right.elements[right_index]:
                continue
            if abs(left_occupancy - float(right.occupancies[right_index])) > occupancy_tol:
                continue
            if _fractional_position_distance(left_position, right.positions[right_index]) > space_tol:
                continue
            if np.max(np.abs(left_moment - np.asarray(right.moments[right_index], dtype=float))) > moment_tol:
                continue
            matched_index = right_index
            break
        if matched_index is None:
            return False
        unmatched.remove(matched_index)
    return not unmatched


def _assert_acc_primitive_cells_equivalent(
    reference_cell: CrystalCell,
    candidate_cell: CrystalCell,
    reference_to_candidate: tuple[np.ndarray, np.ndarray],
    *,
    tol: Tolerances,
    label: str,
) -> None:
    transformed_reference = reference_cell.transform(
        np.asarray(reference_to_candidate[0], dtype=float),
        np.asarray(reference_to_candidate[1], dtype=float),
    )
    if _magnetic_cells_equivalent(
        transformed_reference,
        candidate_cell,
        space_tol=tol.space,
        moment_tol=tol.moment,
        occupancy_tol=tol.occupancy,
        lattice_tol=max(tol.space * 1e-3, 1e-6),
    ):
        return
    raise ValueError(
        f"ACC-aligned primitive validation failed for {label}: "
        "the convention-index P transform does not reproduce the legacy "
        "magnetic primitive structure under the derived setting change."
    )


def _setting_transform_lattice_matrix(
    source_lattice: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Return the lattice generated by ``CrystalCell.transform(matrix, shift)``."""

    return np.linalg.inv(np.asarray(matrix, dtype=float)).T @ np.asarray(source_lattice, dtype=float)


def _assert_acc_primitive_lattice_matrix_matches_magnetic_primitive(
    magnetic_primitive_lattice: np.ndarray,
    candidate_acc_primitive_lattice: np.ndarray,
    *,
    tol: Tolerances,
    label: str,
) -> None:
    """Require the ACC primitive output lattice to be the same primitive lattice.

    The identify-index P matrix is defined for the database conventional gauge.
    If it is applied in a merely equivalent but differently gauged convention
    setting, it can produce a same-volume cell that is not a unimodular basis
    change of the magnetic primitive lattice.  That cell is not a valid
    standalone magnetic primitive POSCAR even though the paired algebraic SSG
    transform exists.
    """

    reference_lattice = np.asarray(magnetic_primitive_lattice, dtype=float)
    candidate_lattice = np.asarray(candidate_acc_primitive_lattice, dtype=float)
    lattice_relation = candidate_lattice @ np.linalg.inv(reference_lattice)
    rounded_relation = np.rint(lattice_relation)
    relation_tol = max(tol.m_matrix_tol, tol.space * 1e-3, 1e-8)
    if not np.allclose(lattice_relation, rounded_relation, atol=relation_tol, rtol=0):
        raise ValueError(
            f"ACC primitive lattice validation failed for {label}: "
            "the identify-index P matrix produced a same-volume lattice that "
            "is not an integer basis change of the magnetic primitive lattice. "
            f"candidate_to_magnetic_primitive={lattice_relation.tolist()}."
        )

    determinant = int(round(float(np.linalg.det(rounded_relation))))
    if abs(determinant) != 1:
        raise ValueError(
            f"ACC primitive lattice validation failed for {label}: "
            "the candidate lattice is integer-related to the magnetic primitive "
            f"lattice but not unimodular, det={determinant}. "
            f"candidate_to_magnetic_primitive={rounded_relation.astype(int).tolist()}."
        )


def _assert_acc_primitive_lattice_matches_magnetic_primitive(
    magnetic_primitive_cell: CrystalCell,
    candidate_acc_primitive_cell: CrystalCell,
    *,
    tol: Tolerances,
    label: str,
) -> None:
    _assert_acc_primitive_lattice_matrix_matches_magnetic_primitive(
        magnetic_primitive_cell.lattice_matrix,
        candidate_acc_primitive_cell.lattice_matrix,
        tol=tol,
        label=label,
    )


def _resolve_acc_primitive_from_selected_standard(
    selected_standard_cell: CrystalCell,
    magnetic_primitive_cell: CrystalCell,
    ssg_primitive: SpinSpaceGroup,
    transformation_input_to_primitive: tuple[np.ndarray, np.ndarray],
    transformation_input_to_selected_standard: tuple[np.ndarray, np.ndarray],
    transformation_input_to_database_standard: tuple[np.ndarray, np.ndarray],
    legacy_acc_primitive_cell: CrystalCell,
    legacy_transformation_input_to_acc_primitive: tuple[np.ndarray, np.ndarray],
    *,
    identify_info: str,
    tol: Tolerances,
) -> tuple[
    CrystalCell,
    SpinSpaceGroup,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict,
]:
    transformation_database_standard_to_acc_primitive = (
        _acc_aligned_convention_to_primitive_transform(identify_info)
    )
    transformation_selected_standard_to_database_standard = _compose_setting_transform(
        transformation_input_to_selected_standard[0],
        transformation_input_to_selected_standard[1],
        transformation_input_to_database_standard[0],
        transformation_input_to_database_standard[1],
    )
    transformation_selected_standard_to_acc_primitive = _chain_setting_transform(
        transformation_selected_standard_to_database_standard[0],
        transformation_selected_standard_to_database_standard[1],
        transformation_database_standard_to_acc_primitive[0],
        transformation_database_standard_to_acc_primitive[1],
    )
    acc_primitive_cell = selected_standard_cell.transform(
        *transformation_selected_standard_to_acc_primitive
    )
    _assert_acc_primitive_lattice_matches_magnetic_primitive(
        magnetic_primitive_cell,
        acc_primitive_cell,
        tol=tol,
        label=identify_info,
    )
    transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_selected_standard[0],
        transformation_input_to_selected_standard[1],
        transformation_selected_standard_to_acc_primitive[0],
        transformation_selected_standard_to_acc_primitive[1],
    )
    transformation_primitive_to_acc_primitive = _compose_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    acc_primitive_ssg = ssg_primitive.transform(
        *transformation_primitive_to_acc_primitive
    )
    primitive_acc_primitive_cell = magnetic_primitive_cell.transform(
        *transformation_primitive_to_acc_primitive
    )
    if not _magnetic_cells_equivalent(
        primitive_acc_primitive_cell,
        acc_primitive_cell,
        space_tol=tol.space,
        moment_tol=tol.moment,
        occupancy_tol=tol.occupancy,
        lattice_tol=max(tol.space * 1e-3, 1e-6),
    ):
        raise ValueError(
            f"ACC primitive validation failed for {identify_info}: "
            "the final-convention cell path and the primitive-composed SSG path "
            "do not produce the same magnetic primitive cell."
        )
    # Force the lazy symbol/index invariants now. Invalid P-derived cells should
    # fail here instead of being hidden by a later serialization path.
    _ = acc_primitive_ssg.international_symbol_type
    transformation_legacy_acc_to_acc_primitive = _compose_setting_transform(
        legacy_transformation_input_to_acc_primitive[0],
        legacy_transformation_input_to_acc_primitive[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    _assert_acc_primitive_cells_equivalent(
        legacy_acc_primitive_cell,
        acc_primitive_cell,
        transformation_legacy_acc_to_acc_primitive,
        tol=tol,
        label=identify_info,
    )
    return (
        acc_primitive_cell,
        acc_primitive_ssg,
        transformation_input_to_acc_primitive,
        transformation_selected_standard_to_acc_primitive,
        {"strategy": "identify_index_p"},
    )


def _resolve_acc_primitive_from_primitive_standard_transform(
    magnetic_primitive_cell: CrystalCell,
    ssg_primitive: SpinSpaceGroup,
    transformation_input_to_primitive: tuple[np.ndarray, np.ndarray],
    transformation_input_to_selected_standard: tuple[np.ndarray, np.ndarray],
    legacy_acc_primitive_cell: CrystalCell,
    legacy_transformation_input_to_acc_primitive: tuple[np.ndarray, np.ndarray],
    *,
    identify_info: str,
    tol: Tolerances,
) -> tuple[
    CrystalCell,
    SpinSpaceGroup,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict,
]:
    transformation_selected_standard_to_acc_primitive = (
        _acc_aligned_convention_to_primitive_transform(identify_info)
    )
    transformation_primitive_to_selected_standard = _compose_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        transformation_input_to_selected_standard[0],
        transformation_input_to_selected_standard[1],
    )
    transformation_primitive_to_acc_primitive = _chain_setting_transform(
        transformation_primitive_to_selected_standard[0],
        transformation_primitive_to_selected_standard[1],
        transformation_selected_standard_to_acc_primitive[0],
        transformation_selected_standard_to_acc_primitive[1],
    )
    acc_primitive_cell = magnetic_primitive_cell.transform(
        *transformation_primitive_to_acc_primitive
    )
    acc_primitive_ssg = ssg_primitive.transform(
        *transformation_primitive_to_acc_primitive
    )
    # Force lazy symbol/index invariants now. Invalid P-derived transforms
    # should fail here instead of being hidden by a later serialization path.
    _ = acc_primitive_ssg.international_symbol_type
    transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        transformation_primitive_to_acc_primitive[0],
        transformation_primitive_to_acc_primitive[1],
    )
    transformation_legacy_acc_to_acc_primitive = _compose_setting_transform(
        legacy_transformation_input_to_acc_primitive[0],
        legacy_transformation_input_to_acc_primitive[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    _assert_acc_primitive_cells_equivalent(
        legacy_acc_primitive_cell,
        acc_primitive_cell,
        transformation_legacy_acc_to_acc_primitive,
        tol=tol,
        label=identify_info,
    )
    return (
        acc_primitive_cell,
        acc_primitive_ssg,
        transformation_input_to_acc_primitive,
        transformation_selected_standard_to_acc_primitive,
        {"strategy": "identify_index_p_direct_composed"},
    )


def _identity_setting_transform() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3), np.zeros(3)


def _identify_affine_4x4_to_setting_transform(
    matrix4x4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the identify-index internal 4x4 affine layout into the public
    direct-transform pair `(A, b)` with:

        x_target = A x_source + b

    Internal identify-index matrices use the nonstandard layout generated by
    `make_4d_matrix(...)`, where the affine block is stored as:

        [[1, 0, 0, 0],
         [b0, A00, A01, A02],
         [b1, A10, A11, A12],
         [b2, A20, A21, A22]]
    """
    matrix4x4 = np.asarray(matrix4x4, dtype=float)
    if matrix4x4.shape != (4, 4):
        raise ValueError(f"Expected identify affine 4x4 matrix, got shape {matrix4x4.shape}.")
    return matrix4x4[1:, 1:], matrix4x4[1:, 0]


def _acc_setting_allows_input_collapse(acc_symbol: str | None) -> bool:
    if not acc_symbol:
        return False
    return acc_symbol.rstrip().endswith("P")


def _cartesianized_input_cell(cell: CrystalCell) -> CrystalCell:
    moments_cartesian = cell.moments_cartesian
    return CrystalCell(
        lattice=np.asarray(cell.lattice_matrix, dtype=float),
        positions=np.asarray(cell.positions, dtype=float),
        occupancies=list(cell.occupancies),
        elements=list(cell.elements),
        moments=None if moments_cartesian is None else np.asarray(moments_cartesian, dtype=float),
        spin_setting=None if moments_cartesian is None else "cartesian",
        tol=cell.tol,
    )


def _cell_to_poscar_preserving_lattice(cell: CrystalCell, filename: str) -> str:
    lattice, positions, types, moments = cell.to_spglib(mag=True)
    rows = sorted(
        zip(positions, types, moments),
        key=lambda item: (
            item[1],
            item[2][0],
            item[2][1],
            item[2][2],
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    positions_sorted, types_sorted, moments_sorted = zip(*rows)

    atom_name = ["initial"]
    count = ["initial"]
    for atom_type in types_sorted:
        symbol = cell.atom_types_to_symbol[int(atom_type)]
        if symbol != atom_name[-1]:
            atom_name.append(symbol)
            count.append(1)
        else:
            count[-1] += 1

    return "\n".join(
        [
            filename + f"#FINDSPINGROUP(version{__version__})",
            "1",
            *(" ".join(f"{value:.10f}" for value in row) for row in np.asarray(lattice, dtype=float)),
            " ".join(atom_name[1:]),
            " ".join(map(str, count[1:])),
            "direct",
            *(" ".join(f"{value:.10f}" for value in position) for position in positions_sorted),
            "# MAGMOM="
            + " ".join(
                " ".join(f"{value:.10f}" for value in moment)
                for moment in moments_sorted
            ),
        ]
    )


def _canonicalize_standard_setting_transform(
    source_cell: CrystalCell,
    source_ssg: SpinSpaceGroup,
    transform: np.ndarray,
    shift: np.ndarray,
    *,
    tol: float = 1e-5,
) -> tuple[CrystalCell, SpinSpaceGroup, tuple[np.ndarray, np.ndarray], dict]:
    transform = np.asarray(transform, dtype=float)
    shift = np.asarray(shift, dtype=float)
    audit = audit_spatial_transform_effect(
        source_ssg,
        transform,
        shift,
        tol=tol,
        use_nssg=False,
    )
    if audit["real_ops_exact_same"]:
        return source_cell, source_ssg, _identity_setting_transform(), audit

    target_cell = source_cell.transform(transform, shift)
    target_ssg = source_ssg.transform(transform, shift)
    normalized = (
        transform,
        normalize_vector_to_zero(shift, atol=1e-10),
    )
    return target_cell, target_ssg, normalized, audit


def _canonicalize_input_to_standard_setting(
    input_cell_cartesian: CrystalCell,
    target_cell: CrystalCell,
    target_ssg: SpinSpaceGroup,
    transform_input_to_target: tuple[np.ndarray, np.ndarray],
    *,
    allow_identity_collapse: bool = True,
    tol: float = 1e-5,
) -> tuple[CrystalCell, SpinSpaceGroup, tuple[np.ndarray, np.ndarray], dict]:
    transform = np.asarray(transform_input_to_target[0], dtype=float)
    shift = np.asarray(transform_input_to_target[1], dtype=float)
    audit = audit_spatial_transform_effect(
        target_ssg,
        transform,
        shift,
        tol=tol,
        use_nssg=False,
    )
    if allow_identity_collapse and audit["real_ops_exact_same"]:
        target_to_input = _invert_setting_transform(transform, shift)
        input_basis_ssg = target_ssg.transform(*target_to_input)
        return input_cell_cartesian, input_basis_ssg, _identity_setting_transform(), audit
    return (
        target_cell,
        target_ssg,
        (transform, normalize_vector_to_zero(shift, atol=1e-10)),
        audit,
    )


def _format_basis_transform_component(value: float, symbol: str, *, tol: float = 1e-10) -> str:
    numeric = float(value)
    if abs(numeric) <= tol:
        return ""
    fraction = Fraction(numeric).limit_denominator(12)
    if abs(float(fraction) - numeric) > 1e-9:
        coeff = f"{numeric:.6f}".rstrip("0").rstrip(".")
    elif fraction.denominator == 1:
        coeff = str(fraction.numerator)
    else:
        coeff = f"{fraction.numerator}/{fraction.denominator}"
    if coeff == "1":
        return symbol
    if coeff == "-1":
        return f"-{symbol}"
    return f"{coeff}{symbol}"


def _format_basis_transform_rows(matrix: np.ndarray, symbols: tuple[str, str, str], *, tol: float = 1e-10) -> str:
    matrix = np.asarray(matrix, dtype=float)
    rows = []
    for row in matrix:
        pieces = [
            _format_basis_transform_component(value, symbol, tol=tol)
            for value, symbol in zip(row, symbols)
        ]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            rows.append("0")
            continue
        rendered = pieces[0]
        for piece in pieces[1:]:
            rendered += piece if piece.startswith("-") else f"+{piece}"
        rows.append(rendered)
    return ",".join(rows)


def _build_candidate_transform_chen_pp_abcs_hex_spatial_cubic_spin_from_identify(
    *,
    current_space_to_input_basis: np.ndarray,
    identify_point_group_transformation: np.ndarray,
) -> dict:
    """
    Provisional current-file -> Chen transform for the 3.24 audit slice.

    Source side:
    - current emitted `.scif` spatial basis = current `G0std_oriented` hex basis `(a,b,c)`
    - current emitted spin basis = file-declared `(as,bs,cs)` with `transform_spinframe_P_abc = 'a,b,c'`

    Target side for this slice:
    - Chen spatial basis = same hex basis `(A,B,C) = (a,b,c)`
    - Chen spin basis = cubic basis aligned to the input cubic setting and
      ordered/oriented according to the identify-index point-group map

    `current_space_to_input_basis` is the row-matrix whose rows are the input
    cubic basis vectors expressed in the current emitted spatial basis. The spin
    basis construction uses the corresponding column basis.
    `identify_point_group_transformation` is the identify-index 3x3 point-group
    transformation returned for the Chen equivalent-map resolution.
    """
    current_space_to_input_basis = np.asarray(current_space_to_input_basis, dtype=float)
    identify_point_group_transformation = np.asarray(
        identify_point_group_transformation,
        dtype=float,
    )
    spin_basis_current_to_chen = identify_point_group_transformation @ current_space_to_input_basis.T
    space_basis_current_to_chen = np.eye(3)
    origin_current_to_chen = np.zeros(3)
    return {
        "from_spatial_setting": "current_scif_ssg_convention_oriented_hex",
        "from_spin_frame": "current_file_spinframe_(as,bs,cs)",
        "to_spatial_setting": "chen_hex_spatial",
        "to_spin_frame": "chen_cubic_spin_basis",
        "space_basis_rows_abc": space_basis_current_to_chen.tolist(),
        "origin_shift_p": origin_current_to_chen.tolist(),
        "spin_basis_rows_abcs": spin_basis_current_to_chen.tolist(),
        "transform_Chen_Pp_abcs": (
            f"{_format_basis_transform_rows(space_basis_current_to_chen, ('a', 'b', 'c'))};"
            f"0,0,0;"
            f"{_format_basis_transform_rows(spin_basis_current_to_chen, ('as', 'bs', 'cs'))}"
        ),
    }


def _spin_transform_to_in_lattice(cell: CrystalCell) -> np.ndarray:
    actual_basis = np.array(
        [vector / np.linalg.norm(vector) for vector in np.asarray(cell.lattice_matrix, dtype=float)],
        dtype=float,
    ).T
    return np.linalg.inv(actual_basis)


def _spin_transform_to_oriented_abc(cell: CrystalCell) -> np.ndarray:
    lattice_col = np.asarray(cell.lattice_matrix, dtype=float).T
    return np.linalg.inv(lattice_col)


def _scif_spin_transform_for_frame(cell: CrystalCell, spin_frame: str) -> np.ndarray:
    if spin_frame == SCIF_SPIN_FRAME_CARTESIAN:
        return np.eye(3)
    if spin_frame == SCIF_SPIN_FRAME_ORIENTED:
        return _spin_transform_to_oriented_abc(cell)
    raise ValueError(f"Unsupported SCIF spin frame: {spin_frame}")


def _scif_moment_transform_for_frame(cell: CrystalCell, spin_frame: str) -> tuple[np.ndarray, str]:
    if spin_frame == SCIF_SPIN_FRAME_CARTESIAN:
        return np.eye(3), "cartesian"
    if spin_frame == SCIF_SPIN_FRAME_ORIENTED:
        # Formal SCIF atom moment axes are unit-coordinate components. Keep the
        # established oriented SCIF contract: operations use true abc, while
        # axis_u/v/w are stored along normalized lattice directions.
        return _spin_transform_to_in_lattice(cell), "in_lattice"
    raise ValueError(f"Unsupported SCIF spin frame: {spin_frame}")


def _scif_spinframe_basis_abc_rows(cell: CrystalCell, spin_transform: np.ndarray) -> np.ndarray:
    lattice_col = _lattice_column_matrix(cell)
    spin_basis_cartesian = np.linalg.inv(np.asarray(spin_transform, dtype=float))
    # Rows are the emitted spin-frame basis vectors written in the current
    # real-space lattice basis. For cartesian spin frame this is L^{-1}e_i;
    # for oriented spin frame this collapses to a,b,c.
    return (np.linalg.inv(lattice_col) @ spin_basis_cartesian).T


def _scif_moment_basis_cartesian(cell: CrystalCell, spin_frame: str) -> np.ndarray:
    if spin_frame == SCIF_SPIN_FRAME_CARTESIAN:
        return np.eye(3)
    if spin_frame == SCIF_SPIN_FRAME_ORIENTED:
        unit_lattice_basis = np.array(
            [vector / np.linalg.norm(vector) for vector in np.asarray(cell.lattice_matrix, dtype=float)],
            dtype=float,
        ).T
        return unit_lattice_basis
    raise ValueError(f"Unsupported SCIF spin frame: {spin_frame}")


def _build_scif_export_targets(
    *,
    input_cell: CrystalCell,
    acc_magnetic_primitive_cell: CrystalCell,
    acc_magnetic_primitive_ssg: SpinSpaceGroup,
    database_standard_cell: CrystalCell,
    database_standard_ssg: SpinSpaceGroup,
    database_standard_setting: str,
    convention_cell: CrystalCell,
    convention_ssg: SpinSpaceGroup,
    convention_setting: str,
    transformation_input_to_acc_primitive: tuple[np.ndarray, np.ndarray],
    transformation_input_to_database_standard: tuple[np.ndarray, np.ndarray],
    transformation_input_to_convention: tuple[np.ndarray, np.ndarray],
    transformation_input_to_G0std: tuple[np.ndarray, np.ndarray],
    transformation_input_to_L0std: tuple[np.ndarray, np.ndarray],
    input_identified_ssg: SpinSpaceGroup | None = None,
):
    def _basis_tag_transforms_for_export(
        transformation_input_to_export: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            "input": transformation_input_to_export,
            "magnetic_primitive": _compose_setting_transform(
                transformation_input_to_acc_primitive[0],
                transformation_input_to_acc_primitive[1],
                transformation_input_to_export[0],
                transformation_input_to_export[1],
            ),
            "G0std": _compose_setting_transform(
                transformation_input_to_G0std[0],
                transformation_input_to_G0std[1],
                transformation_input_to_export[0],
                transformation_input_to_export[1],
            ),
            "L0std": _compose_setting_transform(
                transformation_input_to_L0std[0],
                transformation_input_to_L0std[1],
                transformation_input_to_export[0],
                transformation_input_to_export[1],
            ),
        }

    def _target(
        *,
        cell_mode: str,
        base_cell: CrystalCell,
        base_ssg: SpinSpaceGroup,
        transformation_input_to_export: tuple[np.ndarray, np.ndarray],
        setting_name: str,
        spin_frame: str,
        is_input_setting: bool = False,
    ):
        spin_transform = _scif_spin_transform_for_frame(base_cell, spin_frame)
        moment_transform, moment_setting = _scif_moment_transform_for_frame(base_cell, spin_frame)
        return cell_mode, {
            "export_cell": base_cell.transform_spin(moment_transform, moment_setting),
            "export_ssg": base_ssg.transform_spin(spin_transform),
            "transform_input_to_export": transformation_input_to_export,
            "basis_tag_transforms": _basis_tag_transforms_for_export(
                transformation_input_to_export,
            ),
            "setting_name": setting_name,
            "spin_frame": spin_frame,
            "spinframe_basis_abc_rows": _scif_spinframe_basis_abc_rows(base_cell, spin_transform),
            "moment_basis_cartesian": _scif_moment_basis_cartesian(base_cell, spin_frame),
            "is_input_setting": is_input_setting,
        }

    targets = dict([
        _target(
            cell_mode=SCIF_CELL_MODE_SSG_CONVENTION_CARTESIAN,
            base_cell=convention_cell,
            base_ssg=convention_ssg,
            transformation_input_to_export=transformation_input_to_convention,
            setting_name=convention_setting,
            spin_frame=SCIF_SPIN_FRAME_CARTESIAN,
        ),
        _target(
            cell_mode=SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED,
            base_cell=convention_cell,
            base_ssg=convention_ssg,
            transformation_input_to_export=transformation_input_to_convention,
            setting_name=convention_setting,
            spin_frame=SCIF_SPIN_FRAME_ORIENTED,
        ),
        _target(
            cell_mode=SCIF_CELL_MODE_DATABASE_STANDARD_CARTESIAN,
            base_cell=database_standard_cell,
            base_ssg=database_standard_ssg,
            transformation_input_to_export=transformation_input_to_database_standard,
            setting_name=database_standard_setting,
            spin_frame=SCIF_SPIN_FRAME_CARTESIAN,
        ),
        _target(
            cell_mode=SCIF_CELL_MODE_DATABASE_STANDARD_ORIENTED,
            base_cell=database_standard_cell,
            base_ssg=database_standard_ssg,
            transformation_input_to_export=transformation_input_to_database_standard,
            setting_name=database_standard_setting,
            spin_frame=SCIF_SPIN_FRAME_ORIENTED,
        ),
        _target(
            cell_mode=SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_CARTESIAN,
            base_cell=acc_magnetic_primitive_cell,
            base_ssg=acc_magnetic_primitive_ssg,
            transformation_input_to_export=transformation_input_to_acc_primitive,
            setting_name=ACC_PRIMITIVE_SETTING,
            spin_frame=SCIF_SPIN_FRAME_CARTESIAN,
        ),
        _target(
            cell_mode=SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_ORIENTED,
            base_cell=acc_magnetic_primitive_cell,
            base_ssg=acc_magnetic_primitive_ssg,
            transformation_input_to_export=transformation_input_to_acc_primitive,
            setting_name=ACC_PRIMITIVE_SETTING,
            spin_frame=SCIF_SPIN_FRAME_ORIENTED,
        ),
    ])
    if input_identified_ssg is not None:
        identity = (np.eye(3), np.zeros(3))
        targets.update(dict([
            _target(
                cell_mode=SCIF_CELL_MODE_INPUT_CARTESIAN,
                base_cell=input_cell,
                base_ssg=input_identified_ssg,
                transformation_input_to_export=identity,
                setting_name="input",
                spin_frame=SCIF_SPIN_FRAME_CARTESIAN,
                is_input_setting=True,
            ),
            _target(
                cell_mode=SCIF_CELL_MODE_INPUT_ORIENTED,
                base_cell=input_cell,
                base_ssg=input_identified_ssg,
                transformation_input_to_export=identity,
                setting_name="input",
                spin_frame=SCIF_SPIN_FRAME_ORIENTED,
                is_input_setting=True,
            ),
        ]))
    return targets


def _identify_parent_space_group_for_export_cell(
    export_cell: CrystalCell,
    *,
    symprec: float,
    source_parent_space_group: dict | None = None,
    reuse_source_transforms: bool = False,
):
    cell = export_cell.to_spglib(mag=False)
    dataset = get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return None, {
            "status": "generation_failed",
            "matches_input": None,
            "input_name_H_M_alt": (
                None if source_parent_space_group is None else source_parent_space_group.get("name_H_M_alt")
            ),
            "input_IT_number": (
                None if source_parent_space_group is None else source_parent_space_group.get("IT_number")
            ),
        }
    if dataset.number in SG_HALL_MAPPING:
        dataset = get_symmetry_dataset(cell, symprec=symprec, hall_number=SG_HALL_MAPPING[dataset.number])

    generated = {
        "name_H_M_alt": str(dataset.international),
        "IT_number": int(dataset.number),
        "transform_to_parent_space_group_Pp": "",
    }
    try:
        generated["transform_to_parent_space_group_Pp"] = affine_matrix_to_xyz_expression(
            np.asarray(dataset.transformation_matrix, dtype=float).T,
            normalize_vector_to_zero(np.asarray(dataset.origin_shift, dtype=float), atol=1e-9),
            ('a', 'b', 'c'),
            separate_translation=True,
            coeff_precision=6,
        )
    except Exception as exc:
        warnings.warn(
            f"Unable to serialize parent-space-group transform for export cell: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        generated["child_transform_Pp_abc"] = affine_matrix_to_xyz_expression(
            np.asarray(dataset.transformation_matrix, dtype=float).T,
            normalize_vector_to_zero(np.asarray(dataset.origin_shift, dtype=float), atol=1e-9),
            ('a', 'b', 'c'),
            separate_translation=True,
            coeff_precision=6,
        )
    except Exception as exc:
        warnings.warn(
            f"Unable to serialize parent-space-group child transform for export cell: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    matches_input = None
    if source_parent_space_group is not None:
        checks = []
        input_number = source_parent_space_group.get("IT_number")
        input_name = source_parent_space_group.get("name_H_M_alt")
        if input_number is not None:
            checks.append(int(round(float(input_number))) == generated["IT_number"])
        if input_name is not None:
            checks.append(
                re.sub(r"\s+", "", str(input_name).strip())
                == re.sub(r"\s+", "", generated["name_H_M_alt"])
            )
        if checks:
            matches_input = all(checks)
        if matches_input and reuse_source_transforms:
            parent_transform = source_parent_space_group.get("transform_Pp_abc")
            child_transform = source_parent_space_group.get("child_transform_Pp_abc")
            if input_name is not None:
                generated["name_H_M_alt"] = str(input_name).strip()
            if input_number is not None:
                generated["IT_number"] = int(round(float(input_number)))
            if parent_transform is not None:
                generated["transform_Pp_abc"] = parent_transform
            if child_transform is not None and "child_transform_Pp_abc" not in generated:
                generated["child_transform_Pp_abc"] = child_transform
        elif matches_input:
            if input_name is not None:
                generated["name_H_M_alt"] = str(input_name).strip()
            if input_number is not None:
                generated["IT_number"] = int(round(float(input_number)))

    if matches_input is True:
        status = "matches_input_metadata"
    elif matches_input is False:
        status = "generated_differs_from_input_metadata"
    else:
        status = "generated_without_input_metadata"

    return generated, {
        "status": status,
        "matches_input": matches_input,
        "input_name_H_M_alt": (
            None if source_parent_space_group is None else source_parent_space_group.get("name_H_M_alt")
        ),
        "input_IT_number": (
            None if source_parent_space_group is None else source_parent_space_group.get("IT_number")
        ),
    }


def _primitive_msg_ops_from_ssg(ssg_ops, tol: float, time_reversal_resolver=None) -> list[list]:
    primitive_msg_ops = []
    for op in ssg_ops:
        if time_reversal_resolver is None:
            time_reversal = op.magnetic_time_reversal(atol=tol)
        else:
            time_reversal = time_reversal_resolver(op)
        if time_reversal is None:
            continue
        primitive_msg_ops.append(
            [
                int(time_reversal),
                np.asarray(op[1], dtype=float),
                np.asarray(op[2], dtype=float),
            ]
        )
    return primitive_msg_ops


def _get_magnetic_little_group(kpoint, primitive_msg_operations, tol: float) -> list[list]:
    magnetic_little_group = []
    primitive_kpoint = np.asarray(kpoint, dtype=float)
    for time_reversal, rotation, translation in primitive_msg_operations:
        transformed_kpoint = time_reversal * np.asarray(rotation, dtype=float) @ primitive_kpoint
        if getNormInf(transformed_kpoint % 1, primitive_kpoint) < tol:
            magnetic_little_group.append([time_reversal, rotation, translation])
    return magnetic_little_group


def _get_ssg_little_groups(ssg: SpinSpaceGroup, *, tol: float) -> list[list[SpinSpaceGroupOperation]]:
    kpoints = ssg.kpoints_primitive if ssg.is_primitive else ssg.kpoints_conventional
    ops = list(ssg.ops)
    effective_ops = [
        np.linalg.det(op.spin_rotation) * np.linalg.inv(np.asarray(op.rotation, dtype=float)).T
        for op in ops
    ]

    if ssg.cptrans is None or np.allclose(ssg.cptrans, np.eye(3), atol=tol):
        little_groups = []
        for kpoint in kpoints:
            primitive_kpoint = np.asarray(kpoint, dtype=float)
            little_group = []
            for op, effective_op in zip(ops, effective_ops):
                target_kpoint = effective_op @ primitive_kpoint % 1
                if getNormInf(primitive_kpoint % 1, target_kpoint) < tol:
                    little_group.append(op)
            little_groups.append(little_group)
        return little_groups

    cptrans = np.asarray(ssg.cptrans, dtype=float)
    cptrans_inv = np.linalg.inv(cptrans)
    conjugated_effective_ops = [
        cptrans_inv @ effective_op @ cptrans
        for effective_op in effective_ops
    ]
    little_groups = []
    for kpoint in kpoints:
        kpoint_array = np.asarray(kpoint, dtype=float)
        little_group = []
        for op, effective_op, conjugated_effective_op in zip(ops, effective_ops, conjugated_effective_ops):
            if ssg.is_primitive:
                target_kpoint = effective_op @ kpoint_array % 1
                if getNormInf(kpoint_array % 1, target_kpoint) < tol:
                    little_group.append(op)
            else:
                primitive_kpoint = cptrans.T @ kpoint_array % 1
                transformed_primitive = conjugated_effective_op @ primitive_kpoint % 1
                if getNormInf(primitive_kpoint, transformed_primitive) < tol:
                    little_group.append(op)
        little_groups.append(little_group)
    return little_groups


def _get_spin_constraint_for_msg_little_groups(
    little_groups: list[list[list]],
    cell: CrystalCell,
    tol: float,
    spin_frame_rotation: np.ndarray | None = None,
) -> list[list[str]]:
    lattice_col = _lattice_column_matrix(cell)
    target_rotation = None if spin_frame_rotation is None else np.asarray(spin_frame_rotation, dtype=float)
    target_rotation_inv = None if target_rotation is None else np.linalg.inv(target_rotation)
    constraints = []
    for little_group in little_groups:
        spin_matrices = []
        for time_reversal, rotation, _ in little_group:
            rotation_cartesian = _cartesianize_similarity(rotation, lattice_col)
            if target_rotation is not None:
                rotation_cartesian = target_rotation @ rotation_cartesian @ target_rotation_inv
            spin_matrices.append(time_reversal * np.linalg.det(rotation_cartesian) * rotation_cartesian - np.eye(3))
        spinmatrices = np.vstack(deduplicate_matrix_pairs(spin_matrices, tol=tol))
        constraints.append(combine_parametric_solutions(rref_with_tolerance(spinmatrices)))
    return constraints


def _build_msg_little_group_payload(
    ssg: SpinSpaceGroup,
    cell: CrystalCell,
    tol: float,
    spin_frame_rotation: np.ndarray | None = None,
) -> tuple[list[list], list[str | None], list[list[str]]]:
    primitive_msg_ops, little_groups, little_group_symbols = _build_msg_little_group_core(
        ssg,
        tol=tol,
    )
    spin_constraints = _get_spin_constraint_for_msg_little_groups(
        little_groups,
        cell=cell,
        tol=tol,
        spin_frame_rotation=spin_frame_rotation,
    )
    return primitive_msg_ops, little_group_symbols, spin_constraints


def _build_msg_little_group_core(
    ssg: SpinSpaceGroup,
    *,
    tol: float,
) -> tuple[list[list], list[list[list]], list[str | None]]:
    primitive_msg_ops = _primitive_msg_ops_from_ssg(
        ssg.msg_ops,
        tol=tol,
        time_reversal_resolver=ssg.classify_magnetic_operation,
    )
    little_groups = [
        _get_magnetic_little_group(kpoint, primitive_msg_ops, tol=tol)
        for kpoint in ssg.kpoints_primitive
    ]
    little_group_symbols = []
    for group in little_groups:
        if not group:
            little_group_symbols.append("1")
            continue
        msg_info = get_magnetic_space_group_from_operations(group)
        if msg_info is None:
            little_group_symbols.append(None)
        else:
            little_group_symbols.append(msg_info["mpg_symbol"])
    return primitive_msg_ops, little_groups, little_group_symbols


def _quasi2d_in_plane_acc_kpoint_indices(quasi_2d: dict | None) -> list[int]:
    if not isinstance(quasi_2d, dict):
        return []
    indices = []
    acc_index = 0
    for row in quasi_2d.get("kpoints") or []:
        if row.get("kind") != "acc_table":
            continue
        if row.get("plane_classification") == "in_plane":
            indices.append(acc_index)
        acc_index += 1
    return indices


def _select_indices(values, indices: list[int]) -> list:
    return [values[index] for index in indices if 0 <= index < len(values)]


def _build_quasi2d_little_group_payload(
    *,
    quasi_2d: dict | None,
    acc_primitive_ssg: SpinSpaceGroup,
    ssg_little_groups: list[list[SpinSpaceGroupOperation]],
    msg_little_groups: list[list[list]],
    msg_little_group_symbols: list[str | None],
    msg_spin_polarizations: list[list[str]],
    tol: float,
) -> dict:
    indices = _quasi2d_in_plane_acc_kpoint_indices(quasi_2d)
    if not indices:
        return {
            "ssg_little_group_symbol_2d": [],
            "msg_little_group_symbol_2d": [],
            "msg_spin_polarization_2d": [],
            "ssg_little_group_ops_2d": [],
            "ssg_little_group_seitz_latex_2d": [],
            "msg_little_group_ops_2d": [],
            "msg_little_group_seitz_latex_2d": [],
        }

    ssg_little_group_symbols = list(acc_primitive_ssg.little_groups_symbols)
    ssg_little_groups_2d = _select_indices(ssg_little_groups, indices)
    msg_little_groups_2d = _select_indices(msg_little_groups, indices)
    return {
        "ssg_little_group_symbol_2d": _select_indices(ssg_little_group_symbols, indices),
        "msg_little_group_symbol_2d": _select_indices(msg_little_group_symbols, indices),
        "msg_spin_polarization_2d": _select_indices(msg_spin_polarizations, indices),
        "ssg_little_group_ops_2d": _serialize_ssg_little_group_ops(ssg_little_groups_2d),
        "ssg_little_group_seitz_latex_2d": _serialize_ssg_little_group_seitz_latex(
            ssg_little_groups_2d,
            tol=tol,
        ),
        "msg_little_group_ops_2d": _serialize_msg_little_group_ops(msg_little_groups_2d),
        "msg_little_group_seitz_latex_2d": _serialize_msg_little_group_seitz_latex(
            msg_little_groups_2d,
            tol=tol,
        ),
    }


def _format_wp_with_site_dof(wp_symbol, dof):
    if dof is None:
        return wp_symbol
    return f"{wp_symbol}({int(dof)})"


def _make_wp_chain_and_site_order(
    wp_sg,
    wp_ssg,
    wp_msg,
    cell,
    atom_types_dict,
    *,
    ssg_dof_by_site=None,
    msg_dof_by_site=None,
):
    ssg_dof_by_site = {} if ssg_dof_by_site is None else ssg_dof_by_site
    msg_dof_by_site = {} if msg_dof_by_site is None else msg_dof_by_site
    chain = tuple(
        (
            atom_types_dict[int(cell[2][i])],
            wp_sg[i][0],
            wp_sg[i][1],
            _format_wp_with_site_dof(wp_ssg[i][0], ssg_dof_by_site.get(i)),
            wp_ssg[i][1],
            _format_wp_with_site_dof(wp_msg[i][0], msg_dof_by_site.get(i)),
            wp_msg[i][1],
        )
        for i in range(min(len(wp_sg), len(wp_ssg), len(wp_msg)))
    )
    site_order = sorted(range(len(chain)), key=lambda index: (str(chain[index][0]), chain[index][1:]))
    sorted_chain = []
    seen = set()
    for index in site_order:
        row = chain[index]
        if row in seen:
            continue
        sorted_chain.append(row)
        seen.add(row)
    if len(site_order) != len(cell[2]):
        site_order = None
    return sorted_chain, site_order


def _make_wp_chain(wp_sg, wp_ssg, wp_msg, cell, atom_types_dict):
    wp_chain, _ = _make_wp_chain_and_site_order(wp_sg, wp_ssg, wp_msg, cell, atom_types_dict)
    return wp_chain


def _get_wp_for_original_sites(dataset, site_count: int):
    # get_G0_dataset_for_cell appends a synthetic generic-site orbit; public
    # Wyckoff chains must describe only the physical sites in the input cell.
    return get_wp_from_dataset(dataset, max=False)[:site_count]


def _build_wp_chain_payload_and_site_order(
    g0_cell: CrystalCell,
    g0_ssg: SpinSpaceGroup,
    tol_cfg: Tolerances,
    *,
    annotate_magnetic_site_dof: bool = False,
):
    sg_dataset = get_symmetry_dataset(g0_cell.to_spglib(), symprec=tol_cfg.space)
    oriented_ssg = _ossg_oriented_spin_frame_ssg(g0_ssg, g0_cell)
    msg_ops = [[op[1], op[2]] for op in oriented_ssg.msg_ops]
    if not msg_ops:
        return [], None
    msg_dataset = get_G0_dataset_for_cell(msg_ops, g0_cell.to_spglib(mag=True), tol_cfg.space)
    ssg_dataset = get_G0_dataset_for_cell(g0_ssg.G0_ops, g0_cell.to_spglib(mag=True), tol_cfg.space)
    site_count = len(g0_cell.to_spglib(mag=True)[1])
    wp_extended_sg = get_wp_from_dataset(sg_dataset, max=False)
    wp_extended_ssg = _get_wp_for_original_sites(ssg_dataset, site_count)
    wp_extended_msg = _get_wp_for_original_sites(msg_dataset, site_count)
    ssg_dof_by_site = None
    msg_dof_by_site = None
    if annotate_magnetic_site_dof:
        ssg_dof_by_site, msg_dof_by_site = _magnetic_site_dof_maps_for_cell(
            g0_cell,
            g0_ssg,
            tol_cfg,
        )
    return _make_wp_chain_and_site_order(
        wp_extended_sg,
        wp_extended_ssg,
        wp_extended_msg,
        g0_cell.to_spglib(mag=True),
        g0_cell.atom_types_to_symbol,
        ssg_dof_by_site=ssg_dof_by_site,
        msg_dof_by_site=msg_dof_by_site,
    )


def _build_wp_chain_payload(g0_cell: CrystalCell, g0_ssg: SpinSpaceGroup, tol_cfg: Tolerances):
    wp_chain, _ = _build_wp_chain_payload_and_site_order(g0_cell, g0_ssg, tol_cfg)
    return wp_chain


def _magnetic_site_dof_maps_for_cell(
    cell: CrystalCell,
    ssg: SpinSpaceGroup,
    tol_cfg: Tolerances,
) -> tuple[dict[int, int], dict[int, int]]:
    sg_dataset = get_symmetry_dataset(cell.to_spglib(), symprec=tol_cfg.space)
    site_count = len(cell.to_spglib(mag=True)[1])
    nonzero_moment_indices = [] if cell.magnetic_atom_indices is None else list(cell.magnetic_atom_indices)
    magnetic_indices, _selection = _expand_magnetic_indices_by_sg_orbit(
        sg_dataset,
        nonzero_moment_indices,
        site_count,
    )
    _ssg_magnetic_indices, _ssg_classes, ssg_dof, ssg_spin_classes, ssg_constraints = (
        get_spin_wyckoff(
            cell,
            ssg.ops,
            atol=tol_cfg.m_matrix_tol,
            magnetic_indices=magnetic_indices,
        )
    )
    ssg_dof_rows = _site_dof_rows(ssg_spin_classes, ssg_dof, ssg_constraints)
    ssg_dof_by_site, _ssg_constraints_by_site, _ssg_representative_by_site = _site_dof_maps(
        ssg_dof_rows
    )

    msg_dof_by_site = {}
    oriented_ssg = _ossg_oriented_spin_frame_ssg(ssg, cell)
    msg_ops = list(oriented_ssg.msg_ops)
    if msg_ops:
        _msg_magnetic_indices, _msg_classes, msg_dof, msg_spin_classes, msg_constraints = (
            get_spin_wyckoff(
                cell,
                msg_ops,
                atol=tol_cfg.m_matrix_tol,
                magnetic_indices=magnetic_indices,
            )
        )
        msg_dof_rows = _site_dof_rows(msg_spin_classes, msg_dof, msg_constraints)
        msg_dof_by_site, _msg_constraints_by_site, _msg_representative_by_site = _site_dof_maps(
            msg_dof_rows
        )

    return ssg_dof_by_site, msg_dof_by_site


def _is_fm_fim_spin_point_group_symbol(symbol: str) -> bool:
    normalized = str(symbol).strip()
    return (
        bool(re.match(r"^C\d+(?!h)", normalized))
        or bool(re.match(r"^Cs", normalized))
        or normalized == "C∞v"
        or normalized == "C∞ v"
        or bool(re.match(r"^C_\{\\infty} v", normalized))
        or bool(re.match(r"^C\*v", normalized))
        or normalized == "∞m"
    )


def _is_fm_fim_spin_point_group(*symbols: str) -> bool:
    return any(_is_fm_fim_spin_point_group_symbol(symbol) for symbol in symbols if symbol is not None)


def classify_magnetic_phase(
    *,
    conf,
    full_spin_part_point_group_hm,
    full_spin_part_point_group_s,
    net_moment,
    mpg_identifier,
    is_ss_gp,
    net_moment_tol=None,
):
    net_moment_value = float(net_moment)
    zero_net_moment_tol = float(
        DEFAULT_TOL.moment if net_moment_tol is None else net_moment_tol
    )
    zero_net_moment = abs(net_moment_value) < zero_net_moment_tol
    fm_like_by_spin_point_group = _is_fm_fim_spin_point_group(
        full_spin_part_point_group_hm,
        full_spin_part_point_group_s,
    )
    som_by_mpg = mpg_identifier in MSGMPG_DB.FMMPG_INTlist if mpg_identifier is not None else False

    if fm_like_by_spin_point_group:
        base_phase = 'Compensated FiM' if zero_net_moment else 'FM/FiM'
        classification_rule = 'fm_like_spin_point_group'
    else:
        base_phase = 'AFM'
        classification_rule = (
            'afm_with_spin_orbit_magnet'
            if som_by_mpg
            else 'default_antiferromagnetic'
        )

    ss_wo_soc = spin_splitting_wo_soc(base_phase, is_ss_gp)
    alter_tag = is_alter(conf, base_phase, ss_wo_soc)
    som_tag = '(SOM)' if base_phase == 'AFM' and som_by_mpg else ''
    phase = base_phase + alter_tag
    if som_tag:
        phase += '\n' + som_tag

    return {
        'phase': phase,
        'base_phase': base_phase,
        'modifier': alter_tag,
        'spin_orbit_magnet_tag': som_tag,
        'details': {
            'conf': conf,
            'full_spin_part_point_group_hm': full_spin_part_point_group_hm,
            'full_spin_part_point_group_s': full_spin_part_point_group_s,
            'mpg_identifier': mpg_identifier,
            'net_moment': net_moment_value,
            'zero_net_moment_tol': zero_net_moment_tol,
            'zero_net_moment': zero_net_moment,
            'fm_like_by_spin_point_group': fm_like_by_spin_point_group,
            'som_by_mpg': som_by_mpg,
            'classification_rule': classification_rule,
            'base_phase': base_phase,
            'modifier': alter_tag,
            'spin_orbit_magnet_tag': som_tag,
            'spin_splitting_without_soc': ss_wo_soc,
            'is_altermagnet': bool(alter_tag),
            'is_spin_orbit_magnet': bool(som_tag),
        },
        'spin_splitting_without_soc': ss_wo_soc,
        'is_alter': alter_tag,
        'is_spin_orbit_magnet': som_tag,
    }


def get_magnetic_phase(
    full_spin_part_point_group_hm,
    full_spin_part_point_group_s,
    net_moment,
    mpg,
    conf=None,
    is_ss_gp=None,
    net_moment_tol=None,
):
    if conf is None or is_ss_gp is None:
        hm_symbol = full_spin_part_point_group_hm
        s_symbol = full_spin_part_point_group_s
        if full_spin_part_point_group_s is None:
            hm_symbol = None
            s_symbol = full_spin_part_point_group_hm
        return classify_magnetic_phase(
            conf='Unknown',
            full_spin_part_point_group_hm=hm_symbol,
            full_spin_part_point_group_s=s_symbol,
            net_moment=net_moment,
            net_moment_tol=net_moment_tol,
            mpg_identifier=mpg,
            is_ss_gp='spin splitting',
        )['base_phase']

    return classify_magnetic_phase(
        conf=conf,
        full_spin_part_point_group_hm=full_spin_part_point_group_hm,
        full_spin_part_point_group_s=full_spin_part_point_group_s,
        net_moment=net_moment,
        net_moment_tol=net_moment_tol,
        mpg_identifier=mpg,
        is_ss_gp=is_ss_gp,
    )['base_phase']



def getNormInf(matrix1, matrix2, mode=True):
    if mode == True:
        a = np.array(matrix1) % 1
        b = np.array(matrix2) % 1
        c = [1, 2, 3]
        for i in range(3):
            if a[i] > b[i]:
                c[i] = min(a[i] - b[i], 1 + b[i] - a[i])
            if a[i] < b[i]:
                c[i] = min(b[i] - a[i], 1 + a[i] - b[i])
            if a[i] == b[i]:
                c[i] = 0
        max_value = max(c)
    else:
        diff = np.abs(matrix1 - matrix2)
        max_value = np.max(diff)
    return max_value

def combine_parametric_solutions(rref_matrix, tol=1e-3):
    import numpy as np

    A = np.array(rref_matrix, dtype=float)
    rows, cols = A.shape
    pivot_cols = []


    for i in range(rows):
        for j in range(cols):
            if abs(A[i, j]) > tol:
                pivot_cols.append(j)
                break

    pivot_cols = set(pivot_cols)
    free_vars = [j for j in range(cols) if j not in pivot_cols]


    symbols = ['Sx', 'Sy', 'Sz']
    vector_expr = ['0'] * cols

    for free_idx, var_col in enumerate(free_vars):
        coeffs = [0] * cols
        coeffs[var_col] = 1
        for row_idx in range(rows):
            row = A[row_idx]
            pivot_col = next((j for j in range(cols) if abs(row[j]) > tol), None)
            if pivot_col is not None and abs(row[var_col]) > tol:
                coeffs[pivot_col] = -row[var_col]

        if len(free_vars) == 1:
            first_nonzero_component = next(i for i, value in enumerate(coeffs) if abs(value) > tol)
            var_name = symbols[first_nonzero_component]
        else:
            var_name = symbols[free_idx]


        for i in range(cols):
            c = coeffs[i]
            if abs(c) < tol:
                continue
            if vector_expr[i] == '0':
                if abs(c - 1) < tol:
                    vector_expr[i] = var_name
                elif abs(c + 1) < tol:
                    vector_expr[i] = f"-{var_name}"
                else:
                    vector_expr[i] = f"{format_symbolic_scalar(c)}*{var_name}"
            else:
                if abs(c - 1) < tol:
                    vector_expr[i] += f" + {var_name}"
                elif abs(c + 1) < tol:
                    vector_expr[i] += f" - {var_name}"
                elif c > 0:
                    vector_expr[i] += f" + {format_symbolic_scalar(c)}*{var_name}"
                else:
                    vector_expr[i] += f" - {format_symbolic_scalar(abs(c))}*{var_name}"

    return vector_expr

def calculate_freedom_degree(matrices : list[np.ndarray],tol=0.01):
    """
        calculate freedom degree from matrices
    """
    stack_matrices = np.vstack(matrices-np.eye(3)).astype(np.float64)

    # rref(stack_matrices, tol=0.01)
    # pending for (mx,my,mz) representation
    constraints = combine_parametric_solutions(rref_with_tolerance(stack_matrices))
    return 3 - np.linalg.matrix_rank(stack_matrices,tol=tol), constraints

def get_spin_wyckoff(
    ssg_cell: CrystalCell,
    ssg_ops,
    atol=0.001,
    magnetic_indices: list[int] | None = None,
) -> (list, list):
    """
    Calculate spin Wyckoff positions information.

    Parameters:
        ssg_cell_spglib (list): A list containing cell information.
                         - ssg_cell[1]: Atomic positions (numpy array).
                         - ssg_cell[3]: Magnetic moments (numpy array).
        ssg_ops (list): A list of symmetry operations, where each operation is a np list (Rs ||Rr | t).

    Returns:
        Tuple[dict, dict]:
            - magnetic_index: A dictionary mapping magnetic atom indices to their multiplicities.
            - magnetic_index_site_symmetry: A dictionary mapping magnetic atom indices to their site symmetry operations.
    """

    if not ssg_cell or not ssg_ops:
        raise ValueError("Input ssg_cell and ssg_ops cannot be empty.")
    ssg_cell_spglib = ssg_cell.to_spglib(mag=True)

    coords = np.array(ssg_cell_spglib[1])
    atom_types = list(ssg_cell.atom_types)

    bins = max(1, int(np.ceil(1.0 / max(atol, 1e-12))))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(atol / bucket_width)))

    def _bucket_key(position):
        wrapped = np.mod(np.asarray(position, dtype=float), 1.0)
        indices = np.floor(wrapped * bins).astype(int) % bins
        return tuple(int(value) for value in indices)

    def _neighbor_keys(bucket_key):
        for dx in range(-neighbor_radius, neighbor_radius + 1):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                for dz in range(-neighbor_radius, neighbor_radius + 1):
                    yield (
                        (bucket_key[0] + dx) % bins,
                        (bucket_key[1] + dy) % bins,
                        (bucket_key[2] + dz) % bins,
                    )

    typed_position_buckets: dict[tuple, list[int]] = {}
    for index, coord in enumerate(coords):
        typed_position_buckets.setdefault((atom_types[index], _bucket_key(coord)), []).append(index)

    # Get indices of magnetic atoms and initialization

    if magnetic_indices is None:
        magnetic_index = [] if ssg_cell.magnetic_atom_indices is None else list(ssg_cell.magnetic_atom_indices)
    else:
        magnetic_index = sorted({int(index) for index in magnetic_indices})
    magnetic_index_set = set(magnetic_index)

    num_atoms = len(coords)
    assigned = [False] * num_atoms
    equivalence_classes = []

    equivalence_classes_spin = []

    for i in range(num_atoms):
        if assigned[i]:
            continue
        class_i = []
        site_symmetry_ops = []
        for op in ssg_ops:
            Rr = np.array(op[1])
            t = np.array(op[2])
            trans = normalize_vector_to_zero(Rr @ coords[i] + t)
            candidate_indices = []
            seen_candidates = set()
            for neighbor_key in _neighbor_keys(_bucket_key(trans)):
                for candidate in typed_position_buckets.get((atom_types[i], neighbor_key), ()):
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    candidate_indices.append(candidate)
            best_match = None
            best_score = None
            for j in candidate_indices:
                dist = getNormInf(trans, coords[j])
                if dist < atol:
                    score = (dist, 0 if i == j else 1, j)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_match = j
            if best_match is not None:
                j = best_match
                if j not in class_i:
                    class_i.append(j)
                    assigned[j] = True
                # Collect every operation that stabilizes the representative
                # site.  Near-coincident sites can share a tolerance bucket, so
                # this must use the nearest site rather than the lowest index.
                if i == j:
                    site_symmetry_ops.append(np.array(op[0]))
        equivalence_classes.append({
            "representative_index": i,
            "class_indices": class_i,
            "site_symmetry_ops": site_symmetry_ops
        })
        if magnetic_index_set.intersection(class_i):
            equivalence_classes_spin.append({
                "representative_index": i,
                "class_indices": class_i,
                "site_symmetry_ops": site_symmetry_ops
            })
        # print(class_i)

    # Calculate site symmetry of representative magnetic atoms

    # get degree of freedom of moment
    magnetic_representative_dof = {}
    constraints = []
    for info in equivalence_classes_spin:
        dof, constraint = calculate_freedom_degree(info['site_symmetry_ops'], tol=atol)
        magnetic_representative_dof[info['representative_index']] = int(dof)
        constraints.append(constraint)

    return magnetic_index, equivalence_classes, magnetic_representative_dof,equivalence_classes_spin,constraints


def _parse_parent_child_expansion(child_transform: str | None):
    if child_transform is None:
        return None
    text = str(child_transform).strip().strip("'\"")
    if not text or text == ".":
        return None
    expr = text.split(";", 1)[0]
    matrices, _time_reversals = general_positions_to_matrix(
        [f"{expr},+1"],
        variables=("a", "b", "c"),
    )
    basis_rows, _shift = matrices[0]
    basis_change = np.asarray(basis_rows, dtype=float).T
    determinant = abs(float(np.linalg.det(basis_change)))
    rounded = int(round(determinant))
    if np.isclose(determinant, rounded, atol=1e-8, rtol=0.0):
        return rounded
    return determinant


def _magnetic_orbit_count_from_dataset(dataset, magnetic_indices: list[int]) -> int:
    if not magnetic_indices:
        return 0
    orbit_labels = _dataset_wyckoff_orbits(dataset)
    return len({int(orbit_labels[index]) for index in magnetic_indices})


def _expand_magnetic_indices_by_sg_orbit(
    dataset,
    magnetic_indices: list[int],
    site_count: int,
) -> tuple[list[int], dict]:
    """Include zero-moment sites split from the same SG orbit as magnetic sites."""
    source_indices = sorted(
        {int(index) for index in magnetic_indices if 0 <= int(index) < site_count}
    )
    if not source_indices:
        return [], {
            "mode": "sg_orbit_closure_of_nonzero_moment_sites",
            "source_nonzero_moment_indices": [],
            "included_zero_moment_indices": [],
            "parent_sg_orbit_labels": [],
        }

    orbit_labels = _dataset_wyckoff_orbits(dataset)
    parent_orbit_labels = sorted({int(orbit_labels[index]) for index in source_indices})
    parent_orbit_label_set = set(parent_orbit_labels)
    expanded_indices = [
        int(index)
        for index, orbit_label in enumerate(orbit_labels[:site_count])
        if int(orbit_label) in parent_orbit_label_set
    ]
    source_index_set = set(source_indices)
    included_zero_moment_indices = [
        int(index)
        for index in expanded_indices
        if index not in source_index_set
    ]
    return expanded_indices, {
        "mode": "sg_orbit_closure_of_nonzero_moment_sites",
        "source_nonzero_moment_indices": source_indices,
        "included_zero_moment_indices": included_zero_moment_indices,
        "parent_sg_orbit_labels": parent_orbit_labels,
    }


def _site_dof_rows(equivalence_classes_spin, dof_by_representative, constraints):
    rows = []
    for info, constraint in zip(equivalence_classes_spin, constraints):
        representative = int(info["representative_index"])
        rows.append(
            {
                "representative_index": representative,
                "class_indices": [int(index) for index in info["class_indices"]],
                "dof": int(dof_by_representative[representative]),
                "constraints": list(constraint),
            }
        )
    return rows


def _max_site_dof(dof_rows):
    if not dof_rows:
        return None
    return max(int(row["dof"]) for row in dof_rows)


def _cell_lattice_volume(cell: CrystalCell) -> float:
    lattice, _positions, _types = cell.to_spglib(mag=False)
    return abs(float(np.linalg.det(np.asarray(lattice, dtype=float))))


def _magnetic_to_nonmagnetic_primitive_cell_expansion(cell: CrystalCell):
    nonmagnetic_primitive_cell, _transform = cell.get_primitive_structure(magnetic=False)
    magnetic_volume = _cell_lattice_volume(cell)
    nonmagnetic_volume = _cell_lattice_volume(nonmagnetic_primitive_cell)
    if nonmagnetic_volume < 1e-12:
        return None
    ratio = magnetic_volume / nonmagnetic_volume
    rounded = int(round(ratio))
    if np.isclose(ratio, rounded, atol=1e-8, rtol=0.0):
        return rounded
    return ratio


def _total_site_dof(magnetic_wp_dof_rows, key):
    total = 0
    has_value = False
    for row in magnetic_wp_dof_rows:
        value = row.get(key)
        if value is None:
            continue
        total += int(value)
        has_value = True
    return total if has_value else None


def _site_dof_maps(dof_rows):
    dof_by_site = {}
    constraints_by_site = {}
    representative_by_site = {}
    for row in dof_rows:
        representative = int(row["representative_index"])
        constraints = list(row.get("constraints") or [])
        for index in row.get("class_indices") or []:
            site_index = int(index)
            dof_by_site[site_index] = int(row["dof"])
            constraints_by_site[site_index] = constraints
            representative_by_site[site_index] = representative
    return dof_by_site, constraints_by_site, representative_by_site


def _build_magnetic_wp_dof_rows(
    wp_sg,
    wp_ssg,
    wp_msg,
    cell,
    atom_types_dict,
    magnetic_indices,
    *,
    ssg_dof_by_site,
    ssg_constraints_by_site,
    ssg_representative_by_site,
    msg_dof_by_site,
    msg_constraints_by_site,
    msg_representative_by_site,
):
    rows_by_key = {}
    for index in sorted(int(value) for value in magnetic_indices):
        if index >= min(len(wp_sg), len(wp_ssg), len(wp_msg), len(cell[2])):
            continue
        element = atom_types_dict[int(cell[2][index])]
        ssg_dof = ssg_dof_by_site.get(index)
        msg_dof = msg_dof_by_site.get(index)
        ssg_constraints = tuple(ssg_constraints_by_site.get(index) or ())
        msg_constraints = tuple(msg_constraints_by_site.get(index) or ())
        key = (
            element,
            wp_sg[index][0],
            int(wp_sg[index][1]),
            wp_ssg[index][0],
            int(wp_ssg[index][1]),
            ssg_dof,
            ssg_constraints,
            wp_msg[index][0],
            int(wp_msg[index][1]),
            msg_dof,
            msg_constraints,
        )
        if key not in rows_by_key:
            rows_by_key[key] = {
                "element": element,
                "site_indices": [],
                "site_count": 0,
                "sg_wyckoff": wp_sg[index][0],
                "sg_wyckoff_index": int(wp_sg[index][1]),
                "ssg_wyckoff": wp_ssg[index][0],
                "ssg_wyckoff_with_dof": _format_wp_with_site_dof(
                    wp_ssg[index][0],
                    ssg_dof,
                ),
                "ssg_wyckoff_index": int(wp_ssg[index][1]),
                "ssg_site_dof": None if ssg_dof is None else int(ssg_dof),
                "ssg_orbit_total_dof": None if ssg_dof is None else int(ssg_dof),
                "ssg_constraints": list(ssg_constraints),
                "ssg_representative_index": ssg_representative_by_site.get(index),
                "msg_wyckoff": wp_msg[index][0],
                "msg_wyckoff_with_dof": _format_wp_with_site_dof(
                    wp_msg[index][0],
                    msg_dof,
                ),
                "msg_wyckoff_index": int(wp_msg[index][1]),
                "msg_site_dof": None if msg_dof is None else int(msg_dof),
                "msg_orbit_total_dof": None if msg_dof is None else int(msg_dof),
                "msg_constraints": list(msg_constraints),
                "msg_representative_index": msg_representative_by_site.get(index),
            }
        rows_by_key[key]["site_indices"].append(index)

    rows = list(rows_by_key.values())
    for row in rows:
        row["site_indices"].sort()
        row["site_count"] = len(row["site_indices"])
    rows.sort(
        key=lambda row: (
            str(row["element"]),
            row["sg_wyckoff"],
            row["ssg_wyckoff"],
            row["msg_wyckoff"],
            row["site_indices"],
        )
    )
    return rows


def _build_magnetic_site_summary(
    cell: CrystalCell,
    ssg: SpinSpaceGroup,
    identify_info: str,
    tol_cfg: Tolerances,
    *,
    setting: str,
):
    sg_dataset = get_symmetry_dataset(cell.to_spglib(), symprec=tol_cfg.space)
    site_count = len(cell.to_spglib(mag=True)[1])
    nonzero_moment_indices = [] if cell.magnetic_atom_indices is None else list(cell.magnetic_atom_indices)
    magnetic_indices, magnetic_atom_selection = _expand_magnetic_indices_by_sg_orbit(
        sg_dataset,
        nonzero_moment_indices,
        site_count,
    )
    ssg_dataset = get_G0_dataset_for_cell(
        ssg.G0_ops,
        cell.to_spglib(mag=True),
        tol_cfg.space,
    )
    oriented_ssg = _ossg_oriented_spin_frame_ssg(ssg, cell)
    msg_ops = list(oriented_ssg.msg_ops)

    msg_dataset = None
    if msg_ops:
        msg_dataset = get_G0_dataset_for_cell(
            [[op[1], op[2]] for op in msg_ops],
            cell.to_spglib(mag=True),
            tol_cfg.space,
        )

    _ssg_magnetic_indices, _ssg_classes, ssg_dof, ssg_spin_classes, ssg_constraints = (
        get_spin_wyckoff(
            cell,
            ssg.ops,
            atol=tol_cfg.m_matrix_tol,
            magnetic_indices=magnetic_indices,
        )
    )
    ssg_dof_rows = _site_dof_rows(ssg_spin_classes, ssg_dof, ssg_constraints)
    (
        ssg_dof_by_site,
        ssg_constraints_by_site,
        ssg_representative_by_site,
    ) = _site_dof_maps(ssg_dof_rows)

    msg_dof_rows = []
    msg_dof_by_site = {}
    msg_constraints_by_site = {}
    msg_representative_by_site = {}
    if msg_ops:
        _msg_magnetic_indices, _msg_classes, msg_dof, msg_spin_classes, msg_constraints = (
            get_spin_wyckoff(
                cell,
                msg_ops,
                atol=tol_cfg.m_matrix_tol,
                magnetic_indices=magnetic_indices,
            )
        )
        msg_dof_rows = _site_dof_rows(msg_spin_classes, msg_dof, msg_constraints)
        (
            msg_dof_by_site,
            msg_constraints_by_site,
            msg_representative_by_site,
        ) = _site_dof_maps(msg_dof_rows)

    wp_extended_sg = get_wp_from_dataset(sg_dataset, max=False)
    wp_extended_ssg = _get_wp_for_original_sites(ssg_dataset, site_count)
    wp_extended_msg = [] if msg_dataset is None else _get_wp_for_original_sites(msg_dataset, site_count)
    cell_spglib = cell.to_spglib(mag=True)
    if wp_extended_msg:
        magnetic_wp_dof_rows = _build_magnetic_wp_dof_rows(
            wp_extended_sg,
            wp_extended_ssg,
            wp_extended_msg,
            cell_spglib,
            cell.atom_types_to_symbol,
            magnetic_indices,
            ssg_dof_by_site=ssg_dof_by_site,
            ssg_constraints_by_site=ssg_constraints_by_site,
            ssg_representative_by_site=ssg_representative_by_site,
            msg_dof_by_site=msg_dof_by_site,
            msg_constraints_by_site=msg_constraints_by_site,
            msg_representative_by_site=msg_representative_by_site,
        )
    else:
        magnetic_wp_dof_rows = []

    cell_expansion = _magnetic_to_nonmagnetic_primitive_cell_expansion(cell)

    magnetic_orbits_msg = (
        None
        if msg_dataset is None
        else _magnetic_orbit_count_from_dataset(msg_dataset, magnetic_indices)
    )

    return {
        "status": "ok",
        "setting": setting,
        "SG": {
            "number": int(sg_dataset.number),
            "symbol": str(sg_dataset.international),
            "hall_number": int(sg_dataset.hall_number),
            "choice": getattr(sg_dataset, "choice", None),
        },
        "cell_expansion": cell_expansion,
        "cell_expansion_source": "magnetic_primitive_volume/nonmagnetic_primitive_volume",
        "cell_expansion_transform": None,
        "ssg_index": identify_info,
        "magnetic_atom_count": len(magnetic_indices),
        "magnetic_atom_indices": [int(index) for index in magnetic_indices],
        "magnetic_atom_selection": magnetic_atom_selection,
        "magnetic_atom_selection_mode": magnetic_atom_selection["mode"],
        "nonzero_moment_atom_count": len(nonzero_moment_indices),
        "nonzero_moment_atom_indices": [int(index) for index in nonzero_moment_indices],
        "zero_moment_magnetic_atom_count": len(
            magnetic_atom_selection["included_zero_moment_indices"]
        ),
        "zero_moment_magnetic_atom_indices": magnetic_atom_selection["included_zero_moment_indices"],
        "n_magnetic_orbits_sg": _magnetic_orbit_count_from_dataset(sg_dataset, magnetic_indices),
        "n_magnetic_orbits_ssg": _magnetic_orbit_count_from_dataset(ssg_dataset, magnetic_indices),
        "n_magnetic_orbits_msg": magnetic_orbits_msg,
        "max_magnetic_site_dof_ssg": _max_site_dof(ssg_dof_rows),
        "max_magnetic_site_dof_msg": _max_site_dof(msg_dof_rows),
        "total_magnetic_site_dof_ssg": _total_site_dof(
            magnetic_wp_dof_rows,
            "ssg_orbit_total_dof",
        ),
        "total_magnetic_site_dof_msg": _total_site_dof(
            magnetic_wp_dof_rows,
            "msg_orbit_total_dof",
        ),
        "ssg_magnetic_site_dofs": ssg_dof_rows,
        "msg_magnetic_site_dofs": msg_dof_rows,
        "magnetic_wp_dof_rows": magnetic_wp_dof_rows,
        "msg_operation_count": len(msg_ops),
        "msg_available": bool(msg_ops),
    }


def _identify_ssg_index_details(file_name,ssg_primitive:SpinSpaceGroup,tol = 0.001):
    """
    only for G0std_nofrac
    """
    from findspingroup.data.SG_SYMBOL import SGgeneratorDict
    from findspingroup.data.PG_SYMBOL import PG_SCH_TO_ID_INDEX
    from findspingroup.core.identify_index.functions import make_4d_matrix
    from findspingroup.core.identify_index.functions.find_ssg_reduce import find_ssg_transformation
    from findspingroup.core.identify_index.functions.get_stand_trans import get_stand_trans

    def _normalized_direction(vector):
        direction = np.asarray(vector, dtype=float).reshape(-1)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Cannot normalize a zero-length identify-index direction.")
        return direction / norm

    def _eigen_direction(matrix, eigenvalue):
        eigenvalues, eigenvectors = np.linalg.eig(np.asarray(matrix, dtype=float))
        for idx, value in enumerate(eigenvalues):
            if np.isclose(value, eigenvalue, atol=tol):
                return _normalized_direction(eigenvectors[:, idx].real)
        raise ValueError(
            f"Cannot find eigen-direction with eigenvalue {eigenvalue} for "
            f"identify-index matrix {np.asarray(matrix, dtype=float).tolist()}."
        )

    def _resolve_order_two_coplanar_suffix(ssg_std_nofrac):
        if ssg_std_nofrac.conf != 'Coplanar' or len(ssg_std_nofrac.n_spin_part_point_ops) != 2:
            return None
        spin_only_mirror = next(
            op[0]
            for op in ssg_std_nofrac.sog
            if not np.allclose(op[0], np.eye(3), atol=tol)
        )
        spin_twofold = next(
            op[0]
            for op in ssg_std_nofrac.nssg
            if not np.allclose(op[0], np.eye(3), atol=tol)
        )
        mirror_normal = _eigen_direction(spin_only_mirror, -1.0)
        twofold_axis = _eigen_direction(spin_twofold, 1.0)
        alignment = abs(float(np.dot(mirror_normal, twofold_axis)))
        axis_tol = max(tol, 1e-3)
        if alignment >= 1.0 - axis_tol:
            return 'P1'
        if alignment <= axis_tol:
            return 'P2'
        raise ValueError(
            "Ambiguous P1/P2 classification for "
            f"{file_name}: |dot(mirror_normal, twofold_axis)|={alignment:.6f}."
        )

    def _canonicalize_axis_sign(direction):
        normalized = _normalized_direction(direction)
        for value in normalized:
            if abs(value) < max(tol, 1e-6):
                continue
            return normalized if value > 0 else -normalized
        return normalized

    def _classify_axis_aligned_mirror(matrix):
        matrix = np.asarray(matrix, dtype=float)
        if np.allclose(matrix, np.diag([1.0, 1.0, -1.0]), atol=tol):
            return "Mz"
        if np.allclose(matrix, np.diag([1.0, -1.0, 1.0]), atol=tol):
            return "My"
        if np.allclose(matrix, np.diag([-1.0, 1.0, 1.0]), atol=tol):
            return "Mx"
        raise ValueError(
            f"Unsupported Coplanar+D2 target spin-only matrix {matrix.tolist()} for {file_name}."
        )

    def _canonical_branch_spin_transform(target_spin_only_matrix):
        target_label = _classify_axis_aligned_mirror(target_spin_only_matrix)
        if target_label == "Mz":
            return np.eye(3), target_label
        if target_label == "My":
            return np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=float,
            ), target_label
        if target_label == "Mx":
            return np.array(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=float,
            ), target_label
        raise ValueError(
            f"Unsupported Coplanar+D2 target spin-only label {target_label!r} for {file_name}."
        )

    def _build_coplanar_d2_spin_normalization(current_ops):
        temp_group = IdentifyNoFracGroup(current_ops, conf='Coplanar', tol=tol)
        spin_only_mirror = _select_preferred_candidate(
            [
                op
                for op in temp_group.sog
                if not np.allclose(op[0], np.eye(3), atol=tol)
            ],
            [],
        )
        if spin_only_mirror is None:
            raise ValueError(
                f"Cannot find nontrivial spin-only mirror for Coplanar+D2 identify branch in {file_name}."
            )
        mirror_normal = _canonicalize_axis_sign(_eigen_direction(spin_only_mirror[0], -1.0))

        in_plane_axis = None
        for op in sorted(
            [
                op
                for op in temp_group.nssg
                if not np.allclose(op[0], np.eye(3), atol=tol)
            ],
            key=_operation_candidate_sort_key,
        ):
            candidate_axis = _eigen_direction(op[0], 1.0)
            projected = candidate_axis - float(np.dot(candidate_axis, mirror_normal)) * mirror_normal
            if np.linalg.norm(projected) > max(tol, 1e-4):
                in_plane_axis = _canonicalize_axis_sign(projected)
                break

        if in_plane_axis is None:
            for basis_vector in np.eye(3):
                projected = basis_vector - float(np.dot(basis_vector, mirror_normal)) * mirror_normal
                if np.linalg.norm(projected) > max(tol, 1e-4):
                    in_plane_axis = _canonicalize_axis_sign(projected)
                    break
        if in_plane_axis is None:
            raise ValueError(
                f"Cannot determine in-plane reference axis for Coplanar+D2 identify branch in {file_name}."
            )

        z_axis = mirror_normal
        x_axis = _normalized_direction(in_plane_axis)
        y_axis = _normalized_direction(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < max(tol, 1e-4):
            raise ValueError(
                f"Degenerate Coplanar+D2 spin normalization frame in {file_name}."
            )
        y_axis = _canonicalize_axis_sign(y_axis)
        frame = np.vstack([x_axis, y_axis, z_axis])
        if np.linalg.det(frame) < 0:
            y_axis = -y_axis
            frame = np.vstack([x_axis, y_axis, z_axis])
        return frame, mirror_normal

    def _transform_ops_preserving_integer_shifts(ops, transformation_matrix, origin_shift, frac=True, all_trans=True):
        transformation_matrix = np.asarray(transformation_matrix, dtype=float)
        origin_shift = np.asarray(origin_shift, dtype=float)
        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        if frac:
            lattice_shifts = integer_points_in_new_cell(transformation_matrix_inv.T)
        else:
            lattice_shifts = [np.zeros(3)]
        if not all_trans:
            lattice_shifts = [np.zeros(3)]

        transformed_ops = []
        for op in ops:
            spin_rotation = np.asarray(op[0], dtype=float)
            real_rotation = np.asarray(op[1], dtype=float)
            translation = np.asarray(op[2], dtype=float)
            for lattice_shift in lattice_shifts:
                lifted_translation = translation + np.asarray(lattice_shift, dtype=float)
                new_rotation = transformation_matrix @ real_rotation @ transformation_matrix_inv
                if frac:
                    new_translation = normalize_vector_to_zero(
                        ((np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ lifted_translation),
                        atol=1e-4,
                    )
                else:
                    new_translation = (
                        (np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ lifted_translation
                    )
                transformed_ops.append(
                    SpinSpaceGroupOperation(
                        spin_rotation,
                        new_rotation,
                        new_translation,
                    )
                )
        return transformed_ops

    def _transform_spin_ops_preserving_order(ops, spin_transformation_matrix):
        spin_transformation_matrix = np.asarray(spin_transformation_matrix, dtype=float)
        spin_transformation_matrix_inv = np.linalg.inv(spin_transformation_matrix)
        transformed_ops = []
        for op in ops:
            transformed_ops.append(
                SpinSpaceGroupOperation(
                    spin_transformation_matrix @ np.asarray(op[0], dtype=float) @ spin_transformation_matrix_inv,
                    np.asarray(op[1], dtype=float),
                    np.asarray(op[2], dtype=float),
                )
            )
        return transformed_ops

    def _operation_candidate_sort_key(op):
        spin_rotation = np.asarray(op[0], dtype=float)
        nontrivial_spin = not np.allclose(spin_rotation, np.eye(3), atol=tol)
        spin_distance = float(np.linalg.norm(spin_rotation - np.eye(3)))
        spin_signature = tuple(np.round(spin_rotation, 6).flatten())
        return (
            0 if nontrivial_spin else 1,
            -round(spin_distance, 6),
            spin_signature,
        )

    def _select_preferred_candidate(exact_candidates, equivalent_candidates):
        candidates = exact_candidates or equivalent_candidates
        if not candidates:
            return None
        return sorted(candidates, key=_operation_candidate_sort_key)[0]

    def _match_name_generators(sg_num: int, canonical_nssg_ops, pure_translation_lattice):
        """
        identify name_maps follow the canonical point-group generator
        convention used by the identify database / point-group table.
        """
        sg_info = SGgeneratorDict[sg_num]
        generators = []
        for ind in range((len(sg_info) - 1) // 2):
            gen_rot, gen_t = eval(sg_info[2 * ind + 2])
            gen_t = np.array(gen_t, dtype=float)
            gen_rot = np.array(gen_rot).reshape((3, 3))
            generators.append([gen_rot, gen_t])

        matched_generators = []
        for gen_rot, gen_t in generators:
            exact_candidates = []
            equivalent_candidates = []
            for op in canonical_nssg_ops:
                if not np.allclose(gen_rot, op[1], atol=tol):
                    continue
                if _exact_translation_distance(gen_t, op[2]) < tol:
                    exact_candidates.append(op)
                    continue
                if _translations_equivalent_mod_pure_translations(
                    gen_t,
                    op[2],
                    pure_translation_lattice,
                    tol,
                ):
                    equivalent_candidates.append(op)
            preferred = _select_preferred_candidate(exact_candidates, equivalent_candidates)
            if preferred is None:
                raise ValueError(
                    f"Cannot find canonical identify generator {(gen_rot.tolist(), gen_t.tolist())} "
                    "in G0std_nofrac.nssg."
                )
            matched_generators.append(SpinSpaceGroupOperation(preferred[0], preferred[1], preferred[2]))
        return matched_generators

    def _match_translation_generators(translation_source_ops, lattice_translations):
        """
        identify translation_maps encode the SSG operations corresponding to the
        three nofrac lattice translations [1,0,0], [0,1,0], [0,0,1].
        """
        generators_trans = [
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        ]
        matched_translations = []
        for gen_t in generators_trans:
            exact_candidates = []
            equivalent_candidates = []
            for op in translation_source_ops:
                if not np.allclose(op[1], np.eye(3), atol=tol):
                    continue
                translation = np.asarray(op[2], dtype=float)
                if _exact_translation_distance(gen_t, translation) < tol:
                    exact_candidates.append(op)
                    continue
                for lattice_t in lattice_translations:
                    if _exact_translation_distance(gen_t, translation - lattice_t) < tol:
                        equivalent_candidates.append(op)
                        break
            preferred = _select_preferred_candidate(exact_candidates, equivalent_candidates)
            if preferred is None:
                matched_translations.append(SpinSpaceGroupOperation(np.eye(3), np.eye(3), gen_t))
            else:
                matched_translations.append(SpinSpaceGroupOperation(preferred[0], preferred[1], gen_t))
        return matched_translations
    spin_T = np.eye(3)
    identify_ops_nofrac = _transform_ops_preserving_integer_shifts(
        ssg_primitive._input_ops,
        ssg_primitive.transformation_to_G0std,
        ssg_primitive.origin_shift_to_G0std,
    )
    identify_ops_nofrac = _transform_ops_preserving_integer_shifts(
        identify_ops_nofrac,
        ssg_primitive.transformation_to_G0std_id @ np.linalg.inv(ssg_primitive.transformation_to_G0std),
        np.array([0, 0, 0]),
        frac=False,
    )
    identify_ops_nofrac = _transform_spin_ops_preserving_order(
        identify_ops_nofrac,
        np.linalg.inv(ssg_primitive.n_spin_part_std_transformation),
    )
    G0_num = ssg_primitive.G0_num
    L0_num = ssg_primitive.L0_num
    it = ssg_primitive.it
    ik = ssg_primitive.ik
    nsspg_order = len(ssg_primitive.n_spin_part_point_ops)
    use_222_contract = (
        ssg_primitive.conf == 'Coplanar'
        and ssg_primitive.n_spin_part_point_group_symbol_s == 'D2'
        and has_coplanar_222_lookup_group((L0_num, G0_num), (nsspg_order, it, ik))
    )
    coplanar_d2_spin_normalization = np.eye(3)
    if use_222_contract:
        coplanar_d2_spin_normalization, _ = _build_coplanar_d2_spin_normalization(identify_ops_nofrac)
        identify_ops_nofrac = _transform_spin_ops_preserving_order(
            identify_ops_nofrac,
            coplanar_d2_spin_normalization,
        )
    identify_nofrac_group = IdentifyNoFracGroup(
        identify_ops_nofrac,
        conf=ssg_primitive.conf,
        tol=tol,
    )
    identify_generator_source_ops = list(identify_nofrac_group.nssg)
    identify_translation_lattice = identify_nofrac_group.pure_translations
    if it * ik != nsspg_order:
        raise ValueError(
            "Inconsistent NSSPG invariants for "
            f"{file_name}: it*ik={it * ik}, |nsspg|={nsspg_order}, "
            f"it={it}, ik={ik}, spin_pg={ssg_primitive.n_spin_part_point_group_symbol_s}, "
            f"G0={G0_num}, L0={L0_num}."
        )
    coplanar_suffix = _resolve_order_two_coplanar_suffix(identify_nofrac_group)
    n_spin_part_pg_symbol = ssg_primitive.n_spin_part_point_group_symbol_s
    if n_spin_part_pg_symbol not in PG_SCH_TO_ID_INDEX:
        raise ValueError(
            "Cannot identify point-group map number for "
            f"{file_name}: n-spin part point group {n_spin_part_pg_symbol!r} "
            "is not in the identify-index table."
        )
    pg = PG_SCH_TO_ID_INDEX[n_spin_part_pg_symbol] # map to identify-pg list
    name_generators = _match_name_generators(
        G0_num,
        identify_generator_source_ops,
        identify_translation_lattice,
    )
    translation_generators = _match_translation_generators(
        identify_generator_source_ops,
        identify_translation_lattice,
    )
    generators_hm = [[(spin_T@ i[0]@np.linalg.inv(spin_T)).round(5).tolist(),[i[1].round(5).tolist(),i[2].round(5).tolist()]] for i in name_generators]
    generators_lattice =[[(spin_T@ i[0]@np.linalg.inv(spin_T)).round(5).tolist(),[i[1].round(5).tolist(),i[2].round(5).tolist()]] for i in  translation_generators]
    transformation_G0std_to_L0std = [ssg_primitive.G0std_L0std_transformation.round(5),ssg_primitive.G0std_L0std_origin_shift.round(5)]

    transformation_L0std_to_G0std = np.linalg.inv(np.block([[transformation_G0std_to_L0std[0],transformation_G0std_to_L0std[1].reshape(3,1)],[np.zeros((1,3)), np.ones((1,1))]]))
    transformation_L0std_to_G0std = [transformation_L0std_to_G0std[:3,:3].tolist(),transformation_L0std_to_G0std[:3,3:4].reshape(3).tolist()]

    data = {
        'filename':file_name,
        'L0_id':L0_num,
        'G0_id':G0_num,
        't_index':it,
        'k_index':ik,
        'point_group_id':pg,
        'name_maps':generators_hm,
        'translation_maps':generators_lattice,
        'transformation_matrix':transformation_L0std_to_G0std
    }
    L0_id, G0_id, it, ik, iso,T = data['L0_id'], data['G0_id'], data['t_index'], data['k_index'], data['point_group_id'],  data['transformation_matrix']
    name_maps, translation_maps = data['name_maps'], data['translation_maps']

    if ssg_primitive.conf == 'Collinear':
        last_index = '.L'
    elif ssg_primitive.conf == 'Coplanar':
        last_index = f'.{coplanar_suffix}' if coplanar_suffix is not None else '.P'
    else:
        last_index = ''
    identify_reduction = find_ssg_transformation(
        L0_id,
        G0_id,
        it,
        ik,
        iso,
        make_4d_matrix(T),
        tol=tol,
        use_222_contract=use_222_contract,
    )
    try:
        map_result = get_stand_trans(
            L0_id,
            G0_id,
            it,
            ik,
            iso,
            T,
            name_maps,
            translation_maps,
            tol=tol,
            use_222_contract=use_222_contract,
            return_map_info=use_222_contract,
        )
    except IndexError as exc:
        context = {
            "file_name": file_name,
            "L0_id": L0_id,
            "G0_id": G0_id,
            "t_index": it,
            "k_index": ik,
            "point_group_id": iso,
            "configuration": ssg_primitive.conf,
            "n_spin_part_point_group_symbol_s": n_spin_part_pg_symbol,
            "nsspg_order": nsspg_order,
            "use_222_contract": use_222_contract,
            "transformation_matrix": T,
            "name_maps": name_maps,
            "translation_maps": translation_maps,
        }
        raise ValueError(
            "Identify-index adapter call failed with an internal IndexError "
            "before returning a standard-generator map. This is an exposed "
            "adapter/input consistency error; audit the context before "
            "modifying identify-index internals. context="
            f"{json.dumps(context, cls=NumpyEncoder, sort_keys=True)}"
        ) from exc
    if use_222_contract:
        map_num, trans1, trans2, identify_map_info = map_result
        lookup_entry = get_coplanar_222_lookup_entry(
            (L0_id, G0_id),
            (nsspg_order, it, ik),
            map_num,
        )
        if lookup_entry is None:
            raise ValueError(
                "Missing Coplanar+D2 Excel-backed lookup entry for "
                f"(L0,G0)=({L0_id},{G0_id}), (total,t,k)=({nsspg_order},{it},{ik}), map_num={map_num}."
            )
        last_index = f".{lookup_entry['configuration_suffix']}"
        final_index = str(lookup_entry["final_index"])
        final_index_parts = final_index.split(".")
        if len(final_index_parts) != 5:
            raise ValueError(
                f"Unexpected Coplanar+D2 lookup index format {final_index!r} for {file_name}."
            )
        map_num = int(final_index_parts[3])
        trans2_raw = np.asarray(trans2, dtype=float)
        q_transform = np.asarray(coplanar_d2_spin_normalization, dtype=float)
        branch_spin_transform, branch_spin_target = _canonical_branch_spin_transform(
            lookup_entry["spin_only_matrix"]
        )
        trans2 = branch_spin_transform @ trans2_raw @ q_transform
        final_index_string = final_index
    else:
        map_num, trans1, trans2 = map_result
        final_index_string = f'{G0_id}.{L0_id}.{ik}.{map_num}{last_index}'

    public_space_transform, public_space_shift = _identify_affine_4x4_to_setting_transform(trans1)

    return {
        'filename': file_name,
        'index': final_index_string,
        'configuration': ssg_primitive.conf,
        'G0_id': G0_id,
        'L0_id': L0_id,
        't_index': it,
        'k_index': ik,
        'point_group_id': pg,
        'identify_cell_size': identify_reduction.get('cell_size'),
        'equivalent_map_index': map_num,
        'configuration_suffix': last_index.lstrip('.'),
        'name_maps': name_maps,
        'translation_maps': generators_lattice,
        'transformation_matrix': transformation_L0std_to_G0std,
        'space_group_transformation': [
            np.asarray(public_space_transform, dtype=float).tolist(),
            np.asarray(public_space_shift, dtype=float).tolist(),
        ],
        'space_group_transformation_raw_4x4': np.asarray(trans1, dtype=float).tolist(),
        'point_group_transformation': np.asarray(trans2, dtype=float).tolist(),
        'point_group_transformation_raw': (
            np.asarray(trans2_raw, dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_q_transform': (
            np.asarray(q_transform, dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_b_transform': (
            np.asarray(branch_spin_transform, dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_target_spin_only_matrix': (
            np.asarray(lookup_entry["spin_only_matrix"], dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_target_spin_only_label': (
            branch_spin_target if use_222_contract else None
        ),
        'equivalent_map_resolution': 'database',
        'canonical_transformations_available': True,
        'special_record': None,
    }


def _identify_ssg_index(file_name,ssg_primitive:SpinSpaceGroup,tol = 0.001):
    return _identify_ssg_index_details(file_name, ssg_primitive, tol=tol)['index']





def get_G0_dataset_for_cell(space_group_operations, cell, symprec):
    # weirdSite = np.array([0.4275710, 0.591580, 0.233338700])
    weirdSite = np.array([0.1715870, 0.27754210, 0.737388700])
    # weirdSite = np.array([0.1, 0.2, 0.7])
    # weirdSite = np.array([0,0,0])
    defaultpos = [i for i in cell[1]]
    defaulttypes = [i for i in cell[2]]
    # print(defaulttypes)
    typesForGerator = [max(defaulttypes) + 1]
    # print(typesForGerator)
    generatePosition = [weirdSite]
    for i in space_group_operations:
        # print(i)
        temp = normalize_vector_to_zero(i[0]@weirdSite+i[1] ,atol=1e-8)
        # print(temp)
        if not any(np.allclose(temp, j, atol=1e-4) for j in generatePosition):
            generatePosition.append(temp)
            typesForGerator.append(max(defaulttypes) + 1)
    cells = (cell[0], defaultpos + generatePosition, defaulttypes + typesForGerator)
    space_group_dataset =get_symmetry_dataset(cells, symprec=symprec)
    if space_group_dataset is None:
        raise SpaceToleranceDegeneracyError(
            "spglib could not identify the G0 dataset from accepted operations "
            "under the current space_tol; the spatial tolerance has made the "
            "operation orbit inconsistent with the decorated magnetic cell."
        )
    if space_group_dataset.number in SG_HALL_MAPPING:
        space_group_dataset =get_symmetry_dataset(cells, symprec=symprec, hall_number=SG_HALL_MAPPING[space_group_dataset.number])
        if space_group_dataset is None:
            raise SpaceToleranceDegeneracyError(
                "spglib could not identify the mapped G0 dataset from accepted "
                "operations under the current space_tol."
            )

    return space_group_dataset

#------------------
# Wyckoff
def _dataset_wyckoff_orbits(dataset):
    crystallographic_orbits = getattr(dataset, "crystallographic_orbits", None)
    if crystallographic_orbits is not None:
        return crystallographic_orbits
    return dataset.equivalent_atoms


def get_wp_from_dataset(dataset,max=True):
    temp_eq = {}
    first_index = {}
    last_index = 0
    orbit_labels = _dataset_wyckoff_orbits(dataset)
    for ind, eq_label in enumerate(orbit_labels):
        if eq_label not in temp_eq:
            temp_eq[eq_label] = 1
            first_index[eq_label] = ind
            last_index = ind
        else:
            temp_eq[eq_label] += 1
    di = {key:str(value)+ dataset.wyckoffs[first_index[key]]for key,value in temp_eq.items()}

    if max:
        wp = [(di[i],i) for i in orbit_labels[:last_index]]
    else:
        wp = [(di[i],i) for i in orbit_labels]
    return wp

def get_msg_from_ossg(ossg_ops,tol=0.01):
    """
    Get magnetic space group operations from oriented spin space group operations.

    Parameters:
    ossg_ops (list): A list of oriented spin space group operations.
    tol (float): Tolerance for numerical comparisons.

    Returns:
    list: A list of operations satisfying the MSG condition Rs = +/- Rr.
    """
    msg_ops = []
    for op in ossg_ops:
        if op.magnetic_time_reversal(atol=tol) is not None:
            msg_ops.append(op)
    return msg_ops


def _find_spin_group_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    source_metadata: dict | None = None,
    parser_atol: float | None = None,
    input_spin_setting: str = "in_lattice",
    calculation_mode: str | None = "3d",
    vacuum_axis: str | None = "c",
) -> MagSymmetryResult:
    input_lattice_for_cell = lattice_factors
    input_positions_for_cell = positions
    input_lattice_array = np.asarray(lattice_factors, dtype=float)
    if input_lattice_array.shape == (3, 3):
        input_lattice_matrix = input_lattice_array
    else:
        a, b, c, alpha, beta, gamma = input_lattice_array.reshape(6).tolist()
        input_lattice_matrix = np.asarray(
            calculate_vector_coordinates_from_latticefactors(a, b, c, alpha, beta, gamma),
            dtype=float,
        )
    (
        quasi2d_lattice_for_cell,
        quasi2d_positions_for_cell,
        quasi2d_vacuum_padding,
    ) = prepare_quasi2d_input_cell(
        input_lattice_matrix,
        positions,
        calculation_mode=calculation_mode,
        vacuum_axis=vacuum_axis,
    )
    if quasi2d_vacuum_padding is not None:
        input_lattice_for_cell = quasi2d_lattice_for_cell
        input_positions_for_cell = quasi2d_positions_for_cell
    input_cell = CrystalCell(
        input_lattice_for_cell,
        input_positions_for_cell,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    magnetic_primitive_cell: CrystalCell
    magnetic_primitive_cell,Tmatrix_Tp_input__p_primitive = input_cell.get_primitive_structure(magnetic=True)
    identify_result = identify_spin_space_group_result(
        magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    ssg_primitive: SpinSpaceGroup = identify_result.ssg
    _assert_ssg_ops_consistency(
        "input magnetic primitive",
        ssg_primitive,
        tol=tol_cfg,
    )
    input_space_group = identify_result.input_space_group
    input_space_group_number = None if input_space_group is None else input_space_group.number
    input_space_group_symbol = None if input_space_group is None else input_space_group.symbol
    input_space_group_basis_or_setting = (
        None if input_space_group is None else input_space_group.basis_or_setting
    )

    primitive_ossg_for_phase = _ossg_oriented_spin_frame_ssg(
        ssg_primitive,
        magnetic_primitive_cell,
    )

    magnetic_phase_payload = classify_magnetic_phase(
        conf=ssg_primitive.conf,
        full_spin_part_point_group_hm=ssg_primitive.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=ssg_primitive.spin_part_point_group_symbol_s,
        net_moment=magnetic_primitive_cell.net_moment,
        net_moment_tol=tol_cfg.moment,
        mpg_identifier=primitive_ossg_for_phase.mpg_num,
        is_ss_gp=ssg_primitive.is_spinsplitting[-1],
    )
    magnetic_phase = magnetic_phase_payload['phase']
    magnetic_phase_base = magnetic_phase_payload['base_phase']
    magnetic_phase_modifier = magnetic_phase_payload['modifier']
    magnetic_phase_details = magnetic_phase_payload['details']
    ss_w_soc = spin_splitting_w_soc(ssg_primitive)
    ahc_w_soc = is_ahc(primitive_ossg_for_phase.mpg_num)
    ss_wo_soc = magnetic_phase_payload['spin_splitting_without_soc']
    alter = magnetic_phase_payload['is_alter']


    transformation_input_to_primitive = (
        Tmatrix_Tp_input__p_primitive,
        np.zeros(3),
    )
    input_cell_cartesian = _cartesianized_input_cell(input_cell)
    identify_index_details = None
    identify_info = None
    try:
        identify_index_details = _identify_ssg_index_details(
            source_name,
            ssg_primitive,
            tol=tol_cfg.space,
        )
        identify_info = identify_index_details['index']
        _assert_ssg_ops_consistency(
            "input magnetic primitive",
            ssg_primitive,
            tol=tol_cfg,
            identify_index_details=identify_index_details,
        )
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        identify_info = _handle_missing_identify_index(source_name, exc)
    input_magnetic_primitive_poscar = magnetic_primitive_cell.to_poscar(source_name)
    raw_transformation_primitive_to_G0std = (
        np.asarray(ssg_primitive.transformation_to_G0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_G0std, dtype=float),
    )
    raw_transformation_primitive_to_L0std = (
        np.asarray(ssg_primitive.transformation_to_L0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_L0std, dtype=float),
    )
    legacy_transformation_primitive_to_acc_primitive = (
        np.asarray(ssg_primitive.acc_primitive_trans, dtype=float),
        np.asarray(ssg_primitive.acc_primitive_origin_shift, dtype=float),
    )
    legacy_acc_magnetic_primitive_cell = magnetic_primitive_cell.transform(
        *legacy_transformation_primitive_to_acc_primitive
    )
    legacy_transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        legacy_transformation_primitive_to_acc_primitive[0],
        legacy_transformation_primitive_to_acc_primitive[1],
    )
    if identify_index_details is None:
        selected_standard_setting = G0_STANDARD_SETTING
        selected_transformation_primitive_to_standard = raw_transformation_primitive_to_G0std
        standard_transform_selection_audit = {
            "strategy": "identify_index_unavailable",
            "status": "skipped",
            "standard_setting": selected_standard_setting,
            "selected_strategy": "raw_G0std_without_identify_index",
            "selected_matrix": np.asarray(
                selected_transformation_primitive_to_standard[0], dtype=float
            ).tolist(),
            "selected_origin_shift": np.asarray(
                selected_transformation_primitive_to_standard[1], dtype=float
            ).tolist(),
            "identify_index": identify_info,
        }
    else:
        (
            selected_standard_setting,
            selected_transformation_primitive_to_standard,
            standard_transform_selection_audit,
        ) = _select_standard_transform_for_acc_alignment(
            ssg_primitive,
            magnetic_primitive_cell,
            {
                G0_STANDARD_SETTING: raw_transformation_primitive_to_G0std,
                L0_STANDARD_SETTING: raw_transformation_primitive_to_L0std,
            },
            legacy_transformation_primitive_to_acc_primitive,
            legacy_acc_magnetic_primitive_cell,
            identify_info=identify_info,
            identify_index_details=identify_index_details,
            tol=tol_cfg,
        )
    if selected_standard_setting == G0_STANDARD_SETTING:
        raw_transformation_primitive_to_G0std = selected_transformation_primitive_to_standard
    else:
        raw_transformation_primitive_to_L0std = selected_transformation_primitive_to_standard
    raw_G0std_cell = magnetic_primitive_cell.transform(*raw_transformation_primitive_to_G0std)
    raw_L0std_cell = magnetic_primitive_cell.transform(*raw_transformation_primitive_to_L0std)
    raw_G0std_ssg = ssg_primitive.transform(*raw_transformation_primitive_to_G0std)
    raw_L0std_ssg = ssg_primitive.transform(*raw_transformation_primitive_to_L0std)
    G0std_axis_collapse_matrix, G0std_axis_collapse_audit = _select_G0std_axis_collapse(
        ssg_primitive,
        raw_G0std_ssg,
        identify_index_details=identify_index_details,
        tol=tol_cfg.space,
    )

    raw_transformation_input_to_G0std = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        raw_transformation_primitive_to_G0std[0],
        raw_transformation_primitive_to_G0std[1],
    )
    raw_transformation_input_to_L0std = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        raw_transformation_primitive_to_L0std[0],
        raw_transformation_primitive_to_L0std[1],
    )

    allow_input_collapse = _acc_setting_allows_input_collapse(ssg_primitive.acc)

    G0std_cell, G0std_ssg, transformation_input_to_G0std, _ = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        raw_G0std_cell,
        raw_G0std_ssg,
        raw_transformation_input_to_G0std,
        allow_identity_collapse=allow_input_collapse,
    )
    L0std_cell, L0std_ssg, transformation_input_to_L0std, _ = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        raw_L0std_cell,
        raw_L0std_ssg,
        raw_transformation_input_to_L0std,
        allow_identity_collapse=allow_input_collapse,
    )

    transformation_primitive_to_input = _invert_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
    )
    if selected_standard_setting == G0_STANDARD_SETTING:
        selected_standard_cell = G0std_cell
        selected_standard_ssg = G0std_ssg
        database_standard_cell = raw_G0std_cell
        database_standard_ssg = raw_G0std_ssg
        transformation_input_to_selected_standard = transformation_input_to_G0std
        transformation_input_to_database_standard = raw_transformation_input_to_G0std
    else:
        selected_standard_cell = L0std_cell
        selected_standard_ssg = L0std_ssg
        database_standard_cell = raw_L0std_cell
        database_standard_ssg = raw_L0std_ssg
        transformation_input_to_selected_standard = transformation_input_to_L0std
        transformation_input_to_database_standard = raw_transformation_input_to_L0std

    if identify_index_details is None:
        acc_magnetic_primitive_cell = legacy_acc_magnetic_primitive_cell
        acc_magnetic_primitive_ssg = ssg_primitive.transform(
            *legacy_transformation_primitive_to_acc_primitive
        )
        transformation_input_to_acc_primitive = legacy_transformation_input_to_acc_primitive
        transformation_selected_standard_to_acc_primitive = _compose_setting_transform(
            transformation_input_to_selected_standard[0],
            transformation_input_to_selected_standard[1],
            transformation_input_to_acc_primitive[0],
            transformation_input_to_acc_primitive[1],
        )
        acc_primitive_resolution_audit = {
            "strategy": "legacy_acc_transform_without_identify_index",
            "status": "identify_index_unavailable",
            "identify_index": identify_info,
            "selected_standard_setting": selected_standard_setting,
            "note": (
                "identify-index database details are unavailable, so ACC P-table "
                "validation is skipped; non-ACC symmetry outputs remain available."
            ),
        }
    else:
        (
            acc_magnetic_primitive_cell,
            acc_magnetic_primitive_ssg,
            transformation_input_to_acc_primitive,
            transformation_selected_standard_to_acc_primitive,
            acc_primitive_resolution_audit,
        ) = _resolve_acc_primitive_from_selected_standard(
            selected_standard_cell,
            magnetic_primitive_cell,
            ssg_primitive,
            transformation_input_to_primitive,
            transformation_input_to_selected_standard,
            transformation_input_to_database_standard,
            legacy_acc_magnetic_primitive_cell,
            legacy_transformation_input_to_acc_primitive,
            identify_info=identify_info,
            tol=tol_cfg,
        )
    acc_primitive_resolution_audit["standard_transform_selection"] = standard_transform_selection_audit
    if selected_standard_setting == G0_STANDARD_SETTING:
        acc_primitive_resolution_audit["G0std_transform_selection"] = standard_transform_selection_audit
    else:
        acc_primitive_resolution_audit["L0std_transform_selection"] = standard_transform_selection_audit
    transformation_G0std_to_primitive = _compose_setting_transform(
        transformation_input_to_G0std[0],
        transformation_input_to_G0std[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    transformation_L0std_to_primitive = _compose_setting_transform(
        transformation_input_to_L0std[0],
        transformation_input_to_L0std[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    acc_real_cartesian_to_poscar_spin_frame = _poscar_spin_frame_rotation(acc_magnetic_primitive_cell)
    poscar_spin_frame_to_acc_real_cartesian = np.linalg.inv(acc_real_cartesian_to_poscar_spin_frame)
    acc_magnetic_primitive_ssg_in_poscar_spin_frame = acc_magnetic_primitive_ssg.transform_spin(
        acc_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(
        acc_magnetic_primitive_ssg,
        acc_magnetic_primitive_cell,
    )
    internal_msg_info = acc_primitive_ossg.msg_info
    msg_num = None if internal_msg_info is None else internal_msg_info.get("msg_int_num")
    msg_type = None if internal_msg_info is None else internal_msg_info.get("msg_type")
    msg_symbol = None if internal_msg_info is None else internal_msg_info.get("msg_bns_symbol")

    actual_transformation_acc_primitive_to_acc_conventional = (
        np.asarray(acc_magnetic_primitive_ssg.cptrans, dtype=float),
        np.zeros(3),
    )
    actual_acc_conventional_cell = acc_magnetic_primitive_cell.transform(
        *actual_transformation_acc_primitive_to_acc_conventional
    )
    actual_acc_conventional_ssg = acc_magnetic_primitive_ssg.transform(
        *actual_transformation_acc_primitive_to_acc_conventional
    )
    actual_transformation_selected_standard_to_acc_conventional = _chain_setting_transform(
        transformation_selected_standard_to_acc_primitive[0],
        transformation_selected_standard_to_acc_primitive[1],
        actual_transformation_acc_primitive_to_acc_conventional[0],
        actual_transformation_acc_primitive_to_acc_conventional[1],
    )
    actual_selected_standard_to_acc_conventional_audit = audit_spatial_transform_effect(
        selected_standard_ssg,
        actual_transformation_selected_standard_to_acc_conventional[0],
        actual_transformation_selected_standard_to_acc_conventional[1],
        tol=tol_cfg.m_matrix_tol,
        use_nssg=False,
    )
    actual_selected_standard_to_acc_conventional_label = (
        "self_automorphism"
        if actual_selected_standard_to_acc_conventional_audit["real_ops_exact_same"]
        else "setting_change"
    )
    acc_primitive_output_cell = acc_magnetic_primitive_cell
    acc_primitive_output_ssg = acc_magnetic_primitive_ssg
    transformation_input_to_acc_primitive_output = transformation_input_to_acc_primitive
    transformation_G0std_to_acc_primitive_output = transformation_G0std_to_primitive
    transformation_L0std_to_acc_primitive_output = transformation_L0std_to_primitive
    acc_conventional_cell = actual_acc_conventional_cell
    acc_conventional_ssg = actual_acc_conventional_ssg
    transformation_selected_standard_to_acc_conventional = actual_transformation_selected_standard_to_acc_conventional
    selected_standard_to_acc_conventional_audit = actual_selected_standard_to_acc_conventional_audit
    selected_standard_to_acc_conventional_label = actual_selected_standard_to_acc_conventional_label
    convention_setting = selected_standard_setting
    convention_cell = selected_standard_cell
    convention_cell_snapshot = _serialize_cell_snapshot(selected_standard_cell)
    convention_ssg = selected_standard_ssg
    transformation_input_to_convention = transformation_input_to_selected_standard
    transformation_convention_to_primitive = transformation_selected_standard_to_acc_primitive
    transformation_convention_to_acc_conventional = transformation_selected_standard_to_acc_conventional
    convention_to_acc_conventional_audit = selected_standard_to_acc_conventional_audit
    convention_to_acc_conventional_label = selected_standard_to_acc_conventional_label

    input_cell_is_convention = _is_identity_setting_transform(
        transformation_input_to_convention,
        tol=tol_cfg.m_matrix_tol,
    )
    input_setting_warning = None
    if input_cell_is_convention:
        input_setting_ssg = convention_ssg
        input_setting_matches_true_ssg = True
        input_setting_identify_info = identify_info
        input_setting_identify_index_details = identify_index_details
    else:
        true_input_setting_ssg = ssg_primitive.transform(*transformation_primitive_to_input)
        input_compatible_ssg = _input_compatible_ssg_from_transformed_primitive(
            input_cell_cartesian,
            true_input_setting_ssg,
            tol=tol_cfg,
        )
        if (
            input_compatible_ssg is not None
            and len(input_compatible_ssg.ops) == len(true_input_setting_ssg.ops)
        ):
            input_setting_ssg = true_input_setting_ssg
            input_setting_matches_true_ssg = True
            input_setting_identify_info = identify_info
            input_setting_identify_index_details = identify_index_details
        elif input_compatible_ssg is not None:
            input_setting_ssg = input_compatible_ssg
            input_setting_matches_true_ssg = False
            input_setting_identify_info = _diagnostic_ssg_index(
                source_name,
                input_setting_ssg,
                tol=tol_cfg.m_matrix_tol,
            )
            input_setting_identify_index_details = None
            input_setting_warning = (
                "Input-cell SSG differs from the magnetic-primitive SSG transformed "
                f"to the input setting; input_ssg_index={input_setting_identify_info}."
            )
        else:
            input_identify_result = identify_spin_space_group_result(
                input_cell_cartesian,
                find_primitive=False,
                tol=tol_cfg,
            )
            input_setting_ssg = input_identify_result.ssg
            input_setting_matches_true_ssg = _spin_space_group_operation_sets_match(
                input_setting_ssg.ops,
                true_input_setting_ssg.ops,
                tol=tol_cfg.m_matrix_tol,
            )
            if input_setting_matches_true_ssg:
                input_setting_identify_info = identify_info
                input_setting_identify_index_details = identify_index_details
            else:
                input_setting_identify_info = _diagnostic_ssg_index(
                    source_name,
                    input_setting_ssg,
                    tol=tol_cfg.m_matrix_tol,
                )
                input_setting_identify_index_details = None
                input_setting_warning = (
                    "Input-cell SSG differs from the magnetic-primitive SSG transformed "
                    f"to the input setting; input_ssg_index={input_setting_identify_info}."
                )
    input_setting_ossg = _ossg_oriented_spin_frame_ssg(
        input_setting_ssg,
        input_cell_cartesian,
    )
    input_setting_index_differs = input_setting_identify_info != identify_info

    public_ossg_ssg = _ossg_oriented_spin_frame_ssg(convention_ssg, convention_cell)
    G0std_ops_nofrac_transform = None
    g0_standard_ssg_ops = G0std_ssg.ops
    public_convention_ssg_ops = public_ossg_ssg.ops
    if G0std_axis_collapse_audit and G0std_axis_collapse_audit["strategy"] == "axis_collapse":
        G0std_ops_nofrac_transform = _chain_setting_transform(
            transformation_input_to_G0std[0],
            transformation_input_to_G0std[1],
            G0std_axis_collapse_matrix,
            np.zeros(3),
        )
        g0_standard_ssg_ops = G0std_ssg.transform(G0std_axis_collapse_matrix, np.zeros(3)).ops
        if convention_setting == G0_STANDARD_SETTING:
            public_convention_ssg_ops = public_ossg_ssg.transform(
                G0std_axis_collapse_matrix,
                np.zeros(3),
            ).ops
    try:
        msg_acc = SpinSpaceGroup(public_ossg_ssg.msg_ops).acc
    except Exception:
        msg_acc = None
    gspg_payload = _build_gspg_payload(
        public_ossg_ssg,
        real_space_setting=convention_setting,
        spin_frame_setting=OSSG_ORIENTED_SPIN_FRAME_SETTING,
        spin_analysis_transform=_lattice_column_matrix(convention_cell),
    )
    ahc_wo_soc = is_ahc(gspg_payload["gspg_effective_mpg_symbol"])
    msg_parent_info = msg_parent_space_group_info(msg_num)
    ossg_space_group_number = None if identify_index_details is None else identify_index_details.get("G0_id")
    ssg_space_group_number = int(ssg_primitive.G0_num)
    source_parent_space_group = (
        None if source_metadata is None else source_metadata.get("parent_space_group")
    )
    domain_reversal_coset_analysis = None
    soc_domain_reversal_coset_analysis = None
    if ssg_primitive.conf == "Collinear":
        try:
            domain_reversal_coset_analysis = _build_domain_reversal_coset_analysis(
                source_metadata=source_metadata,
                g0std_cell=G0std_cell,
                g0std_ssg=G0std_ssg,
                ordered_space_group_number=ossg_space_group_number,
                tol_cfg=tol_cfg,
            )
        except Exception as exc:
            domain_reversal_coset_analysis = {
                "status": "not_evaluated_parent_ordered_coset_construction_failed",
                "basis_setting": G0_STANDARD_SETTING,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "candidate_reversal_domains": [],
            }
        if msg_parent_info["is_polar"] is True:
            try:
                soc_domain_reversal_coset_analysis = (
                    _build_g0std_soc_domain_reversal_coset_analysis(
                        g0std_cell=G0std_cell,
                        g0std_ssg=G0std_ssg,
                        msg_parent_space_group_number=msg_parent_info[
                            "bns_parent_space_group_number"
                        ],
                        tol_cfg=tol_cfg,
                    )
                )
            except Exception as exc:
                soc_domain_reversal_coset_analysis = {
                    "status": "not_evaluated_soc_parent_msg_coset_construction_failed",
                    "basis_setting": G0_STANDARD_SETTING,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "candidate_reversal_domains": [],
                }
    ferroelectric_switching = build_ferroelectric_switching_payload(
        input_space_group_number=input_space_group_number,
        input_space_group_symbol=input_space_group_symbol,
        ssg_space_group_number=ssg_space_group_number,
        ossg_space_group_number=ossg_space_group_number,
        msg_num=msg_num,
        msg_symbol=msg_symbol,
        msg_parent_space_group_number=msg_parent_info['bns_parent_space_group_number'],
        source_parent_space_group=source_parent_space_group,
        magnetic_phase=magnetic_phase,
        magnetic_phase_base=magnetic_phase_base,
        magnetic_configuration=ssg_primitive.conf,
        spin_splitting_without_soc=ss_wo_soc,
        is_altermagnet=alter,
        domain_reversal_coset_analysis=domain_reversal_coset_analysis,
        soc_domain_reversal_coset_analysis=soc_domain_reversal_coset_analysis,
        ordered_real_space_ops=[
            (np.asarray(op[1], dtype=float), np.asarray(op[2], dtype=float))
            for op in public_convention_ssg_ops
        ],
        ordered_real_space_ops_setting=convention_setting,
        soc_real_space_ops=[
            (np.asarray(op[1], dtype=float), np.asarray(op[2], dtype=float))
            for op in public_ossg_ssg.msg_ops
        ],
        soc_real_space_ops_setting=convention_setting,
        tol=tol_cfg.m_matrix_tol,
    )
    convention_sg_symmetry = _nonmagnetic_space_group_polar_symmetry_in_cell_basis(
        convention_cell,
        setting=convention_setting,
        tol_cfg=tol_cfg,
    )
    polar_axes_by_symmetry = build_polar_axes_by_symmetry_payload(
        sg_symmetry=convention_sg_symmetry,
        sg_space_group_number=None,
        ossg_symmetry=ferroelectric_switching.get("ordered_spin_space_symmetry"),
        msg_symmetry=ferroelectric_switching.get("soc_magnetic_symmetry"),
        tol=tol_cfg.m_matrix_tol,
    )
    acc_output_real_cartesian_to_poscar_spin_frame = _poscar_spin_frame_rotation(acc_primitive_output_cell)
    poscar_spin_frame_to_acc_output_real_cartesian = np.linalg.inv(
        acc_output_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_output_ssg_in_poscar_spin_frame = acc_primitive_output_ssg.transform_spin(
        acc_output_real_cartesian_to_poscar_spin_frame
    )
    transformation_acc_primitive_to_G0std = _invert_setting_transform(
        transformation_G0std_to_acc_primitive_output[0],
        transformation_G0std_to_acc_primitive_output[1],
    )
    transformation_acc_primitive_to_L0std = _invert_setting_transform(
        transformation_L0std_to_acc_primitive_output[0],
        transformation_L0std_to_acc_primitive_output[1],
    )
    transformation_G0std_to_input = _invert_setting_transform(
        transformation_input_to_G0std[0],
        transformation_input_to_G0std[1],
    )
    transformation_L0std_to_input = _invert_setting_transform(
        transformation_input_to_L0std[0],
        transformation_input_to_L0std[1],
    )
    transformation_acc_primitive_to_input = _invert_setting_transform(
        transformation_input_to_acc_primitive_output[0],
        transformation_input_to_acc_primitive_output[1],
    )
    transformation_convention_to_input = _invert_setting_transform(
        transformation_input_to_convention[0],
        transformation_input_to_convention[1],
    )

    KPOINTS = acc_primitive_output_ssg.KPOINTS
    SS =  acc_primitive_output_ssg.spin_polarizations
    SS_poscar = acc_primitive_output_ssg_in_poscar_spin_frame.spin_polarizations
    quasi_2d_diagnostics = build_quasi2d_diagnostics(
        input_cell_detail=_serialize_cell_snapshot(input_cell_cartesian),
        transformation_input_to_acc_primitive=transformation_input_to_acc_primitive_output,
        acc_primitive_ssg=acc_primitive_output_ssg,
        base_is_alter=alter,
        tol=tol_cfg.m_matrix_tol,
        calculation_mode=calculation_mode,
        vacuum_axis=vacuum_axis,
        vacuum_padding=quasi2d_vacuum_padding,
    )
    quasi_2d_magnetic_phase = _build_quasi2d_magnetic_phase(
        parent_magnetic_phase_payload=magnetic_phase_payload,
        quasi_2d=quasi_2d_diagnostics,
    )
    if quasi_2d_magnetic_phase is not None:
        quasi_2d_diagnostics['magnetic_phase'] = quasi_2d_magnetic_phase
    ssg_little_groups = _get_ssg_little_groups(
        acc_primitive_output_ssg,
        tol=tol_cfg.m_matrix_tol,
    )
    primitive_msg_ops, msg_little_groups, msg_little_group_symbols = _build_msg_little_group_core(
        acc_primitive_ossg,
        tol=tol_cfg.m_matrix_tol,
    )
    msg_spin_polarizations = _get_spin_constraint_for_msg_little_groups(
        msg_little_groups,
        acc_magnetic_primitive_cell,
        tol=tol_cfg.m_matrix_tol,
    )
    msg_spin_polarizations_poscar = _get_spin_constraint_for_msg_little_groups(
        msg_little_groups,
        acc_magnetic_primitive_cell,
        tol=tol_cfg.m_matrix_tol,
        spin_frame_rotation=acc_real_cartesian_to_poscar_spin_frame,
    )
    if quasi_2d_diagnostics is not None:
        quasi_2d_diagnostics.update(
            _build_quasi2d_little_group_payload(
                quasi_2d=quasi_2d_diagnostics,
                acc_primitive_ssg=acc_primitive_output_ssg,
                ssg_little_groups=ssg_little_groups,
                msg_little_groups=msg_little_groups,
                msg_little_group_symbols=msg_little_group_symbols,
                msg_spin_polarizations=msg_spin_polarizations_poscar,
                tol=tol_cfg.m_matrix_tol,
            )
        )
    tensor_outputs = _compute_tensor_outputs(
        acc_magnetic_primitive_ssg,
        acc_magnetic_primitive_cell,
        tol=tol_cfg.m_matrix_tol,
    )

    convention_nssg_ops = public_ossg_ssg.nssg
    convention_nssg_seitz, convention_nssg_seitz_latex = _serialize_op_list_seitz_symbols(
        convention_nssg_ops,
        tol=public_ossg_ssg.symbol_calibration_tol,
    )
    acc_primitive_oriented_seitz_descriptions = _seitz_descriptions_with_cartesian_spin_symbols(
        acc_primitive_ossg,
        spin_to_cartesian=_lattice_column_matrix(acc_primitive_output_cell),
        tol=acc_primitive_ossg.symbol_calibration_tol,
    )
    (
        acc_primitive_oriented_seitz,
        acc_primitive_oriented_seitz_latex,
    ) = _seitz_symbols_from_descriptions(acc_primitive_oriented_seitz_descriptions)
    input_oriented_seitz_descriptions = _seitz_descriptions_with_cartesian_spin_symbols(
        input_setting_ossg,
        spin_to_cartesian=_lattice_column_matrix(input_cell_cartesian),
        tol=input_setting_ossg.symbol_calibration_tol,
    )
    (
        _input_oriented_seitz,
        input_oriented_seitz_latex,
    ) = _seitz_symbols_from_descriptions(input_oriented_seitz_descriptions)

    public_convention_oriented_ssg = public_ossg_ssg
    if (
        len(public_convention_ssg_ops) != len(public_ossg_ssg.ops)
        or any(
            not public_op.is_same_with(ossg_op, atol=public_ossg_ssg.tol)
            for public_op, ossg_op in zip(public_convention_ssg_ops, public_ossg_ssg.ops)
        )
    ):
        public_convention_oriented_ssg = SpinSpaceGroup(
            list(public_convention_ssg_ops),
            tol=public_ossg_ssg.tol,
            real_space_metric=public_ossg_ssg.real_space_metric,
        )
    if quasi_2d_diagnostics is not None:
        quasi_2d_diagnostics.update(
            _quasi2d_spin_texture_config_from_ossg_convention(
                quasi_2d=quasi_2d_diagnostics,
                convention_ossg=public_convention_oriented_ssg,
                convention_cell=convention_cell,
                transformation_input_to_convention=transformation_input_to_convention,
                tol=tol_cfg.m_matrix_tol,
                calibration_atol_limit=max(tol_cfg.m_matrix_tol, tol_cfg.moment),
            )
        )
    public_convention_cartesian_ssg = public_convention_oriented_ssg.transform_spin(
        _lattice_column_matrix(convention_cell)
    )
    acc_primitive_cartesian_ops_payload = _serialize_ssg_operation_matrices(
        list(acc_primitive_output_ssg.ops)
    )
    acc_primitive_oriented_ops_payload = _serialize_ssg_operation_matrices(
        list(acc_primitive_ossg.ops)
    )
    input_cartesian_ops_payload = _serialize_ssg_operation_matrices(
        list(input_setting_ssg.ops)
    )
    input_oriented_ops_payload = _serialize_ssg_operation_matrices(
        list(input_setting_ossg.ops)
    )
    convention_oriented_ops_payload = _serialize_ssg_operation_matrices(
        list(public_convention_oriented_ssg.ops)
    )
    convention_cartesian_ops_payload = _serialize_ssg_operation_matrices(
        list(public_convention_cartesian_ssg.ops)
    )

    convention_oriented_generator_ops = _symbol_generator_ops_for_current_basis(
        public_convention_oriented_ssg
    )
    transformation_convention_to_acc_primitive_output = _compose_setting_transform(
        transformation_input_to_convention[0],
        transformation_input_to_convention[1],
        transformation_input_to_acc_primitive_output[0],
        transformation_input_to_acc_primitive_output[1],
    )
    convention_lattice_col = _lattice_column_matrix(convention_cell)
    acc_primitive_lattice_col = _lattice_column_matrix(acc_primitive_output_cell)
    input_lattice_col = _lattice_column_matrix(input_cell_cartesian)
    acc_primitive_setting_generator_ops = _transform_operation_generators(
        convention_oriented_generator_ops,
        transformation_convention_to_acc_primitive_output[0],
        transformation_convention_to_acc_primitive_output[1],
        tol=acc_primitive_ossg.tol,
        real_space_metric=acc_primitive_ossg.real_space_metric,
    )
    input_setting_generator_ops = _transform_operation_generators(
        convention_oriented_generator_ops,
        transformation_convention_to_input[0],
        transformation_convention_to_input[1],
        tol=input_setting_ossg.tol,
        real_space_metric=input_setting_ossg.real_space_metric,
    )
    acc_primitive_oriented_generator_ops = _transform_spin_generators(
        acc_primitive_setting_generator_ops,
        np.linalg.inv(acc_primitive_lattice_col) @ convention_lattice_col,
    )
    input_oriented_generator_ops = _transform_spin_generators(
        input_setting_generator_ops,
        np.linalg.inv(input_lattice_col) @ convention_lattice_col,
    )
    convention_cartesian_generator_ops = _transform_spin_generators(
        convention_oriented_generator_ops,
        convention_lattice_col,
    )
    acc_primitive_cartesian_generator_ops = _transform_spin_generators(
        acc_primitive_oriented_generator_ops,
        acc_primitive_lattice_col,
    )
    input_cartesian_generator_ops = _transform_spin_generators(
        input_oriented_generator_ops,
        input_lattice_col,
    )
    operation_views = _build_operation_views(
        {
            "convention_cartesian": {
                "ssg": public_convention_cartesian_ssg,
                "ops_payload": convention_cartesian_ops_payload,
                "seitz_latex": public_convention_cartesian_ssg.seitz_symbols_latex,
                "setting_label": convention_setting,
                "spin_frame": "cartesian",
                "generator_ops": convention_cartesian_generator_ops,
            },
            "convention_oriented": {
                "ssg": public_convention_oriented_ssg,
                "ops_payload": convention_oriented_ops_payload,
                "seitz_latex": public_convention_oriented_ssg.seitz_symbols_latex,
                "setting_label": convention_setting,
                "spin_frame": OSSG_ORIENTED_SPIN_FRAME_SETTING,
                "generator_ops": convention_oriented_generator_ops,
            },
            "magnetic_primitive_cartesian": {
                "ssg": acc_primitive_output_ssg,
                "ops_payload": acc_primitive_cartesian_ops_payload,
                "seitz_latex": acc_primitive_output_ssg.seitz_symbols_latex,
                "setting_label": ACC_PRIMITIVE_SETTING,
                "spin_frame": "cartesian",
                "generator_ops": acc_primitive_cartesian_generator_ops,
            },
            "magnetic_primitive_oriented": {
                "ssg": acc_primitive_ossg,
                "ops_payload": acc_primitive_oriented_ops_payload,
                "seitz_latex": acc_primitive_oriented_seitz_latex,
                "setting_label": ACC_PRIMITIVE_SETTING,
                "spin_frame": OSSG_ORIENTED_SPIN_FRAME_SETTING,
                "generator_ops": acc_primitive_oriented_generator_ops,
            },
            "input_cartesian": {
                "ssg": input_setting_ssg,
                "ops_payload": input_cartesian_ops_payload,
                "seitz_latex": input_setting_ssg.seitz_symbols_latex,
                "setting_label": "input",
                "spin_frame": "cartesian",
                "generator_ops": input_cartesian_generator_ops,
            },
            "input_oriented": {
                "ssg": input_setting_ossg,
                "ops_payload": input_oriented_ops_payload,
                "seitz_latex": input_oriented_seitz_latex,
                "setting_label": "input",
                "spin_frame": OSSG_ORIENTED_SPIN_FRAME_SETTING,
                "generator_ops": input_oriented_generator_ops,
            },
        }
    )

    scif_export_targets = _build_scif_export_targets(
        input_cell=input_cell_cartesian,
        acc_magnetic_primitive_cell=acc_magnetic_primitive_cell,
        acc_magnetic_primitive_ssg=acc_magnetic_primitive_ssg,
        database_standard_cell=database_standard_cell,
        database_standard_ssg=database_standard_ssg,
        database_standard_setting=selected_standard_setting,
        convention_cell=convention_cell,
        convention_ssg=convention_ssg,
        convention_setting=convention_setting,
        transformation_input_to_acc_primitive=transformation_input_to_acc_primitive,
        transformation_input_to_database_standard=transformation_input_to_database_standard,
        transformation_input_to_convention=transformation_input_to_convention,
        transformation_input_to_G0std=transformation_input_to_G0std,
        transformation_input_to_L0std=transformation_input_to_L0std,
        input_identified_ssg=input_setting_ssg,
    )
    for input_scif_mode in (SCIF_CELL_MODE_INPUT_CARTESIAN, SCIF_CELL_MODE_INPUT_ORIENTED):
        scif_export_targets[input_scif_mode].update(
            {
                "spin_space_group_index": input_setting_identify_info,
                "spin_space_group_name_linear": (
                    input_setting_ossg.international_symbol_linear_current_frame
                ),
                "spin_space_group_name_latex": (
                    input_setting_ossg.international_symbol_latex_current_frame
                ),
                "identify_index_details": input_setting_identify_index_details,
                "input_setting_warning": input_setting_warning,
                "suppress_repo_local_summary": input_setting_index_differs,
            }
        )
    wp_chain, g0std_wp_site_order = _build_wp_chain_payload_and_site_order(
        G0std_cell,
        G0std_ssg,
        tol_cfg,
        annotate_magnetic_site_dof=True,
    )
    (
        acc_primitive_wp_chain,
        acc_primitive_wp_site_order,
    ) = _build_wp_chain_payload_and_site_order(
        acc_primitive_output_cell,
        acc_primitive_output_ssg,
        tol_cfg,
        annotate_magnetic_site_dof=True,
    )
    input_wp_site_order = None
    if input_setting_matches_true_ssg:
        input_wp_chain, input_wp_site_order = _build_wp_chain_payload_and_site_order(
            input_cell_cartesian,
            input_setting_ssg,
            tol_cfg,
            annotate_magnetic_site_dof=True,
        )
    else:
        input_wp_chain = None
    try:
        magnetic_site_summary = _build_magnetic_site_summary(
            acc_primitive_output_cell,
            acc_primitive_output_ssg,
            identify_info,
            tol_cfg,
            setting=ACC_PRIMITIVE_SETTING,
        )
    except Exception as exc:
        magnetic_site_summary = {
            "status": "error",
            "setting": ACC_PRIMITIVE_SETTING,
            "ssg_index": identify_info,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    canonical_scif_target = scif_export_targets[SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED]
    actual_chen_linear_name = _build_chen_linear_name(
        source_name,
        canonical_scif_target["export_cell"],
        canonical_scif_target["export_ssg"],
        canonical_scif_target["basis_tag_transforms"],
        ssg_primitive,
        identify_index_details,
    )

    scif_outputs = {}
    for cell_mode, export_target in scif_export_targets.items():
        export_cell = export_target["export_cell"]
        export_ssg = export_target["export_ssg"]
        is_input_like_scif = bool(export_target.get("is_input_setting", False))
        export_spin_space_group_index = export_target.get("spin_space_group_index", identify_info)
        export_spin_space_group_name_linear = export_target.get(
            "spin_space_group_name_linear",
            public_ossg_ssg.international_symbol_linear_current_frame,
        )
        export_spin_space_group_name_latex = export_target.get(
            "spin_space_group_name_latex",
            public_ossg_ssg.international_symbol_latex_current_frame,
        )
        export_identify_index_details = export_target.get(
            "identify_index_details",
            identify_index_details,
        )
        export_wyckoff = get_spin_wyckoff(export_cell, export_ssg.ops)
        source_parent_space_group = (
            None if source_metadata is None else source_metadata.get("parent_space_group")
        )
        generated_parent_space_group, parent_space_group_comparison = (
            _identify_parent_space_group_for_export_cell(
                export_cell,
                symprec=tol_cfg.space,
                source_parent_space_group=source_parent_space_group,
                reuse_source_transforms=is_input_like_scif,
            )
        )
        source_cell_parameter_strings = (
            None
            if source_metadata is None or not is_input_like_scif
            else source_metadata.get("cell_parameter_strings")
        )
        scif_outputs[cell_mode] = generate_scif(
            source_name,
            export_cell,
            export_ssg,
            export_wyckoff,
            export_target["basis_tag_transforms"],
            ssg_primitive,
            spin_space_group_index=export_spin_space_group_index,
            spin_space_group_name=export_spin_space_group_name_linear,
            spin_space_group_name_chen=(
                actual_chen_linear_name
                if export_spin_space_group_index == identify_info
                else None
            ),
            spin_space_group_name_linear=export_spin_space_group_name_linear,
            spin_space_group_name_latex=export_spin_space_group_name_latex,
            magnetic_phase=magnetic_phase,
            identify_index_details=export_identify_index_details,
            source_cell_parameter_strings=source_cell_parameter_strings,
            parent_space_group=generated_parent_space_group,
            source_parent_space_group=source_parent_space_group,
            parent_space_group_comparison=parent_space_group_comparison,
            input_setting_warning=export_target.get("input_setting_warning"),
            suppress_repo_local_summary=bool(export_target.get("suppress_repo_local_summary", False)),
            spinframe_basis_abc_rows=export_target.get("spinframe_basis_abc_rows"),
            moment_basis_cartesian=export_target.get("moment_basis_cartesian"),
            real_space_setting=export_target.get("setting_name"),
            spin_frame_setting=export_target.get("spin_frame"),
            quasi_2d=quasi_2d_diagnostics,
        )

    scif = scif_outputs[SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED]

    acc_primitive_output_cell_snapshot = _serialize_cell_snapshot(
        acc_primitive_output_cell,
        site_order=acc_primitive_wp_site_order,
    )
    acc_primitive_output_cell_tuple = _cell_to_spglib_in_snapshot_order(
        acc_primitive_output_cell,
        site_order=acc_primitive_wp_site_order,
    )
    acc_primitive_output_poscar = _cell_to_poscar_in_snapshot_order(
        acc_primitive_output_cell,
        source_name,
        site_order=acc_primitive_wp_site_order,
    )
    acc_p_c_poscar = acc_primitive_output_poscar
    spin_texture_config = _spin_texture_config_for_public_output(identify_info)
    spin_texture_config_no_soc, spin_texture_config_soc = _spin_texture_config_from_ossg_convention(
        public_convention_oriented_ssg,
        convention_cell,
        tol=tol_cfg.m_matrix_tol,
        calibration_atol_limit=max(tol_cfg.m_matrix_tol, tol_cfg.moment),
        reference=spin_texture_config,
        generator_ops=convention_oriented_generator_ops,
    )

    result = {
        'index':identify_info,
        'spin_part_pg':ssg_primitive.spin_part_point_group_symbol_hm,
        'conf':ssg_primitive.conf,
        'id_index_info':identify_info,
        'scif':scif,
        'scif_outputs': scif_outputs,
        'scif_cell_modes': sorted(scif_export_targets.keys()),
        'poscar_mp':acc_primitive_output_poscar,
        'acc':ssg_primitive.acc,
        'msg_acc': msg_acc,
        'spin_texture_config': spin_texture_config,
        'spin_texture_config_no_soc': spin_texture_config_no_soc,
        'spin_texture_config_soc': spin_texture_config_soc,
        'KPOINTS':KPOINTS,
        'quasi_2d': quasi_2d_diagnostics,
        'polar_axes_by_symmetry': polar_axes_by_symmetry,
        'ferroelectric_switching': ferroelectric_switching,
    }

    cell = {
        'input_cell_detail': _serialize_cell_snapshot(
            input_cell_cartesian,
            site_order=input_wp_site_order,
        ),
        'input_magnetic_primitive_cell': magnetic_primitive_cell.to_spglib(mag=True),
        'input_magnetic_primitive_cell_setting': INPUT_MAGNETIC_PRIMITIVE_SETTING,
        'input_magnetic_primitive_cell_poscar': input_magnetic_primitive_poscar,
        'input_magnetic_primitive_cell_detail': _serialize_cell_snapshot(magnetic_primitive_cell),
        'acc_conventional_cell': acc_conventional_cell.to_spglib(mag=True),
        'acc_conventional_cell_setting': ACC_CONVENTIONAL_SETTING,
        'acc_conventional_cell_detail': _serialize_cell_snapshot(acc_conventional_cell),
        'magnetic_primitive_cell': acc_primitive_output_cell_tuple,
        'magnetic_primitive_cell_setting': ACC_PRIMITIVE_SETTING,
        'magnetic_primitive_cell_poscar': acc_p_c_poscar,
        'magnetic_primitive_cell_detail': acc_primitive_output_cell_snapshot,
        'primitive_magnetic_cell':acc_primitive_output_cell_tuple,
        'primitive_magnetic_cell_setting': ACC_PRIMITIVE_SETTING,
        'primitive_magnetic_cell_poscar':acc_p_c_poscar,
        'scif': scif,
        'scif_outputs': scif_outputs,
        'scif_cell_modes': sorted(scif_export_targets.keys()),
        'primitive_magnetic_cell_detail': acc_primitive_output_cell_snapshot,
        'acc_primitive_magnetic_cell': acc_primitive_output_cell_tuple,
        'acc_primitive_magnetic_cell_setting': ACC_PRIMITIVE_SETTING,
        'acc_primitive_magnetic_cell_poscar': acc_primitive_output_poscar,
        'acc_primitive_magnetic_cell_detail': acc_primitive_output_cell_snapshot,
        'g0_standard_cell': _serialize_cell_snapshot(
            G0std_cell,
            site_order=g0std_wp_site_order,
        ),
        'l0_standard_cell': _serialize_cell_snapshot(L0std_cell),
        'convention_cell': convention_cell.to_spglib(mag=True),
        'convention_cell_setting': convention_setting,
        'convention_cell_detail': convention_cell_snapshot,
        'wp_chain': wp_chain,
        'acc_primitive_wp_chain': acc_primitive_wp_chain,
        'input_wp_chain': input_wp_chain,
        'scif':scif,
    }
    symmetry = {'index':identify_info,
                'configuration':ssg_primitive.conf,
                'magnetic_phase':magnetic_phase,
                'magnetic_phase_base': magnetic_phase_base,
                'magnetic_phase_modifier': magnetic_phase_modifier,
                'magnetic_phase_spin_orbit_magnet': magnetic_phase_payload['spin_orbit_magnet_tag'],
                'magnetic_phase_details': magnetic_phase_details,
                'spin_texture_config': spin_texture_config,
                'spin_texture_config_no_soc': spin_texture_config_no_soc,
                'spin_texture_config_soc': spin_texture_config_soc,
                'acc':ssg_primitive.acc,
                'msg_acc': msg_acc,
                'G0_symbol': ssg_primitive.G0_symbol,
                'G0_num': int(ssg_primitive.G0_num),
                'L0_symbol': ssg_primitive.L0_symbol,
                'L0_num': int(ssg_primitive.L0_num),
                'it': int(ssg_primitive.it),
                'ik': int(ssg_primitive.ik),
                'SSPG_symbol_hm': ssg_primitive.spin_part_point_group_symbol_hm,
                'SSPG_symbol_s': ssg_primitive.spin_part_point_group_symbol_s,
                'input_space_group_number': input_space_group_number,
                'input_space_group_symbol': input_space_group_symbol,
                'sg_is_centrosymmetric': space_group_is_centrosymmetric(input_space_group_number),
                'sg_is_polar': space_group_is_polar(input_space_group_number),
                'sg_is_chiral': space_group_is_chiral(input_space_group_number),
                'input_space_group_basis_or_setting': input_space_group_basis_or_setting,
                'source_structure_metadata': source_metadata,
                'source_parent_space_group': (
                    None if source_metadata is None else source_metadata.get('parent_space_group')
                ),
                'source_cell_parameter_strings': (
                    None if source_metadata is None else source_metadata.get('cell_parameter_strings')
                ),
                'magnetic_site_summary': magnetic_site_summary,
                'KPOINTS':KPOINTS,
                'KPOINTS_setting': ACC_PRIMITIVE_SETTING,
                'KPOINTS_real_space_setting': ACC_PRIMITIVE_SETTING,
                'quasi_2d': quasi_2d_diagnostics,
                'polar_axes_by_symmetry': polar_axes_by_symmetry,
                'ferroelectric_switching': ferroelectric_switching,
                'operation_views': operation_views,
                'input_magnetic_primitive_ssg_ops': ssg_primitive.ops,
                'input_magnetic_primitive_ssg_setting': INPUT_MAGNETIC_PRIMITIVE_SETTING,
                'input_magnetic_primitive_ssg_seitz': ssg_primitive.seitz_symbols,
                'input_magnetic_primitive_ssg_seitz_latex': ssg_primitive.seitz_symbols_latex,
                'input_magnetic_primitive_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    ssg_primitive.seitz_descriptions
                ),
                'input_magnetic_primitive_ssg_international_linear': ssg_primitive.international_symbol_linear,
                'input_magnetic_primitive_ssg_international_latex': ssg_primitive.international_symbol_latex,
                'input_magnetic_primitive_ssg_symbol_calibration_tol': ssg_primitive.symbol_calibration_tol,
                'input_magnetic_primitive_ssg_type': ssg_primitive.international_symbol_type,
                'magnetic_primitive_ssg_ops': acc_magnetic_primitive_ssg.ops,
                'magnetic_primitive_ssg_setting': ACC_PRIMITIVE_SETTING,
                'magnetic_primitive_ssg_seitz': acc_magnetic_primitive_ssg.seitz_symbols,
                'magnetic_primitive_ssg_seitz_latex': acc_magnetic_primitive_ssg.seitz_symbols_latex,
                'magnetic_primitive_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_magnetic_primitive_ssg.seitz_descriptions
                ),
                'magnetic_primitive_ssg_international_linear': acc_magnetic_primitive_ssg.international_symbol_linear,
                'magnetic_primitive_ssg_international_latex': acc_magnetic_primitive_ssg.international_symbol_latex,
                'magnetic_primitive_ssg_symbol_calibration_tol': acc_magnetic_primitive_ssg.symbol_calibration_tol,
                'magnetic_primitive_ssg_type': acc_magnetic_primitive_ssg.international_symbol_type,
                'primitive_magnetic_cell_ssg_ops':acc_magnetic_primitive_ssg.ops,
                'primitive_magnetic_cell_ssg_setting': ACC_PRIMITIVE_SETTING,
                'primitive_magnetic_cell_ssg_seitz':acc_magnetic_primitive_ssg.seitz_symbols,
                'primitive_magnetic_cell_ssg_seitz_latex':acc_magnetic_primitive_ssg.seitz_symbols_latex,
                'primitive_magnetic_cell_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_magnetic_primitive_ssg.seitz_descriptions
                ),
                'primitive_magnetic_cell_ssg_international_linear':acc_magnetic_primitive_ssg.international_symbol_linear,
                'primitive_magnetic_cell_ssg_international_latex':acc_magnetic_primitive_ssg.international_symbol_latex,
                'primitive_magnetic_cell_ssg_symbol_calibration_tol': acc_magnetic_primitive_ssg.symbol_calibration_tol,
                'acc_primitive_ssg_ops': acc_primitive_output_ssg.ops,
                'acc_primitive_ssg_setting': ACC_PRIMITIVE_SETTING,
                'acc_primitive_ssg_seitz': acc_primitive_output_ssg.seitz_symbols,
                'acc_primitive_ssg_seitz_latex': acc_primitive_output_ssg.seitz_symbols_latex,
                'acc_primitive_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_primitive_output_ssg.seitz_descriptions
                ),
                'acc_primitive_ssg_international_linear': acc_primitive_output_ssg.international_symbol_linear,
                'acc_primitive_ssg_international_latex': acc_primitive_output_ssg.international_symbol_latex,
                'acc_primitive_ssg_symbol_calibration_tol': acc_primitive_output_ssg.symbol_calibration_tol,
                'acc_primitive_ssg_ops_cartesian': acc_primitive_cartesian_ops_payload,
                'acc_primitive_ssg_seitz_cartesian': acc_primitive_output_ssg.seitz_symbols,
                'acc_primitive_ssg_seitz_latex_cartesian': acc_primitive_output_ssg.seitz_symbols_latex,
                'acc_primitive_ssg_ops_oriented': acc_primitive_oriented_ops_payload,
                'acc_primitive_ssg_seitz_oriented': acc_primitive_oriented_seitz,
                'acc_primitive_ssg_seitz_latex_oriented': acc_primitive_oriented_seitz_latex,
                'acc_primitive_spin_only_direction_cartesian': _format_spin_only_direction(
                    acc_primitive_output_ssg.sog_direction
                ),
                'acc_primitive_spin_only_direction_poscar_spin_frame': _format_spin_only_direction(
                    acc_primitive_output_ssg_in_poscar_spin_frame.sog_direction
                ),
                'input_ssg_ops_spin_cartesian': input_cartesian_ops_payload,
                'input_ssg_seitz_latex_spin_cartesian': input_setting_ssg.seitz_symbols_latex,
                'input_ssg_ops_spin_oriented': input_oriented_ops_payload,
                'input_ssg_seitz_latex_spin_oriented': input_oriented_seitz_latex,
                'input_spin_only_direction_spin_cartesian': _format_spin_only_direction(
                    input_setting_ssg.sog_direction
                ),
                'input_spin_only_direction_spin_oriented': _format_spin_only_direction(
                    input_setting_ossg.sog_direction
                ),
                'input_ssg_may_be_incomplete': not input_setting_matches_true_ssg,
                'input_setting_warning': input_setting_warning,
                'symbol_calibration_tol': acc_magnetic_primitive_ssg.symbol_calibration_tol,
                'primitive_magnetic_cell_ssg_type':acc_magnetic_primitive_ssg.international_symbol_type,
                'full_spin_part_point_group':ssg_primitive.spin_part_point_group_symbol_hm,
                'identify_index_details':identify_index_details,
                'acc_primitive_resolution_audit': acc_primitive_resolution_audit,
                'g0std_axis_collapse_audit': G0std_axis_collapse_audit,
                'g0_standard_ssg_ops': g0_standard_ssg_ops,
                'g0_standard_ssg_seitz': G0std_ssg.seitz_symbols,
                'g0_standard_ssg_seitz_latex': G0std_ssg.seitz_symbols_latex,
                'g0_standard_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    G0std_ssg.seitz_descriptions
                ),
                'l0_standard_ssg_ops': L0std_ssg.ops,
                'l0_standard_ssg_seitz': L0std_ssg.seitz_symbols,
                'l0_standard_ssg_seitz_latex': L0std_ssg.seitz_symbols_latex,
                'l0_standard_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    L0std_ssg.seitz_descriptions
                ),
                'acc_conventional_ssg_ops': acc_conventional_ssg.ops,
                'acc_conventional_ssg_setting': ACC_CONVENTIONAL_SETTING,
                'acc_conventional_ssg_seitz': acc_conventional_ssg.seitz_symbols,
                'acc_conventional_ssg_seitz_latex': acc_conventional_ssg.seitz_symbols_latex,
                'acc_conventional_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_conventional_ssg.seitz_descriptions
                ),
                'acc_conventional_ssg_international_linear': acc_conventional_ssg.international_symbol_linear,
                'acc_conventional_ssg_international_latex': acc_conventional_ssg.international_symbol_latex,
                'acc_conventional_ssg_symbol_calibration_tol': acc_conventional_ssg.symbol_calibration_tol,
                'convention_ssg_ops': public_convention_ssg_ops,
                'convention_ssg_setting': convention_setting,
                'convention_ssg_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'ossg_space_group_number': ossg_space_group_number,
                'ossg_is_centrosymmetric': space_group_is_centrosymmetric(ossg_space_group_number),
                'ossg_is_polar': space_group_is_polar(ossg_space_group_number),
                'ossg_is_chiral': space_group_is_chiral(ossg_space_group_number),
                'convention_spin_only_direction': _format_spin_only_direction(public_ossg_ssg.sog_direction),
                'convention_spin_only_direction_cartesian': _format_spin_only_direction(
                    _cartesian_spin_only_direction_from_oriented(
                        public_ossg_ssg.sog_direction,
                        convention_cell,
                    )
                ),
                'convention_ssg_seitz': public_ossg_ssg.seitz_symbols,
                'convention_ssg_seitz_latex': public_ossg_ssg.seitz_symbols_latex,
                'convention_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    public_ossg_ssg.seitz_descriptions
                ),
                'convention_nssg_ops': convention_nssg_ops,
                'convention_nssg_seitz': convention_nssg_seitz,
                'convention_nssg_seitz_latex': convention_nssg_seitz_latex,
                'convention_ssg_international_linear': public_ossg_ssg.international_symbol_linear_current_frame,
                'convention_ssg_international_latex': public_ossg_ssg.international_symbol_latex_current_frame,
                'convention_ssg_symbol_calibration_tol': public_ossg_ssg.symbol_calibration_tol,
                'primitive_msg_ops': [
                    [int(item[0]), np.asarray(item[1], dtype=float).tolist(), np.asarray(item[2], dtype=float).tolist()]
                    for item in primitive_msg_ops
                ],
                'primitive_msg_ops_setting': ACC_PRIMITIVE_SETTING,
                'primitive_msg_ops_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'magnetic_primitive_msg_ops': [
                    [int(item[0]), np.asarray(item[1], dtype=float).tolist(), np.asarray(item[2], dtype=float).tolist()]
                    for item in primitive_msg_ops
                ],
                'magnetic_primitive_msg_ops_setting': ACC_PRIMITIVE_SETTING,
                'magnetic_primitive_msg_ops_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'acc_primitive_msg_ops': [
                    [int(item[0]), np.asarray(item[1], dtype=float).tolist(), np.asarray(item[2], dtype=float).tolist()]
                    for item in primitive_msg_ops
                ],
                'acc_primitive_msg_ops_setting': ACC_PRIMITIVE_SETTING,
                'acc_primitive_msg_ops_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'ssg_little_group_ops': _serialize_ssg_little_group_ops(ssg_little_groups),
                'ssg_little_group_seitz_latex': _serialize_ssg_little_group_seitz_latex(
                    ssg_little_groups,
                    tol=acc_primitive_output_ssg.symbol_calibration_tol,
                ),
                'msg_little_group_ops': _serialize_msg_little_group_ops(msg_little_groups),
                'msg_little_group_seitz_latex': _serialize_msg_little_group_seitz_latex(
                    msg_little_groups,
                    tol=tol_cfg.m_matrix_tol,
                ),
                'msg_little_group_symbols': msg_little_group_symbols,
                'msg_spin_polarizations': msg_spin_polarizations_poscar,
                'msg_spin_polarizations_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'msg_spin_polarizations_real_space_setting': ACC_PRIMITIVE_SETTING,
                'msg_spin_polarizations_spin_frame': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'msg_spin_polarizations_acc_cartesian': msg_spin_polarizations,
                'msg_spin_polarizations_acc_cartesian_setting': ACC_PRIMITIVE_CARTESIAN_SETTING,
                'msg_spin_polarizations_acc_poscar_spin_frame': msg_spin_polarizations_poscar,
                'msg_spin_polarizations_acc_poscar_spin_frame_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'T_input_to_G0std': (
                    np.asarray(transformation_input_to_G0std[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_G0std[1], dtype=float).tolist(),
                ),
                'T_input_to_G0std_ops_nofrac': (
                    None
                    if G0std_ops_nofrac_transform is None
                    else (
                        np.asarray(G0std_ops_nofrac_transform[0], dtype=float).tolist(),
                        np.asarray(G0std_ops_nofrac_transform[1], dtype=float).tolist(),
                    )
                ),
                'raw_T_input_to_G0std': (
                    np.asarray(raw_transformation_input_to_G0std[0], dtype=float).tolist(),
                    np.asarray(raw_transformation_input_to_G0std[1], dtype=float).tolist(),
                ),
                'T_G0std_to_primitive': (
                    np.asarray(transformation_G0std_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_G0std_to_primitive[1], dtype=float).tolist(),
                ),
                'T_G0std_to_acc_primitive': (
                    np.asarray(transformation_G0std_to_acc_primitive_output[0], dtype=float).tolist(),
                    np.asarray(transformation_G0std_to_acc_primitive_output[1], dtype=float).tolist(),
                ),
                'T_acc_primitive_to_G0std': (
                    np.asarray(transformation_acc_primitive_to_G0std[0], dtype=float).tolist(),
                    np.asarray(transformation_acc_primitive_to_G0std[1], dtype=float).tolist(),
                ),
                'T_G0std_to_input': (
                    np.asarray(transformation_G0std_to_input[0], dtype=float).tolist(),
                    np.asarray(transformation_G0std_to_input[1], dtype=float).tolist(),
                ),
                'T_input_to_L0std': (
                    np.asarray(transformation_input_to_L0std[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_L0std[1], dtype=float).tolist(),
                ),
                'raw_T_input_to_L0std': (
                    np.asarray(raw_transformation_input_to_L0std[0], dtype=float).tolist(),
                    np.asarray(raw_transformation_input_to_L0std[1], dtype=float).tolist(),
                ),
                'T_L0std_to_primitive': (
                    np.asarray(transformation_L0std_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_L0std_to_primitive[1], dtype=float).tolist(),
                ),
                'T_L0std_to_acc_primitive': (
                    np.asarray(transformation_L0std_to_acc_primitive_output[0], dtype=float).tolist(),
                    np.asarray(transformation_L0std_to_acc_primitive_output[1], dtype=float).tolist(),
                ),
                'T_acc_primitive_to_L0std': (
                    np.asarray(transformation_acc_primitive_to_L0std[0], dtype=float).tolist(),
                    np.asarray(transformation_acc_primitive_to_L0std[1], dtype=float).tolist(),
                ),
                'T_L0std_to_input': (
                    np.asarray(transformation_L0std_to_input[0], dtype=float).tolist(),
                    np.asarray(transformation_L0std_to_input[1], dtype=float).tolist(),
                ),
                'T_input_to_convention': (
                    np.asarray(transformation_input_to_convention[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_convention[1], dtype=float).tolist(),
                ),
                'T_convention_to_input': (
                    np.asarray(transformation_convention_to_input[0], dtype=float).tolist(),
                    np.asarray(transformation_convention_to_input[1], dtype=float).tolist(),
                ),
                'T_convention_to_acc_primitive': (
                    np.asarray(transformation_convention_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_convention_to_primitive[1], dtype=float).tolist(),
                ),
                'T_convention_to_acc_conventional': (
                    np.asarray(transformation_convention_to_acc_conventional[0], dtype=float).tolist(),
                    np.asarray(transformation_convention_to_acc_conventional[1], dtype=float).tolist(),
                ),
                'T_convention_to_acc_conventional_is_convention_self_automorphism': (
                    convention_to_acc_conventional_audit["real_ops_exact_same"]
                ),
                'T_convention_to_acc_conventional_label': convention_to_acc_conventional_label,
                'T_convention_to_acc_conventional_audit': {
                    'real_ops_exact_same': convention_to_acc_conventional_audit['real_ops_exact_same'],
                    'real_ops_same_mod_integer': convention_to_acc_conventional_audit['real_ops_same_mod_integer'],
                    'real_ops_same_mod_pure_translations': convention_to_acc_conventional_audit['real_ops_same_mod_pure_translations'],
                    'paired_spin_changed_count': convention_to_acc_conventional_audit['paired_spin_changed_count'],
                    'determinant': convention_to_acc_conventional_audit['determinant'],
                    'volume_preserving': convention_to_acc_conventional_audit['volume_preserving'],
                },
                'selected_standard_setting': selected_standard_setting,
                'T_selected_standard_to_acc_conventional': (
                    np.asarray(transformation_selected_standard_to_acc_conventional[0], dtype=float).tolist(),
                    np.asarray(transformation_selected_standard_to_acc_conventional[1], dtype=float).tolist(),
                ),
                'T_selected_standard_to_acc_conventional_is_self_automorphism': (
                    selected_standard_to_acc_conventional_audit['real_ops_exact_same']
                ),
                'T_selected_standard_to_acc_conventional_label': (
                    selected_standard_to_acc_conventional_label
                ),
                'T_selected_standard_to_acc_conventional_audit': {
                    'real_ops_exact_same': selected_standard_to_acc_conventional_audit['real_ops_exact_same'],
                    'real_ops_same_mod_integer': selected_standard_to_acc_conventional_audit['real_ops_same_mod_integer'],
                    'real_ops_same_mod_pure_translations': selected_standard_to_acc_conventional_audit['real_ops_same_mod_pure_translations'],
                    'paired_spin_changed_count': selected_standard_to_acc_conventional_audit['paired_spin_changed_count'],
                    'determinant': selected_standard_to_acc_conventional_audit['determinant'],
                    'volume_preserving': selected_standard_to_acc_conventional_audit['volume_preserving'],
                },
                'T_input_to_mag_primitive': (
                    np.asarray(transformation_input_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_primitive[1], dtype=float).tolist(),
                ),
                'T_input_to_input_magnetic_primitive': (
                    np.asarray(transformation_input_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_primitive[1], dtype=float).tolist(),
                ),
                'T_input_to_acc_primitive': (
                    np.asarray(transformation_input_to_acc_primitive_output[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_acc_primitive_output[1], dtype=float).tolist(),
                ),
                'T_acc_primitive_to_input': (
                    np.asarray(transformation_acc_primitive_to_input[0], dtype=float).tolist(),
                    np.asarray(transformation_acc_primitive_to_input[1], dtype=float).tolist(),
                ),
                'msg_num': msg_num,
                'msg_type': msg_type,
                'msg_bns_number': msg_parent_info['bns_number'],
                'msg_og_number': msg_parent_info['og_number'],
                'msg_parent_space_group_number': msg_parent_info['bns_parent_space_group_number'],
                'msg_is_centrosymmetric': msg_parent_info['is_centrosymmetric'],
                'msg_is_polar': msg_parent_info['is_polar'],
                'msg_is_chiral': msg_parent_info['is_chiral'],
                'tolerances': {
                    'space_tol': float(tol_cfg.space),
                    'mtol': float(tol_cfg.moment),
                    'meigtol': float(tol_cfg.m_eig),
                    'matrix_tol': float(tol_cfg.m_matrix_tol),
                    'parser_atol': None if parser_atol is None else float(parser_atol),
                },
                'spin_polarizations':SS_poscar,
                'spin_polarizations_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'spin_polarizations_real_space_setting': ACC_PRIMITIVE_SETTING,
                'spin_polarizations_spin_frame': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'spin_polarizations_acc_cartesian': SS,
                'spin_polarizations_acc_cartesian_setting': ACC_PRIMITIVE_CARTESIAN_SETTING,
                'acc_primitive_real_cartesian_to_poscar_spin_frame': np.asarray(
                    acc_output_real_cartesian_to_poscar_spin_frame, dtype=float
                ).tolist(),
                'poscar_spin_frame_to_acc_primitive_real_cartesian': np.asarray(
                    poscar_spin_frame_to_acc_output_real_cartesian, dtype=float
                ).tolist(),
                'real_cartesian_to_spin_frame': np.asarray(
                    acc_output_real_cartesian_to_poscar_spin_frame, dtype=float
                ).tolist(),
                'spin_frame_to_real_cartesian': np.asarray(
                    poscar_spin_frame_to_acc_output_real_cartesian, dtype=float
                ).tolist(),
                'spin_polarizations_acc_poscar_spin_frame': SS_poscar,
                'spin_polarizations_acc_poscar_spin_frame_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'msg_symbol':msg_symbol,
                **gspg_payload}
    properties = {
        'ss_w_soc':ss_w_soc,
        'ss_wo_soc':ss_wo_soc,
        'ahc_w_soc':ahc_w_soc,
        'ahc_wo_soc':ahc_wo_soc,
        'is_alter':alter,
        'is_spin_orbit_magnet': magnetic_phase_payload['is_spin_orbit_magnet'],
        'magnetic_phase_base': magnetic_phase_base,
        'magnetic_phase_modifier': magnetic_phase_modifier,
        'tensor_outputs': tensor_outputs,
    }

    return MagSymmetryResult(cell,symmetry,properties)


def _find_spin_group_basic_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    input_spin_setting: str = "in_lattice",
) -> dict:
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    magnetic_primitive_cell, transformation_input_to_primitive = input_cell.get_primitive_structure(
        magnetic=True
    )
    identify_result = identify_spin_space_group_result(
        magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    ssg_primitive: SpinSpaceGroup = identify_result.ssg
    _assert_ssg_ops_consistency(
        "basic magnetic primitive",
        ssg_primitive,
        tol=tol_cfg,
    )
    input_space_group = identify_result.input_space_group
    input_space_group_number = None if input_space_group is None else input_space_group.number
    input_space_group_symbol = None if input_space_group is None else input_space_group.symbol

    identify_index_details = None
    identify_info = None
    try:
        identify_index_details = _identify_ssg_index_details(
            source_name,
            ssg_primitive,
            tol=tol_cfg.space,
        )
        identify_info = identify_index_details["index"]
        _assert_ssg_ops_consistency(
            "basic magnetic primitive",
            ssg_primitive,
            tol=tol_cfg,
            identify_index_details=identify_index_details,
        )
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        identify_info = _handle_missing_identify_index(source_name, exc)

    primitive_ossg_for_phase = _ossg_oriented_spin_frame_ssg(
        ssg_primitive,
        magnetic_primitive_cell,
    )

    magnetic_phase_payload = classify_magnetic_phase(
        conf=ssg_primitive.conf,
        full_spin_part_point_group_hm=ssg_primitive.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=ssg_primitive.spin_part_point_group_symbol_s,
        net_moment=magnetic_primitive_cell.net_moment,
        net_moment_tol=tol_cfg.moment,
        mpg_identifier=primitive_ossg_for_phase.mpg_num,
        is_ss_gp=ssg_primitive.is_spinsplitting[-1],
    )
    ss_w_soc = spin_splitting_w_soc(ssg_primitive)
    ahc_w_soc = is_ahc(primitive_ossg_for_phase.mpg_num)
    ss_wo_soc = magnetic_phase_payload["spin_splitting_without_soc"]
    empg_symbol = ssg_primitive.gspg.empg_symbol
    ahc_wo_soc = is_ahc(empg_symbol)

    transformation_input_to_primitive_setting = (
        np.asarray(transformation_input_to_primitive, dtype=float),
        np.zeros(3),
    )
    legacy_transformation_primitive_to_acc_primitive = (
        np.asarray(ssg_primitive.acc_primitive_trans, dtype=float),
        np.asarray(ssg_primitive.acc_primitive_origin_shift, dtype=float),
    )
    legacy_acc_magnetic_primitive_cell = magnetic_primitive_cell.transform(
        *legacy_transformation_primitive_to_acc_primitive
    )
    legacy_transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_primitive_setting[0],
        transformation_input_to_primitive_setting[1],
        legacy_transformation_primitive_to_acc_primitive[0],
        legacy_transformation_primitive_to_acc_primitive[1],
    )
    raw_transformation_primitive_to_G0std = (
        np.asarray(ssg_primitive.transformation_to_G0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_G0std, dtype=float),
    )
    raw_transformation_primitive_to_L0std = (
        np.asarray(ssg_primitive.transformation_to_L0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_L0std, dtype=float),
    )
    if identify_index_details is None:
        selected_standard_setting = G0_STANDARD_SETTING
        selected_transformation_primitive_to_standard = raw_transformation_primitive_to_G0std
        standard_transform_selection_audit = {
            "strategy": "identify_index_unavailable",
            "status": "skipped",
            "standard_setting": selected_standard_setting,
            "selected_strategy": "raw_G0std_without_identify_index",
            "selected_matrix": np.asarray(
                selected_transformation_primitive_to_standard[0], dtype=float
            ).tolist(),
            "selected_origin_shift": np.asarray(
                selected_transformation_primitive_to_standard[1], dtype=float
            ).tolist(),
            "identify_index": identify_info,
        }
    else:
        (
            selected_standard_setting,
            selected_transformation_primitive_to_standard,
            standard_transform_selection_audit,
        ) = _select_standard_transform_for_acc_alignment(
            ssg_primitive,
            magnetic_primitive_cell,
            {
                G0_STANDARD_SETTING: raw_transformation_primitive_to_G0std,
                L0_STANDARD_SETTING: raw_transformation_primitive_to_L0std,
            },
            legacy_transformation_primitive_to_acc_primitive,
            legacy_acc_magnetic_primitive_cell,
            identify_info=identify_info,
            identify_index_details=identify_index_details,
            tol=tol_cfg,
        )
    if selected_standard_setting == G0_STANDARD_SETTING:
        raw_transformation_primitive_to_G0std = selected_transformation_primitive_to_standard
    else:
        raw_transformation_primitive_to_L0std = selected_transformation_primitive_to_standard
    raw_transformation_input_to_G0std = _chain_setting_transform(
        transformation_input_to_primitive_setting[0],
        transformation_input_to_primitive_setting[1],
        raw_transformation_primitive_to_G0std[0],
        raw_transformation_primitive_to_G0std[1],
    )
    raw_transformation_input_to_L0std = _chain_setting_transform(
        transformation_input_to_primitive_setting[0],
        transformation_input_to_primitive_setting[1],
        raw_transformation_primitive_to_L0std[0],
        raw_transformation_primitive_to_L0std[1],
    )
    transformation_input_to_selected_standard = (
        raw_transformation_input_to_G0std
        if selected_standard_setting == G0_STANDARD_SETTING
        else raw_transformation_input_to_L0std
    )
    selected_transformation_primitive_to_standard = (
        raw_transformation_primitive_to_G0std
        if selected_standard_setting == G0_STANDARD_SETTING
        else raw_transformation_primitive_to_L0std
    )
    selected_standard_cell = magnetic_primitive_cell.transform(
        *selected_transformation_primitive_to_standard
    )
    selected_standard_ssg = ssg_primitive.transform(
        *selected_transformation_primitive_to_standard
    )
    selected_standard_ossg = _ossg_oriented_spin_frame_ssg(
        selected_standard_ssg,
        selected_standard_cell,
    )
    if identify_index_details is None:
        acc_magnetic_primitive_cell = legacy_acc_magnetic_primitive_cell
        acc_magnetic_primitive_ssg = ssg_primitive.transform(
            *legacy_transformation_primitive_to_acc_primitive
        )
        transformation_input_to_acc_primitive = legacy_transformation_input_to_acc_primitive
        transformation_selected_standard_to_acc_primitive = _compose_setting_transform(
            transformation_input_to_selected_standard[0],
            transformation_input_to_selected_standard[1],
            transformation_input_to_acc_primitive[0],
            transformation_input_to_acc_primitive[1],
        )
        acc_primitive_resolution_audit = {
            "strategy": "legacy_acc_transform_without_identify_index",
            "status": "identify_index_unavailable",
            "identify_index": identify_info,
            "selected_standard_setting": selected_standard_setting,
            "note": (
                "identify-index database details are unavailable, so ACC P-table "
                "validation is skipped; non-ACC symmetry outputs remain available."
            ),
        }
    else:
        (
            acc_magnetic_primitive_cell,
            acc_magnetic_primitive_ssg,
            transformation_input_to_acc_primitive,
            transformation_selected_standard_to_acc_primitive,
            acc_primitive_resolution_audit,
        ) = _resolve_acc_primitive_from_selected_standard(
            selected_standard_cell,
            magnetic_primitive_cell,
            ssg_primitive,
            transformation_input_to_primitive_setting,
            transformation_input_to_selected_standard,
            transformation_input_to_selected_standard,
            legacy_acc_magnetic_primitive_cell,
            legacy_transformation_input_to_acc_primitive,
            identify_info=identify_info,
            tol=tol_cfg,
        )
    acc_primitive_resolution_audit["standard_transform_selection"] = standard_transform_selection_audit
    if selected_standard_setting == G0_STANDARD_SETTING:
        acc_primitive_resolution_audit["G0std_transform_selection"] = standard_transform_selection_audit
    else:
        acc_primitive_resolution_audit["L0std_transform_selection"] = standard_transform_selection_audit
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(
        acc_magnetic_primitive_ssg,
        acc_magnetic_primitive_cell,
    )
    internal_msg_info = acc_primitive_ossg.msg_info
    msg_num = None if internal_msg_info is None else internal_msg_info.get("msg_int_num")
    msg_symbol = None if internal_msg_info is None else internal_msg_info.get("msg_bns_symbol")
    msg_type = None if internal_msg_info is None else internal_msg_info.get("msg_type")
    msg_parent_info = msg_parent_space_group_info(msg_num)
    magnetic_phase_details = magnetic_phase_payload["details"]

    ssg_space_group_number = int(ssg_primitive.G0_num)
    ferroelectric_switching = build_ferroelectric_switching_payload(
        input_space_group_number=input_space_group_number,
        input_space_group_symbol=input_space_group_symbol,
        ssg_space_group_number=ssg_space_group_number,
        ossg_space_group_number=None,
        msg_num=msg_num,
        msg_symbol=msg_symbol,
        msg_parent_space_group_number=msg_parent_info["bns_parent_space_group_number"],
        source_parent_space_group=None,
        magnetic_phase=magnetic_phase_payload["phase"],
        magnetic_phase_base=magnetic_phase_payload["base_phase"],
        magnetic_configuration=ssg_primitive.conf,
        spin_splitting_without_soc=ss_wo_soc,
        is_altermagnet=magnetic_phase_payload["is_alter"],
        ordered_real_space_ops=acc_primitive_ossg.G0_ops,
        ordered_real_space_ops_setting=ACC_PRIMITIVE_SETTING,
        soc_real_space_ops=[
            (np.asarray(op[1], dtype=float), np.asarray(op[2], dtype=float))
            for op in acc_primitive_ossg.msg_ops
        ],
        soc_real_space_ops_setting=ACC_PRIMITIVE_SETTING,
        tol=tol_cfg.m_matrix_tol,
    )
    acc_primitive_sg_symmetry = _nonmagnetic_space_group_polar_symmetry_in_cell_basis(
        acc_magnetic_primitive_cell,
        setting=ACC_PRIMITIVE_SETTING,
        tol_cfg=tol_cfg,
    )
    polar_axes_by_symmetry = build_polar_axes_by_symmetry_payload(
        sg_symmetry=acc_primitive_sg_symmetry,
        sg_space_group_number=None,
        ossg_symmetry=ferroelectric_switching.get("ordered_spin_space_symmetry"),
        msg_symmetry=ferroelectric_switching.get("soc_magnetic_symmetry"),
        tol=tol_cfg.m_matrix_tol,
    )
    spin_texture_config = _spin_texture_config_for_public_output(identify_info)
    spin_texture_config_no_soc, spin_texture_config_soc = _spin_texture_config_from_ossg_convention(
        selected_standard_ossg,
        selected_standard_cell,
        tol=tol_cfg.m_matrix_tol,
        calibration_atol_limit=max(tol_cfg.m_matrix_tol, tol_cfg.moment),
        reference=spin_texture_config,
    )

    return {
        "index": identify_info,
        "identify_index_details": identify_index_details,
        "g0_symbol": ssg_primitive.G0_symbol,
        "g0_number": int(ssg_primitive.G0_num),
        "l0_symbol": ssg_primitive.L0_symbol,
        "l0_number": int(ssg_primitive.L0_num),
        "it": int(ssg_primitive.it),
        "ik": int(ssg_primitive.ik),
        "nsspg": ssg_primitive.n_spin_part_point_group_symbol_hm,
        "sspg": ssg_primitive.spin_part_point_group_symbol_hm,
        "acc_symbol": ssg_primitive.acc,
        "acc_primitive_resolution_audit": acc_primitive_resolution_audit,
        "T_input_to_acc_primitive": (
            np.asarray(transformation_input_to_acc_primitive[0], dtype=float).tolist(),
            np.asarray(transformation_input_to_acc_primitive[1], dtype=float).tolist(),
        ),
        "T_selected_standard_to_acc_primitive": (
            np.asarray(transformation_selected_standard_to_acc_primitive[0], dtype=float).tolist(),
            np.asarray(transformation_selected_standard_to_acc_primitive[1], dtype=float).tolist(),
        ),
        "space_group_symbol": input_space_group_symbol,
        "space_group_number": input_space_group_number,
        "msg_symbol": msg_symbol,
        "msg_type": msg_type,
        "msg_bns_number": msg_parent_info["bns_number"],
        "msg_og_number": msg_parent_info["og_number"],
        "empg": empg_symbol,
        "conf": ssg_primitive.conf,
        "phase": magnetic_phase_payload["phase"],
        "magnetic_phase": magnetic_phase_payload["phase"],
        "magnetic_phase_base": magnetic_phase_payload["base_phase"],
        "magnetic_phase_modifier": magnetic_phase_payload["modifier"],
        "magnetic_phase_details": magnetic_phase_details,
        "spin_texture_config": spin_texture_config,
        "spin_texture_config_no_soc": spin_texture_config_no_soc,
        "spin_texture_config_soc": spin_texture_config_soc,
        "net_moment": magnetic_phase_details["net_moment"],
        "zero_net_moment_tol": magnetic_phase_details["zero_net_moment_tol"],
        "properties": {
            "ss_w_soc": ss_w_soc,
            "ss_wo_soc": ss_wo_soc,
            "ahc_w_soc": ahc_w_soc,
            "ahc_wo_soc": ahc_wo_soc,
            "is_alter": magnetic_phase_payload["is_alter"],
            "is_spin_orbit_magnet": magnetic_phase_payload["is_spin_orbit_magnet"],
            "magnetic_phase_base": magnetic_phase_payload["base_phase"],
            "magnetic_phase_modifier": magnetic_phase_payload["modifier"],
        },
        "is_alter": magnetic_phase_payload["is_alter"],
        "is_som": magnetic_phase_payload["is_spin_orbit_magnet"],
        "sg_is_polar": space_group_is_polar(input_space_group_number),
        "sg_is_chiral": space_group_is_chiral(input_space_group_number),
        "ssg_is_polar": space_group_is_polar(ssg_space_group_number),
        "ssg_is_chiral": space_group_is_chiral(ssg_space_group_number),
        "msg_is_polar": msg_parent_info["is_polar"],
        "msg_is_chiral": msg_parent_info["is_chiral"],
        "quasi_2d": None,
        "polar_axes_by_symmetry": polar_axes_by_symmetry,
        "ferroelectric_switching": ferroelectric_switching,
        "tolerances": {
            "space_tol": float(tol_cfg.space),
            "mtol": float(tol_cfg.moment),
            "meigtol": float(tol_cfg.m_eig),
            "matrix_tol": float(tol_cfg.m_matrix_tol),
        },
    }


def _find_spin_group_acc_primitive_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    input_spin_setting: str = "in_lattice",
) -> dict:
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    magnetic_primitive_cell, transformation_input_to_primitive_matrix = input_cell.get_primitive_structure(
        magnetic=True
    )
    transformation_input_to_primitive = (
        np.asarray(transformation_input_to_primitive_matrix, dtype=float),
        np.zeros(3),
    )
    identify_result = identify_spin_space_group_result(
        magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    ssg_primitive: SpinSpaceGroup = identify_result.ssg
    _assert_ssg_ops_consistency(
        "ACC primitive magnetic primitive",
        ssg_primitive,
        tol=tol_cfg,
    )

    identify_index_details = None
    identify_info = None
    try:
        identify_index_details = ssg_primitive.identify_index_details(
            source_name,
            tol=tol_cfg.space,
        )
        identify_info = identify_index_details["index"]
        _assert_ssg_ops_consistency(
            "ACC primitive magnetic primitive",
            ssg_primitive,
            tol=tol_cfg,
            identify_index_details=identify_index_details,
        )
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        identify_info = _handle_missing_identify_index(source_name, exc)

    legacy_transformation_primitive_to_acc_primitive = (
        np.asarray(ssg_primitive.acc_primitive_trans, dtype=float),
        np.asarray(ssg_primitive.acc_primitive_origin_shift, dtype=float),
    )
    legacy_acc_primitive_cell = magnetic_primitive_cell.transform(
        *legacy_transformation_primitive_to_acc_primitive
    )
    legacy_transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        legacy_transformation_primitive_to_acc_primitive[0],
        legacy_transformation_primitive_to_acc_primitive[1],
    )

    raw_transformation_primitive_to_G0std = (
        np.asarray(ssg_primitive.transformation_to_G0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_G0std, dtype=float),
    )
    raw_transformation_primitive_to_L0std = (
        np.asarray(ssg_primitive.transformation_to_L0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_L0std, dtype=float),
    )
    if identify_index_details is None:
        selected_standard_setting = G0_STANDARD_SETTING
        selected_transformation_primitive_to_standard = raw_transformation_primitive_to_G0std
        standard_transform_selection_audit = {
            "strategy": "identify_index_unavailable",
            "status": "skipped",
            "standard_setting": selected_standard_setting,
            "selected_strategy": "raw_G0std_without_identify_index",
            "selected_matrix": np.asarray(
                selected_transformation_primitive_to_standard[0], dtype=float
            ).tolist(),
            "selected_origin_shift": np.asarray(
                selected_transformation_primitive_to_standard[1], dtype=float
            ).tolist(),
            "identify_index": identify_info,
        }
    else:
        (
            selected_standard_setting,
            selected_transformation_primitive_to_standard,
            standard_transform_selection_audit,
        ) = _select_standard_transform_for_acc_alignment(
            ssg_primitive,
            magnetic_primitive_cell,
            {
                G0_STANDARD_SETTING: raw_transformation_primitive_to_G0std,
                L0_STANDARD_SETTING: raw_transformation_primitive_to_L0std,
            },
            legacy_transformation_primitive_to_acc_primitive,
            legacy_acc_primitive_cell,
            identify_info=identify_info,
            identify_index_details=identify_index_details,
            tol=tol_cfg,
        )
    if selected_standard_setting == G0_STANDARD_SETTING:
        raw_transformation_primitive_to_G0std = selected_transformation_primitive_to_standard
    else:
        raw_transformation_primitive_to_L0std = selected_transformation_primitive_to_standard
    raw_G0std_cell = magnetic_primitive_cell.transform(*raw_transformation_primitive_to_G0std)
    raw_L0std_cell = magnetic_primitive_cell.transform(*raw_transformation_primitive_to_L0std)
    raw_G0std_ssg = ssg_primitive.transform(*raw_transformation_primitive_to_G0std)
    raw_L0std_ssg = ssg_primitive.transform(*raw_transformation_primitive_to_L0std)
    G0std_axis_collapse_matrix, G0std_axis_collapse_audit = _select_G0std_axis_collapse(
        ssg_primitive,
        raw_G0std_ssg,
        identify_index_details=identify_index_details,
        tol=tol_cfg.space,
    )
    raw_transformation_input_to_G0std = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        raw_transformation_primitive_to_G0std[0],
        raw_transformation_primitive_to_G0std[1],
    )
    raw_transformation_input_to_L0std = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        raw_transformation_primitive_to_L0std[0],
        raw_transformation_primitive_to_L0std[1],
    )
    input_cell_cartesian = _cartesianized_input_cell(input_cell)
    allow_input_collapse = _acc_setting_allows_input_collapse(ssg_primitive.acc)
    G0std_cell, G0std_ssg, transformation_input_to_G0std, _ = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        raw_G0std_cell,
        raw_G0std_ssg,
        raw_transformation_input_to_G0std,
        allow_identity_collapse=allow_input_collapse,
    )
    L0std_cell, L0std_ssg, transformation_input_to_L0std, _ = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        raw_L0std_cell,
        raw_L0std_ssg,
        raw_transformation_input_to_L0std,
        allow_identity_collapse=allow_input_collapse,
    )
    if selected_standard_setting == G0_STANDARD_SETTING:
        selected_standard_cell = G0std_cell
        selected_standard_ssg = G0std_ssg
        transformation_input_to_selected_standard = transformation_input_to_G0std
        transformation_input_to_database_standard = raw_transformation_input_to_G0std
    else:
        selected_standard_cell = L0std_cell
        selected_standard_ssg = L0std_ssg
        transformation_input_to_selected_standard = transformation_input_to_L0std
        transformation_input_to_database_standard = raw_transformation_input_to_L0std
    selected_standard_ossg = _ossg_oriented_spin_frame_ssg(
        selected_standard_ssg,
        selected_standard_cell,
    )

    if identify_index_details is None:
        acc_primitive_cell = legacy_acc_primitive_cell
        acc_primitive_ssg = ssg_primitive.transform(
            *legacy_transformation_primitive_to_acc_primitive
        )
        transformation_input_to_acc_primitive = legacy_transformation_input_to_acc_primitive
        transformation_convention_to_acc_primitive = _compose_setting_transform(
            transformation_input_to_selected_standard[0],
            transformation_input_to_selected_standard[1],
            transformation_input_to_acc_primitive[0],
            transformation_input_to_acc_primitive[1],
        )
        acc_primitive_resolution_audit = {
            "strategy": "legacy_acc_transform_without_identify_index",
            "status": "identify_index_unavailable",
            "identify_index": identify_info,
            "selected_standard_setting": selected_standard_setting,
            "note": (
                "identify-index database details are unavailable, so ACC P-table "
                "validation is skipped; non-ACC symmetry outputs remain available."
            ),
        }
    else:
        (
            acc_primitive_cell,
            acc_primitive_ssg,
            transformation_input_to_acc_primitive,
            transformation_convention_to_acc_primitive,
            acc_primitive_resolution_audit,
        ) = _resolve_acc_primitive_from_selected_standard(
            selected_standard_cell,
            magnetic_primitive_cell,
            ssg_primitive,
            transformation_input_to_primitive,
            transformation_input_to_selected_standard,
            transformation_input_to_database_standard,
            legacy_acc_primitive_cell,
            legacy_transformation_input_to_acc_primitive,
            identify_info=identify_info,
            tol=tol_cfg,
        )
    acc_primitive_resolution_audit["standard_transform_selection"] = standard_transform_selection_audit
    if selected_standard_setting == G0_STANDARD_SETTING:
        acc_primitive_resolution_audit["G0std_transform_selection"] = standard_transform_selection_audit
    else:
        acc_primitive_resolution_audit["L0std_transform_selection"] = standard_transform_selection_audit
    acc_real_cartesian_to_poscar_spin_frame = _poscar_spin_frame_rotation(acc_primitive_cell)
    poscar_spin_frame_to_acc_real_cartesian = np.linalg.inv(
        acc_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_ssg_in_poscar_spin_frame = acc_primitive_ssg.transform_spin(
        acc_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(
        acc_primitive_ssg,
        acc_primitive_cell,
    )
    transformation_acc_primitive_to_G0std = _compose_setting_transform(
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
        transformation_input_to_G0std[0],
        transformation_input_to_G0std[1],
    )
    transformation_acc_primitive_to_L0std = _compose_setting_transform(
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
        transformation_input_to_L0std[0],
        transformation_input_to_L0std[1],
    )
    (
        acc_primitive_wp_chain,
        acc_primitive_wp_site_order,
    ) = _build_wp_chain_payload_and_site_order(
        acc_primitive_cell,
        acc_primitive_ssg,
        tol_cfg,
        annotate_magnetic_site_dof=True,
    )
    acc_primitive_poscar = _cell_to_poscar_in_snapshot_order(
        acc_primitive_cell,
        source_name,
        site_order=acc_primitive_wp_site_order,
    )
    acc_primitive_oriented_seitz_descriptions = _seitz_descriptions_with_cartesian_spin_symbols(
        acc_primitive_ossg,
        spin_to_cartesian=_lattice_column_matrix(acc_primitive_cell),
        tol=acc_primitive_ossg.symbol_calibration_tol,
    )
    (
        acc_primitive_oriented_seitz,
        acc_primitive_oriented_seitz_latex,
    ) = _seitz_symbols_from_descriptions(acc_primitive_oriented_seitz_descriptions)
    acc_primitive_cartesian_ops_payload = _serialize_ssg_operation_matrices(
        list(acc_primitive_ssg.ops)
    )
    acc_primitive_oriented_ops_payload = _serialize_ssg_operation_matrices(
        list(acc_primitive_ossg.ops)
    )
    operation_views = _build_operation_views(
        {
            "magnetic_primitive_cartesian": {
                "ssg": acc_primitive_ssg,
                "ops_payload": acc_primitive_cartesian_ops_payload,
                "seitz_latex": acc_primitive_ssg.seitz_symbols_latex,
                "setting_label": ACC_PRIMITIVE_SETTING,
                "spin_frame": "cartesian",
            },
            "magnetic_primitive_oriented": {
                "ssg": acc_primitive_ossg,
                "ops_payload": acc_primitive_oriented_ops_payload,
                "seitz_latex": acc_primitive_oriented_seitz_latex,
                "setting_label": ACC_PRIMITIVE_SETTING,
                "spin_frame": OSSG_ORIENTED_SPIN_FRAME_SETTING,
            },
        }
    )
    magnetic_phase_payload = classify_magnetic_phase(
        conf=ssg_primitive.conf,
        full_spin_part_point_group_hm=ssg_primitive.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=ssg_primitive.spin_part_point_group_symbol_s,
        net_moment=magnetic_primitive_cell.net_moment,
        net_moment_tol=tol_cfg.moment,
        mpg_identifier=selected_standard_ossg.mpg_num,
        is_ss_gp=ssg_primitive.is_spinsplitting[-1],
    )
    spin_texture_config = _spin_texture_config_for_public_output(identify_info)
    spin_texture_config_no_soc, spin_texture_config_soc = _spin_texture_config_from_ossg_convention(
        selected_standard_ossg,
        selected_standard_cell,
        tol=tol_cfg.m_matrix_tol,
        calibration_atol_limit=max(tol_cfg.m_matrix_tol, tol_cfg.moment),
        reference=spin_texture_config,
    )

    return {
        "index": identify_info,
        "identify_index_details": identify_index_details,
        "acc_symbol": ssg_primitive.acc,
        "conf": ssg_primitive.conf,
        "spin_texture_config": spin_texture_config,
        "spin_texture_config_no_soc": spin_texture_config_no_soc,
        "spin_texture_config_soc": spin_texture_config_soc,
        "quasi_2d": None,
        "operation_views": operation_views,
        "acc_primitive_resolution_audit": acc_primitive_resolution_audit,
        "acc_primitive_standard_setting": selected_standard_setting,
        "acc_primitive_cell_setting": ACC_PRIMITIVE_SETTING,
        "acc_primitive_cell_detail": _serialize_cell_snapshot(
            acc_primitive_cell,
            site_order=acc_primitive_wp_site_order,
        ),
        "acc_primitive_poscar": acc_primitive_poscar,
        "acc_primitive_ssg_setting": ACC_PRIMITIVE_SETTING,
        "acc_primitive_ssg_international_linear": acc_primitive_ssg.international_symbol_linear,
        "acc_primitive_ssg_operation_matrices": acc_primitive_cartesian_ops_payload,
        "acc_primitive_ssg_ops_cartesian": acc_primitive_cartesian_ops_payload,
        "acc_primitive_ssg_seitz_cartesian": acc_primitive_ssg.seitz_symbols,
        "acc_primitive_ssg_seitz_latex_cartesian": acc_primitive_ssg.seitz_symbols_latex,
        "acc_primitive_ssg_ops_oriented": acc_primitive_oriented_ops_payload,
        "acc_primitive_ssg_seitz_oriented": acc_primitive_oriented_seitz,
        "acc_primitive_ssg_seitz_latex_oriented": acc_primitive_oriented_seitz_latex,
        "acc_primitive_poscar_spin_frame_setting": ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
        "acc_primitive_poscar_spin_frame_ssg_operation_matrices": _serialize_ssg_operation_matrices(
            list(acc_primitive_ssg_in_poscar_spin_frame.ops)
        ),
        "acc_primitive_spin_only_direction_cartesian": _format_spin_only_direction(
            acc_primitive_ssg.sog_direction
        ),
        "acc_primitive_spin_only_direction_poscar_spin_frame": _format_spin_only_direction(
            acc_primitive_ssg_in_poscar_spin_frame.sog_direction
        ),
        "acc_primitive_wp_chain": acc_primitive_wp_chain,
        "acc_primitive_real_cartesian_to_poscar_spin_frame": np.asarray(
            acc_real_cartesian_to_poscar_spin_frame, dtype=float
        ).tolist(),
        "poscar_spin_frame_to_acc_primitive_real_cartesian": np.asarray(
            poscar_spin_frame_to_acc_real_cartesian, dtype=float
        ).tolist(),
        "T_input_to_acc_primitive": (
            np.asarray(transformation_input_to_acc_primitive[0], dtype=float).tolist(),
            np.asarray(transformation_input_to_acc_primitive[1], dtype=float).tolist(),
        ),
        "T_acc_primitive_to_G0std": (
            np.asarray(transformation_acc_primitive_to_G0std[0], dtype=float).tolist(),
            np.asarray(transformation_acc_primitive_to_G0std[1], dtype=float).tolist(),
        ),
        "T_acc_primitive_to_L0std": (
            np.asarray(transformation_acc_primitive_to_L0std[0], dtype=float).tolist(),
            np.asarray(transformation_acc_primitive_to_L0std[1], dtype=float).tolist(),
        ),
    }


def find_spin_group_from_data(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    source_metadata: dict | None = None,
    input_spin_setting: str = "in_lattice",
    space_tol = 0.02,
    mtol = 0.02,
    meigtol = 0.00002,
    matrix_tol = 0.01,
    calculation_mode: str | None = "3d",
    vacuum_axis: str | None = "c",
) -> MagSymmetryResult:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    return _find_spin_group_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        source_metadata=source_metadata,
        parser_atol=None,
        input_spin_setting=input_spin_setting,
        calculation_mode=calculation_mode,
        vacuum_axis=vacuum_axis,
    )


def find_spin_group_basic_from_data(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    input_spin_setting="in_lattice",
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    return _find_spin_group_basic_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_acc_primitive_from_data(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    input_spin_setting="in_lattice",
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    return _find_spin_group_acc_primitive_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def _has_explicit_magnetic_moments(moments, *, tol: float = 1e-8) -> bool:
    if moments is None:
        return False
    array = np.asarray(moments, dtype=float)
    if array.size == 0:
        return False
    if array.ndim == 1:
        return bool(np.linalg.norm(array) > tol)
    return bool(np.any(np.linalg.norm(array, axis=1) > tol))


def _find_spin_group_input_ssg_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    *,
    input_spin_setting: str,
    source_format: str,
) -> dict:
    if not _has_explicit_magnetic_moments(moments):
        raise ValueError(
            f"Input magnetic-SSG route requires explicit magnetic moments; none were found in {source_name}."
        )

    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    identify_cell = (
        input_cell
        if input_cell.spin_setting == "cartesian"
        else _cartesianized_input_cell(input_cell)
    )
    input_magnetic_primitive_cell, transformation_input_to_input_magnetic_primitive = (
        identify_cell.get_primitive_structure(magnetic=True)
    )
    primitive_transform = np.asarray(
        transformation_input_to_input_magnetic_primitive,
        dtype=float,
    )
    primitive_det = float(np.linalg.det(primitive_transform))
    is_input_magnetic_primitive = bool(np.isclose(abs(primitive_det), 1.0, atol=1e-6))

    primitive_identify_result = identify_spin_space_group_result(
        input_magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    primitive_ssg: SpinSpaceGroup = primitive_identify_result.ssg
    _assert_ssg_ops_consistency(
        "input route primitive SSG",
        primitive_ssg,
        tol=tol_cfg,
    )

    primitive_identify_info = None
    try:
        primitive_identify_info = primitive_ssg.identify_index(
            source_name,
            tol=tol_cfg.space,
        )
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        primitive_identify_info = _handle_missing_identify_index(source_name, exc)

    primitive_ossg = _ossg_oriented_spin_frame_ssg(primitive_ssg, input_magnetic_primitive_cell)

    if is_input_magnetic_primitive:
        primitive_to_input = _invert_setting_transform(
            primitive_transform,
            np.zeros(3),
        )
        input_ssg = primitive_ssg.transform(*primitive_to_input)
        _assert_ssg_ops_consistency(
            "input route input SSG",
            input_ssg,
            tol=tol_cfg,
        )
        identify_info = primitive_identify_info
    else:
        input_identify_result = identify_spin_space_group_result(
            identify_cell,
            find_primitive=False,
            tol=tol_cfg,
        )
        input_ssg = input_identify_result.ssg
        _assert_ssg_ops_consistency(
            "input route input SSG",
            input_ssg,
            tol=tol_cfg,
        )
        identify_info = None
        try:
            identify_info = input_ssg.identify_index(
                source_name,
                tol=tol_cfg.space,
            )
        except ValueError as exc:
            if not _should_degrade_identify_index_error(exc):
                raise
            identify_info = _handle_missing_identify_index(source_name, exc)

    input_ossg = _ossg_oriented_spin_frame_ssg(input_ssg, identify_cell)
    magnetic_phase_payload = classify_magnetic_phase(
        conf=input_ssg.conf,
        full_spin_part_point_group_hm=input_ssg.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=input_ssg.spin_part_point_group_symbol_s,
        net_moment=identify_cell.net_moment,
        net_moment_tol=tol_cfg.moment,
        mpg_identifier=input_ossg.mpg_num,
        is_ss_gp=input_ssg.is_spinsplitting[-1],
    )
    warning = None
    if not is_input_magnetic_primitive:
        warning = (
            "Input cell is not a magnetic primitive cell; the input-cell SSG may be missing "
            "symmetry operations relative to the magnetic primitive setting."
        )

    input_poscar = None
    if source_format != "poscar":
        input_poscar = _cell_to_poscar_preserving_lattice(
            identify_cell,
            Path(source_name).name,
        )

    magnetic_primitive_poscar = None
    if not is_input_magnetic_primitive:
        magnetic_primitive_poscar = _cell_to_poscar_preserving_lattice(
            input_magnetic_primitive_cell,
            f"{Path(source_name).name}_magnetic_primitive"
        )

    return {
        "summary": {
            "input_ssg_index": identify_info,
            "primitive_ssg_index": primitive_identify_info,
            "input_conf": input_ssg.conf,
            "input_spin_only_direction": _format_spin_only_direction(input_ossg.sog_direction),
            "input_magnetic_phase": magnetic_phase_payload["phase"],
            "input_ssg_database_symbol": input_ssg.international_symbol_linear,
            "input_msg_num": input_ossg.msg_int_num,
            "primitive_msg_num": primitive_ossg.msg_int_num,
            "input_msg_bns_number": input_ossg.msg_bns_num,
            "primitive_msg_bns_number": primitive_ossg.msg_bns_num,
            "input_msg_symbol": input_ossg.msg_bns_symbol,
            "is_input_magnetic_primitive": is_input_magnetic_primitive,
            "input_ssg_may_be_incomplete": not is_input_magnetic_primitive,
            "warning": warning,
        },
        "ssg": {
            "setting": INPUT_POSCAR_SETTING,
            "spin_frame_setting": identify_cell.spin_setting,
            "ops": _serialize_ssg_operation_matrices(list(input_ssg.ops)),
        },
        "msg": {
            "setting": INPUT_POSCAR_SETTING,
            "spin_frame_setting": OSSG_ORIENTED_SPIN_FRAME_SETTING,
            "ops": _serialize_msg_operation_matrices(list(input_ossg.msg_ops), tol=input_ossg.tol),
        },
        "primitive_relation": {
            "T_input_to_input_magnetic_primitive": primitive_transform.tolist(),
            "determinant": primitive_det,
        },
        "quasi_2d": None,
        "input_poscar": input_poscar,
        "magnetic_primitive_poscar": magnetic_primitive_poscar,
    }


def find_spin_group(
    cif: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    parser_atol=0.02,
    calculation_mode: str | None = "3d",
    vacuum_axis: str | None = "c",
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> MagSymmetryResult:
    """
    Find the spin space group of a crystal structure given in a CIF file.

    Parameters:
    cif (str): Path to the CIF file.
    space_tol (float): Tolerance for space group determination.
    mtol (float): Tolerance for magnetic moment determination.
    meigtol (float): Tolerance for eigenvalue determination.
    matrix_tol (float): Tolerance for point-group standardization matrices.
    parser_atol (float): Parsing tolerance for CIF / SCIF structure expansion.

    Returns:
    dict: A dictionary containing the spin space group information and related data.
    """

    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    parsed, source_metadata = parse_structure_file(
        cif,
        atol=parser_atol,
        return_metadata=True,
        poscar_allow_incar_magmom=poscar_allow_incar_magmom,
        poscar_prefer_incar_magmom=poscar_prefer_incar_magmom,
    )
    lattice_factors,positions, elements, occupancies, labels, moments = parsed
    input_spin_setting = (
        "in_lattice" if source_metadata is None else source_metadata.get("spin_setting", "in_lattice")
    )
    return _find_spin_group_from_parsed(
        cif,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        source_metadata=source_metadata,
        parser_atol=parser_atol,
        input_spin_setting=input_spin_setting,
        calculation_mode=calculation_mode,
        vacuum_axis=vacuum_axis,
    )


def find_spin_group_basic(
    cif: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    parser_atol=0.02,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    parsed, _source_metadata = parse_structure_file(
        cif,
        atol=parser_atol,
        return_metadata=True,
        poscar_allow_incar_magmom=poscar_allow_incar_magmom,
        poscar_prefer_incar_magmom=poscar_prefer_incar_magmom,
    )
    lattice_factors, positions, elements, occupancies, labels, moments = parsed
    input_spin_setting = (
        "in_lattice" if _source_metadata is None else _source_metadata.get("spin_setting", "in_lattice")
    )
    return _find_spin_group_basic_from_parsed(
        cif,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_acc_primitive(
    cif: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    parser_atol=0.02,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    parsed, _source_metadata = parse_structure_file(
        cif,
        atol=parser_atol,
        return_metadata=True,
        poscar_allow_incar_magmom=poscar_allow_incar_magmom,
        poscar_prefer_incar_magmom=poscar_prefer_incar_magmom,
    )
    lattice_factors, positions, elements, occupancies, labels, moments = parsed
    input_spin_setting = (
        "in_lattice" if _source_metadata is None else _source_metadata.get("spin_setting", "in_lattice")
    )
    return _find_spin_group_acc_primitive_from_parsed(
        cif,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_input_ssg(
    structure_file: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> dict:
    """
    Identify the spin-space-group operations in the input cell setting.

    This route is intended for file-facing workflows that need the symmetry
    operations of the cell supplied by the user, not only the standardized or
    accepted primitive setting used by the full pipeline. The returned payload
    includes input-cell SSG operations, input-cell oriented MSG operations,
    summary identifiers, the relation to the input magnetic primitive cell, and
    POSCAR text outputs when useful.

    If the supplied cell is not already a magnetic primitive cell, the returned
    input-cell SSG operations may be incomplete relative to the full symmetry of
    the magnetic primitive cell. In that case the payload also includes
    primitive-side identifiers and a warning, so callers can distinguish the
    input-cell answer from the magnetic-primitive reference.

    POSCAR inputs must contain an embedded ``MAGMOM`` payload by default.
    Callers may opt into reading a sibling INCAR by setting
    ``poscar_allow_incar_magmom=True``. CIF, mCIF, and SCIF inputs must contain
    explicit magnetic moments. POSCAR moments are treated as Cartesian, while
    CIF/mCIF/SCIF moments are converted into the route's Cartesian input-cell
    frame before identification and export.
    """
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    path = Path(structure_file)
    suffix = path.suffix.lower()
    basename = path.name.lower()
    if suffix in {".vasp", ".poscar"} or basename in {"poscar", "contcar"}:
        lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(
            structure_file,
            allow_incar_magmom=poscar_allow_incar_magmom,
            prefer_incar_magmom=poscar_prefer_incar_magmom,
            require_embedded_magmom=not poscar_allow_incar_magmom,
        )
        source_format = "poscar"
        input_spin_setting = "cartesian"
    else:
        parsed, source_metadata = parse_structure_file(
            structure_file,
            return_metadata=True,
        )
        lattice_factors, positions, elements, occupancies, labels, moments = parsed
        source_format = "unknown" if source_metadata is None else source_metadata.get("source_format", "unknown")
        input_spin_setting = (
            "in_lattice" if source_metadata is None else source_metadata.get("spin_setting", "in_lattice")
        )
    return _find_spin_group_input_ssg_from_parsed(
        structure_file,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
        source_format=source_format,
    )


def find_spin_group_poscar_ssg(
    poscar: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> dict:
    return find_spin_group_input_ssg(
        poscar,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
        poscar_allow_incar_magmom=poscar_allow_incar_magmom,
        poscar_prefer_incar_magmom=poscar_prefer_incar_magmom,
    )


def write_ssg_operation_matrices(path: str | Path, operations: list[dict]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(operations, indent=2, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    return output_path


def _is_json_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_structured_json(value, indent: int = 0) -> str:
    current_indent = "  " * indent
    next_indent = "  " * (indent + 1)

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        items = list(value.items())
        for idx, (key, item) in enumerate(items):
            comma = "," if idx < len(items) - 1 else ""
            rendered = _format_structured_json(item, indent + 1)
            if "\n" not in rendered:
                lines.append(f"{next_indent}{json.dumps(key, ensure_ascii=False)}: {rendered}{comma}")
                continue
            rendered_lines = rendered.splitlines()
            lines.append(f"{next_indent}{json.dumps(key, ensure_ascii=False)}: {rendered_lines[0]}")
            lines.extend(rendered_lines[1:-1])
            lines.append(f"{rendered_lines[-1]}{comma}")
        lines.append(f"{current_indent}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(_is_json_scalar(item) for item in value):
            return json.dumps(value, ensure_ascii=False)
        if all(isinstance(item, list) and all(_is_json_scalar(entry) for entry in item) for item in value):
            lines = ["["]
            for idx, row in enumerate(value):
                comma = "," if idx < len(value) - 1 else ""
                lines.append(f"{next_indent}{json.dumps(row, ensure_ascii=False)}{comma}")
            lines.append(f"{current_indent}]")
            return "\n".join(lines)
        lines = ["["]
        for idx, item in enumerate(value):
            comma = "," if idx < len(value) - 1 else ""
            rendered = _format_structured_json(item, indent + 1)
            if "\n" not in rendered:
                lines.append(f"{next_indent}{rendered}{comma}")
                continue
            rendered_lines = rendered.splitlines()
            lines.append(f"{next_indent}{rendered_lines[0]}")
            lines.extend(rendered_lines[1:-1])
            lines.append(f"{rendered_lines[-1]}{comma}")
        lines.append(f"{current_indent}]")
        return "\n".join(lines)

    return json.dumps(value, ensure_ascii=False)


def write_poscar_ssg_symmetry_dat(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "summary": payload.get("summary") or {},
        "ssg": payload.get("ssg") or {},
        "msg": payload.get("msg") or {},
        "primitive_relation": payload.get("primitive_relation") or {},
        "input_poscar": payload.get("input_poscar"),
        "magnetic_primitive_poscar": payload.get("magnetic_primitive_poscar"),
        "format": "findspingroup.poscar_ssg.v1",
    }

    normalized_document = json.loads(json.dumps(document, ensure_ascii=False, cls=NumpyEncoder))

    output_path.write_text(
        _format_structured_json(normalized_document) + "\n",
        encoding="utf-8",
    )
    return output_path
