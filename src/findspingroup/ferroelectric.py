from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import spglib

from findspingroup.io.scif_generator import affine_matrix_to_xyz_expression
from findspingroup.utils.matrix_utils import (
    normalize_vector_to_zero,
)
from findspingroup.utils.space_group_flags import (
    format_polar_axis_vector,
    space_group_has_real_space_inversion,
    space_group_is_polar,
    space_group_polar_axis_basis,
)


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _polar_axis_payload(
    space_group_number: int | None,
    *,
    setting: str,
) -> list[dict[str, Any]] | None:
    basis = space_group_polar_axis_basis(space_group_number)
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


def _space_group_payload(
    *,
    source: str,
    space_group_number: int | None,
    space_group_symbol: str | None = None,
    msg_num: int | None = None,
    msg_symbol: str | None = None,
) -> dict[str, Any]:
    standard_direct_basis = f"{source}_space_group_standard_direct_basis"
    return {
        "source": source,
        "msg_num": msg_num,
        "msg_symbol": msg_symbol,
        "space_group_number": space_group_number,
        "space_group_symbol": space_group_symbol,
        "is_polar": space_group_is_polar(space_group_number),
        "has_real_space_inversion": space_group_has_real_space_inversion(space_group_number),
        "allowed_polar_axes": (
            [] if space_group_number is None else _polar_axis_payload(
                space_group_number,
                setting=standard_direct_basis,
            )
        ),
        "allowed_polar_axes_setting": standard_direct_basis,
    }


def _axis_labels(space_group_number: int | None) -> tuple[str, ...] | None:
    basis = space_group_polar_axis_basis(space_group_number)
    if basis is None:
        return None
    return tuple(format_polar_axis_vector(vector) for vector in basis)


def _canonical_translation(t: Any, *, tol: float) -> np.ndarray:
    translation = np.mod(np.asarray(t, dtype=float), 1.0)
    translation[np.isclose(translation, 1.0, atol=tol)] = 0.0
    translation[np.isclose(translation, 0.0, atol=tol)] = 0.0
    return translation


def _real_op_same(
    op_a: tuple[np.ndarray, np.ndarray],
    op_b: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> bool:
    rot_a, trans_a = op_a
    rot_b, trans_b = op_b
    return bool(
        np.allclose(rot_a, rot_b, atol=tol, rtol=0)
        and np.max(
            np.abs(
                _canonical_translation(trans_a, tol=tol)
                - _canonical_translation(trans_b, tol=tol)
            )
        )
        < tol
    )


def _dedupe_real_ops(ops: Any, *, tol: float) -> list[tuple[np.ndarray, np.ndarray]]:
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for op in ops:
        rot, trans = op
        candidate = (np.asarray(rot, dtype=float), np.asarray(trans, dtype=float))
        if not any(_real_op_same(candidate, existing, tol=tol) for existing in unique):
            unique.append(candidate)
    return unique


def _multiply_real_ops(
    op_left: tuple[np.ndarray, np.ndarray],
    op_right: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    rot_left, trans_left = op_left
    rot_right, trans_right = op_right
    return (
        np.asarray(rot_left, dtype=float) @ np.asarray(rot_right, dtype=float),
        _canonical_translation(
            np.asarray(rot_left, dtype=float) @ np.asarray(trans_right, dtype=float)
            + np.asarray(trans_left, dtype=float),
            tol=tol,
        ),
    )


def _contains_real_op(
    ops: list[tuple[np.ndarray, np.ndarray]],
    candidate: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> bool:
    return any(_real_op_same(candidate, existing, tol=tol) for existing in ops)


def _left_coset_partition(
    group_ops: list[tuple[np.ndarray, np.ndarray]],
    subgroup_ops: list[tuple[np.ndarray, np.ndarray]],
    *,
    tol: float,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Return left cosets gH, where H is the ordered-domain stabilizer."""

    unused = list(group_ops)
    cosets: list[list[tuple[np.ndarray, np.ndarray]]] = []
    while unused:
        representative = unused[0]
        coset = [
            _multiply_real_ops(representative, subgroup_op, tol=tol)
            for subgroup_op in subgroup_ops
        ]
        coset = _dedupe_real_ops(coset, tol=tol)
        cosets.append(coset)
        unused = [op for op in unused if not _contains_real_op(coset, op, tol=tol)]
    return cosets


def _op_key(op: tuple[np.ndarray, np.ndarray], *, tol: float) -> tuple[Any, ...]:
    rot = np.rint(np.asarray(op[0], dtype=float)).astype(int)
    trans = _canonical_translation(op[1], tol=tol)
    return tuple(rot.flatten().tolist() + [round(float(x), 8) for x in trans.tolist()])


def _magnetic_op_same(
    op_a: tuple[np.ndarray, np.ndarray, int],
    op_b: tuple[np.ndarray, np.ndarray, int],
    *,
    tol: float,
) -> bool:
    return int(op_a[2]) == int(op_b[2]) and _real_op_same(
        (op_a[0], op_a[1]),
        (op_b[0], op_b[1]),
        tol=tol,
    )


def _dedupe_magnetic_ops(
    ops: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    tol: float,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    unique: list[tuple[np.ndarray, np.ndarray, int]] = []
    for rot, trans, time_reversal in ops:
        candidate = (
            np.asarray(rot, dtype=float),
            _canonical_translation(trans, tol=tol),
            int(time_reversal),
        )
        if not any(_magnetic_op_same(candidate, existing, tol=tol) for existing in unique):
            unique.append(candidate)
    return unique


def _magnetic_operation_payload(
    op: tuple[np.ndarray, np.ndarray, int],
    *,
    tol: float,
) -> dict[str, Any]:
    payload = _operation_payload((op[0], op[1]), tol=tol)
    payload["time_reversal"] = int(op[2])
    return payload


def _vector_in_lattice(vector: np.ndarray, lattice_basis: np.ndarray, *, tol: float) -> bool:
    coordinates = np.linalg.inv(np.asarray(lattice_basis, dtype=float)) @ np.asarray(vector, dtype=float)
    return bool(np.allclose(coordinates, np.rint(coordinates), atol=tol, rtol=0.0))


def _integer_lattice_translation_representatives(
    lattice_basis: np.ndarray,
    *,
    tol: float,
) -> list[np.ndarray]:
    """Return representatives of Z^3 / lattice_basis Z^3.

    The basis is expected to be a finite-index integer sublattice in the parent
    standard direct basis.  A small bounded search is enough for the current
    magnetic supercell indices, and the result is cached by the caller.
    """

    matrix = np.asarray(lattice_basis, dtype=float)
    det = max(1, int(round(abs(float(np.linalg.det(matrix))))))
    representatives: list[np.ndarray] = []
    bound = max(1, det)
    while len(representatives) < det:
        for i in range(-bound, bound + 1):
            for j in range(-bound, bound + 1):
                for k in range(-bound, bound + 1):
                    candidate = np.array([i, j, k], dtype=float)
                    if any(
                        _vector_in_lattice(candidate - existing, matrix, tol=tol)
                        for existing in representatives
                    ):
                        continue
                    representatives.append(candidate)
                    if len(representatives) == det:
                        return representatives
        bound *= 2
        if bound > 8 * det + 16:
            raise ValueError("unable to enumerate parent/ordered translation representatives")
    return representatives


def _parent_candidate_same_left_coset(
    candidate_left: tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray],
    candidate_right: tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray],
    subgroup_parent_ops: list[tuple[np.ndarray, np.ndarray, int]],
    child_basis_in_parent: np.ndarray,
    *,
    tol: float,
) -> bool:
    left_rotation, left_translation, left_time, _, _ = candidate_left
    right_rotation, right_translation, right_time, _, _ = candidate_right
    if int(left_time) != int(right_time):
        return False

    relative_rotation = np.linalg.inv(left_rotation) @ right_rotation
    relative_translation = np.linalg.inv(left_rotation) @ (
        np.asarray(right_translation, dtype=float)
        - np.asarray(left_translation, dtype=float)
    )
    for subgroup_rotation, subgroup_translation, subgroup_time in subgroup_parent_ops:
        if int(subgroup_time) != 1:
            continue
        if not np.allclose(relative_rotation, subgroup_rotation, atol=tol, rtol=0.0):
            continue
        if _vector_in_lattice(
            relative_translation - subgroup_translation,
            child_basis_in_parent,
            tol=tol,
        ):
            return True
    return False


def _parent_left_coset_partition(
    parent_candidates: list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]],
    subgroup_parent_ops: list[tuple[np.ndarray, np.ndarray, int]],
    child_basis_in_parent: np.ndarray,
    *,
    tol: float,
) -> list[list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]]:
    unused = list(range(len(parent_candidates)))
    cosets: list[list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]] = []
    while unused:
        representative_index = unused[0]
        coset_indices = [
            index
            for index in unused
            if _parent_candidate_same_left_coset(
                parent_candidates[representative_index],
                parent_candidates[index],
                subgroup_parent_ops,
                child_basis_in_parent,
                tol=tol,
            )
        ]
        cosets.append([parent_candidates[index] for index in coset_indices])
        coset_index_set = set(coset_indices)
        unused = [index for index in unused if index not in coset_index_set]
    return cosets


def _parent_space_group_contains(
    parent_database_ops: list[tuple[np.ndarray, np.ndarray]],
    candidate: tuple[np.ndarray, np.ndarray, int],
    *,
    tol: float,
) -> bool:
    candidate_rotation, candidate_translation, _candidate_time = candidate
    identity_lattice = np.eye(3)
    return any(
        np.allclose(candidate_rotation, parent_rotation, atol=tol, rtol=0.0)
        and _vector_in_lattice(
            candidate_translation - parent_translation,
            identity_lattice,
            tol=tol,
        )
        for parent_rotation, parent_translation in parent_database_ops
    )


def _coset_keys(
    cosets: list[list[tuple[np.ndarray, np.ndarray]]],
    *,
    tol: float,
) -> set[tuple[tuple[Any, ...], ...]]:
    return {
        tuple(sorted(_op_key(op, tol=tol) for op in coset))
        for coset in cosets
    }


def _operation_payload(
    op: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> dict[str, Any]:
    rot, trans = op
    rot_array = np.asarray(rot, dtype=float)
    trans_array = _canonical_translation(trans, tol=tol)
    return {
        "real_rotation": rot_array.tolist(),
        "translation": trans_array.tolist(),
        "xyzt": affine_matrix_to_xyz_expression(rot_array, trans_array),
    }


def _axis_reversal_payload(
    rotation: np.ndarray,
    axes: tuple[tuple[float, float, float], ...],
    *,
    tol: float,
) -> tuple[bool, list[str]]:
    reversed_axis_labels = []
    for axis in axes:
        axis_array = np.asarray(axis, dtype=float)
        if np.allclose(rotation @ axis_array, -axis_array, atol=tol, rtol=0):
            reversed_axis_labels.append(format_polar_axis_vector(axis))

    if reversed_axis_labels:
        return True, reversed_axis_labels

    if not axes:
        return False, []

    basis = np.asarray(axes, dtype=float).T
    rank = np.linalg.matrix_rank(rotation @ basis + basis, tol=tol)
    has_reversal_vector = rank < basis.shape[1]
    return bool(has_reversal_vector), []


def _axis_relation_payload(
    rotation: np.ndarray,
    axes: tuple[tuple[float, float, float], ...],
    *,
    tol: float,
) -> tuple[str, list[str]]:
    reversed_status, reversed_axes = _axis_reversal_payload(rotation, axes, tol=tol)
    if reversed_status:
        return "P -> -P", reversed_axes

    preserved_axis_labels = []
    for axis in axes:
        axis_array = np.asarray(axis, dtype=float)
        if np.allclose(rotation @ axis_array, axis_array, atol=tol, rtol=0):
            preserved_axis_labels.append(format_polar_axis_vector(axis))
    if preserved_axis_labels:
        return "P -> P", preserved_axis_labels
    return "P -> other", []


def _periodic_norm_inf(vector: np.ndarray) -> float:
    delta = np.asarray(vector, dtype=float)
    delta = delta - np.rint(delta)
    return float(np.max(np.abs(delta)))


def _position_bucket_key(position: np.ndarray, bins: int) -> tuple[int, int, int]:
    values = np.floor((np.asarray(position, dtype=float) % 1.0) * bins).astype(int)
    return tuple((values % bins).tolist())


def _position_neighbor_keys(key: tuple[int, int, int], bins: int):
    for offset in product((-1, 0, 1), repeat=3):
        yield tuple((key[index] + offset[index]) % bins for index in range(3))


def _site_lookup(
    positions: np.ndarray,
    atom_types: list[int],
    *,
    tol: float,
) -> tuple[dict[tuple[int, tuple[int, int, int]], list[int]], int]:
    bins = max(8, int(np.ceil(1.0 / max(float(tol), 1e-5))))
    lookup: dict[tuple[int, tuple[int, int, int]], list[int]] = {}
    for index, (position, atom_type) in enumerate(zip(positions, atom_types)):
        key = (int(atom_type), _position_bucket_key(position, bins))
        lookup.setdefault(key, []).append(index)
    return lookup, bins


def _match_transformed_sites(
    positions: np.ndarray,
    atom_types: list[int],
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    tol: float,
) -> list[int] | None:
    lookup, bins = _site_lookup(positions, atom_types, tol=tol)
    mapping = [-1] * len(positions)
    used: set[int] = set()
    for source_index, (position, atom_type) in enumerate(zip(positions, atom_types)):
        target_position = normalize_vector_to_zero(
            np.asarray(rotation, dtype=float) @ np.asarray(position, dtype=float)
            + np.asarray(translation, dtype=float),
            atol=tol,
        ) % 1.0
        bucket_key = _position_bucket_key(target_position, bins)
        matched_index = None
        for neighbor_key in _position_neighbor_keys(bucket_key, bins):
            for candidate_index in lookup.get((int(atom_type), neighbor_key), ()):
                if candidate_index in used:
                    continue
                if _periodic_norm_inf(target_position - positions[candidate_index]) < tol:
                    matched_index = candidate_index
                    break
            if matched_index is not None:
                break
        if matched_index is None:
            return None
        mapping[source_index] = matched_index
        used.add(matched_index)
    return mapping


def _collinear_pattern_context(
    ordered_cell: Any | None,
    collinear_axis: Any | None,
    *,
    tol: float,
) -> dict[str, Any] | None:
    if ordered_cell is None or collinear_axis is None:
        return None
    moments = getattr(ordered_cell, "moments_cartesian", None)
    if moments is None:
        return None
    positions = np.asarray(getattr(ordered_cell, "positions"), dtype=float)
    atom_types = [int(item) for item in getattr(ordered_cell, "atom_types")]
    axis = np.asarray(collinear_axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < tol:
        return None
    axis = axis / axis_norm
    scalars = np.asarray(moments, dtype=float) @ axis
    signed_pattern = np.zeros(len(scalars), dtype=int)
    signed_pattern[scalars > tol] = 1
    signed_pattern[scalars < -tol] = -1
    if not np.any(signed_pattern):
        return None
    lattice = np.asarray(getattr(ordered_cell, "lattice_matrix"), dtype=float)
    return {
        "positions": positions,
        "atom_types": atom_types,
        "axis": axis,
        "signed_pattern": signed_pattern,
        "lattice": lattice,
    }


def _transformed_collinear_pattern(
    context: dict[str, Any] | None,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    spin_branch: int,
    tol: float,
) -> tuple[np.ndarray | None, str]:
    if context is None:
        return None, "not_evaluated_missing_collinear_pattern"
    mapping = _match_transformed_sites(
        context["positions"],
        context["atom_types"],
        rotation,
        translation,
        tol=tol,
    )
    if mapping is None:
        return None, "not_evaluated_site_mapping_failed"

    source_pattern = np.asarray(context["signed_pattern"], dtype=int)
    transformed = np.zeros_like(source_pattern)
    assigned = np.zeros(len(source_pattern), dtype=bool)
    for source_index, target_index in enumerate(mapping):
        transformed[target_index] = int(spin_branch) * int(source_pattern[source_index])
        assigned[target_index] = True
    if not np.all(assigned):
        return None, "not_evaluated_site_mapping_incomplete"
    return transformed, "ok"


def _collinear_pattern_relation(
    transformed_pattern: np.ndarray | None,
    context: dict[str, Any] | None,
) -> str:
    if transformed_pattern is None or context is None:
        return "not_evaluated"
    source_pattern = np.asarray(context["signed_pattern"], dtype=int)
    if np.array_equal(transformed_pattern, source_pattern):
        return "L -> L"
    if np.array_equal(transformed_pattern, -source_pattern):
        return "L -> -L"
    return "signed_pattern_changed"


def _pattern_key(pattern: np.ndarray | None) -> str | None:
    if pattern is None:
        return None
    return ",".join(str(int(item)) for item in np.asarray(pattern, dtype=int).tolist())


def _relation_label(p_relation: str, magnetic_relation: str) -> str:
    if p_relation == "P -> -P" and magnetic_relation == "L -> -L":
        return "p_and_magnetic_order_reversed"
    if p_relation == "P -> -P" and magnetic_relation == "L -> L":
        return "p_reversed_magnetic_order_preserved"
    if p_relation == "P -> P" and magnetic_relation == "L -> -L":
        return "p_preserved_magnetic_order_reversed"
    return "other_collinear_relation"


def _spin_branch_relation(spin_branch: int) -> str:
    return "S -> S" if int(spin_branch) == 1 else "S -> -S"


def _spin_branch_relation_label(p_relation: str, spin_branch: int) -> str:
    branch_relation = _spin_branch_relation(spin_branch)
    if p_relation == "P -> -P" and branch_relation == "S -> -S":
        return "p_and_spin_branch_reversed"
    if p_relation == "P -> -P" and branch_relation == "S -> S":
        return "p_reversed_spin_branch_preserved"
    if p_relation == "P -> P" and branch_relation == "S -> -S":
        return "p_preserved_spin_branch_reversed"
    return "other_spin_branch_relation"


def _cartesian_rotation_from_fractional(
    rotation: np.ndarray,
    lattice_matrix: np.ndarray | None,
) -> np.ndarray:
    if lattice_matrix is None:
        return np.asarray(rotation, dtype=float)
    direct_to_cartesian = np.asarray(lattice_matrix, dtype=float).T
    return direct_to_cartesian @ np.asarray(rotation, dtype=float) @ np.linalg.inv(direct_to_cartesian)


def _msg_compatible_collinear_branch(
    *,
    rotation: np.ndarray,
    time_reversal: int,
    spin_branch: int,
    context: dict[str, Any] | None,
    tol: float,
) -> bool | None:
    if context is None:
        return None
    lattice = context.get("lattice")
    axis = np.asarray(context["axis"], dtype=float)
    rotation_cart = _cartesian_rotation_from_fractional(rotation, lattice)
    det_sign = 1 if np.linalg.det(rotation_cart) >= 0 else -1
    locked_spin = int(time_reversal) * det_sign * rotation_cart
    locked_axis = locked_spin @ axis
    norm = float(np.linalg.norm(locked_axis))
    if norm < tol:
        return None
    locked_axis = locked_axis / norm
    target_axis = int(spin_branch) * axis
    return bool(np.allclose(locked_axis, target_axis, atol=tol, rtol=0.0))


def _collinear_branch_relation_payloads(
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    time_reversal: int,
    p_relation: str,
    p_axis_labels: list[str],
    context: dict[str, Any] | None,
    tol: float,
) -> list[dict[str, Any]]:
    payloads = []
    for spin_branch in (1, -1):
        transformed_pattern, pattern_status = _transformed_collinear_pattern(
            context,
            rotation,
            translation,
            spin_branch=spin_branch,
            tol=tol,
        )
        magnetic_relation = _collinear_pattern_relation(transformed_pattern, context)
        msg_compatible = _msg_compatible_collinear_branch(
            rotation=rotation,
            time_reversal=time_reversal,
            spin_branch=spin_branch,
            context=context,
            tol=tol,
        )
        valid_domain_relation = pattern_status == "ok"
        representative_class = (
            "not_evaluated"
            if msg_compatible is None
            else "msg_compatible"
            if msg_compatible
            else "exchange_only"
        )
        payloads.append(
            {
                "spin_space_operation": "+1" if spin_branch == 1 else "-1",
                "spin_branch": int(spin_branch),
                "spin_branch_relation": _spin_branch_relation(spin_branch),
                "spin_branch_relation_label": _spin_branch_relation_label(
                    p_relation,
                    spin_branch,
                ),
                "p_relation": p_relation,
                "p_axis_labels": list(p_axis_labels),
                "magnetic_order_relation": magnetic_relation,
                "signed_collinear_pattern_relation": magnetic_relation,
                "relation_label": _relation_label(p_relation, magnetic_relation),
                "pattern_status": pattern_status,
                "transformed_pattern_key": _pattern_key(transformed_pattern),
                "msg_compatible": msg_compatible,
                "representative_class": representative_class,
                "soc_allowed": bool(msg_compatible) if valid_domain_relation else False,
                "exchange_only": (
                    bool(msg_compatible is False) if valid_domain_relation else False
                ),
                "valid_domain_relation": valid_domain_relation,
            }
        )
    return payloads


def build_domain_reversal_coset_analysis(
    *,
    parent_ops: Any,
    ordered_ops: Any,
    ordered_space_group_number: int | None,
    parent_space_group_number: int | None = None,
    parent_space_group_symbol: str | None = None,
    basis_setting: str,
    msg_ops: Any | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Screen parent/ordered cosets for operations that can map P to -P.

    The coset object is the structural parent real-space group modulo the
    ordered spin-space real-space projection, both expressed in the same direct
    basis.  This is a symmetry-only domain-relation screen; structural path and
    barrier checks remain downstream.
    """

    axes = space_group_polar_axis_basis(ordered_space_group_number)
    if axes is None:
        return {
            "status": "not_evaluated_missing_ordered_space_group",
            "basis_setting": basis_setting,
            "candidate_reversal_domains": [],
        }
    if not axes:
        return {
            "status": "not_applicable_ordered_symmetry_nonpolar",
            "basis_setting": basis_setting,
            "candidate_reversal_domains": [],
        }

    parent_real_ops = _dedupe_real_ops(parent_ops, tol=tol)
    ordered_real_ops = _dedupe_real_ops(ordered_ops, tol=tol)
    msg_real_ops = [] if msg_ops is None else _dedupe_real_ops(msg_ops, tol=tol)
    ordered_subset = all(
        _contains_real_op(parent_real_ops, ordered_op, tol=tol)
        for ordered_op in ordered_real_ops
    )
    if not ordered_subset:
        unmatched = [
            _operation_payload(ordered_op, tol=tol)
            for ordered_op in ordered_real_ops
            if not _contains_real_op(parent_real_ops, ordered_op, tol=tol)
        ]
        return {
            "status": "not_evaluated_ordered_group_not_subset_of_parent",
            "basis_setting": basis_setting,
            "parent_space_group_number": parent_space_group_number,
            "parent_space_group_symbol": parent_space_group_symbol,
            "parent_operation_count": len(parent_real_ops),
            "ordered_operation_count": len(ordered_real_ops),
            "ordered_subset_of_parent": False,
            "unmatched_ordered_operation_count": len(unmatched),
            "unmatched_ordered_operations": unmatched[:8],
            "candidate_reversal_domains": [],
        }

    left_cosets = _left_coset_partition(parent_real_ops, ordered_real_ops, tol=tol)
    right_cosets = []
    unused = list(parent_real_ops)
    while unused:
        representative = unused[0]
        coset = [
            _multiply_real_ops(subgroup_op, representative, tol=tol)
            for subgroup_op in ordered_real_ops
        ]
        coset = _dedupe_real_ops(coset, tol=tol)
        right_cosets.append(coset)
        unused = [op for op in unused if not _contains_real_op(coset, op, tol=tol)]

    candidate_domains = []
    for coset_index, coset in enumerate(left_cosets):
        selected_op = None
        selected_axis_labels: list[str] = []
        generic_reversal = False
        for op in coset:
            has_reversal, axis_labels = _axis_reversal_payload(op[0], axes, tol=tol)
            if not has_reversal:
                continue
            selected_op = op
            selected_axis_labels = axis_labels
            generic_reversal = not axis_labels
            break
        if selected_op is None:
            continue

        msg_compatible = (
            None
            if not msg_real_ops
            else _contains_real_op(msg_real_ops, selected_op, tol=tol)
        )
        candidate_domains.append(
            {
                "coset_index": coset_index,
                "coset_size": len(coset),
                "representative": _operation_payload(selected_op, tol=tol),
                "maps_p_to_minus_p": True,
                "reversed_polar_axes": selected_axis_labels,
                "reverses_some_allowed_polar_vector": generic_reversal,
                "representative_class": (
                    "msg_compatible"
                    if msg_compatible is True
                    else "spin_domain_relation_pending"
                ),
                "msg_compatible": msg_compatible,
                "classification_note": (
                    "MSG compatibility for non-stabilizer parent cosets needs "
                    "the spin-domain map; this screen currently validates the "
                    "real-space P -> -P relation."
                ),
            }
        )

    status = (
        "candidate_reversal_domains_found"
        if candidate_domains
        else "no_parent_ordered_coset_maps_p_to_minus_p"
    )
    return {
        "status": status,
        "basis_setting": basis_setting,
        "parent_group_source": "nonmagnetic_space_group_of_ordered_standard_cell",
        "ordered_subgroup_source": "ordered_spin_space_real_space_projection",
        "parent_space_group_number": parent_space_group_number,
        "parent_space_group_symbol": parent_space_group_symbol,
        "ordered_space_group_number": ordered_space_group_number,
        "ordered_polar_axes": [
            {
                "label": format_polar_axis_vector(axis),
                "components": [float(component) for component in axis],
                "setting": basis_setting,
            }
            for axis in axes
        ],
        "parent_operation_count": len(parent_real_ops),
        "ordered_operation_count": len(ordered_real_ops),
        "ordered_subset_of_parent": True,
        "left_coset_convention": "domain cosets are gH with H the ordered subgroup",
        "left_coset_count": len(left_cosets),
        "left_coset_sizes": [len(coset) for coset in left_cosets],
        "right_coset_count": len(right_cosets),
        "right_coset_sizes": [len(coset) for coset in right_cosets],
        "left_equals_right": _coset_keys(left_cosets, tol=tol) == _coset_keys(right_cosets, tol=tol),
        "candidate_reversal_domain_count": len(candidate_domains),
        "candidate_reversal_domains": candidate_domains,
    }


def build_parent_standard_supercell_domain_coset_analysis(
    *,
    parent_space_group_number: int | None,
    parent_space_group_symbol: str | None,
    parent_hall_number: int | None,
    child_basis_in_parent: np.ndarray,
    child_origin_in_parent: np.ndarray,
    ordered_magnetic_ops: list[tuple[np.ndarray, np.ndarray, int]],
    ordered_space_group_number: int | None,
    basis_setting: str,
    ordered_cell: Any | None = None,
    collinear_axis: Any | None = None,
    tol: float = 1e-6,
) -> dict[str, Any] | None:
    """Build parent-grey / ordered-domain cosets in an internally generated basis.

    The parent group is generated from spglib's standard parent setting and
    transported into the current ordered standard supercell.  This deliberately
    does not use MCIF parent/child/BNS transform tags, which are source-file
    coordinate provenance rather than the FindSpinGroup coset convention.
    """

    if parent_space_group_number is None or parent_hall_number is None:
        return None

    axes = space_group_polar_axis_basis(ordered_space_group_number)
    if axes is None:
        return {
            "status": "not_evaluated_missing_ordered_space_group",
            "basis_setting": basis_setting,
            "candidate_reversal_domains": [],
        }
    if not axes:
        return {
            "status": "not_applicable_ordered_symmetry_nonpolar",
            "basis_setting": basis_setting,
            "candidate_reversal_domains": [],
        }

    child_to_parent = np.asarray(child_basis_in_parent, dtype=float)
    parent_to_child = np.linalg.inv(child_to_parent)
    child_origin_in_parent = np.asarray(child_origin_in_parent, dtype=float)
    collinear_context = _collinear_pattern_context(
        ordered_cell,
        collinear_axis,
        tol=tol,
    )

    parent_sym = spglib.get_symmetry_from_database(int(parent_hall_number))
    if parent_sym is None:
        return None

    parent_database_ops = [
        (np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float))
        for rotation, translation in zip(parent_sym["rotations"], parent_sym["translations"])
    ]
    translation_rep_cache: dict[tuple[float, ...], list[np.ndarray]] = {}
    parent_candidates: list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]] = []
    for rotation, translation in zip(parent_sym["rotations"], parent_sym["translations"]):
        parent_rotation = np.asarray(rotation, dtype=float)
        parent_translation = np.asarray(translation, dtype=float)
        child_rotation = parent_to_child @ parent_rotation @ child_to_parent
        child_translation = parent_to_child @ (
            parent_rotation @ child_origin_in_parent
            + parent_translation
            - child_origin_in_parent
        )
        translated_child_lattice = parent_rotation @ child_to_parent
        rep_key = tuple(np.round(translated_child_lattice.flatten(), 10).tolist())
        translation_reps = translation_rep_cache.setdefault(
            rep_key,
            _integer_lattice_translation_representatives(
                translated_child_lattice,
                tol=tol,
            ),
        )
        for parent_lattice_translation in translation_reps:
            parent_translation_with_lattice = parent_translation + parent_lattice_translation
            child_translation_with_lattice = parent_to_child @ (
                parent_rotation @ child_origin_in_parent
                + parent_translation_with_lattice
                - child_origin_in_parent
            )
            for time_reversal in (1, -1):
                parent_candidates.append(
                    (
                        parent_rotation,
                        parent_translation_with_lattice,
                        time_reversal,
                        child_rotation,
                        child_translation_with_lattice,
                    )
                )

    ordered_magnetic_ops = _dedupe_magnetic_ops(ordered_magnetic_ops, tol=tol)
    if not ordered_magnetic_ops:
        return None

    ordered_parent_ops = [
        (
            child_to_parent @ np.asarray(child_rotation, dtype=float) @ parent_to_child,
            child_to_parent @ np.asarray(child_translation, dtype=float)
            + child_origin_in_parent
            - (
                child_to_parent
                @ np.asarray(child_rotation, dtype=float)
                @ parent_to_child
            )
            @ child_origin_in_parent,
            int(time_reversal),
        )
        for child_rotation, child_translation, time_reversal in ordered_magnetic_ops
    ]

    ordered_subset = all(
        _parent_space_group_contains(parent_database_ops, ordered_op, tol=tol)
        for ordered_op in ordered_parent_ops
    )
    if not ordered_subset:
        unmatched = [
            _magnetic_operation_payload(ordered_op, tol=tol)
            for ordered_op in ordered_parent_ops
            if not _parent_space_group_contains(parent_database_ops, ordered_op, tol=tol)
        ]
        return {
            "status": "not_evaluated_ordered_group_not_subset_of_generated_parent_grey_group",
            "basis_setting": basis_setting,
            "parent_group_source": "spglib_standard_parent_lifted_to_ordered_standard_supercell",
            "ordered_subgroup_source": "ordered_spin_space_real_space_projection",
            "parent_space_group_number": parent_space_group_number,
            "parent_space_group_symbol": parent_space_group_symbol,
            "parent_hall_number": parent_hall_number,
            "child_basis_in_parent": np.asarray(child_to_parent, dtype=float).tolist(),
            "child_origin_in_parent": np.asarray(child_origin_in_parent, dtype=float).tolist(),
            "child_transform_determinant": int(round(abs(float(np.linalg.det(child_to_parent))))),
            "parent_action_scope": "parent_space_group_mod_ordered_translation_lattice",
            "parent_operation_count": len(parent_candidates) // 2,
            "parent_grey_operation_count": len(parent_candidates),
            "ordered_operation_count": len(ordered_magnetic_ops),
            "ordered_subset_of_parent": False,
            "unmatched_ordered_operation_count": len(unmatched),
            "unmatched_ordered_operations": unmatched[:8],
            "candidate_reversal_domains": [],
        }

    cosets = _parent_left_coset_partition(
        parent_candidates,
        ordered_parent_ops,
        child_to_parent,
        tol=tol,
    )
    candidate_domains = []
    collinear_domain_relation_candidates = []
    dedup_reversal_domains: dict[tuple[str, str, str], dict[str, Any]] = {}

    def record_dedup_domain(
        *,
        coset_index: int,
        representative_payload: dict[str, Any],
        branch_payload: dict[str, Any],
    ) -> None:
        if not branch_payload["valid_domain_relation"]:
            return
        pattern_key = branch_payload.get("transformed_pattern_key")
        if pattern_key is None:
            return
        key = (
            branch_payload["p_relation"],
            branch_payload["magnetic_order_relation"],
            pattern_key,
        )
        domain = dedup_reversal_domains.setdefault(
            key,
            {
                "domain_index": len(dedup_reversal_domains),
                "p_relation": branch_payload["p_relation"],
                "magnetic_order_relation": branch_payload["magnetic_order_relation"],
                "relation_label": branch_payload["relation_label"],
                "transformed_pattern_key": pattern_key,
                "coset_indices": [],
                "representative_count": 0,
                "soc_allowed_exists": False,
                "exchange_only_exists": False,
                "example_representative": representative_payload,
                "representative_classes": [],
            },
        )
        if coset_index not in domain["coset_indices"]:
            domain["coset_indices"].append(coset_index)
        domain["representative_count"] += 1
        domain["soc_allowed_exists"] = bool(
            domain["soc_allowed_exists"] or branch_payload["soc_allowed"]
        )
        domain["exchange_only_exists"] = bool(
            domain["exchange_only_exists"] or branch_payload["exchange_only"]
        )
        representative_class = branch_payload["representative_class"]
        if representative_class not in domain["representative_classes"]:
            domain["representative_classes"].append(representative_class)

    for coset_index, coset in enumerate(cosets):
        reversal_representatives = []
        general_relation_representatives = []
        for op in coset:
            child_rotation = op[3]
            child_translation = op[4]
            p_relation, axis_labels = _axis_relation_payload(child_rotation, axes, tol=tol)
            if p_relation == "P -> other":
                continue
            representative_payload = _magnetic_operation_payload(
                (
                    child_rotation,
                    child_translation,
                    op[2],
                ),
                tol=tol,
            )
            representative_payload["time_reversal"] = int(op[2])
            branch_payloads = _collinear_branch_relation_payloads(
                rotation=child_rotation,
                translation=child_translation,
                time_reversal=int(op[2]),
                p_relation=p_relation,
                p_axis_labels=axis_labels,
                context=collinear_context,
                tol=tol,
            )
            relation_entry = {
                "representative": representative_payload,
                "p_relation": p_relation,
                "axis_labels": axis_labels,
                "collinear_branch_relations": branch_payloads,
                "soc_allowed_exists": any(
                    branch["soc_allowed"] for branch in branch_payloads
                ),
                "exchange_only_exists": any(
                    branch["exchange_only"] for branch in branch_payloads
                ),
            }
            if p_relation == "P -> -P":
                reversal_representatives.append(relation_entry)
                for branch_payload in branch_payloads:
                    record_dedup_domain(
                        coset_index=coset_index,
                        representative_payload=representative_payload,
                        branch_payload=branch_payload,
                    )
            elif any(
                branch["relation_label"] == "p_preserved_magnetic_order_reversed"
                for branch in branch_payloads
            ):
                general_relation_representatives.append(relation_entry)

        if general_relation_representatives:
            first_general = general_relation_representatives[0]
            collinear_domain_relation_candidates.append(
                {
                    "coset_index": coset_index,
                    "relation_scope": "P -> P magnetic-order-reversal",
                    "representative": first_general["representative"],
                    "representative_count": len(general_relation_representatives),
                    "collinear_branch_relations": first_general[
                        "collinear_branch_relations"
                    ],
                    "soc_allowed_exists": any(
                        item["soc_allowed_exists"]
                        for item in general_relation_representatives
                    ),
                    "exchange_only_exists": any(
                        item["exchange_only_exists"]
                        for item in general_relation_representatives
                    ),
                }
            )

        if not reversal_representatives:
            continue
        selected_entry = reversal_representatives[0]
        selected_op_payload = selected_entry["representative"]
        selected_axis_labels = selected_entry["axis_labels"]
        generic_reversal = not selected_axis_labels
        selected_branch_payloads = selected_entry["collinear_branch_relations"]
        candidate_domains.append(
            {
                "coset_index": coset_index,
                "coset_size": len(coset),
                "representative": selected_op_payload,
                "maps_p_to_minus_p": True,
                "reversed_polar_axes": selected_axis_labels,
                "reverses_some_allowed_polar_vector": generic_reversal,
                "time_reversal": int(selected_op_payload["time_reversal"]),
                "representative_class": (
                    "time_reversal_branch"
                    if int(selected_op_payload["time_reversal"]) == -1
                    else "unitary_branch"
                ),
                "reversal_representative_count": len(reversal_representatives),
                "collinear_branch_relations": selected_branch_payloads,
                "soc_allowed_exists": any(
                    branch["soc_allowed"] for branch in selected_branch_payloads
                ),
                "exchange_only_exists": any(
                    branch["exchange_only"] for branch in selected_branch_payloads
                ),
                "soc_exchange_classification_status": (
                    "classified_collinear_branches"
                    if collinear_context is not None
                    else "not_evaluated_missing_collinear_pattern"
                ),
                "msg_compatibility_rule": (
                    "In the oriented spin frame, a spin-space operation "
                    "{S||R|t} is MSG-compatible iff S equals "
                    "theta * det(R_cart) * R_cart after expressing the "
                    "real-space rotation in the same spin Cartesian frame; "
                    "theta=-1 is the time-reversal branch."
                ),
                "domain_equivalence_status": (
                    "deduplicated_by_transformed_signed_collinear_pattern"
                    if collinear_context is not None
                    else "not_evaluated_missing_collinear_pattern"
                ),
                "classification_note": (
                    "This is an internally generated parent-grey/domain "
                    "representative in the ordered standard supercell. "
                    "For collinear cases, +1/-1 spin branches are classified "
                    "against the signed magnetic pattern and the MSG spin-real "
                    "locking map."
                ),
            }
        )

    status = (
        "candidate_reversal_domains_found"
        if candidate_domains
        else "no_parent_ordered_coset_maps_p_to_minus_p"
    )
    return {
        "status": status,
        "basis_setting": basis_setting,
        "parent_group_source": "spglib_standard_parent_lifted_to_ordered_standard_supercell",
        "ordered_subgroup_source": "ordered_spin_space_real_space_projection",
        "parent_space_group_number": parent_space_group_number,
        "parent_space_group_symbol": parent_space_group_symbol,
        "parent_hall_number": parent_hall_number,
        "ordered_space_group_number": ordered_space_group_number,
        "ordered_polar_axes": [
            {
                "label": format_polar_axis_vector(axis),
                "components": [float(component) for component in axis],
                "setting": basis_setting,
            }
            for axis in axes
        ],
        "child_basis_in_parent": np.asarray(child_to_parent, dtype=float).tolist(),
        "child_origin_in_parent": np.asarray(child_origin_in_parent, dtype=float).tolist(),
        "child_transform_determinant": int(round(abs(float(np.linalg.det(child_to_parent))))),
        "parent_action_scope": "parent_space_group_mod_ordered_translation_lattice",
        "parent_operation_count": len(parent_candidates) // 2,
        "parent_grey_operation_count": len(parent_candidates),
        "ordered_operation_count": len(ordered_magnetic_ops),
        "ordered_subset_of_parent": True,
        "translation_quotient_status": (
            "physical_domains_deduplicated_by_signed_collinear_pattern"
            if collinear_context is not None
            else "raw_cosets_only_missing_collinear_pattern"
        ),
        "translation_quotient_note": (
            "Raw parent-grey cosets are retained for diagnostics. Physical "
            "domain counts use transformed signed collinear patterns, which "
            "quotients common-supercell translations that do not change the "
            "magnetic domain."
        ),
        "left_coset_convention": (
            "domain cosets are left cosets gH in the parent grey space group "
            "modulo the ordered translation lattice; H is the ordered "
            "spin-space real-space projection on the unit time branch"
        ),
        "left_coset_count": len(cosets),
        "left_coset_sizes": [len(coset) for coset in cosets],
        "candidate_reversal_domain_count": len(candidate_domains),
        "candidate_reversal_domains": candidate_domains,
        "physical_reversal_domain_count": len(dedup_reversal_domains),
        "deduplicated_reversal_domains": list(dedup_reversal_domains.values()),
        "collinear_domain_relation_candidate_count": len(collinear_domain_relation_candidates),
        "collinear_domain_relation_candidates": collinear_domain_relation_candidates,
    }


def _ordered_symmetry_source(ossg_space_group_number: int | None) -> str:
    if ossg_space_group_number is None:
        return "ssg_g0_real_space_projection"
    return "ossg_real_space_projection"


def _classify_polarity(
    *,
    structural_parent_number: int | None,
    ordered_space_group_number: int | None,
) -> tuple[str, str, bool | None]:
    parent_is_polar = space_group_is_polar(structural_parent_number)
    ordered_is_polar = space_group_is_polar(ordered_space_group_number)

    if parent_is_polar is None or ordered_is_polar is None:
        return "unknown_symmetry", "insufficient_symmetry_data", None

    if not ordered_is_polar:
        return "ordered_symmetry_nonpolar", "ordered_symmetry_forbids_polarization", False

    if not parent_is_polar:
        return (
            "magnetically_induced_polar_candidate",
            "candidate_requires_parent_ordered_coset",
            None,
        )

    if structural_parent_number == ordered_space_group_number:
        parent_axes = _axis_labels(structural_parent_number)
        ordered_axes = _axis_labels(ordered_space_group_number)
        if parent_axes == ordered_axes:
            return (
                "parent_polar_axis_preserved",
                "parent_polar_ordered_polar_no_switching_claim",
                None,
            )

    return (
        "parent_polar_ordered_polar_transport_required",
        "parent_polar_axis_relation_requires_coordinate_transport",
        None,
    )


def _domain_relation_status(polarity_status: str) -> str:
    if polarity_status == "ordered_symmetry_nonpolar":
        return "not_applicable_ordered_symmetry_nonpolar"
    if polarity_status == "unknown_symmetry":
        return "not_evaluated_missing_symmetry_data"
    if polarity_status == "parent_polar_axis_preserved":
        return "not_evaluated_no_new_polar_axis_from_ordered_symmetry"
    return "not_evaluated_missing_parent_ordered_cosets"


def _configuration_collinear_only_status(configuration: str | None) -> str | None:
    if configuration is None or configuration == "Collinear":
        return None
    if configuration == "Coplanar":
        return "not_evaluated_coplanar_order_collinear_only"
    if configuration == "Noncoplanar":
        return "not_evaluated_noncollinear_order_collinear_only"
    return "not_evaluated_unknown_magnetic_configuration_collinear_only"


def _is_k_dependent_nonrelativistic_spin_splitting(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "k-dependent"


def _is_altermagnet_tag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip() == "(Altermagnet)"


def _ferroelectric_altermagnet_screening_payload(
    *,
    ordered_space_group_number: int | None,
    polarity_status: str,
    magnetic_phase: str | None,
    magnetic_phase_base: str | None,
    spin_splitting_without_soc: str | None,
    is_altermagnet: Any,
) -> dict[str, Any]:
    ordered_is_polar = space_group_is_polar(ordered_space_group_number)
    has_k_dependent_spin_splitting = _is_k_dependent_nonrelativistic_spin_splitting(
        spin_splitting_without_soc
    )
    altermagnet_flag = _is_altermagnet_tag(is_altermagnet)

    if ordered_is_polar is not True:
        status = "not_candidate_ordered_symmetry_nonpolar"
    elif has_k_dependent_spin_splitting is None:
        status = "unknown_missing_nonrelativistic_spin_splitting"
    elif not has_k_dependent_spin_splitting:
        status = "not_candidate_no_k_dependent_nonrelativistic_spin_splitting"
    elif altermagnet_flag is True:
        status = "candidate"
    elif altermagnet_flag is None:
        status = "candidate_k_dependent_spin_splitting_needs_altermagnet_review"
    else:
        status = "candidate_k_dependent_spin_splitting_not_flagged_altermagnet"

    return {
        "status": status,
        "literature_basis": {
            "source": "Gu et al., Phys. Rev. Lett. 134, 106802 (2025)",
            "doi": "10.1103/PhysRevLett.134.106802",
            "supplemental_material": (
                "Table SI lists 22 ferroelectric altermagnet candidates; "
                "Section II narrows switchable candidates by structural "
                "polarization reversibility and P/S domain comparison."
            ),
        },
        "screening_rules": [
            {
                "name": "ordered_spin_space_real_space_projection_is_polar",
                "satisfied": ordered_is_polar,
            },
            {
                "name": "nonrelativistic_spin_splitting_is_k_dependent",
                "satisfied": has_k_dependent_spin_splitting,
            },
            {
                "name": "magnetic_phase_is_flagged_as_altermagnet",
                "satisfied": altermagnet_flag,
            },
        ],
        "evidence": {
            "polarity_status": polarity_status,
            "magnetic_phase": magnetic_phase,
            "magnetic_phase_base": magnetic_phase_base,
            "spin_splitting_without_soc": spin_splitting_without_soc,
            "is_altermagnet": altermagnet_flag,
        },
    }


def _switchable_altermagnet_screening_payload(
    *,
    ferroelectric_altermagnet_status: str,
    domain_status: str,
) -> dict[str, Any]:
    if domain_status.endswith("_collinear_only"):
        status = domain_status
    elif ferroelectric_altermagnet_status == "candidate":
        status = "candidate_requires_p_s_coset_and_barrier_validation"
    elif ferroelectric_altermagnet_status.startswith("candidate_"):
        status = "candidate_requires_altermagnet_classification_review"
    else:
        status = "not_candidate_from_current_ferroelectric_altermagnet_screening"

    return {
        "status": status,
        "domain_relation_status": domain_status,
        "required_symmetry_tests": [
            "use_domain_reversal_symmetry_screening_for_P_to_minus_P_candidates",
            "test_spin_splitting_secondary_descriptor_S_to_minus_S_for_each_candidate",
            "classify_candidate_representative_as_msg_compatible_or_spin_space_only",
        ],
        "representative_classes": {
            "msg_compatible": (
                "SOC-allowed magnetic-domain representative with spin and "
                "real-space locking compatible with the MSG map"
            ),
            "spin_space_only": (
                "exchange-allowed nonrelativistic representative present in "
                "spin-space symmetry but not MSG-compatible"
            ),
        },
    }


def _domain_reversal_symmetry_screening_payload(
    *,
    domain_status: str,
    coset_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if domain_status == "not_applicable_ordered_symmetry_nonpolar":
        status = "not_applicable_ordered_symmetry_nonpolar"
    elif domain_status == "not_evaluated_missing_symmetry_data":
        status = "not_evaluated_missing_symmetry_data"
    elif domain_status.endswith("_collinear_only"):
        status = domain_status
    elif coset_analysis is not None:
        status = coset_analysis.get("status", "parent_ordered_coset_screened")
    else:
        status = "requires_parent_ordered_coset_validation"

    payload = {
        "status": status,
        "scope": "symmetry_only",
        "primary_order_parameter": {
            "name": "electric_polarization",
            "symbol": "P",
            "required_test": "real_space_part_maps_P_to_minus_P",
        },
        "secondary_order_parameter": {
            "name": "pending_explicit_descriptor",
            "examples": [
                "collinear_neel_vector",
                "spin_splitting_S",
                "weak_moment_M",
                "magnetic_order_parameter",
            ],
            "required_test": (
                "transport_descriptor_under_same_representative_and_compare_to_requested_target"
            ),
        },
        "candidate_operation_tests": [
            "construct_parent_ordered_coset_representatives",
            "test_real_space_part_maps_P_to_minus_P",
            "deduplicate_transformed_magnetic_structures_by_equivalence",
            "test_optional_secondary_descriptor_transform_for_each_surviving_candidate",
            "classify_representative_as_msg_compatible_or_spin_space_only",
        ],
        "representative_classes": {
            "msg_compatible": "SOC-allowed candidate relation",
            "spin_space_only": "exchange_allowed_nonrelativistic_candidate_relation",
        },
        "candidate_reversal_domains": [],
    }
    if coset_analysis is not None:
        payload.update(coset_analysis)
    return payload


def _post_fsg_path_validation_requirements_payload() -> dict[str, Any]:
    return {
        "status": "not_evaluated_by_findspingroup",
        "scope": "outside_fsg_symmetry_only_contract",
        "checks": [
            "deduplicate_transformed_magnetic_structures_by_equivalence",
            "whether_polarization_reversal_is_structurally_switchable",
            "which_minus_p_domain_is_selected_by_the_practical_electric_field_path",
            "whether_the_practical_minus_p_domain_has_the_requested_secondary_descriptor_state",
            "whether_the_candidate_path_has_an_accessible_energy_barrier",
        ],
        "note": (
            "These are not FSG failure modes. FSG may list symmetry candidates; "
            "structural path selection and barriers require a separate "
            "calculation."
        ),
    }


def _energy_barrier_workflow_payload() -> dict[str, Any]:
    return {
        "status": "not_computed_by_findspingroup",
        "symmetry_inference": (
            "MSG-compatible versus spin-space-only representative labels do not "
            "determine the switching barrier. The barrier depends on the "
            "structural path, unstable modes, and atomic displacements."
        ),
        "required_inputs": [
            "initial_domain_structure",
            "target_domain_structure_or_generating_operation",
            "polar_axis_basis_or_optional_polarization_vector",
            "candidate_structural_path_or_mode_decomposition",
            "dft_total_energy_or_neb_profile",
        ],
        "recommended_workflows": [
            "mode_decomposition_for_hybrid_improper_ferroelectrics",
            "linear_interpolation_screening_between_domain_structures",
            "nudged_elastic_band_or_constrained_relaxation_for_final_barrier",
        ],
        "barrier_units": "eV_per_unit_cell_or_eV_per_formula_unit",
    }


def build_ferroelectric_switching_payload(
    *,
    input_space_group_number: int | None,
    input_space_group_symbol: str | None = None,
    ssg_space_group_number: int | None = None,
    ossg_space_group_number: int | None = None,
    msg_num: int | None = None,
    msg_symbol: str | None = None,
    msg_parent_space_group_number: int | None = None,
    source_parent_space_group: dict | None = None,
    magnetic_phase: str | None = None,
    magnetic_phase_base: str | None = None,
    magnetic_configuration: str | None = None,
    spin_splitting_without_soc: str | None = None,
    is_altermagnet: Any = None,
    domain_reversal_coset_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative symmetry-only ferroelectric switching payload.

    The payload separates the structural parent, ordered spin-space symmetry,
    and SOC magnetic subgroup. A positive switching claim still requires a
    validated parent/ordered coset representative that maps the chosen
    polarization vector to its negative.
    """

    exact_parent_number = None
    exact_parent_symbol = None
    if domain_reversal_coset_analysis is not None:
        exact_parent_number = _as_int_or_none(
            domain_reversal_coset_analysis.get("parent_space_group_number")
        )
        exact_parent_symbol = domain_reversal_coset_analysis.get("parent_space_group_symbol")
    structural_parent_number = (
        exact_parent_number
        if exact_parent_number is not None
        else _as_int_or_none(input_space_group_number)
    )
    structural_parent_symbol = (
        exact_parent_symbol
        if exact_parent_symbol is not None
        else input_space_group_symbol
    )
    ordered_space_group_number = (
        _as_int_or_none(ossg_space_group_number)
        if ossg_space_group_number is not None
        else _as_int_or_none(ssg_space_group_number)
    )
    msg_parent_number = _as_int_or_none(msg_parent_space_group_number)
    ordered_source = _ordered_symmetry_source(ossg_space_group_number)

    polarity_status, switching_status, switching_detected = _classify_polarity(
        structural_parent_number=structural_parent_number,
        ordered_space_group_number=ordered_space_group_number,
    )
    domain_status = _domain_relation_status(polarity_status)
    collinear_only_status = _configuration_collinear_only_status(magnetic_configuration)
    if (
        collinear_only_status is not None
        and polarity_status != "ordered_symmetry_nonpolar"
    ):
        switching_status = collinear_only_status
        switching_detected = None
        domain_status = collinear_only_status
        domain_reversal_coset_analysis = None

    required_inputs = []
    if switching_detected is None and collinear_only_status is None:
        required_inputs = [
            "deduplicated_transformed_magnetic_domain_structures",
            "secondary_order_parameter_definition_and_transport",
            "oriented_msg_compatibility_classification_for_coset_representatives",
            "structural_path_or_barrier_validation_for_switching_claim",
        ]
        if polarity_status == "parent_polar_ordered_polar_transport_required":
            required_inputs.append("parent_to_ordered_polar_axis_coordinate_transport")

    source_parent_number = None
    source_parent_name = None
    source_parent_has_inversion = None
    source_parent_is_polar = None
    if source_parent_space_group is not None:
        source_parent_number = _as_int_or_none(source_parent_space_group.get("IT_number"))
        source_parent_name = source_parent_space_group.get("name_H_M_alt")
        source_parent_has_inversion = space_group_has_real_space_inversion(source_parent_number)
        source_parent_is_polar = space_group_is_polar(source_parent_number)

    structural_parent = _space_group_payload(
        source="current_ordered_exact_parent",
        space_group_number=structural_parent_number,
        space_group_symbol=structural_parent_symbol,
    )
    input_structure_parent = _space_group_payload(
        source="input_structure_exact_parent",
        space_group_number=_as_int_or_none(input_space_group_number),
        space_group_symbol=input_space_group_symbol,
    )
    ordered_spin_space = _space_group_payload(
        source=ordered_source,
        space_group_number=ordered_space_group_number,
    )
    soc_magnetic = _space_group_payload(
        source="msg_parent_space_group",
        space_group_number=msg_parent_number,
        msg_num=msg_num,
        msg_symbol=msg_symbol,
    )
    ferroelectric_altermagnet_screening = _ferroelectric_altermagnet_screening_payload(
        ordered_space_group_number=ordered_space_group_number,
        polarity_status=polarity_status,
        magnetic_phase=magnetic_phase,
        magnetic_phase_base=magnetic_phase_base,
        spin_splitting_without_soc=spin_splitting_without_soc,
        is_altermagnet=is_altermagnet,
    )
    switchable_altermagnet_screening = _switchable_altermagnet_screening_payload(
        ferroelectric_altermagnet_status=ferroelectric_altermagnet_screening["status"],
        domain_status=domain_status,
    )
    domain_reversal_symmetry_screening = _domain_reversal_symmetry_screening_payload(
        domain_status=domain_status,
        coset_analysis=domain_reversal_coset_analysis,
    )
    coset_domains = domain_reversal_symmetry_screening.get("candidate_reversal_domains", [])
    analysis_level = (
        "symmetry_only_parent_ordered_coset_screened"
        if domain_reversal_coset_analysis is not None
        else "symmetry_only_collinear_switching_not_evaluated"
        if collinear_only_status is not None
        else "symmetry_only_parent_ordered_coset_pending"
    )

    return {
        "status": switching_status,
        "switching_detected": switching_detected,
        "analysis_level": analysis_level,
        "polarity_status": polarity_status,
        "structural_parent_symmetry": structural_parent,
        "input_structure_parent_symmetry": input_structure_parent,
        "parent_selection": {
            "default": "current_ordered_exact_parent",
            "source": (
                "domain_reversal_coset_analysis"
                if exact_parent_number is not None
                else "input_structure_exact_parent"
            ),
            "high_temperature_parent_status": "not_inferred_from_fsg_inputs",
            "override_status": "not_enabled_in_current_draft",
        },
        "ordered_spin_space_symmetry": ordered_spin_space,
        "soc_magnetic_symmetry": soc_magnetic,
        "governing_symmetry": ordered_spin_space,
        "comparison_symmetry": {
            "input_space_group_number": _as_int_or_none(input_space_group_number),
            "input_space_group_symbol": input_space_group_symbol,
            "input_is_polar": space_group_is_polar(_as_int_or_none(input_space_group_number)),
            "current_ordered_exact_parent_space_group_number": structural_parent_number,
            "current_ordered_exact_parent_space_group_symbol": structural_parent_symbol,
            "current_ordered_exact_parent_is_polar": space_group_is_polar(
                structural_parent_number
            ),
            "ssg_space_group_number": ssg_space_group_number,
            "ssg_is_polar": space_group_is_polar(ssg_space_group_number),
            "ossg_space_group_number": ossg_space_group_number,
            "ossg_is_polar": space_group_is_polar(ossg_space_group_number),
            "msg_parent_space_group_number": msg_parent_number,
            "msg_is_polar": space_group_is_polar(msg_parent_number),
        },
        "allowed_polar_axes": ordered_spin_space["allowed_polar_axes"],
        "allowed_polar_axes_setting": ordered_spin_space["allowed_polar_axes_setting"],
        "parent_allowed_polar_axes": structural_parent["allowed_polar_axes"],
        "ordered_allowed_polar_axes": ordered_spin_space["allowed_polar_axes"],
        "soc_allowed_polar_axes": soc_magnetic["allowed_polar_axes"],
        "domain_reversal_symmetry_screening": domain_reversal_symmetry_screening,
        "polarization_test_contract": {
            "mode": "polar_axis_basis_only",
            "claim": (
                "The current screen tests whether a representative reverses an "
                "allowed polar-axis direction; it does not use a computed Berry "
                "phase polarization vector."
            ),
        },
        "secondary_order_parameter_contract": {
            "status": "pending_definition_and_transport",
            "default_for_collinear_discussion": "neel_vector",
            "notes": [
                "Gu-style S->-S is not claimed yet.",
                "For collinear AFM cases the natural first descriptor is the Neel vector.",
                "Other cases may need spin_splitting_S, weak_moment_M, or an explicit order parameter.",
            ],
        },
        "polarization_coupling_contract": {
            "scope": "collinear_only",
            "input_magnetic_configuration": magnetic_configuration,
            "magnetically_induced_polarization": {
                "definition": (
                    "The exact nonmagnetic parent of the ordered structure is "
                    "nonpolar, while the ordered spin-space real-space "
                    "projection is polar."
                ),
                "current_status_field": "polarity_status",
                "current_status_value": "magnetically_induced_polar_candidate",
            },
            "magnetically_controlled_polarization": {
                "definition": (
                    "A symmetry-allowed relation between polarization and the "
                    "collinear magnetic order descriptor under a domain "
                    "operation. Symmetry can classify possible relations, but "
                    "the practical switching path remains case dependent."
                ),
                "traditional_strong_condition": (
                    "both polarization and the magnetic order descriptor reverse"
                ),
                "fsg_reporting_contract": (
                    "report all three collinear symmetry possibilities without "
                    "promoting them to validated switching claims"
                ),
            },
            "collinear_spin_space_branches": [
                {
                    "spin_space_operation": "+1",
                    "meaning": "preserve the signed collinear order descriptor",
                },
                {
                    "spin_space_operation": "-1",
                    "meaning": "reverse the signed collinear order descriptor",
                },
            ],
            "collinear_relation_classes": [
                {
                    "label": "p_and_magnetic_order_reversed",
                    "polarization_relation": "P -> -P",
                    "magnetic_order_relation": "L -> -L",
                    "traditional_magnetically_controlled": True,
                },
                {
                    "label": "p_reversed_magnetic_order_preserved",
                    "polarization_relation": "P -> -P",
                    "magnetic_order_relation": "L -> L",
                    "traditional_magnetically_controlled": False,
                },
                {
                    "label": "p_preserved_magnetic_order_reversed",
                    "polarization_relation": "P -> P",
                    "magnetic_order_relation": "L -> -L",
                    "traditional_magnetically_controlled": False,
                },
            ],
            "current_evaluation_status": (
                collinear_only_status
                if collinear_only_status is not None
                else "definitions_recorded_collinear_classification_pending"
            ),
        },
        "msg_compatibility_rule": {
            "status": "rule_recorded_classification_pending",
            "oriented_spin_frame_condition": (
                "For an oriented spin-space operation {S||R|t}, MSG compatibility "
                "requires S = theta * det(R_cart) * R_cart in the same spin "
                "Cartesian frame, where theta=+1/-1 is the unitary/time-reversal branch."
            ),
            "soc_label": "msg_compatible",
            "exchange_only_label": "valid_spin_space_operation_not_msg_compatible",
        },
        "domain_deduplication_contract": {
            "mode": "transformed_magnetic_structure_equivalence",
            "status": "required_before_validated_switching_claim",
            "representative_level_output": (
                "keep each symmetry representative and its SOC/exchange class"
            ),
            "domain_level_output": (
                "after magnetic-structure equivalence deduplication, aggregate "
                "soc_allowed_exists and exchange_only_exists for the domain"
            ),
        },
        "claim_level_contract": {
            "current_positive_level": "p_reversal_symmetry_candidate",
            "next_level": "switchable_candidate_pending_secondary_descriptor_and_path",
            "final_level": "validated_switchable",
        },
        "translation_quotient_contract": {
            "status": "implemented_by_signed_pattern_dedup_for_collinear_output",
            "principle": (
                "Translations introduced only to embed parent operations in a "
                "common supercell are gauge/redundant; translations outside the "
                "actual magnetic-cell translation breaking should not create "
                "additional physical domains."
            ),
            "implementation_target": (
                "retain raw parent-grey cosets for diagnostics, then deduplicate "
                "physical domains by transformed signed collinear pattern"
            ),
        },
        "domain_relation_output_contract": {
            "candidate_reversal_domains_scope": "P -> -P candidates only",
            "general_collinear_domain_relation_scope": (
                "record P/L relation classes including P -> P, L -> -L outside "
                "candidate_reversal_domains"
            ),
            "internal_descriptor": "signed_collinear_magnetic_pattern",
            "display_descriptor_note": (
                "Neel vector L, magnetization M, or signed pattern labels are "
                "presentation choices; classification should use the signed "
                "collinear pattern."
            ),
        },
        "ferroelectric_altermagnet_screening": ferroelectric_altermagnet_screening,
        "switchable_altermagnet_screening": switchable_altermagnet_screening,
        "post_fsg_path_validation_requirements": _post_fsg_path_validation_requirements_payload(),
        "energy_barrier_workflow": _energy_barrier_workflow_payload(),
        "candidate_reversal_domains": coset_domains,
        "candidate_reversal_domain_status": domain_reversal_symmetry_screening["status"],
        "domain_switching_relation": {
            "status": domain_status,
            "scope": "collinear_only",
            "parent_group": "current_ordered_exact_parent",
            "ordered_subgroup": ordered_source,
            "switching_test": "real_space_coset_representative_maps_P_to_minus_P",
            "soc_allowed_if": (
                "the same representative is compatible with the MSG spin-real "
                "locking map"
            ),
            "exchange_allowed_if": (
                "the representative is valid in spin-space symmetry but is not "
                "MSG-compatible"
            ),
            "barrier_note": (
                "Representative class is a symmetry label, not a barrier "
                "estimate; energy barriers require a structural path calculation."
            ),
        },
        "source_parent_space_group": {
            "space_group_number": source_parent_number,
            "space_group_symbol": source_parent_name,
            "is_polar": source_parent_is_polar,
            "has_real_space_inversion": source_parent_has_inversion,
        },
        "required_inputs_for_switching_claim": required_inputs,
        "special_coset": {
            "status": (
                "replaced_by_sg_to_ossg_coset_screening"
                if domain_reversal_coset_analysis is not None
                else "not_promoted_to_switching_claim"
            ),
            "reason": (
                "The switching screen uses the structural parent SG modulo the "
                "ordered OSSG real-space projection. A representative is only "
                "a reversal candidate after the real-space P -> -P test; pure "
                "translation does not flip polarization."
            ),
        },
    }
