from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import spglib

from . import find_spin_group
from .core.identify_symmetry_from_ops import get_space_group_from_operations
from .io.scif_generator import affine_matrix_to_xyz_expression
from .structure import SpinSpaceGroup
from .structure.cell import CrystalCell
from .utils.matrix_utils import getNormInf, in_space_group


def _canonical_translation(t: np.ndarray, *, tol: float) -> np.ndarray:
    translation = np.mod(np.asarray(t, dtype=float), 1.0)
    translation[np.isclose(translation, 1.0, atol=tol)] = 0.0
    translation[np.isclose(translation, 0.0, atol=tol)] = 0.0
    return translation


def _real_op_same(op_a, op_b, *, tol: float) -> bool:
    rot_a, trans_a = op_a
    rot_b, trans_b = op_b
    return np.allclose(rot_a, rot_b, atol=tol) and getNormInf(
        _canonical_translation(trans_a, tol=tol),
        _canonical_translation(trans_b, tol=tol),
    ) < tol


def _dedupe_real_ops(ops: list[tuple[np.ndarray, np.ndarray]], *, tol: float) -> list[tuple[np.ndarray, np.ndarray]]:
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for rot, trans in ops:
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
    new_rot = np.asarray(rot_left, dtype=float) @ np.asarray(rot_right, dtype=float)
    new_trans = _canonical_translation(
        np.asarray(rot_left, dtype=float) @ np.asarray(trans_right, dtype=float)
        + np.asarray(trans_left, dtype=float),
        tol=tol,
    )
    return new_rot, new_trans


def _contains_real_op(
    ops: list[tuple[np.ndarray, np.ndarray]],
    candidate: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> bool:
    return any(_real_op_same(candidate, existing, tol=tol) for existing in ops)


def _coset_partition(
    group_ops: list[tuple[np.ndarray, np.ndarray]],
    subgroup_ops: list[tuple[np.ndarray, np.ndarray]],
    *,
    left: bool,
    tol: float,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    unused = list(group_ops)
    cosets: list[list[tuple[np.ndarray, np.ndarray]]] = []
    while unused:
        representative = unused[0]
        if left:
            coset = [
                _multiply_real_ops(representative, subgroup_op, tol=tol)
                for subgroup_op in subgroup_ops
            ]
        else:
            coset = [
                _multiply_real_ops(subgroup_op, representative, tol=tol)
                for subgroup_op in subgroup_ops
            ]
        coset = _dedupe_real_ops(coset, tol=tol)
        cosets.append(coset)
        unused = [op for op in unused if not _contains_real_op(coset, op, tol=tol)]
    return cosets


def _op_to_xyzt(op: tuple[np.ndarray, np.ndarray]) -> str:
    return affine_matrix_to_xyz_expression(op[0], op[1])


def _op_key(op: tuple[np.ndarray, np.ndarray], *, tol: float) -> tuple[Any, ...]:
    rot = np.rint(np.asarray(op[0], dtype=float)).astype(int)
    trans = _canonical_translation(op[1], tol=tol)
    return tuple(rot.flatten().tolist() + [round(float(x), 6) for x in trans.tolist()])


def _coset_keys(
    cosets: list[list[tuple[np.ndarray, np.ndarray]]],
    *,
    tol: float,
) -> set[tuple[tuple[Any, ...], ...]]:
    return {
        tuple(sorted(_op_key(op, tol=tol) for op in coset))
        for coset in cosets
    }


def _g0_hall_number(expected_number: int, expected_symbol: str) -> int | None:
    matches: list[int] = []
    for hall_number in range(1, 531):
        info = spglib.get_spacegroup_type(hall_number)
        if info is None or int(info.number) != int(expected_number):
            continue
        if getattr(info, "international_short", None) == expected_symbol:
            return int(hall_number)
        matches.append(int(hall_number))
    return matches[0] if matches else None


def analyze_g0std_space_group_cosets(
    source_path: str | Path,
    *,
    symprec: float = 0.02,
    tol: float = 1e-6,
) -> dict[str, Any]:
    path = Path(source_path)
    result = find_spin_group(str(path), space_tol=symprec)

    g0_ssg = SpinSpaceGroup(result.g0_standard_ssg_ops)
    g0_cell = CrystalCell(
        result.g0_standard_cell["lattice"],
        result.g0_standard_cell["positions"],
        result.g0_standard_cell["occupancies"],
        result.g0_standard_cell["elements"],
        result.g0_standard_cell["moments"],
        spin_setting="in_lattice",
    )

    dataset = spglib.get_symmetry_dataset(g0_cell.to_spglib(mag=False), symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not determine a non-magnetic symmetry dataset for the G0std cell.")

    sg_ops = _dedupe_real_ops(
        [(np.asarray(rot, dtype=float), np.asarray(trans, dtype=float)) for rot, trans in zip(dataset.rotations, dataset.translations)],
        tol=tol,
    )
    ssg_real_ops = _dedupe_real_ops(
        [(np.asarray(rot, dtype=float), np.asarray(trans, dtype=float)) for rot, trans in g0_ssg.G0_ops],
        tol=tol,
    )
    ops_identified_dataset = get_space_group_from_operations(
        [[rot, trans] for rot, trans in sg_ops],
        symprec=symprec,
        bz=False,
    )

    subset = all(in_space_group(op, sg_ops, tol=tol) for op in ssg_real_ops)

    right_cosets = _coset_partition(sg_ops, ssg_real_ops, left=False, tol=tol) if subset else []
    left_cosets = _coset_partition(sg_ops, ssg_real_ops, left=True, tol=tol) if subset else []

    expected_hall_number = _g0_hall_number(g0_ssg.G0_num, g0_ssg.G0_symbol)
    expected_database_ops_count = None
    expected_hall_choice = None
    if expected_hall_number is not None:
        db_sym = spglib.get_symmetry_from_database(expected_hall_number)
        expected_database_ops_count = len(db_sym["rotations"])
        expected_hall_choice = spglib.get_spacegroup_type(expected_hall_number).choice

    actual_hall_type = spglib.get_spacegroup_type(dataset.hall_number)

    return {
        "case_id": path.as_posix(),
        "index": result.index,
        "conf": result.conf,
        "expected_g0_number": int(g0_ssg.G0_num),
        "expected_g0_symbol": g0_ssg.G0_symbol,
        "expected_g0_hall_number": expected_hall_number,
        "expected_g0_hall_choice": expected_hall_choice,
        "expected_g0_database_op_count": expected_database_ops_count,
        "g0std_spglib_number": int(dataset.number),
        "g0std_spglib_symbol": dataset.international,
        "g0std_spglib_hall_number": int(dataset.hall_number),
        "g0std_spglib_hall_choice": None if actual_hall_type is None else actual_hall_type.choice,
        "matches_expected_g0_number": int(dataset.number) == int(g0_ssg.G0_num),
        "ops_identified_number": int(ops_identified_dataset.number),
        "ops_identified_symbol": ops_identified_dataset.international,
        "ops_identified_hall_number": int(ops_identified_dataset.hall_number),
        "ops_identified_choice": ops_identified_dataset.choice,
        "sg_op_count": len(sg_ops),
        "ssg_real_op_count": len(ssg_real_ops),
        "ssg_real_subset_of_sg": subset,
        "right_coset_count": len(right_cosets) if subset else None,
        "right_coset_sizes": [len(coset) for coset in right_cosets] if subset else [],
        "left_coset_count": len(left_cosets) if subset else None,
        "left_coset_sizes": [len(coset) for coset in left_cosets] if subset else [],
        "left_equals_right": _coset_keys(left_cosets, tol=tol) == _coset_keys(right_cosets, tol=tol) if subset else None,
        "sg_operations_xyzt": [_op_to_xyzt(op) for op in sg_ops],
        "ssg_real_operations_xyzt": [_op_to_xyzt(op) for op in ssg_real_ops],
        "right_cosets_xyzt": [[_op_to_xyzt(op) for op in coset] for coset in right_cosets] if subset else [],
        "left_cosets_xyzt": [[_op_to_xyzt(op) for op in coset] for coset in left_cosets] if subset else [],
    }


def analyze_g0std_space_group_cosets_json(
    source_path: str | Path,
    *,
    symprec: float = 0.02,
    tol: float = 1e-6,
) -> str:
    payload = analyze_g0std_space_group_cosets(source_path, symprec=symprec, tol=tol)
    return json.dumps(payload, ensure_ascii=False, indent=2)
