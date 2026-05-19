from __future__ import annotations

from functools import lru_cache

import numpy as np
import spglib

from findspingroup.data import MSGMPG_DB
from findspingroup.data.PG_SYMBOL import SG_HALL_MAPPING


@lru_cache(maxsize=None)
def _hall_number_for_space_group(space_group_number: int) -> int:
    number = int(space_group_number)
    mapped = SG_HALL_MAPPING.get(number)
    if mapped is not None:
        return int(mapped)

    for hall_number in range(1, 531):
        info = spglib.get_spacegroup_type(hall_number)
        if info and info.number == number:
            return int(hall_number)

    raise ValueError(f"Unable to resolve a Hall number for space group {space_group_number}.")


@lru_cache(maxsize=None)
def space_group_is_centrosymmetric(space_group_number: int | None) -> bool | None:
    if space_group_number is None:
        return None

    hall_number = _hall_number_for_space_group(int(space_group_number))
    dataset = spglib.get_symmetry_from_database(hall_number)
    rotations = np.asarray(dataset["rotations"], dtype=int)
    minus_identity = -np.eye(3, dtype=int)
    return bool(any(np.array_equal(rotation, minus_identity) for rotation in rotations))


def space_group_has_real_space_inversion(space_group_number: int | None) -> bool | None:
    return space_group_is_centrosymmetric(space_group_number)


@lru_cache(maxsize=None)
def space_group_is_chiral(space_group_number: int | None) -> bool | None:
    if space_group_number is None:
        return None

    hall_number = _hall_number_for_space_group(int(space_group_number))
    dataset = spglib.get_symmetry_from_database(hall_number)
    rotations = np.asarray(dataset["rotations"], dtype=int)
    return bool(all(int(round(np.linalg.det(rotation))) == 1 for rotation in rotations))


@lru_cache(maxsize=None)
def space_group_is_polar(space_group_number: int | None) -> bool | None:
    if space_group_number is None:
        return None

    return bool(space_group_polar_axis_basis(space_group_number))


@lru_cache(maxsize=None)
def space_group_polar_axis_basis(space_group_number: int | None) -> tuple[tuple[float, float, float], ...] | None:
    if space_group_number is None:
        return None

    hall_number = _hall_number_for_space_group(int(space_group_number))
    dataset = spglib.get_symmetry_from_database(hall_number)
    rotations = np.asarray(dataset["rotations"], dtype=float)
    unique_rotations: list[np.ndarray] = []
    for rotation in rotations:
        if not any(np.array_equal(rotation, existing) for existing in unique_rotations):
            unique_rotations.append(rotation)

    constraint_matrix = np.concatenate(
        [rotation - np.eye(3, dtype=float) for rotation in unique_rotations],
        axis=0,
    )
    _, singular_values, vh = np.linalg.svd(constraint_matrix)
    rank = int(np.sum(singular_values > 1e-8))
    basis = vh[rank:]
    if basis.size == 0:
        return ()
    normalized = []
    for vector in basis:
        vector = np.asarray(vector, dtype=float)
        max_abs = float(np.max(np.abs(vector)))
        if max_abs < 1e-12:
            continue
        vector = vector / max_abs
        vector[np.abs(vector) < 1e-10] = 0.0
        nonzero_indices = np.where(np.abs(vector) >= 1e-10)[0]
        if nonzero_indices.size and vector[int(nonzero_indices[0])] < 0:
            vector = -vector
        normalized.append(tuple(float(round(component, 12)) for component in vector))
    return tuple(normalized)


def format_polar_axis_vector(vector: tuple[float, float, float] | list[float]) -> str:
    labels = ("a", "b", "c")
    parts = []
    for coefficient, label in zip(vector, labels):
        if abs(coefficient) < 1e-10:
            continue
        sign = "-" if coefficient < 0 else ""
        magnitude = abs(coefficient)
        if abs(magnitude - 1.0) < 1e-10:
            parts.append(f"{sign}{label}")
        else:
            parts.append(f"{sign}{magnitude:g}{label}")
    return " + ".join(parts).replace("+ -", "- ") if parts else "0"


def space_group_polar_axis_labels(space_group_number: int | None) -> list[str] | None:
    basis = space_group_polar_axis_basis(space_group_number)
    if basis is None:
        return None
    return [format_polar_axis_vector(vector) for vector in basis]



def msg_parent_space_group_info(msg_num: int | None) -> dict[str, int | str | bool | None]:
    if msg_num is None:
        return {
            "bns_number": None,
            "og_number": None,
            "bns_parent_space_group_number": None,
            "og_parent_space_group_number": None,
            "is_centrosymmetric": None,
            "is_polar": None,
            "is_chiral": None,
        }

    bns_number, _bns_symbol = MSGMPG_DB.MSG_INT_TO_BNS[msg_num]
    og_number = MSGMPG_DB.BNS_TO_OG_NUM[bns_number]
    bns_parent = int(str(bns_number).split(".")[0])
    og_parent = int(str(og_number).split(".")[0])
    bns_flag = space_group_is_centrosymmetric(bns_parent)
    og_flag = space_group_is_centrosymmetric(og_parent)
    bns_polar = space_group_is_polar(bns_parent)
    og_polar = space_group_is_polar(og_parent)
    bns_chiral = space_group_is_chiral(bns_parent)
    og_chiral = space_group_is_chiral(og_parent)

    if bns_flag != og_flag:
        raise ValueError(
            "MSG minus-one rule disagrees between BNS and OG parent SG numbers "
            f"for msg_num={msg_num}, BNS={bns_number}, OG={og_number}."
        )
    if bns_polar != og_polar:
        raise ValueError(
            "MSG polar rule disagrees between BNS and OG parent SG numbers "
            f"for msg_num={msg_num}, BNS={bns_number}, OG={og_number}."
        )
    if bns_chiral != og_chiral:
        raise ValueError(
            "MSG chiral rule disagrees between BNS and OG parent SG numbers "
            f"for msg_num={msg_num}, BNS={bns_number}, OG={og_number}."
        )

    return {
        "bns_number": bns_number,
        "og_number": og_number,
        "bns_parent_space_group_number": bns_parent,
        "og_parent_space_group_number": og_parent,
        "is_centrosymmetric": bns_flag,
        "is_polar": bns_polar,
        "is_chiral": bns_chiral,
    }
