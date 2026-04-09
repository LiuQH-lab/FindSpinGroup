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
def space_group_has_real_space_inversion(space_group_number: int | None) -> bool | None:
    if space_group_number is None:
        return None

    hall_number = _hall_number_for_space_group(int(space_group_number))
    dataset = spglib.get_symmetry_from_database(hall_number)
    rotations = np.asarray(dataset["rotations"], dtype=int)
    minus_identity = -np.eye(3, dtype=int)
    return bool(any(np.array_equal(rotation, minus_identity) for rotation in rotations))


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
    nullity = vh[rank:].T.shape[1]
    return bool(nullity > 0)


def msg_parent_space_group_info(msg_num: int | None) -> dict[str, int | str | bool | None]:
    if msg_num is None:
        return {
            "bns_number": None,
            "og_number": None,
            "bns_parent_space_group_number": None,
            "og_parent_space_group_number": None,
            "has_real_space_inversion": None,
            "is_polar": None,
            "is_chiral": None,
        }

    bns_number, _bns_symbol = MSGMPG_DB.MSG_INT_TO_BNS[msg_num]
    og_number = MSGMPG_DB.BNS_TO_OG_NUM[bns_number]
    bns_parent = int(str(bns_number).split(".")[0])
    og_parent = int(str(og_number).split(".")[0])
    bns_flag = space_group_has_real_space_inversion(bns_parent)
    og_flag = space_group_has_real_space_inversion(og_parent)
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
        "has_real_space_inversion": bns_flag,
        "is_polar": bns_polar,
        "is_chiral": bns_chiral,
    }
