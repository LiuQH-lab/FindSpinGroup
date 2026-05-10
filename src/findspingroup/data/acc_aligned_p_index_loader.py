from __future__ import annotations

import json
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files
from typing import TypeAlias


Matrix: TypeAlias = tuple[tuple[Fraction, Fraction, Fraction], ...]


@lru_cache(maxsize=1)
def load_acc_aligned_p_index() -> dict:
    data_file = files("findspingroup.data").joinpath("acc_aligned_p_index.json")
    return json.loads(data_file.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def matrix_by_id() -> dict[str, Matrix]:
    data = load_acc_aligned_p_index()
    return {
        p_id: tuple(tuple(Fraction(cell) for cell in row) for row in matrix)
        for p_id, matrix in data["matrix_by_id"].items()
    }


def get_p_id_for_ssg_label(label: str) -> str:
    data = load_acc_aligned_p_index()
    return data["ssg_label_to_p_id"][label]


def get_acc_aligned_conventional_to_primitive_p(label: str) -> Matrix:
    return matrix_by_id()[get_p_id_for_ssg_label(label)]
