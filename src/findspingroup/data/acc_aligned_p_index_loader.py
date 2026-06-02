from __future__ import annotations

import json
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files
from typing import TypeAlias


Matrix: TypeAlias = tuple[tuple[Fraction, Fraction, Fraction], ...]


@lru_cache(maxsize=1)
def load_acc_aligned_kpoint_runtime_index() -> dict:
    data_file = files("findspingroup.data").joinpath("acc_aligned_kpoint_runtime_index.json")
    return json.loads(data_file.read_text(encoding="utf-8"))


def load_acc_aligned_p_index() -> dict:
    """Compatibility alias for the ACC-aligned runtime index."""
    return load_acc_aligned_kpoint_runtime_index()


@lru_cache(maxsize=1)
def matrix_by_id() -> dict[str, Matrix]:
    data = load_acc_aligned_kpoint_runtime_index()
    return {
        p_id: tuple(tuple(Fraction(cell) for cell in row) for row in matrix)
        for p_id, matrix in data["matrix_by_id"].items()
    }


@lru_cache(maxsize=1)
def acc_kpoint_symbols_by_number() -> dict[int, dict]:
    data = load_acc_aligned_kpoint_runtime_index()
    return {
        int(payload["acc_number"]): payload
        for payload in data["acc_kpoint_symbols_by_acc"].values()
    }


def get_p_id_for_ssg_label(label: str) -> str:
    data = load_acc_aligned_kpoint_runtime_index()
    return data["ssg_label_to_p_id"][label]


def get_acc_aligned_conventional_to_primitive_p(label: str) -> Matrix:
    return matrix_by_id()[get_p_id_for_ssg_label(label)]


def get_pair_id_for_ssg_label(label: str) -> str:
    data = load_acc_aligned_kpoint_runtime_index()
    return data["label_to_pair_id"][label]


def get_acc_kpoint_symbols_by_acc_number(acc_number: int) -> dict:
    return acc_kpoint_symbols_by_number()[int(acc_number)]


def get_ssg_conventional_kpoint_symbols_for_label(label: str) -> tuple[str, ...]:
    data = load_acc_aligned_kpoint_runtime_index()
    pair_id = data["label_to_pair_id"][label]
    return tuple(data["ssg_conventional_kpoint_symbols_by_pair_id"][pair_id])
