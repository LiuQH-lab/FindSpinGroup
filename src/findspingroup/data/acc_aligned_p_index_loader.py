from __future__ import annotations

import json
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files
from typing import TypeAlias

from findspingroup.spin_splitting import (
    _append_basis_remainder_ascii,
    _append_basis_remainder_latex,
    combine_spin_texture_basis,
    spin_texture_basis_latex,
)


Matrix: TypeAlias = tuple[tuple[Fraction, Fraction, Fraction], ...]
RUNTIME_INDEX_FILE = "ssg_label_acc_kpoint_wave_index_20260604.json"


@lru_cache(maxsize=1)
def load_acc_aligned_kpoint_runtime_index() -> dict:
    data_file = files("findspingroup.data").joinpath(RUNTIME_INDEX_FILE)
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


@lru_cache(maxsize=1)
def ssg_label_records() -> dict[str, dict]:
    data = load_acc_aligned_kpoint_runtime_index()
    columns = tuple(data["ssg_label_record_columns"])
    return {
        label: dict(zip(columns, values, strict=True))
        for label, values in data["ssg_label_records"].items()
    }


def get_p_id_for_ssg_label(label: str) -> str:
    return ssg_label_records()[label]["p_id"]


def get_acc_aligned_conventional_to_primitive_p(label: str) -> Matrix:
    return matrix_by_id()[get_p_id_for_ssg_label(label)]


def get_pair_id_for_ssg_label(label: str) -> str:
    return ssg_label_records()[label]["pair_id"]


def get_acc_kpoint_symbols_by_acc_number(acc_number: int) -> dict:
    return acc_kpoint_symbols_by_number()[int(acc_number)]


def get_ssg_conventional_kpoint_symbols_for_label(label: str) -> tuple[str, ...]:
    data = load_acc_aligned_kpoint_runtime_index()
    pair_id = get_pair_id_for_ssg_label(label)
    return tuple(data["ssg_conventional_kpoint_symbols_by_pair_id"][pair_id])


def _spin_texture_config_record(payload: dict) -> dict:
    record = dict(payload)
    legacy_type_key = "wave" + "_type"
    if legacy_type_key in record and "spin_texture_type" not in record:
        record["spin_texture_type"] = record.pop(legacy_type_key)
    basis = record.get("basis")
    order = record.get("order")
    remainder_order = int(order) if order is not None else None
    if basis and not any(" + o(" in str(expression) for expression in basis):
        basis = combine_spin_texture_basis(basis)
        record["basis"] = _append_basis_remainder_ascii(basis, remainder_order)
        basis_latex = spin_texture_basis_latex(basis)
        record["basis_latex"] = _append_basis_remainder_latex(basis_latex, remainder_order)
    else:
        if basis:
            record["basis"] = combine_spin_texture_basis(basis)
            record["basis_latex"] = spin_texture_basis_latex(record["basis"])
        else:
            record.setdefault("basis_latex", spin_texture_basis_latex(record.get("basis")))
    return record


def get_spin_texture_config_id_for_ssg_label(label: str) -> str:
    record = ssg_label_records()[label]
    return record.get("spin_texture_config_id") or record["wave" + "_spin_config_id"]


def get_spin_texture_config_for_ssg_label(label: str) -> dict:
    data = load_acc_aligned_kpoint_runtime_index()
    config_id = get_spin_texture_config_id_for_ssg_label(label)
    records = data.get("spin_texture_config_by_id") or data["wave" + "_spin_config_by_id"]
    return _spin_texture_config_record(records[config_id])
