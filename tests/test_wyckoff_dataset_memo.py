import importlib

import numpy as np


find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")


class _FakeCell:
    def __init__(self):
        self.calls = []

    def to_spglib(self, mag=False):
        self.calls.append(bool(mag))
        return ("magnetic" if mag else "nonmagnetic",)


def test_request_local_wyckoff_dataset_memo_reuses_only_exact_inputs(monkeypatch):
    symmetry_calls = []
    g0_calls = []

    def fake_symmetry_dataset(cell, *, symprec):
        symmetry_calls.append((cell, symprec))
        return object()

    def fake_g0_dataset(operations, cell, symprec):
        g0_calls.append((operations, cell, symprec))
        return object()

    monkeypatch.setattr(
        find_spin_group_module,
        "get_symmetry_dataset",
        fake_symmetry_dataset,
    )
    monkeypatch.setattr(
        find_spin_group_module,
        "get_G0_dataset_for_cell",
        fake_g0_dataset,
    )

    cell = _FakeCell()
    memo = {}
    operations = (
        (np.eye(3), np.zeros(3)),
        (np.diag([-1.0, -1.0, 1.0]), np.array([0.0, 0.0, 0.5])),
    )

    first_symmetry = find_spin_group_module._symmetry_dataset_for_analysis(
        cell, 0.02, memo=memo
    )
    second_symmetry = find_spin_group_module._symmetry_dataset_for_analysis(
        cell, 0.02, memo=memo
    )
    first_g0 = find_spin_group_module._g0_dataset_for_analysis(
        operations, cell, 0.02, memo=memo
    )
    second_g0 = find_spin_group_module._g0_dataset_for_analysis(
        operations, cell, 0.02, memo=memo
    )
    reversed_g0 = find_spin_group_module._g0_dataset_for_analysis(
        tuple(reversed(operations)), cell, 0.02, memo=memo
    )

    assert first_symmetry is second_symmetry
    assert first_g0 is second_g0
    assert reversed_g0 is not first_g0
    assert len(symmetry_calls) == 1
    assert len(g0_calls) == 2
    assert cell.calls.count(False) == 1
    assert cell.calls.count(True) == 1
