import itertools
import importlib
import warnings

import numpy as np
import pytest

from findspingroup.structure import SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.structure import CrystalCell
from findspingroup.io.scif_generator import _transform_ssg_ops_to_chen_frame
from findspingroup.utils.international_symbol import (
    _axis_period,
    _integer_bezout_coefficients,
    _select_preferred_primitive_translation_match,
)


def operation(spin, translation):
    return SpinSpaceGroupOperation(np.asarray(spin, float), np.eye(3), translation)


def test_nofrac_and_spin_frame_transport_the_known_period_not_the_origin():
    source = SpinSpaceGroup([SpinSpaceGroupOperation.identity()])
    basis = np.array([[2, 1, 0], [0, 3, 0], [0, 0, 1]], float)
    lifted = source.transform(basis, [0.173, 0.281, 0.397], frac=False)
    np.testing.assert_array_equal(lifted._translation_period_basis, basis)
    reframed = lifted.transform_spin(np.diag([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(reframed._translation_period_basis, basis)
    restored = reframed.transform(np.linalg.inv(basis), [0, 0, 0], frac=False)
    np.testing.assert_allclose(restored._translation_period_basis, np.eye(3), atol=1e-15, rtol=0)


def test_oriented_metric_and_chen_frame_copies_preserve_the_period_contract():
    module = importlib.import_module("findspingroup.find_spin_group")
    basis = np.diag([2, 1, 3])
    group = SpinSpaceGroup([SpinSpaceGroupOperation.identity()], _translation_period_basis=basis)
    cell = CrystalCell(np.diag([4, 5, 6]), [[0, 0, 0]], [1], ["Fe"], [[0, 0, 1]])
    oriented = module._ossg_oriented_spin_frame_ssg(group, cell)
    np.testing.assert_array_equal(oriented._translation_period_basis, basis)
    matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0.5]])
    chen = _transform_ssg_ops_to_chen_frame(oriented, {
        "space_matrix": matrix, "space_shift": [0.17, 0.23, 0.31],
        "spin_basis_rows": np.diag([2, 3, 4]),
    })
    np.testing.assert_array_equal(chen._translation_period_basis, matrix @ basis)


def test_complete_mod1_expansion_resets_period_and_keeps_explicit_centering():
    source = SpinSpaceGroup([SpinSpaceGroupOperation.identity()])
    expanded = source.transform(np.diag([0.5, 1, 1]), [0, 0, 0])
    np.testing.assert_array_equal(expanded._translation_period_basis, np.eye(3))
    assert len(expanded.ops) == 2
    _, vector = _select_preferred_primitive_translation_match(
        expanded.identity_real_nssg_ops, 0, period_basis=expanded._translation_period_basis)
    np.testing.assert_array_equal(vector, [0.5, 0, 0])


@pytest.mark.parametrize("basis", [np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2)])
def test_invalid_known_period_is_rejected(basis):
    with pytest.raises(ValueError, match="translation period"):
        SpinSpaceGroup([SpinSpaceGroupOperation.identity()], _translation_period_basis=basis)


def test_axis_selector_does_not_confuse_g0_integer_lifts_with_spin_only():
    identity = SpinSpaceGroupOperation.identity()
    flip = operation(np.diag([-1, -1, 1]), [1, 0, 1 - 8e-14])
    basis = np.diag([2, 1, 1])
    source, vector = _select_preferred_primitive_translation_match([identity, flip], 0, period_basis=basis)
    assert source is flip
    np.testing.assert_allclose(vector, [1, 0, 0], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(flip.translation, [1, 0, 1 - 8e-14])
    group = SpinSpaceGroup([identity, flip], _translation_period_basis=basis)
    assert len(group.sog) == 1


def test_threefold_axis_action_is_independent_of_positive_or_negative_lifts():
    angle = 2 * np.pi / 3
    spin = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    basis = np.diag([3, 1, 1])
    for shifts in ([0, 0, 0], [5, -7, 3], [-4, 2, -8]):
        ops = [operation(np.linalg.matrix_power(spin, i),
                         np.array([i, 0, 0]) + basis @ [shifts[i], i - 1, -i])
               for i in range(3)]
        selected, vector = _select_preferred_primitive_translation_match(ops, 0, period_basis=basis)
        np.testing.assert_allclose(selected.spin_rotation, spin, atol=1e-12, rtol=0)
        np.testing.assert_allclose(vector, [1, 0, 0], atol=1e-12, rtol=0)


def test_off_axis_centering_coset_is_not_an_axis_translation():
    centered = operation(np.diag([-1, -1, 1]), [0.5, 0.5, 0])
    identity = SpinSpaceGroupOperation.identity()
    for axis in range(3):
        chosen, vector = _select_preferred_primitive_translation_match([centered, identity], axis)
        assert chosen is identity
        np.testing.assert_array_equal(vector, np.zeros(3))


def test_conflicting_spin_actions_are_not_silently_replaced_by_identity():
    ops = [SpinSpaceGroupOperation.identity(), operation(-np.eye(3), [1, 0, 0])]
    with pytest.raises(ValueError, match="Conflicting spin actions"):
        _select_preferred_primitive_translation_match(ops, 0)


@pytest.mark.parametrize("basis", [np.diag([2, 3, 1]),
                                   np.array([[2, 1, 0], [0, 2, 0], [0, 0, 3]]),
                                   np.array([[1, 0, 0], [1, 2, 0], [0, 1, 2]])])
def test_axis_coset_selector_matches_exhaustive_lattice_images(basis):
    images = np.array(list(itertools.product(range(-5, 6), repeat=3))) @ basis.T
    rng = np.random.default_rng(208)
    ops = [operation(np.eye(3), basis @ (np.array(coset) / 2 + rng.integers(-2, 3, 3)))
           for coset in itertools.product(range(2), repeat=3)]
    for axis in range(3):
        expected = []
        off_axes = [i for i in range(3) if i != axis]
        for op in ops:
            points = images + op.translation
            mask = (np.max(np.abs(points[:, off_axes]), axis=1) < 1e-10) & (points[:, axis] > 1e-10)
            expected.extend(points[mask, axis])
        _, vector = _select_preferred_primitive_translation_match(ops, axis, period_basis=basis)
        assert vector[axis] == pytest.approx(min(expected))


def test_bezout_and_rational_axis_period():
    for direction in ([3, 5, 0], [-5, 0, 3], [0, -7, -11], [0, 0, -1]):
        coefficients = _integer_bezout_coefficients(direction)
        assert sum(a * b for a, b in zip(coefficients, direction)) == 1
    inverse, repeat, coefficients = _axis_period(np.diag([3, 2, 1]), 0, tol=1e-4)
    assert repeat == 3
    np.testing.assert_allclose(repeat * inverse[:, 0], [1, 0, 0], atol=1e-15, rtol=0)
    assert coefficients == [1, 0, 0]


def test_noncrystallographic_period_is_not_silently_rationalized():
    with pytest.raises(ValueError, match="rational translation period"):
        _axis_period(np.diag([np.sqrt(2), 1, 1]), 0, tol=1e-4)


@pytest.mark.parametrize("case", ["2.63_DyCrO3", "2.21_TbOOH", "1.357_Ho3Ge4", "1.13_Ba3Nb2NiO9"])
def test_all_scif_frames_keep_chen_name_for_lifted_translation_groups(case):
    module = importlib.import_module("findspingroup.find_spin_group")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = module.find_spin_group(f"tests/testset/mcif_241130_no2186/{case}.mcif")
    assert not any("Unable to build Chen/database-facing linear name" in str(item.message) for item in caught)
    assert len(result.scif_outputs) == 8
    for mode, text in result.scif_outputs.items():
        line = next(row for row in text.splitlines() if row.startswith("_space_group_spin.name_Chen_Liu "))
        value = line.split(maxsplit=1)[1].strip()
        assert value not in (".", "?", "''", '\"\"'), mode
        # These Chen-frame groups have a nontrivial axial twofold spin action;
        # a nonempty but all-identity translation factor is also incorrect.
        expected = {
            "2.63_DyCrO3": "(1,1,2_{001})",
            "2.21_TbOOH": "(1,1,2_{010})",
            "1.13_Ba3Nb2NiO9": "(3^{2}_{001},3^{2}_{001},2_{001})",
        }.get(case)
        if expected is not None:
            assert expected in value, mode
