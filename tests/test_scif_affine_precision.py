import numpy as np
import pytest

from findspingroup.io.scif_generator import affine_matrix_to_xyz_expression, _stabilize_fractional_boundary_value
from findspingroup.utils import general_positions_to_matrix


@pytest.mark.parametrize("translation", [[.0004, -.0006, .5012], [1.23456789, -.1244, .3756], [1/3, 1/4, -1/6]])
@pytest.mark.parametrize("separate", [False, True])
def test_scif_affine_translation_roundtrip_is_precision_bounded(translation, separate):
    text = affine_matrix_to_xyz_expression(np.eye(3), translation, separate_translation=separate, coeff_precision=15)
    if separate:
        expr, shift = text.split(";")
        from findspingroup.utils.matrix_utils import evaluate_numeric_expression
        actual_translation = np.array([evaluate_numeric_expression(v) for v in shift.split(",")])
        ops, _ = general_positions_to_matrix([expr])
    else:
        ops, _ = general_positions_to_matrix([text])
        actual_translation = ops[0][1]
    np.testing.assert_allclose(ops[0][0], np.eye(3), atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual_translation, translation, atol=1e-13, rtol=0)


def test_small_spin_rotation_coefficients_are_not_dropped():
    angle = .0004
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    text = affine_matrix_to_xyz_expression(matrix, coeff_precision=15)
    ops, _ = general_positions_to_matrix([text], variables=("u", "v", "w"))
    np.testing.assert_allclose(ops[0][0], matrix, atol=1e-13, rtol=0)


def test_default_affine_text_is_suitable_for_computational_reuse():
    matrix = np.array([[1, -1.23456789123e-6, 0], [0, 1, 0], [0, 0, 1]])
    text = affine_matrix_to_xyz_expression(matrix)
    ops, _ = general_positions_to_matrix([text], variables=("u", "v", "w"))
    np.testing.assert_allclose(ops[0][0], matrix, atol=1e-14, rtol=0)


@pytest.mark.parametrize("value", [5e-6, 1-5e-6, -5e-6, 5e-9])
def test_scif_site_boundary_cleanup_does_not_consume_resolved_coordinates(value):
    difference = (_stabilize_fractional_boundary_value(value) - value + .5) % 1 - .5
    assert abs(difference) < 1e-14


def test_fractional_simple_values_remain_compact():
    assert affine_matrix_to_xyz_expression(np.eye(3), [1/2, 1/3, 1/4], coeff_precision=15) == "x+1/2,y+1/3,z+1/4"
