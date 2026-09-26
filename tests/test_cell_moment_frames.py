import numpy as np
import pytest

from findspingroup.structure.cell import CrystalCell, calculate_lattice_params, transform_moments


@pytest.mark.parametrize("lattice", [
    [[0., 2, 0], [-3, 0, 0], [0, 0, 4]],
    [[2., 0, 0], [1, 3, 0], [.5, 1, -4]],
    [[1., 2, 3], [-3, 1, 2], [2, -1, 4]],
])
def test_in_lattice_moments_use_actual_normalized_lattice_directions(lattice):
    lattice = np.array(lattice)
    moments = np.array([[.7, -.8, .9]])
    cell = CrystalCell(lattice, [[.1, .2, .3]], [1], ["Fe"], moments, spin_setting="in_lattice")
    expected = moments @ (lattice / np.linalg.norm(lattice, axis=1)[:, None])
    np.testing.assert_allclose(cell.moments_cartesian, expected, atol=1e-12, rtol=0)
    actual = transform_moments(moments, calculate_lattice_params(lattice), lattice_matrix=lattice)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    restored = transform_moments(actual, calculate_lattice_params(lattice), inverse=True, lattice_matrix=lattice)
    np.testing.assert_allclose(restored, moments, atol=1e-12, rtol=0)


@pytest.mark.parametrize("matrix", [np.diag([.5, 1, 1]),
                                    [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                                    [[0, 1, 0], [1, 0, 0], [0, 0, -1]]])
def test_in_lattice_transport_preserves_physical_cartesian_moments(matrix):
    cell = CrystalCell([[3., 0, 0], [1, 4, 0], [.5, 1, 5]], [[.1, .2, .3]],
                       [1], ["Fe"], [[.7, -.8, .9]], spin_setting="in_lattice")
    expected = cell.moments_cartesian[0]
    transformed = cell.transform(np.array(matrix), np.array([.03, -.01, .05]))
    assert transformed.spin_setting == "in_lattice"
    for moment in transformed.moments_cartesian:
        np.testing.assert_allclose(moment, expected, atol=1e-12, rtol=0)


def test_identify_without_primitive_uses_cartesian_spin_action():
    from findspingroup.core.identify_spin_space_group import identify_spin_space_group
    lattice = np.array([[0., 3., 0], [-4., 1, 0], [.5, .4, 5.]])
    cell = CrystalCell(lattice, [[.12, .23, .34]], [1], ["Fe"], [[.7, -.8, .9]], spin_setting="in_lattice")
    group = identify_spin_space_group(cell, find_primitive=False)
    moment = cell.moments_cartesian[0]
    for operation in group.ops:
        np.testing.assert_allclose(operation[0] @ moment, moment, atol=1e-8, rtol=0)
    assert cell.spin_setting == "in_lattice"


def test_magnetic_presence_is_a_physical_norm_in_nonorthogonal_frame():
    cell = CrystalCell([[1., 0, 0], [1., 1e-7, 0], [0, 0, 1]], [[0, 0, 0]],
                       [1], ["Fe"], [[1., -1., 0]], spin_setting="in_lattice")
    assert cell.moments is None
    assert cell.magnetic_atom_indices is None
    assert cell.net_moment < 1e-5
