"""Cell-copy identity is topological; contraction residuals are physical."""
import numpy as np
import pytest

from findspingroup.core.tolerances import Tolerances
from findspingroup.structure.cell import CrystalCell, SpaceToleranceDegeneracyError, change_cell_settings
from findspingroup.utils.periodic import periodic_cartesian_distance


def test_expansion_keeps_different_source_sites_even_with_large_tolerance():
    old = (np.eye(3), [[0, 0, 0], [.001, 0, 0]], [1, 1], [[0, 0, 1], [0, 0, -1]])
    new = change_cell_settings(old, np.diag([.5, 1, 1]), np.zeros(3), eps=.1)
    assert len(new[1]) == 4
    assert sum(np.array_equal(m, [0, 0, 1]) for m in new[3]) == 2
    assert sum(np.array_equal(m, [0, 0, -1]) for m in new[3]) == 2


@pytest.mark.parametrize("length", [.1, 1, 100])
@pytest.mark.parametrize("residual,accepted", [(.005, True), (.025, False)])
def test_contraction_uses_length_not_fractional_or_volume_scaled_tolerance(length, residual, accepted):
    lattice = np.diag([length, 2., 3.])
    old = (lattice, [[.1, .2, .3], [.6 + residual / length, .2, .3]], [1, 1], [[0, 0, 1]]*2)
    if not accepted:
        with pytest.raises(SpaceToleranceDegeneracyError):
            change_cell_settings(old, np.diag([2., 1, 1]), np.zeros(3), eps=.01)
    else:
        new = change_cell_settings(old, np.diag([2., 1, 1]), np.zeros(3), eps=.01)
        assert len(new[1]) == 1
        assert any(periodic_cartesian_distance(new[1][0], np.diag([2., 1, 1]) @ p, new[0]) < 1e-12
                   for p in old[1])


def test_noncommensurate_volume_preserving_map_is_not_a_valid_cell_change():
    matrix = np.diag([1.25, .8, 1])
    old = (np.eye(3), [[.2, .3, .4]], [1])
    with pytest.raises(SpaceToleranceDegeneracyError, match="translation"):
        change_cell_settings(old, matrix, np.zeros(3))


def test_crystal_transform_uses_its_physical_tolerances():
    cell = CrystalCell(np.diag([100., 2, 3]), [[.1, .2, .3], [.60005, .2, .3]],
                       [1, 1], ["Fe", "Fe"], [[0, 0, 1], [0, 0, 1.015]],
                       tol=Tolerances(space=.01, moment=.02))
    new = cell.transform(np.diag([2., 1, 1]), np.zeros(3))
    assert len(new.positions) == 1
    assert new.tol == cell.tol


def test_contraction_rejects_ambiguous_site_identity():
    old = (np.eye(3), [[0, 0, 0], [.01, 0, 0], [.5, 0, 0], [.51, 0, 0]], [1]*4)
    with pytest.raises(SpaceToleranceDegeneracyError, match="ambig"):
        change_cell_settings(old, np.diag([2., 1, 1]), np.zeros(3), eps=.02)


def test_mixed_cell_change_keeps_expected_magnetic_translation_quotient():
    old = (np.eye(3), [[.1, .2, .3], [.6, .2, .3]], [1, 1], [[1, 2, 3]]*2)
    new = change_cell_settings(old, np.diag([2., .5, 1]), [.03, .04, .05], eps=.001)
    assert len(new[1]) == 2
    np.testing.assert_allclose(new[0], np.diag([.5, 2., 1]))
    assert all(np.array_equal(m, [1, 2, 3]) for m in new[3])


@pytest.mark.parametrize("basis", [np.diag([2, 3, 1]), [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
                                   [[1, 0, 0], [1, 2, 0], [1, 1, -1]]])
@pytest.mark.parametrize("lattice", [np.diag([3., 8, 12]), [[3., 0, 0], [2.8, 1., 0], [.2, .3, 5.]]])
def test_expansion_contraction_roundtrip_and_source_permutation(basis, lattice):
    basis, lattice = np.asarray(basis, float), np.asarray(lattice, float)
    old = (lattice, np.array([[.123, .234, .345], [.321, .412, .731]]), [1, 2],
           np.array([[1., 2, 3], [-2., 1, 2]]))
    shift = np.array([.017, -.028, .039])
    expanded = change_cell_settings(old, np.linalg.inv(basis), shift)
    assert len(expanded[1]) == round(2*abs(np.linalg.det(basis)))
    for order in [np.arange(len(expanded[1])), np.arange(len(expanded[1]))[::-1]]:
        permuted = (expanded[0], *[np.asarray(x)[order] for x in expanded[1:]])
        restored = change_cell_settings(permuted, basis, -basis @ shift)
        np.testing.assert_allclose(restored[0], old[0], atol=1e-12, rtol=0)
        assert len(restored[1]) == 2
        for i, position in enumerate(old[1]):
            j = list(restored[2]).index(old[2][i])
            assert periodic_cartesian_distance(restored[1][j], position, lattice) < 1e-12
            np.testing.assert_allclose(restored[3][j], old[3][i], atol=1e-12, rtol=0)


def test_contraction_moment_budget_is_a_closed_vector_norm_not_componentwise():
    old = (np.eye(3), [[.1, .2, .3], [.6, .2, .3]], [1, 1], [[0, 0, 1], [.015, .015, 1]])
    with pytest.raises(SpaceToleranceDegeneracyError, match="moments"):
        change_cell_settings(old, np.diag([2., 1, 1]), np.zeros(3), moment_eps=.02)
    old = (*old[:3], [[0, 0, 1], [.012, .016, 1]])
    assert len(change_cell_settings(old, np.diag([2., 1, 1]), np.zeros(3), moment_eps=.02)[1]) == 1
