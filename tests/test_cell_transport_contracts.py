import numpy as np
import pytest

from findspingroup.structure.cell import change_cell_settings, SpaceToleranceDegeneracyError
from findspingroup.utils.periodic import periodic_cartesian_distance
import findspingroup.structure.cell as cell_module


def _cell(positions, lattice=None, moments=None):
    return (np.eye(3) if lattice is None else np.asarray(lattice), np.asarray(positions),
            [1] * len(positions), np.zeros((len(positions), 3)) if moments is None else moments)


@pytest.mark.parametrize("shift", [-5e-9, 5e-9, -5e-5, 5e-5])
@pytest.mark.parametrize("matrix", [np.eye(3), np.diag([0.5, 1, 1])])
def test_cell_origin_transport_keeps_resolved_displacements(shift, matrix):
    source = _cell([[0, 0, 0]], np.diag([10, 11, 12]))
    result = change_cell_settings(source, matrix, [shift, 0, 0])
    assert any(periodic_cartesian_distance(p, [shift, 0, 0], result[0]) < 1e-12 for p in result[1])


def test_near_integer_matrix_is_not_a_silent_replacement_cell():
    with pytest.raises(SpaceToleranceDegeneracyError, match="multiplicity"):
        change_cell_settings(_cell([[0, 0, 0]]), np.diag([1.00005, 1, 1]), [0, 0, 0])


def test_roundoff_near_integer_matrix_keeps_the_supplied_affine_map():
    matrix = np.diag([1 + 5e-12, 1, 1])
    source = _cell([[.23, .34, .45]], np.diag([10, 11, 12]))
    result = change_cell_settings(source, matrix, [0, 0, 0])
    np.testing.assert_allclose(result[0], np.linalg.inv(matrix).T @ source[0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result[1][0], matrix @ source[1][0], atol=1e-14, rtol=0)


def test_unimodular_shear_preserves_distinct_source_atom_identities():
    source = _cell([[0, 0, 0], [.00991, -.000099, 0]], np.eye(3) * 50,
                   [[0, 0, 1], [0, 0, -1]])
    matrix = np.array([[1, 100, 0], [0, 1, 0], [0, 0, 1]])
    result = change_cell_settings(source, matrix, np.zeros(3))
    assert len(result[1]) == 2
    for pos, moment in zip(source[1], source[3]):
        assert any(periodic_cartesian_distance(matrix @ pos, target, result[0]) < 1e-12
                   and np.array_equal(moment, target_moment)
                   for target, target_moment in zip(result[1], result[3]))


def test_identity_transform_does_not_rediscover_or_merge_nearby_sites():
    source = _cell([[0, 0, 0], [.05, 0, 0]], moments=[[0, 0, 1], [0, 0, -1]])
    result = change_cell_settings(source, np.eye(3), np.zeros(3), eps=.1)
    assert len(result[1]) == 2


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2),
                                   np.eye(3) * 1e-200, np.eye(3) * 1e200])
def test_cell_transform_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        change_cell_settings(_cell([[0, 0, 0]]), matrix, np.zeros(3))


def test_cell_transform_rejects_nonintegral_atom_multiplicity():
    with pytest.raises(SpaceToleranceDegeneracyError, match="multiplicity"):
        change_cell_settings(_cell([[0, 0, 0]]), np.diag([2., 1, 1]), np.zeros(3))


@pytest.mark.parametrize("matrix", [np.eye(3), np.array([[1, 2, 0], [0, 1, 1], [0, 0, 1]]),
                                   np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])])
def test_fast_and_general_paths_use_the_same_affine_map_and_image_order(monkeypatch, matrix):
    source = _cell([[.11, .12, .13], [.31, .32, .33]], np.array([[2, 0, 0], [.2, 3, 0], [.1, .4, 4]]),
                   [[0, 0, 1], [1, 0, 0]])
    shift = np.array([-.13, .52, 1.29])
    fast = change_cell_settings(source, matrix, shift)
    monkeypatch.setattr(cell_module, "_change_cell_settings_unimodular_fast_path", lambda *a, **kw: None)
    general = change_cell_settings(source, matrix, shift)
    for actual, expected in zip(fast, general):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
