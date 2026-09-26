"""Site actions use cell-length budgets, independent of spin-matrix rank."""
import numpy as np
import pytest

from findspingroup.structure import CrystalCell, SpinSpaceGroupOperation
from findspingroup.core.tolerances import Tolerances
from findspingroup.find_spin_group import _spin_space_site_orbits


def cell(lattice, positions, space=.02):
    n = len(positions)
    return CrystalCell(lattice, positions, [1]*n, ['Fe']*n, [[0, 0, 1]]*n,
                       tol=Tolerances(space=space))


def operation(rotation=None, translation=None, spin=None):
    return SpinSpaceGroupOperation(np.eye(3) if spin is None else spin,
                                   np.eye(3) if rotation is None else rotation,
                                   np.zeros(3) if translation is None else translation)


def test_site_position_budget_is_not_the_matrix_budget():
    structure = cell(np.diag([.1, 1., 1.]), [[0, 0, 0]])
    ops = [operation(), operation(translation=[.1, 0, 0], spin=np.diag([-1, -1, 1]))]
    _, orbits, _ = _spin_space_site_orbits(structure, ops, atol=1e-6)
    assert len(orbits[0]['site_symmetry_ops']) == 2


def test_large_lattice_does_not_inflate_allowed_displacement():
    structure = cell(np.diag([100., 1., 1.]), [[0, 0, 0]])
    with pytest.raises(ValueError, match="site action"):
        _spin_space_site_orbits(structure, [operation(translation=[.001, 0, 0])], atol=.01)


def test_nearest_site_matching_must_be_bijective():
    structure = cell(np.eye(3), [[0, 0, 0], [.003, 0, 0]], space=.01)
    with pytest.raises(ValueError, match="bijective"):
        _spin_space_site_orbits(structure, [operation(translation=[.002, 0, 0])], atol=.01)


def test_site_stabilizer_keeps_identity_on_nearby_distinct_sites():
    structure = cell(np.eye(3), [[0, 0, 0], [.003, 0, 0]], space=.01)
    _, orbits, _ = _spin_space_site_orbits(structure, [operation()], atol=.01)
    assert [r['class_indices'] for r in orbits] == [[0], [1]]
    assert all(len(r['site_symmetry_ops']) == 1 for r in orbits)


def test_skew_cell_action_uses_true_minimum_image():
    lattice = np.array([[1., 0, 0], [1., .01, 0], [0, 0, 1.]])
    structure = cell(lattice, [[0, 0, 0]], space=.006)
    _, orbits, _ = _spin_space_site_orbits(
        structure, [operation(translation=[.5, -.5, 0])], atol=1e-6)
    assert len(orbits[0]['site_symmetry_ops']) == 1


def test_real_action_cache_does_not_drop_spin_only_stabilizers():
    structure = cell(np.eye(3), [[0, 0, 0], [.5, 0, 0]])
    ops = [operation(), operation(spin=np.diag([-1, -1, 1])),
           operation(translation=[.5, 0, 0]),
           operation(translation=[.5, 0, 0], spin=np.diag([-1, -1, 1]))]
    _, orbits, _ = _spin_space_site_orbits(structure, ops)
    assert len(orbits) == 1
    assert orbits[0]['class_indices'] == [0, 1]
    assert len(orbits[0]['site_symmetry_ops']) == 2
