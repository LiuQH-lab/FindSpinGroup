import itertools

import numpy as np
import pytest

from findspingroup.core.identify_spin_space_group import (
    _best_magnetic_residual_match,
    _build_magnetic_atom_preservation_checker,
    _has_bijective_site_matching,
    _mag_atom_neighbor_keys,
    _magnetic_action_residual_details,
    _magnetic_residual_targets,
)
from findspingroup.core.tolerances import Tolerances
from findspingroup.structure import AtomicSite, SpinSpaceGroupOperation
from findspingroup.utils.periodic import periodic_cartesian_distance


def _site(position, lattice, moment=(0, 0, 1), element="Fe", occupancy=1.0):
    return AtomicSite(position, moment, occupancy, element, lattice_matrix=lattice)


def test_preservation_and_residual_use_the_same_physical_length():
    lattice = np.eye(3) * 30
    atom = _site([0, 0, 0], lattice)
    tol = Tolerances(space=0.02)
    preserves = _build_magnetic_atom_preservation_checker([atom], tol)
    for shift, expected in [(0.001, False), (0.0006, True)]:
        translation = [shift, 0, 0]
        assert preserves(np.eye(3), np.eye(3), translation) is expected
        residual = _magnetic_action_residual_details(
            SpinSpaceGroupOperation(np.eye(3), np.eye(3), translation), [atom], tol
        )
        assert residual["max_position"] == pytest.approx(30 * shift)
        assert (residual["normalized"] <= 1) is expected
    assert not atom.is_equivalent(_site([0.001, 0, 0], lattice), tol)
    assert _site([0, 0, 0], None).is_equivalent(_site([0.001, 0, 0], None), tol)


@pytest.mark.parametrize("change", [np.eye(3), [[1, 7, 0], [0, 1, 0], [0, 0, 1]]])
def test_site_matching_is_origin_and_basis_invariant(change):
    lattice = np.array([[3, 0, 0], [0.8, 4, 0], [0, 0, 30.]])
    positions = np.array([[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]])
    change = np.asarray(change)
    inverse = np.linalg.inv(change)
    tol = Tolerances(space=0.02)
    for origin in [np.zeros(3), np.array([0.137, 0.271, 0.31])]:
        sites = [_site(p, change @ lattice) for p in (positions + origin) @ inverse]
        preserves = _build_magnetic_atom_preservation_checker(sites, tol)
        for epsilon, expected in [(0.0006, True), (0.001, False)]:
            translation = np.array([0.5, 0, epsilon]) @ inverse
            assert preserves(np.eye(3), np.eye(3), translation) is expected
            residual = _magnetic_action_residual_details(
                SpinSpaceGroupOperation(np.eye(3), np.eye(3), translation), sites, tol
            )
            assert residual["max_position"] == pytest.approx(30 * epsilon)


@pytest.mark.parametrize("candidates,expected", [
    ([[2], [2], [0, 1, 2]], False),
    ([[0, 1], [0, 2], [0, 2]], True),
    ([[], [1]], False),
    ([[0], [1]], True),
])
def test_site_matches_must_form_a_bijection(candidates, expected):
    assert _has_bijective_site_matching(candidates) is expected


def test_bijection_matches_exhaustive_permutations():
    rng = np.random.default_rng(36)
    for _ in range(30):
        candidates = [np.flatnonzero(rng.random(5) < 0.4).tolist() for _ in range(5)]
        expected = any(all(target in candidates[i] for i, target in enumerate(permutation))
                       for permutation in itertools.permutations(range(5)))
        assert _has_bijective_site_matching(candidates) is expected


def test_residual_bounds_match_exhaustive_minimum_image_search():
    rng = np.random.default_rng(36)
    lattice = np.array([[1., 0, 0], [0.9, 0.1, 0], [0, 0, 12.]])
    sites = [_site(rng.random(3), lattice, rng.normal(size=3), "Fe", rng.random()) for _ in range(12)]
    sites.append(_site([0, 0, 0], lattice, element="O"))
    targets = _magnetic_residual_targets(sites)
    tol = Tolerances(space=0.02, moment=0.01, occupancy=0.05)
    for _ in range(10):
        atom = _site(rng.random(3), lattice, rng.normal(size=3), occupancy=rng.random())
        expected = []
        for index, target in enumerate(sites):
            if atom.element_symbol != target.element_symbol:
                continue
            position = periodic_cartesian_distance(atom.position, target.position, lattice)
            moment = np.linalg.norm(atom.magnetic_moment - target.magnetic_moment)
            occupancy = abs(atom.occupancy - target.occupancy)
            normalized = max(position / tol.space, moment / tol.moment, occupancy / tol.occupancy)
            expected.append(((normalized, position, moment, occupancy), index))
        residual, target_index = _best_magnetic_residual_match(atom, targets["Fe"], tol)
        expected_residual, expected_index = min(expected)
        assert residual == pytest.approx(expected_residual)
        assert target_index == expected_index


def test_large_fractional_search_radius_does_not_duplicate_buckets():
    assert list(_mag_atom_neighbor_keys((0, 0, 0), 1, 100)) == [(0, 0, 0)]
    assert len(set(_mag_atom_neighbor_keys((0, 0, 0), 3, 100))) == 27
