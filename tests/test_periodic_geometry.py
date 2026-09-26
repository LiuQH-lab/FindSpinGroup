import itertools

import numpy as np
import pytest

from findspingroup.utils.periodic import (
    periodic_cartesian_distance,
    positions_within_cartesian_tolerance,
)


def test_position_tolerance_measures_length_not_fractional_components():
    assert not positions_within_cartesian_tolerance([0, 0, 0], [0.019, 0, 0], np.eye(3) * 30, 0.02)
    assert positions_within_cartesian_tolerance([0, 0, 0], [0.0006, 0, 0], np.eye(3) * 30, 0.02)


def test_skew_lattice_requires_more_than_componentwise_wrapping():
    lattice = np.array([[1.0, 0, 0], [0.9, 0.1, 0], [0, 0, 2.0]])
    delta = np.array([0.49, 0.49, 0])
    expected = min(np.linalg.norm((delta - shift) @ lattice)
                   for shift in itertools.product(range(-2, 3), repeat=3))

    assert periodic_cartesian_distance(delta, np.zeros(3), lattice) == pytest.approx(expected)
    assert np.linalg.norm(delta @ lattice) > expected * 5


@pytest.mark.parametrize("change", [np.eye(3), [[1, 7, 0], [0, 1, 0], [0, 0, 1]],
                                    [[0, 1, 0], [1, 0, 0], [0, 0, -1]]])
def test_periodic_distance_is_invariant_under_unimodular_basis_change(change):
    lattice = np.array([[3.0, 0, 0], [1.1, 4.0, 0], [0.2, 0.4, 5.0]])
    change = np.asarray(change)
    positions = np.random.default_rng(36).random((8, 3))
    for left, right in zip(positions, positions[::-1]):
        before = periodic_cartesian_distance(left, right, lattice)
        after = periodic_cartesian_distance(left @ np.linalg.inv(change), right @ np.linalg.inv(change), change @ lattice)
        assert after == pytest.approx(before, abs=1e-12)


def test_periodic_distance_preserves_origin_rotation_and_input_arrays():
    lattice = np.array([[3.0, 0, 0], [1.1, 4.0, 0], [0.2, 0.4, 5.0]])
    original = lattice.copy()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    left, right = np.array([0.12, 0.91, 0.37]), np.array([0.9, 0.17, 0.61])
    shift = np.array([0.137, 0.271, 0.192731])
    distance = periodic_cartesian_distance(left, right, lattice)

    assert periodic_cartesian_distance(left + shift, right + shift, lattice @ rotation.T) == pytest.approx(distance)
    np.testing.assert_array_equal(lattice, original)


def test_long_vacuum_axis_does_not_change_in_plane_distance():
    assert periodic_cartesian_distance([0.01, 0.02, 0], [0, 0, 0], np.diag([3, 4, 10000])) == pytest.approx(np.hypot(0.03, 0.08))
