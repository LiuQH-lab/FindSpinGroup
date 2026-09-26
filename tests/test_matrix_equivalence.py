import numpy as np
import pytest

from findspingroup.core.identify_symmetry_from_ops import (
    deduplicate_matrix_pairs,
    is_close_matrix_pair,
)


@pytest.mark.parametrize("tol", [1e-5, 1e-3, 1e-2])
@pytest.mark.parametrize("reverse", [False, True])
def test_matrix_deduplication_checks_both_sides_of_rounding_boundary(tol, reverse):
    first = np.eye(3)
    second = np.eye(3)
    first[0, 1] = 4.9 * tol
    second[0, 1] = 5.1 * tol
    items = [first, second][:: -1 if reverse else 1]

    unique = deduplicate_matrix_pairs(items, tol=tol)

    assert len(unique) == 1
    np.testing.assert_array_equal(unique[0], items[0])


def test_matrix_pair_deduplication_handles_mixed_component_shapes():
    first = [np.eye(3), np.array([0.049, 0.0, 0.0])]
    second = [np.eye(3), np.array([0.051, 0.0, 0.0])]

    assert len(deduplicate_matrix_pairs([first, second], tol=0.01)) == 1


def test_absolute_matrix_comparison_does_not_grow_with_matrix_entries():
    first = np.eye(3)
    second = np.eye(3)
    first[0, 1] = 10000.0
    second[0, 1] = 10000.02

    assert not is_close_matrix_pair(first, second, tol=1e-4)
    assert len(deduplicate_matrix_pairs([first, second], tol=1e-4)) == 2


def test_matrix_deduplication_keeps_distinct_operations_and_input_arrays():
    first = np.eye(3)
    second = np.diag([-1.0, -1.0, 1.0])
    originals = [first.copy(), second.copy()]

    unique = deduplicate_matrix_pairs([first, second, first.copy()], tol=1e-5)

    assert len(unique) == 2
    for actual, original in zip((first, second), originals):
        np.testing.assert_array_equal(actual, original)


def test_bucket_deduplication_agrees_with_exhaustive_absolute_comparison():
    random = np.random.default_rng(36)
    tolerance = 1e-3
    items = []
    for _ in range(24):
        matrix = random.normal(size=(3, 3))
        items.extend([matrix, matrix + random.uniform(-0.9, 0.9, size=(3, 3)) * tolerance])
    random.shuffle(items)
    expected = []
    for item in items:
        if not any(is_close_matrix_pair(item, previous, tolerance) for previous in expected):
            expected.append(item)

    actual = deduplicate_matrix_pairs(items, tol=tolerance)

    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
