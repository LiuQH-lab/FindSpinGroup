import numpy as np
import pytest

from findspingroup.core.identify_spin_space_group import (
    _spin_rotations_are_clean_finite_group,
)


def test_individually_finite_order_mirrors_need_not_form_a_group():
    first = np.diag([-1., 1., 1.])
    normal = np.array([.001, 1., 0.])
    normal /= np.linalg.norm(normal)
    second = np.eye(3) - 2*np.outer(normal, normal)
    candidate = [np.eye(3), first, second, np.diag([-1., -1., 1.])]
    for matrix in candidate:
        np.testing.assert_allclose(matrix @ matrix, np.eye(3), atol=1e-14, rtol=0)
    assert not _spin_rotations_are_clean_finite_group(candidate)


def test_exact_group_is_clean_after_an_arbitrary_cartesian_rotation():
    q, _ = np.linalg.qr(np.array([[1., 2., 4.], [3., 5., 7.], [6., 8., 11.]]))
    group = [np.diag(signs) for signs in [(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1)]]
    assert _spin_rotations_are_clean_finite_group([q @ m @ q.T for m in group])


@pytest.mark.parametrize('matrices', [[], [np.diag([-1., 1., 1.])], [np.full((3,3), np.nan)]])
def test_clean_group_requires_finite_matrices_and_identity(matrices):
    assert not _spin_rotations_are_clean_finite_group(matrices)


def test_cartesian_spin_group_must_preserve_the_euclidean_metric():
    # This oblique involution is valid in another metric, not the Cartesian
    # spin representation accepted by this internal model-fitting stage.
    assert not _spin_rotations_are_clean_finite_group(
        [np.eye(3), np.array([[-1., 2., 0.], [0., 1., 0.], [0., 0., 1.]])])


def test_0270_accepted_collinear_representation_is_closed():
    from findspingroup import find_spin_group

    result = find_spin_group('tests/testset/mcif_241130_no2186/0.270_Tb2MnNiO6.mcif')
    assert result.index == '14.14.1.1.L'
    rotations = np.asarray([op.spin_rotation for op in result.acc_primitive_ssg_ops])
    assert _spin_rotations_are_clean_finite_group(rotations, clean_tol=1e-10)
