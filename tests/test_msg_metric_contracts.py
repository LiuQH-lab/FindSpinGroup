"""MSG membership budgets refer to physical action in an oriented basis."""
import numpy as np
import pytest

from findspingroup.structure import SpinSpaceGroup, SpinSpaceGroupOperation


@pytest.mark.parametrize("shear", [0., 3., 20.])
def test_msg_membership_is_covariant_in_the_declared_metric(shear):
    basis = np.array([[1., shear, 0.], [0., 1., 0.], [0., 0., 1.]])
    angle = .009
    noise = np.array([[np.cos(angle), -np.sin(angle), 0.],
                      [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    real = np.diag([-1., -1., 1.])
    inverse = np.linalg.inv(basis)
    op = SpinSpaceGroupOperation(inverse @ noise @ real @ basis, inverse @ real @ basis, np.zeros(3))
    group = SpinSpaceGroup([SpinSpaceGroupOperation.identity(), op], tol=.01,
                           real_space_metric=basis.T @ basis)
    assert group.classify_magnetic_operation(op) == 1


def test_full_rotation_error_is_not_hidden_by_componentwise_bounds():
    axis = np.ones(3)/np.sqrt(3)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    angle = .015
    noise = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
    real = np.diag([-1., -1., 1.])
    op = SpinSpaceGroupOperation(noise @ real, real, np.zeros(3))
    group = SpinSpaceGroup([SpinSpaceGroupOperation.identity(), op], tol=.01,
                           real_space_metric=np.eye(3))
    assert group.classify_magnetic_operation(op) is None


@pytest.mark.parametrize("time_sign", [-1, 1])
def test_improper_real_operation_uses_axial_action(time_sign):
    op = SpinSpaceGroupOperation(time_sign*np.eye(3), -np.eye(3), np.zeros(3))
    assert op.magnetic_time_reversal(atol=0., metric=np.eye(3)) == time_sign
    assert op.is_magnetic_space_group_operation(atol=0., metric=np.eye(3))


@pytest.mark.parametrize("metric", [np.zeros((3,3)), np.diag([1.,1.,-1.]),
                                     np.full((3,3), np.nan),
                                     np.array([[1.,.1,0.],[0.,1.,0.],[0.,0.,1.]])])
def test_invalid_metric_is_not_silently_used(metric):
    with pytest.raises(ValueError, match="metric"):
        SpinSpaceGroup([SpinSpaceGroupOperation.identity()], real_space_metric=metric)


def test_algebraic_mode_does_not_add_implicit_relative_tolerance():
    op = SpinSpaceGroupOperation(np.diag([1.0000005, 1, 1]), np.eye(3), np.zeros(3))
    assert op.magnetic_time_reversal(atol=1e-8) is None
