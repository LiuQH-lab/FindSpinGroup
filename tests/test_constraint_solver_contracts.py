"""Numerical rank is an accepted-model decision, not an equation-count vote."""
import numpy as np

from findspingroup.kpoint_spin_polarization import _structured_spin_constraint
from findspingroup.structure.group import solve_spin_constraint_from_stacked


def test_repeating_operation_constraints_does_not_remove_a_spin_direction():
    block = np.diag([.0005, 1., 1.])
    one = solve_spin_constraint_from_stacked(block, tol=.001)
    repeated = solve_spin_constraint_from_stacked(np.tile(block, (9, 1)), tol=.001)
    assert one == repeated == ("spin splitting", ["Sx", "0", "0"])


def test_identity_padding_and_order_leave_the_subspace_unchanged():
    blocks = [np.diag([0., 1., 1.]), np.diag([0., 2., 2.])]
    expected = _structured_spin_constraint(np.vstack(blocks), tol=.001)
    actual = _structured_spin_constraint(np.vstack([np.zeros((3,3)), *blocks[::-1]]*5), tol=.001)
    assert actual['dimension'] == expected['dimension'] == 1
    np.testing.assert_allclose(actual['projector_acc_primitive_cartesian'],
                               expected['projector_acc_primitive_cartesian'], atol=1e-12, rtol=0)


def test_readable_constraint_preserves_small_resolved_direction_components():
    axis = np.array([.0007, 0., 1.])
    axis /= np.linalg.norm(axis)
    rotation = 2*np.outer(axis,axis)-np.eye(3)
    status, constraint = solve_spin_constraint_from_stacked(rotation-np.eye(3))
    assert status == "spin splitting"
    assert constraint == ["0.0007*Sz", "0", "Sz"]
