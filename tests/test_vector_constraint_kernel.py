import numpy as np
import pytest

from findspingroup.utils.vector_constraints import solve_vector_constraints


def test_duplicates_and_zero_blocks_do_not_change_rank_or_basis():
    block = np.diag([.0005, 1., 1.])
    one = solve_vector_constraints(block, tol=.001)
    many = solve_vector_constraints(np.vstack([block]*9 + [np.zeros((3,3))]*5), tol=.001)
    assert one.dimension == many.dimension == 1
    np.testing.assert_allclose(one.physical_basis @ one.physical_basis.T,
                               many.physical_basis @ many.physical_basis.T, atol=1e-12, rtol=0)
    assert many.distinct_blocks == 1
    assert many.max_operation_residual == pytest.approx(.0005)


def test_rms_cannot_hide_a_violated_operation():
    stacked = np.vstack([np.diag([.0012,0.,0.]), np.diag([0.,1.,1.])])
    with pytest.raises(ValueError, match="full-operation budget"):
        solve_vector_constraints(stacked, tol=.001)


def test_constraint_metric_covariance():
    basis = np.array([[2.,3.,0.],[0.,.5,0.],[0.,0.,4.]])
    inverse = np.linalg.inv(basis)
    n = np.array([1.,2.,3.])/np.sqrt(14)
    action = np.eye(3)-np.outer(n,n)
    source = solve_vector_constraints(action, tol=1e-8)
    target = solve_vector_constraints(inverse @ action @ basis, tol=1e-8, frame=basis)
    assert target.dimension == 1
    np.testing.assert_allclose(source.physical_basis @ source.physical_basis.T,
                               target.physical_basis @ target.physical_basis.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(action @ basis @ target.basis, 0., atol=1e-12, rtol=0)


def test_distinct_operation_order_is_deterministic():
    a, b = np.diag([0.,1.,1.]), np.diag([0.,2.,2.])
    left = solve_vector_constraints(np.vstack([a,b]), tol=1e-8)
    right = solve_vector_constraints(np.vstack([b,a]), tol=1e-8)
    np.testing.assert_array_equal(left.basis, right.basis)


def test_exact_generators_and_full_set_have_the_same_kernel():
    c2 = np.diag([-1.,-1.,1.])
    mirror = np.diag([1.,-1.,1.])
    a = solve_vector_constraints(np.vstack([c2-np.eye(3),mirror-np.eye(3)]), tol=1e-8)
    b = solve_vector_constraints(np.vstack([op-np.eye(3) for op in
                                           [np.eye(3),c2,mirror,c2@mirror]]), tol=1e-8)
    assert a.dimension == b.dimension == 1
    np.testing.assert_allclose(a.basis@a.basis.T,b.basis@b.basis.T,atol=1e-12,rtol=0)


def test_short_equations_preserve_the_complete_right_nullspace():
    result = solve_vector_constraints([[1.,0.,0.]], tol=1e-8)
    assert result.dimension == 2
    np.testing.assert_allclose(result.basis@result.basis.T,np.diag([0,1,1]),atol=1e-12,rtol=0)


def test_empty_equations_are_unconstrained():
    assert solve_vector_constraints(np.zeros((0,3)),tol=1e-8).dimension == 3


def test_tall_stack_uses_economy_svd(monkeypatch):
    original = np.linalg.svd
    flags = []
    def tracked(matrix,*args,**kwargs):
        flags.append((matrix.shape,kwargs.get('full_matrices')))
        return original(matrix,*args,**kwargs)
    monkeypatch.setattr(np.linalg,'svd',tracked)
    blocks = [np.diag([0.,1.,1.+i*1e-5]) for i in range(1000)]
    assert solve_vector_constraints(np.vstack(blocks),tol=1e-8).dimension == 1
    assert ((3000,3),False) in flags
