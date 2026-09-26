import numpy as np
import pytest

from findspingroup.spin_splitting import svd_nullspace


@pytest.mark.parametrize('shape',[(600,3),(3,3),(2,5)])
def test_polynomial_svd_allocates_only_the_required_singular_spaces(monkeypatch,shape):
    matrix=np.random.default_rng(208).normal(size=shape)
    matrix[:,-1]=matrix[:,0]
    original=np.linalg.svd
    _,s,vh=original(matrix,full_matrices=True)
    rank=int(np.count_nonzero(s>1e-10))
    expected=vh[rank:].T
    calls=[]
    def tracked(a,*,full_matrices=True,**kwargs):
        calls.append(full_matrices)
        return original(a,full_matrices=full_matrices,**kwargs)
    monkeypatch.setattr(np.linalg,'svd',tracked)
    result=svd_nullspace(matrix,rtol=0.,atol=1e-10,confidence_gap=100.)
    basis=result[-1]
    assert calls==[shape[0]<shape[1]]
    assert basis.shape==(shape[1],shape[1]-rank)
    np.testing.assert_allclose(basis@basis.T,expected@expected.T,atol=1e-13,rtol=0)
    np.testing.assert_allclose(matrix@basis,0.,atol=1e-12,rtol=0)


def test_empty_polynomial_system_keeps_all_free_coefficients():
    result=svd_nullspace(np.zeros((0,6)),rtol=0.,atol=1e-10,confidence_gap=100.)
    assert result[0]==0
    np.testing.assert_array_equal(result[-1],np.eye(6))
