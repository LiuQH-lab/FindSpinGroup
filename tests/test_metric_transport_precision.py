from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from findspingroup.structure import SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.structure.group import _normalize_metric


METRIC = np.array([[5033.589352000001,0.,-17353.57668],
                   [0.,56.791295999999996,0.],[-17353.57668,0.,59828.079888]])
TRANSFORM = np.array([[264.99999999911125,0.,-911.9999999969414],
                      [4.2428905399545384e-15,-.9999999999999998,-1.461440074873234e-14],
                      [76.99999999974176,0.,-264.99999999911125]])


def test_sheared_metric_transport_keeps_derived_gram_symmetric():
    inverse = np.linalg.inv(TRANSFORM)
    direct = inverse.T @ METRIC @ inverse
    assert np.max(abs(direct-direct.T)) > 64*np.finfo(float).eps*np.max(abs(direct))
    source = SpinSpaceGroup([SpinSpaceGroupOperation.identity()],real_space_metric=METRIC)
    result = source.transform(TRANSFORM,np.zeros(3),frac=False)
    actual = result.real_space_metric
    np.testing.assert_array_equal(actual,actual.T)
    assert np.linalg.eigvalsh(actual)[0] > 0
    # Independent high-precision congruence for the *same floating-point* P^-1.
    with localcontext() as context:
        context.prec = 60
        g = [[Decimal.from_float(float(v)) for v in row] for row in METRIC]
        p = [[Decimal.from_float(float(v)) for v in row] for row in inverse]
        expected = np.array([[float(sum(p[k][i]*g[k][l]*p[l][j]
                                        for k in range(3) for l in range(3)))
                              for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(source.real_space_metric,METRIC)


def test_raw_asymmetric_user_metric_is_still_rejected():
    bad = METRIC.copy()
    bad[0,2] += 2e-7
    with pytest.raises(ValueError,match='symmetric'):
        _normalize_metric(bad)


@pytest.mark.parametrize('factor',[1e-10,1.,1e10])
def test_metric_transport_preserves_the_quadratic_form_under_unit_rescaling(factor):
    frame = factor*np.array([[2.,.7,.4],[0.,3.,.9],[0.,0.,4.]])
    metric = frame.T@frame
    p = np.array([[1.,2.,0.],[0.,1.,-1.],[0.,0.,1.]])
    ssg = SpinSpaceGroup([SpinSpaceGroupOperation.identity()],real_space_metric=metric)
    new = ssg.transform(p,np.zeros(3),frac=False).real_space_metric
    expected_frame = frame@np.linalg.inv(p)
    np.testing.assert_allclose(new/factor**2,expected_frame.T@expected_frame/factor**2,
                               rtol=1e-14,atol=1e-13)


@pytest.mark.parametrize('name',['2.63_DyCrO3','2.64_DyCrO3'])
def test_sheared_dycro3_poscar_roundtrip_preserves_basic_contract(name,tmp_path):
    from findspingroup.batch_poscar_roundtrip import run_poscar_roundtrip_batch
    path=Path(__file__).parent/'testset'/'mcif_241130_no2186'/f'{name}.mcif'
    summary=run_poscar_roundtrip_batch([path],tmp_path,workers=1,save_poscar=False,quiet=True)
    assert summary['success_count']==1
    assert summary['error_count']==summary['mismatch_count']==0
