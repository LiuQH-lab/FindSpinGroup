import importlib
import numpy as np
import pytest

from findspingroup.core.tolerances import Tolerances
from findspingroup.structure import SpinSpaceGroupOperation

identify = importlib.import_module('findspingroup.core.identify_spin_space_group')


def test_algebraic_closure_budget_does_not_use_position_length_tolerance():
    ops=[SpinSpaceGroupOperation.identity(),
         SpinSpaceGroupOperation(np.eye(3),np.eye(3),[.49,0.,0.])]
    for space in (.001,.02,1.,100.):
        tolerance=identify._candidate_audit_tol(Tolerances(space=space,m_matrix_tol=.001))
        assert tolerance == .001
        assert identify._spin_space_group_closure_failure(ops,tolerance,'test') is not None


def test_physical_preservation_cache_does_not_merge_nearby_products(monkeypatch):
    checked=[]
    def preserves(s,r,t):
        checked.append(float(t[0]))
        return not np.isclose(t[0],.50202,atol=1e-10,rtol=0)
    monkeypatch.setattr(identify,'_build_magnetic_atom_preservation_checker',lambda *args:preserves)
    monkeypatch.setattr(identify,'_magnetic_action_residual',lambda *args:(0.,.0201))
    cache={}
    for shift in [.251]:
        identify._complete_ssg_ops_by_closure(
            [SpinSpaceGroupOperation.identity(),SpinSpaceGroupOperation(np.eye(3),np.eye(3),[shift,0.,0.])],
            [],group_tol=Tolerances(),preserve_cache=cache)
    with pytest.raises(ValueError,match='does not preserve'):
        identify._complete_ssg_ops_by_closure(
            [SpinSpaceGroupOperation.identity(),SpinSpaceGroupOperation(np.eye(3),np.eye(3),[.25101,0.,0.])],
            [],group_tol=Tolerances(),preserve_cache=cache)
    assert any(np.isclose(v,.50202,atol=1e-10,rtol=0) for v in checked)


@pytest.mark.parametrize('spatial',[False,True])
def test_lookup_bucket_never_accepts_a_difference_above_a_tight_budget(spatial):
    rotation=np.eye(3);shift=np.zeros(3)
    moved=np.array([2e-9,0.,0.])
    if spatial:
        lookup=identify._SpatialOperationLookup([(rotation,shift)],tol=1e-12)
        assert not lookup.contains((rotation,moved))
    else:
        lookup=identify._SpinSpaceOperationLookup([SpinSpaceGroupOperation.identity()],tol=1e-12)
        assert not lookup.contains(SpinSpaceGroupOperation(rotation,rotation,moved))


def test_operation_audit_has_no_implicit_relative_matrix_slack():
    rotation=np.array([[1.,10000.,0.],[0.,-1.,0.],[0.,0.,1.]])
    first=SpinSpaceGroupOperation(np.eye(3),rotation,np.zeros(3))
    altered=rotation.copy();altered[0,1]+=.02
    second=SpinSpaceGroupOperation(np.eye(3),altered,np.zeros(3))
    assert not identify._spin_space_operation_same(first,second,1e-4)
    assert not identify._SpinSpaceOperationLookup([first],tol=1e-4).contains(second)


def test_spatial_composition_does_not_snap_a_resolved_translation():
    rotation,shift=identify._compose_spatial_operation(
        (np.eye(3),np.array([.499999,0.,0.])),(np.eye(3),np.array([.5,0.,0.])))
    assert abs(shift[0]-.999999) <= np.finfo(float).eps
    inverse=identify._invert_spatial_operation((rotation,np.array([1e-9,0.,0.])))
    assert inverse[1][0] > .99999999
