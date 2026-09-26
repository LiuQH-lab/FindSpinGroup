from types import SimpleNamespace
import numpy as np

from findspingroup.core.tolerances import Tolerances
from findspingroup.ferroelectric import (
    _match_transformed_sites, _collinear_pattern_context,
    _transformed_collinear_pattern, build_parent_standard_supercell_domain_coset_analysis,
    _msg_compatible_collinear_branch,
)


def test_domain_site_matching_uses_a_length_budget():
    assert _match_transformed_sites(np.array([[0.,0.,0.]]),[1],np.eye(3),[.001,0.,0.],
                                    lattice=np.diag([100.,1.,1.]),tol=.02) is None


def test_domain_site_mapping_uses_the_nearest_typed_site():
    x=np.array([[0.,0.,0.],[.009,0.,0.]])
    assert _match_transformed_sites(x,[1,1],-np.eye(3),[.009,0.,0.],
                                    lattice=np.eye(3),tol=.02)==[1,0]


def test_nonbijective_nearest_domain_action_is_not_greedily_repaired():
    x=np.array([[0.,0.,0.],[.009,0.,0.]])
    assert _match_transformed_sites(x,[1,1],-np.eye(3),[.004,0.,0.],
                                    lattice=np.eye(3),tol=.02) is None


def cell(moments):
    return SimpleNamespace(positions=np.array([[0.,0.,0.],[.5,0.,0.]]),
                           atom_types=[1,1],moments_cartesian=np.array(moments),
                           lattice_matrix=np.eye(3),tol=Tolerances())


def test_signed_order_does_not_drop_nonzero_moments_at_a_matrix_tolerance():
    context=_collinear_pattern_context(cell([[0,0,.005],[0,0,-.005]]),[0,0,1],tol=.01)
    assert context is not None
    np.testing.assert_array_equal(context['signed_pattern'],[1,-1])


def test_signed_order_reports_unresolved_projection_instead_of_a_nonmagnetic_site():
    context=_collinear_pattern_context(cell([[.005,0,0],[0,0,1.]]),[0,0,1],tol=.01)
    pattern,status=_transformed_collinear_pattern(context,np.eye(3),np.zeros(3),spin_branch=1,tol=.01)
    assert pattern is None
    assert status=='not_evaluated_unresolved_collinear_projection'


def test_complete_soc_axes_are_solved_in_the_child_basis_not_a_number_table():
    result=build_parent_standard_supercell_domain_coset_analysis(
        parent_space_group_number=47,parent_space_group_symbol='Pmmm',parent_hall_number=227,
        child_basis_in_parent=np.eye(3),child_origin_in_parent=np.zeros(3),
        ordered_magnetic_ops=[(np.eye(3),np.zeros(3),1),
                              (np.diag([1.,-1.,-1.]),np.zeros(3),1)],
        ordered_space_group_number=3,basis_setting='child',relation_layer='soc_magnetic',
        subgroup_time_branch_scope='full')
    assert result['ordered_polar_axes'][0]['components']==[1.,0.,0.]


def test_domain_soc_classification_uses_a_vector_action_budget():
    axis=np.array([1.,1.,.006*np.sqrt(2)]);axis/=np.linalg.norm(axis)
    rotation=2*np.outer(axis,axis)-np.eye(3)
    context={'axis':np.array([0.,0.,1.]),'lattice':np.eye(3)}
    assert not _msg_compatible_collinear_branch(rotation=rotation,time_reversal=-1,
                                               spin_branch=1,context=context,tol=.01)


def test_soc_domain_stabilizer_uses_oriented_msg_not_cartesian_component_comparison(monkeypatch):
    import importlib
    from findspingroup.structure import CrystalCell, SpinSpaceGroup, SpinSpaceGroupOperation
    fsg=importlib.import_module('findspingroup.find_spin_group')
    frame=np.array([[1.,3.,0.],[0.,1.,0.],[0.,0.,1.]])
    real=np.linalg.inv(frame)@np.diag([1.,-1.,-1.])@frame
    spins=[np.eye(3),np.diag([1.,-1.,-1.]),np.diag([1.,1.,-1.]),np.diag([1.,-1.,1.])]
    group=SpinSpaceGroup([SpinSpaceGroupOperation(s,r,np.zeros(3)) for s in spins
                          for r in [np.eye(3),real]])
    structure=CrystalCell(frame.T,[[0,0,0]],[1.],['Fe'],[[1.,0.,0.]],spin_setting='cartesian')
    monkeypatch.setattr(fsg,'_build_g0std_parent_coset_analysis',lambda **kwargs:kwargs)
    result=fsg._build_g0std_soc_domain_reversal_coset_analysis(
        g0std_cell=structure,g0std_ssg=group,msg_parent_space_group_number=3,tol_cfg=Tolerances())
    assert len(result['ordered_magnetic_ops'])==2
    assert result['g0std_ssg'] is group


def test_incomplete_soc_operation_set_does_not_use_standard_axes_in_an_unknown_basis():
    result=build_parent_standard_supercell_domain_coset_analysis(
        parent_space_group_number=47,parent_space_group_symbol='Pmmm',parent_hall_number=227,
        child_basis_in_parent=np.eye(3),child_origin_in_parent=np.zeros(3),
        ordered_magnetic_ops=[(np.eye(3),np.zeros(3),1)],ordered_space_group_number=3,
        basis_setting='child',relation_layer='soc_magnetic',subgroup_time_branch_scope='unit')
    assert result['status']=='not_evaluated_incomplete_soc_axis_constraints'
