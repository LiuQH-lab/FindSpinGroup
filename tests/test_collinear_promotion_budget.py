import numpy as np
import pytest

from findspingroup.structure import SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.structure.group import _collinear_axis_action_matches


def group_with_axis(axis, real, basis=None):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    first = np.cross(axis, np.array([0., 1., 0.]))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    spin_only = [np.eye(3), 2*np.outer(axis,axis)-np.eye(3),
                 np.eye(3)-2*np.outer(first,first), np.eye(3)-2*np.outer(second,second)]
    frame = np.eye(3) if basis is None else np.asarray(basis, float)
    inv = np.linalg.inv(frame)
    ops = [SpinSpaceGroupOperation(inv@s@frame, inv@r@frame, np.zeros(3))
           for s in spin_only for r in [np.eye(3), real]]
    return SpinSpaceGroup(ops, tol=.01, real_space_metric=frame.T@frame)


def test_near_parallel_real_rotation_is_not_an_exact_spin_only_rotation():
    # The old 1-cos(theta)<tol admitted an eight-degree tilt at tol=.01.
    angle = .1
    real = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    group = group_with_axis([np.sin(angle),0.,np.cos(angle)], real)
    assert group.collinear_spin_promotion_order == 2
    for spin in group._collinear_spin_only_promotion_rotations():
        assert np.linalg.norm((spin-np.eye(3))@group.collinear_axis) <= .01


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_perpendicular_candidate_budget_does_not_depend_on_lattice_units(scale):
    real = np.diag([1.,-1.,-1.])
    group = group_with_axis([0.,0.,1.], real, scale*np.eye(3))
    assert group._build_collinear_perpendicular_direct_candidates()


def test_perpendicular_candidate_uses_action_error_not_cosine_error():
    # |axis dot n|=.007 < .01, but the promoted 2-fold changes n by .014.
    real = np.diag([1.,-1.,-1.])
    group = group_with_axis([.007,0.,1.], real)
    assert not group._build_collinear_perpendicular_direct_candidates()


@pytest.mark.parametrize('shear', [0., 3., 20.])
@pytest.mark.parametrize('sign', [-1, 1])
def test_axis_action_is_covariant_under_nonorthogonal_frame(shear, sign):
    frame=np.array([[1.,shear,0.],[0.,1.,0.],[0.,0.,1.]])
    rotation=np.diag([1.,-1.,-1.])
    direction=np.array([1.,0.,0.]) if sign==1 else np.array([0.,1.,0.])
    inverse=np.linalg.inv(frame)
    assert _collinear_axis_action_matches(inverse@rotation@frame, inverse@direction,
                                           sign=sign, metric=frame.T@frame, tol=1e-10)


@pytest.mark.parametrize('case', ['0.3_Ca3LiOsO6.mcif','0.200_Mn3Sn.mcif','1.501_Ba2CoO2Cu2S2.mcif'])
def test_scif_magnetic_acc_is_independent_of_declared_spin_frame(case):
    from findspingroup import find_spin_group
    from findspingroup.io.cif_parser import ScifParser

    result=find_spin_group('tests/testset/mcif_241130_no2186/'+case)
    tag='_space_group_spin.fsg_magnetic_arithmetic_crystal_class_symbol'
    for setting in ['ssg_convention','magnetic_primitive','input','database_standard']:
        cart=ScifParser(source_text=result.scif_outputs[setting+'_cartesian']).parse()[tag]
        oriented=ScifParser(source_text=result.scif_outputs[setting+'_oriented']).parse()[tag]
        assert cart==oriented,(setting,cart,oriented)


def test_1501_msg_agrees_with_independent_physical_moment_detection():
    import spglib
    from findspingroup import find_spin_group

    result=find_spin_group('tests/testset/mcif_241130_no2186/1.501_Ba2CoO2Cu2S2.mcif')
    cell=result.input_cell_detail
    for moment_tol in [.005,.02,.05]:
        dataset=spglib.get_magnetic_symmetry_dataset(
            (cell['lattice'],cell['positions'],cell['type_ids'],cell['moments']),
            symprec=.02,mag_symprec=moment_tol)
        assert result.msg_num==dataset.uni_number==7
