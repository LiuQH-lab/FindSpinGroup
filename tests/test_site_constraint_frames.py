import numpy as np

from findspingroup.find_spin_group import calculate_freedom_degree
from findspingroup.io.cif_parser import ScifParser
from findspingroup.io.scif_generator import (
    write_scif_atoms, _solver_constraints_to_matrix, _format_spinframe_transform_abc,
    _solver_constraints_to_relative_symmform, _solver_constraints_to_absolute_symmform,
    _solver_matrix_to_symmform,
)
from findspingroup.utils import general_positions_to_matrix


def test_site_dof_and_text_use_the_same_rank_threshold():
    dof, constraint = calculate_freedom_degree(np.array([np.eye(3)+np.diag([.005, 1., 1.])]), tol=.01)
    assert dof == 1
    assert constraint == ["Sx", "0", "0"]


def _site_tags(spin_lengths):
    moment = np.array(spin_lengths)*[1.,1.,0.]
    cell = (np.diag([2.,3.,4.]), np.array([[0.,0.,0.]]), [1], moment[None,:])
    orbit = [{'representative_index':0, 'class_indices':[0]}]
    text = write_scif_atoms(cell,{1:1.},{1:'Fe'},orbit,orbit,
                            symmetry_constraints=[['Sx','Sx','0']],
                            moment_basis_cartesian=np.eye(3),
                            spin_basis_lengths=spin_lengths)
    return ScifParser(source_text='data_case\n'+text).parse()


def test_cartesian_scif_constraint_does_not_use_real_lattice_lengths():
    tags = _site_tags(np.ones(3))
    assert tags['_atom_site_spin_moment.symmform_uvw'] == ['u,u,0']
    assert tags['_atom_site_spin_moment.symmform_rel_uvw'] == ['u,u,0']


def test_oriented_scif_constraint_uses_actual_spin_basis_lengths():
    tags = _site_tags(np.array([2.,3.,4.]))
    assert tags['_atom_site_spin_moment.symmform_uvw'] == ['u,3/2u,0']
    assert tags['_atom_site_spin_moment.symmform_rel_uvw'] == ['u,u,0']


def test_site_solver_transports_a_physical_near_nullspace_back_to_its_frame():
    basis = np.array([[1.,3.,0.],[0.,1.,0.],[0.,0.,1.]])
    axis = np.array([1.,2.,3.])/np.sqrt(14)
    constraint = np.eye(3)-.995*np.outer(axis,axis)
    relative = np.linalg.inv(basis)@constraint@basis
    dof, expressions, audit = calculate_freedom_degree(
        [np.eye(3)+relative],tol=.01,spin_basis=basis,return_audit=True)
    assert dof == 1
    coefficients = _solver_constraints_to_matrix(expressions)
    physical = basis@coefficients
    q, singular, _ = np.linalg.svd(physical,full_matrices=False)
    q = q[:,singular>1e-10]
    np.testing.assert_allclose(q@q.T,np.outer(axis,axis),atol=1e-12,rtol=0)
    assert audit['max_operation_residual'] < .01


def test_scif_fallback_relative_constraint_uses_declared_basis_lengths():
    cell = (np.diag([2.,3.,4.]), np.array([[0.,0.,0.]]), [1], np.array([[2.,3.,0.]]))
    orbit = [{'representative_index':0,'class_indices':[0]}]
    text = write_scif_atoms(cell,{1:1.},{1:'Fe'},orbit,orbit,moment_basis_cartesian=np.eye(3),
                            spin_basis_lengths=[2.,3.,4.])
    tags = ScifParser(source_text='data_case\n'+text).parse()
    assert tags['_atom_site_spin_moment.symmform_uvw'] == ['u,3/2u,0']
    assert tags['_atom_site_spin_moment.symmform_rel_uvw'] == ['u,u,0']


def test_spinframe_tag_honors_machine_readable_precision():
    lattice = np.array([[4.123456789,0.,0.],[.31,5.87654321,0.],[.12,.27,6.012345678]])
    rows = np.linalg.inv(lattice.T).T
    tag = _format_spinframe_transform_abc(rows,coeff_precision=15)
    expression = tag.split("'")[1]
    transforms,_ = general_positions_to_matrix([expression],variables=('a','b','c'))
    recovered = lattice.T@transforms[0][0].T
    np.testing.assert_allclose(recovered,np.eye(3),atol=1e-13,rtol=0)


def test_relative_and_absolute_symmforms_do_not_snap_coefficients_independently():
    constraints=['Sx','0.99993535*Sx','0']
    relative=_solver_constraints_to_relative_symmform(constraints)
    absolute=_solver_constraints_to_absolute_symmform(constraints,np.diag([2.,3.,4.]))
    r=general_positions_to_matrix([relative],variables=('u','v','w'))[0][0][0]
    a=general_positions_to_matrix([absolute],variables=('u','v','w'))[0][0][0]
    expected=np.array([1.,.99993535,0.])
    np.testing.assert_allclose(r[:,0],expected,atol=1e-12,rtol=0)
    physical_r=np.diag([2.,3.,4.])@r
    np.testing.assert_allclose(physical_r@np.linalg.pinv(physical_r),
                               a@np.linalg.pinv(a),atol=1e-12,rtol=0)


def test_symmform_keeps_independent_parameters_when_a_leading_component_is_tiny():
    matrix=np.array([[3.6e-11,1.,0.],[-1.,0.,0.],[0.,-1.61,0.]])
    text=_solver_matrix_to_symmform(matrix)
    recovered=general_positions_to_matrix([text],variables=('u','v','w'))[0][0][0]
    assert np.max(np.abs(recovered)) <= 2.
    assert np.linalg.matrix_rank(recovered,tol=1e-9) == 2
    np.testing.assert_allclose(recovered@np.linalg.pinv(recovered),
                               matrix@np.linalg.pinv(matrix),atol=1e-12,rtol=0)
