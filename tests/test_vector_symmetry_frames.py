import numpy as np
import pytest

from findspingroup.ferroelectric import _vector_axis_basis_from_ops, _axis_relation_payload


def test_vector_axes_use_the_supplied_physical_frame():
    frame=np.array([[1.,3.,0.],[0.,1.,0.],[0.,0.,2.]])
    n=np.array([1.,2.,3.])/np.sqrt(14)
    c=np.eye(3)-.995*np.outer(n,n)
    relative=np.linalg.inv(frame)@(np.eye(3)+c)@frame
    axes=_vector_axis_basis_from_ops([relative],representation_matrix=lambda x:x,
                                    tol=.01,frame=frame)
    assert len(axes)==1
    physical=frame@np.array(axes).T
    projector=physical@np.linalg.pinv(physical)
    np.testing.assert_allclose(projector,np.outer(n,n),atol=1e-10,rtol=0)


def test_unconstrained_vector_keeps_simple_coordinate_axes_in_any_frame():
    frame=np.array([[1.,3.,0.],[0.,1.,0.],[0.,0.,2.]])
    axes=_vector_axis_basis_from_ops([np.eye(3)],representation_matrix=lambda x:x,
                                    tol=.01,frame=frame)
    assert axes==((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def test_vector_axes_are_independent_of_repeated_operations():
    operation=np.eye(3)+np.diag([.0005,1.,1.])
    first=_vector_axis_basis_from_ops([operation],representation_matrix=lambda x:x,tol=.001)
    repeated=_vector_axis_basis_from_ops([operation]*15,representation_matrix=lambda x:x,tol=.001)
    assert first==repeated==((1.,0.,0.),)


@pytest.mark.parametrize('scale',[1e-15,1.,1e15])
def test_direction_does_not_disappear_when_lattice_length_units_change(scale):
    axes=_vector_axis_basis_from_ops([np.diag([-1.,-1.,1.])],
                                    representation_matrix=lambda x:x,tol=1e-8,
                                    frame=scale*np.eye(3))
    assert axes==((0.,0.,1.),)


def test_tight_vector_budget_is_not_lost_in_output_rounding():
    axis=np.array([5.0551e-11,0.,1.]);axis/=np.linalg.norm(axis)
    rotation=2*np.outer(axis,axis)-np.eye(3)
    axes=_vector_axis_basis_from_ops([rotation],representation_matrix=lambda x:x,tol=1e-14)
    vector=np.array(axes[0]);vector/=np.linalg.norm(vector)
    assert np.linalg.norm((rotation-np.eye(3))@vector) < 1e-14


def test_preserved_polar_direction_can_be_a_combination_of_displayed_axes():
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    axes=((1.,0.,1.),(0.,1.,1.),(1.,1.,0.))
    assert _axis_relation_payload(rotation,axes,tol=1e-8)==('P -> P',[])


@pytest.mark.parametrize('scale',[1e-12,1.,1e12])
def test_polar_reversal_is_invariant_to_axis_basis_scaling_and_shear(scale):
    frame=np.array([[1.,3.,0.],[0.,1.,0.],[0.,0.,2.]])
    rotation=np.diag([-1.,-1.,1.])
    inverse=np.linalg.inv(frame)
    axes=tuple(tuple(v) for v in (inverse@(scale*np.array([[1.,0.],[0.,1.],[1.,1.]]))).T)
    assert _axis_relation_payload(inverse@rotation@frame,axes,tol=1e-8,frame=frame)[0]=='P -> -P'


def test_polar_axis_relation_obeys_physical_not_component_error():
    frame=np.diag([1.,100.,1.])
    rotation=np.array([[-1.,0.,0.],[.005,1.,0.],[0.,0.,1.]])
    # Componentwise error .005 hid physical error .5 in the short axis.
    assert _axis_relation_payload(rotation,((1.,0.,0.),),tol=.01,frame=frame)==('P -> other',[])
