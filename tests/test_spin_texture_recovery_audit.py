import importlib
import numpy as np

from findspingroup.spin_splitting import classify_public_spin_texture_config, normalize_operations

fsg=importlib.import_module('findspingroup.find_spin_group')


def pair(spin):
    return {'Q':np.eye(3),'S':np.asarray(spin,float)}


def reference(axis='z'):
    return {'spin_texture_type':'s-wave','order':0,'nullity':1,'spin_rank':1,
            'momentum_space_spin_configuration':'collinear','basis':[f'C1*sigma_{axis}']}


def test_reference_match_does_not_override_full_operation_constraints():
    primary=[pair(np.diag([-1.,1.,-1.]))]
    full=[*primary,pair(np.diag([1.,-1.,-1.]))]
    result=fsg._classify_spin_texture_config_with_reference(
        primary,primary_source='generators',reference=reference('y'),
        calibration_atol_limit=.001,spin_texture_basis_max_order=None,
        fallback_operations=full,fallback_source='full')
    assert result['spin_texture_type']=='forbidden'
    audit=result['calibration']
    assert audit['strict_primary']['spin_texture_type']=='s-wave'
    assert not audit['strict_primary']['constraint_validation']['passed']
    assert audit['strict_full']['spin_texture_type']=='forbidden'
    assert audit['reference']['basis']==['C1*sigma_y']


def test_bounded_recovery_retains_strict_full_reference_and_selected_results():
    operations=[pair(np.diag([-1.,-1.,1.00001]))]
    result=fsg._classify_spin_texture_config_with_reference(
        operations,primary_source='generators',reference=reference(),
        calibration_atol_limit=.001,spin_texture_basis_max_order=None,
        fallback_operations=operations,fallback_source='full')
    assert result['spin_texture_type']=='s-wave'
    audit=result['calibration']
    assert audit['strict_primary']['spin_texture_type']=='forbidden'
    assert audit['strict_full']['spin_texture_type']=='forbidden'
    assert audit['selected']['basis']==result['basis']
    assert result['constraint_validation']['passed']
    assert result['constraint_validation']['raw_max_operation_residual']<=audit['atol']
    assert audit['atol']<=audit['boundary_atol']==.001


def test_recovery_does_not_exceed_the_explicit_bound_to_obtain_a_label():
    operations=[pair(np.diag([-1.,-1.,1.00001]))]
    result=fsg._classify_spin_texture_config_with_reference(
        operations,primary_source='generators',reference=reference(),
        calibration_atol_limit=1e-7,spin_texture_basis_max_order=None,
        fallback_operations=operations,fallback_source='full')
    assert result['spin_texture_type']=='forbidden'
    assert result['calibration']['status']=='reference_mismatch'
    assert all(a['atol']<=1e-7 for a in result['calibration']['attempts'])


def test_display_cleanup_cannot_violate_the_accepted_polynomial_kernel():
    axis=np.array([1e-7,0.,1.]);axis/=np.linalg.norm(axis)
    operations=[pair(2*np.outer(axis,axis)-np.eye(3))]
    result=classify_public_spin_texture_config(
        operations,source='test',validation_operations=operations,rtol=0.,atol=1e-10)
    audit=result['constraint_validation']
    assert audit['passed']
    assert audit['raw_dimension']==audit['rendered_dimension']==1
    assert audit['rendered_max_operation_residual']<=1e-10
    assert audit['presentation_refined']
    assert 'sigma_x' in result['basis'][0]


def test_validation_includes_requested_higher_orders_not_only_the_leading_term():
    result=classify_public_spin_texture_config(
        [],source='test',basis_orders_through=1,include_diagnostics=True,
        validation_operations=[{'Q':-np.eye(3),'S':np.eye(3)}])
    assert not result['constraint_validation']['passed']
    orders=result['constraint_validation']['orders']
    assert orders[0]['passed'] and not orders[1]['passed']


def test_latex_does_not_erase_a_resolved_small_coefficient():
    axis=np.array([1e-9,0.,1.]);axis/=np.linalg.norm(axis)
    operations=[pair(2*np.outer(axis,axis)-np.eye(3))]
    result=classify_public_spin_texture_config(
        operations,source='test',validation_operations=operations,rtol=0.,atol=1e-14)
    assert result['constraint_validation']['passed']
    assert 'sigma_x' in result['basis'][0]
    assert r'10^{-9}' in result['basis_latex'][0]
    assert r'\sigma_{x}' in result['basis_latex'][0]


def test_rounded_pair_keys_do_not_discard_distinct_constraint_matrices():
    operations=[pair(np.diag([-1.,-1.,1.])),pair(np.diag([-1.,-1.,1.+1e-9]))]
    assert len(normalize_operations(operations,key_decimals=8))==2


def test_quasi2d_recovery_keeps_the_same_evidence_without_inventing_a_reference():
    result=fsg._classify_quasi2d_spin_texture_config(
        [{'Q':np.eye(2),'S':np.diag([-1.,-1.,1.00001])}],source='test2d',
        operation_audit={'non_plane_preserving_operation_count':0},in_plane_axes=['b','c'],
        k_names=('ky','kz'),calibration_atol_limit=.001,relax_without_reference=True)
    assert result['spin_texture_type']=='s-wave'
    audit=result['calibration']
    assert audit['reference'] is None
    assert audit['strict_full']['spin_texture_type']=='forbidden'
    assert audit['selected']['basis']==result['basis']
    assert result['constraint_validation']['passed']
    assert result['k_variable_labels']=={'ky':'input reciprocal b*','kz':'input reciprocal c*'}
