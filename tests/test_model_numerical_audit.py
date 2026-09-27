import json
import numpy as np
import pytest

from findspingroup.core.identify_spin_space_group import (
    identify_spin_space_group_result, _project_ssg_spin_rotations_to_exact_point_group,
)
from findspingroup.core.tolerances import Tolerances
from findspingroup.structure import CrystalCell, SpinSpaceGroupOperation
from findspingroup.find_spin_group import classify_magnetic_phase, find_spin_group_basic_from_data


def test_identifier_retains_existing_selected_model_residuals():
    cell=CrystalCell([[2.,0.,0.],[.2,3.,0.],[.3,.5,4.]],[[0,0,0]],[1.],['Fe'],
                     [[0.,0.,1.]],spin_setting='cartesian')
    result=identify_spin_space_group_result(cell,find_primitive=False)
    audit=result.numerical_audit
    assert audit['spin_frame']=='Cartesian'
    assert audit['physical_action_residual']['max_moment']==0.
    assert audit['physical_budgets']['moment']==cell.tol.moment
    assert audit['spin_representation']['max_spin_matrix_change']==0.
    assert audit['worst_source_site']['element']=='Fe'
    json.dumps(audit,allow_nan=False)


def test_projection_retains_its_matrix_change_without_mutating_the_input():
    normal=np.array([0.,1.,.002]);normal/=np.linalg.norm(normal)
    spins=[np.eye(3),np.diag([-1.,-1.,1.]),np.diag([-1.,1.,1.]),np.eye(3)-2*np.outer(normal,normal)]
    ops=[SpinSpaceGroupOperation(s,np.eye(3),np.zeros(3)) for s in spins]
    original=[op.spin_rotation.copy() for op in ops]
    cell=CrystalCell(np.eye(3),[[0,0,0]],[1.],['Fe'],[[0.,0.,1.]],spin_setting='cartesian')
    audit={}
    projected=_project_ssg_spin_rotations_to_exact_point_group(ops,cell.atoms,Tolerances(),audit=audit)
    assert audit['status']=='projected_and_physically_validated'
    assert 0 < audit['max_spin_matrix_change'] <= audit['spin_matrix_projection_limit']
    assert all(np.linalg.norm(op.spin_rotation@np.array([0.,0.,1.])-np.array([0.,0.,1.])) <= .02
               for op in projected)
    for op,before in zip(ops,original):np.testing.assert_array_equal(op.spin_rotation,before)


@pytest.mark.parametrize('value,zero', [
    (np.nextafter(.02, 0.), False), (.02, False),
    (np.nextafter(.02, np.inf), False), (-.02, False),
])
def test_net_moment_margin_retains_raw_values_with_nonzero_boundary_ties(value, zero):
    payload = classify_magnetic_phase(
        conf='Collinear', full_spin_part_point_group_hm='m',
        full_spin_part_point_group_s='Cs', net_moment=value,
        mpg_identifier=None, is_ss_gp='spin splitting', net_moment_tol=.02,
        magnetic_atom_orbit_analysis={'count': 2})
    assert payload['base_phase'] == ('Compensated FiM' if zero else 'FiM')
    decision = payload['details']['net_moment_decision']
    assert decision['margin'] == .02-abs(value)
    assert decision['ratio'] == abs(value)/.02
    assert decision['at_threshold'] is True
    assert decision['boundary_policy'] == 'nonzero'
    assert payload['details']['net_moment'] == value


def test_zero_net_moment_budget_has_no_division_or_new_zero_policy():
    payload = classify_magnetic_phase(
        conf='Collinear', full_spin_part_point_group_hm='m',
        full_spin_part_point_group_s='Cs', net_moment=0.,
        mpg_identifier=None, is_ss_gp='spin splitting', net_moment_tol=0.)
    assert not payload['details']['zero_net_moment']
    assert payload['details']['net_moment_decision']['ratio'] is None
    json.dumps(payload, allow_nan=False)


def test_basic_output_reuses_the_identifier_audit_without_a_new_top_level_field():
    result = find_spin_group_basic_from_data(
        'audit', [[2.,0.,0.],[.2,3.,0.],[.3,.5,4.]], [[0,0,0]], ['Fe'],
        [1.], [[0,0,1.]], input_spin_setting='cartesian')
    assert 'accepted_group_audit' not in result
    audit = result['magnetic_phase_details']['accepted_group_audit']
    assert audit['setting'] == 'identifier_cell'
    assert np.asarray(audit['lattice_rows']).shape == (3,3)
    assert audit['physical_action_residual']['max_moment'] == 0.


def test_group_diagnostic_maxima_are_independent_of_the_worst_normalized_operation(monkeypatch):
    import importlib
    module = importlib.import_module('findspingroup.core.identify_spin_space_group')
    entries = [
        dict(normalized=.8, max_position=.016, max_moment=.004, max_occupancy=0.,
             worst_atom=0, matched_atom=0),
        dict(normalized=.9, max_position=.001, max_moment=.018, max_occupancy=0.,
             worst_atom=1, matched_atom=1),
    ]
    monkeypatch.setattr(module, '_magnetic_action_residual_details',
                        lambda op, *args, **kwargs: entries[op])
    details = module._ssg_group_residual_details([0,1], [], Tolerances())
    assert details['max_position'] == .016
    assert details['max_moment'] == .018
    assert details['normalized'] == .9
    assert details['worst_op'] == 1
