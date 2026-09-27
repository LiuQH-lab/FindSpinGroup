"""Classification ties do not change moments, frames, or physical tolerances."""
from pathlib import Path

import numpy as np
import pytest

from findspingroup import find_spin_group, find_spin_group_basic_from_data
from findspingroup.find_spin_group import classify_magnetic_phase, get_magnetic_phase
from findspingroup.io import parse_scif_text


def classify(moment, threshold=.02, **kwargs):
    options = dict(
        conf='Collinear', full_spin_part_point_group_hm='m',
        full_spin_part_point_group_s='Cs', mpg_identifier=None,
        is_ss_gp='spin splitting', magnetic_atom_orbit_analysis={'count': 2},
    )
    options.update(kwargs)
    return classify_magnetic_phase(net_moment=moment, net_moment_tol=threshold, **options)


@pytest.mark.parametrize('threshold', [1e-12, .02, .05, 1., 1e6])
@pytest.mark.parametrize('offset', [-10., -.5, 0., .5, 10.])
def test_relative_boundary_is_unit_covariant_and_does_not_broaden_zero(threshold, offset):
    eta = np.sqrt(np.finfo(float).eps) * threshold
    value = threshold + offset * eta
    result = classify(value, threshold)
    decision = result['details']['net_moment_decision']
    assert result['base_phase'] == ('Compensated FiM' if offset < -1 else 'FiM')
    assert decision['at_threshold'] == (abs(offset) <= 1)
    assert decision['numerical_tolerance'] == eta
    assert decision['numerical_relative_tolerance'] == np.sqrt(np.finfo(float).eps)
    assert decision['threshold'] == threshold
    assert result['details']['net_moment'] == value
    assert decision['margin'] == threshold - value


@pytest.mark.parametrize('value', [.019999999999999844, .019999999963993216,
                                  .020000000000000018, .02000000007893199])
def test_nicro2o4_recorded_frame_readbacks_share_the_same_boundary(value):
    result = classify(value)
    assert result['base_phase'] == 'FiM'
    assert result['details']['net_moment_decision']['at_threshold']


@pytest.mark.parametrize('sign', [-1., 1.])
def test_exact_numerical_band_endpoints_belong_to_the_tie(sign):
    eta = np.sqrt(np.finfo(float).eps)
    result = classify(1. + sign * eta, 1.)
    assert result['base_phase'] == 'FiM'
    assert result['details']['net_moment_decision']['at_threshold']


@pytest.mark.parametrize('threshold', [0., 1e-16, .02])
def test_exact_zero_and_zero_threshold_keep_the_strict_physical_meaning(threshold):
    result = classify(0., threshold)
    assert result['details']['zero_net_moment'] == (threshold > 0)
    assert result['details']['net_moment_decision']['at_threshold'] == (threshold == 0)


def test_boundary_tie_does_not_relabel_afm_or_single_orbit_fm_as_fim():
    afm = classify(.02, full_spin_part_point_group_hm='mmm',
                   full_spin_part_point_group_s='D2h')
    assert afm['base_phase'] == 'AFM'
    fm = classify(.02, magnetic_atom_orbit_analysis={'count': 1})
    assert fm['base_phase'] == 'FM'
    assert get_magnetic_phase('m', 'Cs', np.nextafter(.02, 0), None,
                              magnetic_atom_orbit_analysis={'count': 2}) == 'FiM'


def test_boundary_is_stable_under_physical_rotations_and_summation_order():
    moments = np.zeros((6, 3))
    moments[:4, 0] = .84
    moments[4:, 0] = -1.69
    original = moments.copy()
    rng = np.random.default_rng(28)
    for _ in range(30):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        transformed = (moments @ rotation.T)[rng.permutation(len(moments))]
        value = np.linalg.norm(transformed.sum(axis=0))
        result = classify(value)
        assert result['base_phase'] == 'FiM'
        assert result['details']['net_moment_decision']['at_threshold']
    np.testing.assert_array_equal(moments, original)


def test_nicro2o4_all_scif_modes_preserve_phase_without_changing_index_or_msg():
    source = Path(__file__).parent / 'testset/mcif_241130_no2186/0.892_NiCr2O4.mcif'
    result = find_spin_group(str(source))
    assert result.index == '141.141.1.1.L'
    assert result.magnetic_phase == 'FiM'
    assert len(result.scif_outputs) == 8
    for mode, text in result.scif_outputs.items():
        parsed, metadata = parse_scif_text(text, return_metadata=True)
        lattice, positions, elements, occupancies, _, moments = parsed
        readback = find_spin_group_basic_from_data(
            mode, lattice, positions, elements, occupancies, moments,
            input_spin_setting=metadata['spin_setting'],
        )
        assert readback['index'] == result.index, mode
        assert readback['msg_bns_number'] == result.msg_bns_number, mode
        assert readback['magnetic_phase'] == 'FiM', mode
        assert readback['magnetic_phase_details']['net_moment_decision']['at_threshold'], mode
