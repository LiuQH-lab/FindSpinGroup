import numpy as np
import pytest

from findspingroup.find_spin_group import _format_spin_only_direction
from findspingroup.io.scif_generator import write_scif_spin_only
from findspingroup.utils.matrix_utils import evaluate_numeric_expression
from findspingroup.utils.symbolic_format import format_direction_components


def parse(text):
    return np.array([evaluate_numeric_expression(value) for value in text.split(',')])


@pytest.mark.parametrize('direction', [[3.186411846e-5, 0., -1.],
                                     [1., 1.00009, 0.], [1., 2e-9, .5]])
def test_reusable_direction_preserves_resolved_components(direction):
    np.testing.assert_allclose(parse(_format_spin_only_direction(direction)), direction,
                               rtol=0, atol=4e-15)
    for conf, tag in [('Collinear', 'collinear_direction_xyz'), ('Coplanar', 'coplanar_perp_uvw')]:
        text = write_scif_spin_only(conf, direction)
        line = next(line for line in text.splitlines() if '.'+tag in line)
        actual = parse(line.split("'")[1])
        np.testing.assert_allclose(actual / np.linalg.norm(actual),
                                   np.array(direction) / np.linalg.norm(direction),
                                   rtol=0, atol=8e-15)


@pytest.mark.parametrize('direction,expected', [([.3,.3,0.], '1,1,0'),
                                             ([.2,-.4,0.], '1,-2,0'),
                                             ([0.,0.,-1e-20], '0,0,-1')])
def test_exact_scale_free_directions_still_have_compact_integer_symbols(direction, expected):
    assert format_direction_components(direction, integer_direction=True) == expected


def test_public_direction_normalization_and_shape_are_unchanged():
    direction = [np.sqrt(2)/2, np.sqrt(2)/2, 0.]
    assert _format_spin_only_direction(direction) == 'sqrt(2)/2,sqrt(2)/2,0'
    assert _format_spin_only_direction(None) == ''
    with pytest.raises(ValueError, match='3-vector'):
        format_direction_components(np.eye(3))


def test_small_coordinate_frame_scale_does_not_erase_the_whole_direction():
    direction = np.array([1.2e-20, -2.3e-20, 0.])
    np.testing.assert_allclose(parse(format_direction_components(direction)), direction,
                               rtol=4e-15, atol=0)
