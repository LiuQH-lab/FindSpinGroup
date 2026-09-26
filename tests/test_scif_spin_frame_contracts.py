import numpy as np
import pytest

from findspingroup.io import parse_scif_text, parse_structure_file


def _text(frame, operation="-u,v,-w"):
    return f"""data_spin_frame
_cell_length_a 1
_cell_length_b 1
_cell_length_c 1
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_spin.transform_spinframe_P_abc '{frame}'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe .1 .2 .3
loop_
_atom_site_spin_moment.label
_atom_site_spin_moment.axis_u
_atom_site_spin_moment.axis_v
_atom_site_spin_moment.axis_w
Fe1 1 2 3
loop_
_space_group_symop_spin_operation.id
_space_group_symop_spin_operation.xyzt
_space_group_symop_spin_operation.uvw
1 x,y,z,+1 u,v,w
2 x+1/2,y,z,+1 {operation}
"""


def test_absolute_moments_and_relative_operations_use_different_scales():
    parsed, metadata = parse_scif_text(_text("2a,b,c", "1/2v,2u,-w"), return_metadata=True)
    np.testing.assert_allclose(parsed[-1], [[1, 2, 3], [2, 1, -3]], atol=1e-12, rtol=0)
    assert metadata["spin_setting"] == "cartesian"


def test_rotated_spin_frame_returns_canonical_cartesian_moments(tmp_path):
    path = tmp_path / "permuted.scif"
    path.write_text(_text("b,-a,c"))
    parsed, metadata = parse_structure_file(path, return_metadata=True)
    np.testing.assert_allclose(parsed[-1], [[-2, 1, 3], [-2, -1, -3]], atol=1e-12, rtol=0)
    assert metadata["spin_setting"] == "cartesian"


@pytest.mark.parametrize("frame", ["a,b,c", "1a,1b,1c"])
def test_default_lattice_frame_retains_in_lattice_return_contract(frame):
    parsed, metadata = parse_scif_text(_text(frame), return_metadata=True)
    np.testing.assert_allclose(parsed[-1], [[1, 2, 3], [-1, 2, -3]], atol=1e-12, rtol=0)
    assert metadata["spin_setting"] == "in_lattice"


def test_missing_spin_frame_means_the_current_lattice_frame():
    text = _text("a,b,c").replace("_space_group_spin.transform_spinframe_P_abc 'a,b,c'\n", "")
    parsed, metadata = parse_scif_text(text, return_metadata=True)
    assert metadata["spin_setting"] == "in_lattice"
    np.testing.assert_allclose(parsed[-1], [[1, 2, 3], [-1, 2, -3]], atol=1e-12, rtol=0)


@pytest.mark.parametrize("frame", ["a,a,c", "a+1,b,c", "0a,b,c"])
def test_invalid_spin_frame_is_not_silently_treated_as_cartesian(frame):
    with pytest.raises(ValueError, match="spin.frame|spin frame"):
        parse_scif_text(_text(frame))


def test_nondefault_legacy_matrix_requires_an_unambiguous_basis_declaration():
    text = _text("a,b,c").replace("_space_group_spin.transform_spinframe_P_abc 'a,b,c'",
        "_space_group_spin.transform_spinframe_P_matrix '[[0,1,0],[1,0,0],[0,0,1]]'")
    with pytest.raises(ValueError, match="explicit"):
        parse_scif_text(text)
