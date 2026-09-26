"""Expansion compares physical positions and physical moment vectors separately."""
import numpy as np
import pytest

from findspingroup.io import parse_cif_file, parse_scif_text, parse_structure_file


def _header(a=2, b=3, c=4):
    return f"""data_contract
_cell_length_a {a}
_cell_length_b {b}
_cell_length_c {c}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"""


def _scif(frame, sites, moments, *, a=2, b=3, c=4, operations=None):
    if operations is None:
        operations = ["1 x,y,z,+1 u,v,w"]
    return (_header(a, b, c) + sites + "\n" +
            f"_space_group_spin.transform_spinframe_P_abc '{frame}'\n" +
            "loop_\n_atom_site_spin_moment.label\n_atom_site_spin_moment.axis_u\n"
            "_atom_site_spin_moment.axis_v\n_atom_site_spin_moment.axis_w\n" + moments + "\n" +
            "loop_\n_space_group_symop_spin_operation.id\n_space_group_symop_spin_operation.xyzt\n"
            "_space_group_symop_spin_operation.uvw\n" + "\n".join(operations) + "\n")


def test_general_scif_frame_is_not_mistaken_for_world_cartesian(tmp_path):
    text = _scif("b,a,c", "Fe1 Fe .1 .2 .3", "Fe1 1 2 3",
                 operations=["1 x,y,z,+1 u,v,w", "2 x+1/2,y,z,+1 -u,v,-w"])
    path = tmp_path / "frame.scif"
    path.write_text(text)
    parsed, metadata = parse_structure_file(path, return_metadata=True)
    assert metadata["spin_setting"] == "cartesian"
    expected = [[2., 1, 3], [2., -1, -3]]
    for actual, target in zip(parsed[-1], expected):
        np.testing.assert_allclose(actual, target, atol=1e-12, rtol=0)


def test_scif_nonorthogonal_frame_moment_consistency_uses_physical_norm():
    text = _scif("a,a+b,c", "Fe1 Fe .1 .2 .3\nFe2 Fe .1 .2 .3",
                 "Fe1 1 1 0\nFe2 1.015 1.015 0", a=1, b=1, c=1)
    with pytest.raises(ValueError, match="inconsistent moments"):
        parse_scif_text(text, atol=.02)


@pytest.mark.parametrize("a, expected_count", [(1., 1), (20., 2)])
def test_scif_position_budget_has_length_units(a, expected_count):
    text = _scif("a,b,c", "Fe1 Fe .1 .2 .3\nFe2 Fe .103 .2 .3",
                 "Fe1 0 0 1\nFe2 0 0 1", a=a)
    parsed = parse_scif_text(text, position_atol=.02)
    assert len(parsed[1]) == expected_count


def _mcif(sites, moments, operations="x,y,z,+1", centering="x,y,z,+1"):
    return (_header() + sites + "\nloop_\n_atom_site_moment.label\n"
            "_atom_site_moment.crystalaxis_x\n_atom_site_moment.crystalaxis_y\n"
            "_atom_site_moment.crystalaxis_z\n" + moments +
            "\nloop_\n_space_group_symop_magn_operation.xyz\n" + operations +
            "\nloop_\n_space_group_symop_magn_centering.xyz\n" + centering + "\n")


def test_mcif_rejects_conflicting_explicit_moments_at_same_site(tmp_path):
    path = tmp_path / "inconsistent.mcif"
    path.write_text(_mcif("Fe1 Fe .1 .2 .3", "Fe1 0 0 1", "x,y,z,+1\nx,y,z,-1"))
    with pytest.raises(ValueError, match="inconsistent moments"):
        parse_cif_file(path, atol=.02)


def test_missing_moment_is_not_an_explicit_zero_constraint(tmp_path):
    path = tmp_path / "unspecified.mcif"
    path.write_text(_mcif("Fe1 Fe .1 .2 .3\nFe2 Fe .1 .2 .3", "Fe1 0 0 1"))
    parsed = parse_cif_file(path, atol=.02)
    assert len(parsed[1]) == 1
    np.testing.assert_allclose(parsed[-1][0], [0, 0, 1], atol=1e-12, rtol=0)


def test_mcif_affine_composition_rotates_centering_translation(tmp_path):
    path = tmp_path / "composition.mcif"
    path.write_text(_mcif("Fe1 Fe .2 .1 .3", "Fe1 0 0 1", "-x,-y,z,+1", "x+1/3,y,z,+1"))
    parsed = parse_cif_file(path)
    np.testing.assert_allclose(parsed[1][0], [7/15, .9, .3], atol=1e-12, rtol=0)


def test_mcif_centering_rotation_also_acts_on_axial_moment(tmp_path):
    path = tmp_path / "composition.mcif"
    path.write_text(_mcif("Fe1 Fe .2 .1 .3", "Fe1 1 2 3", "-x,-y,z,+1", "x,-y,-z,-1"))
    parsed = parse_cif_file(path)
    np.testing.assert_allclose(parsed[-1][0], [1, -2, 3], atol=1e-12, rtol=0)


@pytest.mark.parametrize("frame", ["a,b,c", "2a,3b,4c"])
def test_explicit_zero_is_checked_but_missing_scif_moment_is_not(frame):
    sites = "Fe1 Fe .1 .2 .3\nFe2 Fe .1 .2 .3"
    parsed = parse_scif_text(_scif(frame, sites, "Fe2 0 0 1"))
    assert len(parsed[1]) == 1
    np.testing.assert_allclose(parsed[-1][0], [0, 0, 1], atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="inconsistent moments"):
        parse_scif_text(_scif(frame, sites, "Fe1 0 0 0\nFe2 0 0 1"))


def test_parser_moment_budget_is_absolute_without_relative_or_floor_tolerance():
    sites = "Fe1 Fe .1 .2 .3\nFe2 Fe .1 .2 .3"
    with pytest.raises(ValueError, match="inconsistent moments"):
        parse_scif_text(_scif("a,b,c", sites, "Fe1 0 0 10000\nFe2 0 0 10000.03"), atol=.02)
    with pytest.raises(ValueError, match="inconsistent moments"):
        parse_scif_text(_scif("a,b,c", sites, "Fe1 0 0 0\nFe2 0 0 .0000001"), atol=1e-9)


def test_parser_rejects_affine_spin_shift():
    text = _scif("a,b,c", "Fe1 Fe .1 .2 .3", "Fe1 0 0 1",
                 operations=["1 x,y,z,+1 u+.00001,v,w"])
    with pytest.raises(ValueError, match="affine spin shifts"):
        parse_scif_text(text)


def test_periodic_physical_position_comparison_crosses_unit_boundary():
    text = _scif("a,b,c", "Fe1 Fe .999 .2 .3\nFe2 Fe .001 .2 .3",
                 "Fe1 0 0 1\nFe2 0 0 1", a=2)
    assert len(parse_scif_text(text, position_atol=.005)[1]) == 1
    assert len(parse_scif_text(text, position_atol=.003)[1]) == 2


def test_occupancy_budget_does_not_use_relative_tolerance():
    from findspingroup.io._expansion import ExpansionSites
    sites = ExpansionSites(np.eye(3), np.eye(3), position_atol=.02, moment_atol=.02,
                           occupancy_atol=1e-6, format_name="SCIF")
    for occupancy in [1., .999995]:
        sites.add([0, 0, 0], "Fe", occupancy, "Fe", [0, 0, 1], moment_known=True)
    assert len(sites.positions) == 2


def test_nontransitive_position_matches_are_diagnosed():
    text = _scif("a,b,c", "Fe1 Fe .1 .2 .3\nFe2 Fe .13 .2 .3\nFe3 Fe .115 .2 .3",
                 "Fe1 0 0 1\nFe2 0 0 1\nFe3 0 0 1", a=1)
    with pytest.raises(ValueError, match="ambiguous site matches"):
        parse_scif_text(text, position_atol=.02)
