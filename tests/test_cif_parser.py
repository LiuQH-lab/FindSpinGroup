import numpy as np
from pathlib import Path

from findspingroup.io import parse_cif_file, parse_cif_metadata, parse_structure_file
from findspingroup.io.cif_parser import CifParser


def test_cif_parser_preserves_quoted_symmetry_operation_tokens():
    data = CifParser("tests/testset/errorset/yzplane.mcif").parse()

    assert data["_space_group_symop_operation_xyz"] == ["x, y, z"]


def test_parse_cif_file_accepts_single_quoted_symmetry_loop_rows():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/errorset/yzplane.mcif"
    )

    assert np.allclose(lattice_factors, [10.0, 10.0, 20.0, 90.0, 90.0, 120.0])
    assert len(positions) == 3
    assert elements == ["Fe", "Fe", "Fe"]
    assert occupancies == [1.0, 1.0, 1.0]
    assert sorted(labels) == ["Fe1", "Fe2", "Fe3"]
    assert any(np.allclose(moment, [0.0, 0.0, 1.0]) for moment in moments)


def test_parse_cif_metadata_reads_parent_space_group_and_cell_strings():
    metadata = parse_cif_metadata("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert metadata["parent_space_group"]["name_H_M_alt"] == "I m -3"
    assert metadata["parent_space_group"]["IT_number"] == 204
    assert metadata["parent_space_group"]["transform_Pp_abc"] == "a,b,c;0,0,0"
    assert metadata["parent_space_group"]["child_transform_Pp_abc"] == "2a,2b,2c;0,0,0"
    assert metadata["cell_parameter_strings"]["_cell_length_a"] == "14.88540"
    assert metadata["cell_parameter_strings"]["_cell_angle_alpha"] == "90.00000"


def test_parse_structure_file_can_return_cif_metadata():
    parsed, metadata = parse_structure_file(
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
        return_metadata=True,
    )

    assert len(parsed) == 6
    assert metadata["parent_space_group"]["IT_number"] == 204
    assert metadata["cell_parameter_strings"]["_cell_length_a"] == "14.88540"


def test_parse_cif_file_accepts_plain_cif_symmetry_equiv_loop_for_p1_magnetic_input(tmp_path):
    cif_text = """# generated using pymatgen
data_V2Te2O
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.04300022
_cell_length_b   4.04300022
_cell_length_c   23.85330009
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   V2Te2O
_chemical_formula_sum   'V2 Te2 O1'
_cell_volume   389.90248418
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  V  V0  1  0.00000000  0.50000000  0.00000000  1
  V  V1  1  0.50000000  0.00000000  0.00000000  1
  Se  Se2  1  0.50000000  0.50000000  0.91868674  1
  Se  Se3  1  0.50000000  0.50000000  0.08131326  1
  O  O4  1  0.00000000  0.00000000  0.00000000  1
loop_
 _atom_site_moment_label
 _atom_site_moment_crystalaxis_x
 _atom_site_moment_crystalaxis_y
 _atom_site_moment_crystalaxis_z
  V0  -5.00000000 0.00000000  0.00000000
  V1  5.00000000 0.00000000  0.00000000
"""
    path = Path(tmp_path) / "v2te2o_p1.mcif"
    path.write_text(cif_text, encoding="utf-8")

    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(path)

    assert np.allclose(lattice_factors, [4.04300022, 4.04300022, 23.85330009, 90.0, 90.0, 90.0])
    assert len(positions) == 5
    assert elements == ["V", "V", "Se", "Se", "O"]
    assert occupancies == [1.0] * 5
    assert labels == ["V0", "V1", "Se2", "Se3", "O4"]
    assert any(np.allclose(moment, [-5.0, 0.0, 0.0]) for moment in moments)
    assert any(np.allclose(moment, [5.0, 0.0, 0.0]) for moment in moments)
