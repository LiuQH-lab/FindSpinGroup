import numpy as np
import pytest
from pathlib import Path

from findspingroup.io import parse_cif_file, parse_cif_metadata, parse_structure_file
from findspingroup.io.cif_parser import CifParser, convert_string_to_float


def test_cif_parser_preserves_quoted_symmetry_operation_tokens():
    data = CifParser("tests/testset/errorset/yzplane.mcif").parse()

    assert data["_space_group_symop_operation_xyz"] == ["x, y, z"]


def test_cif_parser_packs_loop_values_across_physical_lines():
    data = CifParser(
        source_text="""data_test
loop_
_test.id
_test.value
row1
1
row2 2 row3 3
"""
    ).parse()

    assert data["_test.id"] == ["row1", "row2", "row3"]
    assert data["_test.value"] == ["1", "2", "3"]


def test_cif_parser_preserves_bracketed_loop_values():
    data = CifParser(
        source_text="""data_test
loop_
_parent_propagation_vector.id
_parent_propagation_vector.kxkykz
k1 [0 1/2 -1/2]
"""
    ).parse()

    assert data["_parent_propagation_vector.id"] == ["k1"]
    assert data["_parent_propagation_vector.kxkykz"] == ["[0 1/2 -1/2]"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1e-3", 1e-3),
        ("-2.5E+2", -250.0),
        ("3.14159(12)", 3.14159),
        ("1.25(3)e-2", 0.0125),
        ("'4.5e1'", 45.0),
        ("−0.0318(7)", -0.0318),
        ("5..88848(6)", 5.88848),
        ("-3.11.", -3.11),
    ],
)
def test_convert_string_to_float_supports_cif_number_syntax(value, expected):
    assert convert_string_to_float(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["?", ".", "1/2", "1.0 trailing"])
def test_convert_string_to_float_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError, match="Invalid CIF numeric value"):
        convert_string_to_float(value)


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


def test_parse_structure_file_treats_plain_filename_as_poscar(tmp_path):
    path = tmp_path / "magnetic_input"
    path.write_text(
        "\n".join(
            [
                "plain POSCAR filename",
                "1.0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "Fe",
                "1",
                "Direct",
                "0 0 0",
                "# MAGMOM= 0 0 1",
            ]
        ),
        encoding="utf-8",
    )

    parsed, metadata = parse_structure_file(path, return_metadata=True)
    lattice_factors, positions, elements, occupancies, labels, moments = parsed

    assert metadata["source_format"] == "poscar"
    assert metadata["spin_setting"] == "cartesian"
    assert np.allclose(lattice_factors, np.eye(3))
    assert np.allclose(positions, [[0.0, 0.0, 0.0]])
    assert elements == ["Fe"]
    assert occupancies == [1.0]
    assert labels == ["Fe_1"]
    assert np.allclose(moments, [[0.0, 0.0, 1.0]])


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


def test_parse_cif_file_defaults_missing_atom_site_occupancy_to_one(tmp_path):
    cif_text = """# generated by external mcif source
data_missing_occupancy
_cell_length_a   4
_cell_length_b   5
_cell_length_c   6
_cell_angle_alpha   90
_cell_angle_beta   90
_cell_angle_gamma   90
loop_
 _space_group_symop_operation_xyz
  'x,y,z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Fe1 Fe 0 0 0 1
  Ga1 Ga 0.25 0.25 0.25
loop_
 _atom_site_moment.label
 _atom_site_moment.crystalaxis_x
 _atom_site_moment.crystalaxis_y
 _atom_site_moment.crystalaxis_z
  Fe1 1 0 0
"""
    path = Path(tmp_path) / "missing_occupancy.mcif"
    path.write_text(cif_text, encoding="utf-8")

    _lattice, _positions, elements, occupancies, labels, _moments = parse_cif_file(path)

    assert elements == ["Fe", "Ga"]
    assert labels == ["Fe1", "Ga1"]
    assert occupancies == [1.0, 1.0]
