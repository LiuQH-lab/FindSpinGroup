import importlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from findspingroup import find_spin_group, find_spin_group_from_data
from findspingroup.find_spin_group import (
    SCIF_CELL_MODE_G0STD_ORIENTED,
    SCIF_CELL_MODE_INPUT,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
)
from findspingroup.core import Molecule, PointGroupAnalyzer
from findspingroup.core.identify_spin_space_group import dedup_moments_with_tol
from findspingroup.io import parse_cif_file, parse_scif_file, parse_scif_metadata, parse_scif_text
from findspingroup.structure.cell import (
    CrystalCell,
    are_positions_equivalent,
    calculate_lattice_params,
    calculate_vector_coordinates_from_latticefactors,
)
from findspingroup.structure import SpinSpaceGroup
from findspingroup.utils import general_positions_to_matrix
from findspingroup.utils.matrix_utils import normalize_vector_to_zero


def _roundtrip_index_from_scif_data(scif_path: Path):
    lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_file(scif_path)
    return find_spin_group_from_data(
        str(scif_path),
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )


def _roundtrip_index_from_scif_text(scif_text: str, source_name: str):
    scif_path = Path("/tmp") / source_name
    scif_path.write_text(scif_text, encoding="utf-8")
    return _roundtrip_index_from_scif_data(scif_path)


def _parse_pp_transform(pp_string: str):
    # The emitted `Pp` string is interpreted as a basis change:
    #   x_current = P x_target + p
    # Convert it back into the direct transform expected by `.transform(...)`:
    #   x_target = P^{-1} x_current - P^{-1} p
    expr, translation = pp_string.split(";")
    matrices, _ = general_positions_to_matrix([f"{expr},+1"], variables=("a", "b", "c"))
    basis_rows, _ = matrices[0]
    basis_change = np.asarray(basis_rows, dtype=float)
    origin_current = np.array([float(Fraction(token)) for token in translation.split(",")], dtype=float)
    direct_matrix = np.linalg.inv(basis_change)
    direct_shift = normalize_vector_to_zero(-direct_matrix @ origin_current, atol=1e-10)
    return direct_matrix, direct_shift


def _serialize_spatial_ops(ops):
    return {
        (
            tuple(np.round(np.asarray(op[1], dtype=float), 6).flatten()),
            tuple(np.round(np.mod(np.asarray(op[2], dtype=float), 1.0), 6).flatten()),
        )
        for op in ops
    }


def _invert_setting_transform(transform: np.ndarray, shift: np.ndarray):
    transform_inv = np.linalg.inv(transform)
    shift_inv = np.mod(-transform_inv @ shift, 1.0)
    shift_inv[np.isclose(shift_inv, 1.0, atol=1e-8)] = 0.0
    shift_inv[np.isclose(shift_inv, 0.0, atol=1e-8)] = 0.0
    return transform_inv, shift_inv


def _actual_basis_spin_transform(cell: CrystalCell) -> np.ndarray:
    actual_basis = np.array(
        [vector / np.linalg.norm(vector) for vector in np.asarray(cell.lattice_matrix, dtype=float)],
        dtype=float,
    ).T
    return np.linalg.inv(actual_basis)


def _canonical_basis_spin_transform(cell: CrystalCell) -> np.ndarray:
    canonical_basis = calculate_vector_coordinates_from_latticefactors(
        1,
        1,
        1,
        *np.asarray(cell.lattice_factors, dtype=float)[3:],
    )
    return np.linalg.inv(canonical_basis)


def _roundtrip_index_with_spin_transform_strategy(
    source_path: str,
    strategy,
    strategy_name: str,
):
    find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")
    original_strategy = find_spin_group_module._spin_transform_to_in_lattice
    try:
        find_spin_group_module._spin_transform_to_in_lattice = strategy
        original = find_spin_group(source_path)
        roundtrip = _roundtrip_index_from_scif_text(
            original.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED),
            f"{Path(source_path).stem}_{strategy_name}.scif",
        )
        return original, roundtrip
    finally:
        find_spin_group_module._spin_transform_to_in_lattice = original_strategy


def test_parse_scif_file_roundtrips_generated_mnte_scif(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    scif_path = Path(tmp_path) / "mnte.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_file(scif_path)

    assert np.allclose(lattice_factors, [4.148, 4.147, 6.71, 90.0, 90.0, 120.0], atol=1e-3)
    assert len(positions) == 4
    assert len(elements) == 4
    assert len(occupancies) == 4
    assert len(labels) == 4
    assert len(moments) == 4
    assert sorted(elements) == ["Mn", "Mn", "Te", "Te"]
    assert any(np.linalg.norm(moment) > 0 for moment in moments)


def test_find_spin_group_accepts_generated_scif_input(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    scif_path = Path(tmp_path) / "mnte.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    roundtrip = find_spin_group(str(scif_path))

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf
    assert roundtrip.gspg_symbol_linear == original.gspg_symbol_linear


def test_parse_scif_data_roundtrips_back_into_findspingroup_mainline(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    scif_path = Path(tmp_path) / "mnte.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    roundtrip = _roundtrip_index_from_scif_data(scif_path)

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_find_spin_group_exposes_mainline_scif_output():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    assert "loop_\n_atom_type_symbol\nMn\n" in result.scif
    assert "_space_group_spin.collinear_direction_xyz" in result.scif
    assert "_space_group_spin.collinear_direction_xyz '1,1,0'" in result.scif
    assert f'_space_group_spin.number_Chen  "{result.index}"' in result.scif
    assert "_space_group_spin.name_Chen" in result.scif
    assert metadata["space_group_spin"]["spin_space_group_name_chen"] is not None
    assert "_space_group_spin.fsg_spin_space_group_name_linear" not in result.scif
    assert (
        f'_space_group_spin.fsg_oriented_spin_space_group_name_linear     "{result.convention_ssg_international_linear}"'
        in result.scif
    )
    assert (
        f'_space_group_spin.fsg_oriented_spin_space_group_name_latex     "{result.convention_ssg_international_latex}"'
        in result.scif
    )
    assert "_space_group_spin.rotation_axis  ." in result.scif
    assert "_space_group_spin.rotation_angle ." in result.scif
    assert (
        f"_space_group_spin.fsg_G0_number  {int(result.index.split('.')[0])}"
        in result.scif
    )
    assert sorted(result.scif_cell_modes) == sorted(
        [
            SCIF_CELL_MODE_G0STD_ORIENTED,
            SCIF_CELL_MODE_INPUT,
            SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
        ]
    )
    assert result.to_scif() == result.scif
    assert result.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED) == result.scif


def test_scif_atom_type_loop_lists_all_emitted_species_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    atom_type_block = result.scif.split("loop_\n_atom_type_symbol\n", 1)[1].split("\n\n", 1)[0]
    assert "\nFe\n" in f"\n{atom_type_block}\n"
    assert "\nCa\n" in f"\n{atom_type_block}\n"
    assert "\nO\n" in f"\n{atom_type_block}\n"
    assert "\nTi\n" in f"\n{atom_type_block}\n"


def test_parse_scif_data_roundtrips_back_into_findspingroup_all_cell_modes_mnte():
    original = find_spin_group("examples/0.800_MnTe.mcif")

    for cell_mode in [
        SCIF_CELL_MODE_INPUT,
        SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
        SCIF_CELL_MODE_G0STD_ORIENTED,
    ]:
        roundtrip = _roundtrip_index_from_scif_text(
            original.to_scif(cell_mode=cell_mode),
            f"mnte_{cell_mode}.scif",
        )
        assert roundtrip.index == original.index
        assert roundtrip.conf == original.conf


def test_parse_scif_data_roundtrips_back_into_findspingroup_all_cell_modes_324():
    original = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    for cell_mode in [
        SCIF_CELL_MODE_INPUT,
        SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
        SCIF_CELL_MODE_G0STD_ORIENTED,
    ]:
        roundtrip = _roundtrip_index_from_scif_text(
            original.to_scif(cell_mode=cell_mode),
            f"ca_fe_ti_324_{cell_mode}.scif",
        )
        assert roundtrip.index == original.index
        assert roundtrip.conf == original.conf


def test_input_mode_scif_preserves_source_parent_tags_and_cell_strings_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    input_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_INPUT)

    assert '_parent_space_group.name_H-M_alt  "I m -3"' in input_scif
    assert "_parent_space_group.IT_number  204" in input_scif
    assert '_parent_space_group.transform_Pp_abc  "a,b,c;0,0,0"' in input_scif
    assert '_parent_space_group.child_transform_Pp_abc  "2a,2b,2c;0,0,0"' in input_scif
    assert any(
        line.split() == ["_cell_length_a", "14.88540"]
        for line in input_scif.splitlines()
    )
    assert any(
        line.split() == ["_cell_angle_alpha", "90.00000"]
        for line in input_scif.splitlines()
    )

    metadata = parse_scif_metadata(source_text=input_scif)
    assert metadata["parent_space_group"]["name_H_M_alt"] == "I m -3"
    assert metadata["parent_space_group"]["IT_number"] == 204.0
    assert metadata["parent_space_group"]["transform_Pp_abc"] == "a,b,c;0,0,0"
    assert metadata["parent_space_group"]["child_transform_Pp_abc"] == "2a,2b,2c;0,0,0"
    assert metadata["cell_parameter_strings"]["_cell_length_a"] == "14.88540"
    assert metadata["space_group_spin"]["parent_space_group_status"] is None
    assert metadata["space_group_spin"]["parent_space_group_matches_input"] is None


def test_non_input_scif_uses_six_decimal_computed_cell_constants_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    g0std_scif = result.to_scif(
        cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED,
    )

    assert any(
        line.split() == ["_cell_length_a", "21.051135"]
        for line in g0std_scif.splitlines()
    )
    assert any(
        line.split() == ["_cell_angle_gamma", "120.000000"]
        for line in g0std_scif.splitlines()
    )
    assert "_cell_length_a       14.88540" not in g0std_scif
    assert '_parent_space_group.name_H-M_alt  "I m -3"' in g0std_scif
    assert "_parent_space_group.IT_number  204" in g0std_scif
    assert "_parent_space_group.transform_Pp_abc" not in g0std_scif
    assert (
        '_parent_space_group.child_transform_Pp_abc  "-2a+2c,2a+2b,-1/2a+1/2b-1/2c;0,0,0"'
        in g0std_scif
    )

    metadata = parse_scif_metadata(source_text=g0std_scif)
    assert metadata["parent_space_group"]["name_H_M_alt"] == "I m -3"
    assert metadata["parent_space_group"]["IT_number"] == 204.0
    assert metadata["parent_space_group"]["transform_Pp_abc"] is None
    assert metadata["parent_space_group"]["child_transform_Pp_abc"] == "-2a+2c,2a+2b,-1/2a+1/2b-1/2c;0,0,0"
    assert metadata["space_group_spin"]["parent_space_group_status"] is None
    assert metadata["space_group_spin"]["parent_space_group_matches_input"] is None
    assert metadata["space_group_spin"]["transform_to_input_Pp"] == (
        "2/3a-1/3b-1/3c,1/3a-2/3b+1/3c,-4/3a-4/3b-4/3c;0,0,0"
    )


def test_parse_scif_data_roundtrips_back_into_findspingroup_all_cell_modes_0396():
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.396_MnPtGa.mcif")

    for cell_mode in [
        SCIF_CELL_MODE_INPUT,
        SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
        SCIF_CELL_MODE_G0STD_ORIENTED,
    ]:
        roundtrip = _roundtrip_index_from_scif_text(
            original.to_scif(cell_mode=cell_mode),
            f"mnptga_0396_{cell_mode}.scif",
        )
        assert roundtrip.index == original.index
        assert roundtrip.conf == original.conf


def test_parse_scif_error_message_points_to_find_spin_group_parser_atol():
    path = Path(
        "tests/error_info/run_v0.13.3_20260316_154737_legacy/scif/legacy/0.37_U3Al2Si3.legacy.scif"
    )

    with pytest.raises(ValueError) as exc_info:
        parse_scif_file(path, atol=0.01)

    message = str(exc_info.value)
    assert "find_spin_group(..., parser_atol=...)" in message
    assert "parse_scif_file(..., atol=...)" in message


def test_parse_scif_data_roundtrips_back_into_findspingroup_037_with_relaxed_parser_atol():
    roundtrip = find_spin_group(
        "tests/error_info/run_v0.13.3_20260315_232200_legacy/scif/legacy/0.37_U3Al2Si3.legacy.scif",
        parser_atol=0.02,
    )

    assert roundtrip.index == "79.5.1.2.P2"
    assert roundtrip.conf == "Coplanar"


def test_point_group_symmetry_operations_do_not_overexpand_037_roundtrip():
    path = Path(
        "tests/error_info/run_v0.13.3_20260315_232200_legacy/scif/legacy/0.37_U3Al2Si3.legacy.scif"
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_file(path, atol=0.02)
    parsed_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    )
    primitive_cell, _ = parsed_cell.get_primitive_structure(magnetic=True)

    primitive_moments = np.asarray(primitive_cell.moments, dtype=float)
    primitive_types = np.asarray(primitive_cell.atom_types)
    nonzero = np.linalg.norm(primitive_moments, axis=1) > 2e-5
    unique_types, unique_moments = dedup_moments_with_tol(
        primitive_types[nonzero],
        primitive_moments[nonzero],
        tol=0.02,
    )

    pg = PointGroupAnalyzer(
        Molecule(unique_types.copy(), unique_moments.copy()),
        tolerance=0.02,
        eigen_tolerance=2e-5,
    )

    assert str(pg.get_pointgroup()) == "C2v"
    assert len(pg.get_pointgroup()) == 4
    assert len(pg.get_symmetry_operations()) == 4


def test_parse_scif_data_roundtrips_back_into_findspingroup_0506():
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.506_Cs2Cu3SnF12.mcif")

    roundtrip = _roundtrip_index_from_scif_text(
        original.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED),
        "cs2cu3snf12_0506_g0std_oriented.scif",
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_scif_data_roundtrips_back_into_findspingroup_0876():
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.876_La2ZnIrO6.mcif")

    roundtrip = _roundtrip_index_from_scif_text(
        original.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED),
        "la2zniro6_0876_g0std_oriented.scif",
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_scif_data_roundtrips_back_into_findspingroup_0427():
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.427_Sm2Ti2O7.mcif")

    roundtrip = _roundtrip_index_from_scif_text(
        original.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED),
        "sm2ti2o7_0427_g0std_oriented.scif",
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_scif_data_roundtrips_back_into_findspingroup_0265():
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.265_Mn3(Co0.61Mn0.39)N.mcif")

    roundtrip = _roundtrip_index_from_scif_text(
        original.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED),
        "mn3_co061mn039_n_0265_g0std_oriented.scif",
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_actual_basis_export_strategy_keeps_0427_and_other_sensitive_cases():
    for path in [
        "tests/testset/mcif_241130_no2186/0.427_Sm2Ti2O7.mcif",
        "tests/testset/mcif_241130_no2186/0.265_Mn3(Co0.61Mn0.39)N.mcif",
        "tests/testset/mcif_241130_no2186/0.876_La2ZnIrO6.mcif",
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
    ]:
        original, roundtrip = _roundtrip_index_with_spin_transform_strategy(
            path,
            _actual_basis_spin_transform,
            "actual_basis",
        )
        assert roundtrip.index == original.index
        assert roundtrip.conf == original.conf


def test_ops_true_abc_reference_files_identify_as_324():
    for path in [
        "scif_info/3.24_CaFe3Ti4O12_fsg_v0.13.3_ops_true_abc_symbolic.scif",
        "scif_info/3.24_CaFe3Ti4O12_fsg_v0.13.3_ops_true_abc_fullfloat.scif",
    ]:
        roundtrip = find_spin_group(path)
        assert roundtrip.index == "148.2.4.1"
        assert roundtrip.conf == "Noncoplanar"


def test_old_324_legacy_reference_is_incompatible_with_true_abc_parser_contract():
    with pytest.raises(ValueError):
        find_spin_group("scif_info/3.24_CaFe3Ti4O12_fsg_v0.13.2_legacy.scif")


def test_pure_actual_basis_scif_write_and_read_split_is_explicit_for_0427():
    find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")
    original_strategy = find_spin_group_module._spin_transform_to_in_lattice
    try:
        find_spin_group_module._spin_transform_to_in_lattice = _actual_basis_spin_transform
        result = find_spin_group("tests/testset/mcif_241130_no2186/0.427_Sm2Ti2O7.mcif")
        scif_text = result.to_scif(
            cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED,
        )
        lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_text(scif_text)
    finally:
        find_spin_group_module._spin_transform_to_in_lattice = original_strategy

    parsed_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    )
    actual_basis = np.array(
        [
            vector / np.linalg.norm(vector)
            for vector in np.asarray(result.g0_standard_cell["lattice"], dtype=float)
        ],
        dtype=float,
    ).T
    canonical_basis = calculate_vector_coordinates_from_latticefactors(
        1,
        1,
        1,
        *np.asarray(lattice_factors, dtype=float)[3:],
    )

    idx = next(i for i, moment in enumerate(moments) if np.linalg.norm(moment) > 1e-6)
    coeff = np.asarray(moments[idx], dtype=float)
    actual_cart = actual_basis @ coeff
    canonical_cart = canonical_basis @ coeff
    internal_cart = np.asarray(parsed_cell.moments_cartesian[idx], dtype=float)

    source_matches = [
        source_index
        for source_index, (source_position, source_element) in enumerate(
            zip(result.g0_standard_cell["positions"], result.g0_standard_cell["elements"])
        )
        if are_positions_equivalent(positions[idx], source_position)
        and source_element == elements[idx]
    ]
    assert source_matches
    source_idx = source_matches[0]

    assert np.allclose(
        actual_cart,
        np.asarray(result.g0_standard_cell["moments"][source_idx], dtype=float),
        atol=1e-6,
    )
    assert np.allclose(internal_cart, canonical_cart, atol=1e-9)
    assert not np.allclose(internal_cart, actual_cart, atol=1e-6)


def test_parse_scif_data_roundtrips_back_into_findspingroup_crse(tmp_path):
    original = find_spin_group("examples/2.35_CrSe.mcif")
    scif_path = Path(tmp_path) / "crse.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    roundtrip = _roundtrip_index_from_scif_data(scif_path)

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_general_positions_to_matrix_parses_decimal_uvw_coefficients():
    expr = (
        "-0.333333u-0.666667v+0.544331w,"
        "-0.666667u-0.333333v-0.544331w,"
        "0.816497u-0.816497v-0.333333w"
    )

    matrices, time_reversal = general_positions_to_matrix([expr], variables=("u", "v", "w"))
    matrix, shift = matrices[0]

    assert time_reversal == [1]
    assert np.allclose(
        matrix,
        [
            [-0.333333, -0.666667, 0.544331],
            [-0.666667, -0.333333, -0.544331],
            [0.816497, -0.816497, -0.333333],
        ],
        atol=1e-6,
    )
    assert np.allclose(shift, [0.0, 0.0, 0.0], atol=1e-9)


def test_general_positions_to_matrix_parses_fraction_and_sqrt_uvw_coefficients():
    expr = "sqrt(6)/3u-2*sqrt(6)/9v-1/3w,2/3u-v-2*sqrt(6)/9w,-sqrt(6)/3u-1/3w"

    matrices, time_reversal = general_positions_to_matrix([expr], variables=("u", "v", "w"))
    matrix, shift = matrices[0]

    assert time_reversal == [1]
    assert np.allclose(
        matrix,
        [
            [np.sqrt(6) / 3, -2 * np.sqrt(6) / 9, -1 / 3],
            [2 / 3, -1, -2 * np.sqrt(6) / 9],
            [-np.sqrt(6) / 3, 0, -1 / 3],
        ],
        atol=1e-9,
    )
    assert np.allclose(shift, [0.0, 0.0, 0.0], atol=1e-9)


def test_parse_scif_data_roundtrips_back_into_findspingroup_324(tmp_path):
    original = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    scif_path = Path(tmp_path) / "ca_fe_ti_324.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    roundtrip = _roundtrip_index_from_scif_data(scif_path)

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_dedup_moments_with_tol_keeps_distinct_atom_types():
    atom_types = np.array([2, 1, 1, 2])
    moments = np.array(
        [
            [0.0, -0.48, 0.0],
            [0.0, -0.46, 0.0],
            [0.0, 0.46, 0.0],
            [0.0, 0.48, 0.0],
        ],
        dtype=float,
    )

    unique_types, unique_moments = dedup_moments_with_tol(atom_types, moments, tol=0.02)

    got = sorted(
        (int(t), round(float(m[1]), 2))
        for t, m in zip(unique_types, unique_moments)
    )
    assert got == [(1, -0.46), (1, 0.46), (2, -0.48), (2, 0.48)]


def test_parse_scif_data_roundtrips_back_into_findspingroup_0429(tmp_path):
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.429_CaCr0.86Fe3.14As3.mcif")
    scif_path = Path(tmp_path) / "ca_cr_fe_0429.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    roundtrip = _roundtrip_index_from_scif_data(scif_path)

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_g0std_cell_in_lattice_roundtrips_back_into_findspingroup_crse():
    original = find_spin_group("examples/2.35_CrSe.mcif")
    g0_standard = original.g0_standard_cell

    lattice = np.asarray(g0_standard["lattice"], dtype=float)
    normed_lattice = lattice / np.linalg.norm(lattice, axis=1)[:, None]

    # `g0_standard_cell` stores moments in cartesian form. Rebuilding the
    # in-lattice spin frame should preserve the identified index.
    g0std_cell_in_lattice = CrystalCell(
        calculate_lattice_params(lattice),
        g0_standard["positions"],
        g0_standard["occupancies"],
        g0_standard["elements"],
        g0_standard["moments"],
        spin_setting="cartesian",
    ).transform_spin(np.linalg.inv(normed_lattice.T), "in_lattice")

    roundtrip = find_spin_group_from_data(
        "g0std_cell_in_lattice",
        g0std_cell_in_lattice.lattice_factors,
        g0std_cell_in_lattice.positions,
        g0std_cell_in_lattice.elements,
        g0std_cell_in_lattice.occupancies,
        g0std_cell_in_lattice.moments,
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_scif_file_can_return_structured_metadata(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    scif_path = Path(tmp_path) / "mnte.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    parsed, metadata = parse_scif_file(scif_path, return_metadata=True)

    assert len(parsed) == 6
    assert metadata["space_group_spin"]["source_tags"]["collinear_direction"] == (
        "_space_group_spin.collinear_direction_xyz"
    )
    assert metadata["space_group_spin"]["collinear_direction"]["numeric_components"] is not None
    assert metadata["space_group_spin"]["G0_number"] == 194.0
    assert metadata["space_group_spin"]["L0_number"] == 164.0
    assert metadata["space_group_spin"]["spin_space_point_group_name"] == "∞/mm"
    assert metadata["space_group_spin"]["spin_part_point_group"] == "∞/mm"
    assert metadata["space_group_spin"]["repo_local_extensions"]["G0_number"] == 194.0
    assert metadata["space_group_spin"]["repo_local_extensions"]["spin_space_point_group_name"] == "∞/mm"
    assert metadata["space_group_spin"]["repo_local_extensions"]["spin_part_point_group"] == "∞/mm"
    assert metadata["space_group_symop_spin_operation"]["xyzt"] is not None
    assert metadata["atom_site_spin_moment"]["label"] is not None


def test_parse_scif_metadata_reads_project_and_spincif_style_fields(tmp_path):
    scif_text = """#\\#CIF_2.0
data_test
_space_group_spin.collinear_direction "0,0,1"
_space_group_spin.coplanar_perp_uvw .
_space_group_spin.spin_space_group_name_Chen "P(3)63/(1)m(2)m(2)c.P"
_space_group_spin.fsg_spin_space_group_name_linear "P 1|m 3^1_001|-3 2_100|m m_001|1"
_space_group_spin.fsg_spin_space_group_name_latex "P latex"
_space_group_spin.spin_space_group_number_Chen "194.147.1.1.P3"
_space_group_spin.rotation_angle 60
_space_group_spin.rotation_axis_xyz "0,0,1"
_space_group_spin.rotation_axis_cartn [ 0 0 1 ]
_space_group_spin.transform_spinframe_P_abc 'a,b,c'
"""
    scif_path = Path(tmp_path) / "metadata_only.scif"
    scif_path.write_text(scif_text, encoding="utf-8")

    metadata = parse_scif_metadata(scif_path)

    assert metadata["space_group_spin"]["collinear_direction"]["numeric_components"] == [0.0, 0.0, 1.0]
    assert metadata["space_group_spin"]["spin_space_group_name_chen"] == "P(3)63/(1)m(2)m(2)c.P"
    assert (
        metadata["space_group_spin"]["spin_space_group_name_linear"]
        == "P 1|m 3^1_001|-3 2_100|m m_001|1"
    )
    assert metadata["space_group_spin"]["spin_space_group_name_latex"] == "P latex"
    assert metadata["space_group_spin"]["spin_space_group_number_chen"] == "194.147.1.1.P3"
    assert metadata["space_group_spin"]["rotation_angle"] == 60.0
    assert metadata["space_group_spin"]["rotation_axis_xyz"]["numeric_components"] == [0.0, 0.0, 1.0]
    assert metadata["space_group_spin"]["rotation_axis_cartn"]["numeric_components"] == [0.0, 0.0, 1.0]
    assert metadata["space_group_spin"]["transform_spinframe_P_abc"] == "a,b,c"


def test_parse_scif_metadata_reads_symbolic_spin_vectors(tmp_path):
    scif_text = """#\\#CIF_2.0
data_test
_space_group_spin.collinear_direction "sqrt(2)/2,0,sqrt(2)/2"
"""
    scif_path = Path(tmp_path) / "symbolic_metadata.scif"
    scif_path.write_text(scif_text, encoding="utf-8")

    metadata = parse_scif_metadata(scif_path)

    assert metadata["space_group_spin"]["collinear_direction"]["numeric_components"] == [
        np.sqrt(2) / 2,
        0.0,
        np.sqrt(2) / 2,
    ]


def test_parse_scif_metadata_keeps_backward_compatibility_for_old_fsg_tags(tmp_path):
    scif_text = """#\\#CIF_2.0
data_test
_space_group_spin.fsg.spin_space_group_name_linear "legacy linear symbol"
_space_group_spin.fsg.transform_to_input_Pp 'a,b,c;0,0,0'
"""
    scif_path = Path(tmp_path) / "legacy_fsg_tags.scif"
    scif_path.write_text(scif_text, encoding="utf-8")

    metadata = parse_scif_metadata(scif_path)

    assert metadata["space_group_spin"]["spin_space_group_name_linear"] == "legacy linear symbol"
    assert metadata["space_group_spin"]["transform_to_input_Pp"] == "a,b,c;0,0,0"


def test_parse_scif_metadata_reads_orbital_moment_category(tmp_path):
    scif_text = """#\\#CIF_2.0
data_test
loop_
_atom_site_orbital_moment.label
_atom_site_orbital_moment.axis_x
_atom_site_orbital_moment.axis_y
_atom_site_orbital_moment.axis_z
_atom_site_orbital_moment.symmform_xyz
_atom_site_orbital_moment.magnitude
Mn1 0.1 0.2 0.3 mx,0,0 0.374
"""
    scif_path = Path(tmp_path) / "orbital_metadata.scif"
    scif_path.write_text(scif_text, encoding="utf-8")

    metadata = parse_scif_metadata(scif_path)

    assert metadata["atom_site_orbital_moment"]["label"] == ["Mn1"]
    assert metadata["atom_site_orbital_moment"]["crystalaxis_x"] == ["0.1"]
    assert metadata["atom_site_orbital_moment"]["crystalaxis_y"] == ["0.2"]
    assert metadata["atom_site_orbital_moment"]["crystalaxis_z"] == ["0.3"]
    assert metadata["atom_site_orbital_moment"]["symmform_xyz"] == ["mx,0,0"]
    assert metadata["atom_site_orbital_moment"]["magnitude"] == ["0.374"]


def test_parse_scif_metadata_reads_uvw_id_references_as_metadata_only(tmp_path):
    scif_text = """#\\#CIF_2.0
data_test
loop_
_space_group_symop_spin_operation.id
_space_group_symop_spin_operation.xyzt
_space_group_symop_spin_operation.uvw_id
1 x,y,z,+1 7

loop_
_space_group_symop_spin_lattice.id
_space_group_symop_spin_lattice.xyzt
_space_group_symop_spin_lattice.uvw_id
1 x,y,z,+1 1
"""
    scif_path = Path(tmp_path) / "uvw_id_only.scif"
    scif_path.write_text(scif_text, encoding="utf-8")

    metadata = parse_scif_metadata(scif_path)

    assert metadata["space_group_symop_spin_operation"]["uvw"] is None
    assert metadata["space_group_symop_spin_operation"]["uvw_id"] == ["7"]
    assert metadata["space_group_symop_spin_lattice"]["uvw"] is None
    assert metadata["space_group_symop_spin_lattice"]["uvw_id"] == ["1"]


def test_parse_scif_metadata_reads_mainline_output(tmp_path):
    result = find_spin_group("examples/0.800_MnTe.mcif")
    scif_path = Path(tmp_path) / "mnte.scif"
    scif_path.write_text(result.scif, encoding="utf-8")

    metadata = parse_scif_metadata(scif_path)

    assert metadata["space_group_spin"]["source_tags"]["collinear_direction"] == (
        "_space_group_spin.collinear_direction_xyz"
    )
    assert metadata["space_group_spin"]["source_tags"]["spin_space_group_number_chen"] == (
        "_space_group_spin.number_Chen"
    )
    assert metadata["space_group_spin"]["spin_space_group_number_chen"] == result.index
    assert metadata["space_group_spin"]["spin_space_group_name_chen"] is not None
    assert metadata["space_group_spin"]["spin_space_group_name_chen"] != (
        result.convention_ssg_international_linear
    )
    assert metadata["space_group_spin"]["source_tags"]["spin_space_group_name_chen"] == (
        "_space_group_spin.name_Chen"
    )
    assert metadata["space_group_spin"]["spin_space_group_name_linear"] == (
        result.convention_ssg_international_linear
    )
    assert metadata["space_group_spin"]["source_tags"]["spin_space_group_name_linear"] == (
        "_space_group_spin.fsg_oriented_spin_space_group_name_linear"
    )
    assert metadata["space_group_spin"]["spin_space_group_name_latex"] == (
        result.convention_ssg_international_latex
    )
    assert metadata["space_group_spin"]["source_tags"]["spin_space_group_name_latex"] == (
        "_space_group_spin.fsg_oriented_spin_space_group_name_latex"
    )
    assert metadata["space_group_spin"]["source_tags"]["rotation_axis_xyz"] == (
        "_space_group_spin.rotation_axis"
    )
    assert metadata["space_group_spin"]["rotation_axis_cartn"] is None
    assert metadata["space_group_spin"]["source_tags"]["rotation_axis_cartn"] is None
    assert metadata["space_group_spin"]["transform_spinframe_P_abc"] == "a,b,c"
    assert metadata["space_group_spin"]["source_tags"]["transform_spinframe_P_abc"] == (
        "_space_group_spin.transform_spinframe_P_abc"
    )
    assert metadata["space_group_spin"]["transform_spinframe_P_matrix"] is None
    assert isinstance(metadata["space_group_spin"]["transform_to_input_Pp"], str)
    assert isinstance(metadata["space_group_spin"]["transform_to_magnetic_primitive_Pp"], str)
    assert isinstance(metadata["space_group_spin"]["transform_to_L0std_Pp"], str)
    assert isinstance(metadata["space_group_spin"]["transform_to_G0std_Pp"], str)
    assert metadata["space_group_spin"]["magnetic_phase"] == result.magnetic_phase.replace("\n", "")
    assert metadata["space_group_spin"]["repo_local_extensions"]["spin_space_group_name_linear"] == (
        result.convention_ssg_international_linear
    )


def test_generated_scif_uses_cif_legal_fsg_tags_and_full_precision_operations():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert "_space_group_spin.number_Chen" in result.scif
    assert "_space_group_spin.name_Chen" in result.scif
    assert " : " in result.convention_ssg_international_linear
    assert "_space_group_spin.fsg_spin_space_group_name_linear" not in result.scif
    assert "_space_group_spin.fsg_oriented_spin_space_group_name_linear" in result.scif
    assert "_space_group_spin.fsg_oriented_spin_space_group_name_latex" in result.scif
    assert "_space_group_spin.spin_space_point_group_name" not in result.scif
    assert (
        f'_space_group_spin.fsg_spin_space_point_group_name  "{result.spin_part_point_group}"'
        in result.scif
    )
    assert "_space_group_spin.fsg_G0_number" in result.scif
    assert "_space_group_spin.fsg_G0_number  148" in result.scif
    assert '_space_group_spin.fsg_magnetic_phase  "AFM(SOM)"' in result.scif
    assert "_space_group_spin.fsg_transform_to_input_Pp" in result.scif
    assert "_space_group_spin.fsg.transform_to_input_Pp" not in result.scif
    assert "_space_group_spin.fsg_transform_to_parent_space_group_Pp" not in result.scif
    assert (
        "_space_group_spin.transform_Chen_Pp_abcs  "
        "'a,b,c;0,0,0;-25.782269as+25.782269bs-6.445567cs,-25.782269bs-6.445567cs,25.782269as-6.445567cs'"
        in result.scif
    )
    assert "1/3u-1/3w,2/3u-v-1/6w,-8/3u-1/3w" in result.scif
    assert "0.333333333333333u" not in result.scif
    assert "_space_group_spin.rotation_axis  ." in result.scif
    assert "_space_group_spin.rotation_angle ." in result.scif
    assert (
        '_parent_space_group.child_transform_Pp_abc  "-2a+2c,2a+2b,-1/2a+1/2b-1/2c;0,0,0"'
        in result.scif
    )


def test_generated_scif_uses_solver_derived_symmform_uvw_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    assert (
        "Fe1\t-1.781909088590099\t1.781909088590101\t-2.182384017536785\t"
        "u,-u,sqrt(6)/2u\tu,-u,4u\t3.780"
    ) in result.scif
    assert "Sx,Sy,Sz" not in result.scif
    assert metadata["atom_site_spin_moment"]["symmform_uvw"] == ["u,-u,sqrt(6)/2u"]
    assert metadata["atom_site_spin_moment"]["symmform_rel_uvw"] == ["u,-u,4u"]


def test_generated_scif_uses_solver_derived_symmform_uvw_for_mnte():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    assert "Mn1\t4.6\t4.599999999999999\t0\tu,u,0\tu,u,0\t4.600" in result.scif
    assert "Sx,Sy,Sz" not in result.scif
    assert metadata["atom_site_spin_moment"]["symmform_uvw"] == ["u,u,0"]
    assert metadata["atom_site_spin_moment"]["symmform_rel_uvw"] == ["u,u,0"]
    assert metadata["space_group_spin"]["transform_Chen_Pp_abcs"] == "a,b,c;0,0,0;as,bs,cs"


def test_generated_scif_uses_solver_derived_symmform_uvw_for_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    assert metadata["atom_site_spin_moment"]["symmform_uvw"] == ["u,-u,0"]
    assert metadata["atom_site_spin_moment"]["symmform_rel_uvw"] == ["u,-u,0"]


def test_generated_scif_uses_solver_derived_symmform_uvw_for_conbs_tripleq_default_branch():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    assert metadata["atom_site_spin_moment"]["symmform_uvw"] == ["u,-u,-0.612408u"]
    assert metadata["atom_site_spin_moment"]["symmform_rel_uvw"] == ["u,-u,-0.592417u"]


def test_generated_scif_transform_to_g0std_maps_current_setting_to_g0std_equivalent_ops():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)
    transform_matrix, origin_shift = _parse_pp_transform(metadata["space_group_spin"]["transform_to_G0std_Pp"])

    current_ssg = SpinSpaceGroup(result.g0_standard_ssg_ops)
    transformed = current_ssg.transform(transform_matrix, origin_shift)
    target = current_ssg.transform(current_ssg.transformation_to_G0std, current_ssg.origin_shift_to_G0std)

    assert _serialize_spatial_ops(transformed.ops) == _serialize_spatial_ops(target.ops)


def test_generated_scif_transform_to_input_maps_current_setting_to_input_equivalent_ops():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)
    transform_matrix, origin_shift = _parse_pp_transform(metadata["space_group_spin"]["transform_to_input_Pp"])
    expected_matrix, expected_shift = _invert_setting_transform(
        np.asarray(result.T_input_to_G0std[0], dtype=float),
        np.asarray(result.T_input_to_G0std[1], dtype=float),
    )

    assert np.allclose(transform_matrix, expected_matrix, atol=1e-8)
    assert np.allclose(origin_shift, expected_shift, atol=1e-8)


def test_generated_scif_transform_to_input_recovers_1669_source_magnetic_fe_semantics():
    source_path = Path("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")
    result = find_spin_group(str(source_path))
    metadata = parse_scif_metadata(source_text=result.scif)

    transform_matrix, origin_shift = _parse_pp_transform(metadata["space_group_spin"]["transform_to_input_Pp"])
    lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_file(source_text=result.scif)
    scif_cell = CrystalCell(lattice_factors, positions, occupancies, elements, moments, "in_lattice")
    back_in_input_setting = scif_cell.transform(transform_matrix, origin_shift)

    source_lattice_factors, source_positions, source_elements, source_occupancies, _, source_moments = parse_cif_file(
        source_path
    )
    source_cell = CrystalCell(
        source_lattice_factors,
        source_positions,
        source_occupancies,
        source_elements,
        source_moments,
        "in_lattice",
    )

    # `fsg_transform_to_input_Pp` is spatial-only; after applying it, the moments are still
    # expressed in the emitted SCIF in-lattice frame. Convert them into the input-cell
    # in-lattice frame before comparing against the source magnetic semantics.
    source_spin_basis = _actual_basis_spin_transform(source_cell)
    back_spin_basis = _actual_basis_spin_transform(back_in_input_setting)
    spin_back_to_source = source_spin_basis @ np.linalg.inv(back_spin_basis)
    back_in_source_spin_frame = back_in_input_setting.transform_spin(spin_back_to_source, "in_lattice")

    source_fe = [
        (np.asarray(position, dtype=float), np.asarray(moment, dtype=float))
        for position, element, moment in zip(source_cell.positions, source_cell.elements, source_cell.moments)
        if element == "Fe" and np.linalg.norm(moment) > 1e-8
    ]
    recovered_fe = [
        (np.asarray(position, dtype=float), np.asarray(moment, dtype=float))
        for position, element, moment in zip(
            back_in_source_spin_frame.positions,
            back_in_source_spin_frame.elements,
            back_in_source_spin_frame.moments,
        )
        if element == "Fe" and np.linalg.norm(moment) > 1e-8
    ]

    assert len(source_fe) == 12
    assert len(recovered_fe) == 12

    unused = set(range(len(recovered_fe)))
    for source_position, source_moment in source_fe:
        match_idx = next(
            (
                idx
                for idx in unused
                if are_positions_equivalent(source_position, recovered_fe[idx][0], tolerance=1e-4)
                and np.allclose(source_moment, recovered_fe[idx][1], atol=1e-4)
            ),
            None,
        )
        assert match_idx is not None, (
            "Expected `fsg_transform_to_input_Pp` plus the corresponding in-lattice spin-frame "
            "reconciliation to recover the source Fe magnetic semantics for 1.669_KFe(PO3F)2."
        )
        unused.remove(match_idx)

    assert not unused


def test_generated_scif_stabilizes_boundary_fractional_coordinates_for_1669():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert "0.00000111" not in result.scif
    assert "0.00000222" not in result.scif
    assert "Fe1\tFe\t0\t0.33333222\t0.375" in result.scif
    assert "K1\tK\t0\t0\t0" in result.scif


def test_generated_scif_prefers_symbolic_sqrt_coefficients_for_1669():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert "0.57735200925825" not in result.scif
    assert "0.577348529119253" not in result.scif
    assert "-sqrt(3)/3u-sqrt(3)/3v" in result.scif
    assert "2*sqrt(3)/3u-sqrt(3)/3v" in result.scif
