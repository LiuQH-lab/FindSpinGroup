import json
import importlib
import numpy as np
import pytest

import findspingroup.core.identify_symmetry_from_ops as identify_symmetry_from_ops_module
import findspingroup.structure.group as group_module
from findspingroup import find_spin_group, find_spin_group_from_data
from findspingroup.core.identify_index.functions import (
    find_stand_gen_maps,
    is_matrix_equal,
    make_4d_matrix,
    map_transformation,
)
from findspingroup.core.identify_index.functions.get_stand_trans import get_stand_trans
from findspingroup.core.identify_index.functions.find_ssg_reduce import (
    find_ssg_transformation,
)
from findspingroup.core.identify_spin_space_group import (
    NONMAGNETIC_MTOL_ERROR,
    UNSTABLE_MTOL_ERROR,
    dedup_moments_with_tol,
    get_pg,
    identify_spin_space_group,
    identify_spin_space_group_result,
    _classify_moment_configuration,
)
from findspingroup.core.identify_symmetry_from_ops import (
    analyze_transition_matrix_problem,
    deduplicate_matrix_pairs,
    find_transition_matrix_deterministic,
    get_magnetic_space_group_from_operations,
    identify_point_group,
)
from findspingroup.core import Molecule, PointGroupAnalyzer
from findspingroup.core.pg_analyzer import SymmOp, generate_full_symmops
from findspingroup.find_spin_group import (
    SCIF_CELL_MODE_G0STD_ORIENTED,
    SCIF_CELL_MODE_INPUT,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
    _canonicalize_input_to_standard_setting,
    audit_spatial_transform_effect,
    _build_candidate_transform_chen_pp_abcs_hex_spatial_cubic_spin_from_identify,
    _ossg_oriented_spin_frame_ssg,
    _spin_transform_to_in_lattice,
    _spin_transform_to_oriented_abc,
    _build_msg_little_group_payload,
    _get_magnetic_little_group,
    _primitive_msg_ops_from_ssg,
    _tensor_ops_wo_soc,
    _translations_equivalent_mod_pure_translations,
    combine_parametric_solutions,
)
from findspingroup.io import parse_cif_file, parse_scif_metadata
from findspingroup.structure import SpinSpaceGroup
from findspingroup.structure.cell import CrystalCell, standardize_lattice
from findspingroup.utils.international_symbol import (
    _compose_setting_transform as _compose_symbol_setting_transform,
    _default_centering_vectors,
    _find_real_operation,
    _parse_sg_generator_ops,
    _select_preferred_primitive_translation_match,
    _select_preferred_translation_match,
    _transport_standard_generators_to_current_basis,
    build_international_symbol,
)
from findspingroup.utils.space_group_flags import (
    msg_parent_space_group_info,
    space_group_is_chiral,
    space_group_is_polar,
    space_group_has_real_space_inversion,
)
from findspingroup.utils.seitz_symbol import describe_point_operation
from findspingroup.utils import general_positions_to_matrix
from web_app.website6 import serialize_data

find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")


def _serialize_gspg_pairs(ops):
    return [
        [
            np.asarray(spin_rotation, dtype=float).tolist(),
            np.asarray(space_rotation, dtype=float).tolist(),
        ]
        for spin_rotation, space_rotation in ops
    ]


def _serialize_effective_mpg_ops(ops):
    return [
        [int(time_reversal), np.asarray(rotation, dtype=float).tolist()]
        for time_reversal, rotation in ops
    ]


def _serialize_msg_ops(ops):
    return [
        [
            int(time_reversal),
            np.asarray(rotation, dtype=float).tolist(),
            np.asarray(translation, dtype=float).tolist(),
        ]
        for time_reversal, rotation, translation in ops
    ]


def _serialize_rotation_ops(ops):
    return [np.asarray(rotation, dtype=float).tolist() for rotation in ops]


def _primitive_magnetic_cell_from_cif(path: str) -> CrystalCell:
    lattice_factors, positions, elements, occupancies, _labels, moments = parse_cif_file(path)
    primitive_cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    return primitive_cell


def _rotation_order(rotation: np.ndarray, *, max_order: int = 12, tol: float = 1e-6) -> int | None:
    power = np.eye(3)
    rotation = np.asarray(rotation, dtype=float)
    for order in range(1, max_order + 1):
        power = power @ rotation
        if np.allclose(power, np.eye(3), atol=tol):
            return order
    return None


def _effective_proper_axis_from_space_rotation(rotation: np.ndarray, *, tol: float = 1e-6) -> np.ndarray | None:
    rotation = np.asarray(rotation, dtype=float)
    effective = rotation if np.linalg.det(rotation) > 0 else -rotation
    eigenvalues, eigenvectors = np.linalg.eig(effective)
    matches = np.isclose(eigenvalues, 1.0, atol=tol)
    if not np.any(matches):
        return None
    axis = eigenvectors[:, matches][:, 0].real
    axis = axis / np.linalg.norm(axis)
    for value in axis:
        if abs(value) > tol:
            if value < 0:
                axis = -axis
            break
    return axis


def _serialize_ssg_ops(ops):
    return [
        [
            np.asarray(op[0], dtype=float).tolist(),
            np.asarray(op[1], dtype=float).tolist(),
            np.asarray(op[2], dtype=float).tolist(),
        ]
        for op in ops
    ]


def test_combine_parametric_solutions_uses_axis_named_single_free_variable_for_z_only():
    rref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert combine_parametric_solutions(rref) == ["0", "0", "Sz"]


def test_combine_parametric_solutions_uses_axis_named_single_free_variable_for_first_nonzero_component():
    # x = 0, y - 0.57 z = 0 -> (0, 0.57*t, t)
    rref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -0.57]])
    assert combine_parametric_solutions(rref) == ["0", "0.57*Sy", "Sy"]


def test_combine_parametric_solutions_snaps_common_sqrt_coefficients():
    rref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -0.57735200925825]])
    assert combine_parametric_solutions(rref) == ["0", "sqrt(3)/3*Sy", "Sy"]


def test_tensor_output_display_snaps_common_sqrt_coefficients():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    relations = result.tensor_outputs["AHE_wSOC"]["relations"]
    components = result.tensor_outputs["AHE_wSOC"]["components"]

    assert relations == [
        r"\sigma_{xz} = -sqrt(3)/3\sigma_{yz} = -\sigma_{zx} = sqrt(3)/3\sigma_{zy}"
    ]
    assert components == [
        ["0", "0", r"sqrt(3)/3\sigma_{zy}"],
        ["0", "0", r"-\sigma_{zy}"],
        [r"-sqrt(3)/3\sigma_{zy}", r"\sigma_{zy}", "0"],
    ]


def test_combine_parametric_solutions_keeps_multi_free_variable_ordering():
    rref = np.array([[1.0, 0.0, 0.0]])
    assert combine_parametric_solutions(rref) == ["0", "Sx", "Sy"]


def test_space_group_has_real_space_inversion_lookup_matches_basic_examples():
    assert space_group_has_real_space_inversion(1) is False
    assert space_group_has_real_space_inversion(2) is True
    assert space_group_has_real_space_inversion(33) is False
    assert space_group_has_real_space_inversion(14) is True
    assert space_group_has_real_space_inversion(110) is False
    assert space_group_has_real_space_inversion(123) is True


def test_space_group_is_polar_lookup_matches_reference_examples():
    for sg in [1, 3, 4, 5, 25, 75, 99, 143, 156, 168, 183]:
        assert space_group_is_polar(sg) is True
    for sg in [2, 14, 16, 47, 62, 123, 195, 207]:
        assert space_group_is_polar(sg) is False


def test_space_group_is_chiral_lookup_matches_reference_examples():
    for sg in [1, 3, 4, 5, 16, 75, 143, 168, 195, 207, 214]:
        assert space_group_is_chiral(sg) is True
    for sg in [2, 14, 25, 47, 62, 99, 156, 183]:
        assert space_group_is_chiral(sg) is False


def test_msg_bns_and_og_first_segments_agree_on_real_space_inversion_rule():
    from findspingroup.data import MSGMPG_DB

    for msg_num in MSGMPG_DB.MSG_INT_TO_BNS:
        if msg_num is None:
            continue
        info = msg_parent_space_group_info(msg_num)
        assert space_group_has_real_space_inversion(
            info["bns_parent_space_group_number"]
        ) == space_group_has_real_space_inversion(
            info["og_parent_space_group_number"]
        )
        assert space_group_is_polar(
            info["bns_parent_space_group_number"]
        ) == space_group_is_polar(
            info["og_parent_space_group_number"]
        )
        assert space_group_is_chiral(
            info["bns_parent_space_group_number"]
        ) == space_group_is_chiral(
            info["og_parent_space_group_number"]
        )


def test_find_spin_group_exposes_real_space_inversion_flags_from_identified_numbers():
    noncentro = find_spin_group("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif")
    centro = find_spin_group("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif")

    assert noncentro.input_space_group_number == 33
    assert noncentro.sg_has_real_space_inversion is False
    assert noncentro.ossg_space_group_number == 33
    assert noncentro.ossg_has_real_space_inversion is False
    assert noncentro.msg_parent_space_group_number == 33
    assert noncentro.msg_has_real_space_inversion is False

    assert centro.input_space_group_number == 14
    assert centro.sg_has_real_space_inversion is True
    assert centro.ossg_space_group_number == 14
    assert centro.ossg_has_real_space_inversion is True
    assert centro.msg_parent_space_group_number == 14
    assert centro.msg_has_real_space_inversion is True


def test_find_spin_group_exposes_polar_and_chiral_flags_from_identified_numbers():
    polar_chiral = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    centro_nonpolar = find_spin_group("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif")

    assert polar_chiral.input_space_group_number == 182
    assert polar_chiral.sg_is_polar is False
    assert polar_chiral.sg_is_chiral is True
    assert polar_chiral.ossg_space_group_number == 182
    assert polar_chiral.ossg_is_polar is False
    assert polar_chiral.ossg_is_chiral is True
    assert polar_chiral.msg_parent_space_group_number == 150
    assert polar_chiral.msg_is_polar is False
    assert polar_chiral.msg_is_chiral is True

    assert centro_nonpolar.input_space_group_number == 14
    assert centro_nonpolar.sg_is_polar is False
    assert centro_nonpolar.sg_is_chiral is False
    assert centro_nonpolar.ossg_space_group_number == 14
    assert centro_nonpolar.ossg_is_polar is False
    assert centro_nonpolar.ossg_is_chiral is False
    assert centro_nonpolar.msg_parent_space_group_number == 14
    assert centro_nonpolar.msg_is_polar is False
    assert centro_nonpolar.msg_is_chiral is False


def test_generate_full_symmops_raises_on_runaway_non_group_closure():
    bad_generator = SymmOp(np.diag([1.01, 1.0, 1.0, 1.0]), tol=1e-8)
    with pytest.raises(ValueError, match="maximum generated operation count"):
        generate_full_symmops([bad_generator], tol=1e-8, max_generated_ops=20)


def test_find_spin_group_uses_acc_primitive_ossg_msg_info_for_public_msg_fields():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    internal_msg_info = get_magnetic_space_group_from_operations(result.primitive_msg_ops)

    assert result.msg_num == internal_msg_info["msg_int_num"]
    assert result.msg_type == internal_msg_info["msg_type"]
    assert result.msg_symbol == internal_msg_info["msg_bns_symbol"]


def test_find_spin_group_exposes_used_tolerances_in_result_metadata():
    result = find_spin_group(
        "examples/CoNb3S6_tripleQ.mcif",
        space_tol=0.03,
        mtol=0.05,
        meigtol=1e-4,
        matrix_tol=0.02,
        parser_atol=0.08,
    )

    assert result.tolerances == {
        "space_tol": 0.03,
        "mtol": 0.05,
        "meigtol": 1e-4,
        "matrix_tol": 0.02,
        "parser_atol": 0.08,
    }


def _identify_translation_vectors(details):
    return [
        np.asarray(item[1][1], dtype=float)
        for item in details["translation_maps"]
    ]


def _build_similarity_transformed_generator(standard_generator, transform):
    transform = np.asarray(transform, dtype=float)
    return transform @ np.asarray(standard_generator, dtype=float) @ np.linalg.inv(transform)


def test_analyze_transition_matrix_problem_reports_stable_candidate_for_mirror_group():
    mirror = np.diag([1.0, 1.0, -1.0])
    transform = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.0, 1.3, 0.2],
            [0.1, 0.0, 0.9],
        ],
        dtype=float,
    )
    transformed_mirror = _build_similarity_transformed_generator(mirror, transform)

    analysis = analyze_transition_matrix_problem([transformed_mirror], "m", id=True)

    assert analysis["group_symbol"] == "m"
    assert analysis["expected_null_space_dimension"] == 5
    assert analysis["null_space_dimension"] >= 1
    assert len(analysis["basis_metrics"]) == analysis["null_space_dimension"]
    assert analysis["best_candidate"] is not None
    assert analysis["best_candidate"]["passes_residual_tol"] is True
    assert analysis["best_candidate"]["sigma_min"] > 1e-8


def test_find_transition_matrix_deterministic_is_repeatable_for_near_mirror_generator():
    mirror = np.diag([1.0, 1.0, -1.0])
    transform = np.array(
        [
            [1.0, 0.25, 0.05],
            [0.0, 1.2, 0.15],
            [0.08, -0.03, 0.95],
        ],
        dtype=float,
    )
    transformed_mirror = _build_similarity_transformed_generator(mirror, transform)

    solution_a = find_transition_matrix_deterministic([transformed_mirror], "m", id=True)
    solution_b = find_transition_matrix_deterministic([transformed_mirror], "m", id=True)

    assert np.allclose(solution_a, solution_b, atol=1e-10)
    assert abs(np.linalg.det(solution_a)) > 1e-8


def test_find_spin_group_forwards_parser_atol_to_parse_structure_file(monkeypatch):
    captured = {}
    fake_parsed = (
        np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0], dtype=float),
        [np.zeros(3)],
        ["X"],
        [1.0],
        ["X1"],
        [np.zeros(3)],
    )

    def fake_parse_structure_file(filename, atol=0.01, return_metadata=False):
        captured["filename"] = filename
        captured["atol"] = atol
        captured["return_metadata"] = return_metadata
        return fake_parsed, {"kind": "fake"}

    def fake_find_spin_group_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        source_metadata=None,
    ):
        captured["source_name"] = source_name
        captured["source_metadata"] = source_metadata
        return {"ok": True}

    monkeypatch.setattr(find_spin_group_module, "parse_structure_file", fake_parse_structure_file)
    monkeypatch.setattr(find_spin_group_module, "_find_spin_group_from_parsed", fake_find_spin_group_from_parsed)

    result = find_spin_group_module.find_spin_group("dummy.scif", parser_atol=0.123)

    assert result == {"ok": True}
    assert captured["filename"] == "dummy.scif"
    assert captured["atol"] == 0.123
    assert captured["return_metadata"] is True
    assert captured["source_name"] == "dummy.scif"
    assert captured["source_metadata"] == {"kind": "fake"}


def test_find_transition_matrix_deterministic_error_suggests_pg_standardization_direction(monkeypatch):
    mirror = np.diag([1.0, 1.0, -1.0])
    transform = np.array(
        [
            [1.0, 0.25, 0.05],
            [0.0, 1.2, 0.15],
            [0.08, -0.03, 0.95],
        ],
        dtype=float,
    )
    transformed_mirror = _build_similarity_transformed_generator(mirror, transform)

    monkeypatch.setattr(
        identify_symmetry_from_ops_module,
        "_search_transition_candidate",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError) as exc_info:
        find_transition_matrix_deterministic([transformed_mirror], "m", id=True)

    message = str(exc_info.value)
    assert "无法在零空间中找到非奇异矩阵 P。" in message
    assert "find_spin_group(..., matrix_tol=...)" in message
    assert "meigtol=..." in message


def test_get_stand_trans_error_suggests_database_coverage_direction():
    with pytest.raises(ValueError) as exc_info:
        get_stand_trans(
            143,
            147,
            2,
            12,
            64,
            (np.identity(3), np.zeros(3)),
            [],
            [],
            tol=0.001,
        )

    message = str(exc_info.value)
    assert "No identify-index reduction record for L0=143, G0=147, it=2, ik=12, iso=64." in message
    assert "database/special-case coverage" in message
    assert "do not tune `space_tol`, `mtol`, `meigtol`, or `matrix_tol` first" in message


def _build_identify_standardization_debug(result):
    details = result.identify_index_details
    reduce_info = find_ssg_transformation(
        details["L0_id"],
        details["G0_id"],
        details["t_index"],
        details["k_index"],
        details["point_group_id"],
        make_4d_matrix(details["transformation_matrix"]),
        tol=0.01,
    )
    standardization_transform = np.linalg.inv(make_4d_matrix(reduce_info["TTM"]))
    transformed_name_maps = map_transformation(
        details["name_maps"],
        standardization_transform,
    )
    transformed_translation_maps = map_transformation(
        details["translation_maps"],
        standardization_transform,
    )
    standard_generator_maps = find_stand_gen_maps(
        transformed_name_maps,
        transformed_translation_maps,
        reduce_info["gen_matrices"],
        reduce_info["cell_size"],
    )
    database_standard_generators = [
        np.asarray(make_4d_matrix(generator), dtype=float)
        for generator in reduce_info["gen_matrices"]
    ]
    return {
        "identify_index_details": details,
        "reduce_info": reduce_info,
        "standardization_transform": np.asarray(
            standardization_transform,
            dtype=float,
        ),
        "transformed_name_maps": [
            {
                "point": np.asarray(item[0], dtype=float),
                "space": np.asarray(item[1], dtype=float),
            }
            for item in transformed_name_maps
        ],
        "transformed_translation_maps": [
            {
                "point": np.asarray(item[0], dtype=float),
                "space": np.asarray(item[1], dtype=float),
            }
            for item in transformed_translation_maps
        ],
        "standard_generator_maps": [
            {
                "point": np.asarray(item[0], dtype=float),
                "space": np.asarray(item[1], dtype=float),
            }
            for item in standard_generator_maps
        ],
        "database_standard_generators": database_standard_generators,
    }


def test_find_spin_group_exposes_main_flow_identify_result_for_collinear_case():
    result = find_spin_group("examples/0.800_MnTe.mcif")

    assert result.index == "194.164.1.1.L"
    assert result.conf == "Collinear"
    assert result.spin_part_point_group == "∞/mm"
    assert result.magnetic_phase == "AFM(Altermagnet)"
    assert result.magnetic_phase_base == "AFM"
    assert result.magnetic_phase_modifier == "(Altermagnet)"
    assert result.is_spin_orbit_magnet == ""
    assert result.magnetic_phase_details["classification_rule"] == "default_antiferromagnetic"
    assert result.magnetic_phase_details["is_altermagnet"] is True
    assert result.magnetic_phase_details["is_spin_orbit_magnet"] is False
    assert result.identify_index_details is not None
    assert result.identify_index_details["G0_id"] == 194
    assert result.identify_index_details["L0_id"] == 164
    assert result.identify_index_details["k_index"] == 1
    assert result.identify_index_details["equivalent_map_index"] == 1


def test_find_spin_group_exposes_identify_transformations_for_coplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    details = result.identify_index_details

    assert result.index == "38.6.1.2.P2"
    assert result.conf == "Coplanar"
    assert details["G0_id"] == 38
    assert details["L0_id"] == 6
    assert details["t_index"] == 2
    assert details["k_index"] == 1
    assert details["point_group_id"] == 2
    assert details["equivalent_map_index"] == 2
    assert details["configuration_suffix"] == "P2"

    lattice_transform = details["transformation_matrix"]
    assert np.asarray(lattice_transform[0], dtype=float).shape == (3, 3)
    assert np.asarray(lattice_transform[1], dtype=float).shape == (3,)

    space_transform = np.asarray(details["space_group_transformation"], dtype=float)
    point_transform = np.asarray(details["point_group_transformation"], dtype=float)
    assert space_transform.shape == (4, 4)
    assert point_transform.shape == (3, 3)
    assert abs(np.linalg.det(space_transform[:3, :3])) > 1e-8
    assert abs(np.linalg.det(point_transform)) > 1e-8

    assert details["name_maps"]
    assert len(details["translation_maps"]) == 3


@pytest.mark.parametrize(
    ("source_path", "expected_index", "expected_suffix"),
    [
        ("tests/testset/mcif_241130_no2186/0.1010_C10H6MnN4O4.mcif", "14.1.1.1.P2", "P2"),
        ("tests/testset/mcif_241130_no2186/0.394_Cu2CdB2O6.mcif", "14.1.1.1.P2", "P2"),
        ("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif", "33.1.1.1.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/0.716_HoCrWO6.mcif", "33.1.1.1.P1", "P1"),
        ("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif", "14.2.2.1.P2", "P2"),
        ("tests/testset/mcif_241130_no2186/1.197_Fe4Si2Sn7O16.mcif", "12.2.2.1.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/1.647_Na2.4Ni2TeO6.mcif", "63.13.2.1.P1", "P1"),
        ("tests/testset/mcif_241130_no2186/2.96_GdMn2Si2.mcif", "139.115.2.11.P3", "P3"),
    ],
)
def test_find_spin_group_uses_excel_backed_suffixes_for_coplanar_d2_identify_branch(
    source_path,
    expected_index,
    expected_suffix,
):
    result = find_spin_group(source_path)
    details = result.identify_index_details

    assert result.conf == "Coplanar"
    assert result.index == expected_index
    assert details["configuration_suffix"] == expected_suffix


def test_build_candidate_transform_chen_pp_abcs_for_324_hex_spatial_cubic_spin():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    transform_to_input = metadata["space_group_spin"]["transform_to_input_Pp"]
    basis_expr = transform_to_input.split(";", 1)[0]
    basis_matrices, _ = general_positions_to_matrix([basis_expr], variables=("a", "b", "c"))
    current_space_to_input_basis = np.asarray(basis_matrices[0][0], dtype=float)

    candidate = _build_candidate_transform_chen_pp_abcs_hex_spatial_cubic_spin_from_identify(
        current_space_to_input_basis=current_space_to_input_basis,
        identify_point_group_transformation=np.asarray(
            result.identify_index_details["point_group_transformation"],
            dtype=float,
        ),
    )

    assert candidate["from_spatial_setting"] == "current_scif_g0std_oriented_hex"
    assert candidate["to_spatial_setting"] == "chen_hex_spatial"
    assert candidate["to_spin_frame"] == "chen_cubic_spin_basis"
    assert candidate["transform_Chen_Pp_abcs"] == (
        "a,b,c;0,0,0;"
        "1/3as-1/3bs+4/3cs,1/3as+2/3bs+4/3cs,-2/3as-1/3bs+4/3cs"
    )

    spin_basis_columns = np.asarray(candidate["spin_basis_rows_abcs"], dtype=float).T
    spin_basis_columns_inv = np.linalg.inv(spin_basis_columns)

    current_point_ops = metadata["space_group_symop_spin_operation"]["uvw"][:6]
    point_op_matrices, _ = general_positions_to_matrix(current_point_ops, variables=("u", "v", "w"))
    transformed_point_ops = [
        np.round(spin_basis_columns_inv @ np.asarray(matrix, dtype=float) @ spin_basis_columns, 6)
        for matrix, _ in point_op_matrices
    ]
    expected_point_ops = [
        np.eye(3),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.eye(3),
    ]
    for actual, expected in zip(transformed_point_ops, expected_point_ops):
        assert np.allclose(actual, expected, atol=1e-6)

    current_lattice_ops = metadata["space_group_symop_spin_lattice"]["uvw"][:4]
    lattice_op_matrices, _ = general_positions_to_matrix(current_lattice_ops, variables=("u", "v", "w"))
    transformed_lattice_ops = [
        np.round(spin_basis_columns_inv @ np.asarray(matrix, dtype=float) @ spin_basis_columns, 6)
        for matrix, _ in lattice_op_matrices
    ]
    expected_lattice_ops = [
        np.eye(3),
        np.diag([-1.0, -1.0, 1.0]),
        np.diag([-1.0, -1.0, 1.0]),
        np.diag([-1.0, 1.0, -1.0]),
    ]
    for actual, expected in zip(transformed_lattice_ops, expected_lattice_ops):
        assert np.allclose(actual, expected, atol=1e-6)


def test_find_spin_group_uses_p1_branch_for_parallel_coplanar_order_two_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.317_La0.25Pr0.75Co2P2.mcif")
    details = result.identify_index_details

    assert result.index == "123.123.2.2.P1"
    assert result.conf == "Coplanar"
    assert details["point_group_id"] == 2
    assert details["equivalent_map_index"] == 2
    assert details["configuration_suffix"] == "P1"


@pytest.mark.parametrize(
    ("source_path", "expected_target"),
    [
        ("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif", "My"),
        ("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif", "My"),
        ("tests/testset/mcif_241130_no2186/0.716_HoCrWO6.mcif", "Mz"),
    ],
)
def test_find_spin_group_exposes_total_coplanar_222_spin_transform(source_path, expected_target):
    result = find_spin_group(source_path)
    details = result.identify_index_details

    assert result.conf == "Coplanar"
    assert details["point_group_id"] == 14
    assert details["point_group_transformation_raw"] is not None
    assert details["coplanar_222_q_transform"] is not None
    assert details["coplanar_222_b_transform"] is not None
    assert details["coplanar_222_target_spin_only_matrix"] is not None
    assert details["coplanar_222_target_spin_only_label"] == expected_target

    raw = np.asarray(details["point_group_transformation_raw"], dtype=float)
    q_transform = np.asarray(details["coplanar_222_q_transform"], dtype=float)
    b_transform = np.asarray(details["coplanar_222_b_transform"], dtype=float)
    total = np.asarray(details["point_group_transformation"], dtype=float)
    target_matrix = np.asarray(details["coplanar_222_target_spin_only_matrix"], dtype=float)

    assert np.allclose(total, b_transform @ raw @ q_transform, atol=1e-6)

    mz = np.diag([1.0, 1.0, -1.0])
    assert np.allclose(
        b_transform @ mz @ np.linalg.inv(b_transform),
        target_matrix,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("source_path", "expected_index", "expected_suffix", "expected_phase"),
    [
        ("tests/testset/mcif_241130_no2186/0.1000_Fe4O5.mcif", "36.8.1.2.P2", "P2", "FM/FiM"),
        ("tests/testset/mcif_241130_no2186/0.188_CeMnAsO.mcif", "59.25.1.2.P1", "P1", "AFM"),
        ("tests/testset/mcif_241130_no2186/0.196_Co4Nb2O9.mcif", "165.158.1.2.P1", "P1", "AFM"),
    ],
)
def test_find_spin_group_matches_coplanar_configuration_suffixes_seen_in_batch_regression_scan(
    source_path,
    expected_index,
    expected_suffix,
    expected_phase,
):
    result = find_spin_group(source_path)
    details = result.identify_index_details

    assert result.conf == "Coplanar"
    assert result.index == expected_index
    assert result.magnetic_phase == expected_phase
    assert details["equivalent_map_index"] == 2
    assert details["configuration_suffix"] == expected_suffix


def test_find_spin_group_uses_single_equivalent_map_for_iso_zero_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.333_Mn2Mo3O8.mcif")
    details = result.identify_index_details

    assert result.index == "186.186.1.1.L"
    assert details["point_group_id"] == 0
    assert details["equivalent_map_index"] == 1


def test_ktb3f10_spin_point_group_sentinel_stays_c3v_with_origin_anchor():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.1120_KTb3F10.mcif"
    )
    primitive_cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    non_zero_indices = np.where(np.linalg.norm(primitive_cell.moments, axis=1) > 2e-5)[0]
    filtered_moments = np.array([primitive_cell.moments[i] for i in non_zero_indices])
    filtered_types = np.array([primitive_cell.atom_types[i] for i in non_zero_indices])
    unique_types, unique_moments = dedup_moments_with_tol(filtered_types, filtered_moments, 0.02)
    pg = PointGroupAnalyzer(Molecule(unique_types, unique_moments), tolerance=0.02, eigen_tolerance=2e-5)

    assert pg.sch_symbol == "C3v"
    assert len(pg.get_symmetry_operations()) == 6


@pytest.mark.parametrize(
    ("path", "expected_symbol"),
    [
        ("tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif", "C2v"),
        ("tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif", "C2v"),
        ("tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif", "C2v"),
        ("tests/testset/mcif_241130_no2186/1.412_Au72Al14Tb14.mcif", "Th"),
        ("tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif", "D3d"),
        ("tests/testset/mcif_241130_no2186/1.798_Tb2O3.mcif", "Th"),
        ("examples/CoNb3S6_tripleQ.mcif", "C3v"),
        ("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif", "D3h"),
    ],
)
def test_get_pg_recovers_expected_symbols_for_magnetic_point_sets(path, expected_symbol):
    primitive_cell = _primitive_magnetic_cell_from_cif(path)

    pg_symbol, _pg_operations = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        2e-5,
    )

    assert pg_symbol == expected_symbol


@pytest.mark.parametrize(
    ("path", "expected_conf", "expected_msg_num", "expected_msg_type"),
    [
        ("tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif", "Coplanar", 199, 3),
        ("tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif", "Coplanar", 82, 1),
        ("tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif", "Coplanar", 82, 1),
        ("tests/testset/mcif_241130_no2186/1.713_CsCr0.98Al0.02F4.mcif", "Coplanar", 332, 4),
        ("tests/testset/mcif_241130_no2186/1.748_TbAuIn.mcif", "Coplanar", 1366, 4),
        ("tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif", "Noncoplanar", 1233, 4),
    ],
)
def test_find_spin_group_recovers_pg_boundary_residual_msg_numbers(
    path,
    expected_conf,
    expected_msg_num,
    expected_msg_type,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.index is not None
    assert result.conf == expected_conf
    assert result.msg_num == expected_msg_num
    assert result.msg_type == expected_msg_type
    assert acc_primitive_ossg.msg_int_num == expected_msg_num
    assert acc_primitive_ossg.msg_type == expected_msg_type


def test_find_spin_group_keeps_conbnb3s6_tripleq_pg_boundary_recoverable():
    primitive_cell = _primitive_magnetic_cell_from_cif("examples/CoNb3S6_tripleQ.mcif")
    default_pg_symbol, _ = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        2e-5,
    )
    loose_pg_symbol, _ = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        1e-4,
    )

    default_result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    loose_result = find_spin_group("examples/CoNb3S6_tripleQ.mcif", meigtol=1e-4)

    assert default_pg_symbol == "C3v"
    assert loose_pg_symbol == "Td"
    assert default_result.msg_num == loose_result.msg_num == 1257
    assert default_result.msg_type == loose_result.msg_type == 3


def test_little_groups_symbols_recover_for_conbnb3s6_tripleq():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    symbols = ssg.little_groups_symbols

    assert isinstance(symbols, list)
    assert len(symbols) == len(ssg.kpoints_symbol_primitive)
    assert all(symbol != "?" for symbol in symbols)


def test_little_groups_symbols_use_minus3_not_minus6_for_vcl2_trigonal_cogroups():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.237_VCl2.mcif")
    ssg = SpinSpaceGroup(result.acc_primitive_ssg_ops)

    actual = {
        label: symbol
        for label, symbol in zip(ssg.kpoints_symbol_primitive, ssg.little_groups_symbols)
        if label in {"A:(0,0,1/2)", "Γ:(0,0,0)", "H:(1/3,1/3,1/2)", "K:(1/3,1/3,0)"}
    }

    assert actual["A:(0,0,1/2)"] == "^{m}-3^{m}m^{6/m}1"
    assert actual["Γ:(0,0,0)"] == "^{m}-3^{m}m^{6/m}1"
    assert "-6" not in actual["H:(1/3,1/3,1/2)"]
    assert "-6" not in actual["K:(1/3,1/3,0)"]
    assert "-3" in actual["H:(1/3,1/3,1/2)"]
    assert "-3" in actual["K:(1/3,1/3,0)"]


def test_describe_point_operation_keeps_near_hex_threefold_as_3_not_2():
    matrix = np.array(
        [
            [3.833458638e-05, -1.000019165e00, -7.295004068e-05],
            [1.000019167e00, -9.999808303e-01, -5.285192675e-05],
            [1.532712294e-05, -5.886194163e-05, 9.999999980e-01],
        ],
        dtype=float,
    )

    info = describe_point_operation(matrix, tol=1e-4, max_order=12)

    assert info["hm_symbol"] == "3"
    assert info["rotation_power"] == 1
    assert info["axis_direction"] == (0, 0, 1)
    assert info["symbol"] == "3^{1}_{001}"


def test_mn3sn_seitz_symbols_do_not_emit_illegal_minus6_power2_tokens():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.199_Mn3Sn.mcif")

    joined = "\n".join(result.g0_standard_ssg_seitz + result.primitive_magnetic_cell_ssg_seitz)

    assert "-6^{2}_{001}" not in joined
    assert "-6^{5}_{001}" in joined


def test_describe_point_operation_keeps_near_improper_fourfold_as_minus4():
    matrix = np.array(
        [
            [-0.00195873, -0.9999979, -0.00022571],
            [0.99999823, -0.00195903, 0.00015037],
            [0.00015076, 0.00022567, -0.99999996],
        ],
        dtype=float,
    )

    info = describe_point_operation(matrix, tol=1e-4, max_order=120)

    assert info["hm_symbol"] == "-4"
    assert info["axis_direction"] == (0, 0, 1)
    assert info["symbol"] == "-4^{1}_{001}"


def test_audit_spatial_transform_effect_identity_preserves_real_ops_exactly():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)

    audit = audit_spatial_transform_effect(ssg, np.eye(3), np.zeros(3), tol=1e-6)

    assert audit["source_real_op_count"] == audit["transformed_real_op_count"] == 48
    assert audit["real_ops_exact_same"] is True
    assert audit["real_ops_same_mod_integer"] is True
    assert audit["real_ops_same_mod_pure_translations"] is True
    assert audit["paired_spin_changed_count"] == 0
    assert audit["unmatched_source_indices"] == []


def test_audit_spatial_transform_effect_flags_normalizer_like_real_space_invariance_for_conbnb3s6():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    transform_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    origin_shift = np.array([0.0, 0.5, 0.0], dtype=float)

    audit = audit_spatial_transform_effect(ssg, transform_matrix, origin_shift, tol=1e-6)

    assert audit["source_real_op_count"] == audit["transformed_real_op_count"] == 48
    assert audit["real_ops_exact_same"] is True
    assert audit["real_ops_same_mod_integer"] is True
    assert audit["real_ops_same_mod_pure_translations"] is True
    assert audit["paired_spin_changed_count"] > 0
    assert audit["unmatched_source_indices"] == []


def test_audit_spatial_transform_effect_short_circuits_when_volume_changes():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    transform_matrix = 2.0 * np.eye(3)

    audit = audit_spatial_transform_effect(ssg, transform_matrix, np.zeros(3), tol=1e-6, det_tol=1e-2)

    assert abs(audit["determinant"] - 8.0) < 1e-9
    assert audit["volume_preserving"] is False
    assert audit["can_be_affine_normalizer_equivalent"] is False
    assert audit["real_ops_exact_same"] is False
    assert audit["source_real_op_count"] is None
    assert audit["transformed_to_source"] == []


def test_canonicalize_input_to_standard_setting_returns_cartesianized_input_cell_on_identity_collapse():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "examples/0.800_MnTe.mcif"
    )
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    )
    input_cell_cartesian = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        input_cell.moments_cartesian,
        spin_setting="cartesian",
    )
    result = find_spin_group("examples/0.800_MnTe.mcif")
    target_ssg = SpinSpaceGroup(result.g0_standard_ssg_ops)

    collapsed_cell, collapsed_ssg, collapsed_transform, audit = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        input_cell,
        target_ssg,
        (np.eye(3), np.zeros(3)),
    )

    assert audit["real_ops_exact_same"] is True
    assert collapsed_transform[0].tolist() == np.eye(3).tolist()
    assert collapsed_transform[1].tolist() == np.zeros(3).tolist()
    assert collapsed_cell.spin_setting == "cartesian"
    assert np.allclose(collapsed_cell.moments, input_cell_cartesian.moments, atol=1e-8)
    assert len(collapsed_ssg.ops) == len(target_ssg.ops)


def _changed_basis_conb3s6_tripleq_input():
    lattice = [11.498, 11.498, 11.886, 90, 90, 120]
    _, _, elements, occupancies, _labels, _ = parse_cif_file("examples/CoNb3S6_tripleQ.mcif")
    raw = """
0.333333 0.166700 0.750000 1.000000 2.1772 1.08860 -0.666667
0.833300 0.166633 0.750000 1.000000 -1.08860 1.08860 -0.666667
0.833367 0.666667 0.750000 1.000000 -1.08860 -2.1772 -0.666667
0.166633 0.833300 0.250000 1.000000 -1.08860 1.08860 -0.666667
0.166700 0.333333 0.250000 1.000000 -1.08860 -2.1772 -0.666667
0.666667 0.833367 0.250000 1.000000 2.1772 1.08860 -0.666667
0.666667 0.333333 0.250000 1.000000 0.000000 0.000000 2.000000
0.333333 0.666667 0.750000 1.000000 0.000000 0.000000 2.000000
0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.500000 1.000000 0.000000 0.000000 0.000000
0.500000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000
0.000000 0.500000 0.000000 1.000000 0.000000 0.000000 0.000000
0.500000 0.500000 0.000000 1.000000 0.000000 0.000000 0.000000
0.500000 0.000000 0.500000 1.000000 0.000000 0.000000 0.000000
0.000000 0.500000 0.500000 1.000000 0.000000 0.000000 0.000000
0.500000 0.500000 0.500000 1.000000 0.000000 0.000000 0.000000
0.666667 0.333333 0.994400 1.000000 0.000000 0.000000 0.000000
0.333333 0.666667 0.005600 1.000000 0.000000 0.000000 0.000000
0.333333 0.166700 0.494400 1.000000 0.000000 0.000000 0.000000
0.833300 0.166633 0.494400 1.000000 0.000000 0.000000 0.000000
0.833367 0.666667 0.494400 1.000000 0.000000 0.000000 0.000000
0.166633 0.833300 0.505600 1.000000 0.000000 0.000000 0.000000
0.166700 0.333333 0.505600 1.000000 0.000000 0.000000 0.000000
0.666667 0.833367 0.505600 1.000000 0.000000 0.000000 0.000000
0.333333 0.166700 0.005600 1.000000 0.000000 0.000000 0.000000
0.833300 0.166633 0.005600 1.000000 0.000000 0.000000 0.000000
0.833367 0.666667 0.005600 1.000000 0.000000 0.000000 0.000000
0.166633 0.833300 0.994400 1.000000 0.000000 0.000000 0.000000
0.166700 0.333333 0.994400 1.000000 0.000000 0.000000 0.000000
0.666667 0.833367 0.994400 1.000000 0.000000 0.000000 0.000000
0.666667 0.333333 0.505600 1.000000 0.000000 0.000000 0.000000
0.333333 0.666667 0.494400 1.000000 0.000000 0.000000 0.000000
0.166100 0.000500 0.369400 1.000000 0.000000 0.000000 0.000000
0.999500 0.165600 0.369400 1.000000 0.000000 0.000000 0.000000
0.834400 0.833900 0.369400 1.000000 0.000000 0.000000 0.000000
0.165600 0.999500 0.630600 1.000000 0.000000 0.000000 0.000000
0.000500 0.166100 0.630600 1.000000 0.000000 0.000000 0.000000
0.833900 0.834400 0.630600 1.000000 0.000000 0.000000 0.000000
0.499600 0.165700 0.369400 1.000000 0.000000 0.000000 0.000000
0.834300 0.333900 0.369400 1.000000 0.000000 0.000000 0.000000
0.666100 0.500400 0.369400 1.000000 0.000000 0.000000 0.000000
0.333900 0.834300 0.630600 1.000000 0.000000 0.000000 0.000000
0.165700 0.499600 0.630600 1.000000 0.000000 0.000000 0.000000
0.500400 0.666100 0.630600 1.000000 0.000000 0.000000 0.000000
0.334400 0.833900 0.369400 1.000000 0.000000 0.000000 0.000000
0.166100 0.500500 0.369400 1.000000 0.000000 0.000000 0.000000
0.499500 0.665600 0.369400 1.000000 0.000000 0.000000 0.000000
0.500500 0.166100 0.630600 1.000000 0.000000 0.000000 0.000000
0.833900 0.334400 0.630600 1.000000 0.000000 0.000000 0.000000
0.665600 0.499500 0.630600 1.000000 0.000000 0.000000 0.000000
0.833900 0.999600 0.869400 1.000000 0.000000 0.000000 0.000000
0.000400 0.834300 0.869400 1.000000 0.000000 0.000000 0.000000
0.165700 0.166100 0.869400 1.000000 0.000000 0.000000 0.000000
0.834300 0.000400 0.130600 1.000000 0.000000 0.000000 0.000000
0.999600 0.833900 0.130600 1.000000 0.000000 0.000000 0.000000
0.166100 0.165700 0.130600 1.000000 0.000000 0.000000 0.000000
0.500500 0.834400 0.869400 1.000000 0.000000 0.000000 0.000000
0.165600 0.666100 0.869400 1.000000 0.000000 0.000000 0.000000
0.333900 0.499500 0.869400 1.000000 0.000000 0.000000 0.000000
0.666100 0.165600 0.130600 1.000000 0.000000 0.000000 0.000000
0.834400 0.500500 0.130600 1.000000 0.000000 0.000000 0.000000
0.499500 0.333900 0.130600 1.000000 0.000000 0.000000 0.000000
0.665700 0.166100 0.869400 1.000000 0.000000 0.000000 0.000000
0.833900 0.499600 0.869400 1.000000 0.000000 0.000000 0.000000
0.500400 0.334300 0.869400 1.000000 0.000000 0.000000 0.000000
0.499600 0.833900 0.130600 1.000000 0.000000 0.000000 0.000000
0.166100 0.665700 0.130600 1.000000 0.000000 0.000000 0.000000
0.334300 0.500400 0.130600 1.000000 0.000000 0.000000 0.000000
0.666100 0.000500 0.369400 1.000000 0.000000 0.000000 0.000000
0.999500 0.665600 0.369400 1.000000 0.000000 0.000000 0.000000
0.334400 0.333900 0.369400 1.000000 0.000000 0.000000 0.000000
0.665600 0.999500 0.630600 1.000000 0.000000 0.000000 0.000000
0.000500 0.666100 0.630600 1.000000 0.000000 0.000000 0.000000
0.333900 0.334400 0.630600 1.000000 0.000000 0.000000 0.000000
0.333900 0.999600 0.869400 1.000000 0.000000 0.000000 0.000000
0.000400 0.334300 0.869400 1.000000 0.000000 0.000000 0.000000
0.665700 0.666100 0.869400 1.000000 0.000000 0.000000 0.000000
0.334300 0.000400 0.130600 1.000000 0.000000 0.000000 0.000000
0.999600 0.333900 0.130600 1.000000 0.000000 0.000000 0.000000
0.666100 0.665700 0.130600 1.000000 0.000000 0.000000 0.000000
"""
    rows = [tuple(float(x) for x in line.split()) for line in raw.strip().splitlines()]
    positions = [r[:3] for r in rows]
    occupancies_in = [r[3] for r in rows]
    moments = [r[4:] for r in rows]
    assert all(abs(a - b) < 1e-8 for a, b in zip(occupancies_in, occupancies))
    return lattice, positions, elements, occupancies, moments


def test_changed_basis_conb3s6_tripleq_preserves_msg_after_g0_collapse():
    lattice, positions, elements, occupancies, moments = _changed_basis_conb3s6_tripleq_input()

    result = find_spin_group_from_data(
        "changed_basis_Conb3s6",
        lattice,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert result.index == "182.4.4.2"
    assert result.conf == "Noncoplanar"
    assert result.acc == "6mmP"
    assert result.msg_num == 1257
    assert result.msg_type == 3
    assert result.msg_symbol == "P32'1"
    assert result.msg_acc == "3m1P"
    assert "m_{010}" in result.convention_ssg_international_linear


def test_identify_point_group_recovers_td_for_conbnb3s6_gamma_little_group_spin_part():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)
    gamma_index = ssg.kpoints_symbol_primitive.index("Γ:(0,0,0)")
    little_group = ssg.little_groups[gamma_index]
    spin_part = deduplicate_matrix_pairs([np.array(op[0]) for op in little_group])

    group_symbol, *_ = identify_point_group(spin_part)

    assert group_symbol == "-43m"


def test_classify_moment_configuration_uses_mtol_residual_contract():
    moments = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.019],
        ],
        dtype=float,
    )
    assert _classify_moment_configuration(moments, 0.02) == "Coplanar"
    assert _classify_moment_configuration(moments, 0.006) == "Noncoplanar"


@pytest.mark.parametrize(
    ("path", "low_mtol", "mid_mtol", "high_mtol", "expected_low", "expected_mid", "expected_high"),
    [
        (
            "tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif",
            0.019,
            0.02,
            0.021,
            "Coplanar",
            "Coplanar",
            "Collinear",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif",
            0.019,
            0.02,
            0.021,
            "Coplanar",
            "Coplanar",
            "Collinear",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif",
            0.049,
            0.05,
            0.053,
            "Coplanar",
            "Coplanar",
            "Collinear",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.138_MgV2O4.mcif",
            0.041,
            0.043,
            0.061,
            "Noncoplanar",
            "Coplanar",
            "Collinear",
        ),
    ],
)
def test_real_cases_exhibit_mtol_driven_configuration_boundaries(
    path,
    low_mtol,
    mid_mtol,
    high_mtol,
    expected_low,
    expected_mid,
    expected_high,
):
    primitive_cell = _primitive_magnetic_cell_from_cif(path)
    moments = np.asarray(primitive_cell.moments, dtype=float)
    non_zero = moments[np.linalg.norm(moments, axis=1) > primitive_cell.tol.m_eig]

    assert _classify_moment_configuration(non_zero, low_mtol) == expected_low
    assert _classify_moment_configuration(non_zero, mid_mtol) == expected_mid
    assert _classify_moment_configuration(non_zero, high_mtol) == expected_high


def test_identify_spin_space_group_reports_nonmagnetic_error_when_input_is_effectively_nonmagnetic():
    cell = CrystalCell(
        lattice=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
        positions=[[0.0, 0.0, 0.0]],
        occupancies=[1.0],
        elements=["Fe"],
        moments=[[1e-6, 0.0, 0.0]],
        spin_setting="in_lattice",
    )

    with pytest.raises(ValueError, match=NONMAGNETIC_MTOL_ERROR):
        identify_spin_space_group_result(cell, find_primitive=False)


def test_find_spin_group_reports_clear_error_when_extreme_mtol_blocks_late_stage_degeneracy():
    with pytest.raises(ValueError, match=UNSTABLE_MTOL_ERROR):
        find_spin_group("tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif", mtol=5.0)


def test_find_spin_group_reports_clear_error_when_extreme_mtol_destabilizes_real_case():
    with pytest.raises(ValueError, match=UNSTABLE_MTOL_ERROR):
        find_spin_group("tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif", mtol=5.0)


@pytest.mark.parametrize(
    ("path", "expected_index", "expected_msg_num", "expected_msg_type"),
    [
        ("tests/testset/mcif_241130_no2186/1.138_MgV2O4.mcif", "22.1.2.7", 135, 4),
        ("tests/testset/mcif_241130_no2186/1.207_U2Rh2Sn.mcif", "127.2.2.8", 1152, 4),
        ("tests/testset/mcif_241130_no2186/1.501_Ba2CoO2Cu2S2.mcif", "69.65.2.1.L", 7, 4),
    ],
)
def test_find_spin_group_recovers_post_batch_three_residual_regressions(
    path,
    expected_index,
    expected_msg_num,
    expected_msg_type,
):
    result = find_spin_group(path)

    assert result.index == expected_index
    assert result.msg_num == expected_msg_num
    assert result.msg_type == expected_msg_type


def test_find_spin_group_keeps_ktb3f10_out_of_identity_collapse_sentinel():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1120_KTb3F10.mcif")

    assert result.index == "225.69.1.2"
    assert result.identify_index_details["t_index"] == 6
    assert result.identify_index_details["k_index"] == 1
    assert result.identify_index_details["equivalent_map_index"] == 2
    assert result.identify_index_details["point_group_id"] == 32


def test_find_spin_group_gracefully_degrades_when_identify_database_entry_is_missing():
    with pytest.warns(RuntimeWarning, match="Identify-index output unavailable"):
        result = find_spin_group("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert result.index is None
    assert result.conf == "Coplanar"
    assert result.identify_index_details is None

    metadata = parse_scif_metadata(source_text=result.scif)
    assert metadata["space_group_spin"]["spin_space_group_number_chen"] is None
    assert metadata["space_group_spin"]["spin_space_group_name_chen"] is None
    assert metadata["space_group_spin"]["transform_Chen_Pp_abcs"] is None
    assert metadata["space_group_spin"]["spin_space_group_name_linear"] == (
        result.convention_ssg_international_linear
    )


def test_g_type_output_ossg_uses_shortest_nonzero_axis_translations():
    with pytest.warns(RuntimeWarning, match="Identify-index output unavailable"):
        result = find_spin_group("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert " : (3^{1}_{001},3^{1}_{001},4^{1}_{001})" in result.convention_ssg_international_linear


def test_conbnb3s6_tripleq_g_type_translation_part_keeps_nontrivial_a_b_and_identity_c():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")

    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    symbol = ssg.international_symbol

    assert result.convention_ssg_international_linear.endswith(" : (2_{alpha,beta,0},2_{alpha,beta,0},1)")
    assert symbol["translation_terms_linear"] == ["(2_{010},2_{001},1)"]
    assert symbol["translation_details"][:3] == [
        {"label": "t_a", "vector": (1.0, 0.0, 0.0), "spin_symbol": "2_{010}"},
        {"label": "t_b", "vector": (-5.4235237604366743e-17, 1.0, 0.0), "spin_symbol": "2_{001}"},
        {"label": "t_c", "vector": (0.0, 0.0, 0.0), "spin_symbol": "1"},
    ]


def test_find_spin_group_preserves_historical_identify_index_for_srmnvo4oh():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.165_SrMn(VO4)(OH).mcif")

    assert result.index == "19.4.1.2.P2"
    assert result.identify_index_details["equivalent_map_index"] == 2
    assert result.identify_index_details["configuration_suffix"] == "P2"


def test_find_spin_group_preserves_historical_identify_index_for_ndco2():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.403_NdCo2.mcif")

    assert result.index == "227.227.1.1.L"
    assert result.spin_part_point_group == "∞m"
    assert result.magnetic_phase == "FM/FiM"
    assert result.magnetic_phase_base == "FM/FiM"
    assert result.magnetic_phase_modifier == ""
    assert result.is_spin_orbit_magnet == ""
    assert result.magnetic_phase_details["classification_rule"] == "fm_like_spin_point_group"
    assert result.magnetic_phase_details["fm_like_by_spin_point_group"] is True
    assert result.identify_index_details["equivalent_map_index"] == 1
    assert result.identify_index_details["configuration_suffix"] == "L"


def test_find_spin_group_exposes_compensated_fim_classification_details():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.103_Mn2GeO4.mcif")

    assert result.magnetic_phase == "Compensated FiM"
    assert result.magnetic_phase_base == "Compensated FiM"
    assert result.magnetic_phase_modifier == ""
    assert result.is_spin_orbit_magnet == ""
    assert result.magnetic_phase_details["classification_rule"] == "fm_like_spin_point_group"
    assert result.magnetic_phase_details["zero_net_moment"] is True


def test_find_spin_group_exposes_spin_orbit_magnet_classification_details():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.101_Mn2GeO4.mcif")

    assert result.magnetic_phase == "AFM\n(SOM)"
    assert result.magnetic_phase_base == "AFM"
    assert result.magnetic_phase_modifier == ""
    assert result.is_spin_orbit_magnet == "(SOM)"
    assert result.magnetic_phase_details["classification_rule"] == "afm_with_spin_orbit_magnet"
    assert result.magnetic_phase_details["som_by_mpg"] is True


def test_find_spin_group_exposes_independent_alter_and_spin_orbit_magnet_tags():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1008_Sr2ErRuO6.mcif")

    assert result.magnetic_phase == "AFM(Altermagnet)\n(SOM)"
    assert result.magnetic_phase_base == "AFM"
    assert result.magnetic_phase_modifier == "(Altermagnet)"
    assert result.is_alter == "(Altermagnet)"
    assert result.is_spin_orbit_magnet == "(SOM)"
    assert result.magnetic_phase_details["is_altermagnet"] is True
    assert result.magnetic_phase_details["is_spin_orbit_magnet"] is True


def test_identify_name_generator_matching_uses_pure_translation_cosets():
    target = np.array([0.0, 0.0, 0.0], dtype=float)
    candidate = np.array([0.0006, 0.4994, 0.5], dtype=float)
    pure_translations = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.5, 0.0, 0.5], dtype=float),
        np.array([0.0, 0.5, 0.5], dtype=float),
        np.array([0.5, 0.5, 0.0], dtype=float),
    ]

    assert not _translations_equivalent_mod_pure_translations(
        target,
        candidate,
        [np.zeros(3)],
        tol=1e-3,
    )
    assert _translations_equivalent_mod_pure_translations(
        target,
        candidate,
        pure_translations,
        tol=1e-3,
    )


def test_find_spin_group_recovers_msg_little_group_symbols_after_translation_cleanup():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.118_Ba5Co5ClO13.mcif")

    assert result.msg_little_group_symbols
    assert "Unknown" not in set(result.msg_little_group_symbols)


def test_tensor_ops_do_not_depend_on_international_symbol_generation(monkeypatch):
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.1114_CeAlSi.mcif"
    )
    cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    ssg = identify_spin_space_group(cell, find_primitive=False)
    acc_cell = cell.transform(ssg.acc_primitive_trans, ssg.acc_primitive_origin_shift)
    acc_ssg = ssg.transform(ssg.acc_primitive_trans, ssg.acc_primitive_origin_shift)

    def _boom(self, tol=1e-4):
        raise AssertionError("symbol generation should be decoupled from tensor ops")

    monkeypatch.setattr(SpinSpaceGroup, "get_international_symbol", _boom)

    ops_wo_soc = _tensor_ops_wo_soc(acc_ssg, acc_cell)

    assert ops_wo_soc
    assert len(ops_wo_soc) == len(acc_ssg.gspg_ops_raw)


def test_get_magnetic_space_group_from_operations_handles_noisy_fractional_translations():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.403_NdCo2.mcif"
    )
    cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    ssg = identify_spin_space_group(cell, find_primitive=False)
    acc_ssg = ssg.transform(ssg.acc_primitive_trans, ssg.acc_primitive_origin_shift)
    primitive_msg_ops = _primitive_msg_ops_from_ssg(acc_ssg.ops, tol=0.01)
    little_group = _get_magnetic_little_group(
        acc_ssg.kpoints_primitive[0],
        primitive_msg_ops,
        tol=0.01,
    )
    msg_info = get_magnetic_space_group_from_operations(little_group)

    assert msg_info is not None
    assert msg_info["mpg_symbol"] == "-1'"


@pytest.mark.parametrize(
    "source_path",
    [
        "tests/testset/mcif_241130_no2186/0.1114_CeAlSi.mcif",
        "tests/testset/mcif_241130_no2186/0.37_U3Al2Si3.mcif",
        "tests/testset/mcif_241130_no2186/2.115_Er2CuMn5O12.mcif",
    ],
)
def test_find_spin_group_handles_near_involution_seitz_symbol_noise(source_path):
    result = find_spin_group(source_path)

    assert result.primitive_magnetic_cell_ssg_seitz


def test_find_spin_group_stably_handles_repeated_near_involution_symbol_runs():
    path = "tests/testset/mcif_241130_no2186/0.1114_CeAlSi.mcif"

    for _ in range(8):
        result = find_spin_group(path)
        assert result.index == "109.44.1.2.P2"
        assert result.primitive_magnetic_cell_ssg_seitz


def test_identify_index_transform_can_reach_database_standard_generators():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    debug = _build_identify_standardization_debug(result)

    assert result.index == "38.6.1.2.P2"
    assert debug["reduce_info"]
    assert debug["standard_generator_maps"]
    assert len(debug["standard_generator_maps"]) == len(
        debug["database_standard_generators"]
    )

    for generated, database in zip(
        debug["standard_generator_maps"],
        debug["database_standard_generators"],
    ):
        assert generated["space"].shape == (4, 4)
        assert database.shape == (4, 4)
        assert is_matrix_equal(generated["space"], database, tol=0.001)


def test_find_spin_group_exposes_tensor_outputs():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.tensor_outputs
    assert result.BCDTensor is not None
    assert result.MSGBCDTensor is not None
    assert result.QMDTensor is not None
    assert result.MSGQMDTensor is not None
    assert result.IMDTensor is not None
    assert result.MSGIMDTensor is not None
    assert result.AHE_woSOC is not None
    assert result.AHE_wSOC is not None

    assert result.BCDTensor["free_parameters"] == 2
    assert result.MSGBCDTensor["free_parameters"] == 2
    assert len(result.BCDTensor["relations"]) == 2
    assert len(result.MSGBCDTensor["relations"]) == 2
    assert result.AHE_woSOC["is_zero"] is True
    assert result.AHE_wSOC["free_parameters"] == 1
    assert result.AHE_wSOC["is_zero"] is False
    assert result.MSGQMDTensor["free_parameters"] == 3
    assert result.MSGIMDTensor["free_parameters"] == 1


def test_crse_w_soc_tensor_inputs_match_legacy_magnetic_point_group_behavior():
    result = find_spin_group("tests/testset/mcif_241130_no2186/2.35_CrSe.mcif")

    assert result.index == "194.149.3.3"
    assert result.conf == "Noncoplanar"
    assert result.AHE_wSOC["free_parameters"] == 1
    assert result.MSGBCDTensor["free_parameters"] == 1
    assert result.MSGQMDTensor["free_parameters"] == 2
    assert result.MSGIMDTensor["free_parameters"] == 1


@pytest.mark.parametrize(
    ("source_path", "expected_index"),
    [
        ("tests/testset/mcif_241130_no2186/0.435_Pb5Fe3TiO11Cl.mcif", "123.129.2.1.L"),
        ("tests/testset/mcif_241130_no2186/1.234_Ca2Sr2IrO6.mcif", "2.2.2.2.P1"),
        ("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif", "148.2.2.3"),
    ],
)
def test_find_spin_group_preserves_identify_indices_when_translation_representatives_require_integer_shifts(
    source_path,
    expected_index,
):
    result = find_spin_group(source_path)

    assert result.index == expected_index


def test_find_spin_group_prefers_nontrivial_translation_generators_for_identify_when_available():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.435_Pb5Fe3TiO11Cl.mcif")
    translation_spins = [
        np.asarray(item[0], dtype=float)
        for item in result.identify_index_details["translation_maps"]
    ]

    assert np.allclose(translation_spins[0], -np.eye(3), atol=1e-6)
    assert np.allclose(translation_spins[1], -np.eye(3), atol=1e-6)
    assert np.allclose(translation_spins[2], np.eye(3), atol=1e-6)


@pytest.mark.parametrize(
    "source_path",
    [
        "examples/0.800_MnTe.mcif",
        "tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif",
        "tests/testset/mcif_241130_no2186/1.274_DyFeWO6.mcif",
    ],
)
def test_identify_translation_maps_keep_exact_nofrac_basis_translations(source_path):
    result = find_spin_group(source_path)
    translations = _identify_translation_vectors(result.identify_index_details)

    assert len(translations) == 3
    assert np.allclose(translations[0], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(translations[1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(translations[2], np.array([0.0, 0.0, 1.0]), atol=1e-6)


@pytest.mark.parametrize(
    ("source_path", "expected_index"),
    [
        ("tests/testset/mcif_241130_no2186/1.274_DyFeWO6.mcif", "7.1.2.19"),
        ("tests/testset/mcif_241130_no2186/1.828_ZnFe2O4.mcif", "115.1.2.6"),
        ("tests/testset/mcif_241130_no2186/2.108_Tb3NbO7.mcif", "17.1.2.6"),
        ("tests/testset/mcif_241130_no2186/2.75_Sr2Fe3S2O3.mcif", "10.11.4.6"),
        ("tests/testset/mcif_241130_no2186/3.19_CoO.mcif", "134.2.2.2"),
        ("tests/testset/mcif_241130_no2186/3.4_MgCr2O4.mcif", "119.3.4.14"),
    ],
)
def test_find_spin_group_matches_historical_noncoplanar_identify_indices_with_exact_basis_lifts(
    source_path,
    expected_index,
):
    result = find_spin_group(source_path)

    assert result.conf == "Noncoplanar"
    assert result.index == expected_index


def test_find_spin_group_from_data_matches_file_based_flow():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "examples/0.800_MnTe.mcif"
    )
    result = find_spin_group_from_data(
        "0.800_MnTe.mcif",
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert result.index == "194.164.1.1.L"
    assert result.conf == "Collinear"
    assert result.primitive_magnetic_cell_ssg_type == "t"


def test_find_spin_group_exposes_input_space_group_metadata_from_identified_magnetic_primitive():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.11_DyFeO3.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)
    payload = serialize_data(result.to_dict())

    assert result.input_space_group_number == 62
    assert result.input_space_group_symbol == "Pnma"
    assert result.input_space_group_number != ssg.international_symbol["sg_number"]
    assert result.input_space_group_symbol != ssg.international_symbol["sg_symbol"]
    assert payload["input_space_group_number"] == 62
    assert payload["input_space_group_symbol"] == "Pnma"


def test_identify_spin_space_group_result_keeps_input_space_group_context_off_ssg():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.11_DyFeO3.mcif"
    )
    cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    identify_result = identify_spin_space_group_result(cell, find_primitive=False)

    assert identify_result.input_space_group is not None
    assert identify_result.input_space_group.number == 62
    assert identify_result.input_space_group.symbol == "Pnma"
    assert not hasattr(identify_result.ssg, "input_space_group_number")


def test_find_spin_group_exposes_input_space_group_metadata_for_type_k_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    assert result.primitive_magnetic_cell_ssg_type == "k"
    assert result.input_space_group_number == 55
    assert result.input_space_group_symbol == "Pbam"
    assert result.input_space_group_number != ssg.international_symbol["sg_number"]


def test_find_spin_group_from_data_preserves_input_space_group_metadata():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.11_DyFeO3.mcif"
    )
    result = find_spin_group_from_data(
        "0.11_DyFeO3.mcif",
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert result.input_space_group_number == 62
    assert result.input_space_group_symbol == "Pnma"


def test_find_spin_group_exposes_source_structure_metadata_from_mcif():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert result.source_parent_space_group["name_H_M_alt"] == "I m -3"
    assert result.source_parent_space_group["IT_number"] == 204
    assert result.source_parent_space_group["transform_Pp_abc"] == "a,b,c;0,0,0"
    assert result.source_parent_space_group["child_transform_Pp_abc"] == "2a,2b,2c;0,0,0"
    assert result.source_cell_parameter_strings["_cell_length_a"] == "14.88540"
    assert result.source_cell_parameter_strings["_cell_angle_alpha"] == "90.00000"


def test_find_spin_group_exposes_standard_setting_payloads_for_web_app():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.g0_standard_cell is not None
    assert result.l0_standard_cell is not None
    assert result.g0_standard_ssg_ops
    assert result.l0_standard_ssg_ops
    assert np.asarray(result.T_input_to_G0std[0], dtype=float).shape == (3, 3)
    assert np.asarray(result.T_input_to_G0std[1], dtype=float).shape == (3,)
    assert np.asarray(result.T_G0std_to_primitive[0], dtype=float).shape == (3, 3)
    assert np.asarray(result.T_G0std_to_primitive[1], dtype=float).shape == (3,)
    assert np.asarray(result.T_input_to_L0std[0], dtype=float).shape == (3, 3)
    assert np.asarray(result.T_input_to_L0std[1], dtype=float).shape == (3,)
    assert np.asarray(result.T_L0std_to_primitive[0], dtype=float).shape == (3, 3)
    assert np.asarray(result.T_L0std_to_primitive[1], dtype=float).shape == (3,)
    assert result.T_G0std_to_acc_primitive == result.T_G0std_to_primitive
    assert result.T_L0std_to_acc_primitive == result.T_L0std_to_primitive


@pytest.mark.parametrize(
    (
        "path",
        "expected_index",
        "expected_convention_setting",
        "expected_acc",
        "expected_msg_num",
        "expected_msg_symbol",
        "expected_label",
        "expected_is_self_automorphism",
        "expected_convention_count",
        "expected_acc_conventional_count",
        "expected_acc_primitive_count",
    ),
    [
        (
            "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
            "148.2.4.1",
            "G0std",
            "-3R",
            1247,
            "R-3",
            "setting_change",
            False,
            240,
            240,
            80,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.456_RbFeO2.mcif",
            "227.216.1.1.L",
            "G0std",
            "m-3mF",
            1216,
            "I4_1'/a'm'd",
            "self_automorphism",
            True,
            400,
            400,
            100,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.458_CsFeO2.mcif",
            "227.216.1.1.L",
            "G0std",
            "m-3mF",
            1216,
            "I4_1'/a'm'd",
            "self_automorphism",
            True,
            400,
            400,
            100,
        ),
        (
            "examples/CoNb3S6_tripleQ.mcif",
            "182.4.4.2",
            "G0std",
            "6mmP",
            1257,
            "P32'1",
            "self_automorphism",
            True,
            80,
            80,
            80,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.712_VNb3S6.mcif",
            "182.149.1.1.L",
            "G0std",
            "6/mmmP",
            131,
            "C2'2'2_1",
            "self_automorphism",
            True,
            20,
            20,
            20,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif",
            "29.4.1.1.L",
            "G0std",
            "mmmP",
            200,
            "Pca'2_1'",
            "setting_change",
            False,
            36,
            36,
            36,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.2_Cd2Os2O7.mcif",
            "227.2.1.2",
            "G0std",
            "m-3mF",
            1633,
            "Fd-3m'",
            "setting_change",
            False,
            88,
            88,
            22,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
            "6.6.2.4.L",
            "L0std",
            "2/mP",
            35,
            "P_cc",
            "self_automorphism",
            True,
            64,
            64,
            64,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.347_CuFeO2.mcif",
            "13.15.2.1.L",
            "L0std",
            "2/mC",
            98,
            "C_a2/c",
            "setting_change",
            False,
            32,
            32,
            16,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.348_CuFeO2.mcif",
            "12.12.2.1.L",
            "L0std",
            "2/mC",
            97,
            "C_c2/c",
            "self_automorphism",
            True,
            80,
            80,
            40,
        ),
    ],
)
def test_find_spin_group_exposes_convention_to_acc_conventional_chain_for_representative_cases(
    path,
    expected_index,
    expected_convention_setting,
    expected_acc,
    expected_msg_num,
    expected_msg_symbol,
    expected_label,
    expected_is_self_automorphism,
    expected_convention_count,
    expected_acc_conventional_count,
    expected_acc_primitive_count,
):
    result = find_spin_group(path)

    assert result.index == expected_index
    assert result.convention_ssg_setting == expected_convention_setting
    assert result.acc == expected_acc
    assert result.msg_num == expected_msg_num
    assert result.msg_symbol == expected_msg_symbol

    assert result.acc_conventional_cell_setting == "acc_conventional"
    assert result.acc_conventional_ssg_setting == "acc_conventional"
    assert result.selected_standard_setting == expected_convention_setting
    assert result.T_selected_standard_to_acc_conventional_label == expected_label
    assert (
        result.T_selected_standard_to_acc_conventional_is_self_automorphism
        is expected_is_self_automorphism
    )

    assert len(result.convention_cell_detail["positions"]) == expected_convention_count
    assert len(result.acc_conventional_cell_detail["positions"]) == expected_acc_conventional_count
    assert len(result.acc_primitive_magnetic_cell_detail["positions"]) == expected_acc_primitive_count

    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[0], dtype=float),
        np.asarray(result.T_selected_standard_to_acc_conventional[0], dtype=float),
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[1], dtype=float),
        np.asarray(result.T_selected_standard_to_acc_conventional[1], dtype=float),
        atol=1e-8,
    )
    assert np.asarray(result.T_convention_to_acc_conventional[0], dtype=float).shape == (3, 3)
    assert np.asarray(result.T_convention_to_acc_conventional[1], dtype=float).shape == (3,)


@pytest.mark.parametrize(
    ("path", "expected_matrix", "expected_shift", "expected_label", "expected_setting"),
    [
        (
            "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
            np.array([[-1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
            np.zeros(3),
            "setting_change",
            "G0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.456_RbFeO2.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "G0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "L0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.347_CuFeO2.mcif",
            np.eye(3),
            np.array([0.5, 0.125, 0.25]),
            "setting_change",
            "L0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.348_CuFeO2.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "L0std",
        ),
    ],
)
def test_find_spin_group_exposes_expected_representative_convention_to_acc_conventional_transforms(
    path,
    expected_matrix,
    expected_shift,
    expected_label,
    expected_setting,
):
    result = find_spin_group(path)

    assert result.convention_ssg_setting == expected_setting
    assert result.selected_standard_setting == expected_setting
    assert result.T_selected_standard_to_acc_conventional_label == expected_label
    assert np.allclose(
        np.asarray(result.T_selected_standard_to_acc_conventional[0], dtype=float),
        expected_matrix,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_selected_standard_to_acc_conventional[1], dtype=float),
        expected_shift,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[0], dtype=float),
        expected_matrix,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[1], dtype=float),
        expected_shift,
        atol=1e-8,
    )


def test_current_basis_symbol_builder_transports_r_centering_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    payload = build_international_symbol(SpinSpaceGroup(result.g0_standard_ssg_ops), basis_mode="current")

    centering_vectors = {
        item["label"]: np.asarray(item["vector"], dtype=float)
        for item in payload["translation_details"]
        if item["label"].startswith("b_")
    }

    assert np.allclose(centering_vectors["b_1"], np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0]), atol=1e-6)
    assert np.allclose(centering_vectors["b_2"], np.array([1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0]), atol=1e-6)


def test_current_basis_symbol_builder_keeps_p_translation_targets_for_conb3s6():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    payload = build_international_symbol(SpinSpaceGroup(result.g0_standard_ssg_ops), basis_mode="current")

    translation_vectors = {
        item["label"]: np.asarray(item["vector"], dtype=float)
        for item in payload["translation_details"]
        if item["label"].startswith("t_")
    }

    assert np.allclose(translation_vectors["t_a"], np.array([0.5, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(translation_vectors["t_b"], np.array([0.0, 0.5, 0.0]), atol=1e-6)
    assert np.allclose(translation_vectors["t_c"], np.array([0.0, 0.0, 0.0]), atol=1e-6)


def _current_basis_symbol_context(result):
    if result.convention_ssg_setting == "G0std":
        ops = result.g0_standard_ssg_ops
        ssg = SpinSpaceGroup(ops)
        sg_num = int(ssg.G0_num)
        bravais = ssg.G0_symbol[0]
        current_to_standard, current_to_standard_shift = _compose_symbol_setting_transform(
            np.asarray(ssg.transformation_to_G0std, dtype=float),
            np.asarray(ssg.origin_shift_to_G0std, dtype=float),
            np.asarray(ssg.transformation_to_G0std_id, dtype=float),
            np.asarray(ssg.origin_shift_to_G0std_id, dtype=float),
        )
    else:
        ops = result.l0_standard_ssg_ops
        ssg = SpinSpaceGroup(ops)
        sg_num = int(ssg.L0_num)
        bravais = ssg.L0_symbol[0]
        current_to_standard = np.asarray(ssg.transformation_to_L0std, dtype=float)
        current_to_standard_shift = np.asarray(ssg.origin_shift_to_L0std, dtype=float)

    named_ops, centering_trans = _parse_sg_generator_ops(sg_num)
    if not centering_trans:
        centering_trans = [vec for _, vec in _default_centering_vectors(bravais)]
    named_ops_cur, centering_cur = _transport_standard_generators_to_current_basis(
        named_ops,
        centering_trans,
        current_to_standard,
        current_to_standard_shift,
    )
    return ssg, named_ops_cur, centering_cur


@pytest.mark.parametrize(
    "path",
    [
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
        "examples/CoNb3S6_tripleQ.mcif",
        "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
        "tests/testset/mcif_241130_no2186/1.347_CuFeO2.mcif",
        "tests/testset/mcif_241130_no2186/1.348_CuFeO2.mcif",
    ],
)
def test_current_basis_symbol_builder_matches_standard_named_generators_after_transport(path):
    result = find_spin_group(path)
    ssg, named_ops_cur, _ = _current_basis_symbol_context(result)

    for rotation, translation in named_ops_cur:
        assert _find_real_operation(ssg.nssg, rotation, translation, tol=1e-4) is not None


@pytest.mark.parametrize(
    "path",
    [
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
        "examples/CoNb3S6_tripleQ.mcif",
    ],
)
def test_current_basis_symbol_builder_matches_required_translation_targets_after_transport(path):
    result = find_spin_group(path)
    ssg, _, centering_cur = _current_basis_symbol_context(result)

    for axis in range(3):
        assert (
            _select_preferred_primitive_translation_match(
                ssg.nssg,
                axis,
                tol=1e-4,
                identity_real_ops=ssg.identity_real_nssg_ops,
            )
            is not None
        )

    for vec in centering_cur:
        assert (
            _select_preferred_translation_match(
                ssg.nssg,
                vec,
                tol=1e-4,
                identity_real_ops=ssg.identity_real_nssg_ops,
            )
            is not None
        )


@pytest.mark.parametrize(
    ("path", "expected_type", "expected_setting"),
    [
        ("examples/0.800_MnTe.mcif", "t", "G0std"),
        ("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif", "k", "L0std"),
        ("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif", "g", "G0std"),
    ],
)
def test_find_spin_group_exposes_convention_selected_standard_payloads(
    path,
    expected_type,
    expected_setting,
):
    result = find_spin_group(path)
    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)

    assert result.primitive_magnetic_cell_ssg_type == expected_type
    assert result.convention_cell_setting == expected_setting
    assert result.convention_ssg_setting == expected_setting
    assert result.selected_standard_setting == expected_setting
    assert result.convention_ssg_spin_frame_setting == "ossg_oriented_spin_frame"

    assert result.convention_ssg_seitz == public_ossg.seitz_symbols
    assert result.convention_ssg_seitz_latex == public_ossg.seitz_symbols_latex
    assert (
        result.convention_ssg_international_linear
        == public_ossg.international_symbol_linear_current_frame
    )
    assert (
        result.convention_ssg_international_latex
        == public_ossg.international_symbol_latex_current_frame
    )


def test_input_to_standard_transforms_remain_nontrivial_for_1048():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif")

    g0_forward = np.asarray(result.T_input_to_G0std[0], dtype=float)
    g0_shift = np.asarray(result.T_input_to_G0std[1], dtype=float)
    l0_forward = np.asarray(result.T_input_to_L0std[0], dtype=float)
    l0_shift = np.asarray(result.T_input_to_L0std[1], dtype=float)

    assert not np.allclose(g0_forward, np.eye(3), atol=1e-8)
    assert not np.allclose(l0_forward, np.eye(3), atol=1e-8)
    assert not np.allclose(g0_shift, np.zeros(3), atol=1e-8)
    assert not np.allclose(l0_shift, np.zeros(3), atol=1e-8)


def test_input_to_standard_transforms_remain_nontrivial_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    g0_forward = np.asarray(result.T_input_to_G0std[0], dtype=float)
    g0_shift = np.asarray(result.T_input_to_G0std[1], dtype=float)
    l0_forward = np.asarray(result.T_input_to_L0std[0], dtype=float)
    l0_shift = np.asarray(result.T_input_to_L0std[1], dtype=float)

    assert not np.allclose(g0_forward, np.eye(3), atol=1e-8)
    assert not np.allclose(l0_forward, np.eye(3), atol=1e-8)
    assert np.allclose(
        g0_forward,
        np.array(
            [
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
                [-4.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0],
            ]
        ),
        atol=1e-8,
    )
    assert np.allclose(g0_shift, np.zeros(3), atol=1e-8)
    assert np.allclose(l0_shift, np.zeros(3), atol=1e-8)


def test_find_spin_group_exposes_acc_primitive_aliases_and_setting_tags():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.primitive_magnetic_cell_setting == "acc_primitive"
    assert result.acc_primitive_magnetic_cell_setting == "acc_primitive"
    assert result.primitive_magnetic_cell == result.acc_primitive_magnetic_cell
    assert result.primitive_magnetic_cell_detail == result.acc_primitive_magnetic_cell_detail
    assert result.primitive_magnetic_cell_poscar == result.acc_primitive_magnetic_cell_poscar

    assert result.primitive_magnetic_cell_ssg_setting == "acc_primitive"
    assert result.acc_primitive_ssg_setting == "acc_primitive"
    assert result.primitive_magnetic_cell_ssg_ops == result.acc_primitive_ssg_ops
    assert result.primitive_magnetic_cell_ssg_seitz == result.acc_primitive_ssg_seitz
    assert (
        result.primitive_magnetic_cell_ssg_international_linear
        == result.acc_primitive_ssg_international_linear
    )
    assert (
        result.primitive_magnetic_cell_ssg_international_latex
        == result.acc_primitive_ssg_international_latex
    )

    assert result.KPOINTS_setting == "acc_primitive"
    assert result.KPOINTS_real_space_setting == "acc_primitive"
    assert result.spin_polarizations_setting == "acc_primitive_poscar_spin_frame"
    assert result.spin_polarizations_real_space_setting == "acc_primitive"
    assert result.spin_polarizations_spin_frame == "acc_primitive_poscar_spin_frame"
    assert result.spin_polarizations_acc_cartesian_setting == "acc_primitive_cartesian"
    assert result.spin_polarizations == result.spin_polarizations_acc_poscar_spin_frame
    assert result.spin_polarizations_acc_poscar_spin_frame_setting == "acc_primitive_poscar_spin_frame"
    assert result.spin_polarizations_acc_poscar_spin_frame is not None
    assert (
        result.real_cartesian_to_spin_frame
        == result.acc_primitive_real_cartesian_to_poscar_spin_frame
    )
    assert (
        result.spin_frame_to_real_cartesian
        == result.poscar_spin_frame_to_acc_primitive_real_cartesian
    )


def test_find_spin_group_exposes_input_and_public_magnetic_primitive_layers():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert result.input_magnetic_primitive_cell_setting == "input_magnetic_primitive"
    assert result.magnetic_primitive_cell_setting == "acc_primitive"
    assert result.acc_primitive_magnetic_cell_setting == "acc_primitive"

    assert result.magnetic_primitive_cell == result.acc_primitive_magnetic_cell
    assert result.magnetic_primitive_cell_detail == result.acc_primitive_magnetic_cell_detail
    assert result.magnetic_primitive_cell_poscar == result.acc_primitive_magnetic_cell_poscar

    assert result.primitive_magnetic_cell == result.magnetic_primitive_cell
    assert result.primitive_magnetic_cell_detail == result.magnetic_primitive_cell_detail
    assert result.primitive_magnetic_cell_poscar == result.magnetic_primitive_cell_poscar

    assert result.input_magnetic_primitive_cell_detail != result.magnetic_primitive_cell_detail
    assert result.input_magnetic_primitive_cell_poscar != result.magnetic_primitive_cell_poscar

    assert result.magnetic_primitive_ssg_ops == result.acc_primitive_ssg_ops
    assert result.magnetic_primitive_ssg_international_linear == result.acc_primitive_ssg_international_linear
    assert result.primitive_magnetic_cell_ssg_ops == result.magnetic_primitive_ssg_ops


def test_scif_transform_tags_use_basis_relation_contract():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    default_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_G0STD_ORIENTED)
    assert (
        "_space_group_spin.fsg_transform_to_input_Pp  "
        "'2/3a+1/3b-4/3c,-1/3a-2/3b-4/3c,-1/3a+1/3b-4/3c;0,0,0'"
    ) in default_scif
    assert "_space_group_spin.fsg_transform_to_G0std_Pp  'a,b,c;0,0,0'" in default_scif
    assert '_space_group_spin.fsg_spin_arithmetic_crystal_class_symbol  "-3R"' in default_scif
    assert '_space_group_spin.fsg_magnetic_arithmetic_crystal_class_symbol  "-3R"' in default_scif

    input_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_INPUT)
    assert "_space_group_spin.fsg_transform_to_input_Pp  'a,b,c;0,0,0'" in input_scif

    magnetic_primitive_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_MAGNETIC_PRIMITIVE)
    assert (
        "_space_group_spin.fsg_transform_to_magnetic_primitive_Pp  'a,b,c;0,0,0'"
        in magnetic_primitive_scif
    )

    assert result.primitive_msg_ops_setting == "acc_primitive"
    assert result.acc_primitive_msg_ops_setting == "acc_primitive"
    assert result.primitive_msg_ops == result.acc_primitive_msg_ops
    assert result.msg_spin_polarizations_setting == "acc_primitive_poscar_spin_frame"
    assert result.msg_spin_polarizations_real_space_setting == "acc_primitive"
    assert result.msg_spin_polarizations_spin_frame == "acc_primitive_poscar_spin_frame"
    assert result.msg_spin_polarizations_acc_cartesian_setting == "acc_primitive_cartesian"
    assert result.msg_spin_polarizations == result.msg_spin_polarizations_acc_poscar_spin_frame
    assert result.symbol_calibration_tol == result.acc_primitive_ssg_symbol_calibration_tol
    assert result.convention_ssg_symbol_calibration_tol is not None
    assert result.primitive_magnetic_cell_ssg_seitz_descriptions
    assert result.acc_primitive_ssg_seitz_descriptions
    assert result.g0_standard_ssg_seitz_descriptions
    assert result.l0_standard_ssg_seitz_descriptions
    assert result.convention_ssg_seitz_descriptions


def test_find_spin_group_exposes_msg_acc_for_conb3s6_tripleq():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")

    assert result.acc == "6mmP"
    assert result.msg_acc == "3m1P"


def test_find_spin_group_exposes_explicit_gspg_payload_for_coplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    oriented_ssg = SpinSpaceGroup(result.convention_ssg_ops)

    assert result.gspg_output_mode == "explicit_ops"
    assert repr(oriented_ssg.gspg) == result.gspg_symbol_linear
    assert result.gspg_spin_only_symbol_hm == oriented_ssg.gspg.spin_only_symbol_hm
    assert result.gspg_spin_only_symbol_s == oriented_ssg.gspg.spin_only_symbol_s
    assert result.gspg_ops == _serialize_gspg_pairs(oriented_ssg.gspg.ops)
    assert result.gspg_raw_ops == _serialize_gspg_pairs(oriented_ssg.gspg.ops)
    assert result.gspg_public_ops_are_reduced is False
    assert result.gspg_real_space_setting == "G0std"
    assert result.gspg_spin_frame_setting == "ossg_oriented_spin_frame"
    assert result.gspg_symbol_calibration_tol == oriented_ssg.symbol_calibration_tol
    assert result.gspg_effective_mpg_ops == _serialize_effective_mpg_ops(
        oriented_ssg.gspg.effective_magnetic_point_group
    )
    assert result.gspg_effective_mpg_symbol == oriented_ssg.gspg.empg_symbol
    assert result.gspg_detail == oriented_ssg.gspg.to_dict()
    assert result.gspg_effective_k_point_group_ops == _serialize_rotation_ops(oriented_ssg.ekPG)
    assert result.gspg_point_part_linear == oriented_ssg.gspg.point_part_linear
    assert result.gspg_spin_only_part_linear == oriented_ssg.gspg.spin_only_linear
    assert result.gspg_symbol_linear == "1|m 2_{001}|m 2_{001}|2 m|1"
    assert result.gspg_symbol_mode == "npg_x_spin_only"
    assert result.gspg_npg_symbol_hm == oriented_ssg.n_spin_part_point_group_symbol_hm
    assert result.gspg_npg_symbol_s == oriented_ssg.n_spin_part_point_group_symbol_s
    assert result.gspg_spin_only_component_symbol_hm == "m"
    assert result.gspg_spin_only_component_symbol_s == "Cs"
    assert result.gspg_symbol_tentative_hm == "2 x m"
    assert result.gspg_symbol_tentative_s == "C2 x Cs"


def test_find_spin_group_reports_collinear_gspg_as_nssg_times_spin_only():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    oriented_ssg = SpinSpaceGroup(result.convention_ssg_ops)
    expected_nssg_point_ops = deduplicate_matrix_pairs(
        [[op[0], op[1]] for op in oriented_ssg.nssg],
        tol=oriented_ssg.tol,
    )

    assert result.conf == "Collinear"
    assert result.gspg_output_mode == "reduced_point_part_with_spin_only_annotation"
    assert repr(oriented_ssg.gspg) == result.gspg_symbol_linear
    assert result.gspg_spin_only_symbol_hm == "∞m"
    assert result.gspg_spin_only_symbol_s == "C∞v"
    assert result.gspg_ops == _serialize_gspg_pairs(expected_nssg_point_ops)
    assert result.gspg_raw_ops == _serialize_gspg_pairs(oriented_ssg.gspg.ops)
    assert result.gspg_public_ops_are_reduced is True
    assert result.gspg_real_space_setting == "G0std"
    assert result.gspg_spin_frame_setting == "ossg_oriented_spin_frame"
    assert len(result.gspg_ops) < len(oriented_ssg.gspg.ops)
    assert result.gspg_effective_mpg_symbol == oriented_ssg.gspg.empg_symbol
    assert result.gspg_point_part_linear == oriented_ssg.gspg.point_part_linear
    assert result.gspg_spin_only_part_linear == oriented_ssg.gspg.spin_only_linear
    assert result.gspg_symbol_linear == "-1|6_{3}/ -1|m 1|m -1|m ∞_{110}m|1"
    assert result.gspg_symbol_mode == "npg_x_spin_only"
    assert result.gspg_npg_symbol_hm == "-1"
    assert result.gspg_npg_symbol_s == "Ci"
    assert result.gspg_spin_only_component_symbol_hm == "∞m"
    assert result.gspg_spin_only_component_symbol_s == "C∞v"
    assert result.gspg_symbol_tentative_hm == "-1 x ∞m"
    assert result.gspg_symbol_tentative_s == "Ci x C∞v"
    assert result.gspg_detail == oriented_ssg.gspg.to_dict()


@pytest.mark.parametrize(
    ("path", "expected_type"),
    [
        ("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif", "k"),
        ("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif", "g"),
    ],
)
def test_find_spin_group_uses_componentized_gspg_symbol_fallback_for_type_k_and_g(path, expected_type):
    result = find_spin_group(path)

    assert result.primitive_magnetic_cell_ssg_type == expected_type
    assert result.gspg_symbol_mode == "point_part_and_spin_only"
    assert result.gspg_symbol_linear is not None
    assert result.gspg_point_part_linear is not None
    assert result.gspg_spin_only_part_linear is not None
    assert result.gspg_npg_symbol_hm is not None
    assert result.gspg_npg_symbol_s is not None
    assert result.gspg_spin_only_component_symbol_hm is not None
    assert result.gspg_spin_only_component_symbol_s is not None
    assert result.gspg_symbol_tentative_hm is None
    assert result.gspg_symbol_tentative_s is None


def test_find_spin_group_uses_gspg_r_eq_i_spin_only_for_collinear_type_k_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif")

    assert result.conf == "Collinear"
    assert result.gspg_real_space_setting == "L0std"
    assert result.gspg_spin_only_symbol_hm == "∞/mm"
    assert result.gspg_spin_only_symbol_s == "D∞h"
    assert result.gspg_point_part_linear == "1|m"
    assert result.gspg_spin_only_part_linear == "∞_{001}/mm|1"
    assert result.gspg_symbol_linear == "1|m ∞_{001}/mm|1"


def test_find_spin_group_uses_oriented_path_for_public_type_g_gspg_symbol():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif")

    assert result.gspg_real_space_setting == "G0std"
    assert result.gspg_spin_frame_setting == "ossg_oriented_spin_frame"
    assert result.gspg_point_part_linear == "3^{1}_{001}|-3"
    assert result.gspg_spin_only_part_linear == "-1|1"
    assert result.gspg_symbol_linear == "3^{1}_{001}|-3 -1|1"


def test_find_spin_group_public_gspg_is_derived_from_public_ossg():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif")
    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)

    assert result.convention_ssg_international_linear == public_ossg.international_symbol_linear_current_frame
    assert result.convention_ssg_international_latex == public_ossg.international_symbol_latex_current_frame
    assert result.gspg_symbol_linear == public_ossg.gspg.symbol_linear
    assert result.gspg_point_part_linear == public_ossg.gspg.point_part_linear
    assert result.gspg_spin_only_part_linear == public_ossg.gspg.spin_only_linear


def test_find_spin_group_exposes_poscar_spin_frame_transform_and_polarizations():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    forward = np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float)
    backward = np.asarray(result.poscar_spin_frame_to_acc_primitive_real_cartesian, dtype=float)

    assert forward.shape == (3, 3)
    assert backward.shape == (3, 3)
    assert np.allclose(forward @ backward, np.eye(3), atol=1e-8)
    assert np.allclose(backward @ forward, np.eye(3), atol=1e-8)

    expected_forward = standardize_lattice(
        np.asarray(result.acc_primitive_magnetic_cell_detail["lattice"], dtype=float)
    )[1]
    assert np.allclose(forward, expected_forward, atol=1e-8)

    expected_poscar_spin_polarizations = SpinSpaceGroup(result.acc_primitive_ssg_ops).transform_spin(
        forward
    ).spin_polarizations
    assert result.spin_polarizations == expected_poscar_spin_polarizations
    assert result.spin_polarizations_acc_poscar_spin_frame == expected_poscar_spin_polarizations
    assert result.spin_polarizations_acc_poscar_spin_frame != result.spin_polarizations_acc_cartesian


def test_find_spin_group_exposes_convention_nssg_views():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)
    expected_nssg = SpinSpaceGroup(public_ossg.nssg)

    assert _serialize_ssg_ops(result.convention_nssg_ops) == _serialize_ssg_ops(expected_nssg.ops)
    assert result.convention_nssg_seitz == expected_nssg.seitz_symbols
    assert result.convention_nssg_seitz_latex == expected_nssg.seitz_symbols_latex


@pytest.mark.parametrize(
    ("path", "expected_direction"),
    [
        ("examples/0.800_MnTe.mcif", "sqrt(2)/2,sqrt(2)/2,0"),
        ("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif", "0,0,1"),
        ("examples/CoNb3S6_tripleQ.mcif", ""),
    ],
)
def test_find_spin_group_exposes_convention_spin_only_direction(path, expected_direction):
    result = find_spin_group(path)
    assert result.convention_spin_only_direction == expected_direction


@pytest.mark.parametrize(
    ("path", "expect_identity_rotation", "expect_changed"),
    [
        ("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif", False, True),
        ("examples/0.800_MnTe.mcif", True, False),
        ("tests/testset/mcif_241130_no2186/1.317_La0.25Pr0.75Co2P2.mcif", True, False),
    ],
)
def test_poscar_spin_frame_projection_behaves_consistently_across_representative_cases(
    path,
    expect_identity_rotation,
    expect_changed,
):
    result = find_spin_group(path)

    forward = np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float)
    expected_forward = standardize_lattice(
        np.asarray(result.acc_primitive_magnetic_cell_detail["lattice"], dtype=float)
    )[1]
    projected = SpinSpaceGroup(result.acc_primitive_ssg_ops).transform_spin(forward).spin_polarizations

    assert np.allclose(forward, expected_forward, atol=1e-8)
    assert np.allclose(forward, np.eye(3), atol=1e-8) is expect_identity_rotation
    assert result.spin_polarizations == projected
    assert projected == result.spin_polarizations_acc_poscar_spin_frame
    assert (projected != result.spin_polarizations_acc_cartesian) is expect_changed


def test_find_spin_group_exposes_msg_little_groups_and_wp_chain():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.primitive_msg_ops
    assert result.msg_little_group_symbols
    assert result.msg_spin_polarizations
    assert result.msg_spin_polarizations_acc_poscar_spin_frame is not None
    assert (
        result.msg_spin_polarizations_acc_poscar_spin_frame_setting
        == "acc_primitive_poscar_spin_frame"
    )
    assert len(result.msg_little_group_symbols) == len(result.spin_polarizations)
    assert len(result.msg_spin_polarizations) == len(result.spin_polarizations)
    assert result.wp_chain


def test_spin_space_group_exposes_class_level_msg_ops_for_0200_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")

    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)
    msg_ops = public_ossg.msg_ops

    assert msg_ops
    assert _serialize_ssg_ops(msg_ops) == _serialize_ssg_ops(public_ossg.magnetic_space_group_ops)
    assert all(op.is_magnetic_space_group_operation(public_ossg.tol) for op in msg_ops)
    assert _serialize_msg_ops(_primitive_msg_ops_from_ssg(public_ossg.ops, tol=public_ossg.tol)) == (
        _serialize_msg_ops(_primitive_msg_ops_from_ssg(msg_ops, tol=public_ossg.tol))
    )


def test_spin_space_group_msg_info_is_lazy_and_cached(monkeypatch):
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)

    captured = {"count": 0}
    original = group_module.get_magnetic_space_group_from_operations

    def wrapped(operations):
        captured["count"] += 1
        return original(operations)

    monkeypatch.setattr(group_module, "get_magnetic_space_group_from_operations", wrapped)

    assert captured["count"] == 0
    assert "msg_info" not in public_ossg.__dict__

    first = public_ossg.msg_info

    assert captured["count"] == 1
    assert public_ossg.magnetic_space_group_info == first
    assert public_ossg.msg_int_num == 562
    assert public_ossg.msg_bns_num == "63.464"
    assert public_ossg.msg_bns_symbol == "Cm'cm'"
    assert public_ossg.msg_og_num == "63.8.518"
    assert public_ossg.msg_og_symbol == "Cm'cm'"
    assert public_ossg.msg_type == 3
    assert public_ossg.mpg_num == "8.4.27"
    assert public_ossg.mpg_symbol == "m'mm'"

    second = public_ossg.msg_info

    assert captured["count"] == 1
    assert first == second


def test_magnetic_time_reversal_uses_axial_vector_det_rule():
    improper_rotation = np.diag([1.0, -1.0, 1.0])
    op_without_time_reversal = group_module.SpinSpaceGroupOperation(
        -improper_rotation,
        improper_rotation,
        np.zeros(3),
    )
    op_with_time_reversal = group_module.SpinSpaceGroupOperation(
        improper_rotation,
        improper_rotation,
        np.zeros(3),
    )

    assert op_without_time_reversal.magnetic_time_reversal() == 1
    assert op_with_time_reversal.magnetic_time_reversal() == -1


def test_in_lattice_spin_frame_is_not_the_same_as_oriented_abc_for_0200_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )

    in_lattice = _spin_transform_to_in_lattice(acc_cell)
    oriented_abc = _spin_transform_to_oriented_abc(acc_cell)

    assert not np.allclose(in_lattice, oriented_abc, atol=1e-8)


def test_acc_primitive_msg_ops_are_derived_from_acc_primitive_ossg_for_0200_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_ssg = SpinSpaceGroup(result.acc_primitive_ssg_ops)
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(acc_ssg, acc_cell)

    assert _serialize_msg_ops(result.acc_primitive_msg_ops) == _serialize_msg_ops(
        _primitive_msg_ops_from_ssg(acc_primitive_ossg.msg_ops, tol=acc_primitive_ossg.tol)
    )
    assert result.acc_primitive_msg_ops_spin_frame_setting == "ossg_oriented_spin_frame"


def test_acc_primitive_ossg_reconstructs_collinear_msg_for_0712_vnb3s6():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.712_VNb3S6.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert result.acc_primitive_msg_ops_spin_frame_setting == "ossg_oriented_spin_frame"
    assert acc_primitive_ossg.msg_int_num == 131
    assert acc_primitive_ossg.msg_bns_num == "20.33"
    assert acc_primitive_ossg.msg_bns_symbol == "C2'2'2_1"


def test_acc_primitive_ossg_reconstructs_hex_collinear_msg_for_0800_mnte():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert result.acc_primitive_msg_ops_spin_frame_setting == "ossg_oriented_spin_frame"
    assert acc_primitive_ossg.msg_int_num == 555
    assert acc_primitive_ossg.msg_bns_num == "63.457"
    assert acc_primitive_ossg.msg_bns_symbol == "Cmcm"


@pytest.mark.parametrize(
    ("path", "expected_bns_num", "expected_bns_symbol"),
    [
        ("tests/testset/mcif_241130_no2186/0.691_CaCo1.86As2.mcif", "126.386", "P_I4/nnc"),
        ("tests/testset/mcif_241130_no2186/0.454_PrScSb.mcif", "128.410", "P_I4/mnc"),
    ],
)
def test_acc_primitive_ossg_reconstructs_tetragonal_type4_collinear_msg_representatives(
    path,
    expected_bns_num,
    expected_bns_symbol,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.gspg.spin_only_symbol_s == "D∞h"
    assert acc_primitive_ossg.msg_bns_num == expected_bns_num
    assert acc_primitive_ossg.msg_bns_symbol == expected_bns_symbol


@pytest.mark.parametrize(
    ("path", "expected_bns_num", "expected_bns_symbol"),
    [
        ("tests/testset/mcif_241130_no2186/0.1073_Cr2CoAl.mcif", "119.319", "I-4m'2'"),
        ("tests/testset/mcif_241130_no2186/0.229_Ba2MnSi2O7.mcif", "113.267", "P-42_1m"),
        ("tests/testset/mcif_241130_no2186/0.802_CuFeS2.mcif", "122.333", "I-42d"),
        ("tests/testset/mcif_241130_no2186/0.826_MnTeLi0.003.mcif", "12.62", "C2'/m'"),
        ("tests/testset/mcif_241130_no2186/1.188_CeRh2Si2.mcif", "64.480", "C_Amca"),
        ("tests/testset/mcif_241130_no2186/1.33_ErAuGe.mcif", "33.154", "P_Cna2_1"),
        ("tests/testset/mcif_241130_no2186/0.19_MnTiO3.mcif", "148.19", "R-3'"),
        ("tests/testset/mcif_241130_no2186/0.1001_PbMn2Ni6Te3O18.mcif", "176.146", "P6_3/m'"),
        ("tests/testset/mcif_241130_no2186/0.35_Cu2OSeO3.mcif", "146.10", "R3"),
    ],
)
def test_acc_primitive_ossg_reconstructs_high_order_collinear_msg_representatives(
    path,
    expected_bns_num,
    expected_bns_symbol,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.msg_bns_num == expected_bns_num
    assert acc_primitive_ossg.msg_bns_symbol == expected_bns_symbol


@pytest.mark.parametrize(
    ("path", "expected_msg_int_num", "expected_msg_type"),
    [
        ("tests/testset/mcif_241130_no2186/1.0.57_NdAlGe.mcif", 325, 3),
        ("tests/testset/mcif_241130_no2186/1.646_Na2Ni2TeO6.mcif", 332, 4),
        ("tests/testset/mcif_241130_no2186/1.738_TbNiAl.mcif", 345, 4),
    ],
)
def test_metric_aware_collinear_geometry_recovers_last_residual_msg_numbers(
    path,
    expected_msg_int_num,
    expected_msg_type,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.real_space_metric is not None
    assert acc_primitive_ossg.msg_int_num == expected_msg_int_num
    assert acc_primitive_ossg.msg_type == expected_msg_type


@pytest.mark.parametrize(
    ("path", "expected_order"),
    [
        ("tests/testset/mcif_241130_no2186/0.1073_Cr2CoAl.mcif", 4),
        ("tests/testset/mcif_241130_no2186/0.229_Ba2MnSi2O7.mcif", 4),
        ("tests/testset/mcif_241130_no2186/0.802_CuFeS2.mcif", 4),
        ("tests/testset/mcif_241130_no2186/0.826_MnTeLi0.003.mcif", 2),
        ("tests/testset/mcif_241130_no2186/1.188_CeRh2Si2.mcif", 2),
        ("tests/testset/mcif_241130_no2186/1.33_ErAuGe.mcif", 2),
    ],
)
def test_collinear_spin_promotion_order_uses_effective_proper_rotations(
    path,
    expected_order,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.collinear_spin_promotion_order == expected_order


def test_acc_primitive_ossg_recovers_type4_noncollinear_msg_for_1412_au72al14tb14():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.412_Au72Al14Tb14.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Noncoplanar"
    assert acc_primitive_ossg.msg_bns_num == "201.21"
    assert acc_primitive_ossg.msg_bns_symbol == "P_In-3"
    assert acc_primitive_ossg.msg_type == 4


def test_high_order_effective_axis_stays_aligned_with_collinear_axis_for_01073_cr2coal():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1073_Cr2CoAl.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    axes = []
    for op in acc_primitive_ossg.ops:
        order = _rotation_order(np.asarray(op[1], dtype=float), tol=1e-6)
        if order != 4:
            continue
        axis = _effective_proper_axis_from_space_rotation(np.asarray(op[1], dtype=float), tol=1e-4)
        if axis is None:
            continue
        if np.allclose(axis, acc_primitive_ossg.collinear_axis, atol=1e-4) or np.allclose(
            axis, -np.asarray(acc_primitive_ossg.collinear_axis), atol=1e-4
        ):
            axes.append(axis)

    assert result.conf == "Collinear"
    assert np.allclose(acc_primitive_ossg.collinear_axis, [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-4)
    assert axes


@pytest.mark.parametrize(
    ("path", "expect_identity_rotation", "expect_changed"),
    [
        ("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif", False, True),
        ("tests/testset/mcif_241130_no2186/0.13_Ca3Co2-xMnxO6.mcif", False, False),
        ("examples/0.800_MnTe.mcif", True, False),
    ],
)
def test_msg_spin_polarizations_poscar_projection_behaves_consistently_across_representative_cases(
    path,
    expect_identity_rotation,
    expect_changed,
):
    result = find_spin_group(path)

    forward = np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float)
    expected_forward = standardize_lattice(
        np.asarray(result.acc_primitive_magnetic_cell_detail["lattice"], dtype=float)
    )[1]
    _, _, expected_msg_poscar = _build_msg_little_group_payload(
        SpinSpaceGroup(result.acc_primitive_ssg_ops),
        CrystalCell(
            lattice=np.asarray(result.acc_primitive_magnetic_cell_detail["lattice"], dtype=float),
            positions=np.asarray(result.acc_primitive_magnetic_cell_detail["positions"], dtype=float),
            occupancies=result.acc_primitive_magnetic_cell_detail["occupancies"],
            elements=result.acc_primitive_magnetic_cell_detail["elements"],
            moments=np.asarray(result.acc_primitive_magnetic_cell_detail["moments"], dtype=float),
            spin_setting="cartesian",
        ),
        tol=0.01,
        spin_frame_rotation=forward,
    )

    assert np.allclose(forward, expected_forward, atol=1e-8)
    assert np.allclose(forward, np.eye(3), atol=1e-8) is expect_identity_rotation
    assert result.msg_spin_polarizations == expected_msg_poscar
    assert result.msg_spin_polarizations_acc_poscar_spin_frame == expected_msg_poscar
    assert (
        result.msg_spin_polarizations_acc_poscar_spin_frame
        != result.msg_spin_polarizations_acc_cartesian
    ) is expect_changed


def test_result_payload_can_be_json_serialized_for_web_app():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    payload = serialize_data(result.to_dict())

    encoded = json.dumps(payload, ensure_ascii=False)

    assert '"primitive_magnetic_cell_ssg_ops"' in encoded
    assert '"acc_primitive_ssg_ops"' in encoded
    assert '"acc_primitive_magnetic_cell"' in encoded
    assert '"T_G0std_to_acc_primitive"' in encoded
    assert '"spin_polarizations_acc_cartesian"' in encoded
    assert '"spin_polarizations_acc_poscar_spin_frame"' in encoded
    assert '"acc_primitive_real_cartesian_to_poscar_spin_frame"' in encoded
    assert '"msg_spin_polarizations_acc_poscar_spin_frame"' in encoded
    assert '"gspg_output_mode"' in encoded
    assert '"gspg_effective_mpg_symbol"' in encoded
    assert '"gspg_symbol_linear"' in encoded
    assert '"gspg_point_part_linear"' in encoded
    assert '"gspg_spin_only_part_linear"' in encoded
    assert '"gspg_symbol_mode"' in encoded
    assert '"gspg_spin_only_component_symbol_s"' in encoded
