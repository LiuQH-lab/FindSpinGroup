import os

import numpy as np
import pytest

from findspingroup import find_spin_group
from findspingroup.data.POINT_GROUP_MATRIX import operations_hex
from findspingroup.data.SG_SYMBOL import SGdisc
from findspingroup.structure.group import SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.utils.international_symbol import (
    _minimal_k_translation_generators,
    _point_group_token_from_real_token,
    _real_generator_tokens,
    _real_symbol_tokens,
)
from findspingroup.utils.seitz_symbol import (
    calibrated_symbol_tol,
    describe_point_operation,
    format_point_seitz_symbol_latex,
    format_translation_tau_latex,
)


@pytest.mark.parametrize(
    "mcif_file, expected_type",
    [
        ("tests/testset/0.200_Mn3Sn.mcif", "t"),
        ("tests/testset/mcif_241130_no2186/2.54_Sr2Cr3As2O2.mcif", "k"),
        ("tests/testset/mcif_241130_no2186/1.357_Ho3Ge4.mcif", "g"),
    ],
)
def test_international_symbol_type_and_forms(mcif_file, expected_type):
    assert os.path.exists(mcif_file)
    result = find_spin_group(mcif_file)
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    info = ssg.international_symbol
    assert info["type"] == expected_type
    assert isinstance(ssg.international_symbol_linear, str) and ssg.international_symbol_linear
    assert isinstance(ssg.international_symbol_latex, str) and ssg.international_symbol_latex


def test_find_spin_group_exposes_international_symbol_fields():
    result = find_spin_group("tests/testset/0.200_Mn3Sn.mcif")

    assert isinstance(result.primitive_magnetic_cell_ssg_international_linear, str)
    assert isinstance(result.primitive_magnetic_cell_ssg_international_latex, str)
    assert result.primitive_magnetic_cell_ssg_type in {"t", "k", "g"}


def test_point_group_token_drops_braced_screw_subscripts():
    assert _point_group_token_from_real_token("6_{3}/") == "6/"
    assert _point_group_token_from_real_token("4_1") == "4"
    assert _point_group_token_from_real_token("c") == "m"


@pytest.mark.parametrize(
    ("sg_num", "expected_tokens"),
    [
        (149, ["3", "1", "2"]),
        (150, ["3", "2", "1"]),
        (151, ["3_{1}", "1", "2"]),
        (152, ["3_{1}", "2", "1"]),
        (153, ["3_{2}", "1", "2"]),
        (154, ["3_{2}", "2", "1"]),
        (156, ["3", "m", "1"]),
        (157, ["3", "1", "m"]),
        (158, ["3", "c", "1"]),
        (159, ["3", "1", "c"]),
        (162, ["-3", "1", "m"]),
        (163, ["-3", "1", "c"]),
        (164, ["-3", "m", "1"]),
        (165, ["-3", "c", "1"]),
    ],
)
def test_trigonal_hm_symbol_restores_identity_direction_slots(
    sg_num,
    expected_tokens,
):
    generator_tokens = _real_generator_tokens(sg_num, named_count=2)

    assert len(generator_tokens) == 2
    assert _real_symbol_tokens(sg_num, generator_tokens) == expected_tokens


def test_only_trigonal_directional_hm_symbols_need_identity_slots():
    identity_slot_sg_numbers = {
        sg_num
        for sg_num, tokens in SGdisc.items()
        if sg_num != 1 and "1" in tokens[1:]
    }

    assert identity_slot_sg_numbers == {
        149,
        150,
        151,
        152,
        153,
        154,
        156,
        157,
        158,
        159,
        162,
        163,
        164,
        165,
    }


def test_type_g_symbol_distinguishes_p_minus3_m1_setting():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.237_VCl2.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    symbol = ssg.international_symbol_current_frame

    assert result.G0_num == 164
    assert result.G0_symbol == "P-3m1"
    assert result.primitive_magnetic_cell_ssg_type == "g"
    assert result.convention_ssg_international_linear.startswith(
        "P 2_{120}|-3 2_{120}|m 1|1 :"
    )
    assert r"^{2_{120}}\bar{3} ^{2_{120}}m ^{1}1" in (
        result.convention_ssg_international_latex
    )
    assert result.gspg_point_part_linear.endswith("1|1")
    assert symbol["real_generator_pairs_linear"] == [
        "2_{120}|-3",
        "2_{120}|m",
    ]
    assert "1" not in {
        operation["label"] for operation in symbol["generator_operations"]
    }


def test_mag_symmetry_result_repr_uses_linear_symbol_by_default():
    result = find_spin_group("tests/testset/0.200_Mn3Sn.mcif")
    rendered = repr(result)

    assert rendered.startswith(f"<{result.primitive_magnetic_cell_ssg_international_linear}>")
    assert result.primitive_magnetic_cell_ssg_international_latex not in rendered.splitlines()[0]


def test_k_type_uses_minimal_translation_generators_in_linear_and_latex():
    result = find_spin_group("tests/testset/mcif_241130_no2186/2.54_Sr2Cr3As2O2.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)
    info = ssg.international_symbol

    assert info["type"] == "k"
    assert info["translation_terms_linear"] == [
        "2_{100}|(1/2,0,1/2)",
        "2_{010}|(1/2,1/2,0)",
    ]
    assert ssg.international_symbol_linear.startswith(
        "P 1|m 1|n 1|a : 2_{100}|(1/2,0,1/2) 2_{010}|(1/2,1/2,0)"
    )
    assert "^{2_{100}}(\\frac{1}{2},0,\\frac{1}{2}) ^{2_{010}}(\\frac{1}{2},\\frac{1}{2},0)" in ssg.international_symbol_latex


def test_k_type_translation_generators_prefer_single_higher_order_closure():
    identity = np.eye(3)
    spin_c4 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    spin_c2 = spin_c4 @ spin_c4
    spin_c4_inv = np.linalg.inv(spin_c4)

    candidates = [
        SpinSpaceGroupOperation(identity, identity, np.array([0.0, 0.0, 0.0])),
        SpinSpaceGroupOperation(spin_c2, identity, np.array([0.0, 0.0, 0.5])),
        SpinSpaceGroupOperation(spin_c4, identity, np.array([0.5, 0.5, 0.25])),
        SpinSpaceGroupOperation(spin_c4_inv, identity, np.array([0.5, 0.5, 0.75])),
    ]

    selected = _minimal_k_translation_generators(candidates)

    assert len(selected) == 1
    assert np.allclose(selected[0].spin_rotation, spin_c4)
    assert np.allclose(selected[0].translation, [0.5, 0.5, 0.25])


def test_k_type_translation_generators_use_centering_closure():
    identity = np.eye(3)
    spin_inversion = -np.eye(3)
    c_centering = SpinSpaceGroupOperation(
        identity,
        identity,
        np.array([0.5, 0.5, 0.0]),
    )
    candidates = [
        SpinSpaceGroupOperation(spin_inversion, identity, np.array([0.0, 0.5, 0.5])),
        SpinSpaceGroupOperation(spin_inversion, identity, np.array([0.5, 0.0, 0.5])),
    ]

    selected = _minimal_k_translation_generators(
        candidates,
        free_generators=[c_centering],
    )

    assert len(selected) == 1
    assert np.allclose(selected[0].spin_rotation, spin_inversion)
    assert np.allclose(selected[0].translation, [0.0, 0.5, 0.5])


def test_centered_k_type_symbol_uses_implicit_centering_translation():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.826_NdZnPO.mcif")

    assert result.index == "12.12.2.1.L"
    assert result.convention_ssg_international_linear.startswith(
        "C 1|2/ 1|m : -1|(0,1/2,1/2)"
    )
    assert "-1|(1/2,0,1/2)" not in result.convention_ssg_international_linear


def test_spin_only_suffix_is_appended_for_coplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.357_Ho3Ge4.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)
    info = ssg.international_symbol

    assert ssg.international_symbol_type == "g"
    assert info["translation_terms_linear"] == ["(1,1,1;2_{001})"]
    assert info["translation_terms_latex"] == ["(1,1,1;2_{001})"]
    assert " : (1,1,1;2_{001}) " in ssg.international_symbol_linear
    assert r" \mid (1,1,1;2_{001}) " in ssg.international_symbol_latex
    assert ssg.international_symbol_linear.endswith("m_{010}|1")
    assert ssg.international_symbol_latex.endswith("^{m_{010}}1")


def test_collinear_suffix_uses_current_oriented_frame_axis_for_lamno3():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1_LaMnO3.mcif")

    assert result.convention_ssg_international_linear.endswith("∞_{100}m|1")
    assert result.convention_ssg_international_latex.endswith(r"^{\infty_{100}m}1")


def test_symbol_fallback_replaces_known_zero_axis_parameters():
    result = find_spin_group("tests/testset/0.200_Mn3Sn.mcif")

    assert "1/2,-sqrt(3)/2,0" in result.primitive_magnetic_cell_ssg_international_linear
    assert "alpha,beta,0" not in result.primitive_magnetic_cell_ssg_international_linear
    assert "alpha,beta,gamma" not in result.primitive_magnetic_cell_ssg_international_linear
    assert "1/2,-sqrt(3)/2,0" in result.primitive_magnetic_cell_ssg_international_latex

    normal = np.array([1.2345, 0.0, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    mirror = np.eye(3) - 2.0 * np.outer(normal, normal)
    info = describe_point_operation(mirror, tol=1e-6)

    assert info["axis_kind"] == "parameter"
    assert info["axis_subscript_linear"] == "alpha,0,gamma"
    assert info["axis_subscript_latex"] == r"\alpha,0,\gamma"
    assert info["axis_parameter_values"] is not None
    assert info["axis_euler_deg"] is None
    assert abs(info["axis_parameter_values"][1]) < 1e-8


def test_convention_and_gspg_share_parameter_placeholder_direction_for_ndga():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1098_NdGa.mcif")

    assert result.convention_ssg_international_linear.endswith("∞_{alpha,0,gamma}m|1")
    assert result.gspg_symbol_linear.endswith("∞_{alpha,0,gamma}m|1")


def test_ossg_symbol_numbers_distinct_free_axis_placeholders():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1002_SrZn2Fe16O27.mcif")

    assert "m_{alpha1,beta1,0}|m" in result.convention_ssg_international_linear
    assert "m_{alpha2,beta2,0}|c" in result.convention_ssg_international_linear
    assert r"m_{\alpha_{1},\beta_{1},0}" in result.convention_ssg_international_latex
    assert r"m_{\alpha_{2},\beta_{2},0}" in result.convention_ssg_international_latex


def test_symbol_calibration_tol_is_coupled_to_user_matrix_tolerance():
    result = find_spin_group("tests/testset/0.200_Mn3Sn.mcif")
    tight = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops, tol=1e-8)
    loose = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops, tol=1e-2)

    assert tight.symbol_calibration_tol == calibrated_symbol_tol(1e-8)
    assert loose.symbol_calibration_tol == calibrated_symbol_tol(1e-2)
    assert tight.symbol_calibration_tol < loose.symbol_calibration_tol

    structured = loose.seitz_descriptions
    assert structured
    assert any(
        item["spin"].get("axis_parameter_values") is not None
        or item["real"].get("axis_parameter_values") is not None
        for item in structured
    )


def test_seitz_descriptions_and_symbol_lists_expose_latex_forms():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)

    assert ssg.seitz_symbols_latex
    assert len(ssg.seitz_symbols_latex) == len(ssg.seitz_symbols)
    assert "symbol_latex" in ssg.seitz_descriptions[0]
    assert "translation_symbol_latex" in ssg.seitz_descriptions[0]
    assert ssg.seitz_descriptions[0]["symbol_latex"] == ssg.seitz_symbols_latex[0]
    assert ssg.seitz_symbols[0].startswith("{ ")
    assert "tau_{" not in ssg.seitz_symbols[0]
    assert r"\tau" not in ssg.seitz_symbols_latex[0]
    assert ssg.seitz_descriptions[0]["translation_symbol"] == "0,0,0"
    assert ssg.seitz_symbols[0].endswith("| 0,0,0 }")


def test_seitz_point_latex_keeps_minus_sign_prefix():
    assert format_point_seitz_symbol_latex("-3", "direction", (1, -1, 0), None, 1) == "-3^{1}_{1-10}"


def test_seitz_translation_latex_snaps_near_fractional_components():
    assert (
        format_translation_tau_latex(np.array([0.49998, 0.00002, 0.99998]), tol=1e-4)
        == r"\frac{1}{2},0,0"
    )


def test_describe_point_operation_uses_legacy_hex_minus6_branch_labels():
    minus6_minus = None
    for matrix, _, token in operations_hex:
        if token == "-6^5_{001}":
            minus6_minus = np.array(matrix, dtype=float)
            break

    assert minus6_minus is not None

    info = describe_point_operation(minus6_minus, tol=1e-6)

    assert info["hm_symbol"] == "-6"
    assert info["rotation_power"] == 5
    assert info["axis_direction"] == (0, 0, 1)
    assert info["symbol"] == "-6^{5}_{001}"
