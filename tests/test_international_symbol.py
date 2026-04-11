import os

import numpy as np
import pytest

from findspingroup import find_spin_group
from findspingroup.data.POINT_GROUP_MATRIX import operations_hex
from findspingroup.structure.group import SpinSpaceGroup
from findspingroup.utils.seitz_symbol import (
    calibrated_symbol_tol,
    describe_point_operation,
    format_point_seitz_symbol_latex,
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

    assert "alpha,beta,0" in result.primitive_magnetic_cell_ssg_international_linear
    assert "alpha,beta,gamma" not in result.primitive_magnetic_cell_ssg_international_linear
    assert r"\alpha,\beta,0" in result.primitive_magnetic_cell_ssg_international_latex

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
    assert result.gspg_spin_only_part_linear == "∞_{alpha,0,gamma}m|1"


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
    assert "tau_{" in ssg.seitz_symbols[0]
    assert r"\tau_{(" in ssg.seitz_symbols_latex[0]


def test_seitz_point_latex_keeps_minus_sign_prefix():
    assert format_point_seitz_symbol_latex("-3", "direction", (1, -1, 0), None, 1) == "-3^{1}_{1-10}"


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
