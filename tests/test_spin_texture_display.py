"""Display rounding must not change the accepted spin-texture payload."""

from copy import deepcopy

import pytest

from findspingroup.spin_splitting import basis_expression_to_latex, spin_texture_basis_latex


@pytest.mark.parametrize(
    ("coefficient", "display"),
    [
        ("0.123456789123", "0.1235"),
        ("-0.123456789123", "-0.1235"),
        ("1.234500000", "1.2345"),
        ("0.000000622612345", r"6.2261\times 10^{-7}"),
        ("-3.24612345e-7", r"-3.2461\times 10^{-7}"),
        ("0.00001", r"1\times 10^{-5}"),
        ("0.000123456789", r"1.2346\times 10^{-4}"),
        ("1e-300", r"1\times 10^{-300}"),
        ("7161972.442", r"7.162\times 10^{6}"),
        ("9.999999e-6", r"1\times 10^{-5}"),
        ("0.500000", "0.5"),
    ],
)
def test_display_coefficients_have_at_most_four_decimal_places(coefficient, display):
    expression = f"C1*(({coefficient}*kx)*sigma_x) + o(k)"
    rendered = basis_expression_to_latex(expression, decimal_places=4)
    assert display in rendered
    assert rendered.endswith(" + o(k)")
    assert rendered.count(r"\sigma_{x}") == 1


def test_display_keeps_fraction_radical_variable_and_parameter_semantics():
    expression = (
        "C123*((sqrt(3)/3*ky^2)*sigma_x + (1/2*kz^12)*sigma_y"
        " + (0.123456789*ky*kz)*sigma_z) + o(k^12)"
    )
    rendered = basis_expression_to_latex(expression, decimal_places=4)
    assert r"\frac{\sqrt{3}}{3}" in rendered
    assert r"\frac{1}{2}" in rendered
    assert r"C_{123}" in rendered
    assert r"k_{z}^{12}" in rendered
    assert r"k_{x}" not in rendered
    assert rendered.endswith(" + o(k^{12})")
    assert "0.1235" in rendered
    assert r"\frac{1234567}{3456789}" in basis_expression_to_latex(
        "C1*((1234567/3456789*kx)*sigma_x)", decimal_places=4,
    )


def test_display_reuses_sigma_grouping_and_single_remainder():
    expression = "C1*((0.123456789*kx*ky)*sigma_x + (ky*kz)*sigma_x) + o(k^2) + o(k^2)"
    rendered = basis_expression_to_latex(expression, decimal_places=4)
    assert rendered.count(r"\sigma_{x}") == 1
    assert rendered.count("o(k^{2})") == 1
    assert "0.1235" in rendered


@pytest.mark.parametrize("mode", ["3d", "quasi2d"])
@pytest.mark.parametrize("key", ["spin_texture_config_no_soc", "spin_texture_config_soc"])
def test_display_does_not_mutate_any_runtime_result(mode, key):
    basis = ["C1*((0.123456789123*ky)*sigma_y - (6.22612345e-7*kz)*sigma_z) + o(k)"]
    payload = {
        "spin_texture_type": "p-wave",
        "basis": basis,
        "basis_latex": spin_texture_basis_latex(basis),
        "constraint_validation": {"valid": True, "threshold": 1e-8},
    }
    result = {key: payload} if mode == "3d" else {"quasi_2d": {key: payload}}
    original = deepcopy(result)
    rendered = spin_texture_basis_latex(payload["basis"], decimal_places=4)
    assert "0.1235" in rendered[0]
    assert r"6.2261\times 10^{-7}" in rendered[0]
    assert result == original
    assert spin_texture_basis_latex(basis) == payload["basis_latex"]
    assert "0.123456789123" in payload["basis_latex"][0]


def test_empty_and_zero_order_display():
    assert spin_texture_basis_latex(None, decimal_places=4) == []
    assert spin_texture_basis_latex([], decimal_places=4) == []
    assert spin_texture_basis_latex(["C1*(sigma_z) + o(1)"], decimal_places=4) == [
        r"C_{1}\left(\sigma_{z}\right) + o(1)"
    ]


@pytest.mark.parametrize("precision", [-1, 1.2, True, "4"])
def test_display_rejects_invalid_precision(precision):
    with pytest.raises(ValueError, match="decimal_places"):
        basis_expression_to_latex("C1*(sigma_z)", decimal_places=precision)


def test_zero_decimal_places_does_not_truncate_integer_zeros():
    assert "10" in basis_expression_to_latex("C1*((10.123*kx)*sigma_z)", decimal_places=0)
