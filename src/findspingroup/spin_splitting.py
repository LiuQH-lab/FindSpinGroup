from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence

import numpy as np


SIGMA_NAMES = ("sigma_x", "sigma_y", "sigma_z")
K_NAMES = ("kx", "ky", "kz")
SPIN_TEXTURE_TYPE_NAMES = {
    0: "s-wave",
    1: "p-wave",
    2: "d-wave",
    3: "f-wave",
    4: "g-wave",
    5: "h-wave",
    6: "i-wave",
    7: "j-wave",
    8: "k-wave",
    9: "l-wave",
    10: "m-wave",
}
CANONICAL_BASIS_RELATIVE_ZERO_TOL = 1e-6


def _split_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(delimiter, i):
            parts.append(text[start:i])
            i += len(delimiter)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return parts


def _strip_outer_parentheses(text: str) -> str:
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return text
    return text[1:-1].strip()


def _latex_symbol(name: str) -> str:
    if name.startswith("sigma_") and len(name) == len("sigma_x"):
        return rf"\sigma_{{{name[-1]}}}"
    if name in {"kx", "ky", "kz"}:
        return rf"k_{{{name[-1]}}}"
    return name


def _latex_variable_factor(token: str) -> str | None:
    if "^" in token:
        base, power = token.split("^", 1)
    else:
        base, power = token, None
    if base not in {"kx", "ky", "kz"}:
        return None
    symbol = _latex_symbol(base)
    return symbol if power in {None, "1"} else rf"{symbol}^{{{power}}}"


def _latex_radical_token(token: str) -> str:
    token = token.strip()
    if token.startswith("sqrt(") and token.endswith(")"):
        return rf"\sqrt{{{token[5:-1]}}}"
    return token


def _latex_numeric_value(value: float, *, zero_tol: float = 1e-8) -> str | None:
    if abs(value) < zero_tol:
        return "0"
    rounded = round(value)
    if abs(value - rounded) < zero_tol:
        return str(int(rounded))

    rational = Fraction(float(value)).limit_denominator(24)
    rational_value = rational.numerator / rational.denominator
    if abs(value - rational_value) < max(zero_tol, 1e-9):
        if rational.denominator == 1:
            return str(rational.numerator)
        return rf"\frac{{{rational.numerator}}}{{{rational.denominator}}}"

    radical = radical_text(
        value,
        zero_tol=max(zero_tol, 1e-8),
        max_radicand=12,
        max_denominator=24,
        max_multiplier=12,
    )
    if radical is None:
        return None
    return _latex_coefficient_token(radical)


def _latex_coefficient_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    try:
        numeric = float(token)
    except ValueError:
        pass
    else:
        numeric_latex = _latex_numeric_value(numeric)
        return numeric_latex if numeric_latex is not None else token

    if "/" in token:
        numerator, denominator = token.split("/", 1)
        return rf"\frac{{{_latex_coefficient_token(numerator)}}}{{{_latex_coefficient_token(denominator)}}}"
    if "*" in token:
        return "".join(_latex_coefficient_token(part) for part in _split_top_level(token, "*"))
    return _latex_radical_token(token)


def _latex_factor(factor: str) -> tuple[int, str]:
    factor = _strip_outer_parentheses(factor)
    polynomial_terms = _split_signed_terms(factor)
    if len(polynomial_terms) > 1:
        pieces: list[str] = []
        for i, (term_sign, term) in enumerate(polynomial_terms):
            factor_sign, factor_latex = _latex_factor(term)
            sign = term_sign * factor_sign
            if i == 0:
                pieces.append(factor_latex if sign > 0 else rf"-{factor_latex}")
            else:
                pieces.append((" + " if sign > 0 else " - ") + factor_latex)
        return 1, rf"\left({''.join(pieces)}\right)"

    sign = 1
    if factor.startswith("-"):
        sign = -1
        factor = factor[1:].strip()
    elif factor.startswith("+"):
        factor = factor[1:].strip()

    coefficient_parts: list[str] = []
    variable_parts: list[str] = []
    for part in _split_top_level(factor, "*"):
        token = part.strip()
        if not token:
            continue
        variable = _latex_variable_factor(token)
        if variable is not None:
            variable_parts.append(variable)
        else:
            coefficient_parts.append(_latex_coefficient_token(token))

    coefficient = "".join(part for part in coefficient_parts if part not in {"", "1"})
    body = "".join([coefficient, *variable_parts])
    return sign, body or "1"


def _split_signed_terms(text: str) -> list[tuple[int, str]]:
    terms: list[tuple[int, str]] = []
    depth = 0
    sign = 1
    start = 0
    i = 0
    if text.startswith("-"):
        sign = -1
        start = 1
        i = 1
    elif text.startswith("+"):
        start = 1
        i = 1
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(" + ", i):
            terms.append((sign, text[start:i].strip()))
            sign = 1
            i += 3
            start = i
            continue
        elif depth == 0 and text.startswith(" - ", i):
            terms.append((sign, text[start:i].strip()))
            sign = -1
            i += 3
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        terms.append((sign, tail))
    return terms


def _split_basis_remainder_suffix(text: str) -> tuple[str, str]:
    marker = " + o("
    index = text.rfind(marker)
    if index < 0:
        return text, ""
    return text[:index], text[index:]


def _basis_remainder_suffix_to_latex(suffix: str) -> str:
    if not suffix:
        return ""
    marker = " + o("
    if not (suffix.startswith(marker) and suffix.endswith(")")):
        return suffix
    body = suffix[len(marker):-1]
    if body == "1":
        return " + o(1)"
    if body == "k":
        return " + o(k)"
    if body.startswith("k^"):
        return rf" + o(k^{{{body[2:]}}})"
    return suffix


def _split_basis_coefficient(text: str) -> tuple[str | None, str]:
    if "*(" not in text or not text.endswith(")"):
        return None, text
    prefix, maybe_inner = text.split("*(", 1)
    if prefix.startswith("C") and prefix[1:].isdigit():
        return prefix, maybe_inner[:-1]
    return None, text


def _parse_sigma_term(term: str) -> tuple[str, int, str] | None:
    term = term.strip()
    if term in SIGMA_NAMES:
        return term, 1, "1"
    if "*sigma_" not in term:
        return None
    factor, sigma = term.rsplit("*", 1)
    sigma = sigma.strip()
    if sigma not in SIGMA_NAMES:
        return None
    factor = _strip_outer_parentheses(factor.strip())
    if len(_split_signed_terms(factor)) > 1:
        return sigma, 1, factor or "1"
    sign = 1
    if factor.startswith("-"):
        sign = -1
        factor = factor[1:].strip()
    elif factor.startswith("+"):
        factor = factor[1:].strip()
    factor = _strip_outer_parentheses(factor)
    return sigma, sign, factor or "1"


def _render_signed_factor_sum(terms: Sequence[tuple[int, str]]) -> str:
    pieces: list[str] = []
    for i, (sign, factor) in enumerate(terms):
        body = factor.strip() or "1"
        if i == 0:
            pieces.append(body if sign > 0 else f"-{body}")
        else:
            pieces.append((" + " if sign > 0 else " - ") + body)
    return "".join(pieces)


def _render_signed_expression_terms(terms: Sequence[tuple[int, str]]) -> str:
    pieces: list[str] = []
    for i, (sign, body) in enumerate(terms):
        body = body.strip()
        if i == 0:
            pieces.append(body if sign > 0 else f"-{body}")
        else:
            pieces.append((" + " if sign > 0 else " - ") + body)
    return "".join(pieces)


def _combine_spin_texture_inner_terms(inner: str) -> str:
    grouped: dict[str, list[tuple[int, str]]] = {sigma: [] for sigma in SIGMA_NAMES}
    passthrough: list[tuple[int, str]] = []

    for term_sign, term in _split_signed_terms(inner):
        parsed = _parse_sigma_term(term)
        if parsed is None:
            passthrough.append((term_sign, term))
            continue
        sigma, factor_sign, factor = parsed
        grouped[sigma].append((term_sign * factor_sign, factor))

    rendered_terms: list[tuple[int, str]] = []
    for sigma in SIGMA_NAMES:
        factors = grouped[sigma]
        if not factors:
            continue
        if len(factors) == 1:
            sign, factor = factors[0]
            if factor == "1":
                rendered_terms.append((sign, sigma))
            else:
                rendered_terms.append((sign, f"({factor})*{sigma}"))
            continue
        rendered_terms.append((1, f"({_render_signed_factor_sum(factors)})*{sigma}"))

    rendered_terms.extend(passthrough)
    return _render_signed_expression_terms(rendered_terms)


def combine_spin_texture_basis_expression(expression: str) -> str:
    """Group public spin-texture basis terms by sigma component."""

    text = str(expression).strip()
    if text in {"", "0"}:
        return text

    main, remainder = _split_basis_remainder_suffix(text)
    coefficient, inner = _split_basis_coefficient(main)
    combined_inner = _combine_spin_texture_inner_terms(inner)
    if coefficient is None:
        return f"{combined_inner}{remainder}"
    return f"{coefficient}*({combined_inner}){remainder}"


def combine_spin_texture_basis(basis: Sequence[str] | None) -> list[str]:
    if not basis:
        return []
    return [combine_spin_texture_basis_expression(expression) for expression in basis]


def basis_expression_to_latex(expression: str) -> str:
    """Convert the public ASCII spin-texture basis expression to LaTeX."""

    text = combine_spin_texture_basis_expression(str(expression).strip())
    if text in {"", "0"}:
        return "0"
    text, remainder = _split_basis_remainder_suffix(text)

    coefficient = None
    inner = text
    prefix, maybe_inner = _split_basis_coefficient(text)
    if prefix is not None:
        coefficient = rf"C_{{{prefix[1:]}}}"
        inner = maybe_inner

    latex_terms: list[tuple[int, str]] = []
    for term_sign, term in _split_signed_terms(inner):
        if term.strip() in SIGMA_NAMES:
            latex_terms.append((term_sign, _latex_symbol(term.strip())))
            continue
        if "*sigma_" not in term:
            latex_terms.append((term_sign, term))
            continue
        factor, sigma = term.rsplit("*", 1)
        factor_sign, factor_latex = _latex_factor(factor)
        sign = term_sign * factor_sign
        sigma_latex = _latex_symbol(sigma.strip())
        if factor_latex == "1":
            body = sigma_latex
        else:
            body = rf"{factor_latex}\,{sigma_latex}"
        latex_terms.append((sign, body))

    if not latex_terms:
        body = inner
    else:
        pieces: list[str] = []
        for i, (sign, body) in enumerate(latex_terms):
            if i == 0:
                pieces.append(body if sign > 0 else rf"-{body}")
            else:
                pieces.append((" + " if sign > 0 else " - ") + body)
        body = "".join(pieces)

    if coefficient is None:
        return body + _basis_remainder_suffix_to_latex(remainder)
    return rf"{coefficient}\left({body}\right)" + _basis_remainder_suffix_to_latex(remainder)


def spin_texture_basis_latex(basis: Sequence[str] | None) -> list[str]:
    if not basis:
        return []
    return [basis_expression_to_latex(expression) for expression in basis]


def _basis_remainder_ascii(order: int) -> str:
    if order == 0:
        return "o(1)"
    if order == 1:
        return "o(k)"
    return f"o(k^{order})"


def _basis_remainder_latex(order: int) -> str:
    if order == 0:
        return "o(1)"
    if order == 1:
        return "o(k)"
    return rf"o(k^{{{order}}})"


def _append_basis_remainder_ascii(basis: Sequence[str] | None, order: int | None) -> list[str]:
    if not basis:
        return []
    if order is None:
        return [str(expression) for expression in basis]
    suffix = _basis_remainder_ascii(int(order))
    return [f"{expression} + {suffix}" for expression in basis]


def _append_basis_remainder_latex(basis_latex: Sequence[str] | None, order: int | None) -> list[str]:
    if not basis_latex:
        return []
    if order is None:
        return [str(expression) for expression in basis_latex]
    suffix = _basis_remainder_latex(int(order))
    return [rf"{expression} + {suffix}" for expression in basis_latex]


def _resolve_basis_remainder_order(
    leading_order: int | None,
    basis_remainder_order: int | str | None,
) -> int | None:
    if basis_remainder_order is None or leading_order is None:
        return None
    if basis_remainder_order == "leading":
        return int(leading_order)
    return int(basis_remainder_order)


@dataclass(frozen=True)
class OperationPair:
    """One reciprocal/spin operation pair using d(Q k) = S d(k)."""

    Q: np.ndarray
    S: np.ndarray


@dataclass
class OrderDiagnostics:
    order: int
    spin_texture_type: str
    unknown_count: int
    constraint_count: int
    rank: int
    nullity: int
    svd_threshold: float
    min_nonzero_singular: float | None
    max_zero_singular: float | None
    singular_values_head: list[float]
    confidence: str
    residual: float | None


@dataclass
class SpinSplittingResult:
    order: int | None
    spin_texture_type: str
    nullity: int
    basis: list[str]
    basis_latex: list[str]
    basis_by_order: list[dict[str, Any]] | None
    spin_rank: int
    momentum_space_spin_configuration: str
    allowed_orders: list[OrderDiagnostics]
    engine: str = "numeric-coefficient-svd"
    convention: str = "d(Q k) = S d(k)"


def as_float_3x3(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix, got shape {matrix.shape}")
    return matrix


def as_float_square(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")
    return matrix


def operation_pair(q_matrix: Any, spin_matrix: Any) -> OperationPair:
    return OperationPair(Q=as_float_square(q_matrix, name="Q"), S=as_float_3x3(spin_matrix, name="S"))


def reciprocal_q_from_real_rotation(
    real_rotation: Any,
    spin_rotation: Any,
    *,
    p_acc_aligned: Any | None = None,
) -> np.ndarray:
    real = as_float_3x3(real_rotation, name="real_rotation")
    spin = as_float_3x3(spin_rotation, name="spin_rotation")
    det_factor = 1.0 if np.linalg.det(spin) >= 0.0 else -1.0
    q_conv = det_factor * np.linalg.inv(real).T
    if p_acc_aligned is None:
        return q_conv
    p_matrix = as_float_3x3(p_acc_aligned, name="p_acc_aligned")
    return p_matrix.T @ q_conv @ np.linalg.inv(p_matrix).T


def monomials_of_degree(order: int, *, dimension: int = 3) -> list[tuple[int, ...]]:
    if order < 0:
        raise ValueError("order must be non-negative")
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    def _build(remaining_order: int, remaining_dimension: int) -> list[tuple[int, ...]]:
        if remaining_dimension == 1:
            return [(remaining_order,)]
        out: list[tuple[int, ...]] = []
        for power in range(remaining_order + 1):
            for suffix in _build(remaining_order - power, remaining_dimension - 1):
                out.append((power, *suffix))
        return out

    return _build(order, dimension)


def _default_k_names(dimension: int) -> tuple[str, ...]:
    if dimension <= len(K_NAMES):
        return K_NAMES[:dimension]
    return tuple(f"k{index + 1}" for index in range(dimension))


def monomial_text(monomial: tuple[int, ...], *, k_names: Sequence[str] | None = None) -> str:
    if k_names is None:
        k_names = _default_k_names(len(monomial))
    pieces: list[str] = []
    for name, power in zip(k_names, monomial):
        if power == 0:
            continue
        pieces.append(name if power == 1 else f"{name}^{power}")
    return "*".join(pieces) if pieces else "1"


def spin_texture_type_for_order(order: int) -> str:
    return SPIN_TEXTURE_TYPE_NAMES.get(order, f"order-{order}-wave")


def multiply_poly_by_linear(
    poly: dict[tuple[int, ...], float],
    coeffs: np.ndarray,
    *,
    zero_tol: float,
) -> dict[tuple[int, ...], float]:
    out: dict[tuple[int, ...], float] = {}
    for powers, value in poly.items():
        for axis, coeff in enumerate(coeffs):
            if abs(float(coeff)) <= zero_tol:
                continue
            new_powers = list(powers)
            new_powers[axis] += 1
            key = tuple(new_powers)
            out[key] = out.get(key, 0.0) + value * float(coeff)
    return {key: value for key, value in out.items() if abs(value) > zero_tol}


def monomial_transform_numeric(
    q_matrix: np.ndarray,
    monomials: Sequence[tuple[int, ...]],
    *,
    zero_tol: float,
) -> np.ndarray:
    index = {monomial: i for i, monomial in enumerate(monomials)}
    transform = np.zeros((len(monomials), len(monomials)), dtype=np.float64)
    for source_index, powers in enumerate(monomials):
        poly: dict[tuple[int, ...], float] = {tuple(0 for _ in powers): 1.0}
        for row_index, power in enumerate(powers):
            for _ in range(power):
                poly = multiply_poly_by_linear(poly, q_matrix[row_index, :], zero_tol=zero_tol)
        for target, coeff in poly.items():
            target_index = index.get(target)
            if target_index is not None:
                transform[source_index, target_index] += coeff
    transform[np.abs(transform) <= zero_tol] = 0.0
    return transform


def constraint_matrix_for_pair_order_numeric(
    q_matrix: np.ndarray,
    spin_matrix: np.ndarray,
    monomials: Sequence[tuple[int, ...]],
    *,
    zero_tol: float,
) -> np.ndarray:
    nm = len(monomials)
    transform = monomial_transform_numeric(q_matrix, monomials, zero_tol=zero_tol)
    matrix = np.zeros((3 * nm, 3 * nm), dtype=np.float64)
    row_index = 0
    for component in range(3):
        for mono_index in range(nm):
            matrix[row_index, component * nm : (component + 1) * nm] += transform[:, mono_index]
            for source_component in range(3):
                matrix[row_index, source_component * nm + mono_index] -= spin_matrix[
                    component,
                    source_component,
                ]
            row_index += 1
    matrix[np.abs(matrix) <= zero_tol] = 0.0
    return matrix


def matrix_key(matrix: np.ndarray, *, decimals: int) -> tuple[float, ...]:
    rounded = np.round(matrix, decimals)
    rounded[np.abs(rounded) < 10 ** (-decimals)] = 0.0
    return tuple(float(value) for value in rounded.reshape(-1))


def is_identity_pair(pair: OperationPair, *, zero_tol: float) -> bool:
    return bool(
        np.allclose(pair.Q, np.eye(pair.Q.shape[0]), atol=zero_tol, rtol=0.0)
        and np.allclose(pair.S, np.eye(pair.S.shape[0]), atol=zero_tol, rtol=0.0)
    )


def unique_operation_pairs(
    pairs: Iterable[OperationPair],
    *,
    decimals: int,
) -> list[OperationPair]:
    seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    out: list[OperationPair] = []
    for pair in pairs:
        key = (matrix_key(pair.Q, decimals=decimals), matrix_key(pair.S, decimals=decimals))
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def sorted_operation_pairs(
    pairs: Iterable[OperationPair],
    *,
    zero_tol: float,
    key_decimals: int,
) -> list[OperationPair]:
    return sorted(
        pairs,
        key=lambda pair: (
            is_identity_pair(pair, zero_tol=zero_tol),
            matrix_key(pair.Q, decimals=key_decimals),
            matrix_key(pair.S, decimals=key_decimals),
        ),
    )


def constraint_matrix_for_order_numeric(
    pairs: Sequence[OperationPair],
    order: int,
    *,
    k_dimension: int,
    zero_tol: float,
    key_decimals: int,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    monomials = monomials_of_degree(order, dimension=k_dimension)
    blocks: list[np.ndarray] = []
    for pair in sorted_operation_pairs(pairs, zero_tol=zero_tol, key_decimals=key_decimals):
        if is_identity_pair(pair, zero_tol=zero_tol):
            continue
        blocks.append(
            constraint_matrix_for_pair_order_numeric(
                pair.Q,
                pair.S,
                monomials,
                zero_tol=zero_tol,
            )
        )
    if not blocks:
        return np.zeros((0, 3 * len(monomials)), dtype=np.float64), monomials
    return np.vstack(blocks), monomials


def svd_nullspace(
    matrix: np.ndarray,
    *,
    rtol: float,
    atol: float,
    confidence_gap: float,
) -> tuple[int, list[float], float, float | None, float | None, str, np.ndarray]:
    unknown_count = matrix.shape[1]
    if matrix.size == 0:
        return 0, [], atol, None, 0.0, "high", np.eye(unknown_count, dtype=np.float64)

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    smax = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(atol, rtol * smax)
    rank = int(np.sum(singular_values > threshold))
    nullity = unknown_count - rank
    min_nonzero = float(singular_values[rank - 1]) if rank > 0 else None
    max_zero = float(singular_values[rank]) if rank < singular_values.size else (0.0 if nullity else None)

    confidence = "high"
    if min_nonzero is not None and min_nonzero < confidence_gap * threshold:
        confidence = "borderline"
    if max_zero is not None and max_zero > threshold / confidence_gap:
        confidence = "borderline"

    basis = vh[rank:, :].T if nullity > 0 else np.zeros((unknown_count, 0), dtype=np.float64)
    return rank, [float(value) for value in singular_values], threshold, min_nonzero, max_zero, confidence, basis


def _canonical_basis_vector(vector: np.ndarray, *, zero_tol: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).copy()
    if vector.size == 0:
        return vector
    max_abs = float(np.max(np.abs(vector)))
    if max_abs <= zero_tol:
        return np.zeros_like(vector)

    vector = vector / max_abs
    vector[np.abs(vector) < max(zero_tol, CANONICAL_BASIS_RELATIVE_ZERO_TOL)] = 0.0
    first_index = next(
        (index for index, value in enumerate(vector) if abs(value) > zero_tol),
        None,
    )
    if first_index is not None and vector[first_index] < 0:
        vector = -vector
    return vector


def canonicalize_nullspace(basis: np.ndarray, *, zero_tol: float) -> list[np.ndarray]:
    if basis.shape[1] == 0:
        return []
    if basis.shape[1] == 1:
        return [_canonical_basis_vector(basis[:, 0], zero_tol=zero_tol)]

    selected: list[int] = []
    current = np.zeros((0, basis.shape[1]), dtype=np.float64)
    for row in range(basis.shape[0]):
        candidate = np.vstack([current, basis[row : row + 1, :]])
        if np.linalg.matrix_rank(candidate, tol=zero_tol) > len(selected):
            selected.append(row)
            current = candidate
            if len(selected) == basis.shape[1]:
                break
    if len(selected) != basis.shape[1]:
        return [
            _canonical_basis_vector(basis[:, index], zero_tol=zero_tol)
            for index in range(basis.shape[1])
        ]

    pivot_block = basis[selected, :]
    canonical = basis @ np.linalg.inv(pivot_block)
    out: list[np.ndarray] = []
    for index in range(canonical.shape[1]):
        out.append(_canonical_basis_vector(canonical[:, index], zero_tol=zero_tol))
    return out


def fraction_text(value: float, *, zero_tol: float, max_denominator: int) -> str | None:
    fraction = Fraction(float(value)).limit_denominator(max_denominator)
    approx = fraction.numerator / fraction.denominator
    if abs(value - approx) > zero_tol:
        return None
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def radical_text(
    value: float,
    *,
    zero_tol: float,
    max_radicand: int,
    max_denominator: int,
    max_multiplier: int,
) -> str | None:
    sign = "-" if value < 0 else ""
    target = abs(float(value))
    best: tuple[float, str] | None = None

    for radicand in range(2, max_radicand + 1):
        root = math.sqrt(radicand)
        if int(root) ** 2 == radicand:
            continue
        for multiplier in range(1, max_multiplier + 1):
            for denominator in range(1, max_denominator + 1):
                approx = multiplier * root / denominator
                error = abs(target - approx)
                if error > zero_tol:
                    continue
                if multiplier == 1 and denominator == 1:
                    body = f"sqrt({radicand})"
                elif multiplier == 1:
                    body = f"sqrt({radicand})/{denominator}"
                elif denominator == 1:
                    body = f"{multiplier}*sqrt({radicand})"
                else:
                    body = f"{multiplier}*sqrt({radicand})/{denominator}"
                candidate = f"{sign}{body}"
                if best is None or (error, len(candidate), candidate) < (best[0], len(best[1]), best[1]):
                    best = (error, candidate)

    return best[1] if best is not None else None


def format_float(
    value: float,
    *,
    zero_tol: float,
    rational_max_denominator: int = 24,
    radical_max_radicand: int = 12,
    radical_max_multiplier: int = 12,
) -> str:
    if abs(value) < zero_tol:
        return "0"
    rounded = round(value)
    if abs(value - rounded) < zero_tol:
        return str(int(rounded))
    rational = fraction_text(value, zero_tol=zero_tol, max_denominator=rational_max_denominator)
    if rational is not None:
        return rational
    radical = radical_text(
        value,
        zero_tol=zero_tol,
        max_radicand=radical_max_radicand,
        max_denominator=rational_max_denominator,
        max_multiplier=radical_max_multiplier,
    )
    if radical is not None:
        return radical
    return f"{value:.10g}"


def basis_vector_expression_numeric(
    vector: np.ndarray,
    monomials: Sequence[tuple[int, ...]],
    *,
    k_names: Sequence[str] | None = None,
    zero_tol: float,
    rational_max_denominator: int = 24,
    radical_max_radicand: int = 12,
    radical_max_multiplier: int = 12,
) -> str:
    nm = len(monomials)
    pieces: list[str] = []
    for component, sigma in enumerate(SIGMA_NAMES):
        for mono_index, monomial in enumerate(monomials):
            coeff = float(vector[component * nm + mono_index])
            if abs(coeff) < zero_tol:
                continue
            mono = monomial_text(monomial, k_names=k_names)
            coeff_text = format_float(
                coeff,
                zero_tol=zero_tol,
                rational_max_denominator=rational_max_denominator,
                radical_max_radicand=radical_max_radicand,
                radical_max_multiplier=radical_max_multiplier,
            )
            if mono == "1":
                scalar = coeff_text
            elif coeff_text == "1":
                scalar = mono
            elif coeff_text == "-1":
                scalar = f"-{mono}"
            else:
                scalar = f"{coeff_text}*{mono}"
            pieces.append(f"({scalar})*{sigma}")
    return " + ".join(pieces).replace("+ (-", "- (") if pieces else "0"


def spin_rank_and_configuration_from_vectors(
    vectors: Sequence[np.ndarray],
    *,
    tol: float,
) -> tuple[int, str]:
    rows: list[np.ndarray] = []
    for vector in vectors:
        if vector.size == 0:
            continue
        matrix = vector.reshape(3, -1)
        for mono_index in range(matrix.shape[1]):
            spin_vector = matrix[:, mono_index]
            if np.any(np.abs(spin_vector) > tol):
                rows.append(spin_vector)
    if not rows:
        return 0, "zero"
    span = np.vstack(rows)
    rank = int(np.linalg.matrix_rank(span, tol=tol))
    if rank <= 0:
        return 0, "zero"
    if rank == 1:
        return 1, "collinear"
    if rank == 2:
        return 2, "coplanar"
    return 3, "noncoplanar"


def _basis_payload_for_order(
    *,
    order: int,
    basis: np.ndarray,
    monomials: Sequence[tuple[int, ...]],
    k_names: Sequence[str],
    zero_tol: float,
    rational_max_denominator: int,
    radical_max_radicand: int,
    radical_max_multiplier: int,
    basis_remainder_order: int | str | None,
) -> dict[str, Any]:
    canonical = canonicalize_nullspace(basis, zero_tol=zero_tol)
    expressions: list[str] = []
    for index, vector in enumerate(canonical):
        expression = basis_vector_expression_numeric(
            vector,
            monomials,
            k_names=k_names,
            zero_tol=zero_tol,
            rational_max_denominator=rational_max_denominator,
            radical_max_radicand=radical_max_radicand,
            radical_max_multiplier=radical_max_multiplier,
        )
        expressions.append(f"C{index + 1}*({expression})")
    expressions = combine_spin_texture_basis(expressions)
    spin_rank, spin_config = spin_rank_and_configuration_from_vectors(canonical, tol=zero_tol)
    remainder_order = _resolve_basis_remainder_order(order, basis_remainder_order)
    expressions_latex = spin_texture_basis_latex(expressions)
    return {
        "order": int(order),
        "spin_texture_type": spin_texture_type_for_order(order),
        "nullity": int(len(expressions)),
        "spin_rank": int(spin_rank),
        "momentum_space_spin_configuration": spin_config,
        "basis": _append_basis_remainder_ascii(expressions, remainder_order),
        "basis_latex": _append_basis_remainder_latex(expressions_latex, remainder_order),
    }


def _empty_basis_payload_for_order(order: int) -> dict[str, Any]:
    return {
        "order": int(order),
        "spin_texture_type": spin_texture_type_for_order(order),
        "nullity": 0,
        "spin_rank": 0,
        "momentum_space_spin_configuration": "zero",
        "basis": [],
        "basis_latex": [],
    }


def classify_spin_splitting_numeric(
    operations: Iterable[OperationPair | tuple[Any, Any] | dict[str, Any]],
    *,
    max_order: int = 6,
    basis_orders_through: int | None = None,
    basis_remainder_order: int | str | None = None,
    k_dimension: int | None = None,
    k_names: Sequence[str] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    zero_tol: float = 1e-8,
    confidence_gap: float = 100.0,
    key_decimals: int = 8,
    rational_max_denominator: int = 24,
    radical_max_radicand: int = 12,
    radical_max_multiplier: int = 12,
) -> SpinSplittingResult:
    pairs = normalize_operations(operations, key_decimals=key_decimals)
    if k_dimension is None:
        k_dimension = int(pairs[0].Q.shape[0]) if pairs else 3
    if k_dimension <= 0:
        raise ValueError("k_dimension must be positive")
    for pair in pairs:
        if pair.Q.shape != (k_dimension, k_dimension):
            raise ValueError(
                f"All Q matrices must be {k_dimension}x{k_dimension}; got {pair.Q.shape}"
            )
        if pair.S.shape != (3, 3):
            raise ValueError(f"All spin matrices must be 3x3; got {pair.S.shape}")
    if k_names is None:
        k_names = _default_k_names(k_dimension)
    elif len(k_names) != k_dimension:
        raise ValueError(f"k_names length {len(k_names)} does not match k_dimension {k_dimension}")
    allowed_orders: list[OrderDiagnostics] = []
    basis_by_order: list[dict[str, Any]] = []
    leading_payload: dict[str, Any] | None = None
    for order in range(max_order + 1):
        matrix, monomials = constraint_matrix_for_order_numeric(
            pairs,
            order,
            k_dimension=k_dimension,
            zero_tol=zero_tol,
            key_decimals=key_decimals,
        )
        rank, singular_values, threshold, min_nonzero, max_zero, confidence, basis = svd_nullspace(
            matrix,
            rtol=rtol,
            atol=atol,
            confidence_gap=confidence_gap,
        )
        nullity = int(basis.shape[1])
        residual = None
        if nullity:
            denom = max(float(singular_values[0]) if singular_values else 1.0, 1.0)
            residual = float(np.max(np.abs(matrix @ basis)) / denom) if matrix.size else 0.0
        allowed_orders.append(
            OrderDiagnostics(
                order=order,
                spin_texture_type=spin_texture_type_for_order(order),
                unknown_count=matrix.shape[1],
                constraint_count=matrix.shape[0],
                rank=rank,
                nullity=nullity,
                svd_threshold=threshold,
                min_nonzero_singular=min_nonzero,
                max_zero_singular=max_zero,
                singular_values_head=singular_values[: min(12, len(singular_values))],
                confidence=confidence,
                residual=residual,
            )
        )
        if nullity:
            order_payload = _basis_payload_for_order(
                order=order,
                basis=basis,
                monomials=monomials,
                k_names=k_names,
                zero_tol=zero_tol,
                rational_max_denominator=rational_max_denominator,
                radical_max_radicand=radical_max_radicand,
                radical_max_multiplier=radical_max_multiplier,
                basis_remainder_order=basis_remainder_order,
            )
            if basis_orders_through is not None and order <= int(basis_orders_through):
                basis_by_order.append(order_payload)
            if leading_payload is None:
                leading_payload = order_payload
            if basis_orders_through is None or order >= int(basis_orders_through):
                return SpinSplittingResult(
                    order=leading_payload["order"],
                    spin_texture_type=leading_payload["spin_texture_type"],
                    nullity=leading_payload["nullity"],
                    basis=leading_payload["basis"],
                    basis_latex=leading_payload["basis_latex"],
                    basis_by_order=basis_by_order if basis_orders_through is not None else None,
                    spin_rank=leading_payload["spin_rank"],
                    momentum_space_spin_configuration=leading_payload[
                        "momentum_space_spin_configuration"
                    ],
                    allowed_orders=allowed_orders,
                )
        elif basis_orders_through is not None and order <= int(basis_orders_through):
            basis_by_order.append(_empty_basis_payload_for_order(order))
    if leading_payload is not None:
        return SpinSplittingResult(
            order=leading_payload["order"],
            spin_texture_type=leading_payload["spin_texture_type"],
            nullity=leading_payload["nullity"],
            basis=leading_payload["basis"],
            basis_latex=leading_payload["basis_latex"],
            basis_by_order=basis_by_order if basis_orders_through is not None else None,
            spin_rank=leading_payload["spin_rank"],
            momentum_space_spin_configuration=leading_payload[
                "momentum_space_spin_configuration"
            ],
            allowed_orders=allowed_orders,
        )

    return SpinSplittingResult(
        order=None,
        spin_texture_type="forbidden",
        nullity=0,
        basis=[],
        basis_latex=[],
        basis_by_order=basis_by_order if basis_orders_through is not None else None,
        spin_rank=0,
        momentum_space_spin_configuration="zero",
        allowed_orders=allowed_orders,
    )


def normalize_operations(
    operations: Iterable[OperationPair | tuple[Any, Any] | dict[str, Any]],
    *,
    key_decimals: int,
) -> list[OperationPair]:
    pairs: list[OperationPair] = []
    for item in operations:
        if isinstance(item, OperationPair):
            pairs.append(operation_pair(item.Q, item.S))
        elif isinstance(item, dict):
            if "Q" in item and "S" in item:
                pairs.append(operation_pair(item["Q"], item["S"]))
            elif "q" in item and "s" in item:
                pairs.append(operation_pair(item["q"], item["s"]))
            elif "real_rotation" in item and "spin_rotation" in item:
                spin = as_float_3x3(item["spin_rotation"], name="spin_rotation")
                q_matrix = reciprocal_q_from_real_rotation(
                    item["real_rotation"],
                    spin,
                    p_acc_aligned=item.get("p_acc_aligned"),
                )
                pairs.append(OperationPair(Q=q_matrix, S=spin))
            else:
                raise ValueError("operation dict must contain Q/S or real_rotation/spin_rotation")
        else:
            q_matrix, spin_matrix = item
            pairs.append(operation_pair(q_matrix, spin_matrix))
    return unique_operation_pairs(pairs, decimals=key_decimals)


def result_to_jsonable(result: SpinSplittingResult, *, include_diagnostics: bool = False) -> dict[str, Any]:
    payload = asdict(result)
    if not include_diagnostics:
        payload.pop("allowed_orders", None)
        payload.pop("engine", None)
        payload.pop("convention", None)
    if payload.get("basis_by_order") is None:
        payload.pop("basis_by_order", None)
    if "basis_latex" not in payload:
        payload["basis_latex"] = spin_texture_basis_latex(payload.get("basis"))
    return payload


def operation_pairs_from_gspg_ops(ops: Iterable[Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for op in ops:
        spin_rotation, real_rotation = op
        pairs.append(
            {
                "spin_rotation": spin_rotation,
                "real_rotation": real_rotation,
            }
        )
    return pairs


def operation_pairs_from_ssg_ops(ops: Iterable[Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for op in ops:
        spin_rotation, real_rotation = op[0], op[1]
        pairs.append(
            {
                "spin_rotation": spin_rotation,
                "real_rotation": real_rotation,
            }
        )
    return pairs


def operation_pairs_from_msg_ops(ops: Iterable[Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for op in ops:
        time_reversal, real_rotation = int(op[0]), as_float_3x3(op[1], name="real_rotation")
        spin_rotation = time_reversal * np.linalg.det(real_rotation) * real_rotation
        pairs.append(
            {
                "spin_rotation": spin_rotation,
                "real_rotation": real_rotation,
            }
        )
    return pairs


def collinear_axis_constraint_operation(axis: Any) -> dict[str, Any]:
    axis_array = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_array))
    if norm <= 1e-12:
        raise ValueError("collinear axis must be non-zero")
    unit = axis_array / norm
    spin_rotation = 2.0 * np.outer(unit, unit) - np.eye(3)
    return {
        "real_rotation": np.eye(3),
        "spin_rotation": spin_rotation,
    }


def classify_public_spin_texture_config(
    operations: Iterable[OperationPair | tuple[Any, Any] | dict[str, Any]],
    *,
    source: str,
    max_order: int = 6,
    basis_orders_through: int | None = None,
    basis_remainder_order: int | str | None = "leading",
    k_dimension: int | None = None,
    k_names: Sequence[str] | None = None,
    include_diagnostics: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    zero_tol: float = 1e-8,
) -> dict[str, Any]:
    effective_max_order = int(basis_orders_through) if basis_orders_through is not None else max_order
    result = classify_spin_splitting_numeric(
        operations,
        max_order=effective_max_order,
        basis_orders_through=basis_orders_through,
        basis_remainder_order=basis_remainder_order,
        k_dimension=k_dimension,
        k_names=k_names,
        rtol=rtol,
        atol=atol,
        zero_tol=zero_tol,
    )
    payload = result_to_jsonable(result, include_diagnostics=include_diagnostics)
    payload["source"] = source
    payload["classifier_tolerances"] = {
        "rtol": float(rtol),
        "atol": float(atol),
        "zero_tol": float(zero_tol),
    }
    return payload
