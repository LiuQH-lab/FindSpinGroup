from __future__ import annotations

from fractions import Fraction
import math
import re

import numpy as np


def format_direction_components(direction, *, integer_direction=False) -> str:
    """Format a reusable 3-vector in its supplied frame, without physical snapping.

    Only an explicitly scale-free direction may be replaced by small integers.
    Other callers retain the supplied normalization. Compact symbolic scalars
    must agree to machine precision; this is not a symmetry acceptance budget.
    """
    values = np.asarray(direction, dtype=float).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Direction must be a finite single 3-vector")
    scale = float(np.max(np.abs(values)))
    budget = 16 * np.finfo(float).eps * scale
    if integer_direction and scale > 0:
        normalized = values / scale
        for denominator in range(1, 25):
            integers = np.rint(normalized * denominator).astype(int)
            if np.max(np.abs(integers / denominator - normalized)) <= 16*np.finfo(float).eps:
                divisor = math.gcd(*integers.tolist())
                return ",".join(str(int(value // divisor)) for value in integers)
    components = []
    for value in values:
        text = format_symbolic_scalar(value, decimal_precision=15, zero_tol=budget,
                                      rational_tol=budget, sqrt_tol=budget)
        try:
            if abs(float(text) - value) > budget:
                text = repr(float(value))
        except ValueError:
            pass  # Fractions and radicals have already satisfied this budget.
        components.append(text)
    return ",".join(components)


def format_symbolic_scalar(
    value: float,
    *,
    decimal_precision: int = 6,
    zero_tol: float = 1e-12,
    rational_tol: float = 1e-9,
    sqrt_tol: float = 5e-6,
    max_denominator: int = 12,
    sqrt_values: tuple[int, ...] = (2, 3, 5, 6),
) -> str:
    numeric = float(value)
    if abs(numeric) <= zero_tol:
        return "0"

    rational = Fraction(numeric).limit_denominator(max_denominator)
    if abs(float(rational) - numeric) <= rational_tol:
        if rational.denominator == 1:
            return str(rational.numerator)
        return f"{rational.numerator}/{rational.denominator}"

    for sqrt_value in sqrt_values:
        scaled = numeric / np.sqrt(sqrt_value)
        factor = Fraction(float(scaled)).limit_denominator(max_denominator)
        if abs(float(factor) * np.sqrt(sqrt_value) - numeric) > sqrt_tol:
            continue

        numerator = factor.numerator
        denominator = factor.denominator
        sign = "-" if numerator < 0 else ""
        numerator = abs(numerator)

        if numerator == 1 and denominator == 1:
            return f"{sign}sqrt({sqrt_value})"
        if denominator == 1:
            return f"{sign}{numerator}*sqrt({sqrt_value})"
        if numerator == 1:
            return f"{sign}sqrt({sqrt_value})/{denominator}"
        return f"{sign}{numerator}*sqrt({sqrt_value})/{denominator}"

    return f"{numeric:.{decimal_precision}f}".rstrip("0").rstrip(".")


_FLOAT_TOKEN_RE = re.compile(r"(?<![A-Za-z_])([+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)")


def symbolize_numeric_tokens_in_string(
    value: str,
    *,
    sqrt_tol: float = 5e-6,
    rational_tol: float = 1e-9,
) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        try:
            return format_symbolic_scalar(
                float(token),
                sqrt_tol=sqrt_tol,
                rational_tol=rational_tol,
            )
        except Exception:
            return token

    return _FLOAT_TOKEN_RE.sub(_replace, value)
