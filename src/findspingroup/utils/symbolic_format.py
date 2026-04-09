from __future__ import annotations

from fractions import Fraction
import re

import numpy as np


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
