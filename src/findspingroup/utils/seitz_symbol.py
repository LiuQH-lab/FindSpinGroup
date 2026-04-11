from __future__ import annotations

import copy
import math
import re
from fractions import Fraction
from functools import lru_cache

import numpy as np
from findspingroup.data.POINT_GROUP_MATRIX import operations, operations_hex


def calibrated_symbol_tol(
    tol: float,
    *,
    floor: float = 5e-5,
    ceil: float | None = 1e-4,
) -> float:
    value = float(tol)
    if ceil is not None:
        value = min(value, float(ceil))
    return max(value, float(floor))


def _orthogonalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a near-orthogonal matrix to O(3) while keeping det sign."""
    matrix = np.asarray(matrix, dtype=float)
    u, _, vt = np.linalg.svd(matrix)
    ortho = u @ vt
    if np.linalg.det(ortho) * np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        ortho = u @ vt
    return ortho


def _matrix_order(matrix: np.ndarray, max_order: int = 120, tol: float = 1e-6) -> int:
    power = np.eye(3)
    for order in range(1, max_order + 1):
        power = power @ matrix
        if np.allclose(power, np.eye(3), atol=tol, rtol=0):
            return order
    raise ValueError(f"Cannot determine matrix order <= {max_order}.")


def _fallback_low_order(matrix: np.ndarray, tol: float = 1e-6) -> int | None:
    matrix = np.asarray(matrix, dtype=float)
    if np.allclose(matrix, np.eye(3), atol=max(1e-4, 100 * tol), rtol=0):
        return 1
    if np.allclose(matrix @ matrix, np.eye(3), atol=max(1e-3, 20 * tol), rtol=0):
        return 2

    det = float(np.linalg.det(matrix))
    eigenvalues = np.linalg.eigvals(matrix)
    if det < 0:
        unit_eigenvalues = [value / abs(value) for value in eigenvalues if abs(value) > tol]
        minus_one_hits = [
            value for value in unit_eigenvalues
            if abs(value.imag) < max(1e-3, 1000 * tol) and abs(value.real + 1.0) < max(1e-2, 10000 * tol)
        ]
        if minus_one_hits:
            residual_angles = []
            for value in unit_eigenvalues:
                if any(np.allclose(value, hit, atol=max(1e-3, 1000 * tol), rtol=0) for hit in minus_one_hits):
                    continue
                angle = abs(float(np.angle(value)))
                residual_angles.append(min(angle, abs(2.0 * math.pi - angle)))
            if residual_angles and max(residual_angles) < 5e-2:
                return 2
    return None


def _nearest_known_point_operation(
    matrix: np.ndarray,
    *,
    tol: float = 1e-6,
) -> np.ndarray | None:
    """Return the nearest standard point-operation matrix when the match is clear."""
    matrix = np.asarray(matrix, dtype=float)
    comparison_bases = [matrix, _orthogonalize_matrix(matrix)]
    unique_candidates: list[np.ndarray] = []
    for op_set in (operations, operations_hex):
        for raw_matrix, _, _ in op_set:
            candidate = np.asarray(raw_matrix, dtype=float)
            if any(np.allclose(candidate, existing, atol=1e-12, rtol=0) for existing in unique_candidates):
                continue
            unique_candidates.append(candidate)

    candidates: list[tuple[float, np.ndarray]] = []
    for candidate in unique_candidates:
        best_dist = min(float(np.linalg.norm(basis_matrix - candidate)) for basis_matrix in comparison_bases)
        candidates.append((best_dist, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    best_dist, best_matrix = candidates[0]
    second_dist = candidates[1][0] if len(candidates) > 1 else math.inf

    # This branch only runs after direct order probing and spectral fallback
    # both fail. A wider fixed gate is acceptable here for noisy but clearly
    # standard finite-order operations such as the 0.427 near-`-4` case.
    max_match_dist = max(1e-2, 4 * tol)
    min_gap = max(1e-4, 2 * tol)

    if best_dist > max_match_dist:
        return None
    if second_dist - best_dist < min_gap:
        return None
    return np.asarray(best_matrix, dtype=float)


def _parse_direction_subscript_token(token: str) -> tuple[int, int, int]:
    if "," in token:
        parts = [part.strip() for part in token.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid direction token: {token}")
        return int(parts[0]), int(parts[1]), int(parts[2])

    values: list[int] = []
    i = 0
    while i < len(token):
        sign = 1
        if token[i] == "-":
            sign = -1
            i += 1
        if i >= len(token) or not token[i].isdigit():
            raise ValueError(f"Invalid direction token: {token}")
        values.append(sign * int(token[i]))
        i += 1
    if len(values) != 3:
        raise ValueError(f"Invalid direction token: {token}")
    return int(values[0]), int(values[1]), int(values[2])


def _parse_known_point_token(token: str) -> dict:
    match = re.fullmatch(r"(?P<hm>-?\d+|m)(?:\^(?P<power>-?\d+))?(?:_\{(?P<axis>[^}]+)\})?", token)
    if not match:
        raise ValueError(f"Unsupported point-operation token: {token}")

    hm_symbol = match.group("hm")
    power_text = match.group("power")
    axis_text = match.group("axis")
    rotation_power = None if power_text is None else int(power_text)
    axis_direction = None if axis_text is None else _parse_direction_subscript_token(axis_text)
    axis_kind = "direction" if axis_direction is not None else None

    axis_vector = None
    if axis_direction is not None:
        raw = np.asarray(axis_direction, dtype=float)
        norm = np.linalg.norm(raw)
        if norm > 0:
            axis_vector = tuple(float(v) for v in raw / norm)

    return {
        "hm_symbol": hm_symbol,
        "rotation_power": rotation_power,
        "axis_kind": axis_kind,
        "axis_direction": axis_direction,
        "axis_parameter_values": None,
        "axis_vector": axis_vector,
    }


@lru_cache(maxsize=1)
def _known_point_operation_tokens() -> tuple[tuple[np.ndarray, dict], ...]:
    items: list[tuple[np.ndarray, dict]] = []
    for op_set in (operations, operations_hex):
        for raw_matrix, _, token in op_set:
            items.append((np.asarray(raw_matrix, dtype=float), _parse_known_point_token(token)))
    return tuple(items)


def _nearest_known_point_operation_token(
    matrix: np.ndarray,
    *,
    tol: float = 1e-6,
) -> dict | None:
    matrix = np.asarray(matrix, dtype=float)
    comparison_bases = [matrix, _orthogonalize_matrix(matrix)]

    candidates: list[tuple[float, dict]] = []
    for candidate_matrix, token_info in _known_point_operation_tokens():
        best_dist = min(float(np.linalg.norm(basis_matrix - candidate_matrix)) for basis_matrix in comparison_bases)
        candidates.append((best_dist, token_info))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    best_dist, best_info = candidates[0]
    second_dist = candidates[1][0] if len(candidates) > 1 else math.inf

    max_match_dist = max(1e-2, 4 * tol)
    min_gap = max(1e-4, 2 * tol)
    if best_dist > max_match_dist:
        return None
    if second_dist - best_dist < min_gap:
        return None
    return copy.deepcopy(best_info)


def _spectral_order_hint(
    matrix: np.ndarray,
    *,
    max_order: int = 120,
    tol: float = 1e-6,
) -> int | None:
    """Infer a plausible finite order from eigenvalue phases when power probing misses narrowly."""
    matrix = np.asarray(matrix, dtype=float)
    _, fold = _rotation_fraction_from_eigenvalues(matrix, max_fold=max_order, tol=max(tol, 1e-6))
    if fold <= 1 or fold > max_order:
        return None

    power = np.eye(3)
    for _ in range(fold):
        power = power @ matrix
    if np.max(np.abs(power - np.eye(3))) > max(2e-4, 4 * tol):
        return None
    return int(fold)


def _canonicalize_axis(axis: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < tol:
        raise ValueError("Cannot normalize near-zero axis.")
    axis = axis / norm
    for value in axis:
        if abs(value) > tol:
            if value < 0:
                axis = -axis
            break
    return axis


def _eigen_axis(matrix: np.ndarray, eigenvalue: float, tol: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eig(matrix)
    idx = int(np.argmin(np.abs(vals - eigenvalue)))
    axis = np.real(vecs[:, idx])
    if np.linalg.norm(axis) < tol:
        raise ValueError("Failed to find a non-zero eigen-axis.")
    return _canonicalize_axis(axis, tol=tol)


def _perpendicular_unit(axis: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    basis = np.eye(3)
    idx = int(np.argmin(np.abs(basis @ axis)))
    vec = basis[idx] - np.dot(basis[idx], axis) * axis
    norm = np.linalg.norm(vec)
    if norm < tol:
        idx = (idx + 1) % 3
        vec = basis[idx] - np.dot(basis[idx], axis) * axis
        norm = np.linalg.norm(vec)
    if norm < tol:
        raise ValueError("Cannot construct a perpendicular helper vector.")
    return vec / norm


def _rotation_fraction(matrix: np.ndarray, axis: np.ndarray, max_fold: int, tol: float = 1e-6) -> tuple[int, int]:
    """Return reduced (m, n) from an angle 2*pi*m/n around axis."""
    v1 = _perpendicular_unit(axis, tol=tol)
    v2 = matrix @ v1
    v2 = v2 - np.dot(v2, axis) * axis
    norm = np.linalg.norm(v2)
    if norm < tol:
        return 0, 1
    v2 = v2 / norm

    cross_term = float(np.dot(axis, np.cross(v1, v2)))
    dot_term = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle = math.atan2(cross_term, dot_term)
    if angle < 0:
        angle += 2.0 * math.pi

    ratio = (angle / (2.0 * math.pi)) % 1.0
    if ratio < tol or abs(ratio - 1.0) < tol:
        return 0, 1

    frac = Fraction(ratio).limit_denominator(max_fold)
    num = int(frac.numerator)
    den = int(frac.denominator)
    num %= den
    if num == 0:
        num = den
    return num, den


def _rotation_fraction_from_eigenvalues(
    matrix: np.ndarray, max_fold: int, tol: float = 1e-6
) -> tuple[int, int]:
    vals = np.linalg.eigvals(matrix)
    complex_vals = [v for v in vals if abs(v.imag) > tol]
    if not complex_vals:
        return 1, 1

    positive_imag = [v for v in complex_vals if v.imag > 0]
    chosen = positive_imag[0] if positive_imag else complex_vals[0]
    chosen = chosen / abs(chosen)

    angle = math.atan2(float(chosen.imag), float(chosen.real))
    if angle < 0:
        angle += 2.0 * math.pi

    ratio = (angle / (2.0 * math.pi)) % 1.0
    if ratio < tol or abs(ratio - 1.0) < tol:
        return 1, 1

    frac = Fraction(ratio).limit_denominator(max_fold)
    return int(frac.numerator), int(frac.denominator)


def _hm_improper_fold(matrix: np.ndarray, fallback_fold: int, tol: float = 1e-6) -> int:
    """
    Return the crystallographic HM fold for an improper point operation.

    For example, a `-3` operation has matrix order 6, but its HM symbol is
    still `-3`. The HM fold comes from the proper rotation part `-R`.
    """
    candidate = -np.asarray(matrix, dtype=float)
    try:
        return _matrix_order(candidate, max_order=max(12, fallback_fold * 2), tol=tol)
    except Exception:
        return int(fallback_fold)


def _vector_to_integer_direction(
    axis: np.ndarray, max_denom: int = 12, tol: float = 1e-4
) -> tuple[int, int, int] | None:
    axis = _canonicalize_axis(axis, tol=tol)
    ref_idx = int(np.argmax(np.abs(axis)))
    if abs(axis[ref_idx]) < tol:
        return None

    ratios = axis / axis[ref_idx]
    fracs = [Fraction(float(v)).limit_denominator(max_denom) for v in ratios]
    approx = np.array([float(f) for f in fracs], dtype=float)
    if np.max(np.abs(approx - ratios)) > tol:
        return None

    lcm = 1
    for frac in fracs:
        lcm = math.lcm(lcm, frac.denominator)

    ints = np.array([frac.numerator * (lcm // frac.denominator) for frac in fracs], dtype=int)
    non_zero = ints[np.nonzero(ints)]
    if non_zero.size == 0:
        return None

    gcd = int(abs(non_zero[0]))
    for value in non_zero[1:]:
        gcd = math.gcd(gcd, int(abs(value)))
    if gcd > 1:
        ints //= gcd

    for value in ints:
        if value != 0:
            if value < 0:
                ints = -ints
            break
    return int(ints[0]), int(ints[1]), int(ints[2])


def _axis_to_parameter_values(axis: np.ndarray, tol: float = 1e-8) -> tuple[float, float, float]:
    axis = _canonicalize_axis(axis, tol=tol)
    values = []
    for value in axis:
        numeric = float(value)
        if abs(numeric) < tol:
            numeric = 0.0
        values.append(numeric)
    return tuple(values)


def _axis_parameter_subscript(
    axis_parameter_values: tuple[float, float, float] | None,
    *,
    latex: bool = False,
    zero_tol: float = 1e-8,
) -> str:
    labels = (
        (r"\alpha", r"\beta", r"\gamma")
        if latex
        else ("alpha", "beta", "gamma")
    )
    if axis_parameter_values is None:
        return ",".join(labels)

    rendered: list[str] = []
    for label, value in zip(labels, axis_parameter_values):
        rendered.append("0" if abs(float(value)) < zero_tol else label)
    return ",".join(rendered)


def _symbolic_component_token(
    value: float,
    *,
    tol: float = 1e-4,
) -> tuple[str, str] | None:
    numeric = float(value)
    if abs(numeric) < tol:
        return "0", "0"

    sign = "-" if numeric < 0 else ""
    magnitude = abs(numeric)
    candidates = [
        (1.0, "1", "1"),
        (0.5, "1/2", r"\frac{1}{2}"),
        (math.sqrt(2.0) / 2.0, "sqrt(2)/2", r"\frac{\sqrt{2}}{2}"),
        (math.sqrt(3.0) / 2.0, "sqrt(3)/2", r"\frac{\sqrt{3}}{2}"),
        (1.0 / math.sqrt(3.0), "1/sqrt(3)", r"\frac{1}{\sqrt{3}}"),
        (math.sqrt(2.0 / 3.0), "sqrt(2/3)", r"\sqrt{\frac{2}{3}}"),
    ]
    for candidate_value, linear_token, latex_token in candidates:
        if abs(magnitude - candidate_value) < tol:
            return f"{sign}{linear_token}", f"{sign}{latex_token}"
    return None


def _vector_to_symbolic_subscript(
    axis: np.ndarray,
    *,
    tol: float = 1e-4,
) -> tuple[str, str] | None:
    axis = _canonicalize_axis(axis, tol=tol)
    linear_parts: list[str] = []
    latex_parts: list[str] = []
    for value in axis:
        token = _symbolic_component_token(float(value), tol=tol)
        if token is None:
            return None
        linear_parts.append(token[0])
        latex_parts.append(token[1])
    return ",".join(linear_parts), ",".join(latex_parts)


def _point_axis_metadata(
    *,
    axis_kind: str | None,
    axis_direction: tuple[int, int, int] | None,
    axis_parameter_values: tuple[float, float, float] | None,
    axis_symbolic_subscript_linear: str | None = None,
    axis_symbolic_subscript_latex: str | None = None,
) -> dict:
    if axis_kind == "direction" and axis_direction is not None:
        sub_linear = _direction_subscript(axis_direction)
        sub_latex = _direction_subscript(axis_direction)
        parameter_values = None
    elif axis_kind == "symbolic":
        sub_linear = axis_symbolic_subscript_linear
        sub_latex = axis_symbolic_subscript_latex
        parameter_values = axis_parameter_values
    elif axis_kind == "parameter":
        sub_linear = _axis_parameter_subscript(axis_parameter_values, latex=False)
        sub_latex = _axis_parameter_subscript(axis_parameter_values, latex=True)
        parameter_values = axis_parameter_values
    else:
        sub_linear = None
        sub_latex = None
        parameter_values = None

    return {
        "axis_subscript_linear": sub_linear,
        "axis_subscript_latex": sub_latex,
        # Keep a semantically neutral alias for future UI / export use. This
        # does not assert these are a finalized Euler-angle convention.
        "axis_parameter_values": parameter_values,
    }


def _format_float(value: float, tol: float = 1e-8) -> str:
    if abs(value - round(value)) < tol:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_float_latex(value: float, tol: float = 1e-8, max_denominator: int = 12) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < tol:
        return str(int(round(numeric)))

    frac = Fraction(numeric).limit_denominator(max_denominator)
    if abs(float(frac) - numeric) < 1e-6:
        return rf"\frac{{{frac.numerator}}}{{{frac.denominator}}}"

    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _direction_subscript(direction: tuple[int, int, int]) -> str:
    if any(abs(v) > 9 for v in direction):
        return f"{direction[0]},{direction[1]},{direction[2]}"
    return f"{direction[0]}{direction[1]}{direction[2]}"


def _to_latex_point_token(token: str) -> str:
    token = token.replace("alpha", r"\alpha")
    token = token.replace("beta", r"\beta")
    token = token.replace("gamma", r"\gamma")
    token = token.replace("∞", r"\infty ")
    return token


def format_point_seitz_symbol(
    hm_symbol: str,
    axis_kind: str | None,
    axis_direction: tuple[int, int, int] | None,
    axis_parameter_values: tuple[float, float, float] | None,
    rotation_power: int | None,
    axis_symbolic_subscript_linear: str | None = None,
) -> str:
    symbol = hm_symbol
    if rotation_power is not None:
        symbol += f"^{{{rotation_power}}}"

    if axis_kind == "direction" and axis_direction is not None:
        symbol += f"_{{{_direction_subscript(axis_direction)}}}"
    elif axis_kind == "symbolic" and axis_symbolic_subscript_linear is not None:
        symbol += f"_{{{axis_symbolic_subscript_linear}}}"
    elif axis_kind == "parameter":
        symbol += f"_{{{_axis_parameter_subscript(axis_parameter_values, latex=False)}}}"

    return symbol


def format_point_seitz_symbol_latex(
    hm_symbol: str,
    axis_kind: str | None,
    axis_direction: tuple[int, int, int] | None,
    axis_parameter_values: tuple[float, float, float] | None,
    rotation_power: int | None,
    axis_symbolic_subscript_latex: str | None = None,
) -> str:
    symbol = _to_latex_point_token(hm_symbol)
    if rotation_power is not None:
        symbol += f"^{{{rotation_power}}}"

    if axis_kind == "direction" and axis_direction is not None:
        symbol += f"_{{{_direction_subscript(axis_direction)}}}"
    elif axis_kind == "symbolic" and axis_symbolic_subscript_latex is not None:
        symbol += f"_{{{axis_symbolic_subscript_latex}}}"
    elif axis_kind == "parameter":
        symbol += f"_{{{_axis_parameter_subscript(axis_parameter_values, latex=True)}}}"

    return symbol


def _matrix_bytes_key(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype=np.float64).reshape(3, 3).tobytes()


def _restore_matrix_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float64).reshape(3, 3).copy()


@lru_cache(maxsize=4096)
def _describe_point_operation_cached(
    matrix_bytes: bytes,
    tol: float,
    max_order: int,
    max_axis_denom: int,
) -> dict:
    matrix = _restore_matrix_from_bytes(matrix_bytes)
    return _describe_point_operation_impl(
        matrix,
        tol=tol,
        max_order=max_order,
        max_axis_denom=max_axis_denom,
    )


def _describe_point_operation_impl(
    matrix: np.ndarray,
    *,
    tol: float = 1e-6,
    max_order: int = 120,
    max_axis_denom: int = 12,
) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    ortho_matrix = _orthogonalize_matrix(matrix)

    det = float(np.linalg.det(matrix))
    det_sign = 1 if det >= 0 else -1

    order = None
    try:
        order = _matrix_order(matrix, max_order=max_order, tol=tol)
    except ValueError:
        # In some transformed (non-Euclidean) bases, finite-order matrices can
        # be numerically noisy; use orthogonalized fallback for order probing.
        try:
            order = _matrix_order(ortho_matrix, max_order=max_order, tol=tol)
        except ValueError:
            spectral_order = _spectral_order_hint(matrix, max_order=max_order, tol=tol)
            if spectral_order is not None:
                order = spectral_order
            else:
                nearest_match = _nearest_known_point_operation(matrix, tol=tol)
                if nearest_match is not None:
                    return describe_point_operation(
                        nearest_match,
                        tol=min(float(tol), 1e-6),
                        max_order=max_order,
                        max_axis_denom=max_axis_denom,
                    )
                fallback_order = _fallback_low_order(ortho_matrix, tol=tol)
                if fallback_order is None:
                    raise
                order = fallback_order

    is_euclidean_orthogonal = np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3, rtol=0)

    hm_symbol: str
    axis = None
    rotation_power = None

    if order == 1:
        hm_symbol = "1"
    elif order == 2:
        if det_sign > 0:
            hm_symbol = "2"
            axis = _eigen_axis(matrix, 1.0, tol=tol)
        else:
            if np.allclose(matrix, -np.eye(3), atol=10 * tol, rtol=0):
                hm_symbol = "-1"
            else:
                hm_symbol = "m"
                axis = _eigen_axis(matrix, -1.0, tol=tol)
    else:
        target_eig = 1.0 if det_sign > 0 else -1.0
        axis = _eigen_axis(matrix, target_eig, tol=tol)

        if is_euclidean_orthogonal:
            power_num, fold = _rotation_fraction(ortho_matrix, axis, max_fold=max_order, tol=tol)
        else:
            power_num, fold = _rotation_fraction_from_eigenvalues(matrix, max_fold=max_order, tol=tol)

        if fold <= 1:
            power_num, fold = _rotation_fraction_from_eigenvalues(matrix, max_fold=max_order, tol=tol)

        if fold <= 1:
            fold = order
            power_num = 1

        # For finite-order operations, denominator should be compatible with group order.
        if fold > 1 and order % fold != 0 and fold % order != 0:
            fold = order
            power_num = 1

        if det_sign < 0:
            hm_fold = _hm_improper_fold(matrix, fold, tol=tol)
            hm_symbol = f"-{hm_fold}"
            proper_component = -matrix
            proper_ortho = _orthogonalize_matrix(proper_component)
            if is_euclidean_orthogonal:
                branch_num, branch_fold = _rotation_fraction(
                    proper_ortho,
                    axis,
                    max_fold=max_order,
                    tol=tol,
                )
            else:
                branch_num, branch_fold = _rotation_fraction_from_eigenvalues(
                    proper_component,
                    max_fold=max_order,
                    tol=tol,
                )
            if branch_fold != hm_fold:
                branch_num = power_num
        else:
            hm_symbol = f"{fold}"
            branch_num = power_num
        if (hm_fold if det_sign < 0 else fold) > 2:
            rotation_power = branch_num

    axis_kind = None
    axis_direction = None
    axis_parameter_values = None
    axis_symbolic_subscript_linear = None
    axis_symbolic_subscript_latex = None
    if axis is not None:
        axis_direction = _vector_to_integer_direction(axis, max_denom=max_axis_denom, tol=max(5 * tol, 1e-4))
        if axis_direction is not None:
            axis_kind = "direction"
        else:
            axis_parameter_values = _axis_to_parameter_values(axis, tol=max(tol, 1e-8))
            symbolic_subscript = _vector_to_symbolic_subscript(axis, tol=max(5 * tol, 1e-4))
            if symbolic_subscript is not None:
                axis_kind = "symbolic"
                axis_symbolic_subscript_linear, axis_symbolic_subscript_latex = symbolic_subscript
            else:
                axis_kind = "parameter"

    known_token_info = _nearest_known_point_operation_token(matrix, tol=tol)
    if known_token_info is not None:
        hm_symbol = known_token_info["hm_symbol"]
        rotation_power = known_token_info["rotation_power"]
        axis_kind = known_token_info["axis_kind"]
        axis_direction = known_token_info["axis_direction"]
        axis_parameter_values = known_token_info["axis_parameter_values"]
        if known_token_info["axis_vector"] is None:
            axis = None
        else:
            axis = np.asarray(known_token_info["axis_vector"], dtype=float)

    symbol = format_point_seitz_symbol(
        hm_symbol=hm_symbol,
        axis_kind=axis_kind,
        axis_direction=axis_direction,
        axis_parameter_values=axis_parameter_values,
        rotation_power=rotation_power,
        axis_symbolic_subscript_linear=axis_symbolic_subscript_linear,
    )
    symbol_latex = format_point_seitz_symbol_latex(
        hm_symbol=hm_symbol,
        axis_kind=axis_kind,
        axis_direction=axis_direction,
        axis_parameter_values=axis_parameter_values,
        rotation_power=rotation_power,
        axis_symbolic_subscript_latex=axis_symbolic_subscript_latex,
    )
    axis_metadata = _point_axis_metadata(
        axis_kind=axis_kind,
        axis_direction=axis_direction,
        axis_parameter_values=axis_parameter_values,
        axis_symbolic_subscript_linear=axis_symbolic_subscript_linear,
        axis_symbolic_subscript_latex=axis_symbolic_subscript_latex,
    )

    return {
        "hm_symbol": hm_symbol,
        "order": order,
        "rotation_power": rotation_power,
        "axis_kind": axis_kind,
        "axis_vector": None if axis is None else tuple(float(v) for v in axis),
        "axis_direction": axis_direction,
        "axis_euler_deg": None,
        "axis_parameter_values": axis_parameter_values,
        **axis_metadata,
        "symbol": symbol,
        "symbol_latex": symbol_latex,
    }


def describe_point_operation(
    matrix: np.ndarray,
    *,
    tol: float = 1e-6,
    max_order: int = 120,
    max_axis_denom: int = 12,
) -> dict:
    return copy.deepcopy(
        _describe_point_operation_cached(
            _matrix_bytes_key(matrix),
            float(tol),
            int(max_order),
            int(max_axis_denom),
        )
    )


def format_translation_tau(translation: np.ndarray, tol: float = 1e-8) -> str:
    vec = np.mod(np.asarray(translation, dtype=float), 1.0)
    vec[np.isclose(vec, 1.0, atol=tol)] = 0.0
    values = ",".join(_format_float(float(v), tol=tol) for v in vec)
    return f"tau_{{{values}}}"


def format_translation_tau_latex(translation: np.ndarray, tol: float = 1e-8) -> str:
    vec = np.mod(np.asarray(translation, dtype=float), 1.0)
    vec[np.isclose(vec, 1.0, atol=tol)] = 0.0
    values = ",".join(_format_float_latex(float(v), tol=tol) for v in vec)
    return rf"\tau_{{({values})}}"


def _fold_from_hm_symbol(hm_symbol: str) -> int | None:
    if hm_symbol in {"1", "-1", "m"}:
        return None
    if hm_symbol.startswith("-"):
        body = hm_symbol[1:]
    else:
        body = hm_symbol
    return int(body) if body.isdigit() else None


def canonicalize_group_seitz_descriptions(
    descriptions: list[dict],
    *,
    tol: float = 1e-6,
    max_axis_denom: int = 12,
) -> list[dict]:
    """
    Canonicalize axis direction for operations in one group.

    For operations on the same axis line, force one shared axis direction.
    If a label would flip axis sign for n-fold rotations (n>2), convert power as m -> n-m.
    """
    if not descriptions:
        return []

    output = copy.deepcopy(descriptions)
    for component in ("spin", "real"):
        canonical_axis_by_key: dict[tuple[float, float, float], np.ndarray] = {}

        for item in output:
            info = item[component]
            axis_vec = info.get("axis_vector")
            if axis_vec is None:
                continue
            axis = np.asarray(axis_vec, dtype=float)
            if np.linalg.norm(axis) < tol:
                continue
            axis_key_vec = _canonicalize_axis(axis, tol=tol)
            axis_key = tuple(np.round(axis_key_vec, 6))
            canonical_axis_by_key.setdefault(axis_key, axis_key_vec)

        for item in output:
            info = item[component]
            axis_vec = info.get("axis_vector")
            if axis_vec is None:
                continue

            original_axis = np.asarray(axis_vec, dtype=float)
            if np.linalg.norm(original_axis) < tol:
                continue
            axis_key = tuple(np.round(_canonicalize_axis(original_axis, tol=tol), 6))
            canonical_axis = canonical_axis_by_key[axis_key]

            flipped = float(np.dot(original_axis, canonical_axis)) < -tol

            fold = _fold_from_hm_symbol(info["hm_symbol"])
            if flipped and fold is not None and fold > 2 and info.get("rotation_power") is not None:
                m = int(info["rotation_power"]) % fold
                if m == 0:
                    m = fold
                m = fold - m
                if m == 0:
                    m = fold
                info["rotation_power"] = m

            info["axis_vector"] = tuple(float(v) for v in canonical_axis)
            axis_direction = _vector_to_integer_direction(
                canonical_axis,
                max_denom=max_axis_denom,
                tol=max(5 * tol, 1e-4),
            )
            axis_symbolic_subscript_linear = None
            axis_symbolic_subscript_latex = None
            if axis_direction is not None:
                info["axis_kind"] = "direction"
                info["axis_direction"] = axis_direction
                info["axis_euler_deg"] = None
                info["axis_parameter_values"] = None
            else:
                info["axis_parameter_values"] = _axis_to_parameter_values(
                    canonical_axis,
                    tol=max(tol, 1e-8),
                )
                symbolic_subscript = _vector_to_symbolic_subscript(
                    canonical_axis,
                    tol=max(5 * tol, 1e-4),
                )
                if symbolic_subscript is not None:
                    info["axis_kind"] = "symbolic"
                    axis_symbolic_subscript_linear, axis_symbolic_subscript_latex = symbolic_subscript
                else:
                    info["axis_kind"] = "parameter"
                info["axis_direction"] = None
                info["axis_euler_deg"] = None

            info["symbol"] = format_point_seitz_symbol(
                hm_symbol=info["hm_symbol"],
                axis_kind=info["axis_kind"],
                axis_direction=info.get("axis_direction"),
                axis_parameter_values=info.get("axis_parameter_values"),
                rotation_power=info.get("rotation_power"),
                axis_symbolic_subscript_linear=axis_symbolic_subscript_linear,
            )
            info["symbol_latex"] = format_point_seitz_symbol_latex(
                hm_symbol=info["hm_symbol"],
                axis_kind=info["axis_kind"],
                axis_direction=info.get("axis_direction"),
                axis_parameter_values=info.get("axis_parameter_values"),
                rotation_power=info.get("rotation_power"),
                axis_symbolic_subscript_latex=axis_symbolic_subscript_latex,
            )
            info.update(
                _point_axis_metadata(
                    axis_kind=info.get("axis_kind"),
                    axis_direction=info.get("axis_direction"),
                    axis_parameter_values=info.get("axis_parameter_values"),
                    axis_symbolic_subscript_linear=axis_symbolic_subscript_linear,
                    axis_symbolic_subscript_latex=axis_symbolic_subscript_latex,
                )
            )

    for item in output:
        item["symbol"] = f"{{ {item['spin']['symbol']} || {item['real']['symbol']} | {item['translation_symbol']} }}"
        item["symbol_latex"] = (
            rf"\left\{{ {item['spin']['symbol_latex']} \,\middle\|\, "
            rf"{item['real']['symbol_latex']} \,\middle|\, "
            rf"{item['translation_symbol_latex']} \right\}}"
        )

    return output


def describe_spin_space_operation(
    spin_rotation: np.ndarray,
    real_rotation: np.ndarray,
    translation: np.ndarray,
    *,
    tol: float = 1e-6,
    max_order: int = 120,
    max_axis_denom: int = 12,
) -> dict:
    spin_info = describe_point_operation(
        spin_rotation, tol=tol, max_order=max_order, max_axis_denom=max_axis_denom
    )
    real_info = describe_point_operation(
        real_rotation, tol=tol, max_order=max_order, max_axis_denom=max_axis_denom
    )
    tau = format_translation_tau(translation, tol=tol)
    tau_latex = format_translation_tau_latex(translation, tol=tol)
    symbol = f"{{ {spin_info['symbol']} || {real_info['symbol']} | {tau} }}"
    symbol_latex = (
        rf"\left\{{ {spin_info['symbol_latex']} \,\middle\|\, "
        rf"{real_info['symbol_latex']} \,\middle|\, {tau_latex} \right\}}"
    )
    return {
        "spin": spin_info,
        "real": real_info,
        "translation": tuple(float(v) for v in np.asarray(translation, dtype=float)),
        "translation_symbol": tau,
        "translation_symbol_latex": tau_latex,
        "symbol": symbol,
        "symbol_latex": symbol_latex,
    }
