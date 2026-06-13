from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from findspingroup.data.SG_SYMBOL import SGdisc, SGgeneratorDict
from findspingroup.utils.seitz_symbol import (
    _axis_parameter_subscript,
    calibrated_symbol_tol,
    describe_point_operation,
)

if TYPE_CHECKING:
    from findspingroup.structure.group import SpinSpaceGroup, SpinSpaceGroupOperation


def _normalize_mod1(vec: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    vec = np.mod(np.asarray(vec, dtype=float), 1.0)
    vec[np.isclose(vec, 1.0, atol=tol)] = 0.0
    vec[np.isclose(vec, 0.0, atol=tol)] = 0.0
    return vec


def _same_translation_mod1(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    aa = _normalize_mod1(a)
    bb = _normalize_mod1(b)
    diff = np.abs(aa - bb)
    diff = np.minimum(diff, 1.0 - diff)
    return bool(np.max(diff) < tol)


def _parse_sg_generator_ops(sg_num: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Return (named_ops, centering_translations) for an SG."""
    info = SGgeneratorDict[sg_num]
    named_ops: list[tuple[np.ndarray, np.ndarray]] = []
    centering: list[np.ndarray] = []

    for index in range((len(info) - 1) // 2):
        rot_flat, trans = eval(info[2 * index + 2])
        rot = np.array(rot_flat, dtype=float).reshape(3, 3)
        trans_vec = np.array(trans, dtype=float)
        if np.allclose(rot, np.eye(3), atol=1e-10):
            if not np.allclose(_normalize_mod1(trans_vec), np.zeros(3), atol=1e-10):
                centering.append(_normalize_mod1(trans_vec))
        else:
            named_ops.append((rot, _normalize_mod1(trans_vec)))

    return named_ops, centering


def _transport_standard_real_op_to_current_basis(
    rotation: np.ndarray,
    translation: np.ndarray,
    current_to_standard: np.ndarray,
    current_to_standard_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transport a real-space operation written in the standard-setting basis into
    the current basis used by `basis_mode="current"`.

    If p_std = T p_cur + s, then:
        R_cur = T^{-1} R_std T
        t_cur = T^{-1} (t_std + (R_std - I) s)
    """
    current_to_standard = np.asarray(current_to_standard, dtype=float)
    current_to_standard_shift = np.asarray(current_to_standard_shift, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)

    standard_to_current = np.linalg.inv(current_to_standard)
    current_rotation = standard_to_current @ rotation @ current_to_standard
    current_translation = standard_to_current @ (
        translation + (rotation - np.eye(3)) @ current_to_standard_shift
    )
    return current_rotation, _normalize_mod1(current_translation)


def _transport_standard_generators_to_current_basis(
    named_ops: list[tuple[np.ndarray, np.ndarray]],
    centering_trans: list[np.ndarray],
    current_to_standard: np.ndarray,
    current_to_standard_shift: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    current_named_ops = [
        _transport_standard_real_op_to_current_basis(
            rotation,
            translation,
            current_to_standard,
            current_to_standard_shift,
        )
        for rotation, translation in named_ops
    ]
    current_centering = [
        _transport_standard_real_op_to_current_basis(
            np.eye(3),
            translation,
            current_to_standard,
            current_to_standard_shift,
        )[1]
        for translation in centering_trans
    ]
    return current_named_ops, current_centering


def _compose_setting_transform(
    source_matrix: np.ndarray,
    source_shift: np.ndarray,
    target_matrix: np.ndarray,
    target_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = target_matrix @ np.linalg.inv(source_matrix)
    shift = target_shift - target_matrix @ np.linalg.inv(source_matrix) @ source_shift
    return np.asarray(transform, dtype=float), np.asarray(shift, dtype=float)


def _real_generator_tokens(sg_num: int, named_count: int) -> list[str]:
    tokens = list(SGdisc[sg_num][1:])
    if named_count == 0:
        return []
    if len(tokens) == named_count:
        return tokens

    # Trigonal/hex notation can contain inserted "1" placeholders.
    filtered = [tok for tok in tokens if tok != "1"]
    if len(filtered) == named_count:
        return filtered

    if len(tokens) > named_count:
        return tokens[:named_count]
    return tokens + ["?"] * (named_count - len(tokens))


def _to_latex_token(token: str) -> str:
    token = re.sub(r"alpha(\d+)", r"\\alpha_{\1}", token)
    token = re.sub(r"beta(\d+)", r"\\beta_{\1}", token)
    token = re.sub(r"gamma(\d+)", r"\\gamma_{\1}", token)
    token = token.replace("alpha", r"\alpha")
    token = token.replace("beta", r"\beta")
    token = token.replace("gamma", r"\gamma")
    token = re.sub(r"-(\d+)", r"\\bar{\1}", token)
    return token


def _canonical_parameter_axis_key(info: dict, tol: float = 1e-8) -> tuple[float, float, float] | None:
    values = info.get("axis_parameter_values")
    if values is None:
        axis_vector = info.get("axis_vector")
        if axis_vector is None:
            return None
        values = axis_vector

    axis = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < tol:
        return None
    axis = axis / norm
    for value in axis:
        if abs(float(value)) > tol:
            if value < 0:
                axis = -axis
            break
    axis[np.abs(axis) < tol] = 0.0
    return tuple(float(v) for v in np.round(axis, 8))


class _SymbolParameterAxisNamer:
    """Assign stable alpha/beta/gamma suffixes to distinct free directions in one symbol."""

    def __init__(self) -> None:
        self._keys: list[tuple[float, float, float]] = []
        self._index_by_key: dict[tuple[float, float, float], int] = {}

    def add(self, info: dict | None) -> None:
        if info is None or info.get("axis_kind") != "parameter":
            return
        key = _canonical_parameter_axis_key(info)
        if key is None or key in self._index_by_key:
            return
        self._index_by_key[key] = len(self._keys) + 1
        self._keys.append(key)

    @property
    def uses_numbered_labels(self) -> bool:
        return len(self._keys) > 1

    def parameter_index(self, info: dict) -> int | None:
        if not self.uses_numbered_labels:
            return None
        key = _canonical_parameter_axis_key(info)
        if key is None:
            return None
        return self._index_by_key.get(key)


def _point_group_token_from_real_token(token: str) -> str:
    """Drop glide/screw translational decoration while preserving HM structure."""
    point_token = re.sub(r"([1-6])_(?:\d+|\{\d+\})", r"\1", token)
    if point_token in {"a", "b", "c", "d", "n"}:
        return "m"
    return point_token


def _format_number_linear(value: float, tol: float = 1e-8, max_den: int = 12) -> str:
    if abs(value) < tol:
        return "0"
    if abs(value - round(value)) < tol:
        return str(int(round(value)))
    frac = Fraction(float(value)).limit_denominator(max_den)
    if abs(float(frac) - float(value)) < 1e-6:
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_number_latex(value: float, tol: float = 1e-8, max_den: int = 12) -> str:
    if abs(value) < tol:
        return "0"
    if abs(value - round(value)) < tol:
        return str(int(round(value)))
    frac = Fraction(float(value)).limit_denominator(max_den)
    if abs(float(frac) - float(value)) < 1e-6:
        if frac.denominator == 1:
            return str(frac.numerator)
        return rf"\frac{{{frac.numerator}}}{{{frac.denominator}}}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_vector_linear(vec: np.ndarray) -> str:
    vv = _normalize_mod1(vec)
    return "(" + ",".join(_format_number_linear(float(x)) for x in vv) + ")"


def _format_vector_latex(vec: np.ndarray) -> str:
    vv = _normalize_mod1(vec)
    return "(" + ",".join(_format_number_latex(float(x)) for x in vv) + ")"


def _find_real_operation(
    ops: list["SpinSpaceGroupOperation"],
    target_rot: np.ndarray,
    target_trans: np.ndarray,
    tol: float = 1e-4,
) -> "SpinSpaceGroupOperation | None":
    for op in ops:
        if np.allclose(op.rotation, target_rot, atol=tol, rtol=0) and _same_translation_mod1(
            op.translation, target_trans, tol=tol
        ):
            return op
    return None


def _select_preferred_translation_match(
    ops: list["SpinSpaceGroupOperation"],
    target_trans: np.ndarray,
    tol: float = 1e-4,
    *,
    identity_real_ops: list["SpinSpaceGroupOperation"] | None = None,
) -> "SpinSpaceGroupOperation | None":
    target = np.asarray(target_trans, dtype=float)
    identity_rot = np.eye(3)
    search_ops = identity_real_ops if identity_real_ops is not None else ops

    exact_candidates = [
        op
        for op in search_ops
        if np.allclose(op.rotation, identity_rot, atol=tol, rtol=0)
        and np.allclose(np.asarray(op.translation, dtype=float), target, atol=tol, rtol=0)
    ]

    candidates = [
        op
        for op in search_ops
        if np.allclose(op.rotation, identity_rot, atol=tol, rtol=0)
        and _same_translation_mod1(op.translation, target_trans, tol=tol)
    ]

    nonzero_axes = np.flatnonzero(np.abs(target) > tol)
    is_axis_target = len(nonzero_axes) == 1 and np.allclose(
        np.abs(target[nonzero_axes[0]]),
        1.0,
        atol=tol,
        rtol=0,
    )

    def _score(op: "SpinSpaceGroupOperation") -> tuple:
        raw = np.asarray(op.translation, dtype=float)
        abs_raw = np.abs(raw)
        norm = float(np.linalg.norm(raw))
        is_zero = norm < tol

        if is_axis_target:
            axis = int(nonzero_axes[0])
            axis_component = abs_raw[axis]
            if axis_component < tol:
                return (2, float("inf"), float("inf"), float("inf"), tuple(np.round(abs_raw, 6)))
            off_axis = float(np.sum(abs_raw) - axis_component)
            return (
                0 if not is_zero else 1,
                off_axis,
                axis_component,
                norm,
                tuple(np.round(abs_raw, 6)),
            )

        return (
            0 if not is_zero else 1,
            norm,
            tuple(np.round(abs_raw, 6)),
        )

    exact_nonzero_candidates = [op for op in exact_candidates if np.linalg.norm(np.asarray(op.translation, dtype=float)) >= tol]
    if exact_nonzero_candidates:
        return min(exact_nonzero_candidates, key=_score)

    exact_zero_candidates = [op for op in exact_candidates if np.linalg.norm(np.asarray(op.translation, dtype=float)) < tol]
    if exact_zero_candidates and not is_axis_target:
        return min(exact_zero_candidates, key=_score)

    nonzero_candidates = []
    for op in candidates:
        raw = np.asarray(op.translation, dtype=float)
        if np.linalg.norm(raw) < tol:
            continue
        if is_axis_target:
            axis = int(nonzero_axes[0])
            if abs(raw[axis]) < tol:
                continue
        nonzero_candidates.append(op)

    if nonzero_candidates:
        return min(nonzero_candidates, key=_score)

    if is_axis_target:
        axis = int(nonzero_axes[0])
        axis_candidates = []
        for op in search_ops:
            if not np.allclose(op.rotation, identity_rot, atol=tol, rtol=0):
                continue
            raw = np.asarray(op.translation, dtype=float)
            abs_raw = np.abs(raw)
            axis_component = abs_raw[axis]
            if axis_component < tol:
                continue
            axis_candidates.append(op)

        if axis_candidates:
            return min(axis_candidates, key=_score)

    zero_candidates = [op for op in candidates if np.linalg.norm(np.asarray(op.translation, dtype=float)) < tol]
    if zero_candidates:
        return min(zero_candidates, key=_score)

    if candidates:
        return min(candidates, key=_score)

    return None


def _select_preferred_primitive_translation_match(
    ops: list["SpinSpaceGroupOperation"],
    axis_index: int,
    tol: float = 1e-4,
    *,
    identity_real_ops: list["SpinSpaceGroupOperation"] | None = None,
) -> "SpinSpaceGroupOperation | None":
    """
    Select the shortest nonzero identity-rotation translation that lies purely
    along one crystallographic axis. If no such translation exists, fall back
    to the identity translation.
    """
    identity_rot = np.eye(3)
    axis_candidates: list["SpinSpaceGroupOperation"] = []
    zero_candidates: list["SpinSpaceGroupOperation"] = []
    search_ops = identity_real_ops if identity_real_ops is not None else ops

    for op in search_ops:
        if not np.allclose(op.rotation, identity_rot, atol=tol, rtol=0):
            continue
        raw = np.asarray(op.translation, dtype=float)
        norm = float(np.linalg.norm(raw))
        if norm < tol:
            zero_candidates.append(op)
            continue

        abs_raw = np.abs(raw)
        axis_component = abs_raw[axis_index]
        off_axis = float(np.sum(abs_raw) - axis_component)
        if axis_component < tol or off_axis >= tol:
            continue
        axis_candidates.append(op)

    if axis_candidates:
        return min(
            axis_candidates,
            key=lambda op: (
                abs(float(np.asarray(op.translation, dtype=float)[axis_index])),
                float(np.linalg.norm(np.asarray(op.translation, dtype=float))),
                tuple(np.round(np.abs(np.asarray(op.translation, dtype=float)), 6)),
            ),
        )

    if zero_candidates:
        return min(
            zero_candidates,
            key=lambda op: tuple(np.round(np.abs(np.asarray(op.translation, dtype=float)), 6)),
        )

    return None


def _symbol_type(it: int, ik: int) -> str:
    # Definitions from the paper:
    # t-type: i_k = 1 (translation-preserving), k-type: i_t = 1 and i_k > 1, g-type: both > 1.
    if ik == 1:
        return "t"
    if it == 1 and ik > 1:
        return "k"
    return "g"


def _transform_to_g0_basis(ssg: "SpinSpaceGroup") -> "SpinSpaceGroup":
    ssg_g0 = ssg.transform(ssg.transformation_to_G0std, ssg.origin_shift_to_G0std)
    basis_fix = ssg.transformation_to_G0std_id @ np.linalg.inv(ssg.transformation_to_G0std)
    ssg_g0 = ssg_g0.transform(basis_fix, np.array([0, 0, 0]), frac=False)
    return ssg_g0.transform_spin(np.linalg.inv(ssg.n_spin_part_std_transformation))


def _transform_to_l0_basis(ssg: "SpinSpaceGroup") -> "SpinSpaceGroup":
    ssg_l0 = ssg.transform(ssg.transformation_to_L0std, ssg.origin_shift_to_L0std)
    return ssg_l0.transform_spin(np.linalg.inv(ssg.n_spin_part_std_transformation))


def _canonical_spin_info_map(ssg: "SpinSpaceGroup") -> dict[int, dict]:
    # Use canonicalized Seitz descriptions so axis direction is consistent in one group.
    descriptions = ssg.seitz_descriptions
    return {id(op): desc["spin"] for op, desc in zip(ssg.ops, descriptions)}


def _default_centering_vectors(bravais: str) -> list[tuple[str, np.ndarray]]:
    vectors: dict[str, list[np.ndarray]] = {
        "P": [],
        "A": [np.array([0.0, 0.5, 0.5])],
        "B": [np.array([0.5, 0.0, 0.5])],
        "C": [np.array([0.5, 0.5, 0.0])],
        "I": [np.array([0.5, 0.5, 0.5])],
        "F": [np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5])],
        "R": [np.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), np.array([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])],
    }
    data = vectors.get(bravais, [])
    return [(f"b_{i+1}", v) for i, v in enumerate(data)]


def _op_key(op: "SpinSpaceGroupOperation", ndigits: int = 6) -> tuple:
    return (
        tuple(np.round(op.spin_rotation.flatten(), ndigits)),
        tuple(np.round(op.rotation.flatten(), ndigits)),
        tuple(np.round(_normalize_mod1(op.translation), ndigits)),
    )


def _is_identity_operation(op: "SpinSpaceGroupOperation", tol: float = 1e-8) -> bool:
    identity = np.eye(3)
    return (
        np.allclose(op.spin_rotation, identity, atol=tol, rtol=0)
        and np.allclose(op.rotation, identity, atol=tol, rtol=0)
        and np.allclose(_normalize_mod1(op.translation, tol=tol), np.zeros(3), atol=tol, rtol=0)
    )


def _serialized_generator_operation(
    op: "SpinSpaceGroupOperation | None",
    *,
    source: str,
    label: str,
    tol: float,
) -> dict | None:
    if op is None or _is_identity_operation(op, tol=tol):
        return None
    return {
        "source": source,
        "label": label,
        "spin_rotation": np.asarray(op.spin_rotation, dtype=float).tolist(),
        "real_rotation": np.asarray(op.rotation, dtype=float).tolist(),
        "translation": [float(v) for v in _normalize_mod1(op.translation, tol=tol)],
    }


def _append_generator_operation(
    rows: list[dict],
    seen: set[tuple],
    op: "SpinSpaceGroupOperation | None",
    *,
    source: str,
    label: str,
    tol: float,
) -> None:
    row = _serialized_generator_operation(op, source=source, label=label, tol=tol)
    if row is None:
        return
    key = (
        tuple(np.round(np.asarray(row["spin_rotation"], dtype=float).flatten(), 6)),
        tuple(np.round(np.asarray(row["real_rotation"], dtype=float).flatten(), 6)),
        tuple(np.round(np.asarray(row["translation"], dtype=float), 6)),
    )
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def _closure_from_generators(
    generators: list["SpinSpaceGroupOperation"], max_size: int = 4096
) -> set[tuple]:
    if not generators:
        return set()

    identity = generators[0].identity()
    seen = {_op_key(identity)}
    queue = [identity]

    while queue and len(seen) < max_size:
        cur = queue.pop(0)
        for gen in generators:
            for nxt in (cur @ gen, cur @ gen.inv()):
                key = _op_key(nxt)
                if key not in seen:
                    seen.add(key)
                    queue.append(nxt)
    return seen


def _k_generator_sort_key(op: "SpinSpaceGroupOperation") -> tuple:
    info = describe_point_operation(
        op.spin_rotation,
        tol=calibrated_symbol_tol(1e-6),
        max_order=120,
        max_axis_denom=12,
    )
    direction = info.get("axis_direction")
    if direction is None:
        axis_positions = (3,)
        direction_strength = (0, 0, 0)
        direction_signs = (0, 0, 0)
    else:
        direction = tuple(int(v) for v in direction)
        axis_positions = tuple(index for index, value in enumerate(direction) if value != 0)
        direction_strength = tuple(-abs(value) for value in direction)
        direction_signs = tuple(-value for value in direction)

    translation = tuple(np.round(_normalize_mod1(op.translation), 6))
    rotation = tuple(np.round(op.rotation.flatten(), 6))
    spin_rotation = tuple(np.round(op.spin_rotation.flatten(), 6))
    return (
        axis_positions,
        direction_strength,
        direction_signs,
        translation,
        spin_rotation,
        rotation,
    )


def _spin_operation_order(op: "SpinSpaceGroupOperation") -> int:
    info = describe_point_operation(
        op.spin_rotation,
        tol=calibrated_symbol_tol(1e-6),
        max_order=120,
        max_axis_denom=12,
    )
    order = info.get("order")
    return int(order) if order is not None else 0


def _minimal_k_translation_generators(
    ops: list["SpinSpaceGroupOperation"],
    *,
    free_generators: list["SpinSpaceGroupOperation"] | None = None,
) -> list["SpinSpaceGroupOperation"]:
    candidates = [op for op in ops if not np.allclose(op.spin_rotation, np.eye(3), atol=1e-4)]
    if not candidates:
        return []

    candidates = sorted(candidates, key=_k_generator_sort_key)

    target_keys = {_op_key(op) for op in candidates}
    base_generators = list(free_generators or [])
    selected: list["SpinSpaceGroupOperation"] = []
    closure: set[tuple] = _closure_from_generators(base_generators)
    available = list(candidates)

    while not target_keys.issubset(closure):
        ranked: list[tuple[tuple, "SpinSpaceGroupOperation", set[tuple]]] = []
        remaining = target_keys - closure
        for op in available:
            if _op_key(op) in closure:
                continue
            trial_closure = _closure_from_generators(base_generators + selected + [op])
            gain = len(remaining & trial_closure)
            if gain == 0:
                continue
            ranked.append(
                (
                    (-gain, -_spin_operation_order(op), _k_generator_sort_key(op)),
                    op,
                    trial_closure,
                )
            )
        if not ranked:
            break
        _, op, closure = min(ranked, key=lambda item: item[0])
        selected.append(op)
        available = [candidate for candidate in available if candidate is not op]
        if target_keys.issubset(closure):
            break

    if not target_keys.issubset(closure):
        return candidates

    pruned = list(selected)
    for op in sorted(selected, key=_k_generator_sort_key):
        if len(pruned) == 1:
            break
        trial = [candidate for candidate in pruned if candidate is not op]
        if target_keys.issubset(_closure_from_generators(base_generators + trial)):
            pruned = trial

    return sorted(pruned, key=_k_generator_sort_key)


def _pure_centering_translation_generators(
    identity_real_ops: list["SpinSpaceGroupOperation"],
    centering_trans: list[np.ndarray],
    *,
    tol: float,
) -> list["SpinSpaceGroupOperation"]:
    identity = np.eye(3)
    generators: list["SpinSpaceGroupOperation"] = []
    seen: set[tuple[float, float, float]] = set()

    for target in centering_trans:
        matches = [
            op
            for op in identity_real_ops
            if np.allclose(op.spin_rotation, identity, atol=tol, rtol=0)
            and np.allclose(op.rotation, identity, atol=tol, rtol=0)
            and _same_translation_mod1(op.translation, target, tol=tol)
        ]
        if not matches:
            continue
        match = min(
            matches,
            key=lambda op: (
                float(np.linalg.norm(np.asarray(op.translation, dtype=float))),
                tuple(np.round(_normalize_mod1(op.translation), 6)),
            ),
        )
        key = tuple(float(v) for v in np.round(_normalize_mod1(match.translation), 6))
        if key in seen:
            continue
        seen.add(key)
        generators.append(match)

    return sorted(generators, key=_k_generator_sort_key)


def _minimal_spin_translation_generators(
    ops: list["SpinSpaceGroupOperation"],
) -> list["SpinSpaceGroupOperation"]:
    candidates = [op for op in ops if not _is_identity_operation(op, tol=1e-6)]
    if not candidates:
        return []

    candidates = sorted(candidates, key=_k_generator_sort_key)

    target_keys = {_op_key(op) for op in candidates}
    selected: list["SpinSpaceGroupOperation"] = []
    closure: set[tuple] = set()

    for op in candidates:
        if _op_key(op) in closure:
            continue
        selected.append(op)
        closure = _closure_from_generators(selected)
        if target_keys.issubset(closure):
            break

    if not target_keys.issubset(closure):
        return candidates
    return sorted(selected, key=_k_generator_sort_key)


def _direction_subscript(direction: tuple[int, int, int]) -> str:
    if any(abs(v) > 9 for v in direction):
        return f"{direction[0]},{direction[1]},{direction[2]}"
    return f"{direction[0]}{direction[1]}{direction[2]}"


def _axis_subscript_from_info(
    info: dict,
    *,
    latex: bool = False,
    parameter_namer: _SymbolParameterAxisNamer | None = None,
) -> str:
    axis_kind = info.get("axis_kind")
    if axis_kind == "direction" and info.get("axis_direction") is not None:
        direction = tuple(int(v) for v in info["axis_direction"])
        return _direction_subscript(direction)
    if axis_kind == "parameter":
        parameter_index = None if parameter_namer is None else parameter_namer.parameter_index(info)
        return _axis_parameter_subscript(
            info.get("axis_parameter_values"),
            latex=latex,
            parameter_index=parameter_index,
        )
    if axis_kind == "symbolic":
        # International-symbol LaTeX historically keeps compact symbolic axis
        # components as text-like tokens (e.g. sqrt(3)/2) instead of expanding
        # them into full fractions. Keep that contract; parameter placeholders
        # are the only axis labels rewritten here.
        key = "axis_subscript_linear"
        if info.get(key):
            return str(info[key])
    return _axis_parameter_subscript(None, latex=latex)


def _point_token_from_info(
    info: dict | None,
    *,
    latex: bool = False,
    parameter_namer: _SymbolParameterAxisNamer | None = None,
) -> str:
    if info is None:
        return "1"
    if info.get("unresolved"):
        return "?"

    token = _to_latex_token(str(info.get("hm_symbol", "?"))) if latex else str(info.get("hm_symbol", "?"))
    rotation_power = info.get("rotation_power")
    if rotation_power is not None:
        token += f"^{{{int(rotation_power)}}}"

    axis_kind = info.get("axis_kind")
    if axis_kind in {"direction", "parameter", "symbolic"}:
        subscript = _axis_subscript_from_info(
            info,
            latex=latex,
            parameter_namer=parameter_namer,
        )
        token += f"_{{{subscript}}}"
    return token


def _spin_only_suffix_point_info(
    ssg_basis: "SpinSpaceGroup", *, tol: float = 1e-6, max_axis_denom: int = 12
) -> tuple[str, dict] | None:
    symbol_tol = calibrated_symbol_tol(tol)
    if ssg_basis.conf == "Noncoplanar":
        return None

    if ssg_basis.conf == "Collinear":
        candidate = None
        for op in ssg_basis.sog:
            if np.allclose(op.spin_rotation, np.eye(3), atol=tol):
                continue
            if np.linalg.det(op.spin_rotation) > 0:
                candidate = op
                break
        if candidate is None:
            return None
        info = describe_point_operation(
            candidate.spin_rotation, tol=symbol_tol, max_order=120, max_axis_denom=max_axis_denom
        )
        return "infinity_m", info

    # Coplanar: use mirror normal direction in spin space.
    candidate = None
    for op in ssg_basis.sog:
        if np.allclose(op.spin_rotation, np.eye(3), atol=tol):
            continue
        if np.linalg.det(op.spin_rotation) < 0:
            candidate = op
            break
    if candidate is None:
        return None

    info = describe_point_operation(
        candidate.spin_rotation,
        tol=symbol_tol,
        max_order=120,
        max_axis_denom=max_axis_denom,
    )
    return "mirror", info


def _format_spin_only_suffix(
    suffix_info: tuple[str, dict] | None,
    *,
    parameter_namer: _SymbolParameterAxisNamer | None = None,
) -> tuple[str, str]:
    if suffix_info is None:
        return "", ""

    kind, info = suffix_info
    sub_linear = _axis_subscript_from_info(info, latex=False, parameter_namer=parameter_namer)
    sub_latex = _axis_subscript_from_info(info, latex=True, parameter_namer=parameter_namer)
    if kind == "infinity_m":
        return f"∞_{{{sub_linear}}}m|1", rf"^{{\infty_{{{sub_latex}}}m}}1"
    return f"m_{{{sub_linear}}}|1", rf"^{{m_{{{sub_latex}}}}}1"


def build_international_symbol(
    ssg: "SpinSpaceGroup",
    tol: float = 1e-4,
    *,
    basis_mode: str = "standard",
) -> dict:
    """
    Build international SSG symbol in both linear and LaTeX forms.

    - t-type: B g_s1 g1 g_s2 g2 g_s3 g3
    - k-type: B 1 g1 1 g2 1 g3 [g_s(tau)...] (L0 basis)
    - g-type: B g_s1 g1 g_s2 g2 g_s3 g3 | (g_s(t_a),g_s(t_b),g_s(t_c);g_s(b_1),...) (G0 basis)
    """
    it = int(ssg.it)
    ik = int(ssg.ik)
    ssg_type = _symbol_type(it, ik)

    use_l0_basis = ssg_type == "k"
    if basis_mode == "current":
        basis_name = "current"
        if use_l0_basis:
            sg_num = int(ssg.L0_num)
            sg_symbol = ssg.L0_symbol
            current_to_standard = np.asarray(ssg.transformation_to_L0std, dtype=float)
            current_to_standard_shift = np.asarray(ssg.origin_shift_to_L0std, dtype=float)
        else:
            sg_num = int(ssg.G0_num)
            sg_symbol = ssg.G0_symbol
            current_to_standard, current_to_standard_shift = _compose_setting_transform(
                np.asarray(ssg.transformation_to_G0std, dtype=float),
                np.asarray(ssg.origin_shift_to_G0std, dtype=float),
                np.asarray(ssg.transformation_to_G0std_id, dtype=float),
                np.asarray(ssg.origin_shift_to_G0std_id, dtype=float),
            )
        ssg_basis = ssg
    else:
        if use_l0_basis:
            basis_name = "L0"
            sg_num = int(ssg.L0_num)
            sg_symbol = ssg.L0_symbol
            ssg_basis = _transform_to_l0_basis(ssg)
        else:
            basis_name = "G0"
            sg_num = int(ssg.G0_num)
            sg_symbol = ssg.G0_symbol
            ssg_basis = _transform_to_g0_basis(ssg)

    bravais = SGdisc[sg_num][0]
    named_ops, centering_trans = _parse_sg_generator_ops(sg_num)
    if basis_mode == "current":
        named_ops, centering_trans = _transport_standard_generators_to_current_basis(
            named_ops,
            centering_trans,
            current_to_standard,
            current_to_standard_shift,
        )
    real_tokens = _real_generator_tokens(sg_num, len(named_ops))

    spin_info_map = _canonical_spin_info_map(ssg_basis)
    identity_real_ops = ssg_basis.identity_real_nssg_ops

    named_pair_data: list[tuple[dict | None, str, "SpinSpaceGroupOperation | None"]] = []
    sg1_token: str | None = None
    if named_ops:
        for (rot, trans), real_tok in zip(named_ops, real_tokens):
            matched = _find_real_operation(ssg_basis.nssg, rot, trans, tol=tol)
            if ssg_type == "k":
                spin_info = None
            else:
                spin_info = spin_info_map.get(id(matched)) if matched is not None else None
            named_pair_data.append((spin_info, real_tok, matched))
    else:
        # SG #1 has no non-identity named generator; keep trailing "1" for readability.
        if len(SGdisc[sg_num]) > 1:
            sg1_token = SGdisc[sg_num][1]

    k_translation_data: list[tuple["SpinSpaceGroupOperation", dict | None, str, str]] = []
    primitive_translation_data: list[tuple[str, np.ndarray, dict | None, "SpinSpaceGroupOperation | None"]] = []
    centering_translation_data: list[tuple[str, np.ndarray, dict | None, "SpinSpaceGroupOperation | None"]] = []
    spin_translation_generator_ops = _minimal_spin_translation_generators(ssg_basis.spin_translation_group)

    if ssg_type == "k":
        # Nontrivial spin translations with real-space identity.
        # Only keep a minimal generator set in the symbol.  For centered
        # settings, the Bravais centering translations are lattice translations
        # and therefore participate in the closure implicitly.
        centering_generators = _pure_centering_translation_generators(
            identity_real_ops,
            centering_trans,
            tol=tol,
        )
        selected_generators = _minimal_k_translation_generators(
            ssg_basis.n_spin_translation_group,
            free_generators=centering_generators,
        )
        for op in selected_generators:
            spin_info = spin_info_map.get(id(op))
            k_translation_data.append(
                (
                    op,
                    spin_info,
                    _format_vector_linear(op.translation),
                    _format_vector_latex(op.translation),
                )
            )

    elif ssg_type == "g":
        primitive_targets: list[tuple[str, np.ndarray]] = [
            ("t_a", np.array([1.0, 0.0, 0.0])),
            ("t_b", np.array([0.0, 1.0, 0.0])),
            ("t_c", np.array([0.0, 0.0, 1.0])),
        ]

        if centering_trans:
            centering_targets = [(f"b_{i+1}", vec) for i, vec in enumerate(centering_trans)]
        else:
            centering_targets = _default_centering_vectors(bravais)

        for axis_index, (label, target) in enumerate(primitive_targets):
            matched = _select_preferred_primitive_translation_match(
                ssg_basis.nssg,
                axis_index,
                tol=tol,
                identity_real_ops=identity_real_ops,
            )
            vector = np.asarray(matched.translation if matched is not None else target, dtype=float)
            spin_info = spin_info_map.get(id(matched)) if matched is not None else None
            primitive_translation_data.append((label, vector, spin_info, matched))

        for label, target in centering_targets:
            matched = _select_preferred_translation_match(
                ssg_basis.nssg,
                target,
                tol=tol,
                identity_real_ops=identity_real_ops,
            )
            vector = np.asarray(matched.translation if matched is not None else target, dtype=float)
            spin_info = spin_info_map.get(id(matched)) if matched is not None else None
            centering_translation_data.append((label, vector, spin_info, matched))

    # The spin-only suffix should describe the current SpinSpaceGroup frame,
    # not the intermediate standardized basis used to label the real-space part.
    suffix_info = _spin_only_suffix_point_info(ssg, tol=calibrated_symbol_tol(tol))

    parameter_namer = _SymbolParameterAxisNamer()
    for spin_info, _real_tok, _matched in named_pair_data:
        parameter_namer.add(spin_info)
    for _op, spin_info, _tau_linear, _tau_latex in k_translation_data:
        parameter_namer.add(spin_info)
    for _label, _vector, spin_info, _matched in primitive_translation_data + centering_translation_data:
        parameter_namer.add(spin_info)
    if suffix_info is not None:
        parameter_namer.add(suffix_info[1])

    pair_linear_terms: list[str] = []
    pair_latex_terms: list[str] = []
    point_pair_linear_terms: list[str] = []
    point_pair_latex_terms: list[str] = []

    if named_pair_data:
        for spin_info, real_tok, _matched in named_pair_data:
            spin_tok = _point_token_from_info(spin_info, parameter_namer=parameter_namer)
            spin_tok_latex = _point_token_from_info(spin_info, latex=True, parameter_namer=parameter_namer)
            pair_linear_terms.append(f"{spin_tok}|{real_tok}")
            pair_latex_terms.append(rf"^{{{spin_tok_latex}}}{_to_latex_token(real_tok)}")

            point_real_tok = _point_group_token_from_real_token(real_tok)
            point_pair_linear_terms.append(f"{spin_tok}|{point_real_tok}")
            point_pair_latex_terms.append(rf"^{{{spin_tok_latex}}}{_to_latex_token(point_real_tok)}")
    elif sg1_token is not None:
        pair_linear_terms.append(sg1_token)
        pair_latex_terms.append(_to_latex_token(sg1_token))
        point_pair_linear_terms.append(sg1_token)
        point_pair_latex_terms.append(_to_latex_token(sg1_token))

    point_part_linear = " ".join(point_pair_linear_terms).strip()
    point_part_latex = " ".join(point_pair_latex_terms).strip()

    base_linear = " ".join([bravais] + pair_linear_terms).strip()
    base_latex = " ".join([_to_latex_token(bravais)] + pair_latex_terms).strip()

    translation_linear_terms: list[str] = []
    translation_latex_terms: list[str] = []
    translation_details: list[dict] = []

    if ssg_type == "k":
        for op, spin_info, tau_linear, tau_latex in k_translation_data:
            spin_tok = _point_token_from_info(spin_info, parameter_namer=parameter_namer)
            spin_tok_latex = _point_token_from_info(spin_info, latex=True, parameter_namer=parameter_namer)
            translation_linear_terms.append(f"{spin_tok}|{tau_linear}")
            translation_latex_terms.append(rf"^{{{spin_tok_latex}}}{tau_latex}")
            translation_details.append(
                {
                    "label": "tau",
                    "vector": tuple(float(v) for v in _normalize_mod1(op.translation)),
                    "spin_symbol": spin_tok,
                }
            )

    elif ssg_type == "g":
        primitive_spin_symbols: list[str] = []
        primitive_spin_symbols_latex: list[str] = []
        centering_spin_symbols: list[str] = []
        centering_spin_symbols_latex: list[str] = []

        for label, vector, spin_info, _matched in primitive_translation_data:
            spin_tok = _point_token_from_info(spin_info, parameter_namer=parameter_namer)
            spin_tok_latex = _point_token_from_info(spin_info, latex=True, parameter_namer=parameter_namer)
            primitive_spin_symbols.append(spin_tok)
            primitive_spin_symbols_latex.append(spin_tok_latex)
            translation_details.append(
                {
                    "label": label,
                    "vector": tuple(float(v) for v in vector),
                    "spin_symbol": spin_tok,
                }
            )

        for label, vector, spin_info, _matched in centering_translation_data:
            spin_tok = _point_token_from_info(spin_info, parameter_namer=parameter_namer)
            spin_tok_latex = _point_token_from_info(spin_info, latex=True, parameter_namer=parameter_namer)
            centering_spin_symbols.append(spin_tok)
            centering_spin_symbols_latex.append(spin_tok_latex)
            translation_details.append(
                {
                    "label": label,
                    "vector": tuple(float(v) for v in vector),
                    "spin_symbol": spin_tok,
                }
            )

        primitive_linear = ",".join(primitive_spin_symbols)
        primitive_latex = ",".join(primitive_spin_symbols_latex)
        if centering_spin_symbols:
            center_linear = ",".join(centering_spin_symbols)
            center_latex = ",".join(centering_spin_symbols_latex)
            translation_linear_terms.append(f"({primitive_linear};{center_linear})")
            translation_latex_terms.append(f"({primitive_latex};{center_latex})")
        else:
            translation_linear_terms.append(f"({primitive_linear})")
            translation_latex_terms.append(f"({primitive_latex})")

    if translation_linear_terms:
        linear = base_linear + " : " + " ".join(translation_linear_terms)
        latex = base_latex + r" \mid " + " ".join(translation_latex_terms)
    else:
        linear = base_linear
        latex = base_latex

    suffix_linear, suffix_latex = _format_spin_only_suffix(suffix_info, parameter_namer=parameter_namer)
    if suffix_linear:
        linear = f"{linear} {suffix_linear}"
        latex = f"{latex} {suffix_latex}"

    generator_operations: list[dict] = []
    generator_operation_keys: set[tuple] = set()
    for _spin_info, real_tok, matched in named_pair_data:
        _append_generator_operation(
            generator_operations,
            generator_operation_keys,
            matched,
            source="real_generator",
            label=real_tok,
            tol=tol,
        )
    for op, _spin_info, _tau_linear, _tau_latex in k_translation_data:
        _append_generator_operation(
            generator_operations,
            generator_operation_keys,
            op,
            source="spin_translation_generator",
            label="tau",
            tol=tol,
        )
    for op in spin_translation_generator_ops:
        _append_generator_operation(
            generator_operations,
            generator_operation_keys,
            op,
            source="spin_translation_generator",
            label="spin",
            tol=tol,
        )
    for label, _vector, _spin_info, matched in primitive_translation_data:
        _append_generator_operation(
            generator_operations,
            generator_operation_keys,
            matched,
            source="primitive_translation_generator",
            label=label,
            tol=tol,
        )
    for label, _vector, _spin_info, matched in centering_translation_data:
        _append_generator_operation(
            generator_operations,
            generator_operation_keys,
            matched,
            source="centering_translation_generator",
            label=label,
            tol=tol,
        )

    return {
        "type": ssg_type,
        "basis": basis_name,
        "it": it,
        "ik": ik,
        "sg_number": sg_num,
        "sg_symbol": sg_symbol,
        "linear": linear,
        "latex": latex,
        "point_part_linear": point_part_linear,
        "point_part_latex": point_part_latex,
        "spin_only_suffix_linear": suffix_linear,
        "spin_only_suffix_latex": suffix_latex,
        "real_generator_pairs_linear": pair_linear_terms,
        "real_generator_pairs_latex": pair_latex_terms,
        "translation_terms_linear": translation_linear_terms,
        "translation_terms_latex": translation_latex_terms,
        "translation_details": translation_details,
        "generator_operations": generator_operations,
    }
