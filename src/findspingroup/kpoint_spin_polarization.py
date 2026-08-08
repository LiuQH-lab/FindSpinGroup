"""Symmetry-allowed spin-polarization constraints at arbitrary k points.

This module analyzes an exact numerical k point.  It does not infer the
behavior of an entire path from a midpoint and it does not predict a material-
specific spin expectation value.  The result is the spin-vector subspace
allowed by the SSG little group (without SOC) and the MSG little group (with
SOC).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from findspingroup.core.identify_symmetry_from_ops import deduplicate_matrix_pairs
from findspingroup.core.tolerances import DEFAULT_KPOINT_TOL
from findspingroup.structure.group import solve_spin_constraint_from_stacked


_SPIN_CONSTRAINT_TOL = 1e-3
_SPIN_VARIABLE_VECTORS = {
    "Sx": np.array([1.0, 0.0, 0.0]),
    "Sy": np.array([0.0, 1.0, 0.0]),
    "Sz": np.array([0.0, 0.0, 1.0]),
}


def _as_vector3(value, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{label} must contain exactly three components; got shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values.")
    return vector


def _validate_kpoint_tol(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("kpoint_tol must be a finite positive number.")
    if tolerance >= 0.5:
        raise ValueError("kpoint_tol must be smaller than 0.5 in reciprocal fractional coordinates.")
    return tolerance


def _reduce_reciprocal(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reduced = np.mod(np.asarray(vector, dtype=float), 1.0)
    reduced[np.isclose(reduced, 1.0, atol=1e-12)] = 0.0
    shift = np.rint(np.asarray(vector, dtype=float) - reduced).astype(int)
    return reduced, shift


def _wrapped_residuals(effective_operations: np.ndarray, kpoint: np.ndarray) -> np.ndarray:
    transformed = np.einsum("nij,j->ni", effective_operations, kpoint)
    delta = transformed - kpoint
    wrapped = delta - np.rint(delta)
    return np.max(np.abs(wrapped), axis=1)


def _canonicalize_direction(vector: np.ndarray, *, tol: float) -> np.ndarray:
    direction = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= tol:
        return np.zeros(3, dtype=float)
    direction = direction / norm
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    direction[np.abs(direction) < max(tol * 1e-3, 1e-12)] = 0.0
    return direction


def _deterministic_basis(projector: np.ndarray, dimension: int, *, tol: float) -> np.ndarray:
    if dimension <= 0:
        return np.empty((0, 3), dtype=float)
    basis: list[np.ndarray] = []
    for axis in np.eye(3):
        candidate = np.asarray(projector, dtype=float) @ axis
        for existing in basis:
            candidate = candidate - np.dot(existing, candidate) * existing
        if np.linalg.norm(candidate) <= tol:
            continue
        candidate = _canonicalize_direction(candidate, tol=tol)
        basis.append(candidate)
        if len(basis) == dimension:
            break
    if len(basis) != dimension:
        raise RuntimeError(
            f"Could not construct a deterministic basis for a {dimension}-dimensional spin subspace."
        )
    return np.vstack(basis)


def _structured_spin_constraint(stacked: np.ndarray, *, tol: float) -> dict:
    matrix = np.asarray(stacked, dtype=float).reshape(-1, 3)
    if matrix.size == 0:
        raise ValueError("A spin constraint requires at least one little-group operation.")

    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = min(int(np.count_nonzero(singular_values > tol)), 3)
    dimension = 3 - rank
    nullspace = vh[rank:, :]
    projector = nullspace.T @ nullspace if dimension else np.zeros((3, 3), dtype=float)
    projector[np.abs(projector) < max(tol * 1e-6, 1e-12)] = 0.0
    basis = _deterministic_basis(projector, dimension, tol=max(tol * 1e-3, 1e-12))
    spin_splitting, readable_constraint = solve_spin_constraint_from_stacked(matrix, tol=tol)

    if dimension == 0:
        status = "forbidden"
    elif dimension == 3:
        status = "unconstrained"
    else:
        status = "allowed"

    return {
        "status": status,
        "spin_splitting": spin_splitting,
        "dimension": int(dimension),
        "constraint_acc_primitive_cartesian": list(readable_constraint),
        "basis_acc_primitive_cartesian": basis.tolist(),
        "projector_acc_primitive_cartesian": projector.tolist(),
        "direction_acc_primitive_cartesian": (
            basis[0].tolist() if dimension == 1 else None
        ),
        "direction_sign_is_ambiguous": bool(dimension == 1),
        "magnitude_is_determined": False,
        "zero_vector_forced": bool(dimension == 0),
        "singular_values": np.asarray(singular_values, dtype=float).tolist(),
        "constraint_tol": float(tol),
    }


def _evaluate_linear_spin_expression(node):
    if isinstance(node, ast.Expression):
        return _evaluate_linear_spin_expression(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SPIN_VARIABLE_VECTORS:
        return _SPIN_VARIABLE_VECTORS[node.id].copy()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_linear_spin_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "sqrt"
            or len(node.args) != 1
            or node.keywords
        ):
            raise ValueError("Only sqrt(number) is allowed in a readable spin constraint.")
        value = _evaluate_linear_spin_expression(node.args[0])
        if not np.isscalar(value) or value < 0.0:
            raise ValueError("sqrt() requires a non-negative scalar argument.")
        return float(np.sqrt(value))
    if isinstance(node, ast.BinOp):
        left = _evaluate_linear_spin_expression(node.left)
        right = _evaluate_linear_spin_expression(node.right)
        left_scalar = np.isscalar(left)
        right_scalar = np.isscalar(right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left_scalar != right_scalar:
                raise ValueError("Cannot add scalar and vector spin-constraint terms.")
            return left + right if isinstance(node.op, ast.Add) else left - right
        if isinstance(node.op, ast.Mult):
            if not left_scalar and not right_scalar:
                raise ValueError("Spin-constraint expressions must remain linear.")
            return left * right
        if isinstance(node.op, ast.Div):
            if not right_scalar or right == 0.0:
                raise ValueError("Spin-constraint division requires a non-zero scalar divisor.")
            return left / right
    raise ValueError("Unsupported readable spin-constraint expression.")


def _projector_from_readable_constraint(constraint: Sequence[str], *, tol: float) -> tuple[int, np.ndarray]:
    if len(constraint) != 3:
        raise ValueError("A readable spin constraint must contain three expressions.")
    rows = []
    for expression in constraint:
        parsed = ast.parse(str(expression).replace(" ", ""), mode="eval")
        value = _evaluate_linear_spin_expression(parsed)
        if np.isscalar(value):
            if abs(float(value)) > tol:
                raise ValueError("A spin-constraint component cannot contain a non-zero constant.")
            value = np.zeros(3, dtype=float)
        rows.append(np.asarray(value, dtype=float).reshape(3))
    parameterization = np.vstack(rows)
    u, singular_values, _vh = np.linalg.svd(parameterization, full_matrices=True)
    dimension = min(int(np.count_nonzero(singular_values > tol)), 3)
    projector = u[:, :dimension] @ u[:, :dimension].T if dimension else np.zeros((3, 3))
    return dimension, projector


def _translation_distance(left, right) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    wrapped = delta - np.rint(delta)
    return float(np.max(np.abs(wrapped)))


def _ssg_operation_arrays(operation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(operation, dict):
        return (
            np.asarray(operation["spin_rotation"], dtype=float),
            np.asarray(operation["real_rotation"], dtype=float),
            np.asarray(operation["translation"], dtype=float),
        )
    if hasattr(operation, "spin_rotation"):
        return (
            np.asarray(operation.spin_rotation, dtype=float),
            np.asarray(operation.rotation, dtype=float),
            np.asarray(operation.translation, dtype=float),
        )
    spin, real, translation = operation
    return (
        np.asarray(spin, dtype=float),
        np.asarray(real, dtype=float),
        np.asarray(translation, dtype=float),
    )


def _msg_operation_arrays(operation) -> tuple[int, np.ndarray, np.ndarray]:
    if isinstance(operation, dict):
        return (
            int(operation["time_reversal"]),
            np.asarray(operation["real_rotation"], dtype=float),
            np.asarray(operation["translation"], dtype=float),
        )
    time_reversal, rotation, translation = operation
    return int(time_reversal), np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)


def _canonical_array_key(array, *, decimals: int = 10) -> tuple[float, ...]:
    values = np.array(array, dtype=float, copy=True).reshape(-1)
    values[np.abs(values) < 10.0 ** (-decimals)] = 0.0
    return tuple(np.round(values, decimals=decimals).tolist())


def _canonical_translation_key(translation, *, decimals: int = 10) -> tuple[float, ...]:
    reduced = np.mod(np.asarray(translation, dtype=float).reshape(3), 1.0)
    reduced[np.isclose(reduced, 1.0, atol=10.0 ** (-decimals))] = 0.0
    return _canonical_array_key(reduced, decimals=decimals)


def _ssg_operation_key(operation) -> tuple:
    spin, real, translation = _ssg_operation_arrays(operation)
    return (
        _canonical_array_key(spin),
        _canonical_array_key(real),
        _canonical_translation_key(translation),
    )


def _msg_operation_key(operation) -> tuple:
    theta, real, translation = _msg_operation_arrays(operation)
    return (
        int(theta),
        _canonical_array_key(real),
        _canonical_translation_key(translation),
    )


def _match_ssg_operation_index(operation, operations, *, tol: float) -> int | None:
    spin, real, translation = _ssg_operation_arrays(operation)
    for index, candidate in enumerate(operations):
        candidate_spin, candidate_real, candidate_translation = _ssg_operation_arrays(candidate)
        if (
            np.allclose(spin, candidate_spin, atol=tol, rtol=0.0)
            and np.allclose(real, candidate_real, atol=tol, rtol=0.0)
            and _translation_distance(translation, candidate_translation) <= tol
        ):
            return index
    return None


def _match_msg_operation_index(operation, operations, *, tol: float) -> int | None:
    theta, real, translation = _msg_operation_arrays(operation)
    for index, candidate in enumerate(operations):
        candidate_theta, candidate_real, candidate_translation = _msg_operation_arrays(candidate)
        if (
            theta == candidate_theta
            and np.allclose(real, candidate_real, atol=tol, rtol=0.0)
            and _translation_distance(translation, candidate_translation) <= tol
        ):
            return index
    return None


def _precomputed_constraint_cache(
    little_groups,
    constraints,
    operations,
    *,
    matcher,
    operation_key,
    tol: float,
) -> dict[tuple[int, ...], dict]:
    cache: dict[tuple[int, ...], dict] = {}
    if little_groups is None or constraints is None:
        return cache
    if len(little_groups) != len(constraints):
        raise ValueError(
            "Existing little-group operations and spin constraints must have the same length."
        )
    index_by_key = {
        operation_key(operation): index
        for index, operation in enumerate(operations)
    }
    for row_index, (little_group, constraint) in enumerate(zip(little_groups, constraints)):
        indices = []
        for operation in little_group:
            matched_index = index_by_key.get(operation_key(operation))
            if matched_index is None:
                matched_index = matcher(operation, operations, tol=tol)
            if matched_index is None:
                indices = []
                break
            indices.append(matched_index)
        if not indices and little_group:
            continue
        signature = tuple(sorted(set(indices)))
        entry = cache.setdefault(
            signature,
            {
                "constraint": list(constraint),
                "precomputed_kpoint_indices_zero_based": [],
            },
        )
        if entry["constraint"] != list(constraint):
            raise RuntimeError(
                "Existing k-space rows with the same little group have inconsistent spin constraints."
            )
        entry["precomputed_kpoint_indices_zero_based"].append(int(row_index))
    return cache


def _membership_audit(residuals: np.ndarray, signature: tuple[int, ...], tol: float) -> dict:
    included = np.asarray(signature, dtype=int)
    included_max = float(np.max(residuals[included])) if included.size else None
    excluded_mask = np.ones(len(residuals), dtype=bool)
    if included.size:
        excluded_mask[included] = False
    excluded = residuals[excluded_mask]
    excluded_min = float(np.min(excluded)) if excluded.size else None
    near_boundary = bool(
        (included_max is not None and included_max >= 0.5 * tol)
        or (excluded_min is not None and excluded_min <= 2.0 * tol)
    )
    return {
        "largest_included_residual": included_max,
        "smallest_excluded_residual": excluded_min,
        "stability": "near_boundary" if near_boundary else "stable",
    }


class KPointSpinPolarizationResult(dict):
    """Compact arbitrary-k result with diagnostics kept outside the mapping.

    ``print(result)`` and ordinary mapping/JSON serialization expose only the
    user-facing physical result.  Detailed little-group membership and solver
    diagnostics remain available through :attr:`audit` or
    :meth:`to_dict(include_audit=True)`.
    """

    def __init__(self, payload: dict, *, audit: dict):
        super().__init__(deepcopy(payload))
        self.audit = deepcopy(audit)

    def to_dict(self, *, include_audit: bool = False) -> dict:
        payload = deepcopy(dict(self))
        if include_audit:
            payload["audit"] = deepcopy(self.audit)
        return payload


@dataclass
class KPointSpinPolarizationAnalyzer:
    """Reusable arbitrary-k analyzer built from an existing full FSG result."""

    result: object
    constraint_tol: float = _SPIN_CONSTRAINT_TOL
    _analysis_cache_no_soc: dict[tuple[int, ...], dict] = field(default_factory=dict, init=False)
    _analysis_cache_soc: dict[tuple[int, ...], dict] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.constraint_tol = float(self.constraint_tol)
        if not np.isfinite(self.constraint_tol) or self.constraint_tol <= 0.0:
            raise ValueError("constraint_tol must be a finite positive number.")
        self._matrix_tol = float((self.result.tolerances or {}).get("matrix_tol", 1e-6))
        if not np.isfinite(self._matrix_tol) or self._matrix_tol <= 0.0:
            raise ValueError("The result matrix_tol must be a finite positive number.")

        self._ssg_operations = list(self.result.acc_primitive_ssg_ops or [])
        self._msg_operations = list(self.result.acc_primitive_msg_ops or [])
        if not self._ssg_operations:
            raise ValueError("The result does not contain ACC-primitive SSG operations.")
        if not self._msg_operations:
            raise ValueError("The result does not contain ACC-primitive MSG operations.")

        ssg_arrays = [_ssg_operation_arrays(operation) for operation in self._ssg_operations]
        self._ssg_effective_operations = np.stack(
            [np.linalg.det(spin) * np.linalg.inv(real).T for spin, real, _translation in ssg_arrays]
        )
        self._ssg_constraint_matrices = np.stack(
            [spin - np.eye(3) for spin, _real, _translation in ssg_arrays]
        )

        lattice = np.asarray(self.result.acc_primitive_magnetic_cell[0], dtype=float)
        if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1e-12:
            raise ValueError("The result does not contain a valid ACC-primitive lattice.")
        lattice_col = lattice.T
        lattice_col_inv = np.linalg.inv(lattice_col)
        msg_arrays = [_msg_operation_arrays(operation) for operation in self._msg_operations]
        self._msg_effective_operations = np.stack(
            [theta * np.linalg.inv(real).T for theta, real, _translation in msg_arrays]
        )
        msg_constraints = []
        for theta, real, _translation in msg_arrays:
            real_cartesian = lattice_col @ real @ lattice_col_inv
            spin_action = theta * np.linalg.det(real_cartesian) * real_cartesian
            msg_constraints.append(spin_action - np.eye(3))
        self._msg_constraint_matrices = np.stack(msg_constraints)

        transform = self.result.T_input_to_acc_primitive
        if transform is None:
            raise ValueError("The result does not contain T_input_to_acc_primitive.")
        self._input_to_acc = np.asarray(transform[0], dtype=float)
        if self._input_to_acc.shape != (3, 3) or abs(np.linalg.det(self._input_to_acc)) < 1e-12:
            raise ValueError("T_input_to_acc_primitive must contain an invertible 3x3 matrix.")
        rounded_input_to_acc = np.rint(self._input_to_acc).astype(int)
        transform_tol = min(max(self._matrix_tol, 1e-8), 1e-5)
        self._input_kpoint_mapping_is_bijective = bool(
            np.allclose(
                self._input_to_acc,
                rounded_input_to_acc,
                atol=transform_tol,
                rtol=0.0,
            )
            and abs(round(float(np.linalg.det(rounded_input_to_acc)))) == 1
        )

        operation_match_tol = min(max(self._matrix_tol, 1e-8), 1e-4)
        self._precomputed_no_soc = _precomputed_constraint_cache(
            self.result.ssg_little_group_ops,
            self.result.spin_polarizations_acc_cartesian,
            self._ssg_operations,
            matcher=_match_ssg_operation_index,
            operation_key=_ssg_operation_key,
            tol=operation_match_tol,
        )
        self._precomputed_soc = _precomputed_constraint_cache(
            self.result.msg_little_group_ops,
            self.result.msg_spin_polarizations_acc_cartesian,
            self._msg_operations,
            matcher=_match_msg_operation_index,
            operation_key=_msg_operation_key,
            tol=operation_match_tol,
        )

    @classmethod
    def from_result(cls, result, *, constraint_tol: float = _SPIN_CONSTRAINT_TOL):
        return cls(result=result, constraint_tol=float(constraint_tol))

    def _coordinate_payload(self, kpoint, setting: str) -> tuple[np.ndarray, dict]:
        supplied = _as_vector3(kpoint, label="kpoint")
        if setting == "input":
            if not self._input_kpoint_mapping_is_bijective:
                raise ValueError(
                    "The input magnetic cell folds multiple ACC-primitive k points onto the "
                    "same input-cell coordinate. A unique spin-polarization constraint requires "
                    "an unfolded k point with kpoint_setting='acc_primitive'."
                )
            input_raw = supplied
            acc_raw = np.linalg.solve(self._input_to_acc.T, input_raw)
        elif setting == "acc_primitive":
            acc_raw = supplied
            input_raw = self._input_to_acc.T @ acc_raw
        else:
            raise ValueError("kpoint_setting must be 'input' or 'acc_primitive'.")

        input_reduced, input_shift = _reduce_reciprocal(input_raw)
        acc_reduced, acc_shift = _reduce_reciprocal(acc_raw)
        return acc_reduced, {
            "supplied": supplied.tolist(),
            "supplied_setting": setting,
            "input_raw": input_raw.tolist(),
            "input_reduced": input_reduced.tolist(),
            "input_reciprocal_shift": input_shift.tolist(),
            "acc_primitive_raw": acc_raw.tolist(),
            "acc_primitive_reduced": acc_reduced.tolist(),
            "acc_primitive_reciprocal_shift": acc_shift.tolist(),
        }

    def _validate_quasi2d_plane(self, coordinate_payload: dict, *, kpoint_tol: float) -> dict | None:
        quasi_2d = self.result.quasi_2d
        if not isinstance(quasi_2d, dict):
            return None
        if quasi_2d.get("dimension") != "2d":
            if quasi_2d.get("calculation_mode") == "quasi2d":
                raise ValueError(
                    "The quasi-2D symmetry analysis did not resolve a valid two-dimensional setting."
                )
            return None
        vacuum_axis_index = quasi_2d.get("vacuum_axis_index")
        if vacuum_axis_index is None:
            raise ValueError("The quasi-2D result does not define an input-cell vacuum axis.")
        component = float(coordinate_payload["input_raw"][int(vacuum_axis_index)])
        distance = abs(component - round(component))
        if distance >= kpoint_tol:
            axis = quasi_2d.get("vacuum_axis_input") or "unknown"
            raise ValueError(
                f"The k point is outside the quasi-2D reciprocal plane: input {axis}-axis "
                f"component {component:.12g} is {distance:.3g} from an integer, "
                f"which is not smaller than kpoint_tol={kpoint_tol:.3g}."
            )
        return {
            "vacuum_axis_input": quasi_2d.get("vacuum_axis_input"),
            "vacuum_axis_index": int(vacuum_axis_index),
            "vacuum_component_raw": component,
            "distance_to_plane": float(distance),
        }

    def _analyze_group(
        self,
        signature: tuple[int, ...],
        constraint_matrices: np.ndarray,
        precomputed: dict,
        cache: dict,
    ) -> dict:
        if signature in cache:
            return deepcopy(cache[signature])
        selected = constraint_matrices[np.asarray(signature, dtype=int)]
        deduplicated = deduplicate_matrix_pairs(list(selected), tol=self._matrix_tol)
        stacked = np.vstack(deduplicated)
        analysis = _structured_spin_constraint(stacked, tol=self.constraint_tol)
        precomputed_entry = precomputed.get(signature)
        if precomputed_entry is None:
            analysis["source"] = "computed_little_group"
            analysis["precomputed_constraint_match"] = "not_available"
            analysis["precomputed_kpoint_indices_zero_based"] = []
        else:
            existing_constraint = list(precomputed_entry["constraint"])
            if existing_constraint != analysis["constraint_acc_primitive_cartesian"]:
                expected_dimension, expected_projector = _projector_from_readable_constraint(
                    existing_constraint,
                    tol=self.constraint_tol,
                )
                projector_distance = float(
                    np.max(
                        np.abs(
                            expected_projector
                            - np.asarray(
                                analysis["projector_acc_primitive_cartesian"],
                                dtype=float,
                            )
                        )
                    )
                )
                if (
                    expected_dimension != analysis["dimension"]
                    or projector_distance > 2.0 * self.constraint_tol
                ):
                    raise RuntimeError(
                        "The arbitrary-k spin subspace disagrees with the existing k-space "
                        f"result for little-group signature {signature}: "
                        f"dimension {analysis['dimension']} vs {expected_dimension}, "
                        f"projector distance {projector_distance:.6g}."
                    )
                analysis["constraint_acc_primitive_cartesian"] = existing_constraint
                analysis["precomputed_constraint_match"] = "equivalent_subspace"
            else:
                analysis["precomputed_constraint_match"] = "exact"
            analysis["source"] = "precomputed_kspace"
            analysis["precomputed_kpoint_indices_zero_based"] = list(
                precomputed_entry["precomputed_kpoint_indices_zero_based"]
            )
        analysis["little_group_operation_indices"] = [index + 1 for index in signature]
        analysis["little_group_order"] = len(signature)
        cache[signature] = deepcopy(analysis)
        return analysis

    def query(
        self,
        kpoint: Sequence[float],
        *,
        kpoint_setting: str = "acc_primitive",
        kpoint_tol: float = DEFAULT_KPOINT_TOL,
    ) -> KPointSpinPolarizationResult:
        """Analyze symmetry-allowed spin polarization at one exact k point."""

        tolerance = _validate_kpoint_tol(kpoint_tol)
        k_acc, coordinate_payload = self._coordinate_payload(kpoint, kpoint_setting)
        quasi2d_plane = self._validate_quasi2d_plane(
            coordinate_payload,
            kpoint_tol=tolerance,
        )

        no_soc_residuals = _wrapped_residuals(self._ssg_effective_operations, k_acc)
        soc_residuals = _wrapped_residuals(self._msg_effective_operations, k_acc)
        no_soc_signature = tuple(np.flatnonzero(no_soc_residuals < tolerance).tolist())
        soc_signature = tuple(np.flatnonzero(soc_residuals < tolerance).tolist())
        if not no_soc_signature or not soc_signature:
            raise RuntimeError("The exact k-point little group unexpectedly lacks the identity operation.")

        without_soc = self._analyze_group(
            no_soc_signature,
            self._ssg_constraint_matrices,
            self._precomputed_no_soc,
            self._analysis_cache_no_soc,
        )
        with_soc = self._analyze_group(
            soc_signature,
            self._msg_constraint_matrices,
            self._precomputed_soc,
            self._analysis_cache_soc,
        )
        without_soc["membership_audit"] = _membership_audit(
            no_soc_residuals, no_soc_signature, tolerance
        )
        with_soc["membership_audit"] = _membership_audit(
            soc_residuals, soc_signature, tolerance
        )

        warnings = []
        if tolerance > 1e-2:
            warnings.append(
                "kpoint_tol is unusually large and may promote a nearby generic point "
                "to a higher-symmetry little group."
            )
        audit = {
            "status": "ok",
            "fsg_version": getattr(self.result, "fsg_version", None),
            "ssg_index": getattr(self.result, "index", None),
            "msg_bns_number": getattr(self.result, "msg_bns_number", None),
            "calculation_mode": "quasi2d" if quasi2d_plane is not None else "3d",
            "kpoint": coordinate_payload,
            "quasi2d_plane": quasi2d_plane,
            "kpoint_tol": tolerance,
            "without_soc": without_soc,
            "with_soc": with_soc,
            "warnings": warnings,
            "interpretation": (
                "Symmetry-allowed spin-polarization subspaces; magnitudes and band-specific "
                "spin expectation values are not determined."
            ),
        }
        payload = {
            "kpoint": list(coordinate_payload["supplied"]),
            "kpoint_setting": coordinate_payload["supplied_setting"],
            "kpoint_tol": tolerance,
            "spin_frame": "acc_primitive_cartesian",
            "without_soc": {
                "allowed": bool(without_soc["dimension"] > 0),
                "dimension": int(without_soc["dimension"]),
                "constraint": list(
                    without_soc["constraint_acc_primitive_cartesian"]
                ),
                "direction": deepcopy(
                    without_soc["direction_acc_primitive_cartesian"]
                ),
            },
            "with_soc": {
                "allowed": bool(with_soc["dimension"] > 0),
                "dimension": int(with_soc["dimension"]),
                "constraint": list(with_soc["constraint_acc_primitive_cartesian"]),
                "direction": deepcopy(with_soc["direction_acc_primitive_cartesian"]),
            },
        }
        if warnings:
            payload["warnings"] = list(warnings)
        return KPointSpinPolarizationResult(payload, audit=audit)

    def query_many(
        self,
        kpoints: Iterable[Sequence[float]],
        *,
        kpoint_setting: str = "acc_primitive",
        kpoint_tol: float = DEFAULT_KPOINT_TOL,
    ) -> list[KPointSpinPolarizationResult]:
        return [
            self.query(
                kpoint,
                kpoint_setting=kpoint_setting,
                kpoint_tol=kpoint_tol,
            )
            for kpoint in kpoints
        ]

    def validate_precomputed_constraints(self) -> dict:
        """Recompute every distinct existing little-group constraint for validation."""

        for signature in self._precomputed_no_soc:
            self._analyze_group(
                signature,
                self._ssg_constraint_matrices,
                self._precomputed_no_soc,
                self._analysis_cache_no_soc,
            )
        for signature in self._precomputed_soc:
            self._analyze_group(
                signature,
                self._msg_constraint_matrices,
                self._precomputed_soc,
                self._analysis_cache_soc,
            )
        return {
            "without_soc_unique_little_groups": len(self._precomputed_no_soc),
            "with_soc_unique_little_groups": len(self._precomputed_soc),
            "without_soc_precomputed_kpoints": sum(
                len(entry["precomputed_kpoint_indices_zero_based"])
                for entry in self._precomputed_no_soc.values()
            ),
            "with_soc_precomputed_kpoints": sum(
                len(entry["precomputed_kpoint_indices_zero_based"])
                for entry in self._precomputed_soc.values()
            ),
        }


def prepare_kpoint_spin_polarization_analyzer(
    source,
    *,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    parser_atol: float = 0.02,
    calculation_mode: str = "3d",
    vacuum_axis: str = "c",
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> KPointSpinPolarizationAnalyzer:
    """Build a reusable analyzer from a structure path or full FSG result."""

    if isinstance(source, (str, Path)):
        from findspingroup.find_spin_group import find_spin_group

        result = find_spin_group(
            str(source),
            space_tol=space_tol,
            mtol=mtol,
            meigtol=meigtol,
            matrix_tol=matrix_tol,
            parser_atol=parser_atol,
            calculation_mode=calculation_mode,
            vacuum_axis=vacuum_axis,
            poscar_allow_incar_magmom=poscar_allow_incar_magmom,
            poscar_prefer_incar_magmom=poscar_prefer_incar_magmom,
            components=(),
        )
    else:
        result = source
    return KPointSpinPolarizationAnalyzer.from_result(result)


def analyze_kpoint_spin_polarization(
    source,
    kpoint: Sequence[float],
    *,
    kpoint_setting: str = "acc_primitive",
    kpoint_tol: float = DEFAULT_KPOINT_TOL,
    **analysis_options,
) -> KPointSpinPolarizationResult:
    """Convenience one-shot arbitrary-k spin-polarization analysis."""

    analyzer = prepare_kpoint_spin_polarization_analyzer(source, **analysis_options)
    return analyzer.query(
        kpoint,
        kpoint_setting=kpoint_setting,
        kpoint_tol=kpoint_tol,
    )


__all__ = [
    "KPointSpinPolarizationAnalyzer",
    "KPointSpinPolarizationResult",
    "analyze_kpoint_spin_polarization",
    "prepare_kpoint_spin_polarization_analyzer",
]
