import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from spglib import get_symmetry_dataset as gsd, SpglibDataset

from findspingroup.core import Molecule, PointGroupAnalyzer
from findspingroup.core.pg_analyzer import SymmOp
from findspingroup.core.identify_symmetry_from_ops import (
    _load_standard_point_group_generators,
    deduplicate_matrix_pairs,
    identify_point_group,
)
from findspingroup.core.tolerances import Tolerances, DEFAULT_TOL
from findspingroup.structure.cell import MAGNETIC_PRESENCE_TOL
from findspingroup.utils.matrix_utils import getNormInf, normalize_vector_to_zero
from findspingroup.structure import *


@dataclass
class InputSpaceGroupInfo:
    number: int | None
    symbol: str | None
    basis_or_setting: str | None = None
    source: str = "magnetic_primitive_spglib_dataset"


@dataclass
class IdentifySpinSpaceGroupResult:
    primitive_cell: CrystalCell
    ssg: SpinSpaceGroup
    input_space_group: InputSpaceGroupInfo | None = None


class MagneticToleranceDegeneracyError(ValueError):
    """Raised when mtol removes all resolved magnetic spin constraints."""


NONMAGNETIC_MTOL_ERROR = (
    "Under the current mtol the structure has no resolved magnetic-moment "
    "direction and degenerates to nonmagnetic O3 spin-rotation symmetry; "
    "FindSpinGroup does not handle nonmagnetic materials."
)

UNSTABLE_MTOL_ERROR = (
    "mtol is too large for stable PG/SSG identification under the current magnetic structure."
)


def check_atom_in_list(atom:AtomicSite, atom_list, tol:Tolerances=DEFAULT_TOL):
    for a in atom_list:
        if atom.is_equivalent(a, tol=tol):
            return True
    return False


def _distance_within_tolerance(distance: float, tolerance: float) -> bool:
    distance = float(distance)
    tolerance = float(tolerance)
    slack = 64.0 * np.finfo(float).eps * max(1.0, abs(distance), abs(tolerance))
    return distance <= tolerance + slack


def _mag_atom_bucket_params(tol: float) -> tuple[int, int]:
    bins = max(1, int(np.ceil(1.0 / max(float(tol), 1e-12))))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(float(tol) / bucket_width)))
    return bins, neighbor_radius


def _mag_atom_bucket_key(position, bins: int) -> tuple[int, int, int]:
    wrapped = np.mod(np.asarray(position, dtype=float), 1.0)
    indices = np.floor(wrapped * bins).astype(int) % bins
    return tuple(int(value) for value in indices)


def _mag_atom_neighbor_keys(bucket_key: tuple[int, int, int], bins: int, neighbor_radius: int):
    for dx in range(-neighbor_radius, neighbor_radius + 1):
        for dy in range(-neighbor_radius, neighbor_radius + 1):
            for dz in range(-neighbor_radius, neighbor_radius + 1):
                yield (
                    (bucket_key[0] + dx) % bins,
                    (bucket_key[1] + dy) % bins,
                    (bucket_key[2] + dz) % bins,
                )


def _build_magnetic_atom_preservation_checker(mag_atoms, tol: Tolerances):
    positions = np.asarray([atom.position for atom in mag_atoms], dtype=float)
    moments = np.asarray([atom.magnetic_moment for atom in mag_atoms], dtype=float)
    elements = [atom.element_symbol for atom in mag_atoms]
    occupancies = np.asarray([float(atom.occupancy) for atom in mag_atoms], dtype=float)
    bins, neighbor_radius = _mag_atom_bucket_params(tol.space)
    buckets: dict[tuple[object, int, int, int], list[int]] = defaultdict(list)
    for index, position in enumerate(positions):
        buckets[(elements[index], *_mag_atom_bucket_key(position, bins))].append(index)

    spatial_cache: dict[tuple[int, ...], list[list[int]] | None] = {}

    def spatial_key(rotation: np.ndarray, translation: np.ndarray) -> tuple[int, ...]:
        key_tol = 1e-8
        normalized_translation = normalize_vector_to_zero(translation, atol=key_tol)
        return (
            *np.rint(np.asarray(rotation, dtype=float).ravel() / key_tol).astype(np.int64),
            *np.rint(normalized_translation.ravel() / key_tol).astype(np.int64),
        )

    def spatial_candidates_for_op(real_rotation, translation) -> list[list[int]] | None:
        real_rotation = np.asarray(real_rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        key = spatial_key(real_rotation, translation)
        if key in spatial_cache:
            return spatial_cache[key]

        candidates_by_atom: list[list[int]] = []
        for atom_index, position in enumerate(positions):
            transformed_position = normalize_vector_to_zero(
                real_rotation @ position + translation,
                atol=1e-9,
            )
            bucket_key = _mag_atom_bucket_key(transformed_position, bins)
            candidates: list[int] = []
            for neighbor_key in _mag_atom_neighbor_keys(bucket_key, bins, neighbor_radius):
                for candidate_index in buckets.get((elements[atom_index], *neighbor_key), ()):
                    if not _distance_within_tolerance(
                        abs(occupancies[atom_index] - occupancies[candidate_index]),
                        tol.occupancy,
                    ):
                        continue
                    if not _distance_within_tolerance(
                        getNormInf(transformed_position, positions[candidate_index]),
                        tol.space,
                    ):
                        continue
                    candidates.append(candidate_index)
            if not candidates:
                spatial_cache[key] = None
                return None
            candidates_by_atom.append(candidates)

        spatial_cache[key] = candidates_by_atom
        return candidates_by_atom

    def operation_preserves(spin_rotation, real_rotation, translation) -> bool:
        candidates_by_atom = spatial_candidates_for_op(real_rotation, translation)
        if candidates_by_atom is None:
            return False

        spin_rotation = np.asarray(spin_rotation, dtype=float)
        transformed_moments = moments @ spin_rotation.T
        for atom_index, candidates in enumerate(candidates_by_atom):
            if not any(
                _distance_within_tolerance(
                    np.linalg.norm(transformed_moments[atom_index] - moments[candidate_index]),
                    tol.moment,
                )
                for candidate_index in candidates
            ):
                return False
        return True

    return operation_preserves


def get_ssg_ops(sg,pg,mag_atoms, tol: Tolerances = DEFAULT_TOL):
    """
    Get the spin space group operations that leave the magnetic moments invariant.
    Consider only magnetic atoms.

    Parameters:
        sg: list of tuples
            List of space group operations, each represented as a tuple (rotation_matrix, translation_vector).
        pg: list of np.ndarray
            List of point group operations, each represented as a rotation matrix.
        mag_atoms: list of AtomicSite
        mtol: float
            Tolerance for magnetic moment determination.

    Returns:
        ssg_ops: list of SpinSpaceGroupOperation
            List of spin space group operations.

    """

    ssg_ops = []
    operation_preserves = _build_magnetic_atom_preservation_checker(mag_atoms, tol)
    for R, t in sg:
        R = np.array(R, dtype=np.float64)
        t = np.array(t, dtype=np.float64)
        for Rs in pg:
            Rs = np.array(Rs, dtype=np.float64)
            if operation_preserves(Rs, R, t):
                ssg_ops.append(SpinSpaceGroupOperation(Rs, R, t))
    return ssg_ops


def _normalize_candidate_tol(group_tol) -> Tolerances:
    if group_tol is None:
        return DEFAULT_TOL
    if isinstance(group_tol, Tolerances):
        return group_tol
    value = float(group_tol)
    return Tolerances(
        space=value,
        moment=value,
        m_eig=value,
        occupancy=value,
        m_matrix_tol=value,
    )


def _magnetic_action_residual(op: SpinSpaceGroupOperation, mag_atoms, tol: Tolerances):
    max_position = 0.0
    max_moment = 0.0
    for atom in mag_atoms:
        new_atom = op @ atom
        best = None
        for target in mag_atoms:
            if target.element_symbol != new_atom.element_symbol:
                continue
            position_diff = float(np.max(np.minimum(
                np.abs(np.mod(new_atom.position, 1.0) - np.mod(target.position, 1.0)),
                1.0 - np.abs(np.mod(new_atom.position, 1.0) - np.mod(target.position, 1.0)),
            )))
            moment_diff = float(np.linalg.norm(new_atom.magnetic_moment - target.magnetic_moment))
            occupancy_diff = abs(float(new_atom.occupancy) - float(target.occupancy))
            normalized = max(
                position_diff / max(float(tol.space), 1e-12),
                moment_diff / max(float(tol.moment), 1e-12),
                occupancy_diff / max(float(tol.occupancy), 1e-12),
            )
            candidate = (normalized, position_diff, moment_diff, occupancy_diff)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return float("inf"), float("inf")
        _, position_diff, moment_diff, _occupancy_diff = best
        max_position = max(max_position, position_diff)
        max_moment = max(max_moment, moment_diff)
    return max_position, max_moment


def _magnetic_action_residual_details(op: SpinSpaceGroupOperation, mag_atoms, tol: Tolerances) -> dict:
    max_position = 0.0
    max_moment = 0.0
    max_occupancy = 0.0
    max_normalized = 0.0
    for atom_index, atom in enumerate(mag_atoms):
        new_atom = op @ atom
        best = None
        best_target_index = None
        for target_index, target in enumerate(mag_atoms):
            if target.element_symbol != new_atom.element_symbol:
                continue
            position_diff_components = np.abs(
                np.mod(new_atom.position, 1.0) - np.mod(target.position, 1.0)
            )
            position_diff = float(np.max(np.minimum(position_diff_components, 1.0 - position_diff_components)))
            moment_diff = float(np.linalg.norm(new_atom.magnetic_moment - target.magnetic_moment))
            occupancy_diff = abs(float(new_atom.occupancy) - float(target.occupancy))
            normalized = max(
                position_diff / max(float(tol.space), 1e-12),
                moment_diff / max(float(tol.moment), 1e-12),
                occupancy_diff / max(float(tol.occupancy), 1e-12),
            )
            candidate = (normalized, position_diff, moment_diff, occupancy_diff)
            if best is None or candidate < best:
                best = candidate
                best_target_index = target_index
        if best is None:
            return {
                "normalized": float("inf"),
                "max_position": float("inf"),
                "max_moment": float("inf"),
                "max_occupancy": float("inf"),
                "worst_atom": atom_index,
                "matched_atom": None,
            }
        normalized, position_diff, moment_diff, occupancy_diff = best
        if normalized >= max_normalized:
            worst_atom = atom_index
            matched_atom = best_target_index
        max_normalized = max(max_normalized, normalized)
        max_position = max(max_position, position_diff)
        max_moment = max(max_moment, moment_diff)
        max_occupancy = max(max_occupancy, occupancy_diff)
    return {
        "normalized": float(max_normalized),
        "max_position": float(max_position),
        "max_moment": float(max_moment),
        "max_occupancy": float(max_occupancy),
        "worst_atom": locals().get("worst_atom"),
        "matched_atom": locals().get("matched_atom"),
    }


def _ssg_group_residual_details(ssg_ops, mag_atoms, tol: Tolerances) -> dict:
    max_details = {
        "normalized": 0.0,
        "max_position": 0.0,
        "max_moment": 0.0,
        "max_occupancy": 0.0,
        "worst_op": None,
        "worst_atom": None,
        "matched_atom": None,
    }
    for op_index, op in enumerate(ssg_ops):
        details = _magnetic_action_residual_details(op, mag_atoms, tol)
        if details["normalized"] >= max_details["normalized"]:
            max_details.update(details)
            max_details["worst_op"] = op_index
        else:
            max_details["max_position"] = max(max_details["max_position"], details["max_position"])
            max_details["max_moment"] = max(max_details["max_moment"], details["max_moment"])
            max_details["max_occupancy"] = max(max_details["max_occupancy"], details["max_occupancy"])
    return max_details


def _ssg_operation_preserves_magnetic_atoms(
    op: SpinSpaceGroupOperation,
    mag_atoms,
    tol: Tolerances,
) -> bool:
    operation_preserves = _build_magnetic_atom_preservation_checker(mag_atoms, tol)
    return operation_preserves(op.spin_rotation, op.rotation, op.translation)


def _normalize_fractional_translation(translation, *, boundary_tol=1e-8):
    normalized = np.mod(np.asarray(translation, dtype=float), 1.0)
    normalized[np.abs(normalized - 1.0) <= boundary_tol] = 0.0
    normalized[np.abs(normalized) <= boundary_tol] = 0.0
    return normalized


def _matrix_close(left, right, tol: float, *, rtol: float = 1e-05) -> bool:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    return bool(np.max(np.abs(left_array - right_array) - rtol * np.abs(right_array)) <= tol)


def _translation_close_mod_lattice(left, right, tol: float) -> bool:
    left_array = _normalize_fractional_translation(left)
    right_array = _normalize_fractional_translation(right)
    diff = np.abs(left_array - right_array)
    wrapped = np.minimum(diff, 1.0 - diff)
    return bool(np.max(wrapped) < tol)


def _spin_space_operation_same(left: SpinSpaceGroupOperation, right: SpinSpaceGroupOperation, tol: float) -> bool:
    return bool(
        _matrix_close(left.spin_rotation, right.spin_rotation, tol)
        and _matrix_close(left.rotation, right.rotation, tol)
        and _translation_close_mod_lattice(left.translation, right.translation, tol)
    )


def _matrix_signature(matrix, *, key_tol: float) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(matrix, dtype=float).ravel() / key_tol).astype(np.int64))


def _translation_signature(translation, *, key_tol: float) -> tuple[int, ...]:
    normalized = _normalize_fractional_translation(translation, boundary_tol=key_tol)
    return tuple(np.rint(normalized.ravel() / key_tol).astype(np.int64))


class _SpinSpaceOperationLookup:
    """Fast membership index with a fine bucket grid and verified boundary path."""

    def __init__(self, ops=(), *, tol: float):
        self.tol = float(tol)
        self.key_tol = max(self.tol * 0.01, 1e-8)
        self.ops: list[SpinSpaceGroupOperation] = []
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self._spin_arrays = None
        self._rotation_arrays = None
        self._translation_arrays = None
        self._arrays_dirty = True
        for op in ops:
            self.add(op)

    def _key(self, op: SpinSpaceGroupOperation) -> tuple[int, ...]:
        return (
            *_matrix_signature(op.spin_rotation, key_tol=self.key_tol),
            *_matrix_signature(op.rotation, key_tol=self.key_tol),
            *_translation_signature(op.translation, key_tol=self.key_tol),
        )

    def add(self, op: SpinSpaceGroupOperation) -> None:
        index = len(self.ops)
        self.ops.append(op)
        self._buckets[self._key(op)].append(index)
        self._arrays_dirty = True

    def _ensure_arrays(self) -> None:
        if not self._arrays_dirty:
            return
        if not self.ops:
            self._spin_arrays = np.empty((0, 3, 3), dtype=float)
            self._rotation_arrays = np.empty((0, 3, 3), dtype=float)
            self._translation_arrays = np.empty((0, 3), dtype=float)
        else:
            self._spin_arrays = np.asarray([op.spin_rotation for op in self.ops], dtype=float)
            self._rotation_arrays = np.asarray([op.rotation for op in self.ops], dtype=float)
            self._translation_arrays = np.asarray(
                [_normalize_fractional_translation(op.translation) for op in self.ops],
                dtype=float,
            )
        self._arrays_dirty = False

    def contains(self, op: SpinSpaceGroupOperation) -> bool:
        # The bucket grid is much finer than the audit tolerance. If an
        # operation lands in an existing bucket, every encoded matrix/translation
        # component differs from the bucket representative by at most key_tol,
        # hence it is already a valid positive membership match under self.tol.
        if self._buckets.get(self._key(op)):
            return True

        # Rare tolerance-boundary slow path: keep correctness without letting the
        # normal group-audit path pay Python-loop cost for every product.
        self._ensure_arrays()
        if len(self.ops) == 0:
            return False
        spin = np.asarray(op.spin_rotation, dtype=float)
        rotation = np.asarray(op.rotation, dtype=float)
        translation = _normalize_fractional_translation(op.translation)
        rtol = 1e-05
        spin_close = np.all(
            np.abs(spin - self._spin_arrays) <= self.tol + rtol * np.abs(self._spin_arrays),
            axis=(1, 2),
        )
        rotation_close = np.all(
            np.abs(rotation - self._rotation_arrays)
            <= self.tol + rtol * np.abs(self._rotation_arrays),
            axis=(1, 2),
        )
        diff = np.abs(translation - self._translation_arrays)
        wrapped = np.minimum(diff, 1.0 - diff)
        translation_close = np.max(wrapped, axis=1) < self.tol
        return bool(np.any(spin_close & rotation_close & translation_close))


class _SpatialOperationLookup:
    """Fast membership index for spatial operation audits."""

    def __init__(self, ops=(), *, tol: float):
        self.tol = float(tol)
        self.key_tol = max(self.tol * 0.01, 1e-8)
        self.ops = []
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self._rotation_arrays = None
        self._translation_arrays = None
        self._arrays_dirty = True
        for op in ops:
            self.add(op)

    def _key(self, op) -> tuple[int, ...]:
        rotation, translation = op
        return (
            *_matrix_signature(rotation, key_tol=self.key_tol),
            *_translation_signature(translation, key_tol=self.key_tol),
        )

    def add(self, op) -> None:
        normalized = [np.asarray(op[0], dtype=float), _normalize_fractional_translation(op[1])]
        index = len(self.ops)
        self.ops.append(normalized)
        self._buckets[self._key(normalized)].append(index)
        self._arrays_dirty = True

    def _ensure_arrays(self) -> None:
        if not self._arrays_dirty:
            return
        if not self.ops:
            self._rotation_arrays = np.empty((0, 3, 3), dtype=float)
            self._translation_arrays = np.empty((0, 3), dtype=float)
        else:
            self._rotation_arrays = np.asarray([op[0] for op in self.ops], dtype=float)
            self._translation_arrays = np.asarray([op[1] for op in self.ops], dtype=float)
        self._arrays_dirty = False

    def contains(self, op) -> bool:
        if self._buckets.get(self._key(op)):
            return True

        self._ensure_arrays()
        if len(self.ops) == 0:
            return False
        rotation = np.asarray(op[0], dtype=float)
        translation = _normalize_fractional_translation(op[1])
        rotation_close = np.all(np.abs(rotation - self._rotation_arrays) <= self.tol, axis=(1, 2))
        diff = np.abs(translation - self._translation_arrays)
        wrapped = np.minimum(diff, 1.0 - diff)
        translation_close = np.max(wrapped, axis=1) < self.tol
        return bool(np.any(rotation_close & translation_close))


def _spin_space_operation_signature(op: SpinSpaceGroupOperation, *, key_tol: float) -> tuple[int, ...]:
    return (
        *_matrix_signature(op.spin_rotation, key_tol=key_tol),
        *_matrix_signature(op.rotation, key_tol=key_tol),
        *_translation_signature(op.translation, key_tol=key_tol),
    )


def _complete_ssg_ops_by_closure(
    ssg_ops,
    mag_atoms,
    group_tol=DEFAULT_TOL,
    *,
    label="matched SSG",
    preserve_cache: dict[tuple[int, ...], bool] | None = None,
):
    tol = _normalize_candidate_tol(group_tol)
    audit_tol = _candidate_audit_tol(tol)
    completed = []
    lookup = _SpinSpaceOperationLookup(tol=audit_tol)
    for op in ssg_ops:
        if not lookup.contains(op):
            completed.append(op)
            lookup.add(op)

    if not completed:
        return completed

    operation_preserves = _build_magnetic_atom_preservation_checker(mag_atoms, tol)
    limit = max(512, len(completed) * len(completed) * 4 + len(completed) + 1)
    changed = True
    while changed:
        changed = False
        snapshot = list(completed)
        for left_index, left in enumerate(snapshot):
            for right_index, right in enumerate(snapshot):
                product = left @ right
                if lookup.contains(product):
                    continue
                cache_key = _spin_space_operation_signature(product, key_tol=lookup.key_tol)
                preserves = None if preserve_cache is None else preserve_cache.get(cache_key)
                if preserves is None:
                    preserves = operation_preserves(
                        product.spin_rotation,
                        product.rotation,
                        product.translation,
                    )
                    if preserve_cache is not None:
                        preserve_cache[cache_key] = preserves
                if not preserves:
                    position_residual, moment_residual = _magnetic_action_residual(product, mag_atoms, tol)
                    raise ValueError(
                        f"{label} closure product {left_index}*{right_index} does not preserve "
                        "the magnetic structure under the active tolerance "
                        f"(max position residual={position_residual:.6g}, "
                        f"max moment residual={moment_residual:.6g})"
                    )
                completed.append(product)
                lookup.add(product)
                changed = True
                if len(completed) > limit:
                    raise ValueError(
                        f"{label} closure exceeded {limit} operations; the candidate generators "
                        "do not define a finite audited SSG under the active tolerance."
                    )
    return completed

def normalize_vector(u):
    """norm"""
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("zero vector can't be normalized!")
    return u / norm


def _canonicalize_direction(direction, tol=1e-8):
    direction = normalize_vector(np.asarray(direction, dtype=float))
    for value in direction:
        if abs(value) > tol:
            if value < 0:
                direction = -direction
            break
    return direction

def reflection_matrix(axis):
    """give a mirror perpendicular to axis"""
    u = normalize_vector(np.array(axis))
    u = u.reshape(3, 1)
    I = np.eye(3)
    reflection_matrix = I - 2 * np.dot(u, u.T)
    return reflection_matrix


def dedup_moments_with_tol(types, moments, tol=0.01):
    # print(types, moments, tol)
    moments = np.asarray(moments, float)
    types = np.asarray(types)


    idx = np.lexsort((moments[:, 2], moments[:, 1], moments[:, 0], types))
    moments_sorted = moments[idx]
    types_sorted = types[idx]

    new_moments = [moments_sorted[0]]
    new_types = [types_sorted[0]]

    for i in range(1, len(moments_sorted)):
        same_type = types_sorted[i] == new_types[-1]
        same_moment = np.linalg.norm(moments_sorted[i] - new_moments[-1]) <= tol
        if not (same_type and same_moment):
            new_moments.append(moments_sorted[i])
            new_types.append(types_sorted[i])

    return np.array(new_types), np.array(new_moments)


def _best_fit_axis(moments):
    moments = np.asarray(moments, dtype=float)
    if len(moments) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    _u, _s, vh = np.linalg.svd(moments, full_matrices=True)
    return _canonicalize_direction(vh[0])


def _best_fit_plane_normal(moments):
    moments = np.asarray(moments, dtype=float)
    if len(moments) <= 1:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    _u, _s, vh = np.linalg.svd(moments, full_matrices=True)
    return _canonicalize_direction(vh[-1])


def _candidate_directions_from_moments(moments):
    moments = np.asarray(moments, dtype=float)
    candidates = []

    def add(direction):
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return
        candidates.append(_canonicalize_direction(direction))

    if len(moments) == 0:
        add([0.0, 0.0, 1.0])
    else:
        add(_best_fit_axis(moments))
        add(_best_fit_plane_normal(moments))
        for moment in moments:
            add(moment)
        for i in range(len(moments)):
            for j in range(i + 1, len(moments)):
                add(moments[i] + moments[j])
                add(moments[i] - moments[j])
                add(np.cross(moments[i], moments[j]))

    # The candidate set grows quadratically with the number of nonzero moments.
    # Use spatial buckets to avoid an O(N^2) allclose scan while preserving the
    # same final near-duplicate predicate for directions in neighboring buckets.
    unique = []
    buckets = {}
    bucket_width = 2e-5
    neighbor_offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for direction in candidates:
        key = tuple(np.floor(direction / bucket_width).astype(int).tolist())
        duplicate = False
        for offset in neighbor_offsets:
            neighbor_key = (
                key[0] + offset[0],
                key[1] + offset[1],
                key[2] + offset[2],
            )
            for existing in buckets.get(neighbor_key, ()):
                if np.allclose(direction, existing, atol=1e-6):
                    duplicate = True
                    break
            if duplicate:
                break
        if duplicate:
            continue
        unique.append(direction)
        buckets.setdefault(key, []).append(direction)
    return unique


def _candidate_chunk_size(moment_count, target_values=3_000_000):
    return max(1, int(target_values // max(1, moment_count)))


def _best_collinear_axis_from_candidates(moments, candidates):
    candidates = np.asarray(candidates, dtype=float)
    moments = np.asarray(moments, dtype=float)
    best_index = 0
    best_residual = float("inf")
    chunk_size = _candidate_chunk_size(len(moments))
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        residuals = 2.0 * np.linalg.norm(
            np.cross(moments[None, :, :], chunk[:, None, :]),
            axis=2,
        ).max(axis=1)
        local_index = int(np.argmin(residuals))
        local_residual = float(residuals[local_index])
        if local_residual < best_residual - 1e-10:
            best_residual = local_residual
            best_index = start + local_index
    return candidates[best_index], best_residual


def _best_coplanar_normal_from_candidates(moments, candidates):
    candidates = np.asarray(candidates, dtype=float)
    moments = np.asarray(moments, dtype=float)
    best_index = 0
    best_residual = float("inf")
    chunk_size = _candidate_chunk_size(len(moments))
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        residuals = 2.0 * np.abs(chunk @ moments.T).max(axis=1)
        local_index = int(np.argmin(residuals))
        local_residual = float(residuals[local_index])
        if local_residual < best_residual - 1e-10:
            best_residual = local_residual
            best_index = start + local_index
    return candidates[best_index], best_residual


def _collinear_residual(moments, axis):
    axis = _canonicalize_direction(axis)
    moments = np.asarray(moments, dtype=float)
    if len(moments) == 0:
        return 0.0
    # Use the same scale as the finite C_infinity_v representatives used by
    # SSG matching: a C2 rotation about the axis moves the perpendicular
    # component by 2 * |m_perp|.
    return 2.0 * max(float(np.linalg.norm(np.cross(moment, axis))) for moment in moments)


def _coplanar_residual(moments, plane_normal):
    plane_normal = _canonicalize_direction(plane_normal)
    moments = np.asarray(moments, dtype=float)
    if len(moments) == 0:
        return 0.0
    # Report the residual in the same units as a spin-only mirror operation:
    # ||(I - 2nn^T)m - m|| = 2 |m.n|.  This makes "Coplanar under mtol"
    # equivalent to accepting the spin-only mirror under that same mtol.
    return 2.0 * max(float(abs(np.dot(moment, plane_normal))) for moment in moments)


def _best_collinear_axis(moments):
    candidates = _candidate_directions_from_moments(moments)
    return _best_collinear_axis_from_candidates(moments, candidates)


def _best_coplanar_normal(moments):
    candidates = _candidate_directions_from_moments(moments)
    return _best_coplanar_normal_from_candidates(moments, candidates)


def _configuration_details(moments, mtol):
    moments = np.asarray(moments, dtype=float)
    if len(moments) == 0:
        max_moment_norm = 0.0
    else:
        max_moment_norm = float(np.max(np.linalg.norm(moments, axis=1)))

    if max_moment_norm <= mtol:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        return {
            "configuration": "Nonmagnetic",
            "constraint_rank": 0,
            "spin_point_group_semantics": "O3",
            "collinear_axis": axis,
            "collinear_residual": 0.0,
            "coplanar_normal": axis,
            "coplanar_residual": 0.0,
            "max_moment_norm": max_moment_norm,
        }

    if len(moments) <= 1:
        axis = _canonicalize_direction(moments[0]) if len(moments) == 1 else np.array([0.0, 0.0, 1.0])
        return {
            "configuration": "Collinear",
            "constraint_rank": 1,
            "spin_point_group_semantics": "O2_about_axis",
            "collinear_axis": axis,
            "collinear_residual": 0.0,
            "coplanar_normal": axis,
            "coplanar_residual": 0.0,
            "max_moment_norm": max_moment_norm,
        }

    candidates = _candidate_directions_from_moments(moments)
    collinear_axis, collinear_residual = _best_collinear_axis_from_candidates(moments, candidates)
    coplanar_normal, coplanar_residual = _best_coplanar_normal_from_candidates(moments, candidates)
    if collinear_residual <= mtol:
        configuration = "Collinear"
        constraint_rank = 1
        spin_point_group_semantics = "O2_about_axis"
    elif coplanar_residual <= mtol:
        configuration = "Coplanar"
        constraint_rank = 2
        spin_point_group_semantics = "plane_mirror"
    else:
        configuration = "Noncoplanar"
        constraint_rank = 3
        spin_point_group_semantics = "finite_discrete"
    return {
        "configuration": configuration,
        "constraint_rank": constraint_rank,
        "spin_point_group_semantics": spin_point_group_semantics,
        "collinear_axis": collinear_axis,
        "collinear_residual": float(collinear_residual),
        "coplanar_normal": coplanar_normal,
        "coplanar_residual": float(coplanar_residual),
        "max_moment_norm": max_moment_norm,
    }


def _classify_moment_configuration(moments, mtol):
    return _configuration_details(moments, mtol)["configuration"]


def _deduplicate_pg_operations(pg_operations, tol):
    return [np.asarray(op, dtype=np.float64) for op in deduplicate_matrix_pairs(pg_operations, tol=tol)]


def _linear_main_axis(pg):
    eigvals = np.asarray(pg.eigvals, dtype=float)
    index = int(np.argmin(np.abs(eigvals)))
    return normalize_vector(np.asarray(pg.principal_axes[index], dtype=float))


def _materialize_linear_pg_ops(pg_symbol, pg):
    main_axis = _linear_main_axis(pg)
    return _semantic_collinear_pg_ops(main_axis, include_axis_flip=(pg_symbol == 'D*h'))


def _semantic_collinear_pg_ops(axis, *, include_axis_flip=True):
    main_axis = normalize_vector(np.asarray(axis, dtype=float))
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    ref_axis = z_axis if abs(np.dot(main_axis, z_axis)) < 0.95 else x_axis

    plane_normal = np.cross(main_axis, ref_axis)
    if np.linalg.norm(plane_normal) < 1e-8:
        ref_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        plane_normal = np.cross(main_axis, ref_axis)

    mirror_v = reflection_matrix(plane_normal)
    rotate_2 = SymmOp.from_axis_angle_and_translation(main_axis, 180).rotation_matrix
    mirror_2 = rotate_2 @ mirror_v

    extra_ops = [mirror_v, rotate_2, mirror_2]
    if include_axis_flip:
        extra_ops += [-mirror_v, -rotate_2, -mirror_2]
    return extra_ops


def _valid_rotation_for_pg_analyzer(pg, rotation) -> bool:
    try:
        return bool(pg.is_valid_op(SymmOp.from_rotation_and_translation(rotation)))
    except Exception:
        return False


def _rotation_maps_axis_to_sign(rotation, axis, sign: int, tol: float) -> bool:
    axis = normalize_vector(np.asarray(axis, dtype=float))
    mapped = np.asarray(rotation, dtype=float) @ axis
    return bool(np.allclose(mapped, sign * axis, atol=tol, rtol=0))


def _is_nontrivial_axis_preserving_proper_rotation(rotation, axis, tol: float) -> bool:
    rotation = np.asarray(rotation, dtype=float)
    if not _rotation_maps_axis_to_sign(rotation, axis, 1, tol):
        return False
    if float(np.linalg.det(rotation)) <= 0.0:
        return False
    return not np.allclose(rotation, np.eye(3), atol=tol, rtol=0)


def _is_axis_preserving_improper_operation(rotation, axis, tol: float) -> bool:
    rotation = np.asarray(rotation, dtype=float)
    if not _rotation_maps_axis_to_sign(rotation, axis, 1, tol):
        return False
    return float(np.linalg.det(rotation)) < 0.0


def _collinear_finite_representative_compatibility(pg_operations, axis, tol: float) -> int:
    has_axis_rotation = any(
        _is_nontrivial_axis_preserving_proper_rotation(op, axis, tol)
        for op in pg_operations
    )
    has_axis_mirror = any(
        _is_axis_preserving_improper_operation(op, axis, tol)
        for op in pg_operations
    )
    if not (has_axis_rotation and has_axis_mirror):
        return 0
    has_axis_flip = any(_rotation_maps_axis_to_sign(op, axis, -1, tol) for op in pg_operations)
    return 2 if has_axis_flip else 1


def _collinear_symbol_from_operations(pg_operations, axis, tol: float) -> str:
    has_axis_flip = any(_rotation_maps_axis_to_sign(op, axis, -1, tol) for op in pg_operations)
    return "D*h" if has_axis_flip else "C*v"


def _pg_operations_contain_matrix(pg_operations, target, tol):
    target = np.asarray(target, dtype=float)
    return any(np.allclose(np.asarray(op, dtype=float), target, atol=tol, rtol=0) for op in pg_operations)


def _pg_operations_contain_all(pg_operations, targets, tol) -> bool:
    return all(_pg_operations_contain_matrix(pg_operations, target, tol) for target in targets)


def _configuration_compatibility(
    pg_symbol,
    configuration,
    *,
    pg_operations=None,
    configuration_details=None,
    tol=0.01,
):
    linear_symbol = pg_symbol in ('C*v', 'D*h')
    if configuration == "Nonmagnetic":
        return 1 if pg_symbol == "Kh" else 0
    if pg_symbol == "Kh":
        return 0
    if configuration == "Coplanar":
        if linear_symbol:
            return 0
        if pg_operations is None or configuration_details is None:
            return 1
        mirror = reflection_matrix(configuration_details["coplanar_normal"])
        return 1 if _pg_operations_contain_matrix(pg_operations, mirror, tol) else 0
    if configuration == "Collinear":
        if pg_operations is None or configuration_details is None:
            return 1 if linear_symbol else 0
        return _collinear_finite_representative_compatibility(
            pg_operations,
            configuration_details["collinear_axis"],
            tol,
        )
    if configuration == "Noncoplanar":
        return 0 if linear_symbol else 1
    return 0


def _candidate_eigen_tolerances(meigtol):
    candidates = []
    for factor in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        value = max(1e-12, min(1e-1, float(meigtol) * factor))
        candidates.append(value)
    candidates.append(float(meigtol))
    return sorted(set(candidates))


def _pg_operation_signature(pg_operations, digits=8):
    return tuple(
        sorted(
            tuple(np.asarray(op, dtype=float).round(digits).reshape(-1).tolist())
            for op in pg_operations
        )
    )


def _identity_pg_candidate(meigtol):
    return {
        "symbol": "C1",
        "operations": [np.eye(3)],
        "eig_tol": float(meigtol),
        "pg": None,
        "audit_rank": 10_000,
        "is_floor_candidate": True,
    }


def _semantic_rank_pg_candidate(configuration_details, meigtol):
    configuration = configuration_details["configuration"]
    if configuration == "Collinear":
        operations = [
            np.eye(3),
            *_semantic_collinear_pg_ops(
                configuration_details["collinear_axis"],
                include_axis_flip=False,
            ),
        ]
        return {
            "symbol": "C*v",
            "operations": _deduplicate_pg_operations(operations, tol=float("1e-8")),
            "eig_tol": float(meigtol),
            "pg": None,
            "audit_rank": 20_000,
            "is_semantic_rank_candidate": True,
        }
    if configuration == "Coplanar":
        operations = [
            np.eye(3),
            reflection_matrix(configuration_details["coplanar_normal"]),
        ]
        return {
            "symbol": "Cs",
            "operations": _deduplicate_pg_operations(operations, tol=float("1e-8")),
            "eig_tol": float(meigtol),
            "pg": None,
            "audit_rank": 20_000,
            "is_semantic_rank_candidate": True,
        }
    return None


def _candidate_eigvals_key(candidate):
    pg = candidate.get("pg")
    if pg is None:
        return ()
    return tuple(np.round(np.asarray(pg.eigvals, dtype=float), 8))


def _matrix_operation_equivalent(left, right, tol: float) -> bool:
    return _matrix_close(left, right, tol, rtol=0)


class _MatrixOperationLookup:
    def __init__(self, matrices=(), *, tol: float):
        self.tol = float(tol)
        self.key_tol = max(self.tol * 0.01, 1e-8)
        self.ops: list[np.ndarray] = []
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self._array = None
        self._array_dirty = True
        for matrix in matrices:
            self.add(matrix)

    def _key(self, matrix) -> tuple[int, ...]:
        return _matrix_signature(matrix, key_tol=self.key_tol)

    def add(self, matrix) -> None:
        arr = np.asarray(matrix, dtype=float).reshape(3, 3)
        index = len(self.ops)
        self.ops.append(arr)
        self._buckets[self._key(arr)].append(index)
        self._array_dirty = True

    def _ensure_array(self) -> None:
        if not self._array_dirty:
            return
        self._array = (
            np.empty((0, 3, 3), dtype=float)
            if not self.ops
            else np.asarray(self.ops, dtype=float)
        )
        self._array_dirty = False

    def find(self, matrix):
        arr = np.asarray(matrix, dtype=float).reshape(3, 3)
        for index in self._buckets.get(self._key(arr), ()):
            if _matrix_close(arr, self.ops[index], self.tol, rtol=0):
                return self.ops[index]

        self._ensure_array()
        if len(self.ops) == 0:
            return None
        close = np.all(np.abs(arr - self._array) <= self.tol, axis=(1, 2))
        indices = np.flatnonzero(close)
        if len(indices) == 0:
            return None
        return self.ops[int(indices[0])]

    def contains(self, matrix) -> bool:
        return self.find(matrix) is not None


def _find_matching_matrix_operation(matrix, universe, tol: float, lookup: _MatrixOperationLookup | None = None):
    if lookup is None:
        lookup = _MatrixOperationLookup(universe, tol=tol)
    return lookup.find(matrix)


def _point_group_closure_within_universe(generators, universe, tol: float, *, limit: int = 256):
    universe_lookup = _MatrixOperationLookup(universe, tol=tol)
    identity = _find_matching_matrix_operation(np.eye(3), universe, tol, lookup=universe_lookup)
    if identity is None:
        return None

    group = [identity]
    group_lookup = _MatrixOperationLookup(group, tol=tol)
    for generator in generators:
        matched = _find_matching_matrix_operation(generator, universe, tol, lookup=universe_lookup)
        if matched is None:
            return None
        if not group_lookup.contains(matched):
            group.append(matched)
            group_lookup.add(matched)

    index = 0
    while index < len(group):
        left = group[index]
        index += 1
        snapshot = list(group)
        for right in snapshot:
            for product in (left @ right, right @ left):
                matched = _find_matching_matrix_operation(product, universe, tol, lookup=universe_lookup)
                if matched is None:
                    return None
                if group_lookup.contains(matched):
                    continue
                group.append(matched)
                group_lookup.add(matched)
                if len(group) > limit:
                    return None
    return _deduplicate_pg_operations(group, tol=tol)


def _point_group_subgroups_from_operations(pg_operations, tol: float):
    universe = _deduplicate_pg_operations([np.eye(3), *pg_operations], tol=tol)
    generator_sets = [()]
    generator_sets.extend((operation,) for operation in universe)
    generator_sets.extend(itertools.combinations(universe, 2))

    subgroups = []
    seen = set()
    for generators in generator_sets:
        subgroup = _point_group_closure_within_universe(generators, universe, tol)
        if subgroup is None:
            continue
        signature = _pg_operation_signature(subgroup)
        if signature in seen:
            continue
        seen.add(signature)
        subgroups.append(subgroup)

    subgroups.sort(key=lambda ops: (-len(ops), _pg_operation_signature(ops)))
    return subgroups


def _build_pg_candidate_variants(
    moment_types,
    moment_vectors,
    *,
    mtol,
    eig_tol,
    matrix_tol,
    configuration_details=None,
):
    pg = PointGroupAnalyzer(Molecule(moment_types, moment_vectors), tolerance=mtol, eigen_tolerance=eig_tol)
    operation_sets = getattr(pg, "_audited_operation_sets", None)
    if operation_sets is None:
        operation_sets = [pg.get_symmetry_operations()]

    candidates = []
    seen = set()
    for audit_rank, operation_set in enumerate(operation_sets):
        base_pg_operations = [np.array(i.rotation_matrix, dtype=np.float64) for i in operation_set]
        pg_symbol = str(pg.get_pointgroup())
        if operation_set is not getattr(pg, "_symmetry_operations", None):
            info = PointGroupAnalyzer._identify_closed_point_group(
                base_pg_operations,
                tol=min(float(mtol), float(pg.mat_tol)),
            )
            pg_symbol = info[-1]
            if pg_symbol in {"C1v", "C1h"}:
                pg_symbol = "Cs"

        pg_operations = list(base_pg_operations)
        if (
            configuration_details is not None
            and configuration_details.get("configuration") == "Collinear"
        ):
            semantic_ops = _semantic_collinear_pg_ops(
                configuration_details["collinear_axis"],
                include_axis_flip=True,
            )
            pg_operations += [
                op
                for op in semantic_ops
                if _valid_rotation_for_pg_analyzer(pg, op)
            ]
            pg_symbol = _collinear_symbol_from_operations(pg_operations, configuration_details["collinear_axis"], mtol)
        elif str(pg.get_pointgroup()) in ('C*v', 'D*h'):
            pg_operations += _materialize_linear_pg_ops(str(pg.get_pointgroup()), pg)
        if (
            configuration_details is not None
            and configuration_details.get("configuration") == "Coplanar"
        ):
            pg_operations.append(reflection_matrix(configuration_details["coplanar_normal"]))
        pg_operations = _deduplicate_pg_operations(pg_operations, tol=matrix_tol)
        if _point_group_closure_within_universe(
            pg_operations,
            pg_operations,
            tol=matrix_tol,
        ) is None:
            continue

        key = (pg_symbol, _pg_operation_signature(pg_operations))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "symbol": pg_symbol,
                "operations": pg_operations,
                "eig_tol": float(eig_tol),
                "pg": pg,
                "audit_rank": audit_rank,
            }
        )
    return candidates


def _build_pg_candidates(moments, atom_types, mtol, meigtol, matrix_tol=0.01):
    non_zero_indices = np.where(np.linalg.norm(moments, axis=1) > MAGNETIC_PRESENCE_TOL)[0]

    filtered_moments = np.array([moments[i] for i in non_zero_indices], dtype=float)
    filtered_types = np.array([atom_types[i] for i in non_zero_indices])

    configuration_details = _configuration_details(filtered_moments, mtol)
    configuration = configuration_details["configuration"]
    if configuration == "Nonmagnetic":
        raise MagneticToleranceDegeneracyError(
            f"{NONMAGNETIC_MTOL_ERROR} "
            f"mtol={float(mtol):.6g}, max_moment_norm="
            f"{configuration_details['max_moment_norm']:.6g}."
        )

    unique_types, unique_moments = dedup_moments_with_tol(filtered_types, filtered_moments, mtol)

    candidates = []
    candidate_failures = []
    seen = set()
    for eig_tol in _candidate_eigen_tolerances(meigtol):
        try:
            eig_candidates = _build_pg_candidate_variants(
                unique_types,
                unique_moments,
                mtol=mtol,
                eig_tol=eig_tol,
                matrix_tol=matrix_tol,
                configuration_details=configuration_details,
            )
        except ValueError as exc:
            candidate_failures.append(f"eig_tol={eig_tol}: {exc}")
            continue
        if not eig_candidates:
            candidate_failures.append(f"eig_tol={eig_tol}: no audited point-group candidate")
            continue
        for candidate in eig_candidates:
            key = (
                candidate["symbol"],
                len(candidate["operations"]),
                _candidate_eigvals_key(candidate),
                _pg_operation_signature(candidate["operations"]),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    semantic_candidate = _semantic_rank_pg_candidate(configuration_details, meigtol)
    if semantic_candidate is not None:
        key = (
            semantic_candidate["symbol"],
            len(semantic_candidate["operations"]),
            _candidate_eigvals_key(semantic_candidate),
            _pg_operation_signature(semantic_candidate["operations"]),
        )
        if key not in seen:
            seen.add(key)
            candidates.append(semantic_candidate)

    if not candidates:
        candidates.append(_identity_pg_candidate(meigtol))
    else:
        identity_candidate = _identity_pg_candidate(meigtol)
        identity_key = (
            identity_candidate["symbol"],
            len(identity_candidate["operations"]),
            _candidate_eigvals_key(identity_candidate),
            _pg_operation_signature(identity_candidate["operations"]),
        )
        if identity_key not in seen:
            candidates.append(identity_candidate)

    return {
        "configuration": configuration,
        "configuration_details": configuration_details,
        "mtol": float(mtol),
        "filtered_moments": filtered_moments,
        "filtered_types": filtered_types,
        "unique_types": unique_types,
        "unique_moments": unique_moments,
        "candidates": candidates,
        "candidate_failures": candidate_failures,
    }


def _select_get_pg_candidate(bundle, meigtol):
    configuration = bundle["configuration"]
    configuration_details = bundle["configuration_details"]
    candidates = [candidate for candidate in bundle["candidates"] if not candidate.get("is_floor_candidate")]
    if not candidates:
        candidates = list(bundle["candidates"])

    compatible_candidates = [
        candidate
        for candidate in candidates
        if _configuration_compatibility(
            candidate["symbol"],
            configuration,
            pg_operations=candidate["operations"],
            configuration_details=configuration_details,
            tol=max(float(bundle.get("mtol", meigtol)), 1e-12),
        )
        > 0
    ]
    if not compatible_candidates:
        symbols = ", ".join(
            f"{candidate['symbol']}[{len(candidate['operations'])}]"
            for candidate in candidates[:8]
        )
        raise ValueError(
            "Point-group candidates are inconsistent with the magnetic "
            f"{configuration} spin-rank semantics: {symbols}"
        )

    base_candidate = min(compatible_candidates, key=lambda candidate: abs(candidate["eig_tol"] - meigtol))
    if _configuration_compatibility(
        base_candidate["symbol"],
        configuration,
        pg_operations=base_candidate["operations"],
        configuration_details=configuration_details,
        tol=max(float(bundle.get("mtol", meigtol)), 1e-12),
    ) > 0:
        return base_candidate

    return max(
        compatible_candidates,
        key=lambda candidate: (
            _configuration_compatibility(
                candidate["symbol"],
                configuration,
                pg_operations=candidate["operations"],
                configuration_details=configuration_details,
                tol=max(float(bundle.get("mtol", meigtol)), 1e-12),
            ),
            len(candidate["operations"]),
            -abs(np.log10(max(candidate["eig_tol"], 1e-12) / max(meigtol, 1e-12))),
        ),
    )


def _space_operation_signature(rotation, translation, digits=6):
    rotation_key = tuple(np.asarray(rotation, dtype=float).round(digits).reshape(-1).tolist())
    translation = np.mod(np.asarray(translation, dtype=float), 1.0)
    translation[np.isclose(translation, 1.0, atol=10 ** (-digits))] = 0.0
    translation_key = tuple(np.asarray(translation, dtype=float).round(digits).tolist())
    return rotation_key, translation_key


def _distinct_real_space_count(ssg_ops):
    return len({_space_operation_signature(op[1], op[2]) for op in ssg_ops})


def _distinct_space_operation_count(space_operations_list):
    return len({_space_operation_signature(rotation, translation) for rotation, translation in space_operations_list})


def _ssg_signature(ssg_ops):
    return tuple(
        sorted(
            (
                tuple(np.asarray(op[0], dtype=float).round(6).reshape(-1).tolist()),
                *_space_operation_signature(op[1], op[2]),
            )
            for op in ssg_ops
        )
    )


def _ssg_ops_equivalent(left_ops, right_ops, tol):
    if len(left_ops) != len(right_ops):
        return False
    unmatched = list(right_ops)
    for left in left_ops:
        match_index = next(
            (
                index
                for index, right in enumerate(unmatched)
                if left.is_same_with(right, atol=tol)
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return True


def _ssg_ops_subset(subset_ops, superset_ops, tol):
    for left in subset_ops:
        if not any(left.is_same_with(right, atol=tol) for right in superset_ops):
            return False
    return True


def _maximal_audited_ssg_subgroups_from_generators(
    raw_ssg_ops,
    mag_atoms,
    group_tol=DEFAULT_TOL,
    *,
    label="matched SSG",
    preserve_cache: dict[tuple[int, ...], bool] | None = None,
):
    if not raw_ssg_ops:
        return []

    tol = _normalize_candidate_tol(group_tol)
    audit_tol = _candidate_audit_tol(tol)
    lookup = _SpinSpaceOperationLookup(tol=audit_tol)
    unique_ops = []
    for op in [SpinSpaceGroupOperation.identity(), *raw_ssg_ops]:
        if lookup.contains(op):
            continue
        unique_ops.append(op)
        lookup.add(op)

    identity = SpinSpaceGroupOperation.identity()
    generators = [op for op in unique_ops if not op.is_same_with(identity, atol=audit_tol)]
    spin_representatives = {}
    identity_spin_generators = []
    for op in generators:
        if _matrix_close(op.spin_rotation, np.eye(3), audit_tol):
            identity_spin_generators.append(op)
        spin_key = _matrix_signature(op.spin_rotation, key_tol=max(audit_tol * 0.01, 1e-8))
        previous = spin_representatives.get(spin_key)
        op_is_spin_only = (
            _matrix_close(op.rotation, np.eye(3), audit_tol)
            and _translation_close_mod_lattice(op.translation, np.zeros(3), audit_tol)
        )
        previous_is_spin_only = (
            previous is not None
            and _matrix_close(previous.rotation, np.eye(3), audit_tol)
            and _translation_close_mod_lattice(previous.translation, np.zeros(3), audit_tol)
        )
        if previous is None or (op_is_spin_only and not previous_is_spin_only):
            spin_representatives[spin_key] = op

    pair_generators = []
    pair_lookup = _SpinSpaceOperationLookup(tol=audit_tol)
    for op in [*spin_representatives.values(), *identity_spin_generators]:
        if pair_lookup.contains(op):
            continue
        pair_generators.append(op)
        pair_lookup.add(op)

    generator_sets = [()]
    generator_sets.extend((op,) for op in generators)
    generator_sets.extend(itertools.combinations(pair_generators, 2))

    valid = []
    seen = set()
    if preserve_cache is None:
        preserve_cache = {}
    for generator_set in generator_sets:
        seed_ops = [identity, *generator_set]
        try:
            subgroup_ops = _complete_ssg_ops_by_closure(
                seed_ops,
                mag_atoms,
                group_tol=group_tol,
                label=f"{label} generated subgroup",
                preserve_cache=preserve_cache,
            )
            subgroup_ops = _project_ssg_spin_rotations_to_exact_point_group(
                subgroup_ops,
                mag_atoms,
                group_tol=group_tol,
                label=f"{label} generated subgroup",
            )
            subgroup_ops = _complete_ssg_ops_by_closure(
                subgroup_ops,
                mag_atoms,
                group_tol=group_tol,
                label=f"{label} projected generated subgroup",
                preserve_cache=preserve_cache,
            )
        except ValueError:
            continue
        if _candidate_audit_failure(subgroup_ops, group_tol=group_tol) is not None:
            continue
        signature = _ssg_signature(subgroup_ops)
        if signature in seen:
            continue
        seen.add(signature)
        valid.append(subgroup_ops)

    maximal = []
    for subgroup_ops in valid:
        if any(
            other_ops is not subgroup_ops
            and _ssg_ops_subset(subgroup_ops, other_ops, audit_tol)
            for other_ops in valid
        ):
            continue
        maximal.append(subgroup_ops)

    return sorted(
        maximal,
        key=lambda ops: (
            -len(ops),
            -_distinct_real_space_count(ops),
            _ssg_signature(ops),
        ),
    )


def _nsspg_invariant_failure(ssg: SpinSpaceGroup) -> str | None:
    g0_count = len(ssg.G0_ops)
    l0_count = len(ssg.L0_ops)
    spin_translation_count = len(ssg.n_spin_translation_group)
    pure_translation_count = len(ssg.pure_t_group)
    spin_point_count = len(ssg.n_spin_part_point_ops)

    if l0_count == 0:
        return "L0 has no operations"
    if pure_translation_count == 0:
        return "pure translation group has no operations"
    if g0_count % l0_count != 0:
        return f"|G0|={g0_count} is not divisible by |L0|={l0_count}"
    if spin_translation_count % pure_translation_count != 0:
        return (
            f"|spin translation|={spin_translation_count} is not divisible by "
            f"|pure translation|={pure_translation_count}"
        )

    itik = g0_count // l0_count
    ik = spin_translation_count // pure_translation_count
    if ik == 0:
        return "ik is zero"
    if itik % ik != 0:
        return f"itik={itik} is not divisible by ik={ik}"

    it = itik // ik
    if it * ik != spin_point_count:
        return f"it*ik={it * ik} does not match |nsspg|={spin_point_count}"
    return None


def _candidate_audit_tol(group_tol) -> float:
    if isinstance(group_tol, Tolerances):
        return max(float(group_tol.space), float(group_tol.m_matrix_tol))
    return float(group_tol)


def _spin_matrix_projection_tol(mag_atoms, tol: Tolerances) -> float:
    moment_norms = [
        float(np.linalg.norm(atom.magnetic_moment))
        for atom in mag_atoms
        if np.linalg.norm(atom.magnetic_moment) > MAGNETIC_PRESENCE_TOL
    ]
    moment_scale = float(np.median(moment_norms)) if moment_norms else 1.0
    if not np.isfinite(moment_scale) or moment_scale < 1e-12:
        moment_scale = 1.0

    derived = float(tol.moment) / moment_scale
    lower = 0.1 * float(tol.moment)
    upper = float(tol.moment)
    return max(1e-8, min(upper, max(lower, derived)))


def _matrix_group_closure_exact(generators, *, tol: float = 1e-10, limit: int = 256):
    group = [np.eye(3)]
    for generator in generators:
        matrix = np.asarray(generator, dtype=float)
        if not any(np.allclose(matrix, existing, atol=tol, rtol=0) for existing in group):
            group.append(matrix)

    index = 0
    while index < len(group):
        left = group[index]
        index += 1
        snapshot = list(group)
        for right in snapshot:
            for product in (left @ right, right @ left):
                if any(np.allclose(product, existing, atol=tol, rtol=0) for existing in group):
                    continue
                group.append(product)
                if len(group) > limit:
                    raise ValueError(
                        f"Canonical spin point-group closure exceeded {limit} operations."
                    )
    return group


def _orthogonal_factor(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("Cannot project spin rotations using a singular point-group transform.")
    left, _singular_values, right_t = np.linalg.svd(matrix)
    return left @ right_t


def _match_projected_spin_rotation(raw_rotation, projected_rotations, spin_tol: float):
    raw_rotation = np.asarray(raw_rotation, dtype=float)
    distances = [
        float(np.max(np.abs(raw_rotation - projected_rotation)))
        for projected_rotation in projected_rotations
    ]
    best_index = int(np.argmin(distances))
    best_distance = distances[best_index]
    if best_distance > spin_tol:
        raise ValueError(
            "Projected spin rotation moved outside the tolerance implied by mtol "
            f"(max matrix residual={best_distance:.6g}, spin_matrix_tol={spin_tol:.6g})."
        )
    return projected_rotations[best_index], best_distance


def _finite_order_residual(matrix, *, max_order: int = 120):
    matrix = np.asarray(matrix, dtype=float)
    power = np.eye(3)
    best_order = 0
    best_residual = float("inf")
    for order in range(1, max_order + 1):
        power = power @ matrix
        residual = float(np.max(np.abs(power - np.eye(3))))
        if residual < best_residual:
            best_order = order
            best_residual = residual
    return best_order, best_residual


def _spin_rotations_are_clean_finite_group(rotations, *, clean_tol: float = 1e-6) -> bool:
    for rotation in rotations:
        _order, residual = _finite_order_residual(rotation)
        if residual > clean_tol:
            return False
    return True


def _project_ssg_spin_rotations_to_exact_point_group(
    ssg_ops,
    mag_atoms,
    group_tol=DEFAULT_TOL,
    *,
    label="matched SSG",
):
    tol = _normalize_candidate_tol(group_tol)
    spin_tol = _spin_matrix_projection_tol(mag_atoms, tol)
    raw_spin_rotations = [np.asarray(op[0], dtype=float) for op in ssg_ops]
    unique_spin_rotations = deduplicate_matrix_pairs(raw_spin_rotations, tol=spin_tol)
    if _spin_rotations_are_clean_finite_group(unique_spin_rotations):
        return list(ssg_ops)

    try:
        pg_symbol, _ops_info, transform, _generator_indices, sch_symbol = identify_point_group(
            unique_spin_rotations,
            _id=True,
            tol=spin_tol,
        )
    except Exception as exc:
        raise ValueError(
            f"{label} spin rotations cannot be identified as a finite point group "
            f"under spin_matrix_tol={spin_tol:.6g}: {exc}"
        ) from exc

    standard_generators = [
        np.asarray(generator, dtype=float)
        for generator in _load_standard_point_group_generators(pg_symbol, id=True)
    ]
    standard_rotations = _matrix_group_closure_exact(standard_generators)
    if len(standard_rotations) != len(unique_spin_rotations):
        raise ValueError(
            f"{label} canonical spin point-group order mismatch: "
            f"identified {len(unique_spin_rotations)} operations but standard "
            f"{pg_symbol!r} generated {len(standard_rotations)} operations."
        )

    spin_frame = _orthogonal_factor(transform)
    projected_rotations = [
        spin_frame @ standard_rotation @ spin_frame.T
        for standard_rotation in standard_rotations
    ]

    projected_ops = []
    max_projection_residual = 0.0
    operation_preserves = _build_magnetic_atom_preservation_checker(mag_atoms, tol)
    for raw_op in ssg_ops:
        projected_spin, projection_residual = _match_projected_spin_rotation(
            raw_op[0],
            projected_rotations,
            spin_tol,
        )
        max_projection_residual = max(max_projection_residual, projection_residual)
        projected_op = SpinSpaceGroupOperation(projected_spin, raw_op[1], raw_op[2])
        if not operation_preserves(projected_op.spin_rotation, projected_op.rotation, projected_op.translation):
            position_residual, moment_residual = _magnetic_action_residual(projected_op, mag_atoms, tol)
            raise ValueError(
                f"{label} projected spin operation does not preserve the magnetic structure "
                f"under mtol={tol.moment:.6g} "
                f"(max position residual={position_residual:.6g}, "
                f"max moment residual={moment_residual:.6g}, "
                f"max spin projection residual={max_projection_residual:.6g})."
            )
        projected_ops.append(projected_op)

    return projected_ops


def _spin_space_group_closure_failure(ops, tol: float, label: str) -> str | None:
    lookup = _SpinSpaceOperationLookup(ops, tol=tol)
    if not lookup.contains(SpinSpaceGroupOperation.identity()):
        return f"{label} has no identity operation"

    for index, op in enumerate(ops):
        inverse = op.inv()
        if not lookup.contains(inverse):
            return f"{label} operation {index} has no inverse"

    for left_index, left in enumerate(ops):
        for right_index, right in enumerate(ops):
            product = left @ right
            if not lookup.contains(product):
                return f"{label} is not closed under product {left_index}*{right_index}"
    return None


def _translation_same_mod_lattice(left, right, tol: float) -> bool:
    return _translation_close_mod_lattice(left, right, tol)


def _spatial_operation_same(left, right, tol: float) -> bool:
    return bool(
        _matrix_close(left[0], right[0], tol, rtol=0)
        and _translation_same_mod_lattice(left[1], right[1], tol)
    )


def _compose_spatial_operation(left, right):
    left_rotation, left_translation = left
    right_rotation, right_translation = right
    rotation = np.asarray(left_rotation, dtype=float) @ np.asarray(right_rotation, dtype=float)
    translation = (
        np.asarray(left_rotation, dtype=float) @ np.asarray(right_translation, dtype=float)
        + np.asarray(left_translation, dtype=float)
    )
    translation = np.mod(translation, 1.0)
    translation[np.isclose(translation, 1.0, atol=1e-8)] = 0.0
    return [rotation, translation]


def _invert_spatial_operation(op):
    rotation, translation = op
    inverse_rotation = np.linalg.inv(np.asarray(rotation, dtype=float))
    inverse_translation = np.mod(-inverse_rotation @ np.asarray(translation, dtype=float), 1.0)
    inverse_translation[np.isclose(inverse_translation, 1.0, atol=1e-8)] = 0.0
    return [inverse_rotation, inverse_translation]


def _spatial_group_closure_failure(ops, tol: float, label: str) -> str | None:
    lookup = _SpatialOperationLookup(ops, tol=tol)
    identity = [np.eye(3), np.zeros(3)]
    if not lookup.contains(identity):
        return f"{label} has no identity operation"

    for index, op in enumerate(ops):
        inverse = _invert_spatial_operation(op)
        if not lookup.contains(inverse):
            return f"{label} operation {index} has no inverse"

    for left_index, left in enumerate(ops):
        for right_index, right in enumerate(ops):
            product = _compose_spatial_operation(left, right)
            if not lookup.contains(product):
                return f"{label} is not closed under product {left_index}*{right_index}"
    return None


def _candidate_audit_tol_key(group_tol) -> tuple:
    if isinstance(group_tol, Tolerances):
        return (
            "tolerances",
            float(group_tol.space),
            float(group_tol.moment),
            float(group_tol.m_eig),
            float(group_tol.occupancy),
            float(group_tol.m_matrix_tol),
        )
    return ("scalar", float(group_tol))


def _candidate_audit_operation_key(op: SpinSpaceGroupOperation) -> tuple[bytes, bytes, bytes]:
    return (
        np.asarray(op.spin_rotation, dtype=np.float64).reshape(3, 3).tobytes(),
        np.asarray(op.rotation, dtype=np.float64).reshape(3, 3).tobytes(),
        np.asarray(op.translation, dtype=np.float64).reshape(3).tobytes(),
    )


def _candidate_audit_ops_key(ssg_ops) -> tuple[tuple[bytes, bytes, bytes], ...]:
    # Group audits are independent of operation enumeration order. Exact bytes
    # avoid merging numerically close candidates that still require separate
    # tolerance-sensitive validation.
    return tuple(sorted(_candidate_audit_operation_key(op) for op in ssg_ops))


def _restore_candidate_audit_tol(tol_key: tuple):
    if tol_key[0] == "scalar":
        return float(tol_key[1])
    return Tolerances(
        space=float(tol_key[1]),
        moment=float(tol_key[2]),
        m_eig=float(tol_key[3]),
        occupancy=float(tol_key[4]),
        m_matrix_tol=float(tol_key[5]),
    )


def _restore_candidate_audit_ops(ops_key):
    return [
        SpinSpaceGroupOperation(
            np.frombuffer(spin_bytes, dtype=np.float64).reshape(3, 3).copy(),
            np.frombuffer(real_bytes, dtype=np.float64).reshape(3, 3).copy(),
            np.frombuffer(translation_bytes, dtype=np.float64).reshape(3).copy(),
        )
        for spin_bytes, real_bytes, translation_bytes in ops_key
    ]


def _candidate_audit_failure_uncached(ssg_ops, group_tol=DEFAULT_TOL) -> str | None:
    if not ssg_ops:
        return "candidate produced no SSG operations"
    try:
        ssg = SpinSpaceGroup(ssg_ops, tol=group_tol)
        audit_tol = _candidate_audit_tol(group_tol)
        for ops, label in (
            (ssg.ops, "matched SSG"),
            (ssg.nssg, "nSSG"),
            (ssg.n_spin_translation_group, "spin translation group"),
        ):
            failure = _spin_space_group_closure_failure(ops, audit_tol, label)
            if failure is not None:
                return failure
        for ops, label in (
            (ssg.G0_ops, "G0"),
            (ssg.L0_ops, "L0"),
            (ssg.pure_t_group, "pure translation group"),
        ):
            failure = _spatial_group_closure_failure(ops, audit_tol, label)
            if failure is not None:
                return failure
        invariant_failure = _nsspg_invariant_failure(ssg)
        if invariant_failure is not None:
            return invariant_failure
        _ = ssg.spin_part_point_group_symbol_hm
        _ = ssg.n_spin_part_point_group_symbol_hm
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=128)
def _candidate_audit_failure_cached(ops_key: tuple, tol_key: tuple) -> str | None:
    return _candidate_audit_failure_uncached(
        _restore_candidate_audit_ops(ops_key),
        group_tol=_restore_candidate_audit_tol(tol_key),
    )


def _candidate_audit_failure(ssg_ops, group_tol=DEFAULT_TOL) -> str | None:
    ssg_ops = list(ssg_ops)
    if not ssg_ops:
        return "candidate produced no SSG operations"
    return _candidate_audit_failure_cached(
        _candidate_audit_ops_key(ssg_ops),
        _candidate_audit_tol_key(group_tol),
    )


def _format_candidate_audit_failures(profiles, limit=5):
    failures = []
    for profile in profiles[:limit]:
        candidate = profile["candidate"]
        subgroup_order = profile.get("pg_op_count")
        subgroup_label = "" if subgroup_order is None else f"/order={subgroup_order}"
        failures.append(
            f"{candidate['symbol']}{subgroup_label}@eig_tol={candidate['eig_tol']}: "
            f"{profile['audit_failure']}"
        )
    remaining = len(profiles) - len(failures)
    if remaining > 0:
        failures.append(f"... {remaining} more")
    return "; ".join(failures)


def _candidate_profile_sort_key(profile, meigtol):
    residual = profile.get("residual", {}).get("normalized", float("inf"))
    eig_tol = max(float(profile["candidate"].get("eig_tol", meigtol)), 1e-12)
    target_eig_tol = max(float(meigtol), 1e-12)
    eig_distance = abs(np.log10(eig_tol / target_eig_tol))
    return (
        -int(profile["compat"]),
        -int(profile["ssg_op_count"]),
        -int(profile["space_count"]),
        -int(profile["pg_op_count"]),
        float(residual),
        float(eig_distance),
        int(profile["candidate"].get("audit_rank", 0)),
        bool(profile["candidate"].get("is_floor_candidate", False)),
        profile["signature"],
    )


def _build_candidate_profile(
    candidate,
    pg_ops,
    space_operations_list,
    mag_atoms,
    group_tol,
    tol,
    configuration_details,
    preserve_cache: dict[tuple[int, ...], bool] | None = None,
):
    configuration = configuration_details["configuration"]
    raw_ssg_ops = get_ssg_ops(space_operations_list, pg_ops, mag_atoms, tol=tol)
    try:
        ssg_ops = _complete_ssg_ops_by_closure(
            raw_ssg_ops,
            mag_atoms,
            group_tol=group_tol,
            label=f"{candidate['symbol']} order-{len(pg_ops)} matched SSG",
            preserve_cache=preserve_cache,
        )
        ssg_ops = _project_ssg_spin_rotations_to_exact_point_group(
            ssg_ops,
            mag_atoms,
            group_tol=group_tol,
            label=f"{candidate['symbol']} order-{len(pg_ops)} matched SSG",
        )
        ssg_ops = _complete_ssg_ops_by_closure(
            ssg_ops,
            mag_atoms,
            group_tol=group_tol,
            label=f"{candidate['symbol']} order-{len(pg_ops)} projected matched SSG",
            preserve_cache=preserve_cache,
        )
        audit_failure = _candidate_audit_failure(ssg_ops, group_tol=group_tol)
    except ValueError as exc:
        subgroup_options = _maximal_audited_ssg_subgroups_from_generators(
            raw_ssg_ops,
            mag_atoms,
            group_tol=group_tol,
            label=f"{candidate['symbol']} order-{len(pg_ops)} matched SSG",
            preserve_cache=preserve_cache,
        )
        if subgroup_options:
            def subgroup_sort_key(ops):
                matched_spin_ops = _deduplicate_pg_operations(
                    [op[0] for op in ops],
                    tol=_candidate_audit_tol(group_tol),
                )
                compat = _configuration_compatibility(
                    candidate["symbol"],
                    configuration,
                    pg_operations=matched_spin_ops,
                    configuration_details=configuration_details,
                    tol=_candidate_audit_tol(group_tol),
                )
                return (
                    -int(compat),
                    -len(ops),
                    -_distinct_real_space_count(ops),
                    _ssg_signature(ops),
                )

            ssg_ops = sorted(subgroup_options, key=subgroup_sort_key)[0]
            audit_failure = None
        else:
            ssg_ops = raw_ssg_ops
            audit_failure = str(exc)
    residual = (
        _ssg_group_residual_details(ssg_ops, mag_atoms, tol)
        if ssg_ops
        else {"normalized": float("inf")}
    )
    matched_spin_ops = (
        _deduplicate_pg_operations([op[0] for op in ssg_ops], tol=_candidate_audit_tol(group_tol))
        if ssg_ops
        else []
    )
    return {
        "candidate": candidate,
        "ssg_ops": ssg_ops,
        "compat": _configuration_compatibility(
            candidate["symbol"],
            configuration,
            pg_operations=matched_spin_ops,
            configuration_details=configuration_details,
            tol=_candidate_audit_tol(group_tol),
        ),
        "space_count": _distinct_real_space_count(ssg_ops),
        "pg_op_count": len(pg_ops),
        "ssg_op_count": len(ssg_ops),
        "signature": _ssg_signature(ssg_ops),
        "audit_failure": audit_failure,
        "residual": residual,
    }


def _select_identify_pg_candidate(bundle, space_operations_list, mag_atoms, meigtol, group_tol=DEFAULT_TOL):
    group_tol = DEFAULT_TOL if group_tol is None else group_tol
    tol = _normalize_candidate_tol(group_tol)
    pg_subgroup_tol = _candidate_audit_tol(tol)

    profiles = []
    preserve_cache: dict[tuple[int, ...], bool] = {}
    for candidate in bundle["candidates"]:
        full_profile = _build_candidate_profile(
            candidate,
            candidate["operations"],
            space_operations_list,
            mag_atoms,
            group_tol,
            tol,
            bundle["configuration_details"],
            preserve_cache=preserve_cache,
        )
        profiles.append(full_profile)
        if full_profile["audit_failure"] is None:
            continue

        pg_subgroups = _point_group_subgroups_from_operations(candidate["operations"], pg_subgroup_tol)
        full_signature = _pg_operation_signature(candidate["operations"])
        for pg_ops in pg_subgroups:
            if _pg_operation_signature(pg_ops) == full_signature:
                continue
            profiles.append(
                _build_candidate_profile(
                    candidate,
                    pg_ops,
                    space_operations_list,
                    mag_atoms,
                    group_tol,
                    tol,
                    bundle["configuration_details"],
                    preserve_cache=preserve_cache,
                )
            )

    valid_profiles = [profile for profile in profiles if profile["audit_failure"] is None]
    if not valid_profiles:
        raise ValueError(
            "No PG candidate produced an audited SSG after matching with the spatial group: "
            f"{_format_candidate_audit_failures(profiles)}"
        )
    profiles = valid_profiles

    max_compat = max(profile["compat"] for profile in profiles)
    if max_compat <= 0:
        raise ValueError(
            "No audited SSG candidate is consistent with the magnetic "
            f"{bundle['configuration']} spin-rank semantics: "
            f"{_format_candidate_audit_failures(profiles)}"
        )
    profiles = [profile for profile in profiles if profile["compat"] == max_compat]

    distinct_profiles = []
    equivalence_tol = _candidate_audit_tol(group_tol)
    for profile in profiles:
        if any(
            _ssg_ops_equivalent(profile["ssg_ops"], existing["ssg_ops"], equivalence_tol)
            for existing in distinct_profiles
        ):
            continue
        distinct_profiles.append(profile)

    maximal_profiles = []
    for profile in distinct_profiles:
        is_strict_subgroup = any(
            other is not profile
            and _ssg_ops_subset(profile["ssg_ops"], other["ssg_ops"], equivalence_tol)
            for other in distinct_profiles
        )
        if not is_strict_subgroup:
            maximal_profiles.append(profile)
    distinct_profiles = maximal_profiles

    selected = sorted(distinct_profiles, key=lambda profile: _candidate_profile_sort_key(profile, meigtol))[0]
    return selected["candidate"], selected["ssg_ops"]

def get_pg(moments,atom_types,mtol,meigtol,matrix_tol=0.01):
    """
    Get the point group operations that leave the magnetic moments invariant.

    Parameters:
        moments: np.ndarray
            Array of magnetic moments.
        atom_types: np.ndarray
            Array of atom types.
        meigtol: float
            Tolerance for eigenvalue determination.

    Returns:
        pg_symbol: str
            The symbol of the identified point group.
        pg_operations: list of np.ndarray
            List of rotation matrices representing the point group operations.
    """


    bundle = _build_pg_candidates(moments, atom_types, mtol, meigtol, matrix_tol)
    selected = _select_get_pg_candidate(bundle, meigtol)
    return selected["symbol"], selected["operations"]



def identify_spin_space_group_result(
    default_cell,
    find_primitive=True,
    tol: Tolerances = DEFAULT_TOL,
) -> IdentifySpinSpaceGroupResult:
    """
    Identify the spin space group of a given magnetic structure.


    Returns:
        IdentifySpinSpaceGroupResult:
            Identification context including the primitive cell, spin space group,
            and input space-group metadata derived from the magnetic primitive cell.

    """
    if find_primitive == True:
        cell :CrystalCell = default_cell.get_primitive_structure(magnetic=True)[0]
    else:
        cell: CrystalCell = default_cell
    if cell.moments is None or not cell.magnetic_atom_indices:
        raise MagneticToleranceDegeneracyError(NONMAGNETIC_MTOL_ERROR)
    # get space operations
    p_dataset: SpglibDataset = gsd(cell.to_spglib(), symprec=tol.space)

    space_operations_list = list(zip(p_dataset.rotations, p_dataset.translations))
    # get point group operations for spin
    try:
        pg_bundle = _build_pg_candidates(
            cell.moments,
            cell.atom_types,
            tol.moment,
            tol.m_eig,
            tol.m_matrix_tol,
        )
        _selected_pg, ssg_ops = _select_identify_pg_candidate(
            pg_bundle,
            space_operations_list,
            [cell.atoms[i] for i in cell.magnetic_atom_indices],
            tol.m_eig,
            tol,
        )
    except ValueError as exc:
        if str(exc) in {
            "min() iterable argument is empty",
            "Wrong spin only groups. Check tolerance!",
            "Wrong number of co-set. Check tolerance!",
        }:
            raise ValueError(UNSTABLE_MTOL_ERROR) from exc
        raise

    ssg = SpinSpaceGroup(ssg_ops, tol=tol)
    try:
        ssg.validate_nsspg_invariants()
    except ValueError as exc:
        if str(exc) in {
            "cannot divide by zero",
            "Wrong spin only groups. Check tolerance!",
            "Wrong number of co-set. Check tolerance!",
        }:
            raise ValueError(UNSTABLE_MTOL_ERROR) from exc
        raise

    input_space_group = InputSpaceGroupInfo(
        number=int(p_dataset.number),
        symbol=str(p_dataset.international),
        basis_or_setting=getattr(p_dataset, "choice", None) or None,
    )
    return IdentifySpinSpaceGroupResult(
        primitive_cell=cell,
        ssg=ssg,
        input_space_group=input_space_group,
    )


def identify_spin_space_group(default_cell,find_primitive = True,tol:Tolerances=DEFAULT_TOL) -> SpinSpaceGroup:
    return identify_spin_space_group_result(
        default_cell,
        find_primitive=find_primitive,
        tol=tol,
    ).ssg
