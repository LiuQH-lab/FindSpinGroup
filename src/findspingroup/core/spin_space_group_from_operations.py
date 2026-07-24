"""Construct an identified spin space group directly from symmetry operations."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np

from findspingroup.core.identify_spin_space_group import _semantic_collinear_pg_ops
from findspingroup.core.tolerances import Tolerances
from findspingroup.structure.group import SpinSpaceGroup, SpinSpaceGroupOperation


_IDENTITY = np.eye(3)
_ZERO = np.zeros(3)
_CONFIGURATION_NAMES = {
    "collinear": "Collinear",
    "coplanar": "Coplanar",
    "noncoplanar": "Noncoplanar",
    "non-coplanar": "Noncoplanar",
}
_SPIN_FRAMES = {"oriented", "cartesian"}


def _operation_tolerance(tol: float | Tolerances) -> float:
    value = tol.m_matrix_tol if isinstance(tol, Tolerances) else float(tol)
    if not np.isfinite(value) or value <= 0 or value >= 0.5:
        raise ValueError("tol must be finite and satisfy 0 < tol < 0.5")
    return value


def _normalize_configuration(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("_", "-")
    try:
        return _CONFIGURATION_NAMES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_CONFIGURATION_NAMES.values())))
        raise ValueError(
            f"spin_configuration must be one of {allowed}, got {value!r}"
        ) from exc


def _normalize_spin_frame(value: str) -> str:
    frame = str(value).strip().lower()
    if frame not in _SPIN_FRAMES:
        allowed = ", ".join(sorted(_SPIN_FRAMES))
        raise ValueError(f"spin_frame must be one of {allowed}, got {value!r}")
    return frame


def _normalize_real_space_metric(metric) -> np.ndarray | None:
    if metric is None:
        return None
    normalized = np.asarray(metric, dtype=float)
    if normalized.shape != (3, 3) or not np.all(np.isfinite(normalized)):
        raise ValueError("real_space_metric must be a finite 3x3 matrix")
    normalized = 0.5 * (normalized + normalized.T)
    if np.min(np.linalg.eigvalsh(normalized)) <= 0:
        raise ValueError("real_space_metric must be positive definite")
    return normalized


def _normalize_real_space_lattice(lattice) -> np.ndarray | None:
    if lattice is None:
        return None
    normalized = np.asarray(lattice, dtype=float)
    if normalized.shape != (3, 3) or not np.all(np.isfinite(normalized)):
        raise ValueError("real_space_lattice must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(normalized))) <= 1e-12:
        raise ValueError("real_space_lattice must be nonsingular")
    return normalized


def _resolve_real_space_geometry(real_space_lattice, real_space_metric):
    lattice = _normalize_real_space_lattice(real_space_lattice)
    metric = _normalize_real_space_metric(real_space_metric)
    if lattice is None:
        return None, metric
    lattice_metric = lattice @ lattice.T
    if metric is not None and not np.allclose(
        metric,
        lattice_metric,
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError(
            "real_space_metric is inconsistent with real_space_lattice"
        )
    return lattice, lattice_metric


def _setting_to_cartesian_transform(
    real_space_lattice,
    real_space_metric,
) -> np.ndarray:
    """Map setting coordinates to the selected Cartesian spin frame."""
    lattice = _normalize_real_space_lattice(real_space_lattice)
    if lattice is not None:
        return lattice.T

    metric = _normalize_real_space_metric(real_space_metric)
    if metric is None:
        raise ValueError(
            "real_space_lattice or real_space_metric is required when "
            "spin_frame='cartesian'"
        )
    # The upper-triangular Cholesky factor places a along mx, b in the
    # mx-my plane with positive my, and mz = mx x my.
    return np.linalg.cholesky(metric).T


def _normalize_input_spin_frame(
    operations,
    *,
    spin_frame: str,
    spin_only_direction,
    real_space_lattice,
    real_space_metric,
    spin_metric,
):
    """Convert input spin data once to the oriented setting representation."""
    if spin_frame == "oriented":
        effective_spin_metric = (
            real_space_metric
            if spin_metric is None and real_space_metric is not None
            else spin_metric
        )
        return operations, spin_only_direction, effective_spin_metric

    setting_to_cartesian = _setting_to_cartesian_transform(
        real_space_lattice,
        real_space_metric,
    )
    cartesian_to_setting = np.linalg.inv(setting_to_cartesian)
    normalized_operations = [
        SpinSpaceGroupOperation(
            cartesian_to_setting
            @ operation.spin_rotation
            @ setting_to_cartesian,
            operation.rotation,
            operation.translation,
        )
        for operation in operations
    ]
    normalized_direction = None
    if spin_only_direction is not None:
        direction = np.asarray(spin_only_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError(
                "spin_only_direction must be a finite length-3 vector"
            )
        normalized_direction = cartesian_to_setting @ direction
    cartesian_metric = (
        np.eye(3)
        if spin_metric is None
        else np.asarray(spin_metric, dtype=float)
    )
    normalized_spin_metric = (
        setting_to_cartesian.T
        @ cartesian_metric
        @ setting_to_cartesian
    )
    return (
        normalized_operations,
        normalized_direction,
        normalized_spin_metric,
    )


def _normalize_direction(direction, *, metric, name: str) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    norm = float(np.sqrt(vector @ metric @ vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    vector = vector / norm
    for component in vector:
        if abs(component) <= 1e-12:
            continue
        if component < 0:
            vector = -vector
        break
    return vector


def _deserialize_operation(raw, *, index: int, tol: float) -> SpinSpaceGroupOperation:
    if isinstance(raw, SpinSpaceGroupOperation):
        spin_rotation = raw.spin_rotation
        real_rotation = raw.rotation
        translation = raw.translation
    elif isinstance(raw, dict):
        required = {"spin_rotation", "translation"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(
                f"operation {index} is missing fields: {', '.join(sorted(missing))}"
            )
        if "real_rotation" in raw:
            real_rotation = raw["real_rotation"]
        elif "rotation" in raw:
            real_rotation = raw["rotation"]
        else:
            raise ValueError(
                f"operation {index} is missing field: real_rotation"
            )
        spin_rotation = raw["spin_rotation"]
        translation = raw["translation"]
    elif (
        isinstance(raw, (Sequence, np.ndarray))
        and not isinstance(raw, (str, bytes))
        and len(raw) == 3
    ):
        spin_rotation, real_rotation, translation = raw
    else:
        raise TypeError(
            "operations must contain SpinSpaceGroupOperation objects, serialized "
            "dicts, or [spin_rotation, real_rotation, translation] sequences"
        )

    spin_rotation = np.asarray(spin_rotation, dtype=float)
    real_rotation = np.asarray(real_rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if spin_rotation.shape != (3, 3):
        raise ValueError(
            f"operation {index} spin_rotation must have shape (3, 3)"
        )
    if real_rotation.shape != (3, 3):
        raise ValueError(
            f"operation {index} real_rotation must have shape (3, 3)"
        )
    if translation.shape == (3, 1):
        translation = translation.reshape(3)
    if translation.shape != (3,):
        raise ValueError(
            f"operation {index} translation must have shape (3,)"
        )
    if not (
        np.all(np.isfinite(spin_rotation))
        and np.all(np.isfinite(real_rotation))
        and np.all(np.isfinite(translation))
    ):
        raise ValueError(f"operation {index} contains non-finite values")
    if np.any(translation < 0) or np.any(translation >= 1):
        raise ValueError(
            f"operation {index} translation must lie in [0, 1) componentwise"
        )

    spin_determinant = float(np.linalg.det(spin_rotation))
    if not np.isclose(abs(spin_determinant), 1.0, atol=tol, rtol=0):
        raise ValueError(
            f"operation {index} spin_rotation must be a finite metric-preserving "
            "matrix with determinant +1 or -1"
        )

    rounded_real_rotation = np.rint(real_rotation)
    if not np.allclose(real_rotation, rounded_real_rotation, atol=tol, rtol=0):
        raise ValueError(
            f"operation {index} real_rotation must be integral in the input cell basis"
        )
    real_determinant = float(np.linalg.det(rounded_real_rotation))
    if not np.isclose(abs(real_determinant), 1.0, atol=tol, rtol=0):
        raise ValueError(
            f"operation {index} real_rotation must be unimodular"
        )

    return SpinSpaceGroupOperation(
        spin_rotation,
        rounded_real_rotation,
        translation,
    )


def _translation_close(left, right, *, tol: float) -> bool:
    difference = np.mod(
        np.asarray(left, dtype=float) - np.asarray(right, dtype=float) + 0.5,
        1.0,
    ) - 0.5
    return bool(np.max(np.abs(difference)) <= tol)


def _operations_same(left, right, *, tol: float) -> bool:
    if np.max(
        np.abs(left.spin_rotation - right.spin_rotation)
    ) > tol:
        return False
    if np.max(np.abs(left.rotation - right.rotation)) > tol:
        return False
    return _translation_close(
        left.translation,
        right.translation,
        tol=tol,
    )


class _OperationLookup:
    """Tolerance-aware lookup keyed by exact lattice action and periodic bins."""

    def __init__(self, *, tol: float):
        self.tol = tol
        self.translation_bins = max(1, int(np.floor(1.0 / tol)))
        self.operations = []
        self._buckets = {}

    @staticmethod
    def _rotation_key(operation):
        return tuple(np.rint(operation.rotation).astype(int).reshape(-1))

    def _translation_key(self, translation):
        wrapped = np.mod(np.asarray(translation, dtype=float), 1.0)
        return tuple(
            np.floor(wrapped * self.translation_bins).astype(int)
            % self.translation_bins
        )

    def _candidate_keys(self, operation):
        rotation_key = self._rotation_key(operation)
        translation_key = self._translation_key(operation.translation)
        for offsets in product((-1, 0, 1), repeat=3):
            yield (
                rotation_key,
                tuple(
                    (component + offset) % self.translation_bins
                    for component, offset in zip(translation_key, offsets)
                ),
            )

    def contains(self, candidate) -> bool:
        for key in self._candidate_keys(candidate):
            for existing in self._buckets.get(key, ()):
                if _operations_same(candidate, existing, tol=self.tol):
                    return True
        return False

    def add(self, operation) -> bool:
        if self.contains(operation):
            return False
        self.operations.append(operation)
        key = (
            self._rotation_key(operation),
            self._translation_key(operation.translation),
        )
        self._buckets.setdefault(key, []).append(operation)
        return True


def _compose(left, right) -> SpinSpaceGroupOperation:
    translation = np.mod(
        left.rotation @ right.translation + left.translation,
        1.0,
    )
    translation[np.isclose(translation, 1.0, atol=1e-12, rtol=0)] = 0.0
    return SpinSpaceGroupOperation(
        left.spin_rotation @ right.spin_rotation,
        left.rotation @ right.rotation,
        translation,
    )


def _inverse(operation) -> SpinSpaceGroupOperation:
    real_inverse = np.linalg.inv(operation.rotation)
    spin_inverse = np.linalg.inv(operation.spin_rotation)
    translation = np.mod(-real_inverse @ operation.translation, 1.0)
    translation[np.isclose(translation, 1.0, atol=1e-12, rtol=0)] = 0.0
    return SpinSpaceGroupOperation(spin_inverse, real_inverse, translation)


def _close_operations(
    generators,
    *,
    tol: float,
    max_operations: int,
    canonicalize=None,
):
    if max_operations < 1:
        raise ValueError("max_operations must be positive")

    identity = SpinSpaceGroupOperation.identity()
    if canonicalize is not None:
        identity = canonicalize(identity)
    word_lookup = _OperationLookup(tol=tol)
    for generator in generators:
        for word in (generator, _inverse(generator)):
            word = canonicalize(word) if canonicalize is not None else word
            word_lookup.add(word)

    closure_lookup = _OperationLookup(tol=tol)
    closure_lookup.add(identity)
    queue_index = 0
    while queue_index < len(closure_lookup.operations):
        current = closure_lookup.operations[queue_index]
        queue_index += 1
        for word in word_lookup.operations:
            candidate = _compose(current, word)
            if canonicalize is not None:
                candidate = canonicalize(candidate)
            if not closure_lookup.add(candidate):
                continue
            if len(closure_lookup.operations) > max_operations:
                raise ValueError(
                    "operation closure exceeded max_operations; the inputs do not "
                    "define a finite spin space group under the active tolerance"
                )
    return closure_lookup.operations


def _close_operation_input(operations, *, tol: float, max_operations: int):
    """Close either a complete operation list or a generator subset efficiently."""
    input_lookup = _OperationLookup(tol=tol)
    for operation in operations:
        input_lookup.add(operation)

    selected_generators = []
    closure = [SpinSpaceGroupOperation.identity()]
    closure_lookup = _OperationLookup(tol=tol)
    closure_lookup.add(closure[0])
    for operation in operations:
        if closure_lookup.contains(operation):
            continue
        selected_generators.append(operation)
        closure = _close_operations(
            selected_generators,
            tol=tol,
            max_operations=max_operations,
        )
        closure_lookup = _OperationLookup(tol=tol)
        for closed_operation in closure:
            closure_lookup.add(closed_operation)
    if (
        len(input_lookup.operations) == len(closure_lookup.operations)
        and all(input_lookup.contains(operation) for operation in closure)
    ):
        # Preserve the caller's exact matrices when the supplied operations
        # already form the complete group. Reconstructing them as generator
        # words can accumulate avoidable floating-point noise.
        return input_lookup.operations
    # Matrix products of nonorthogonal finite-order generators accumulate
    # machine-epsilon noise. Stabilize only generated representatives; the
    # requested precision remains far below the public operation tolerance.
    return [
        SpinSpaceGroupOperation(
            np.round(operation.spin_rotation, decimals=12),
            np.rint(operation.rotation),
            np.mod(np.round(operation.translation, decimals=12), 1.0),
        )
        for operation in closure
    ]


def _is_spin_only(operation, *, tol: float) -> bool:
    return bool(
        np.allclose(operation.rotation, _IDENTITY, atol=tol, rtol=0)
        and _translation_close(operation.translation, _ZERO, tol=tol)
    )


def _fixed_subspace(spin_rotations, *, tol: float):
    constraints = np.vstack(
        [np.asarray(rotation, dtype=float) - _IDENTITY for rotation in spin_rotations]
    )
    _, singular_values, right_vectors = np.linalg.svd(constraints)
    threshold = max(tol, np.finfo(float).eps * max(constraints.shape))
    rank = int(np.count_nonzero(singular_values > threshold))
    return right_vectors[rank:].T


def _invariant_spin_metric(operations, *, tol: float, spin_metric=None):
    if spin_metric is None:
        metric = sum(
            operation.spin_rotation.T @ operation.spin_rotation
            for operation in operations
        ) / len(operations)
    else:
        metric = np.asarray(spin_metric, dtype=float)
        if metric.shape != (3, 3) or not np.all(np.isfinite(metric)):
            raise ValueError("spin_metric must be a finite 3x3 matrix")
        metric = 0.5 * (metric + metric.T)

    eigenvalues = np.linalg.eigvalsh(metric)
    if np.min(eigenvalues) <= tol:
        raise ValueError("spin_metric must be positive definite")
    metric = metric / float(np.linalg.det(metric)) ** (1.0 / 3.0)
    for operation in operations:
        transformed = operation.spin_rotation.T @ metric @ operation.spin_rotation
        if not np.allclose(transformed, metric, atol=max(tol, 1e-8), rtol=0):
            raise ValueError(
                "spin rotations do not preserve a common finite "
                "positive-definite metric"
            )
    return metric


def _infer_spin_only_semantics(operations, *, metric, tol: float):
    rotations = [
        operation.spin_rotation
        for operation in operations
        if _is_spin_only(operation, tol=tol)
    ]
    if not rotations:
        return "Noncoplanar", None
    if all(np.allclose(rotation, _IDENTITY, atol=tol, rtol=0) for rotation in rotations):
        return "Noncoplanar", None

    fixed_subspace = _fixed_subspace(rotations, tol=tol)
    dimension = fixed_subspace.shape[1]
    if dimension == 1:
        return "Collinear", _normalize_direction(
            fixed_subspace[:, 0],
            metric=metric,
            name="inferred collinear direction",
        )
    if dimension == 2:
        coordinate_normal = np.cross(
            fixed_subspace[:, 0],
            fixed_subspace[:, 1],
        )
        normal = np.linalg.solve(metric, coordinate_normal)
        return "Coplanar", _normalize_direction(
            normal,
            metric=metric,
            name="inferred coplanar plane normal",
        )
    if dimension == 3:
        return "Noncoplanar", None
    raise ValueError(
        "the finite spin-only operations have no common fixed spin subspace; "
        "provide a physically consistent spin-only subgroup"
    )


def _directions_parallel(left, right, *, metric, tol: float) -> bool:
    return bool(abs(float(left @ metric @ right)) >= 1.0 - tol)


def _resolve_spin_only_semantics(
    operations,
    *,
    requested_configuration: str | None,
    requested_direction,
    metric,
    tol: float,
):
    inferred_configuration, inferred_direction = _infer_spin_only_semantics(
        operations,
        metric=metric,
        tol=tol,
    )
    if requested_configuration is None:
        if requested_direction is not None:
            raise ValueError(
                "spin_only_direction requires an explicit spin_configuration"
            )
        return inferred_configuration, inferred_direction

    if (
        inferred_configuration != "Noncoplanar"
        and inferred_configuration != requested_configuration
    ):
        raise ValueError(
            f"spin_configuration={requested_configuration!r} conflicts with the "
            f"finite spin-only subgroup, which implies {inferred_configuration!r}"
        )
    if requested_configuration == "Noncoplanar":
        if requested_direction is not None:
            raise ValueError(
                "spin_only_direction is not defined for noncoplanar spin configurations"
            )
        if inferred_configuration != "Noncoplanar":
            raise ValueError(
                "spin_configuration='Noncoplanar' conflicts with a nontrivial "
                "finite spin-only subgroup"
            )
        return requested_configuration, None

    if requested_direction is None:
        if inferred_direction is None:
            raise ValueError(
                f"spin_only_direction is required for explicit "
                f"{requested_configuration.lower()} input when the operations "
                "do not determine it"
            )
        return requested_configuration, inferred_direction

    direction = _normalize_direction(
        requested_direction,
        metric=metric,
        name="spin_only_direction",
    )
    if inferred_direction is not None and not _directions_parallel(
        direction,
        inferred_direction,
        metric=metric,
        tol=max(tol, 1e-8),
    ):
        raise ValueError(
            "spin_only_direction conflicts with the direction implied by the "
            "finite spin-only subgroup"
        )
    return requested_configuration, direction


def _coplanar_mirror(normal, *, metric) -> np.ndarray:
    normal = np.asarray(normal, dtype=float).reshape(3)
    return _IDENTITY - 2.0 * np.outer(normal, metric @ normal)


def _metric_norm(vector, metric) -> float:
    vector = np.asarray(vector, dtype=float)
    return float(np.sqrt(vector @ metric @ vector))


def _quotient_canonicalizer(configuration: str, direction, *, metric, tol: float):
    if configuration == "Noncoplanar":
        return lambda operation: operation

    if configuration == "Collinear":
        axis = np.asarray(direction, dtype=float)

        def canonicalize(operation):
            mapped = operation.spin_rotation @ axis
            projection = float(mapped @ metric @ axis)
            residual = _metric_norm(mapped - projection * axis, metric)
            if residual > tol or not np.isclose(abs(projection), 1.0, atol=tol, rtol=0):
                raise ValueError(
                    "a spin rotation does not preserve the declared collinear axis"
                )
            spin_rotation = _IDENTITY if projection > 0 else -_IDENTITY
            return SpinSpaceGroupOperation(
                spin_rotation,
                operation.rotation,
                operation.translation,
            )

        return canonicalize

    normal = np.asarray(direction, dtype=float)
    mirror = _coplanar_mirror(normal, metric=metric)

    def canonicalize(operation):
        mapped = operation.spin_rotation @ normal
        projection = float(mapped @ metric @ normal)
        residual = _metric_norm(mapped - projection * normal, metric)
        if residual > tol or not np.isclose(abs(projection), 1.0, atol=tol, rtol=0):
            raise ValueError(
                "a spin rotation does not preserve the declared coplanar spin plane"
            )
        spin_rotation = operation.spin_rotation
        if np.linalg.det(spin_rotation) < 0:
            spin_rotation = spin_rotation @ mirror
        return SpinSpaceGroupOperation(
            spin_rotation,
            operation.rotation,
            operation.translation,
        )

    return canonicalize


def _metric_cartesian_transform(metric):
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    return np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _canonical_spin_only_rotations(configuration: str, direction, *, metric):
    if configuration == "Noncoplanar":
        return [_IDENTITY]
    if configuration == "Coplanar":
        return [_IDENTITY, _coplanar_mirror(direction, metric=metric)]

    to_cartesian = _metric_cartesian_transform(metric)
    from_cartesian = np.linalg.inv(to_cartesian)
    cartesian_axis = to_cartesian @ np.asarray(direction, dtype=float)
    cartesian_axis = cartesian_axis / np.linalg.norm(cartesian_axis)
    cartesian_ops = _semantic_collinear_pg_ops(
        cartesian_axis,
        include_axis_flip=False,
    )
    return [
        _IDENTITY,
        *[
            from_cartesian @ operation @ to_cartesian
            for operation in cartesian_ops
        ],
    ]


def _lift_spin_only_group(
    quotient_operations,
    *,
    configuration: str,
    direction,
    metric,
    tol: float,
):
    lifted = _OperationLookup(tol=tol)
    for spin_only_rotation in _canonical_spin_only_rotations(
        configuration,
        direction,
        metric=metric,
    ):
        for operation in quotient_operations:
            candidate = SpinSpaceGroupOperation(
                spin_only_rotation @ operation.spin_rotation,
                operation.rotation,
                operation.translation,
            )
            lifted.add(candidate)
    return lifted.operations


def _has_internal_spin_only_contract(
    operations,
    *,
    configuration: str,
    tol: float,
) -> bool:
    spin_only_order = sum(
        _is_spin_only(operation, tol=tol)
        for operation in operations
    )
    if configuration == "Noncoplanar":
        return spin_only_order == 1
    if configuration == "Coplanar":
        return spin_only_order == 2
    return spin_only_order in {4, 8}


def get_spin_space_group_from_operations(
    operations,
    *,
    spin_configuration: str | None = None,
    spin_only_direction=None,
    spin_frame: str = "cartesian",
    tol: float | Tolerances = 1e-6,
    identify_tol: float = 1e-3,
    max_operations: int = 4096,
    real_space_lattice=None,
    real_space_metric=None,
    spin_metric=None,
) -> SpinSpaceGroup:
    """Return an identified :class:`SpinSpaceGroup` from operations or generators.

    Examples
    --------
    See ``examples/spin_space_group_from_operations.py`` for executable
    coplanar Mn3Sn and collinear MnTe generator examples.

    ``operations`` may contain the complete group or a generator subset. Each
    entry may be a :class:`SpinSpaceGroupOperation`, a serialized dictionary,
    or a three-item sequence. Centering translations and spin translations
    that are implicit in a printed group symbol must be supplied explicitly
    when they are needed to generate the group.

    Input operations use ``[spin_rotation, real_rotation, translation]`` order.
    Translations must already be reduced componentwise to ``[0, 1)`` in the
    current cell basis. The finite invariant metric of the supplied spin
    matrices is derived automatically; ``spin_metric`` can provide it
    explicitly.

    ``spin_only_direction`` is the collinear axis or the coplanar plane normal.
    When no configuration is supplied, the finite spin-only subgroup is
    inferred. An identity-only spin-only subgroup is interpreted as
    noncoplanar by this operation-only API.

    No real-space cell convention is assumed. ``real_rotation`` and
    ``translation`` are interpreted in the current basis supplied by the
    caller. The frame default is ``spin_frame="cartesian"``.

    ``spin_frame="cartesian"`` means spin matrices and directions use the same
    Cartesian coordinates as the supplied lattice. It is the default and
    requires ``real_space_lattice`` or ``real_space_metric``.
    ``spin_frame="oriented"`` means spin coordinates already use the current
    real-space setting basis.
    The lattice and spin frame are jointly reduced to the default relative
    frame with x parallel to a, y in the ab plane, and z=x cross y. Supplying
    only ``real_space_metric`` selects that default relative frame. Supplying
    ``real_space_lattice`` with row vectors a, b, and c instead preserves its
    actual global Cartesian orientation. The returned group uses the oriented
    representation expected by downstream OSSG/MSG analysis. For example,
    ``ssg.index``, ``ssg.G0_num``, ``ssg.L0_num``, group symbols,
    translations, ACC/k-point information, and operation-derived MSG
    information remain available. Structure-derived atoms, Wyckoff data,
    magnetic phase, SCIF, and POSCAR require a structure-based FindSpinGroup
    route instead.

    For a collinear group, ``len(ssg.ops)`` is the order of FindSpinGroup's
    finite internal representative, not the order of the physical spin space
    group. The latter contains a continuous C-infinity-v spin-only component
    and therefore has infinitely many operations.
    """
    operation_tol = _operation_tolerance(tol)
    normalized_spin_frame = _normalize_spin_frame(spin_frame)
    (
        normalized_real_space_lattice,
        normalized_real_space_metric,
    ) = _resolve_real_space_geometry(
        real_space_lattice,
        real_space_metric,
    )
    if not np.isfinite(identify_tol) or identify_tol <= 0:
        raise ValueError("identify_tol must be a finite positive number")

    try:
        raw_operations = list(operations)
    except TypeError as exc:
        raise TypeError("operations must be an iterable of operations") from exc
    if not raw_operations:
        raise ValueError("operations must contain at least one operation or generator")

    parsed_operations = [
        _deserialize_operation(raw, index=index, tol=operation_tol)
        for index, raw in enumerate(raw_operations)
    ]
    (
        parsed_operations,
        spin_only_direction,
        spin_metric,
    ) = _normalize_input_spin_frame(
        parsed_operations,
        spin_frame=normalized_spin_frame,
        spin_only_direction=spin_only_direction,
        real_space_lattice=normalized_real_space_lattice,
        real_space_metric=normalized_real_space_metric,
        spin_metric=spin_metric,
    )
    requested_configuration = _normalize_configuration(spin_configuration)

    finite_closure = _close_operation_input(
        parsed_operations,
        tol=operation_tol,
        max_operations=max_operations,
    )
    metric = _invariant_spin_metric(
        finite_closure,
        tol=operation_tol,
        spin_metric=spin_metric,
    )
    configuration, direction = _resolve_spin_only_semantics(
        finite_closure,
        requested_configuration=requested_configuration,
        requested_direction=spin_only_direction,
        metric=metric,
        tol=operation_tol,
    )

    if _has_internal_spin_only_contract(
        finite_closure,
        configuration=configuration,
        tol=operation_tol,
    ):
        complete_operations = finite_closure
    else:
        canonicalize = _quotient_canonicalizer(
            configuration,
            direction,
            metric=metric,
            tol=operation_tol,
        )
        quotient_operations = _close_operations(
            finite_closure,
            tol=operation_tol,
            max_operations=max_operations,
            canonicalize=canonicalize,
        )
        complete_operations = _lift_spin_only_group(
            quotient_operations,
            configuration=configuration,
            direction=direction,
            metric=metric,
            tol=operation_tol,
        )
    if len(complete_operations) > max_operations:
        raise ValueError(
            "canonical spin-only completion exceeded max_operations"
        )

    group = SpinSpaceGroup(
        complete_operations,
        tol=operation_tol,
        real_space_metric=normalized_real_space_metric,
        identify_source_name="<operation input>",
        identify_tol=float(identify_tol),
    )
    group.input_spin_frame = normalized_spin_frame
    group.spin_settings = "oriented"
    group.relative_settings = "OSSG"
    group.index
    return group
