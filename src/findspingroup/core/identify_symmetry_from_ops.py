import copy
import itertools
import math
import re
from fractions import Fraction
from seekpath import get_path

import numpy as np

from spglib import SpglibDataset, get_symmetry_dataset, SpglibMagneticDataset, get_magnetic_symmetry_dataset
from findspingroup.data.MSGMPG_DB import MSG_INT_TO_BNS, BNS_TO_OG_NUM, OG_NUM_TO_MPG
from findspingroup.utils import SG_HALL_MAPPING
from findspingroup.utils.matrix_utils import normalize_vector_to_zero

np.set_printoptions(suppress=True)


def get_element_order(element):
    # element := 3*3 np.array

    errorblocker = 0
    order = 1
    temp = element

    while not np.allclose(temp, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), atol=0.1):
        # tolerance can be large
        temp = temp @ element
        order = order + 1
        errorblocker = errorblocker + 1
        if errorblocker > 61:
            raise ValueError('error cannot find the order')
    return order

def rotation_angle(R,axis,eigenvals):
    v1 = np.random.rand(3)
    axis = axis
    v1 -= v1.dot(axis) * axis  # get perpendicular vector v1
    v1 /= np.linalg.norm(v1)
    v2 = R @ v1
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(axis, cross))

    a = [val for val in eigenvals if val.imag > 0.01]
    angle = np.arccos(a[0].real)
    if sign < 0:
        angle = 2*3.14159265357-angle  # countwise rotation

    return angle

def times_of_rotation(rotation_angle, order_hint=None):
    """
        2*pi*m/n = rotation_angle
        m/n = rotation_angle/2pi
    """

    m_n = rotation_angle/(2*3.14159265357)
    if order_hint is not None:
        n = int(order_hint)
        if n < 1:
            raise ValueError("order_hint must be a positive integer")
        m = int(round(m_n * n)) % n
        if m == 0 and abs(m_n) > 1e-8:
            m = n
        return m, n

    f = Fraction(m_n).limit_denominator(120)
    m, n = f.numerator, f.denominator
    return m, n


def _hm_improper_display_order(op: np.ndarray, fallback_order: int) -> int:
    """
    Return the HM display order for an improper operation.

    The matrix order of an improper rotation can be larger than the HM symbol
    order. For example, a crystallographic `-3` operation has matrix order 6,
    but its HM token must still be `-3`. The HM order is determined by the
    proper rotation obtained after multiplying by inversion.
    """
    try:
        return int(get_element_order(-np.asarray(op, dtype=float)))
    except Exception:
        return int(fallback_order)


def costheta(v1, v2):
    """Return the cosine between two non-normalized vectors."""
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0
    return np.dot(v1, v2) / norm_product


def find_rotation(operations, rotation_times, axis, perp_axis=None, exclude=None, improper=False):

    """
    Find a rotation that satisfies the requested geometric constraints.

    Parameters:
    -----------
    operations : list
        Operation list in the form `[..., ..., ..., vector, (power, order), det]`.
    rotation_times : int
        Target rotation order.
    axis : array_like
        Target axis direction.
    perp_axis : array_like, optional
        Optional perpendicular reference axis used to determine handedness.
    exclude : array_like, optional
        Axis to exclude, usually an already fixed principal axis.
    improper : bool
        Whether to search for an improper rotation.

    Returns:
    --------
    (index, direction) : tuple
        The matched operation index and corrected direction vector.
    """

    # Geometric tolerances.
    TOL_PARALLEL = 1e-2
    TOL_CHIRALITY = 1e-4
    MIN_DEVIATION = 1.000001

    best_op_index = None
    best_direction = None
    best_rank = None

    # Target determinant: proper (+1) or improper (-1).
    target_det = -1 if improper else 1

    for i, op in enumerate(operations):
        # Unpack fields for readability.
        op_vec = np.array(op[3])
        op_rot_info = op[4]  # (power, order)
        op_det = op[5]

        # Basic property filters.
        if op_det != target_det:
            continue
        if op_rot_info is None or op_rot_info[1] != rotation_times:
            continue

        # Exclude axes parallel to the forbidden axis.
        if exclude is not None:
            if abs(abs(costheta(exclude, op_vec)) - 1) < TOL_PARALLEL:
                continue

        # Reject vectors that are too parallel to the perpendicular reference.
        if perp_axis is not None:
            if abs(costheta(perp_axis, op_vec)) > 0.8:
                continue

        # Measure alignment against the target axis.
        cos_val = costheta(op_vec, axis)

        current_deviation = abs(abs(cos_val) - 1)
        if current_deviation >= MIN_DEVIATION:
            continue

        # Consider both axis directions. Prefer the standard representative
        # with rotation-time 1; if it does not exist, fall back to the
        # smallest same-axis representative.
        candidates = []

        forward_step = op_rot_info[0]
        reverse_step = op_rot_info[1] - op_rot_info[0]
        candidates.append((op_vec, forward_step))
        # For order-2 operations, +v and -v are the same rotation-time but still
        # represent distinct axis directions in the rigid basis convention.
        candidates.append((-op_vec, reverse_step))

        for candidate_vec, effective_step in candidates:
            is_valid = True

            # Enforce a right-handed convention when both reference axes exist.
            if exclude is not None and perp_axis is not None:
                mixed_product = np.dot(perp_axis, np.cross(exclude, candidate_vec))

                if mixed_product < TOL_CHIRALITY:
                    is_valid = False

            elif perp_axis is not None:
                if costheta(perp_axis, candidate_vec) < -0.1:
                    is_valid = False

            elif exclude is not None:
                if costheta(exclude, candidate_vec) < 0.1:
                    is_valid = False

            if is_valid:
                final_cos = costheta(candidate_vec, axis)
                this_deviation = abs(final_cos - 1)
                step_rank = 0 if effective_step == 1 else 1
                candidate_rank = (step_rank, effective_step, this_deviation)

                # Update the current best candidate.
                if best_rank is None or candidate_rank < best_rank:
                    best_rank = candidate_rank
                    best_op_index = i
                    best_direction = candidate_vec

    if best_op_index is None:
        raise ValueError(f"No rotation found for order={rotation_times}, improper={improper}")

    return best_op_index, best_direction


def find_mirror(operations,axis,perp_axis=None,exclude=None,cubic=None):
    index = None
    distance = 1.001
    for i,op in enumerate(operations):
        if op[2] == 'm':
            if perp_axis is not None: # skip those not perp

                if abs(costheta(perp_axis,op[3])) > 0.9:
                    continue
            if exclude is not None:
                if abs(abs(costheta(exclude,op[3]))-1) < 1e-2:
                    continue
            if cubic is not None:
                if cubic is True and exclude is not None and abs(costheta(exclude,op[3])) < 0.65:
                    continue

                if cubic is not True and (abs(costheta(cubic,op[3])) > 0.65  or abs(costheta(cubic,op[3])) < 0.4): # exclude the nearest mirror of high-symmetry axis
                    continue
            temp_distance = costheta(op[3],axis)+0.001
            if abs(abs(temp_distance)-1)<=distance:
                if np.sign(temp_distance) < 0:
                    if exclude is not None and perp_axis is not None:
                        t = costheta(perp_axis,exclude)
                        tempc= costheta(perp_axis, np.cross(exclude, -op[3]))
                        if tempc < 0:
                            continue
                    if perp_axis is not None and costheta(perp_axis, -op[3]) < -0.1:  # for cubic system
                        continue
                    distance = abs(abs(temp_distance)-1)
                    index = i
                    direction = -op[3]
                else:
                    if exclude is not None and perp_axis is not None and abs(costheta(exclude, op[3])) < 1e-2 and costheta(perp_axis,np.cross(exclude,op[3])) > 0:
                        distance = abs(abs(temp_distance) - 1)
                        index = i
                        direction = op[3]
                        continue
                    if exclude is not None and perp_axis is not None and abs(costheta(exclude, op[3])) < 1e-2 and costheta(perp_axis,np.cross(exclude,-op[3])) > 0:
                        distance = abs(abs(temp_distance) - 1)
                        index = i
                        direction = -op[3]
                        continue
                    if exclude is not None and perp_axis is not None and costheta(perp_axis, np.cross(exclude, op[3])) < 0:
                        continue
                    if perp_axis is not None and costheta(perp_axis, op[3]) < -0.1:  # for cubic system
                        continue
                    distance = abs(abs(temp_distance)-1)
                    index = i
                    direction = op[3]
            if cubic is not None and cubic is not True and index == i:
                if costheta(cubic,op[3]) < 0:
                    direction = -op[3]
                else:
                    direction = op[3]
    if index is None:
        raise ValueError("No mirror found")
    return index, direction


def reverse_direction(operations, direction):
    """
        reverse direction
        operations : op_order_type_direction_addition_det
    """
    for op in operations:
        if op[3] is not None:
            if abs(np.dot(op[3],direction) + 1) < 1e-2: #  opposite direction
                op[3] = -op[3] # reverse direction
                if op[4] != None: # only for rotations
                    op[4] = [op[4][1]-op[4][0],op[4][1]] # [n-m,n]
    return operations


def find_operation_index_by_matrix(operations, target_matrix, tol=1e-2):
    for i, op in enumerate(operations):
        if np.allclose(op[0], target_matrix, atol=tol, rtol=tol):
            return i
    raise ValueError("No operation matches the canonical generator matrix.")


def classify_point_group_operations(point_group_matrices, tol=1e-2):
    """
    Classify raw point-group matrices into the enriched operation records used by
    ``identify_point_group``.

    This is intentionally exposed as a diagnostic helper so we can inspect which
    order/type/axis assignments are being made before the later symbolic
    standardization logic branches on them.
    """
    op_order_type_direction_addition_det = []
    for op in point_group_matrices:
        order = get_element_order(op)

        if order == 1:  # map 1
            op_order_type_direction_addition_det.append([op, order, '1', None, None, 1])

        eigvals, eigvecs = np.linalg.eig(op.astype(np.float64))
        eigvecs = eigvecs.T

        if order > 2:  # map n or -n
            for i, val in enumerate(eigvals):
                if abs(val.imag) < tol:  # find the direction
                    if abs(val.real - 1) < tol:  # n
                        axis = eigvecs[i].real
                        angle = rotation_angle(op, axis, eigvals)
                        m, n = times_of_rotation(angle, order_hint=order)
                        op_order_type_direction_addition_det.append(
                            [op, order, str(n), axis, [m, n], 1]
                        )
                        break

                    if abs(val.real + 1) < tol:  # -n
                        axis = eigvecs[i].real
                        angle = rotation_angle(-op, axis, -eigvals)
                        m, n = times_of_rotation(angle, order_hint=order)
                        display_n = _hm_improper_display_order(op, n)
                        op_order_type_direction_addition_det.append(
                            [op, order, str(-display_n), axis, [m, n], -1]
                        )
                        break
                if i == 2:
                    raise ValueError('can not find eigenvector for rotation, try another tolerance!')

        if order == 2:  # -1, m, 2
            if abs(sum(eigvals) + 3) < tol:  # -1
                op_order_type_direction_addition_det.append([op, order, '-1', None, None, -1])

            if abs(sum(eigvals) + 1) < tol:  # 2
                for i, val in enumerate(eigvals):
                    if abs(val - 1) < tol:  # find direction
                        axis = eigvecs[i]
                        break
                    if i == 2:
                        raise ValueError('can not find eigenvector for rotation 2, try another tolerance!')
                op_order_type_direction_addition_det.append([op, order, '2', axis, [1, 2], 1])

            if abs(sum(eigvals) - 1) < tol:  # m
                for i, val in enumerate(eigvals):
                    if abs(val + 1) < tol:  # find norm vector
                        axis = eigvecs[i]
                        break
                    if i == 2:
                        raise ValueError('can not find norm vector for mirror, try another tolerance!')
                op_order_type_direction_addition_det.append([op, order, 'm', axis, None, -1])
    return op_order_type_direction_addition_det


def _load_standard_point_group_generators(group_symbol, *, id=False):
    if id:
        from findspingroup.data.POINT_GROUP_MATRIX import point_group_generators_cartesian as pg_gens
    else:
        from findspingroup.data.POINT_GROUP_MATRIX import point_group_generators as pg_gens

    if group_symbol not in pg_gens:
        return generate_non_crystallographic_point_groups(group_symbol)
    return [np.array(op) for op in pg_gens[group_symbol]]


def _build_transition_matrix_linear_system(matrices_list, standard_list):
    I = np.eye(3)
    a_blocks = []
    for matrix, standard in zip(matrices_list, standard_list):
        a_blocks.append(np.kron(I, matrix) - np.kron(standard.T, I))
    return np.vstack(a_blocks)


def _axis_line_misalignment(axis, target_axis):
    axis = np.asarray(axis, dtype=np.complex128).real
    target_axis = np.asarray(target_axis, dtype=np.complex128).real
    axis = axis / np.linalg.norm(axis)
    target_axis = target_axis / np.linalg.norm(target_axis)
    return 1.0 - abs(costheta(axis, target_axis))


def _transition_combo_quality(matrices_list, standard_list):
    linear_system = _build_transition_matrix_linear_system(matrices_list, standard_list)
    _, singular_values, _ = np.linalg.svd(linear_system)
    return float(singular_values[-1]), singular_values


def _try_td_generator_candidates(operations, *, id=False, tol=1e-2):
    standard_list = _load_standard_point_group_generators('-43m', id=id)
    standard_ops = classify_point_group_operations(standard_list, tol=tol)
    standard_by_symbol = {op[2]: op for op in standard_ops}

    minus4_candidates = [(idx, op) for idx, op in enumerate(operations) if op[2] == '-4']
    three_candidates = [(idx, op) for idx, op in enumerate(operations) if op[2] == '3']
    mirror_candidates = [(idx, op) for idx, op in enumerate(operations) if op[2] == 'm']

    if not minus4_candidates or not three_candidates or not mirror_candidates:
        raise ValueError("Missing one of the required Td generator families (-4, 3, m).")

    target_minus4 = standard_by_symbol['-4'][3]
    target_three = standard_by_symbol['3'][3]
    target_mirror = standard_by_symbol['m'][3]

    def axis_rank(item, target_axis):
        _, op = item
        return _axis_line_misalignment(op[3], target_axis)

    # First try the user's intended canonical directions directly.
    preferred = [
        min(minus4_candidates, key=lambda item: axis_rank(item, target_minus4)),
        min(three_candidates, key=lambda item: axis_rank(item, target_three)),
        min(mirror_candidates, key=lambda item: axis_rank(item, target_mirror)),
    ]
    preferred_indices = [item[0] for item in preferred]
    preferred_generators = [item[1][0] for item in preferred]
    try:
        find_transition_matrix_deterministic(preferred_generators, '-43m', id=id, tol=tol)
        return preferred_generators, preferred_indices
    except ValueError:
        pass

    ranked = []
    for minus4_index, minus4_op in minus4_candidates:
        for three_index, three_op in three_candidates:
            for mirror_index, mirror_op in mirror_candidates:
                generators = [minus4_op[0], three_op[0], mirror_op[0]]
                min_sv, singular_values = _transition_combo_quality(generators, standard_list)
                ranked.append(
                    (
                        min_sv,
                        axis_rank((minus4_index, minus4_op), target_minus4)
                        + axis_rank((three_index, three_op), target_three)
                        + axis_rank((mirror_index, mirror_op), target_mirror),
                        (minus4_index, three_index, mirror_index),
                        generators,
                    )
                )

    ranked.sort(key=lambda item: (item[0], item[1]))
    last_error = None
    for _min_sv, _axis_score, indices, generators in ranked:
        try:
            find_transition_matrix_deterministic(generators, '-43m', id=id, tol=tol)
            return generators, list(indices)
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("No valid Td generator combination found.")


def _transition_nullspace_threshold(singular_values, tol):
    max_sv = float(np.max(singular_values)) if len(singular_values) else 0.0
    return max(float(tol) * 1e-4, max_sv * 1e-8, 1e-12)


def _expected_transition_nullspace_dimension(standard_list):
    """
    Estimate the exact null-space dimension from the standard generators alone.

    If ``M_i = P D_i P^{-1}`` exactly, then every valid solution has the form
    ``P C`` where ``C`` commutes with the standard generators ``D_i``. The
    dimension of that commutant is therefore the exact null-space dimension of
    the transition system in the noise-free case, and gives a principled lower
    bound for near-null admission under small perturbations.
    """
    standard_system = _build_transition_matrix_linear_system(standard_list, standard_list)
    _, singular_values, _ = np.linalg.svd(standard_system)
    max_sv = float(np.max(singular_values)) if len(singular_values) else 0.0
    exact_tol = max(max_sv * 1e-10, 1e-12)
    return max(1, int(np.sum(singular_values < exact_tol)))


def _expand_transition_nullspace_dimension_via_gap(
    singular_values,
    current_dim,
    *,
    tol,
):
    """
    Recover near-null clusters that the strict relative threshold can miss.

    Some roundtripped low-order groups (for example near-``mm2`` cases) produce
    a tail like ``[..., 2e-6, 2e-6, 1e-16]``.  The current base threshold sees
    only the final singular value as null, which leaves a 1D null space whose
    basis matrix is singular.  If that tail is clearly separated from the bulk
    by a large spectral gap, we treat the whole tail as an effective null-space
    cluster.
    """
    if len(singular_values) < 2:
        return current_dim

    expanded_dim = current_dim
    gap_ratio_required = 1e4
    tail_value_limit = max(float(tol) * 1e-3, 1e-5)

    for i in range(len(singular_values) - 1):
        upper = float(singular_values[i])
        lower = float(singular_values[i + 1])
        tail_dim = len(singular_values) - i - 1
        if tail_dim <= expanded_dim:
            continue
        if lower > tail_value_limit:
            continue
        ratio = upper / max(lower, 1e-300)
        if ratio < gap_ratio_required:
            continue
        expanded_dim = tail_dim

    return expanded_dim


def _transition_dimension_candidates(
    singular_values,
    *,
    initial_dim,
    gap_dim,
    expected_dim,
):
    candidates = []

    def add(dim):
        if 1 <= dim <= len(singular_values) and dim not in candidates:
            candidates.append(dim)

    add(initial_dim)
    add(gap_dim)
    add(expected_dim)
    add(expected_dim + 1)
    add(expected_dim + 2)
    add(min(len(singular_values), expected_dim + 3))
    return candidates


def _transition_residual_tolerances(
    singular_values,
    *,
    dim,
    sv_tol,
):
    admitted = float(singular_values[-dim])
    candidates = [max(sv_tol * 10.0, 1e-8)]
    candidates.extend(max(admitted * factor, 1e-8) for factor in (1.0, 2.0, 5.0, 10.0))

    ordered = []
    seen = set()
    for value in candidates:
        key = round(float(value), 15)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(float(value))
    return ordered


def _search_transition_candidate_progressive(
    singular_values,
    vh,
    linear_system,
    *,
    sv_tol,
    initial_dim,
    gap_dim,
    expected_dim,
):
    best_candidate = None
    best_meta = None
    attempts = []

    for dim in _transition_dimension_candidates(
        singular_values,
        initial_dim=initial_dim,
        gap_dim=gap_dim,
        expected_dim=expected_dim,
    ):
        basis_matrices = [vector.reshape(3, 3) for vector in vh[-dim:, :]]
        basis_metrics = [_transition_basis_metrics(matrix) for matrix in basis_matrices]
        for residual_tol in _transition_residual_tolerances(
            singular_values,
            dim=dim,
            sv_tol=sv_tol,
        ):
            candidate = _search_transition_candidate(
                basis_matrices,
                linear_system,
                residual_tol=residual_tol,
            )
            attempts.append(
                {
                    "dim": int(dim),
                    "residual_tol": float(residual_tol),
                    "basis_metrics": basis_metrics,
                    "candidate": candidate,
                }
            )
            if candidate is None:
                continue
            if not candidate["passes_residual_tol"] or candidate["sigma_min"] < 1e-8:
                continue
            meta = (
                int(dim),
                float(residual_tol),
                *_candidate_sort_key(candidate),
            )
            if best_candidate is None or meta < best_meta:
                best_candidate = candidate
                best_meta = meta
        if best_candidate is not None:
            break

    return best_candidate, attempts


def _transition_basis_metrics(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_min = float(singular_values[-1])
    sigma_max = float(singular_values[0])
    cond = math.inf if sigma_min < 1e-15 else float(sigma_max / sigma_min)
    return {
        "det": float(np.linalg.det(matrix)),
        "fro_norm": float(np.linalg.norm(matrix)),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": cond,
    }


def _score_transition_candidate(matrix, linear_system):
    fro_norm = float(np.linalg.norm(matrix))
    if fro_norm < 1e-15:
        return None
    normalized = np.asarray(matrix, dtype=float) / fro_norm
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    sigma_min = float(singular_values[-1])
    sigma_max = float(singular_values[0])
    cond = math.inf if sigma_min < 1e-15 else float(sigma_max / sigma_min)
    residual = float(np.linalg.norm(linear_system @ normalized.reshape(-1)))
    return {
        "matrix": normalized,
        "residual": residual,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": cond,
        "det": float(abs(np.linalg.det(normalized))),
    }


def _candidate_sort_key(metrics):
    return (
        -metrics["sigma_min"],
        metrics["residual"],
        metrics["condition_number"],
        -metrics["det"],
    )


def _search_transition_candidate(
    basis_matrices,
    linear_system,
    *,
    residual_tol,
):
    best_candidate = None
    best_key = None

    def consider_candidate(candidate_matrix):
        nonlocal best_candidate, best_key
        metrics = _score_transition_candidate(candidate_matrix, linear_system)
        if metrics is None:
            return
        metrics["passes_residual_tol"] = metrics["residual"] <= residual_tol
        key = (
            not metrics["passes_residual_tol"],
            *_candidate_sort_key(metrics),
        )
        if best_candidate is None or key < best_key:
            best_candidate = metrics
            best_key = key

    for coeffs in _iter_transition_coefficients(len(basis_matrices)):
        candidate = np.zeros((3, 3), dtype=float)
        for coeff, basis in zip(coeffs, basis_matrices):
            candidate += coeff * basis
        consider_candidate(candidate)

    if best_candidate is None or not best_candidate["passes_residual_tol"] or best_candidate["sigma_min"] < 1e-8:
        rng = np.random.default_rng(0)
        for _ in range(max(64, 16 * len(basis_matrices))):
            coeffs = rng.normal(size=len(basis_matrices))
            candidate = np.zeros((3, 3), dtype=float)
            for coeff, basis in zip(coeffs, basis_matrices):
                candidate += coeff * basis
            consider_candidate(candidate)

    return best_candidate


def _iter_transition_coefficients(dim):
    base_scales = (1.0, -1.0, 0.5, -0.5, 2.0, -2.0)

    seen = set()

    def emit(coeffs):
        key = tuple(float(value) for value in coeffs)
        if all(abs(value) < 1e-12 for value in key):
            return
        if key in seen:
            return
        seen.add(key)
        return np.asarray(key, dtype=float)

    for i in range(dim):
        for scale in base_scales:
            coeffs = np.zeros(dim)
            coeffs[i] = scale
            candidate = emit(coeffs)
            if candidate is not None:
                yield candidate

    for i, j in itertools.combinations(range(dim), 2):
        for scale_i in base_scales:
            for scale_j in base_scales:
                coeffs = np.zeros(dim)
                coeffs[i] = scale_i
                coeffs[j] = scale_j
                candidate = emit(coeffs)
                if candidate is not None:
                    yield candidate

    deterministic_templates = [
        [1.0, math.sqrt(2.0), math.sqrt(3.0), math.pi, math.e, (1.0 + math.sqrt(5.0)) / 2.0],
        [1.0, -math.sqrt(2.0), math.sqrt(3.0), -math.pi, math.e, -(1.0 + math.sqrt(5.0)) / 2.0],
        [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
        [1.0, -2.0, 3.0, -5.0, 7.0, -11.0],
    ]
    for template in deterministic_templates:
        coeffs = np.array([template[i % len(template)] for i in range(dim)], dtype=float)
        candidate = emit(coeffs)
        if candidate is not None:
            yield candidate
        candidate = emit(-coeffs)
        if candidate is not None:
            yield candidate

    if dim <= 5:
        exhaustive_values = (1.0, -1.0, 0.5, -0.5, 2.0)
        for coeffs in itertools.product(exhaustive_values, repeat=dim):
            candidate = emit(coeffs)
            if candidate is not None:
                yield candidate


def analyze_transition_matrix_problem(matrices_list, group_symbol, id=False, tol=1e-2):
    """
    Diagnose the conjugacy problem solved by ``find_transition_matrix_deterministic``.

    Returns a dictionary containing the singular-value spectrum of the linear
    system, inferred null-space dimension, basis metrics, and the best
    deterministic candidate found by the current candidate search strategy.
    """
    standard_list = _load_standard_point_group_generators(group_symbol, id=id)
    if len(matrices_list) != len(standard_list):
        raise ValueError(
            f"Matrix count ({len(matrices_list)}) does not match the number of standard generators ({len(standard_list)})."
        )

    linear_system = _build_transition_matrix_linear_system(matrices_list, standard_list)
    _, singular_values, vh = np.linalg.svd(linear_system)
    sv_tol = _transition_nullspace_threshold(singular_values, tol)
    initial_null_space_dim = int(np.sum(singular_values < sv_tol))
    if initial_null_space_dim == 0:
        if singular_values[-1] < max(float(tol), 1e-8):
            initial_null_space_dim = 1
        else:
            raise ValueError("No null space found; the input matrices may not form the requested point group.")
    null_space_dim = _expand_transition_nullspace_dimension_via_gap(
        singular_values,
        initial_null_space_dim,
        tol=tol,
    )
    expected_null_space_dim = _expected_transition_nullspace_dimension(standard_list)
    best_candidate, attempts = _search_transition_candidate_progressive(
        singular_values,
        vh,
        linear_system,
        sv_tol=sv_tol,
        initial_dim=initial_null_space_dim,
        gap_dim=null_space_dim,
        expected_dim=expected_null_space_dim,
    )

    return {
        "group_symbol": group_symbol,
        "input_count": len(matrices_list),
        "standard_count": len(standard_list),
        "singular_values": [float(value) for value in singular_values],
        "null_space_tolerance": float(sv_tol),
        "initial_null_space_dimension": initial_null_space_dim,
        "null_space_dimension": null_space_dim,
        "expected_null_space_dimension": expected_null_space_dim,
        "dimension_candidates": _transition_dimension_candidates(
            singular_values,
            initial_dim=initial_null_space_dim,
            gap_dim=null_space_dim,
            expected_dim=expected_null_space_dim,
        ),
        "basis_metrics": []
        if not attempts
        else attempts[0]["basis_metrics"],
        "attempts": [
            {
                "dim": int(attempt["dim"]),
                "residual_tol": float(attempt["residual_tol"]),
                "basis_metrics": attempt["basis_metrics"],
                "candidate": None
                if attempt["candidate"] is None
                else {
                    "residual": float(attempt["candidate"]["residual"]),
                    "sigma_min": float(attempt["candidate"]["sigma_min"]),
                    "sigma_max": float(attempt["candidate"]["sigma_max"]),
                    "condition_number": float(attempt["candidate"]["condition_number"])
                    if math.isfinite(attempt["candidate"]["condition_number"])
                    else math.inf,
                    "det": float(attempt["candidate"]["det"]),
                    "passes_residual_tol": bool(attempt["candidate"]["passes_residual_tol"]),
                },
            }
            for attempt in attempts
        ],
        "best_candidate": None
        if best_candidate is None
        else {
            "residual": float(best_candidate["residual"]),
            "sigma_min": float(best_candidate["sigma_min"]),
            "sigma_max": float(best_candidate["sigma_max"]),
            "condition_number": float(best_candidate["condition_number"])
            if math.isfinite(best_candidate["condition_number"])
            else math.inf,
            "det": float(best_candidate["det"]),
            "passes_residual_tol": bool(best_candidate["passes_residual_tol"]),
        },
    }


def analyze_point_group_identification(point_group_matrices, _id=False, tol=1e-2):
    """
    Diagnostic helper for the early classification phase inside
    ``identify_point_group``.
    """
    operations = classify_point_group_operations(point_group_matrices, tol=tol)
    order_group = len(operations)
    metric_matrix = np.array(
        sum([op[0].T @ op[0] for op in operations]) / order_group,
        dtype=np.float64,
    )
    summary = {
        "operations_count": order_group,
        "classified_operations": operations,
        "metric_matrix": metric_matrix.tolist(),
        "needs_metric_basis_change": not np.allclose(metric_matrix, np.eye(3), rtol=1e-2),
        "max_order": max(op[1] for op in operations),
        "mirror_count": sum(op[2] == "m" for op in operations),
        "has_inversion": any(op[2] == "-1" for op in operations),
        "has_improper": any(op[5] == -1 for op in operations),
    }
    try:
        symbol, _, transformation, generator_indices, symbol_s = identify_point_group(
            point_group_matrices,
            _id=_id,
            tol=tol,
        )
        summary.update(
            {
                "identified_hm_symbol": symbol,
                "identified_s_symbol": symbol_s,
                "transformation": np.asarray(transformation, dtype=float).tolist(),
                "generator_indices": list(generator_indices),
            }
        )
    except Exception as exc:
        summary["identify_error"] = str(exc)
    return summary


def identify_point_group(point_group_matrices,_id= False,tol=1e-2):
    """

        input : point_group_matrices [op,...]

        return symbol_HM, op_symbols(type + direction), transformation matrix, generators_index, symbol_S
    """

    # Step 1 : determination of ops

    op_order_type_direction_addition_det = classify_point_group_operations(
        point_group_matrices,
        tol=tol,
    )


    # Step 2 : determination of point group
    # Step 3 : transformation matrix

    order_group = len(op_order_type_direction_addition_det)


    # change basis for metric
    metric_matrix_G = np.array(sum([op[0].T@op[0] for op in op_order_type_direction_addition_det])/order_group,dtype=np.float64)
    # metric_matrix_G = recover_metric_positive_definite_relaxed([op[0] for op in op_order_type_direction_addition_det])
    # use g = 1/|G| sum(R^T@ I @R) to recover metric
    if np.allclose(metric_matrix_G, np.eye(3), rtol=1e-2):
        P1 = np.eye(3)
        P1_inv = np.eye(3)
        operations = copy.deepcopy(op_order_type_direction_addition_det)
    else:
        P1_inv = np.linalg.cholesky(metric_matrix_G).T
        P1 = np.linalg.inv(P1_inv)
        operations = copy.deepcopy(op_order_type_direction_addition_det)
        for op in operations:
            if op[3] is not None:
                op[0] = P1_inv @ op[0] @ P1
                op[3] = P1_inv @ op[3]



    if order_group == 1:
        group_symbol = '1'
        group_symbol_S = 'C1'
        return '1',op_order_type_direction_addition_det,np.eye(3),[0],group_symbol_S


    counter_high_order_axis = []
    mirror = 0
    minues = False
    max_order = 1
    rotation2_axis = []
    improper = False

    for i in operations:
        if improper:
            pass
        else:
            if i[5] == -1:
                improper = True
        if i[1] > 2: # order > 2
            if counter_high_order_axis == []: # initialization
                counter_high_order_axis.append(i[3])
            else:
                if not any([abs(abs(costheta(i[3], _ ))-1) < tol for _ in counter_high_order_axis]):  # different axis for high order
                    counter_high_order_axis.append(i[3])
        if i[2] == '2':
            rotation2_axis.append(i[3])
        if i[2] == 'm':
            mirror = mirror + 1
        if i[2] == '-1':
            minues = True
        max_order = max(max_order, i[1])



    if len(counter_high_order_axis) > 1: #23 m-3 432 -43m m-3m I Ih
        if order_group == 12: # T
            group_symbol = '23'
            group_symbol_S = 'T'
            r1_index, direction = find_rotation(operations,
                                                      2, np.array([1, 0, 0]))
            operations = reverse_direction(operations,
                                                                     direction)
            r2_index, r2_direction = find_rotation(operations,
                                                      3, np.array([1, 1, 1]),direction)
            operations = reverse_direction(operations,
                                                                     r2_direction)
            generators = [operations[r1_index][0],operations[r2_index][0]]  # matrix
            generators_index = [r1_index,r2_index]
        if order_group == 24: #-43m 432 m-3
            if minues: # Th
                group_symbol = 'm-3'
                group_symbol_S = 'Th'

                ir_index, ir_direction = find_rotation(operations,
                                                       6, np.array([1, 1, 1]), improper=True)
                operations = reverse_direction(operations, ir_direction)

                mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]), cubic=ir_direction)
                operations = reverse_direction(operations, m_direction)
                generators = [operations[mirror_index][0],operations[ir_index][0]]  # matrix
                generators_index = [mirror_index,ir_index]
            else:
                if mirror > 0: # Td
                    group_symbol = '-43m'
                    group_symbol_S = 'Td'
                    generators, generators_index = _try_td_generator_candidates(
                        operations,
                        id=_id,
                        tol=tol,
                    )


                else:  # O
                    group_symbol = '432'
                    group_symbol_S = 'O'
                    r_index, r_direction = find_rotation(operations,
                                                           4, np.array([1, 0, 0]))
                    operations = reverse_direction(operations,
                                                                             r_direction)
                    r2_index, r2_direction = find_rotation(operations,
                                                           3, np.array([1, 1, 1]),r_direction)
                    operations = reverse_direction(operations,
                                                                             r2_direction)
                    generators = [operations[r_index][0],
                                  operations[r2_index][0],
                                  operations[r_index][0] @ operations[r2_index][0] @ operations[r_index][0]@ operations[r_index][0] @ operations[r2_index][0] @operations[r_index][0]@operations[r_index][0]
                                  ]  # matrix 4_100 @ 3+_111 @ 2_100 @ 3+_111 @ 2_100
                    generators_index = [r_index,r2_index,np.where([np.allclose(i[0],operations[r_index][0] @ operations[r2_index][0] @ operations[r_index][0]@ operations[r_index][0] @ operations[r2_index][0] @operations[r_index][0]@operations[r_index][0],rtol=1e-2) for i in operations])[0][0]]

        if order_group == 48: # Oh
            group_symbol = 'm-3m'
            group_symbol_S = 'Oh'

            ir_index, ir_direction = find_rotation(operations,
                                                   6, np.array([1, 1, 1]), improper=True)
            operations = reverse_direction(operations,ir_direction)

            mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]),cubic=ir_direction)
            operations = reverse_direction(operations,m_direction)

            m2_index, m2_direction = find_mirror(operations, ir_direction,ir_direction,m_direction,cubic = True)
            operations = reverse_direction(operations,
                                                                     m2_direction)

            generators = [operations[mirror_index][0],
                          operations[ir_index][0],
                          operations[m2_index][0]
                          ]  # matrix
            generators_index = [mirror_index,ir_index,m2_index]

        if order_group == 60: # I
            group_symbol = '532'
            group_symbol_S = 'I'

        if order_group == 120:  # Ih
            group_symbol = '-5-32'
            group_symbol_S = 'Ih'

    if len(counter_high_order_axis) == 1: # n  or  -n             Cn Cnv Cnh Dn Dnh Dnd Sn           (n>2)
        if improper: # Cnv Cnh Dnh Dnd Sn ---without D2h C2v D2 C2h Cs C2 Ci
            if order_group == max_order: # S2n
                if (order_group / 2) % 2 == 0: # -2n    S2n
                    group_symbol = f'-{max_order}'
                    group_symbol_S = f'S{max_order}'
                    improper_rotation_index, direction = find_rotation(operations,max_order,np.array([0,0,1]),improper=True)
                    operations = reverse_direction(operations,direction)
                    generators = [operations[improper_rotation_index][0]] # matrix
                    generators_index = [improper_rotation_index]
                else:
                    if minues:
                        group_symbol = f'-{int(max_order/2)}'  # -n   S2n
                        group_symbol_S = f'S{max_order}'
                        improper_rotation_index, direction = find_rotation(operations,
                                                                           max_order, np.array([0, 0, 1]), improper=True)
                        operations = reverse_direction(operations,
                                                                                 direction)
                        generators = [operations[improper_rotation_index][0]] # matrix
                        generators_index = [improper_rotation_index]
                    else:
                        group_symbol = f'-{max_order}' # -2n  Cnh
                        group_symbol_S = f'C{int(max_order/2)}h'
                        improper_rotation_index, direction = find_rotation(operations,
                                                                           max_order, np.array([0, 0, 1]), improper=True)
                        operations = reverse_direction(operations,
                                                                                 direction)
                        generators = [operations[improper_rotation_index][0]] # matrix
                        generators_index = [improper_rotation_index]
            else:
                if len(rotation2_axis) < 2: # Cnv Ceh
                    if minues: # n/m    Cnh
                        group_symbol = f'{int(order_group/2)}/m'
                        group_symbol_S = f'C{int(order_group/2)}h'
                        rotation_index, direction = find_rotation(operations,
                                                                           int(order_group/2), np.array([0, 0, 1]))
                        operations = reverse_direction(operations,
                                                                                 direction)
                        mirror_index, m_direction = find_mirror(operations, direction)

                        generators = [operations[rotation_index][0],operations[mirror_index][0]] # matrix
                        generators_index = [rotation_index,mirror_index]
                    else:
                        if (order_group / 2) % 2 == 0: # nmm    Cnv
                            if int(order_group/2) > 9:
                                group_symbol = f'({int(order_group/2)})mm'
                            else:
                                group_symbol = f'{int(order_group/2)}mm'
                            group_symbol_S = f'C{int(order_group/2)}v'
                            rotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]))
                            operations = reverse_direction(
                                operations,
                                direction)
                            mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]))
                            operations = reverse_direction(operations,m_direction)
                            mirror2_index = find_operation_index_by_matrix(
                                operations,
                                operations[rotation_index][0] @ operations[mirror_index][0],
                            )
                            generators = [operations[rotation_index][0],operations[mirror_index][0],operations[mirror2_index][0]]
                            generators_index = [rotation_index,mirror_index,mirror2_index]



                        else: # nm  Cnv
                            if int(order_group/2)>9:
                                group_symbol = f'({int(order_group/2)})m'
                                group_symbol_S = f'C{int(order_group/2)}v'
                            else:
                                group_symbol = f'{int(order_group/2)}m'
                                group_symbol_S = f'C{int(order_group/2)}v'
                            rotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]))
                            operations = reverse_direction(
                                operations,
                                direction)
                            mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]))
                            operations = reverse_direction(operations,m_direction)
                            generators = [operations[rotation_index][0],operations[mirror_index][0]]
                            generators_index = [rotation_index,mirror_index]




                else: # Dnd Dnh     --- without D2h
                    if int(order_group / 4) % 2 == 0:
                        if minues: # n/mmm    Dnh
                            group_symbol = f'{int(order_group/4)}/mmm'
                            group_symbol_S = f'D{int(order_group/4)}h'
                            rotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 4), np.array([0, 0, 1]))
                            operations = reverse_direction(
                                operations,
                                direction)
                            mz_index, mz_direction = find_mirror(operations, direction)
                            m_index, m_direction = find_mirror(operations, np.array([1, 0, 0]),direction)
                            operations = reverse_direction(operations,m_direction)
                            m2_index = find_operation_index_by_matrix(
                                operations,
                                operations[rotation_index][0] @ operations[m_index][0],
                            )
                            generators = [operations[rotation_index][0],operations[mz_index][0],operations[m_index][0],operations[m2_index][0]]
                            generators_index = [rotation_index,mz_index,m_index,m2_index]

                        else: #-(2n)2m    Dnd
                            if int(order_group/2) > 9:
                                group_symbol = f'-({int(order_group / 2)})2m'
                                group_symbol_S = f'D{int(order_group / 4)}d'
                            else:
                                group_symbol = f'-{int(order_group/2)}2m'
                                group_symbol_S = f'D{int(order_group/4)}d'
                            srotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]),improper=True)

                            operations = reverse_direction(
                                operations,
                                direction)
                            r_index, r_direction = find_rotation(operations,
                                                                      2, np.array([1, 0, 0]), direction)
                            operations = reverse_direction(operations,r_direction)
                            m_index = find_operation_index_by_matrix(
                                operations,
                                operations[srotation_index][0] @ operations[r_index][0],
                            )
                            generators = [operations[srotation_index][0],operations[r_index][0],operations[m_index][0]]
                            generators_index = [srotation_index,r_index,m_index]



                    else:
                        if minues: # -nm     Dnd    odd n
                            if int(order_group/4) > 9:
                                group_symbol = f'-({int(order_group / 4)})m'
                                group_symbol_S = f'D{int(order_group / 4)}d'
                            else:
                                group_symbol = f'-{int(order_group/4)}m'
                                group_symbol_S = f'D{int(order_group/4)}d'
                            srotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]),improper=True)
                            operations = reverse_direction(
                                operations,
                                direction)

                            m_index, m_direction = find_mirror(operations, np.array([1, 0, 0]),direction)
                            operations = reverse_direction(operations,m_direction)

                            generators = [operations[srotation_index][0],operations[m_index][0]]
                            generators_index = [srotation_index,m_index]
                        else: # -(2n)2m      Dnh   odd n
                            if int(order_group/2) > 9:
                                group_symbol = f'-({int(order_group / 2)})2m'
                                group_symbol_S = f'D{int(order_group / 4)}h'
                            else:
                                group_symbol = f'-{int(order_group/2)}2m'
                                group_symbol_S = f'D{int(order_group/4)}h'
                            srotation_index, direction = find_rotation(operations,
                                                                      int(order_group / 2), np.array([0, 0, 1]),improper=True)
                            operations = reverse_direction(
                                operations,
                                direction)
                            r_index, r_direction = find_rotation(operations,
                                                                      2, np.array([1, 0, 0]), direction)
                            operations = reverse_direction(operations,r_direction)
                            m_index = find_operation_index_by_matrix(
                                operations,
                                operations[srotation_index][0] @ operations[r_index][0],
                            )
                            generators = [operations[srotation_index][0],operations[r_index][0],operations[m_index][0]]
                            generators_index = [srotation_index,r_index,m_index]
        else: # Cn Dn
            if order_group == max_order: #   n     Cn
                if max_order > 9:
                    group_symbol = f'({max_order})'
                    group_symbol_S = f'C{max_order}'
                else:
                    group_symbol = f'{max_order}'
                    group_symbol_S = f'C{max_order}'
                rotation_index, direction = find_rotation(operations,
                                                           max_order, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)
                generators = [operations[rotation_index][0]]  # matrix
                generators_index = [rotation_index]
            else: # Dn
                if int(order_group / 2) % 2 == 0: # n22   Dn
                    if max_order > 9:
                        group_symbol = f'({max_order})22'
                        group_symbol_S = f'D{max_order}'
                    else:
                        group_symbol = f'{max_order}22'
                        group_symbol_S = f'D{max_order}'
                    rotation_index, direction = find_rotation(operations,
                                                              max_order, np.array([0, 0, 1]))
                    operations = reverse_direction(
                        operations,
                        direction)

                    r_index, r_direction = find_rotation(operations,
                                                              2, np.array([1, 0, 0]),direction)
                    operations = reverse_direction(
                        operations,
                        r_direction)
                    r2_index = find_operation_index_by_matrix(
                        operations,
                        operations[rotation_index][0] @ operations[r_index][0],
                    )
                    generators = [operations[rotation_index][0],operations[r_index][0],operations[r2_index][0]]  # matrix
                    generators_index = [rotation_index,r_index,r2_index]

                else:    # n2     Dn
                    if max_order > 9:
                        group_symbol = f'({max_order})2'
                        group_symbol_S = f'D{max_order}'
                    else:
                        group_symbol = f'{max_order}2'
                        group_symbol_S = f'D{max_order}'
                    rotation_index, direction = find_rotation(operations,
                                                              max_order, np.array([0, 0, 1]))
                    operations = reverse_direction(
                        operations,
                        direction)

                    r_index, r_direction = find_rotation(operations,
                                                              2, np.array([1, 0, 0]),direction)
                    operations = reverse_direction(
                        operations,
                        r_direction)
                    generators = [operations[rotation_index][0],
                                  operations[r_index][0]]  # matrix
                    generators_index = [rotation_index,r_index]
    if len(counter_high_order_axis) == 0: # Ci Cs C2 C2h C2v D2 D2h
        if order_group == 2:
            if minues: #Ci
                group_symbol = '-1'
                group_symbol_S = 'Ci'
                generators = [-np.eye(3)]
                generators_index = [np.where([i[1]==2 for i in operations] )[0][0]]
            elif improper: # Cs
                group_symbol = 'm'
                group_symbol_S = 'Cs'
                m_index, m_direction = find_mirror(operations, np.array([0, 1, 0]))
                operations = reverse_direction(
                    operations,
                    m_direction)
                generators = [operations[m_index][0]]
                generators_index = [m_index]
            else: # C2
                group_symbol = '2'
                group_symbol_S = 'C2'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)
                generators = [operations[rotation_index][0]]
                generators_index = [rotation_index]

        if order_group == 4:
            if minues:  # C2h
                group_symbol = '2/m'
                group_symbol_S = 'C2h'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(operations,
                                                                         direction)
                mirror_index, m_direction = find_mirror(operations, direction)

                generators = [operations[rotation_index][0],
                              operations[mirror_index][0]]  # matrix
                generators_index = [rotation_index,mirror_index]
            elif improper:   #C2v
                group_symbol = 'mm2'
                group_symbol_S = 'C2v'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)
                mirror_index, m_direction = find_mirror(operations, np.array([1, 0, 0]))
                operations = reverse_direction(operations,
                                                                         m_direction)
                mirror2_index = find_operation_index_by_matrix(
                    operations,
                    operations[rotation_index][0] @ operations[mirror_index][0],
                )
                generators = [
                              operations[mirror_index][0],
                              operations[mirror2_index][0],
                              operations[rotation_index][0]]
                generators_index = [mirror_index,mirror2_index,rotation_index]
            else:   # D2
                group_symbol = '222'
                group_symbol_S = 'D2'
                rotation_index, direction = find_rotation(operations,
                                                          2, np.array([0, 0, 1]))
                operations = reverse_direction(
                    operations,
                    direction)

                r_index, r_direction = find_rotation(operations,
                                                     2, np.array([1, 0, 0]), direction)
                operations = reverse_direction(
                    operations,
                    r_direction)
                r2_index, r2_direction = find_rotation(operations,
                                                       2, r_direction, direction, r_direction)
                operations = reverse_direction(
                    operations,
                    r2_direction)
                generators = [operations[rotation_index][0],
                              operations[r_index][0],
                              operations[r2_index][0]]  # matrix
                generators_index = [rotation_index,r_index,r2_index]

        if order_group == 8:    #D2h
            group_symbol = 'mmm'
            group_symbol_S = 'D2h'
            mz_index, mz_direction = find_mirror(operations, np.array([0, 0, 1]))
            m_index, m_direction = find_mirror(operations, np.array([1, 0, 0]), mz_direction)
            operations = reverse_direction(operations, m_direction)
            m2_index, m2_direction = find_mirror(operations, m_direction, mz_direction,
                                                 m_direction)
            operations = reverse_direction(operations, m2_direction)
            generators = [
                          operations[m_index][0],
                          operations[m2_index][0],
                          operations[mz_index][0]]
            generators_index = [m_index,m2_index,mz_index]

    # print(group_symbol)
    # print(generators_index)

    P2 = find_transition_matrix_deterministic(generators, group_symbol,id = _id)
    transformation = P1 @ P2

    for op in operations:
        if op[3] is not None:
            op[0] = P1 @ op[0] @ P1_inv
            op[3] = P1 @ op[3]

    return group_symbol, operations, transformation, generators_index,group_symbol_S


def nonc_point_group_generators(n,index):
    if n < 2 or not isinstance(n,int) :
        raise TypeError('n must be an integer greater than or equal to 2')
    if  index < 0 or index > 9:
        raise ValueError('index must be an integer between 0 and 9')
    generators_lambda = [[lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]])],
                         [lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,1,0],[0,0,-1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]]),lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]) @np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,1,0],[0,0,-1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]]),lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]) @np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]]),lambda n: np.array([[1,0,0],[0,-1,0],[0,0,-1]]),lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]]) @np.array([[1,0,0],[0,-1,0],[0,0,-1]]) ],
                         [lambda n: np.array([[-math.cos(2*math.pi/n),math.sin(2*math.pi/n),0],[-math.sin(2*math.pi/n),-math.cos(2*math.pi/n),0],[0,0,-1]]),lambda n: np.array([[-1,0,0],[0,1,0],[0,0,1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,-1,0],[0,0,-1]]),lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]])@np.array([[1,0,0],[0,-1,0],[0,0,-1]])],
                         [lambda n: np.array([[math.cos(2*math.pi/n),-math.sin(2*math.pi/n),0],[math.sin(2*math.pi/n),math.cos(2*math.pi/n),0],[0,0,1]]),lambda n: np.array([[1,0,0],[0,-1,0],[0,0,-1]])]
                         ]
    return [i(n) for i in generators_lambda[index]]


def generate_non_crystallographic_point_groups(hm_symbol):
    """
        input hm_symbol
        n  -n  n/m  nmm  nm  n/mmm  -n2m  -nm  n22  n2


        return generators(in cartesian system)


    """


    if bool(re.search(r'\(', hm_symbol)):
        pattern_index_pairs = [
            (r'-\((\d+)\)2m', 6),
            (r'\((\d+)\)/mmm', 5),
            (r'\((\d+)\)22', 8),
            (r'\((\d+)\)mm', 3),
            (r'\((\d+)\)/m', 2),
            (r'-\((\d+)\)m', 7),
            (r'\((\d+)\)2', 9),
            (r'\((\d+)\)m', 4),
            (r'-\((\d+)\)', 1),
            (r'\((\d+)\)', 0),
        ]
    else:
        pattern_index_pairs = [
            (r'-(\d+)2m', 6),
            (r'(\d+)/mmm', 5),
            (r'(\d+)22', 8),
            (r'(\d+)mm', 3),
            (r'(\d+)/m', 2),
            (r'-(\d+)m', 7),
            (r'(\d+)2', 9),
            (r'(\d+)m', 4),
            (r'-(\d+)', 1),
            (r'(\d+)', 0),
        ]

    n = None
    pattern_index = None
    for pattern_text, mapped_index in pattern_index_pairs:
        pattern = re.compile(pattern_text)
        match = re.fullmatch(pattern, hm_symbol)
        if match:
            n = int(match.group(1))
            pattern_index = mapped_index
            break
    if n is None:
        raise ValueError('No pattern found')
    generators = nonc_point_group_generators(n,pattern_index)

    return generators






def find_transition_matrix_deterministic(matrices_list, group_symbol, id =False, tol=1e-2):
    """
    Deterministically compute a nonsingular transformation matrix ``P``.

    Compared with the older implementation, this version:
    1. uses a relative singular-value threshold to estimate the null space;
    2. searches the null space deterministically using residual/condition
       scoring instead of relying on ad hoc coefficients or random fallback.
    """
    standard_list = _load_standard_point_group_generators(group_symbol, id=id)

    if len(matrices_list) != len(standard_list):
        raise ValueError(
            f"Matrix count ({len(matrices_list)}) does not match the number of standard generators ({len(standard_list)})."
        )

    linear_system = _build_transition_matrix_linear_system(matrices_list, standard_list)
    _, singular_values, vh = np.linalg.svd(linear_system)

    sv_tol = _transition_nullspace_threshold(singular_values, tol)
    null_space_dim = int(np.sum(singular_values < sv_tol))
    if null_space_dim == 0:
        if singular_values[-1] < max(float(tol), 1e-8):
            null_space_dim = 1
        else:
            raise ValueError(
                "No null space found; the input matrices may not form the requested point group. "
                "Adjust point-group identification tolerances first: "
                "`find_spin_group(..., meigtol=...)`, and then `matrix_tol=...` if needed. "
                "Do not start by changing `.scif` output precision."
            )
    null_space_dim = _expand_transition_nullspace_dimension_via_gap(
        singular_values,
        null_space_dim,
        tol=tol,
    )
    expected_null_space_dim = _expected_transition_nullspace_dimension(standard_list)
    best_candidate, attempts = _search_transition_candidate_progressive(
        singular_values,
        vh,
        linear_system,
        sv_tol=sv_tol,
        initial_dim=null_space_dim,
        gap_dim=null_space_dim,
        expected_dim=expected_null_space_dim,
    )

    if best_candidate is None or not best_candidate["passes_residual_tol"] or best_candidate["sigma_min"] < 1e-8:
        tried_dims = sorted({attempt["dim"] for attempt in attempts})
        raise ValueError(
            "Unable to find a nonsingular matrix P in the null space. "
            "Adjust PG-standardization tolerances first: "
            "`find_spin_group(..., matrix_tol=...)`, and `meigtol=...` if needed. "
            "Do not start by changing `.scif` output precision. "
            f"Tried near-null dimensions: {tried_dims}."
        )

    # Keep `.T`: the current Kronecker construction solves for P^T.
    return best_candidate["matrix"].T
def find_transition_matrix_random(matrices_list, group_symbol, tol=1e-2, max_tries=1000):
    """
    Randomly try candidate combinations until a nonsingular `P` is found.
    """

    from findspingroup.data.POINT_GROUP_MATRIX import point_group_generators

    if group_symbol not in point_group_generators: #
        standard_list = generate_non_crystallographic_point_groups(group_symbol)
        # raise ValueError(f"Unsupported point group: {group_symbol}")
    else:
        standard_list = [np.array(op) for op in point_group_generators[group_symbol]]

    if len(matrices_list) != len(standard_list):
        raise ValueError(
            f"Matrix count ({len(matrices_list)}) does not match the number of standard generators ({len(standard_list)})."
        )

    I = np.eye(3)
    A_blocks = []
    for M, D in zip(matrices_list, standard_list):
        A_block = np.kron(I, M) - np.kron(D.T, I)
        A_blocks.append(A_block)

    A = np.vstack(A_blocks)

    # SVD decomposition.
    U, S, Vh = np.linalg.svd(A)
    null_space_dim = np.sum(S < tol)
    if null_space_dim == 0:
        raise ValueError("No null space found; the group elements may be inconsistent.")

    basis_vectors = Vh[-null_space_dim:, :]  # nullspace basis

    tries = 0
    while tries < max_tries:
        random_coeffs = np.random.randint(-5, 5, size=null_space_dim)
        P_flat = random_coeffs @ basis_vectors
        P = P_flat.reshape(3, 3)

        if abs(np.linalg.det(P)) > 0.3:
            #

            # print(f"Found non-singular P in {tries + 1} tries.")
            return P.T

        tries += 1

    raise ValueError(f"Failed to find a nonsingular P after {max_tries} attempts.")


def getNormInf(matrix1, matrix2, mode=True):
    if mode:
        a = np.mod(np.asarray(matrix1, dtype=float), 1.0)
        b = np.mod(np.asarray(matrix2, dtype=float), 1.0)
        diff = np.abs(a - b)
        wrapped = np.minimum(diff, 1.0 - diff)
        return float(np.max(wrapped))
    diff = np.abs(np.asarray(matrix1, dtype=float) - np.asarray(matrix2, dtype=float))
    return float(np.max(diff))


def is_close_matrix_pair(pair1, pair2, tol=1e-5):
    if len(pair1) != len(pair2):
        raise ValueError("Compare two vectors of different lengths.")
    for i, j in enumerate(pair1):
        if not np.allclose(np.array(pair1[i]), np.array(pair2[i]), atol=tol):
            return False
    return True


def _dedup_bucket_decimals(tol: float) -> int:
    tol = float(max(tol, 1e-12))
    return max(0, int(np.ceil(-np.log10(tol))) - 1)


def _matrix_pair_bucket_key(item, tol=1e-5):
    decimals = _dedup_bucket_decimals(tol)
    key_parts = []
    for value in item:
        arr = np.asarray(value, dtype=float).reshape(-1)
        key_parts.append(tuple(np.round(arr, decimals)))
    return tuple(key_parts)


def deduplicate_matrix_pairs(matrix_list, tol=1e-5):
    unique = []
    buckets = {}
    for item in matrix_list:
        bucket_key = _matrix_pair_bucket_key(item, tol=tol)
        candidates = buckets.get(bucket_key, [])
        if any(is_close_matrix_pair(item, u, tol) for u in candidates):
            continue
        unique.append(item)
        buckets.setdefault(bucket_key, []).append(item)
    return unique


def _canonicalize_fractional_translation(translation, tol=3e-3, max_den=12):
    translation = np.mod(np.asarray(translation, dtype=float), 1.0)
    snapped = []
    for value in translation:
        approx = float(Fraction(float(value)).limit_denominator(max_den))
        if abs(value - approx) < tol:
            value = approx
        if abs(value) < tol or abs(value - 1.0) < tol:
            value = 0.0
        snapped.append(value)
    return np.mod(np.asarray(snapped, dtype=float), 1.0)


def _fractional_bucket_params(tol: float):
    tol = float(max(tol, 1e-12))
    bins = max(1, int(np.ceil(1.0 / tol)))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(tol / bucket_width)))
    return bins, neighbor_radius


def _fractional_bucket_key(position, bins: int):
    wrapped = np.mod(np.asarray(position, dtype=float), 1.0)
    indices = np.floor(wrapped * bins).astype(int) % bins
    return tuple(int(value) for value in indices)


def _fractional_neighbor_keys(bucket_key, bins: int, neighbor_radius: int):
    for dx in range(-neighbor_radius, neighbor_radius + 1):
        for dy in range(-neighbor_radius, neighbor_radius + 1):
            for dz in range(-neighbor_radius, neighbor_radius + 1):
                yield (
                    (bucket_key[0] + dx) % bins,
                    (bucket_key[1] + dy) % bins,
                    (bucket_key[2] + dz) % bins,
                )


def _append_unique_fractional_position(
    position,
    positions: list,
    buckets: dict,
    *,
    tol: float,
    bins: int,
    neighbor_radius: int,
):
    bucket_key = _fractional_bucket_key(position, bins)
    for neighbor_key in _fractional_neighbor_keys(bucket_key, bins, neighbor_radius):
        for existing in buckets.get(neighbor_key, ()):
            if getNormInf(position, existing) < tol:
                return False
    positions.append(position)
    buckets.setdefault(bucket_key, []).append(position)
    return True


def compute_invariant_metric(point_group_rotations):
    """calculate invariant metric tensor g。"""
    if not point_group_rotations:
        raise ValueError("no point group rotations provided.")

    g0 = np.eye(3)
    g = np.zeros((3, 3))
    for R in point_group_rotations:
        g += np.dot(R.T, np.dot(g0, R))
    g /= len(point_group_rotations)

    # check positive definiteness
    eigvals = np.linalg.eigvalsh(g)
    if np.min(eigvals) < 1e-6:
        raise ValueError("cannot compute invariant metric tensor, not positive definite.")

    return g


def get_space_group_from_operations(space_group_operations,symprec = 0.02,bz = False)->SpglibDataset:


    weird_sites = [np.array([0.1715870, 0.27754210, 0.737388700]),np.array([0,0,0])]

    # get point group rotations
    point_group_rotations = deduplicate_matrix_pairs([i[0] for i in space_group_operations])
    g = compute_invariant_metric(point_group_rotations)
    lattice = np.linalg.cholesky(g)  # L @ L.T = g , rows as basis vectors

    positions = []
    types = []
    bins, neighbor_radius = _fractional_bucket_params(1e-3)
    position_buckets = {}
    rotations = np.asarray([op[0] for op in space_group_operations], dtype=float)
    translations = np.mod(np.asarray([op[1] for op in space_group_operations], dtype=float), 1.0)
    for index,site in enumerate(weird_sites):
        # Preserve the original fractional representatives here. G0/L0
        # standard-setting detection feeds identify-index exact generator
        # matching downstream, and snapping translations too early can change
        # the standard origin shift enough to lose canonical generators.
        transformed_positions = np.mod(rotations @ site + translations, 1.0)
        for new_pos in transformed_positions:
            if _append_unique_fractional_position(
                new_pos,
                positions,
                position_buckets,
                tol=1e-3,
                bins=bins,
                neighbor_radius=neighbor_radius,
            ):
                types.append(index+1)

    cell = (lattice*20, positions, types)

    space_group_dataset =get_symmetry_dataset(cell, symprec=symprec)
    if space_group_dataset.number in SG_HALL_MAPPING:
        # corresponding to the same space group setting as in Bilbao Crystallographic Server
        space_group_dataset =get_symmetry_dataset(cell, symprec=symprec, hall_number=SG_HALL_MAPPING[space_group_dataset.number])

    if bz :
        path_info = get_path(cell,with_time_reversal=False,symprec=symprec)
        return  space_group_dataset, path_info
    else:
        return space_group_dataset


def get_magnetic_space_group_from_operations(magnetic_space_group_operations):
    """
    :param magnetic_space_group_operations: [[ time_reversal{1,-1},rotation, translation)],...]
    :return : dict with keys:
        msg_int_num: int, magnetic space group international number
        msg_bns_num: str, magnetic space group BNS number
        msg_bns_symbol: str, magnetic space group BNS symbol
        msg_og_num: int, magnetic space group OG number
        msg_og_symbol: str, magnetic space group OG symbol
        msg_type: int, magnetic space group type (1-4)
        mpg_num: int, magnetic point group number
        mpg_symbol: str, magnetic point group symbol
    """
    symprec = 0.02

    weird_sites = [np.array([0.1715870, 0.27754210, 0.737388700]),np.array([0,0,0])]
    weird_moments = [np.array([1.234,0.789,0.345]),np.array([0,0,0])]

    canonical_operations = [
        [int(op[0]), np.asarray(op[1], dtype=float), _canonicalize_fractional_translation(op[2])]
        for op in magnetic_space_group_operations
    ]

    # get point group rotations
    point_group_rotations = deduplicate_matrix_pairs([i[1] for i in canonical_operations])
    g = compute_invariant_metric(point_group_rotations)
    lattice = np.linalg.cholesky(g)  # L @ L.T = g , rows as basis vectors

    positions = []
    types = []
    moments = []
    bins, neighbor_radius = _fractional_bucket_params(1e-4)
    position_buckets = {}
    rotations = np.asarray([op[1] for op in canonical_operations], dtype=float)
    translations = np.mod(np.asarray([op[2] for op in canonical_operations], dtype=float), 1.0)
    time_reversals = np.asarray([int(op[0]) for op in canonical_operations], dtype=float)
    det_signs = np.asarray([round(np.linalg.det(op[1])) for op in canonical_operations], dtype=float)
    for index,site in enumerate(weird_sites):
        transformed_positions = np.mod(rotations @ site + translations, 1.0)
        transformed_moments = (det_signs * time_reversals)[:, None] * (rotations @ weird_moments[index])
        for new_pos, new_mom in zip(transformed_positions, transformed_moments):
            if _append_unique_fractional_position(
                new_pos,
                positions,
                position_buckets,
                tol=1e-4,
                bins=bins,
                neighbor_radius=neighbor_radius,
            ):
                types.append(index+1)
                moments.append(new_mom)

    cell = (lattice * 5, positions, types, moments@lattice)
    magnetic_space_group_dataset :SpglibMagneticDataset=get_magnetic_symmetry_dataset(cell, symprec=symprec,mag_symprec=0.02)
    if magnetic_space_group_dataset is None:
        return None

    msg_int_num = magnetic_space_group_dataset.uni_number
    msg_bns_num,msg_bns_symbol = MSG_INT_TO_BNS[msg_int_num]
    msg_og_num = BNS_TO_OG_NUM[msg_bns_num]
    msg_og_symbol = OG_NUM_TO_MPG[msg_og_num]["og_label"]
    msg_type = magnetic_space_group_dataset.msg_type
    mpg_num = OG_NUM_TO_MPG[msg_og_num]["pointgroup_no"]
    mpg_symbol = OG_NUM_TO_MPG[msg_og_num]["pointgroup_label"]



    return {"msg_int_num":msg_int_num,
            "msg_bns_num":msg_bns_num,
            "msg_bns_symbol":msg_bns_symbol,
            "msg_og_num":msg_og_num,
            "msg_og_symbol":msg_og_symbol,
            "msg_type":msg_type,
            "mpg_num":mpg_num,
            "mpg_symbol":mpg_symbol}


def get_arithmetic_crystal_class_from_ops(ops, *, include_kpath: bool = True):
    """
    rely on spglib
    :parameter: ops: list of [rotation matrix, translation vector], space group operations
    :return: acc_symbol: str, arithmetic crystal class symbol
    """


    # get point group rotations
    if include_kpath:
        acc_dataset, kpath_info = get_space_group_from_operations(ops, bz=True)
    else:
        acc_dataset = get_space_group_from_operations(ops, bz=False)
        kpath_info = None

    if acc_dataset is None:
        raise ValueError("Can not find spg dataset in arithmetic crystal class ")

    international = acc_dataset.international
    bravais_lattice_letter = acc_dataset.international[0]
    pointgroup = acc_dataset.pointgroup

    # process 66 -> 73 TODO: test mm2C and mm2A
    if pointgroup == '-42m':
        if international[1:4] =='-42':
            pointgroup = '-42m'
        else:
            pointgroup = '-4m2'
    if pointgroup == '32' and bravais_lattice_letter == 'P':
        if international[-2:] == '12':
            pointgroup = '312'
        else:
            pointgroup = '321'

    if pointgroup == '3m' and bravais_lattice_letter == 'P':
        if international[-1] == '1':
            pointgroup = '3m1'
        else:
            pointgroup = '31m'

    if pointgroup == '-3m' and bravais_lattice_letter == 'P':
        if international[-1] == '1':
            pointgroup = '-3m1'
        else:
            pointgroup = '-31m'

    if pointgroup == '-62m':
        if international[-1] == '2':
            pointgroup = '-6m2'
        else:
            pointgroup = '-62m'

    acc_symbol = pointgroup + bravais_lattice_letter
    from ..structure.cell import primitive_cell_transformation


    input_acc_transformation = np.linalg.inv(primitive_cell_transformation(acc_dataset.international)) @ acc_dataset.transformation_matrix
    input_acc_origin_shift = normalize_vector_to_zero(np.linalg.inv(primitive_cell_transformation(acc_dataset.international)) @ acc_dataset.origin_shift ,atol=1e-9)
    # L_input = L_acc_std @ input_acc_std_transformation , L for col vector
    # L_acc = L_acc_std @ primitive_transformation
    # L_input = L_acc @ primitive_transformation^-1 @ input_acc_std_transformation
    # L_input = L_acc @ input_acc_transformation
    return acc_symbol, input_acc_transformation, input_acc_origin_shift, kpath_info
