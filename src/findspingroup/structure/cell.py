import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections.abc import Sequence
from spglib import standardize_cell as sc


from findspingroup.core.tolerances import Tolerances, DEFAULT_TOL
from findspingroup.utils.periodic import positions_within_cartesian_tolerance, fractional_search_radius
from findspingroup.version import __version__
from findspingroup.utils.matrix_utils import normalize_vector_to_zero, reduce_computed_mod1

MAGNETIC_PRESENCE_TOL = 1e-5


class SpaceToleranceDegeneracyError(ValueError):
    """Raised when space_tol makes atomic-site equivalence magnetically inconsistent."""


def _moment_distance(moment_a, moment_b):
    return float(np.linalg.norm(np.asarray(moment_a, dtype=float) - np.asarray(moment_b, dtype=float)))


def _within_closed_tolerance(distance, tolerance):
    distance = float(distance)
    tolerance = float(tolerance)
    slack = 64.0 * np.finfo(float).eps * max(1.0, abs(distance), abs(tolerance))
    return distance <= tolerance + slack


def standardize_lattice(lattice):
    """
        Standardizes the input lattice matrix such that:
          - Vector a is aligned along the x-axis
          - Vector b lies in the x-y plane
          - The system forms a right-handed coordinate system

        Parameters:
            lattice: np.ndarray, shape (3, 3), where each row is a lattice vector [a, b, c]

        Returns:
            normalized_lattice: np.ndarray, shape (3, 3), the standardized basis vectors
            rotation_matrix: np.ndarray, shape (3, 3), the rotation matrix from the original to the standard basis
    """
    a, b, c = lattice

    # Step 1: Define the three axes of the new coordinate system
    x_axis = a / np.linalg.norm(a)

    b_proj = b - np.dot(b, x_axis) * x_axis
    y_axis = b_proj / np.linalg.norm(b_proj)

    z_axis = np.cross(x_axis, y_axis)

    # Step 2: Construct the rotation matrix (columns represent the new basis)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Step 3: Project the original lattice vectors onto the new coordinate system
    normalized_lattice = lattice @ rotation_matrix

    return normalized_lattice, rotation_matrix.T

def angle_between(v1, v2, degrees=True):
    """Return the angle between two vectors."""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Zero vector has no defined angle.")
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle) if degrees else angle


def calculate_lattice_params(lattice):
    """
    lattice = [v1,v2,v3] row vectors
    Return (a, b, c, α, β, γ) from 3×3 lattice vectors.
    """
    lattice = np.asarray(lattice)
    norms = np.linalg.norm(lattice, axis=1)
    a, b, c = norms
    alpha = angle_between(lattice[1], lattice[2])
    beta = angle_between(lattice[2], lattice[0])
    gamma = angle_between(lattice[0], lattice[1])
    return a, b, c, alpha, beta, gamma

def calculate_vector_coordinates_from_latticefactors(a, b, c, alpha, beta, gamma):
    """
    Convert lattice parameters to 3x3 lattice vectors.

    Parameters
    ----------
    a, b, c : float
        Lattice lengths
    alpha, beta, gamma : float
        Angles in degrees

    Returns
    -------
    lattice_vectors : 3x3 list
        Lattice vectors in Cartesian coordinates
    """

    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    v1 = np.array([a, 0, 0])
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])

    c1 = c * np.cos(beta)
    c2 = (c * np.cos(alpha) - np.cos(gamma) * c1) / np.sin(gamma)
    c3_squared = c ** 2 - c1 ** 2 - c2 ** 2
    if c3_squared < 0:
        raise ValueError("Invalid lattice parameters, c3^2 < 0")
    c3 = np.sqrt(c3_squared)
    v3 = np.array([c1, c2, c3])

    if np.dot(np.cross(v1, v2), v3) < 0:
        v3[2] *= -1

    return np.array([v1, v2, v3])


def transform_moments(moments, lattice_factors, inverse=False,lattice_matrix = None):
    """
    Convert magnetic moments between lattice coordinates and Cartesian coordinates.

    Parameters
    ----------
    moments : array-like, shape (N,3)
        Magnetic moments in either lattice or Cartesian coordinates.
    lattice_factors : array-like, length 6
        Lattice factors: [a, b, c, alpha, beta, gamma] (angles in degrees)
    inverse : bool, default False
        If False, convert lattice -> Cartesian.
        If True, convert Cartesian -> lattice.
    lattice_matrix : array-like, optional
        Actual row-vector lattice in world Cartesian coordinates. Without it,
        use the canonical frame with a along x and b in the xy plane. Components
        are along unit lattice directions, not relative fractional spin axes.

    Returns
    -------
    moments_out : ndarray, shape (N,3)
        Magnetic moments in the target coordinate system.
    """
    alpha, beta, gamma = lattice_factors[3:]

    # lattice -> cartesian
    T_matrix = calculate_vector_coordinates_from_latticefactors(1, 1, 1, alpha, beta, gamma)
    if lattice_matrix is not None:
        canonical_lattice, world_to_canonical = standardize_lattice(np.asarray(lattice_matrix, dtype=float))
        if canonical_lattice[2, 2] < 0:
            T_matrix[2, 2] *= -1
        T_matrix = T_matrix @ world_to_canonical

    moments = np.asarray(moments)

    if inverse:
        # cartesian -> lattice
        moments_out = moments @ np.linalg.inv(T_matrix)
    else:
        # lattice -> cartesian
        moments_out = moments @ T_matrix

    return moments_out

def transform_c_moments_to_lattice(moments_in_cartesian, lattice_matrix):
    """
    Transform magnetic moments from Cartesian coordinates to lattice cartesian coordinates.
    :param moments_in_cartesian:
    :param lattice_matrix: row vectors
    :return: moments_in_lattice_cartesian
    """
    moments_in_cartesian = np.asarray(moments_in_cartesian)
    lattice_matrix = np.asarray(lattice_matrix)

    # 1.write moments in lattice_matrix-std-cartesian basis
    normed_lattice_matrix = np.array([v / np.linalg.norm(v) for v in lattice_matrix])
    moments_in_normed_lattice = moments_in_cartesian @ np.linalg.inv(normed_lattice_matrix)

    moments_in_lattice_cartesian = transform_moments(moments_in_normed_lattice, calculate_lattice_params(lattice_matrix), inverse=False)
    return moments_in_lattice_cartesian

def transform_lattice_moments_to_c(moments_in_lattice_cartesian, lattice_matrix):
    """
    :param moments_in_cartesian:
    :param lattice_matrix: row vectors
    :return: moments_in_lattice_cartesian
    """
    moments_in_lattice_cartesian = np.asarray(moments_in_lattice_cartesian)
    lattice_matrix = np.asarray(lattice_matrix)

    # 1.write moments in lattice_matrix-std-cartesian basis
    normed_lattice_matrix = np.array([v / np.linalg.norm(v) for v in lattice_matrix])
    moments_in_cartesian = moments_in_lattice_cartesian @ normed_lattice_matrix

    moments_in_lattice_cartesian = transform_moments(moments_in_cartesian, calculate_lattice_params(lattice_matrix), inverse=False)
    return moments_in_lattice_cartesian

def getNormInf(matrix1, matrix2, mode=True):
    if mode:
        a = np.mod(np.asarray(matrix1, dtype=float), 1.0)
        b = np.mod(np.asarray(matrix2, dtype=float), 1.0)
        diff = np.abs(a - b)
        wrapped = np.minimum(diff, 1.0 - diff)
        return float(np.max(wrapped))
    diff = np.abs(np.asarray(matrix1, dtype=float) - np.asarray(matrix2, dtype=float))
    return float(np.max(diff))

def primitive_cell_transformation(international_symbol):
    primitive_transformation_matrix = {'P':np.array([[1,0,0],[0,1,0],[0,0,1]]),
                                       'A':np.array([[1,0,0],[0,1/2,-1/2],[0,1/2,1/2]]),
                                       'C':np.array([[1/2,1/2,0],[-1/2,1/2,0],[0,0,1]]),
                                       'R':np.array([[2/3,-1/3,-1/3],[1/3,1/3,-2/3],[1/3,1/3,1/3]]),
                                       'I':np.array([[-1/2,1/2,1/2],[1/2,-1/2,1/2],[1/2,1/2,-1/2]]),
                                       'F':np.array([[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]])}
    if international_symbol[0] in primitive_transformation_matrix.keys():
        return primitive_transformation_matrix[international_symbol[0]]
        # column vector
    else:
        raise 'Wrong international symbol'


def classify_by_occupancies_and_elements(data, tol=1e-6):
    """
    data:
    """
    groups = []
    result = []
    group_counts = {}
    group_id_counter = 0
    type_occupancy = {}
    type_symbols = {}
    for idx, atom in enumerate(data):
        gid = None
        # check existing groups
        for (ga, gb_ref, g_id) in groups:
            if atom.element_symbol == ga and abs(atom.occupancy - gb_ref) <= tol:
                gid = g_id
                break

        # if not found, create a new group
        if gid is None:
            group_id_counter += 1
            gid = group_id_counter
            groups.append((atom.element_symbol, atom.occupancy, gid))
            group_counts[gid] = 0

        # update counts and results
        group_counts[gid] += 1
        result.append(gid)
        type_occupancy[gid] = atom.occupancy
        type_symbols[gid] = atom.element_symbol

    return result, type_symbols, type_occupancy




def are_positions_equivalent(pos1: list[float]|np.ndarray, pos2: list[float]|np.ndarray,
                           tolerance: float = 0.005) -> bool:
    """Check if two positions are equivalent within tolerance."""
    return getNormInf(pos1, pos2) < tolerance


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


def _as_integer_unimodular_matrix(matrix, *, tol: float):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        return None
    rounded = np.rint(matrix).astype(int)
    if not np.allclose(matrix, rounded, atol=tol, rtol=0.0):
        return None
    rounded_det = round(np.linalg.det(rounded))
    if abs(rounded_det) != 1:
        return None
    if not np.isclose(np.linalg.det(matrix), rounded_det, atol=tol, rtol=0.0):
        return None
    return rounded


def _fractional_positions_unique_by_type(positions, types, *, eps: float):
    bins, neighbor_radius = _fractional_bucket_params(eps)
    position_buckets: dict[tuple, list[np.ndarray]] = {}
    for position, atom_type in zip(positions, types):
        wrapped = np.mod(np.asarray(position, dtype=float), 1.0)
        bucket_key = _fractional_bucket_key(wrapped, bins)
        for neighbor_key in _fractional_neighbor_keys(bucket_key, bins, neighbor_radius):
            for existing in position_buckets.get((atom_type, neighbor_key), ()):
                if getNormInf(wrapped, existing) < eps:
                    return False
        position_buckets.setdefault((atom_type, bucket_key), []).append(wrapped)
    return True


def _change_cell_settings_unimodular_fast_path(
    old_cell,
    transformation_matrix,
    origin_shift,
    mag,
    *,
    eps: float,
    moment_eps: float,
):
    integer_transformation = _as_integer_unimodular_matrix(transformation_matrix, tol=1e-10)
    if integer_transformation is None:
        return None

    old_positions = np.asarray(old_cell[1], dtype=float)
    old_types = list(old_cell[2])
    # The integer matrix certifies topology only; never replace the supplied
    # affine map used by the paired SSG transformation.
    transformation = np.asarray(transformation_matrix, dtype=float)
    origin_shift = np.asarray(origin_shift, dtype=float)
    inverse_integer_transformation = np.rint(np.linalg.inv(integer_transformation)).astype(int)
    new_cell_lattice = np.linalg.inv(transformation).T @ np.asarray(old_cell[0], dtype=float)
    boundary = np.maximum(eps * np.linalg.norm(np.linalg.inv(new_cell_lattice), axis=0), 1e-12)
    direct_positions = old_positions @ transformation.T + origin_shift
    old_moments = [np.asarray(item, dtype=float) for item in mag]
    candidate_entries = []
    for atom_index, direct_position in enumerate(direct_positions):
        offset_options = []
        for axis, component in enumerate(direct_position):
            base = math.floor(-float(component))
            component_offsets = []
            for offset in range(base - 2, base + 4):
                shifted = float(component) + offset
                if -boundary[axis] < shifted < 1.0 + boundary[axis]:
                    component_offsets.append(offset)
            if not component_offsets:
                return None
            offset_options.append(component_offsets)

        for offset in itertools.product(*offset_options):
            offset = np.asarray(offset, dtype=int)
            shift = inverse_integer_transformation @ offset
            candidate_entries.append(
                (
                    tuple(int(value) for value in shift),
                    atom_index,
                    direct_position + transformation @ shift,
                )
            )
    if not candidate_entries:
        return None

    candidate_entries.sort(key=lambda item: (item[0], item[1]))
    new_cell_positions = []
    new_cell_types = []
    new_cell_moments = []
    seen_source_atoms = set()
    for _shift, atom_index, position in candidate_entries:
        if atom_index in seen_source_atoms:
            continue
        # A unimodular map is a known bijection: only different periodic
        # images of this SAME input atom may be deduplicated here.
        seen_source_atoms.add(atom_index)
        new_cell_positions.append(reduce_computed_mod1(position))
        new_cell_types.append(old_types[atom_index])
        new_cell_moments.append(old_moments[atom_index])

    if len(new_cell_positions) != len(old_positions):
        return None
    return new_cell_lattice, new_cell_positions, new_cell_types, new_cell_moments



def find_cell_border(a, b, c):
    """
    Calculate the minimum and maximum values of the x, y, z components for the linear combination
    of three 3D vectors a, b, c with coefficients A, B, C in the range [0, 1].

    Parameters:
        a (tuple): First 3D vector (ax, ay, az).
        b (tuple): Second 3D vector (bx, by, bz).
        c (tuple): Third 3D vector (cx, cy, cz).

    Returns:
        dict: A dictionary with keys 'x', 'y', 'z', each mapping to a tuple (min, max)
              representing the minimum and maximum values of the respective component.

    Example:
        >>> a = (1, -2, 3)
        >>> b = (-1, 4, 0)
        >>> c = (2, 1, -1)
        >>> find_min_max(a, b, c)
        {'x': (-1, 3), 'y': (-2, 5), 'z': (-1, 3)}
    """

    # Extract x, y, z components of each vector
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    # Generate all possible combinations of coefficients A, B, C in {0, 1}
    combinations = list(itertools.product([0, 1], repeat=3))

    # Initialize lists to store values of x, y, z components for all combinations
    vx_values, vy_values, vz_values = [], [], []

    # Compute the x, y, z components for each combination of A, B, C
    for A, B, C in combinations:
        vx = A * ax + B * bx + C * cx  # Calculate x-component
        vy = A * ay + B * by + C * cy  # Calculate y-component
        vz = A * az + B * bz + C * cz  # Calculate z-component
        vx_values.append(vx)
        vy_values.append(vy)
        vz_values.append(vz)

    # Return the minimum and maximum values for each component
    return {
        'x': (min(vx_values), max(vx_values)),
        'y': (min(vy_values), max(vy_values)),
        'z': (min(vz_values), max(vz_values))
    }


def _validate_cell_translation_periods(positions, types, moments, lattice, periods, eps, moment_eps):
    """Validate new unit translations against the original physical cell."""
    bins, radius = _fractional_bucket_params(fractional_search_radius(lattice, eps))
    buckets = {}
    for i, (position, atom_type) in enumerate(zip(positions, types)):
        buckets.setdefault((atom_type, _fractional_bucket_key(position, bins)), []).append(i)
    permutations = []
    for period in periods.T:
        if np.allclose(period, np.rint(period), atol=1e-10, rtol=0):
            permutations.append(np.arange(len(positions)))
            continue
        permutation = []
        for i, position in enumerate(positions):
            target = position + period
            key = _fractional_bucket_key(target, bins)
            candidates = {j for neighbor in _fractional_neighbor_keys(key, bins, radius)
                          for j in buckets.get((types[i], neighbor), ())}
            matches = [j for j in candidates if positions_within_cartesian_tolerance(
                target, positions[j], lattice, eps)]
            if len(matches) != 1:
                reason = "ambiguous" if matches else "unmatched"
                raise SpaceToleranceDegeneracyError(
                    f"Target cell translation {period.tolist()} has {reason} site identity "
                    f"for source atom {i} under space_tol={eps} (length units)."
                )
            j = matches[0]
            distance = _moment_distance(moments[i], moments[j])
            if not _within_closed_tolerance(distance, moment_eps):
                raise SpaceToleranceDegeneracyError(
                    "space_tol identifies a target-cell translation whose moments differ "
                    f"beyond mtol: atoms {i},{j}, moment residual={distance}, mtol={moment_eps}."
                )
            permutation.append(j)
        if len(set(permutation)) != len(positions):
            raise SpaceToleranceDegeneracyError("Target cell translation is not a site bijection.")
        permutations.append(np.asarray(permutation))
    for left, right in itertools.combinations(permutations, 2):
        if not np.array_equal(left[right], right[left]):
            raise SpaceToleranceDegeneracyError("Target cell translations have noncommuting site permutations.")


def change_cell_settings(old_cell, transformation_matrix, origin_shift, eps=0.0001, moment_eps=None):
    """Change fractional coordinates by x_new=P*x_old+p, without idealization.

    Lattices contain row vectors, so L_new=P^-T*L_old. Cartesian moments are
    unchanged. A unimodular reindexing preserves each input site's identity;
    proximity of two distinct sites is not permission to merge them.

    ``eps`` is a physical position tolerance in the lattice's length unit.
    ``moment_eps`` is a Cartesian moment tolerance. These do not control matrix
    integrality or allow merging distinct source atoms in a pure expansion.
    A contraction must introduce valid, unambiguous source-cell translations.
    """
    eps = float(eps)
    moment_eps = eps if moment_eps is None else float(moment_eps)
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(moment_eps) or moment_eps < 0:
        raise ValueError("Cell comparison tolerances must be finite; eps positive and moment_eps nonnegative.")
    transformation_matrix = np.asarray(transformation_matrix, dtype=float)
    origin_shift = np.asarray(origin_shift, dtype=float).reshape(-1)
    if (transformation_matrix.shape != (3, 3) or origin_shift.shape != (3,)
            or not np.all(np.isfinite(transformation_matrix)) or not np.all(np.isfinite(origin_shift))):
        raise ValueError("Cell transformations require a finite 3x3 matrix and three-vector origin.")
    condition = float(np.linalg.cond(transformation_matrix))
    if not math.isfinite(condition) or condition * np.finfo(float).eps >= 1:
        raise ValueError("Cell transformation is singular or numerically unresolved.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected_count_float = len(old_cell[1]) * abs(np.linalg.det(np.linalg.inv(transformation_matrix)))
    if not math.isfinite(expected_count_float):
        raise ValueError("Cell transformation does not have a finite resolved volume ratio.")
    count_slack = max(1e-8, 64 * np.finfo(float).eps * condition * max(1., expected_count_float))
    expected_count = round(expected_count_float)
    if count_slack >= 0.1 or expected_count < 1 or abs(expected_count_float - expected_count) > count_slack:
        raise SpaceToleranceDegeneracyError(
            f"Cell transformation has non-integral or unresolved atom multiplicity: {expected_count_float:.12g}."
        )

    if len(old_cell) == 3:
        mag = np.array([[0,0,0]]*len(old_cell[1]))
    elif len(old_cell) == 4:
        mag = [np.array(i)for i in copy.deepcopy(old_cell[3])]
    else:
        raise ValueError("old_cell should be a tuple of (lattice,positions,types) or (lattice,positions,types,moments)")

    fast_result = _change_cell_settings_unimodular_fast_path(
        old_cell,
        transformation_matrix,
        origin_shift,
        mag,
        eps=eps,
        moment_eps=moment_eps,
    )
    if fast_result is not None:
        return fast_result

    inverse_transformation = np.linalg.inv(transformation_matrix)
    old_lattice = np.asarray(old_cell[0], dtype=float)
    old_positions = np.asarray(old_cell[1], dtype=float) % 1.0
    old_types = list(old_cell[2])
    old_moments = [np.asarray(item, dtype=float) for item in mag]
    pure_expansion = np.allclose(inverse_transformation, np.rint(inverse_transformation), atol=1e-10, rtol=0)
    if not pure_expansion:
        _validate_cell_translation_periods(old_positions, old_types, old_moments, old_lattice,
                                           inverse_transformation, eps, moment_eps)

    new_cell_lattice = inverse_transformation.T @ old_lattice
    # These are conservative enumeration bounds, not coordinate snapping or
    # duplicate-site acceptance. A length error projects differently on each axis.
    boundary = np.maximum(eps * np.linalg.norm(np.linalg.inv(new_cell_lattice), axis=0), 1e-12)
    old_origin = -inverse_transformation @ origin_shift
    border = find_cell_border(*inverse_transformation.T)
    ranges = [range(math.floor(border[axis][0] + old_origin[i]) - 1,
                    math.ceil(border[axis][1] + old_origin[i]) + 1)
              for i, axis in enumerate(("x", "y", "z"))]
    new_cell_positions = []
    new_cell_types = []
    new_cell_moments = []
    radius = 1e-10 if pure_expansion else fractional_search_radius(new_cell_lattice, eps)
    bins, neighbor_radius = _fractional_bucket_params(radius)
    position_buckets: dict[tuple, list[int]] = {}
    source_images = []
    for shift_tuple in itertools.product(*ranges):
        shift = np.asarray(shift_tuple, dtype=float)
        transformed = (old_positions + shift) @ transformation_matrix.T + origin_shift
        for i, position in enumerate(transformed):
            if not (np.all(position >= -boundary) and np.all(position < 1 + boundary)):
                continue
            group_key = i if pure_expansion else old_types[i]
            bucket_key = _fractional_bucket_key(position, bins)
            candidates = {j for neighbor in _fractional_neighbor_keys(bucket_key, bins, neighbor_radius)
                          for j in position_buckets.get((group_key, neighbor), ())}
            if pure_expansion:
                # Same source atom and same lattice coset, independent of eps.
                matches = [j for j in candidates if np.allclose(
                    transformation_matrix @ (shift - source_images[j]),
                    np.rint(transformation_matrix @ (shift - source_images[j])), atol=1e-10, rtol=0)]
            else:
                matches = [j for j in candidates if positions_within_cartesian_tolerance(
                    position, new_cell_positions[j], new_cell_lattice, eps)]
            if len(matches) > 1:
                raise SpaceToleranceDegeneracyError("Cell contraction has ambiguous output-site identity under space_tol.")
            if matches:
                j = matches[0]
                if not _within_closed_tolerance(_moment_distance(old_moments[i], new_cell_moments[j]), moment_eps):
                    raise SpaceToleranceDegeneracyError("space_tol identifies contracted sites whose moments differ beyond mtol.")
                continue
            new_cell_positions.append(reduce_computed_mod1(position))
            new_cell_types.append(old_types[i])
            new_cell_moments.append(old_moments[i])
            source_images.append(shift)
            position_buckets.setdefault((group_key, bucket_key), []).append(len(new_cell_positions) - 1)
    if len(new_cell_positions) != expected_count:
        raise SpaceToleranceDegeneracyError(
            "space_tol makes transformed atomic positions non-bijective for the "
            "current cell transformation; distinct sites collapse or the "
            "transformation is not valid under this tolerance."
        )

    return new_cell_lattice, new_cell_positions, new_cell_types, new_cell_moments









@dataclass
class AtomicSite:
    """
    Represents an atomic site with position, magnetic moment, occupancy, and element symbol.

    Attributes:
    --------------
    position (np.ndarray):
        3D position of the atom in fractional coordinates.
    magnetic_moment (np.ndarray):
        3D magnetic moment vector of the atom.
    occupancy (float):
        Occupancy of the atomic site.
    element_symbol (str | int):
        Element symbol or atomic number of the atom.
    lattice_matrix (np.ndarray | None):
        Row-vector lattice for physical periodic distances. CrystalCell supplies
        it automatically; standalone sites without it retain fractional matching.

    """
    position: np.ndarray | list[float]
    magnetic_moment: np.ndarray | list[float]
    occupancy: float
    element_symbol: str | int
    lattice_matrix: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __repr__(self):
        return f'AtomicSite(position={self.position}, magnetic_moment={self.magnetic_moment}, occupancy={self.occupancy}, element_symbol="{self.element_symbol}")'

    def __lt__(self, other):
        if not isinstance(other, AtomicSite):
            return NotImplemented
        # Compare element_symbol first
        if self.element_symbol != other.element_symbol:
            return int(self.element_symbol) < int(other.element_symbol)
        # Then compare occupancy
        if not np.isclose(self.occupancy, other.occupancy):
            return self.occupancy < other.occupancy
        # Then compare position
        for i in range(3):
            if not np.isclose(self.position[i], other.position[i]):
                return self.position[i] < other.position[i]
        # Finally compare magnetic_moment
        for i in range(3):
            if not np.isclose(self.magnetic_moment[i], other.magnetic_moment[i]):
                return self.magnetic_moment[i] < other.magnetic_moment[i]
        return False  # They are equal

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64).reshape(3,) % 1
        self.magnetic_moment = np.array(self.magnetic_moment, dtype=np.float64).reshape(3,)

    def is_equivalent(self, other, tol:Tolerances=DEFAULT_TOL):
        """Compare sites in this site's basis; space tolerance is a length if known."""
        if not isinstance(other, AtomicSite):
            return False
        if self.lattice_matrix is None:
            pos_equal = _within_closed_tolerance(getNormInf(self.position, other.position), tol.space)
        else:
            pos_equal = positions_within_cartesian_tolerance(
                self.position, other.position, self.lattice_matrix, tol.space
            )
        mom_equal = _within_closed_tolerance(
            _moment_distance(self.magnetic_moment, other.magnetic_moment),
            tol.moment,
        )
        occ_equal = _within_closed_tolerance(abs(self.occupancy - other.occupancy), tol.occupancy)
        elem_equal = self.element_symbol == other.element_symbol
        return pos_equal and mom_equal and occ_equal and elem_equal

@dataclass
class Lattice:
    """
    input row vector as default
    Lattice([v1,v2,v3]) or Lattice((a,b,c,alpha,beta,gamma))
    """
    raw: Sequence[float] | np.ndarray
    matrix_row: np.ndarray = field(init=False)
    matrix_col: np.ndarray = field(init=False)
    factors: tuple[float, float, float, float, float, float] = field(init=False)
    def __post_init__(self):
        arr = np.asarray(self.raw, dtype=float)
        # -----------------------------
        # Case 1: raw == (a,b,c,α,β,γ)
        # -----------------------------
        if arr.ndim == 1 and arr.size == 6:
            a, b, c, alpha, beta, gamma = arr.tolist()
            M = calculate_vector_coordinates_from_latticefactors(
                a, b, c, alpha, beta, gamma
            )

            self.matrix_row = np.asarray(M, dtype=float)
            self.matrix_col = self.matrix_row.T
            self.factors = (a, b, c, alpha, beta, gamma)
            return

        # -----------------------------
        # Case 2: `raw` is a 3x3 matrix.
        # -----------------------------
        if arr.shape != (3, 3):
            raise ValueError("Lattice input must be 3×3 matrix or (a,b,c,alpha,beta,gamma).")


        self.matrix_row = arr
        self.matrix_col = arr.T


        a, b, c, alpha, beta, gamma = calculate_lattice_params(self.matrix_row)
        self.factors = (a, b, c, alpha, beta, gamma)

@dataclass
class CrystalCell:
    """
    """
    lattice: np.ndarray | List[float]
    positions: np.ndarray | List[List[float]]
    occupancies: np.ndarray | List[float]
    elements: List[str] | List[int]
    moments: Optional[np.ndarray | List[List[float]]] = None


    spin_setting:str|None = "cartesian"  # "in_lattice" | "cartesian" | None

    tol: Tolerances = field(default_factory=lambda: DEFAULT_TOL)


    lattice_matrix: np.ndarray = field(init=False)          # 3x3 row vectors
    lattice_factors: Tuple[float, float, float, float, float, float] = field(init=False)

    atoms: List[AtomicSite] = field(init=False)
    atom_types: List[int] = field(init=False)
    atom_types_to_symbol: Dict[int, str] = field(init=False)
    atom_types_to_occupancies: Dict[int, float] = field(init=False)
    magnetic_atom_indices: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.net_moment = None

        lat = Lattice(self.lattice)
        self.lattice_matrix = lat.matrix_row
        self.lattice_factors = lat.factors

        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3) % 1.0
        self.occupancies = np.asarray(self.occupancies, dtype=float).reshape(-1)
        self.elements = list(self.elements)
        site_count = len(self.positions)
        if site_count == 0:
            raise ValueError("CrystalCell requires at least one atomic site.")
        site_lengths = {
            "positions": site_count,
            "occupancies": len(self.occupancies),
            "elements": len(self.elements),
        }

        if self.moments is None:
            pass
        else:
            self.moments = np.asarray(self.moments, dtype=float).reshape(-1, 3)
            site_lengths["moments"] = len(self.moments)

        mismatched = {
            name: length
            for name, length in site_lengths.items()
            if length != site_count
        }
        if mismatched:
            lengths = ", ".join(
                f"{name}={length}" for name, length in site_lengths.items()
            )
            raise ValueError(
                "CrystalCell per-site arrays must have identical lengths "
                f"({lengths})."
            )

        if self.moments is not None:
            physical_moments = np.asarray(self.moments_cartesian)
            self.net_moment = np.linalg.norm(physical_moments.sum(axis=0))
            if any(np.linalg.norm(i) > MAGNETIC_PRESENCE_TOL for i in physical_moments):
                pass
            else:
                self.moments = None

        if self.moments is None:
            spins = [[0.0, 0.0, 0.0]] * len(self.positions)

        else:
            spins = self.moments

        packed = list(zip(
            self.positions,
            spins,
            self.occupancies,
            self.elements,
        ))

        packed_sorted = sorted(
            packed,
            key=lambda x: (
                np.linalg.norm(x[1]) == 0,
                str(x[3]),
                np.linalg.norm(x[1]),
                x[0].tolist(),
            ),
        )
        self.positions, spins, self.occupancies, self.elements = zip(*packed_sorted)
        if self.moments is None:
            pass
        else:
            self.moments = spins


        self.atoms = [
            AtomicSite(pos, spin, occ, elem, lattice_matrix=self.lattice_matrix)
            for pos, spin, occ, elem in zip(
                self.positions, spins, self.occupancies, self.elements
            )
        ]


        (
            self.atom_types,
            self.atom_types_to_symbol,
            self.atom_types_to_occupancies,
        ) = classify_by_occupancies_and_elements(self.atoms, tol=self.tol.occupancy)


        if self.moments is None:
            self.magnetic_atom_indices = None
        else:
            self.magnetic_atom_indices = [
                i for i, m in enumerate(self.moments_cartesian)
                if np.linalg.norm(m) > MAGNETIC_PRESENCE_TOL
            ]


    def __repr__(self):
        return f"CrystalCell(lattice={self.lattice}, positions={self.positions}, occupancies={self.occupancies}, elements={self.elements}, moments={self.moments},\n  spin_setting='{self.spin_setting}')"

    @property
    def moments_cartesian(self) -> Optional[np.ndarray]:
        if self.moments is None:
            return None
        if self.spin_setting == "cartesian":
            return self.moments
        elif self.spin_setting == "in_lattice":
            return transform_moments(self.moments, self.lattice_factors, inverse=False,
                                     lattice_matrix=self.lattice_matrix)
        else:
            raise ValueError("spin_setting must be 'in_lattice', 'cartesian', or None.")

    def get_primitive_structure(self, magnetic = False) -> 'CrystalCell':
        """
        Convert the current cell to its primitive structure.
        This is a placeholder implementation and should be replaced with actual logic.
        """

        if not magnetic or self.moments is None:
            return self._get_primitive_nonmagnetic()
        else:
            return self._get_primitive_magnetic()


    def _get_primitive_nonmagnetic(self):
        """ primitive cell"""
        cell = self.to_spglib()
        primitive_lattice, primitive_pos, primitive_types = sc(
            cell,
            symprec=self.tol.space,
            to_primitive=True,
            no_idealize=True,
        )

        new_occ = [self.atom_types_to_occupancies[t] for t in primitive_types]
        new_elem = [self.atom_types_to_symbol[t] for t in primitive_types]

        transformation_matrix = np.linalg.inv(primitive_lattice.T) @ self.lattice_matrix.T

        return CrystalCell(
            lattice=primitive_lattice,
            positions=primitive_pos,
            occupancies=new_occ,
            elements=new_elem,
            moments=None,
            spin_setting=None,
            tol=self.tol,
        ),transformation_matrix


    def _get_primitive_magnetic(self):

        # 1. make a L0 nonmagnetic cell
        L0_nonmagnetic_atom_types = self.atom_types.copy()
        L0_nonmagnetic_atom_types_to_symbol = self.atom_types_to_symbol.copy()
        L0_nonmagnetic_atom_types_to_occupancies = self.atom_types_to_occupancies.copy()
        L0_nonmagnetic_atom_types_to_moments = {i: np.array([0.0, 0.0, 0.0]) for i in L0_nonmagnetic_atom_types_to_symbol.keys()}


        moments_cartesian = self.moments_cartesian

        # print(set(self.atom_types))
        # classify magnetic atoms by their types
        mag_atom_types_to_indices = {}
        for index in self.magnetic_atom_indices:
            if self.atom_types[index] not in mag_atom_types_to_indices:
                mag_atom_types_to_indices[self.atom_types[index]] = []
            mag_atom_types_to_indices[self.atom_types[index]].append(index)

        max_type_number = max(L0_nonmagnetic_atom_types)
        for key in mag_atom_types_to_indices.keys():


            group = [] # save the first different moment atom indices
            for index in mag_atom_types_to_indices[key]:
                if group == []:
                    group.append(index)
                    max_type_number += 1
                    L0_nonmagnetic_atom_types[index] = max_type_number
                    L0_nonmagnetic_atom_types_to_symbol[L0_nonmagnetic_atom_types[index]] = self.elements[index]
                    L0_nonmagnetic_atom_types_to_occupancies[L0_nonmagnetic_atom_types[index]] = self.occupancies[index]
                    L0_nonmagnetic_atom_types_to_moments[L0_nonmagnetic_atom_types[index]] = moments_cartesian[index]
                else:
                    ok = True
                    for g in group:
                        if _moment_distance(moments_cartesian[g], moments_cartesian[index]) <= self.tol.moment:

                            L0_nonmagnetic_atom_types[index] = L0_nonmagnetic_atom_types[g]
                            ok = False
                            break

                    if ok:
                        group.append(index)
                        max_type_number += 1
                        L0_nonmagnetic_atom_types[index] = max_type_number
                        L0_nonmagnetic_atom_types_to_symbol[L0_nonmagnetic_atom_types[index]] = self.elements[index]
                        L0_nonmagnetic_atom_types_to_occupancies[L0_nonmagnetic_atom_types[index]] = self.occupancies[index]
                        L0_nonmagnetic_atom_types_to_moments[L0_nonmagnetic_atom_types[index]] = moments_cartesian[index]

        L0_nonmagnetic_cell = tuple([self.lattice_matrix,self.positions,L0_nonmagnetic_atom_types])

        try:
            primitive_result = sc(
                L0_nonmagnetic_cell,
                symprec=self.tol.space,
                to_primitive=True,
                no_idealize=True,
            )
        except Exception as exc:
            raise SpaceToleranceDegeneracyError(
                "spglib failed to find the magnetic primitive cell under the "
                "current space_tol after magnetic sites were split by mtol."
            ) from exc
        if primitive_result is None:
            raise SpaceToleranceDegeneracyError(
                "spglib failed to find the magnetic primitive cell under the "
                "current space_tol after magnetic sites were split by mtol."
            )
        prim_lat, prim_pos, prim_types = primitive_result

        prim_occ = [L0_nonmagnetic_atom_types_to_occupancies[t] for t in prim_types]
        prim_elem = [L0_nonmagnetic_atom_types_to_symbol[t] for t in prim_types]
        prim_mom = [L0_nonmagnetic_atom_types_to_moments[t] for t in prim_types]

        transformation_matrix = np.linalg.inv(prim_lat.T) @ self.lattice_matrix.T


        return CrystalCell(
            lattice=prim_lat,
            positions=prim_pos,
            occupancies=prim_occ,
            elements=prim_elem,
            moments=np.array(prim_mom, dtype=float),
            spin_setting="cartesian",
            tol=self.tol,
        ),transformation_matrix

    def transform(self, matrix: np.ndarray, shift: np.ndarray, change_moment_to_lattice = False):
        """
        Apply a transformation to the cell.
        Default setting to default setting.

        Args:
            matrix (np.ndarray): 3x3 transformation matrix.
            shift (np.ndarray): 3-element shift vector.
            mode (str): The mode to apply the transformation to.

        Returns:
            CrystalCell: A new CrystalCell instance with the transformed cell.
        """
        if self.moments is None:
            new_cell = change_cell_settings(self.to_spglib(mag=False), matrix, shift, eps=self.tol.space)
        else:
            new_cell = change_cell_settings(
                (self.lattice_matrix, self.positions, self.atom_types, self.moments_cartesian),
                matrix,
                shift,
                eps=self.tol.space,
                moment_eps=self.tol.moment,
            )

        new_lattice, new_positions, new_types, new_moments = new_cell

        new_occupancies = [self.atom_types_to_occupancies[t] for t in new_types]
        new_elements = [self.atom_types_to_symbol[t] for t in new_types]

        if self.moments is None:
            final_moments = None
            new_spin_setting = None
        else:
            final_moments = np.asarray(new_moments, dtype=float)
            new_spin_setting = self.spin_setting
            if new_spin_setting == "in_lattice":
                final_moments = transform_moments(
                    final_moments, calculate_lattice_params(new_lattice),
                    inverse=True, lattice_matrix=new_lattice,
                )
        return CrystalCell(
            lattice=new_lattice,
            positions=new_positions,
            occupancies=new_occupancies,
            elements=new_elements,
            moments=final_moments,
            spin_setting=new_spin_setting,
            tol=self.tol,
        )

    def transform_spin(self,transform_matrix,setting):

        new_cell = CrystalCell(
            lattice=self.lattice,
            positions=self.positions,
            occupancies=self.occupancies,
            elements=self.elements,
            moments=[transform_matrix@ i for i in self.moments],
            spin_setting=setting,
            tol=self.tol,
        )

        return new_cell



    def to_spglib(self,mag = False):
        """
        Convert the current cell to a format compatible with spglib.
        Returns:
            tuple: (lattice, positions, occupancies) or (lattice, positions, occupancies, moments)
        """
        if not mag:
            return self.lattice_matrix, self.positions, self.atom_types
        else:
            if self.moments is None:
                raise ValueError("Magnetic moments are not defined for this cell.")
            else:
                return self.lattice_matrix, self.positions, self.atom_types, self.moments



    def to_poscar(self, filename) -> str:

        lattice,positions,types,moments = self.to_spglib(mag=True)
        positions_sorted,types_sorted,moments_sorted = zip(*sorted(zip(positions,types,moments),key=lambda x:(x[1],x[2][0],x[2][1],x[2][2])))
        cell = (lattice,positions_sorted,types_sorted,moments_sorted)
        atom_name = ['initial']
        count = ['initial']
        for i, j in enumerate([self.atom_types_to_symbol[i] for i in types_sorted]):
            if j != atom_name[-1]:
                atom_name.append(j)
                count.append(1)
            else:
                count[-1] += 1

        information = filename + f'#FINDSPINGROUP(version{__version__})'
        scale = '1'
        lattice = '\n'.join(' '.join(f'{value:.9f}' for value in row) for row in cell[0])
        species = ' '.join(atom_name[1:])
        atom_number = ' '.join(map(str, count[1:]))
        cartesian = 'direct'
        positions = '\n'.join(' '.join(f'{v:.8f}' for v in i) for i in cell[1])
        magmom = '# MAGMOM=' + ' '.join(
            ' '.join(f'{x:.8f}' for x in i) for i in cell[3]
        )
        return '\n'.join([information, scale, lattice, species, atom_number, cartesian, positions, magmom])
