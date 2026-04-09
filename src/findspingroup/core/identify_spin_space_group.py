from dataclasses import dataclass

import numpy as np

from spglib import get_symmetry_dataset as gsd, SpglibDataset

from findspingroup.core import Molecule, PointGroupAnalyzer
from findspingroup.core.pg_analyzer import SymmOp
from findspingroup.core.identify_symmetry_from_ops import deduplicate_matrix_pairs
from findspingroup.core.tolerances import Tolerances, DEFAULT_TOL
from findspingroup.structure.cell import MAGNETIC_PRESENCE_TOL
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


NONMAGNETIC_MTOL_ERROR = (
    "Under the current mtol the structure is identified as effectively nonmagnetic; "
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

def get_ssg_ops(sg,pg,mag_atoms):
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
    for R, t in sg:
        R = np.array(R, dtype=np.float64)
        t = np.array(t, dtype=np.float64)
        for Rs in pg:
            Rs = np.array(Rs, dtype=np.float64)
            # check if this (R,Rs) can keep all moments invariant
            ok = True
            ssg_op = SpinSpaceGroupOperation(Rs,R,t)
            for atom in mag_atoms: # use AtomicSite.tol
                new_atom = ssg_op @ atom
                if not check_atom_in_list(new_atom, mag_atoms):
                    ok = False
                    break
            if ok:
                ssg_ops.append(ssg_op)
    return ssg_ops

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

    unique = []
    for direction in candidates:
        if any(np.allclose(direction, existing, atol=1e-6) for existing in unique):
            continue
        unique.append(direction)
    return unique


def _collinear_residual(moments, axis):
    axis = _canonicalize_direction(axis)
    moments = np.asarray(moments, dtype=float)
    if len(moments) == 0:
        return 0.0
    return max(float(np.linalg.norm(np.cross(moment, axis))) for moment in moments)


def _coplanar_residual(moments, plane_normal):
    plane_normal = _canonicalize_direction(plane_normal)
    moments = np.asarray(moments, dtype=float)
    if len(moments) == 0:
        return 0.0
    return max(float(abs(np.dot(moment, plane_normal))) for moment in moments)


def _best_collinear_axis(moments):
    candidates = _candidate_directions_from_moments(moments)
    best_axis = candidates[0]
    best_residual = _collinear_residual(moments, best_axis)
    for axis in candidates[1:]:
        residual = _collinear_residual(moments, axis)
        if residual < best_residual - 1e-10:
            best_axis = axis
            best_residual = residual
    return best_axis, best_residual


def _best_coplanar_normal(moments):
    candidates = _candidate_directions_from_moments(moments)
    best_normal = candidates[0]
    best_residual = _coplanar_residual(moments, best_normal)
    for normal in candidates[1:]:
        residual = _coplanar_residual(moments, normal)
        if residual < best_residual - 1e-10:
            best_normal = normal
            best_residual = residual
    return best_normal, best_residual


def _configuration_details(moments, mtol):
    moments = np.asarray(moments, dtype=float)
    if len(moments) <= 1:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        return {
            "configuration": "Collinear",
            "collinear_axis": axis,
            "collinear_residual": 0.0,
            "coplanar_normal": axis,
            "coplanar_residual": 0.0,
        }

    collinear_axis, collinear_residual = _best_collinear_axis(moments)
    coplanar_normal, coplanar_residual = _best_coplanar_normal(moments)
    if collinear_residual <= mtol:
        configuration = "Collinear"
    elif coplanar_residual <= mtol:
        configuration = "Coplanar"
    else:
        configuration = "Noncoplanar"
    return {
        "configuration": configuration,
        "collinear_axis": collinear_axis,
        "collinear_residual": float(collinear_residual),
        "coplanar_normal": coplanar_normal,
        "coplanar_residual": float(coplanar_residual),
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
    if pg_symbol == 'D*h':
        extra_ops += [-mirror_v, -rotate_2, -mirror_2]
    return extra_ops


def _configuration_compatibility(pg_symbol, configuration):
    linear_symbol = pg_symbol in ('C*v', 'D*h')
    if configuration == "Coplanar":
        return 0 if linear_symbol else 1
    if configuration == "Collinear":
        return 1 if linear_symbol else 0
    return 1


def _candidate_eigen_tolerances(meigtol):
    candidates = []
    for factor in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        value = max(1e-12, min(1e-1, float(meigtol) * factor))
        candidates.append(value)
    candidates.append(float(meigtol))
    return sorted(set(candidates))


def _build_pg_candidate(moment_types, moment_vectors, *, mtol, eig_tol):
    pg = PointGroupAnalyzer(Molecule(moment_types, moment_vectors), tolerance=mtol, eigen_tolerance=eig_tol)
    pg_symbol = str(pg.get_pointgroup())
    pg_operations = [np.array(i.rotation_matrix, dtype=np.float64) for i in pg.get_symmetry_operations()]

    if pg_symbol in ('C*v', 'D*h'):
        pg_operations += _materialize_linear_pg_ops(pg_symbol, pg)

    pg_operations = _deduplicate_pg_operations(pg_operations, tol=mtol)
    return {
        "symbol": pg_symbol,
        "operations": pg_operations,
        "eig_tol": float(eig_tol),
        "pg": pg,
    }


def _build_pg_candidates(moments, atom_types, mtol, meigtol):
    non_zero_indices = np.where(np.linalg.norm(moments, axis=1) > MAGNETIC_PRESENCE_TOL)[0]

    filtered_moments = np.array([moments[i] for i in non_zero_indices], dtype=float)
    filtered_types = np.array([atom_types[i] for i in non_zero_indices])

    unique_types, unique_moments = dedup_moments_with_tol(filtered_types, filtered_moments, mtol)
    configuration_details = _configuration_details(filtered_moments, mtol)
    configuration = configuration_details["configuration"]

    candidates = []
    seen = set()
    for eig_tol in _candidate_eigen_tolerances(meigtol):
        candidate = _build_pg_candidate(unique_types, unique_moments, mtol=mtol, eig_tol=eig_tol)
        key = (
            candidate["symbol"],
            len(candidate["operations"]),
            tuple(np.round(np.asarray(candidate["pg"].eigvals, dtype=float), 8)),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return {
        "configuration": configuration,
        "configuration_details": configuration_details,
        "filtered_moments": filtered_moments,
        "filtered_types": filtered_types,
        "unique_types": unique_types,
        "unique_moments": unique_moments,
        "candidates": candidates,
    }


def _select_get_pg_candidate(bundle, meigtol):
    configuration = bundle["configuration"]
    candidates = bundle["candidates"]

    base_candidate = min(candidates, key=lambda candidate: abs(candidate["eig_tol"] - meigtol))
    if _configuration_compatibility(base_candidate["symbol"], configuration) > 0:
        return base_candidate

    return max(
        candidates,
        key=lambda candidate: (
            _configuration_compatibility(candidate["symbol"], configuration),
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


def _select_identify_pg_candidate(bundle, space_operations_list, mag_atoms, meigtol):
    base_candidate = min(bundle["candidates"], key=lambda candidate: abs(candidate["eig_tol"] - meigtol))
    base_ssg_ops = get_ssg_ops(space_operations_list, base_candidate["operations"], mag_atoms)
    target_space_count = _distinct_space_operation_count(space_operations_list)
    if (
        _configuration_compatibility(base_candidate["symbol"], bundle["configuration"]) > 0
        and _distinct_real_space_count(base_ssg_ops) >= target_space_count
    ):
        return base_candidate, base_ssg_ops

    profiles = []
    for candidate in bundle["candidates"]:
        ssg_ops = get_ssg_ops(space_operations_list, candidate["operations"], mag_atoms)
        profiles.append(
            {
                "candidate": candidate,
                "ssg_ops": ssg_ops,
                "compat": _configuration_compatibility(candidate["symbol"], bundle["configuration"]),
                "space_count": _distinct_real_space_count(ssg_ops),
                "pg_op_count": len(candidate["operations"]),
                "ssg_op_count": len(ssg_ops),
                "signature": _ssg_signature(ssg_ops),
            }
        )

    max_compat = max(profile["compat"] for profile in profiles)
    profiles = [profile for profile in profiles if profile["compat"] == max_compat]

    max_space_count = max(profile["space_count"] for profile in profiles)
    profiles = [profile for profile in profiles if profile["space_count"] == max_space_count]

    min_pg_op_count = min(profile["pg_op_count"] for profile in profiles)
    profiles = [profile for profile in profiles if profile["pg_op_count"] == min_pg_op_count]

    min_ssg_op_count = min(profile["ssg_op_count"] for profile in profiles)
    profiles = [profile for profile in profiles if profile["ssg_op_count"] == min_ssg_op_count]

    distinct_profiles = {profile["signature"]: profile for profile in profiles}
    if len(distinct_profiles) > 1:
        raise ValueError(
            "Ambiguous PG candidates under current mtol; multiple distinct maximal SSG candidates remain."
        )

    selected = next(iter(distinct_profiles.values()))
    return selected["candidate"], selected["ssg_ops"]

def get_pg(moments,atom_types,mtol,meigtol):
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


    bundle = _build_pg_candidates(moments, atom_types, mtol, meigtol)
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
        raise ValueError(NONMAGNETIC_MTOL_ERROR)
    # get space operations
    p_dataset: SpglibDataset = gsd(cell.to_spglib(), symprec=tol.space)

    space_operations_list = list(zip(p_dataset.rotations, p_dataset.translations))
    # get point group operations for spin
    try:
        pg_bundle = _build_pg_candidates(cell.moments, cell.atom_types, tol.moment, tol.m_eig)
        _selected_pg, ssg_ops = _select_identify_pg_candidate(
            pg_bundle,
            space_operations_list,
            [cell.atoms[i] for i in cell.magnetic_atom_indices],
            tol.m_eig,
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
