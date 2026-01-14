import numpy as np

from spglib import get_symmetry_dataset as gsd, SpglibDataset

from findspingroup.core import Molecule, PointGroupAnalyzer
from findspingroup.core.tolerances import Tolerances, DEFAULT_TOL
from findspingroup.structure import *


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

def reflection_matrix(axis):
    """give a mirror perpendicular to axis"""
    u = normalize_vector(np.array(axis))
    u = u.reshape(3, 1)  # 使其成为列向量
    I = np.eye(3)
    reflection_matrix = I - 2 * np.dot(u, u.T)
    return reflection_matrix
def dedup_moments_with_tol(types, moments, tol=0.01):
    # print(types, moments, tol)
    moments = np.asarray(moments, float)
    types = np.asarray(types)


    idx = np.lexsort(moments.T)  # 按 moment 排序
    moments_sorted = moments[idx]
    types_sorted = types[idx]

    new_moments = [moments_sorted[0]]
    new_types = [types_sorted[0]]

    for i in range(1, len(moments_sorted)):
        if np.linalg.norm(moments_sorted[i] - new_moments[-1]) > tol:
            new_moments.append(moments_sorted[i])
            new_types.append(types_sorted[i])

    return np.array(new_types), np.array(new_moments)

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


    # filter out zero moments and deduplicate the same moments with same types
    non_zero_indices = np.where(np.linalg.norm(moments, axis=1) > meigtol)[0]

    filtered_moments = np.array([moments[i] for i in non_zero_indices])
    filtered_types = np.array([atom_types[i] for i in non_zero_indices])

    unique_types,unique_moments  = dedup_moments_with_tol(filtered_types,filtered_moments,mtol)
    # find point group

    # need deduplication before find pg
    pg = PointGroupAnalyzer(Molecule(unique_types,unique_moments),tolerance=mtol,eigen_tolerance=meigtol)
    pg_symbol = str(pg.get_pointgroup())
    pg_operations = [np.array(i.rotation_matrix,dtype=np.float64) for i in pg.get_symmetry_operations()]

    # deal with C*v D*h
    if pg_symbol in ('C*v', 'D*h'):
        z_axis = np.array([0, 0, 1])
        ref_axis = z_axis if abs(
            abs(np.dot(filtered_moments[0], z_axis) / np.linalg.norm(filtered_moments[0])) - 1) > 0.05 else np.array(
            [1, 0, 0])
        #if close to z, use x as ref, else use z as ref

        axis = np.cross(ref_axis, filtered_moments[0])
        mirror_v = reflection_matrix(axis)
        rotate_2 = -reflection_matrix(filtered_moments[0])
        mirror_2 = rotate_2 @ mirror_v

        # add mv, r2, m2
        pg_operations += [mirror_v, rotate_2, mirror_2]

        # if D*h, plus -mv, -r2, -m2
        if pg_symbol == 'D*h':
            pg_operations += [-mirror_v, -rotate_2, -mirror_2]

    return pg_symbol, pg_operations



def identify_spin_space_group(default_cell,find_primitive = True,tol:Tolerances=DEFAULT_TOL) -> (CrystalCell,SpinSpaceGroup):
    """
    Identify the spin space group of a given magnetic structure.


    Returns:
        ssg: SpinSpaceGroup
            The identified spin space group operations and settings.

    """
    if find_primitive == True:
        cell :CrystalCell = default_cell.get_primitive_structure(magnetic=True)[0]
    else:
        cell: CrystalCell = default_cell
    # get space operations
    p_dataset: SpglibDataset = gsd(cell.to_spglib(), symprec=tol.space)

    space_operations_list = list(zip(p_dataset.rotations, p_dataset.translations))
    # get point group operations for spin
    pg_symbol, spin_operations_list = get_pg(cell.moments, cell.atom_types,tol.moment,tol.m_eig)

    ssg_ops: list[SpinSpaceGroupOperation] = get_ssg_ops(space_operations_list, spin_operations_list,
                                                         [cell.atoms[i] for i in cell.magnetic_atom_indices])

    ssg = SpinSpaceGroup(ssg_ops)

    return ssg