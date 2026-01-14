import copy
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections.abc import Sequence
from spglib import standardize_cell as sc


from findspingroup.core.tolerances import Tolerances, DEFAULT_TOL
from findspingroup.version import __version__
from findspingroup.utils.matrix_utils import normalize_vector_to_zero


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

    Returns
    -------
    moments_out : ndarray, shape (N,3)
        Magnetic moments in the target coordinate system.
    """
    alpha, beta, gamma = lattice_factors[3:]

    # lattice -> cartesian
    T_matrix = calculate_vector_coordinates_from_latticefactors(1, 1, 1, alpha, beta, gamma)

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
    if mode == True:
        a = np.array(matrix1) % 1
        b = np.array(matrix2) % 1
        c = [1, 2, 3]
        for i in range(3):
            if a[i] > b[i]:
                c[i] = min(a[i] - b[i], 1 + b[i] - a[i])
            if a[i] < b[i]:
                c[i] = min(b[i] - a[i], 1 + a[i] - b[i])
            if a[i] == b[i]:
                c[i] = 0
        max_value = max(c)
    else:
        diff = np.abs(matrix1 - matrix2)
        max_value = np.max(diff)
    return max_value

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
                           tolerance: float = 0.001) -> bool:
    """Check if two positions are equivalent within tolerance."""
    return getNormInf(pos1, pos2) < tolerance




import itertools


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

def change_cell_settings(old_cell, transformation_matrix, origin_shift,eps=0.0001):
    """
    given the old cell and the transformation matrix and origin shift
    return the transformed cell


    old_cell =: [lattice,positions,types,moments_in_lattice] or [lattice,positions,types]
    transformation_matrix: 3x3 matrix old_cell @ transformation_matrix = new_cell
    origin_shift: 3x1 vector
    eps: tolerance for position comparison
    return new_cell -> [lattice,positions,types,moments]
    """

    # temporary fix
    transformation_matrix = np.linalg.inv(transformation_matrix)
    origin_shift = - transformation_matrix @ origin_shift



    if len(old_cell) == 3:
        mag = np.array([[0,0,0]]*len(old_cell[1]))
    elif len(old_cell) == 4:
        mag = [np.array(i)for i in copy.deepcopy(old_cell[3])]
    else:
        raise ValueError("old_cell should be a tuple of (lattice,positions,types) or (lattice,positions,types,moments)")

    #1.generate the temp cell that includes the new cell
    if abs(np.linalg.det(transformation_matrix)) < 0.001:
        raise ValueError("transformation matrix is not valid")
    eps = eps / abs(np.linalg.det(transformation_matrix)) # adjust eps according to the volume change

    border = find_cell_border(transformation_matrix.T[0], transformation_matrix.T[1],transformation_matrix.T[2])
    x_range = (math.floor(border['x'][0]+origin_shift[0]),math.ceil(border['x'][1]+origin_shift[0]))
    y_range = (math.floor(border['y'][0]+origin_shift[1]),math.ceil(border['y'][1]+origin_shift[1]))
    z_range = (math.floor(border['z'][0]+origin_shift[2]),math.ceil(border['z'][1]+origin_shift[2]))

    temp_cell_positions = []
    temp_cell_types = []
    temp_cell_moments = []
    for x in range(x_range[0], x_range[1]+1):
        for y in range(y_range[0], y_range[1]+1):
            for z in range(z_range[0], z_range[1]+1):
                temp_cell_positions  = temp_cell_positions + [np.array([p[0]+x, p[1]+y, p[2]+z]) for p in old_cell[1]]
                temp_cell_types = temp_cell_types + old_cell[2]

                temp_cell_moments = temp_cell_moments + mag
    # print(x_range, y_range, z_range,origin_shift)
    #2. collect the positions in the new cell
    temp_new_cell_positions = [np.linalg.inv(transformation_matrix) @ p - np.linalg.inv(transformation_matrix) @ origin_shift for p in temp_cell_positions]

    new_cell_lattice =  transformation_matrix.T @ old_cell[0] # row vector
    new_cell_positions = []
    new_cell_types = []
    new_cell_moments = []
    # print(len(temp_new_cell_positions))
    for i,j in enumerate(temp_new_cell_positions):
        if (-eps < j[0] < 1+eps ) and (-eps < j[1] < 1+eps ) and (-eps <= j[2] < 1+eps ) and (not any(getNormInf(j, existing_j)<eps and new_cell_types[ind]==temp_cell_types[i] for ind,existing_j in enumerate(new_cell_positions))):
            # if in (-eps,1+eps) range and not similar to existing positions
            new_cell_positions.append(normalize_vector_to_zero(j,atol=1e-8))
            new_cell_types.append(temp_cell_types[i])
            new_cell_moments.append(temp_cell_moments[i])
    # print(len(new_cell_positions),len(old_cell[1]),abs(np.linalg.det(transformation_matrix)))
    if len(new_cell_positions) != round(len(old_cell[1])*abs(np.linalg.det(transformation_matrix))):
        raise ValueError("The number of new cell positions does not match the number of old cell positions, please change tolerance. Or the transformation matrix is not valid.")

    mul_num_list = [0]*len(new_cell_positions)
    for i,p1 in enumerate(temp_new_cell_positions):
        for j,p2 in enumerate(new_cell_positions):
            if getNormInf(p1%1,p2)<eps and temp_cell_types[i] == new_cell_types[j]:
                # print(p1,p2)
                if sum(abs(temp_cell_moments[i]- new_cell_moments[j]))>eps:
                    raise ValueError(f"Atom moments mismatch after transformation, p:{temp_new_cell_positions[i]}moments:{temp_cell_moments[i]} and p:{new_cell_positions[j]}moments:{new_cell_moments[j]},please change tolerance. Or it's not a valid transformation matrix.")
                mul_num_list[j] +=1
                break

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

    """
    position: np.ndarray | list[float]
    magnetic_moment: np.ndarray | list[float]
    occupancy: float
    element_symbol: str | int

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
        """Check if two AtomicSite instances are equivalent within a tolerance."""
        if not isinstance(other, AtomicSite):
            return False
        pos_equal = getNormInf(self.position, other.position) < tol.space
        mom_equal = np.allclose(self.magnetic_moment, other.magnetic_moment,atol=tol.moment)
        occ_equal = abs(self.occupancy - other.occupancy) < tol.occupancy
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
        # Case 2: raw == 3×3 矩阵
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

        lat = Lattice(self.lattice)
        self.lattice_matrix = lat.matrix_row
        self.lattice_factors = lat.factors

        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3) % 1.0
        self.occupancies = np.asarray(self.occupancies, dtype=float).reshape(-1)



        if self.moments is None:
            pass
        else:
            self.moments = np.asarray(self.moments, dtype=float).reshape(-1, 3)
            self.net_moment= np.linalg.norm([sum(_) for _ in zip(*self.moments_cartesian)])
            if any([np.linalg.norm(i)> 1e-5 for i in self.moments]) :
                pass
            else:
                self.moments = None




        self.elements = list(self.elements)

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
            AtomicSite(pos, spin, occ, elem)
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
                i for i, m in enumerate(self.moments)
                if np.linalg.norm(m) > self.tol.moment
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
            return transform_moments(self.moments, self.lattice_factors, inverse=False)
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
                        if np.allclose(moments_cartesian[g], moments_cartesian[index],atol=self.tol.moment):

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
            prim_lat, prim_pos, prim_types = sc(
            L0_nonmagnetic_cell, symprec=self.tol.space, to_primitive=True, no_idealize=True
        )
        except:
            raise ValueError("Spglib failed to find the primitive cell. Maybe the tolerance is too large or the structure is not valid.")

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
            spin_setting="cartesian"
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
            new_cell = change_cell_settings(self.to_spglib(mag=False), matrix, shift)
        else:
            new_cell = change_cell_settings(self.to_spglib(mag=True), matrix, shift)

        new_lattice, new_positions, new_types, new_moments = new_cell

        new_occupancies = [self.atom_types_to_occupancies[t] for t in new_types]
        new_elements = [self.atom_types_to_symbol[t] for t in new_types]

        if self.moments is None:
            final_moments = None
            new_spin_setting = None
        else:
            final_moments = np.asarray(new_moments, dtype=float)
            new_spin_setting = self.spin_setting
        return CrystalCell(
            lattice=new_lattice,
            positions=new_positions,
            occupancies=new_occupancies,
            elements=new_elements,
            moments=final_moments,
            spin_setting=new_spin_setting
        )

    def transform_spin(self,transform_matrix,setting):

        new_cell = CrystalCell(
            lattice=self.lattice,
            positions=self.positions,
            occupancies=self.occupancies,
            elements=self.elements,
            moments=[transform_matrix@ i for i in self.moments],
            spin_setting=setting
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
        std_lattice , std_rotation = standardize_lattice(cell[0])
        lattice = '\n'.join(' '.join(map(str, i.round(6))) for i in std_lattice)
        species = ' '.join(atom_name[1:])
        atom_number = ' '.join(map(str, count[1:]))
        cartesian = 'direct'
        positions = '\n'.join(' '.join([f'{v:.5f}' for v in i]) for i in cell[1])
        magmom = '# MAGMOM=' + ' '.join(' '.join(map(lambda x: str(round(x, 4)), std_rotation @ i)) for i in cell[3])
        return '\n'.join([information, scale, lattice, species, atom_number, cartesian, positions, magmom])