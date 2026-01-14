from __future__ import annotations
import re
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
import string
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import scipy.cluster
import collections



if TYPE_CHECKING:
    from typing import Any, Literal, Self

    from numpy.typing import ArrayLike, NDArray


    LatticeType = Literal[
        "cubic",
        "hexagonal",
        "monoclinic",
        "orthorhombic",
        "rhombohedral",
        "tetragonal",
        "triclinic",
    ]

logger = logging.getLogger(__name__)




class SymmOp(list):
    """A symmetry operation in Cartesian space. Consists of a rotation plus a
    translation. Implementation is as an affine transformation matrix of rank 4
    for efficiency. Read: https://wikipedia.org/wiki/Affine_transformation.

    Attributes:
        affine_matrix (NDArray): A 4x4 array representing the symmetry operation.
    """

    def __init__(
        self,
        affine_transformation_matrix: ArrayLike,
        tol: float = 0.01,
    ) -> None:
        """Initialize the SymmOp from a 4x4 affine transformation matrix.
        In general, this constructor should not be used unless you are
        transferring rotations. Use the static constructors instead to
        generate a SymmOp from proper rotations and translation.

        Args:
            affine_transformation_matrix (4x4 array): Representing an
                affine transformation.
            tol (float): Tolerance for determining if matrices are equal. Defaults to 0.01.

        Raises:
            ValueError: if matrix is not 4x4.
        """
        affine_transformation_matrix = np.asarray(affine_transformation_matrix)
        shape = affine_transformation_matrix.shape
        if shape != (4, 4):
            raise ValueError(f"Affine Matrix must be a 4x4 numpy array, got {shape=}")
        self.affine_matrix = affine_transformation_matrix
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self.affine_matrix, other.affine_matrix, atol=self.tol)

    def __hash__(self) -> int:
        return 7

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.affine_matrix=})"

    def __str__(self) -> str:
        return "\n".join(
            [
                "Rot:",
                str(self.affine_matrix[:3][:, :3]),
                "tau",
                str(self.affine_matrix[:3][:, 3]),
            ]
        )

    def __mul__(self, other) -> Self:
        """Get a new SymmOp which is equivalent to apply the "other" SymmOp
        followed by this one.
        """
        return type(self)(np.dot(self.affine_matrix, other.affine_matrix))

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation_matrix: ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        translation_vec: ArrayLike = (0, 0, 0),
        tol: float = 0.1,
    ) -> Self:
        """Create a symmetry operation from a rotation matrix and a translation
        vector.

        Args:
            rotation_matrix (3x3 array): Rotation matrix.
            translation_vec (3x1 array): Translation vector.
            tol (float): Tolerance to determine if rotation matrix is valid.

        Returns:
            SymmOp object
        """
        rotation_matrix = np.asarray(rotation_matrix)
        translation_vec = np.asarray(translation_vec)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation Matrix must be a 3x3 numpy array.")
        if translation_vec.shape != (3,):
            raise ValueError("Translation vector must be a rank 1 numpy array with 3 elements.")

        affine_matrix = np.eye(4)
        affine_matrix[:3][:, :3] = rotation_matrix
        affine_matrix[:3][:, 3] = translation_vec
        return cls(affine_matrix, tol)

    def operate(self, point: ArrayLike) -> NDArray:
        """Apply the operation on a point.

        Args:
            point: Cartesian coordinate.

        Returns:
            Coordinates of point after operation.
        """
        affine_point = np.asarray([*point, 1])
        return np.dot(self.affine_matrix, affine_point)[:3]

    def operate_multi(self, points: ArrayLike) -> NDArray:
        """Apply the operation on a list of points.

        Args:
            points: List of Cartesian coordinates

        Returns:
            Numpy array of coordinates after operation
        """
        points = np.asarray(points)
        affine_points = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)
        return np.inner(affine_points, self.affine_matrix)[..., :-1]

    def apply_rotation_only(self, vector: ArrayLike) -> NDArray:
        """Vectors should only be operated by the rotation matrix and not the
        translation vector.

        Args:
            vector (3x1 array): A vector.
        """
        return np.dot(self.rotation_matrix, vector)

    def transform_tensor(self, tensor: NDArray) -> NDArray:
        """Apply rotation portion to a tensor. Note that tensor has to be in
        full form, not the Voigt form.

        Args:
            tensor (numpy array): A rank n tensor

        Returns:
            Transformed tensor.
        """
        dim = tensor.shape
        rank = len(dim)
        if any(val != 3 for val in dim):
            raise ValueError("Some dimension in tensor is not 3.")

        # Build einstein sum string
        lc = string.ascii_lowercase
        indices = lc[:rank], lc[rank : 2 * rank]
        einsum_string = ",".join(a + i for a, i in zip(*indices, strict=True))
        einsum_string += f",{indices[::-1][0]}->{indices[::-1][1]}"
        einsum_args = [self.rotation_matrix] * rank + [tensor]

        return np.einsum(einsum_string, *einsum_args)

    def are_symmetrically_related(
        self,
        point_a: ArrayLike,
        point_b: ArrayLike,
        tol: float = 0.001,
    ) -> bool:
        """Check if two points are symmetrically related.

        Args:
            point_a (3x1 array): First point.
            point_b (3x1 array): Second point.
            tol (float): Absolute tolerance for checking distance. Defaults to 0.001.

        Returns:
            bool: True if self.operate(point_a) == point_b or vice versa.
        """
        return any(np.allclose(self.operate(p1), p2, atol=tol) for p1, p2 in [(point_a, point_b), (point_b, point_a)])

    def are_symmetrically_related_vectors(
        self,
        from_a: ArrayLike,
        to_a: ArrayLike,
        r_a: ArrayLike,
        from_b: ArrayLike,
        to_b: ArrayLike,
        r_b: ArrayLike,
        tol: float = 0.001,
    ) -> tuple[bool, bool]:
        """Check if two vectors, or rather two vectors that connect two points
        each are symmetrically related. r_a and r_b give the change of unit
        cells. Two vectors are also considered symmetrically equivalent if starting
        and end point are exchanged.

        Args:
            from_a (3x1 array): Starting point of the first vector.
            to_a (3x1 array): Ending point of the first vector.
            from_b (3x1 array): Starting point of the second vector.
            to_b (3x1 array): Ending point of the second vector.
            r_a (3x1 array): Change of unit cell of the first vector.
            r_b (3x1 array): Change of unit cell of the second vector.
            tol (float): Absolute tolerance for checking distance.

        Returns:
            tuple[bool, bool]: First bool indicates if the vectors are related,
                the second if the vectors are related but the starting and end point
                are exchanged.
        """
        from_c = self.operate(from_a)
        to_c = self.operate(to_a)

        floored = np.floor([from_c, to_c])
        is_too_close = np.abs([from_c, to_c] - floored) > 1 - tol
        floored[is_too_close] += 1

        r_c = self.apply_rotation_only(r_a) - floored[0] + floored[1]
        from_c %= 1
        to_c %= 1

        if np.allclose(from_b, from_c, atol=tol) and np.allclose(to_b, to_c) and np.allclose(r_b, r_c, atol=tol):
            return True, False
        if np.allclose(to_b, from_c, atol=tol) and np.allclose(from_b, to_c) and np.allclose(r_b, -r_c, atol=tol):
            return True, True
        return False, False

    @property
    def rotation_matrix(self) -> NDArray:
        """A 3x3 numpy.array representing the rotation matrix."""
        return self.affine_matrix[:3][:, :3]

    @property
    def translation_vector(self) -> NDArray:
        """A rank 1 numpy.array of dim 3 representing the translation vector."""
        return self.affine_matrix[:3][:, 3]

    @property
    def inverse(self) -> Self:
        """Inverse of transformation."""
        new_instance = copy.deepcopy(self)
        new_instance.affine_matrix = np.linalg.inv(self.affine_matrix)
        return new_instance

    @staticmethod
    def from_axis_angle_and_translation(
        axis: ArrayLike,
        angle: float,
        angle_in_radians: bool = False,
        translation_vec: ArrayLike = (0, 0, 0),
    ) -> SymmOp:
        """Generate a SymmOp for a rotation about a given axis plus translation.

        Args:
            axis: The axis of rotation in Cartesian space. For example,
                [1, 0, 0]indicates rotation about x-axis.
            angle (float): Angle of rotation.
            angle_in_radians (bool): Set to True if angles are given in
                radians. Or else, units of degrees are assumed.
            translation_vec: A translation vector. Defaults to zero.

        Returns:
            SymmOp for a rotation about given axis and translation.
        """
        if isinstance(axis, tuple | list):
            axis = np.array(axis)

        vec = np.asarray(translation_vec)

        ang = angle if angle_in_radians else angle * np.pi / 180
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        unit_vec = axis / np.linalg.norm(axis)
        rot_mat = np.zeros((3, 3))
        rot_mat[0, 0] = cos_a + unit_vec[0] ** 2 * (1 - cos_a)
        rot_mat[0, 1] = unit_vec[0] * unit_vec[1] * (1 - cos_a) - unit_vec[2] * sin_a
        rot_mat[0, 2] = unit_vec[0] * unit_vec[2] * (1 - cos_a) + unit_vec[1] * sin_a
        rot_mat[1, 0] = unit_vec[0] * unit_vec[1] * (1 - cos_a) + unit_vec[2] * sin_a
        rot_mat[1, 1] = cos_a + unit_vec[1] ** 2 * (1 - cos_a)
        rot_mat[1, 2] = unit_vec[1] * unit_vec[2] * (1 - cos_a) - unit_vec[0] * sin_a
        rot_mat[2, 0] = unit_vec[0] * unit_vec[2] * (1 - cos_a) - unit_vec[1] * sin_a
        rot_mat[2, 1] = unit_vec[1] * unit_vec[2] * (1 - cos_a) + unit_vec[0] * sin_a
        rot_mat[2, 2] = cos_a + unit_vec[2] ** 2 * (1 - cos_a)

        return SymmOp.from_rotation_and_translation(rot_mat, vec)

    @staticmethod
    def from_origin_axis_angle(
        origin: ArrayLike,
        axis: ArrayLike,
        angle: float,
        angle_in_radians: bool = False,
    ) -> SymmOp:
        """Generate a SymmOp for a rotation about a given axis through an
        origin.

        Args:
            origin (3x1 array): The origin which the axis passes through.
            axis (3x1 array): The axis of rotation in Cartesian space. For
                example, [1, 0, 0]indicates rotation about x-axis.
            angle (float): Angle of rotation.
            angle_in_radians (bool): Set to True if angles are given in
                radians. Or else, units of degrees are assumed.

        Returns:
            SymmOp.
        """
        theta = angle if angle_in_radians else angle * np.pi / 180
        a, b, c = origin
        ax_u, ax_v, ax_w = axis
        # Set some intermediate values.
        u2, v2, w2 = ax_u * ax_u, ax_v * ax_v, ax_w * ax_w
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        l2 = u2 + v2 + w2
        lsqrt = math.sqrt(l2)

        # Build the matrix entries element by element.
        m11 = (u2 + (v2 + w2) * cos_t) / l2
        m12 = (ax_u * ax_v * (1 - cos_t) - ax_w * lsqrt * sin_t) / l2
        m13 = (ax_u * ax_w * (1 - cos_t) + ax_v * lsqrt * sin_t) / l2
        m14 = (
            a * (v2 + w2)
            - ax_u * (b * ax_v + c * ax_w)
            + (ax_u * (b * ax_v + c * ax_w) - a * (v2 + w2)) * cos_t
            + (b * ax_w - c * ax_v) * lsqrt * sin_t
        ) / l2

        m21 = (ax_u * ax_v * (1 - cos_t) + ax_w * lsqrt * sin_t) / l2
        m22 = (v2 + (u2 + w2) * cos_t) / l2
        m23 = (ax_v * ax_w * (1 - cos_t) - ax_u * lsqrt * sin_t) / l2
        m24 = (
            b * (u2 + w2)
            - ax_v * (a * ax_u + c * ax_w)
            + (ax_v * (a * ax_u + c * ax_w) - b * (u2 + w2)) * cos_t
            + (c * ax_u - a * ax_w) * lsqrt * sin_t
        ) / l2

        m31 = (ax_u * ax_w * (1 - cos_t) - ax_v * lsqrt * sin_t) / l2
        m32 = (ax_v * ax_w * (1 - cos_t) + ax_u * lsqrt * sin_t) / l2
        m33 = (w2 + (u2 + v2) * cos_t) / l2
        m34 = (
            c * (u2 + v2)
            - ax_w * (a * ax_u + b * ax_v)
            + (ax_w * (a * ax_u + b * ax_v) - c * (u2 + v2)) * cos_t
            + (a * ax_v - b * ax_u) * lsqrt * sin_t
        ) / l2

        return SymmOp(
            [
                [m11, m12, m13, m14],
                [m21, m22, m23, m24],
                [m31, m32, m33, m34],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def reflection(normal: ArrayLike, origin: ArrayLike = (0, 0, 0)) -> SymmOp:
        """Get reflection symmetry operation.

        Args:
            normal (3x1 array): Vector of the normal to the plane of
                reflection.
            origin (3x1 array): A point in which the mirror plane passes
                through.

        Returns:
            SymmOp for the reflection about the plane
        """
        # Normalize the normal vector first.
        normal = np.array(normal, dtype=float) / np.linalg.norm(normal)

        u, v, w = normal

        translation = np.eye(4)
        translation[:3, 3] = -np.asarray(origin)

        xx = 1 - 2 * u**2
        yy = 1 - 2 * v**2
        zz = 1 - 2 * w**2
        xy = -2 * u * v
        xz = -2 * u * w
        yz = -2 * v * w
        mirror_mat = [[xx, xy, xz, 0], [xy, yy, yz, 0], [xz, yz, zz, 0], [0, 0, 0, 1]]

        if np.linalg.norm(origin) > 1e-6:
            mirror_mat = np.dot(np.linalg.inv(translation), np.dot(mirror_mat, translation))
        return SymmOp(mirror_mat)

    @staticmethod
    def inversion(origin: ArrayLike = (0, 0, 0)) -> SymmOp:
        """Inversion symmetry operation about axis.

        Args:
            origin (3x1 array): Origin of the inversion operation. Defaults
                to [0, 0, 0].

        Returns:
            SymmOp representing an inversion operation about the origin.
        """
        mat = -np.eye(4)
        mat[3, 3] = 1
        mat[:3, 3] = 2 * np.asarray(origin)
        return SymmOp(mat)

    @staticmethod
    def rotoreflection(axis: ArrayLike, angle: float, origin: ArrayLike = (0, 0, 0)) -> SymmOp:
        """Get a roto-reflection symmetry operation.

        Args:
            axis (3x1 array): Axis of rotation / mirror normal
            angle (float): Angle in degrees
            origin (3x1 array): Point left invariant by roto-reflection.
                Defaults to (0, 0, 0).

        Returns:
            Roto-reflection operation
        """
        rot = SymmOp.from_origin_axis_angle(origin, axis, angle)
        refl = SymmOp.reflection(axis, origin)
        matrix = np.dot(rot.affine_matrix, refl.affine_matrix)
        return SymmOp(matrix)

    def as_dict(self) -> dict[str, Any]:
        """MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "matrix": self.affine_matrix.tolist(),
            "tolerance": self.tol,
        }

    def as_xyz_str(self) -> str:
        """Get a string of the form 'x, y, z', '-x, -y, z', '-y+1/2, x+1/2, z+1/2', etc.
        Only works for integer rotation matrices.
        """
        # Check for invalid rotation matrix
        if not np.allclose(self.rotation_matrix, np.round(self.rotation_matrix)):
            warnings.warn("Rotation matrix should be integer", stacklevel=2)

        return transformation_to_string(
            self.rotation_matrix,
            translation_vec=self.translation_vector,
            delim=", ",
        )

    @classmethod
    def from_xyz_str(cls, xyz_str: str) -> Self:
        """
        Args:
            xyz_str (str): "x, y, z", "-x, -y, z", "-2y+1/2, 3x+1/2, z-y+1/2", etc.

        Returns:
            SymmOp
        """
        rot_matrix: NDArray = np.zeros((3, 3))
        trans: NDArray = np.zeros(3)
        tokens: list[str] = xyz_str.strip().replace(" ", "").lower().split(",")
        re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
        re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")

        for idx, tok in enumerate(tokens):
            # Build the rotation matrix
            for match in re_rot.finditer(tok):
                factor = -1.0 if match[1] == "-" else 1.0
                if match[2] != "":
                    factor *= float(match[2]) / float(match[3]) if match[3] != "" else float(match[2])
                j = ord(match[4]) - 120
                rot_matrix[idx, j] = factor

            # Build the translation vector
            for match in re_trans.finditer(tok):
                factor = -1 if match[1] == "-" else 1
                num = float(match[2]) / float(match[3]) if match[3] != "" else float(match[2])
                trans[idx] = num * factor

        return cls.from_rotation_and_translation(rot_matrix, trans)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct: dict.

        Returns:
            SymmOp from dict representation.
        """
        return cls(dct["matrix"], dct["tolerance"])

class Site(collections.abc.Hashable):
    """A generalized *non-periodic* site. This is essentially a composition
    at a point in space, with some optional properties associated with it. A
    Composition is used to represent the atoms and occupancy, which allows for
    disordered site representation. Coords are given in standard Cartesian
    coordinates.
    """

    position_atol = 1e-5

    def __init__(
        self,
        species,
        coords: ArrayLike,
        properties: dict | None = None,
        label: str | None = None,

    ) -> None:
        """Create a non-periodic Site.

        Args:
            species: Species on the site. Can be:
                i.  A Composition-type object (preferred)
                ii. An element / species specified either as a string
                    symbols, e.g. "Li", "Fe2+", "P" or atomic numbers,
                    e.g. 3, 56, or actual Element or Species objects.
                iii.Dict of elements/species and occupancies, e.g.
                    {"Fe" : 0.5, "Mn":0.5}. This allows the setup of
                    disordered structures.
            coords (ArrayLike): Cartesian coordinates of site.
            properties (dict): Properties associated with the site, e.g.
                {"magmom": 5}. Defaults to None.
            label (str): Label for the site. Defaults to None.
            skip_checks (bool): Whether to ignore all the usual checks and just
                create the site. Use this if the Site is created in a controlled
                manner and speed is desired.
        """


        self._species = species
        self.coords: np.ndarray = coords
        self.properties: dict = properties or {}
        self._label = label


    def __contains__(self, el) -> bool:
        return el in self.species

    def __repr__(self) -> str:
        name = self.species_string

        if self.label != name:
            name = f"{self.label} ({name})"

        return f"Site: {name} ({self.coords[0]:.4f}, {self.coords[1]:.4f}, {self.coords[2]:.4f})"

    def __str__(self) -> str:
        return f"{self.coords} {self.species_string}"
    def __hash__(self) -> int:
        """Use the composition hash for now."""
        return hash(self.coords)

    @property
    def species(self) :
        """The species on the site as a composition, e.g. Fe0.5Mn0.5."""
        return self._species


    @property
    def label(self) -> str:
        """Site label."""
        return self._label if self._label is not None else self.species_string

    @label.setter
    def label(self, label: str | None) -> None:
        self._label = label


    @property
    def species_string(self) -> str:
        """String representation of species on the site."""

        return self.species


class IMolecule:
    """Basic immutable Molecule object without periodicity. Essentially a
    sequence of sites. IMolecule is made to be immutable so that they can
    function as keys in a dict. For a mutable object, use the Molecule class.

    Molecule extends Sequence and Hashable, which means that in many cases,
    it can be used like any Python sequence. Iterating through a molecule is
    equivalent to going through the sites in sequence.
    """

    def __init__(
        self,
        species,
        coords: Sequence[ArrayLike],
        charge: float = 0.0,
        spin_multiplicity: int | None = None,
        validate_proximity: bool = False,
        site_properties: dict | None = None,
        labels: Sequence[str | None] | None = None,
        charge_spin_check: bool = True,
        properties: dict | None = None,
    ) -> None:
        """Create a IMolecule.

        Args:
            species: list of atomic species. Possible kinds of input include a
                list of dict of elements/species and occupancies, a List of
                elements/specie specified as actual Element/Species, Strings
                ("Fe", "Fe2+") or atomic numbers (1,56).
            coords (3x1 array): list of Cartesian coordinates of each species.
            charge (float): Charge for the molecule. Defaults to 0.
            spin_multiplicity (int): Spin multiplicity for molecule.
                Defaults to None, which means that the spin multiplicity is
                set to 1 if the molecule has no unpaired electrons and to 2
                if there are unpaired electrons.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 1 Ang apart. Defaults to False.
            site_properties (dict): Properties associated with the sites as
                a dict of sequences, e.g. {"magmom":[5,5,5,5]}. The
                sequences have to be the same length as the atomic species
                and fractional_coords. Defaults to None for no properties.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.
            charge_spin_check (bool): Whether to check that the charge and
                spin multiplicity are compatible with each other. Defaults
                to True.
            properties (dict): dictionary containing properties associated
                with the whole molecule.
        """
        self._species = species

        self._charge = charge
        self._charge_spin_check = charge_spin_check
        self.properties = properties or {}
        self._spin_multiplicity = spin_multiplicity
        sites: list[Site] = []
        for idx in range(len(species)):
            prop = None
            if site_properties:
                prop = {k: v[idx] for k, v in site_properties.items()}
            label = labels[idx] if labels else None
            sites.append(Site(species[idx], coords[idx], properties=prop, label=label))

        self._sites = tuple(sites)
        self._validate_proximity = validate_proximity
        self._labels = labels
    def __hash__(self) -> int:
        """Use the composition hash for now."""
        return hash(self._sites)

    def __repr__(self) -> str:
        return "Molecule Summary\n" + "\n".join(map(repr, self))

    @property
    def center_of_mass(self) -> NDArray:
        """Center of mass of molecule."""
        center = np.zeros(3)
        total_weight: float = 0
        for site in self:
            # wt = site.species.weight
            wt = 1
            # for points, it's the same
            center += site.coords * wt
            total_weight += wt
        return center / total_weight

    @property
    def site_properties(self) -> dict[str, Sequence]:
        """The site properties as a dict of sequences.
        E.g. {"magmom": (5, -5), "charge": (-4, 4)}.
        """
        prop_keys: set[str] = set()
        for site in self:
            prop_keys.update(site.properties)

        return {key: [site.properties.get(key) for site in self] for key in prop_keys}

    @property
    def label(self) -> str:
        """Site label."""
        return self._label



    @property
    def cart_coords(self) -> np.ndarray:
        """An np.array of the Cartesian coordinates of sites in the structure."""
        return np.array([site.coords for site in self])

    @property
    def species_and_occu(self):
        """List of species and occupancies at each site of the structure."""
        return [site.species for site in self]

    @property
    def labels(self) -> list[str | None]:
        """Site labels as a list."""
        return [site.label for site in self]

    @property
    def species(self):
        """The species on the site as a composition, e.g. Fe0.5Mn0.5."""
        return self._species

    def get_centered_molecule(self) -> Self:
        """Get a Molecule centered at the center of mass.

        Returns:
            IMolecule centered with center of mass at origin.
        """
        center = self.center_of_mass
        new_coords = np.array(self.cart_coords) - center
        return type(self)(
            self.species_and_occu,
            new_coords,
            charge=self._charge,
            spin_multiplicity=self._spin_multiplicity,
            site_properties=self.site_properties,
            charge_spin_check=self._charge_spin_check,
            labels=self.labels,
            properties=self.properties,
        )






class Molecule(IMolecule, collections.abc.MutableSequence):
    """Mutable Molecule. It has all the methods in IMolecule,
    and allows a user to perform edits on the molecule.
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        species,
        coords: Sequence[ArrayLike],
        charge: float = 0.0,
        spin_multiplicity: int | None = None,
        validate_proximity: bool = False,
        site_properties: dict | None = None,
        labels: Sequence[str | None] | None = None,
        charge_spin_check: bool = True,
        properties: dict | None = None,
    ) -> None:
        """Create a mutable Molecule.

        Args:
            species: list of atomic species. Possible kinds of input include a
                list of dict of elements/species and occupancies, a List of
                elements/specie specified as actual Element/Species, Strings
                ("Fe", "Fe2+") or atomic numbers (1,56).
            coords (3x1 array): list of Cartesian coordinates of each species.
            charge (float): Charge for the molecule. Defaults to 0.
            spin_multiplicity (int): Spin multiplicity for molecule.
                Defaults to None, which means that the spin multiplicity is
                set to 1 if the molecule has no unpaired electrons and to 2
                if there are unpaired electrons.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 1 Ang apart. Defaults to False.
            site_properties (dict): Properties associated with the sites as
                a dict of sequences, e.g. {"magmom":[5,5,5,5]}. The
                sequences have to be the same length as the atomic species
                and fractional_coords. Defaults to None for no properties.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.
            charge_spin_check (bool): Whether to check that the charge and
                spin multiplicity are compatible with each other. Defaults
                to True.
            properties (dict): dictionary containing properties associated
                with the whole molecule.
        """
        super().__init__(
            species,
            coords,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            validate_proximity=validate_proximity,
            site_properties=site_properties,
            labels=labels,
            charge_spin_check=charge_spin_check,
            properties=properties,
        )
        self._sites: list[Site] = list(self._sites)
        self.sites = self._sites

    def __setitem__(
        self,
        idx: int | slice | Sequence[int] ,
        site: Site | Sequence,
    ) -> None:
        """Modify a site in the molecule.

        Args:
            idx (int, list[int], slice, Species-like): Indices to change. You can
                specify these as an int, a list of int, or a species-like string.
            site (PeriodicSite/Species/Sequence): Three options exist. You can
                provide a Site directly, or for convenience, you can provide
                simply a Species-like string/object, or finally a (Species,
                coords) sequence, e.g. ("Fe", [0.5, 0.5, 0.5]).
        """
        if isinstance(idx, int):
            indices = [idx]

        elif isinstance(idx, str ):
            self.replace_species({idx: site})  # type: ignore[dict-item]
            return

        elif isinstance(idx, slice):
            to_mod = self[idx]
            indices = [idx for idx, site in enumerate(self._sites) if site in to_mod]

        else:
            indices = list(idx)

        for ii in indices:
            if isinstance(site, Site):
                self._sites[ii] = site
            elif isinstance(site, str) or not isinstance(site, collections.abc.Sequence):
                self._sites[ii].species = site  # type: ignore[assignment]
            else:
                self._sites[ii].species = site[0]  # type: ignore[assignment, index]
                if len(site) > 1:
                    self._sites[ii].coords = site[1]  # type: ignore[assignment, index]
                if len(site) > 2:
                    self._sites[ii].properties = site[2]  # type: ignore[assignment, index]

    def __getitem__(self, ind: int | slice):
        return self.sites[ind]  # type: ignore[return-value]

    def __delitem__(self, idx: slice) -> None:
        """Deletes a site from the Structure."""
        self._sites.__delitem__(idx)
    def __len__(self) -> int:
        return len(self._sites)
    def insert(
        self,
        idx: int,
        species,
        coords: ArrayLike,
        validate_proximity: bool = False,
        properties: dict | None = None,
        label: str | None = None,
    ) -> Self:
        """Insert a site to the molecule.

        Args:
            idx (int): Index to insert site
            species: species of inserted site
            coords (3x1 array): coordinates of inserted site
            validate_proximity (bool): Whether to check if inserted site is
                too close to an existing site. Defaults to True.
            properties (dict): Dict of properties for the Site.
            label (str): Label of inserted site

        Returns:
            New molecule with inserted site.
        """
        new_site = Site(species, coords, properties=properties, label=label)
        self.sites.insert(idx, new_site)

        return self

    def append(
        self,
        species,
        coords: ArrayLike,
        validate_proximity: bool = False,
        properties: dict | None = None,
    ) -> Self:
        """Append a site to the molecule.

        Args:
            species: Species of inserted site
            coords: Coordinates of inserted site
            validate_proximity (bool): Whether to check if inserted site is
                too close to an existing site. Defaults to False.
            properties (dict): A dict of properties for the Site.

        Returns:
            New molecule with inserted site.
        """
        return self.insert(
            len(self),
            species,
            coords,
            validate_proximity=validate_proximity,
            properties=properties,
        )






class PointGroupOperations(list):
    """Represents a point group, which is a sequence of symmetry operations.

    Attributes:
        sch_symbol (str): Schoenflies symbol of the point group.
    """

    def __init__(
        self,
        sch_symbol: str,
        operations: Sequence[SymmOp],
        tol: float = 0.1,
    ) -> None:
        """
        Args:
            sch_symbol (str): Schoenflies symbol of the point group.
            operations ([SymmOp]): Initial set of symmetry operations. It is
                sufficient to provide only just enough operations to generate
                the full set of symmetries.
            tol (float): Tolerance to generate the full set of symmetry
                operations.
        """
        self.sch_symbol = sch_symbol
        super().__init__(generate_full_symmops(operations, tol))

    def __repr__(self) -> str:
        return self.sch_symbol


class PointGroupAnalyzer:
    """A class to analyze the point group of a molecule.

    The general outline of the algorithm is as follows:

    1. Center the molecule around its center of mass.
    2. Compute the inertia tensor and the eigenvalues and eigenvectors.
    3. Handle the symmetry detection based on eigenvalues.

        a. Linear molecules have one zero eigenvalue. Possible symmetry
           operations are C*v or D*v
        b. Asymmetric top molecules have all different eigenvalues. The
           maximum rotational symmetry in such molecules is 2
        c. Symmetric top molecules have 1 unique eigenvalue, which gives a
           unique rotation axis. All axial point groups are possible
           except the cubic groups (T & O) and I.
        d. Spherical top molecules have all three eigenvalues equal. They
           have the rare T, O or I point groups.

    Attribute:
        sch_symbol (str): Schoenflies symbol of the detected point group.
    """

    inversion_op = SymmOp.inversion()

    def __init__(
        self,
        mol: Molecule,
        tolerance: float = 0.3,
        eigen_tolerance: float = 0.01,
        matrix_tolerance: float = 0.1,
    ) -> None:
        """The default settings are usually sufficient.

        Args:
            mol (Molecule): Molecule to determine point group for.
            tolerance (float): Distance tolerance to consider sites as
                symmetrically equivalent. Defaults to 0.3 Angstrom.
            eigen_tolerance (float): Tolerance to compare eigen values of
                the inertia tensor. Defaults to 0.01.
            matrix_tolerance (float): Tolerance used to generate the full set of
                symmetry operations of the point group.
        """
        mol.append('origin',[0,0,0])
        # add an atom to fix [0,0,0]
        self.mol = mol
        self.centered_mol = mol.get_centered_molecule()
        self.tol = tolerance
        self.eig_tol = eigen_tolerance
        self.mat_tol = matrix_tolerance
        self._analyze()
        if self.sch_symbol in {"C1v", "C1h"}:
            self.sch_symbol: str = "Cs"

    def _analyze(self):
        if len(self.centered_mol) == 1 and np.allclose(self.centered_mol[0].coords,np.array([0,0,0]),atol=0.00001):
            # yutong #
            self.sch_symbol = "Kh"
        else:
            inertia_tensor = np.zeros((3, 3))
            total_inertia = 0
            for site in self.centered_mol:
                c = site.coords
                # wt = site.species.weight
                wt = 1
                # for points, it's the same
                for i in range(3):
                    inertia_tensor[i, i] += wt * (c[(i + 1) % 3] ** 2 + c[(i + 2) % 3] ** 2)
                for i, j in ((0, 1), (1, 2), (0, 2)):
                    inertia_tensor[i, j] += -wt * c[i] * c[j]
                    inertia_tensor[j, i] += -wt * c[j] * c[i]
                total_inertia += wt * np.dot(c, c)

            # Normalize the inertia tensor so that it does not scale with size
            # of the system. This mitigates the problem of choosing a proper
            # comparison tolerance for the eigenvalues.
            inertia_tensor /= total_inertia
            eigvals, eigvecs = np.linalg.eig(inertia_tensor)
            self.principal_axes = eigvecs.T
            self.eigvals = eigvals
            v1, v2, v3 = eigvals
            eig_zero = abs(v1 * v2 * v3) < self.eig_tol
            eig_all_same = abs(v1 - v2) < self.eig_tol and abs(v1 - v3) < self.eig_tol
            eig_all_diff = abs(v1 - v2) > self.eig_tol and abs(v1 - v3) > self.eig_tol and abs(v2 - v3) > self.eig_tol
            # print(self.eigvals)
            self.rot_sym: list = []
            self.symmops: list[SymmOp] = [SymmOp(np.eye(4))]
            if eig_zero:
                logger.debug("Linear molecule detected")
                self._proc_linear()
            elif eig_all_same:
                logger.debug("Spherical top molecule detected")
                self._proc_sph_top()
            elif eig_all_diff:
                logger.debug("Asymmetric top molecule detected")
                self._proc_asym_top()
            else:
                logger.debug("Symmetric top molecule detected")
                self._proc_sym_top()

    def _proc_linear(self) -> None:
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.sch_symbol = "D*h"
            self.symmops.append(PointGroupAnalyzer.inversion_op)
        else:
            self.sch_symbol = "C*v"

    def _proc_asym_top(self) -> None:
        """Handles asymmetric top molecules, which cannot contain rotational symmetry
        larger than 2.
        """
        self._check_R2_axes_asym()
        if len(self.rot_sym) == 0:
            logger.debug("No rotation symmetries detected.")
            self._proc_no_rot_sym()
        elif len(self.rot_sym) == 3:
            logger.debug("Dihedral group detected.")
            self._proc_dihedral()
        else:
            logger.debug("Cyclic group detected.")
            self._proc_cyclic()

    def _proc_sym_top(self) -> None:
        """Handles symmetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.

        More complex handling required to look for R2 axes perpendicular to this unique
        axis.
        """
        if abs(self.eigvals[0] - self.eigvals[1]) < self.eig_tol:
            ind = 2
        elif abs(self.eigvals[1] - self.eigvals[2]) < self.eig_tol:
            ind = 0
        else:
            ind = 1
        logger.debug(f"Eigenvalues = {self.eigvals}.")
        unique_axis = self.principal_axes[ind]
        self._check_rot_sym(unique_axis)
        logger.debug(f"Rotation symmetries = {self.rot_sym}")
        if len(self.rot_sym) > 0:
            self._check_perpendicular_r2_axis(unique_axis)

        if len(self.rot_sym) >= 2:
            self._proc_dihedral()
            # print('here_proc_dihedral')
        elif len(self.rot_sym) == 1 and self.rot_sym[0][1] > 1:
            self._proc_cyclic()
        else:
            self._proc_no_rot_sym()

    def _proc_no_rot_sym(self) -> None:
        """Handles molecules with no rotational symmetry.

        Only possible point groups are C1, Cs and Ci.
        """
        self.sch_symbol = "C1"
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.sch_symbol = "Ci"
            self.symmops.append(PointGroupAnalyzer.inversion_op)
        else:
            for v in self.principal_axes:
                mirror_type = self._find_mirror(v)
                if mirror_type != "":
                    self.sch_symbol = "Cs"
                    break

    def _proc_cyclic(self) -> None:
        """Handles cyclic group molecules."""
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f"C{rot}"
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif mirror_type == "v":
            self.sch_symbol += "v"
        elif mirror_type == "" and self.is_valid_op(SymmOp.rotoreflection(main_axis, angle=180 / rot)):
            self.sch_symbol = f"S{2 * rot}"

    def _proc_dihedral(self) -> None:
        """Handles dihedral group molecules, i.e those with intersecting R2 axes and a
        main axis.
        """
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        # print(rot)
        self.sch_symbol = f"D{rot}"
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif mirror_type != "":
            self.sch_symbol += "d"

    def _check_R2_axes_asym(self) -> None:
        """Test for 2-fold rotation along the principal axes.

        Used to handle asymmetric top molecules.
        """
        for v in self.principal_axes:
            op = SymmOp.from_axis_angle_and_translation(v, 180)
            if self.is_valid_op(op):
                self.symmops.append(op)
                self.rot_sym.append((v, 2))

    def _find_mirror(self, axis: NDArray) -> Literal["h", "d", "v", ""]:
        """Looks for mirror symmetry of specified type about axis.

        Possible types are "h" or "vd". Horizontal (h) mirrors are perpendicular to the
        axis while vertical (v) or diagonal (d) mirrors are parallel. v mirrors has atoms
        lying on the mirror plane while d mirrors do not.
        """
        mirror_type: Literal["h", "d", "v", ""] = ""

        # First test whether the axis itself is the normal to a mirror plane.
        if self.is_valid_op(SymmOp.reflection(axis)):
            self.symmops.append(SymmOp.reflection(axis))
            mirror_type = "h"
        else:
            # Iterate through all pairs of atoms to find mirror
            for s1, s2 in itertools.combinations(self.centered_mol, 2):
                if s1.species == s2.species:
                    normal = s1.coords - s2.coords
                    if np.dot(normal, axis) < self.tol:
                        op = SymmOp.reflection(normal)
                        if self.is_valid_op(op):
                            self.symmops.append(op)
                            if len(self.rot_sym) > 1:
                                mirror_type = "d"
                                for v, _ in self.rot_sym:
                                    if np.linalg.norm(v - axis) >= self.tol and np.dot(v, normal) < self.tol:
                                        mirror_type = "v"
                                        break
                            else:
                                mirror_type = "v"
                            break

        return mirror_type

    def _get_smallest_set_not_on_axis(self, axis: NDArray) -> list:
        """Get the smallest list of atoms with the same species and distance from
        origin AND does not lie on the specified axis.

        This maximal set limits the possible rotational symmetry operations, since atoms
        lying on a test axis is irrelevant in testing rotational symmetryOperations.
        """

        def not_on_axis(site):
            v = np.cross(site.coords, axis)
            return np.linalg.norm(v) > 1e-6
            # return np.linalg.norm(v) > self.tol
            # yutong 其实可以算投影 #

        valid_sets = []
        _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
        for test_set in dist_el_sites.values():
            valid_set = list(filter(not_on_axis, test_set))
            if len(valid_set) > 0:
                valid_sets.append(valid_set)

        return min(valid_sets, key=len)

    def _check_rot_sym(self, axis: NDArray) -> int:
        """Determine the rotational symmetry about supplied axis.

        Used only for symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        max_sym = len(min_set)
        for idx in range(max_sym, 0, -1):
            if max_sym % idx != 0:
                continue
            op = SymmOp.from_axis_angle_and_translation(axis, 360 / idx)
            if self.is_valid_op(op):
                self.symmops.append(op)
                self.rot_sym.append((axis, idx))
                return idx
        return 1

    def _check_perpendicular_r2_axis(self, axis: NDArray) -> None | Literal[True]:
        """Check for R2 axes perpendicular to unique axis.

        For handling symmetric top molecules.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        for s1, s2 in itertools.combinations(min_set, 2):
            test_axis = np.cross(s1.coords - s2.coords, axis)
            if np.linalg.norm(test_axis) > self.tol:
                op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                if self.is_valid_op(op):
                    self.symmops.append(op)
                    self.rot_sym.append((test_axis, 2))
                    return True
        return None

    def _proc_sph_top(self) -> None:
        """Handles Spherical Top Molecules, which belongs to the T, O or I point
        groups.
        """
        self._find_spherical_axes()
        if len(self.rot_sym) == 0:
            logger.debug("Accidental spherical top!")
            self._proc_sym_top()
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        if rot < 3:
            logger.debug("Accidental spherical top!")
            self._proc_sym_top()

        elif rot == 3:
            mirror_type = self._find_mirror(main_axis)
            if mirror_type == "":
                self.sch_symbol = "T"
            elif self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = "Th"
            else:
                self.sch_symbol = "Td"

        elif rot == 4:
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = "Oh"
            else:
                self.sch_symbol = "O"

        elif rot == 5:
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = "Ih"
            else:
                self.sch_symbol = "I"

    def _find_spherical_axes(self) -> None:
        """Looks for R5, R4, R3 and R2 axes in spherical top molecules.

        Point group T molecules have only one unique 3-fold and one unique 2-fold axis. O
        molecules have one unique 4, 3 and 2-fold axes. I molecules have a unique 5-fold
        axis.
        """
        rot_present: dict[int, bool] = defaultdict(bool)
        _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
        test_set = min(dist_el_sites.values(), key=len)
        coords = [s.coords for s in test_set]
        for c1, c2, c3 in itertools.combinations(coords, 3):
            for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self.tol:
                        op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                        rot_present[2] = self.is_valid_op(op)
                        if rot_present[2]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, 2))

            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self.tol:
                for r in (3, 4, 5):
                    if not rot_present[r]:
                        op = SymmOp.from_axis_angle_and_translation(test_axis, 360 / r)
                        rot_present[r] = self.is_valid_op(op)
                        if rot_present[r]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, r))
                            break
            if rot_present[2] and rot_present[3] and (rot_present[4] or rot_present[5]):
                break

    def get_pointgroup(self) -> PointGroupOperations:
        """Get a PointGroup object for the molecule."""
        return PointGroupOperations(self.sch_symbol, self.symmops, self.mat_tol)

    def get_symmetry_operations(self) -> Sequence[SymmOp]:
        """Get symmetry operations.

        Returns:
            list[SymmOp]: symmetry operations in Cartesian coord.
        """
        return generate_full_symmops(self.symmops, self.tol)


    def is_valid_op(self, symm_op: SymmOp) -> bool:
        """Check if a particular symmetry operation is a valid symmetry operation for a
        molecule, i.e., the operation maps all atoms to another equivalent atom.

        Args:
            symm_op (SymmOp): Symmetry operation to test.

        Returns:
            bool: True if SymmOp is valid for Molecule.
        """
        coords = self.centered_mol.cart_coords
        for site in self.centered_mol:
            coord = symm_op.operate(site.coords)
            ind = find_in_coord_list(coords, coord, self.tol)

            if not (any([self.mol[i].species == site.species for i in ind])):
                # 很可能出现一个点有多个原子的情况，只要有一个对就可以
            # if not (len(ind) == 1 and self.centered_mol[ind[0]].species == site.species):
            # yutong #
                return False
        return True




def cluster_sites(
    mol: Molecule,
    tol: float,
    give_only_index: bool = False,
):
    """Cluster sites based on distance and species type.

    Args:
        mol (Molecule): Molecule **with origin at center of mass**.
        tol (float): Tolerance to use.
        give_only_index (bool): Whether to return only the index of the
            origin site, instead of the site itself. Defaults to False.

    Returns:
        tuple[Site | None, dict]: origin_site is a site at the center
            of mass (None if there are no origin atoms). clustered_sites is a
            dict of {(avg_dist, species_and_occu): [list of sites]}
    """
    # Cluster works for dim > 2 data. We just add a dummy 0 for second
    # coordinate.
    dists: list[list[float]] = [[float(np.linalg.norm(site.coords)), 0] for site in mol]

    f_cluster = scipy.cluster.hierarchy.fclusterdata(dists, tol, criterion="distance")
    clustered_dists: dict[str, list[list[float]]] = defaultdict(list)
    for idx in range(len(mol)):
        clustered_dists[f_cluster[idx]].append(dists[idx])
    avg_dist = {key: np.mean(val) for key, val in clustered_dists.items()}
    clustered_sites = defaultdict(list)
    origin_site = None
    for idx, site in enumerate(mol):
        if avg_dist[f_cluster[idx]] < tol:
            origin_site = idx if give_only_index else site
        elif give_only_index:
            clustered_sites[avg_dist[f_cluster[idx]], site.species].append(idx)
        else:
            clustered_sites[avg_dist[f_cluster[idx]], site.species].append(site)
    return origin_site, clustered_sites


def find_in_coord_list(coord_list, coord, atol: float = 1e-8):
    """Find the indices of matches of a particular coord in a coord_list.

    Args:
        coord_list: List of coords to test
        coord: Specific coordinates
        atol: Absolute tolerance. Defaults to 1e-8. Accepts both scalar and
            array.

    Returns:
        Indices of matches, e.g. [0, 1, 2, 3]. Empty list if not found.
    """
    if len(coord_list) == 0:
        return []
    diff = np.array(coord_list) - np.array(coord)[None, :]

    return np.where([math.sqrt(i[0]**2+i[1]**2+i[2]**2) < atol for i in diff])[0]
    # yt for more physical meaning#

def generate_full_symmops(
    symmops: Sequence[SymmOp],
    tol: float,
) -> Sequence[SymmOp]:
    """Recursive algorithm to permute through all possible combinations of the initially
    supplied symmetry operations to arrive at a complete set of operations mapping a
    single atom to all other equivalent atoms in the point group. This assumes that the
    initial number already uniquely identifies all operations.

    Args:
        symmops (list[SymmOp]): Initial set of symmetry operations.
        tol (float): Tolerance for detecting symmetry.

    Returns:
        list[SymmOp]: Full set of symmetry operations.
    """
    # Uses an algorithm described in:
    # Gregory Butler. Fundamental Algorithms for Permutation Groups.
    # Lecture Notes in Computer Science (Book 559). Springer, 1991. page 15
    identity = np.eye(4)
    generators = [op.affine_matrix for op in symmops if not np.allclose(op.affine_matrix, identity)]
    if not generators:
        # C1 symmetry breaks assumptions in the algorithm afterwards
        return [SymmOp(identity)]

    full = list(generators)

    for g in full:
        for s in generators:
            op = np.dot(g, s)
            d = np.abs(full - op) < tol
            if not np.any(np.all(np.all(d, axis=2), axis=1)):
                full.append(op)
            if len(full) > 1000:
                warnings.warn(
                    f"{len(full)} matrices have been generated. The tol may be too small. Please terminate"
                    " and rerun with a different tolerance.",
                    stacklevel=2,
                )

    d = np.abs(full - identity) < tol
    if not np.any(np.all(np.all(d, axis=2), axis=1)):
        full.append(identity)
    return [SymmOp(op) for op in full]


def transformation_to_string(
    matrix: ArrayLike,
    translation_vec = (0, 0, 0),
    components: tuple[str, str, str] = ("x", "y", "z"),
    c: str = "",
    delim: str = ",",
) -> str:
    """Convenience method. Given matrix returns string, e.g. x+2y+1/4.

    Args:
        matrix (ArrayLike): A 3x3 matrix.
        translation_vec (Vector3D): The translation vector. Defaults to (0, 0, 0).
        components(tuple[str, str, str]): The components. Either ('x', 'y', 'z') or ('a', 'b', 'c').
            Defaults to ('x', 'y', 'z').
        c (str): An optional additional character to print (used for magmoms). Defaults to "".
        delim (str): A delimiter. Defaults to ",".

    Returns:
        str: xyz string.
    """
    parts = []
    for idx in range(3):
        string = ""
        mat = matrix[idx]
        offset = translation_vec[idx]
        for j, dim in enumerate(components):
            if mat[j] != 0:
                f = Fraction(mat[j]).limit_denominator()
                if string != "" and f >= 0:
                    string += "+"
                if abs(f.numerator) != 1:
                    string += str(f.numerator)
                elif f < 0:
                    string += "-"
                string += c + dim
                if f.denominator != 1:
                    string += f"/{f.denominator}"
        if offset != 0:
            string += ("+" if (offset > 0 and string != "") else "") + str(Fraction(offset).limit_denominator())
        if string == "":
            string += "0"
        parts.append(string)
    return delim.join(parts)
