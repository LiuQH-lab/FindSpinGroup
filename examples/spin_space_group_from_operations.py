"""Identify spin space groups directly from operation generators.

Run from the repository root with:

    PYTHONPATH=src python examples/spin_space_group_from_operations.py

There is no implicit real-space convention in the operation-input API. Each
example names the setting used by its real rotations and translations. The
default is ``spin_frame="cartesian"`` and requires a lattice or metric.
Examples whose spin matrices are already written in an oriented setting say so
explicitly.

Translations are already reduced componentwise to the required [0, 1) range.
"""

from math import sqrt

from findspingroup import get_spin_space_group_from_operations


IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
MINUS_IDENTITY = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
C3_ORIENTED = [[0, -1, 0], [1, -1, 0], [0, 0, 1]]
MN3SN_CONVENTION_LATTICE = [
    [5.665, 0, 0],
    [-2.8325, 4.906033912438845, 0],
    [0, 0, 4.531],
]
MN3SN_MAGNETIC_PRIMITIVE_LATTICE = [
    [-5.665, 0, 0],
    [2.8325, -4.906033912438845, 0],
    [0, 0, 4.531],
]


def _mn3sn_nssg_generators():
    """Return the four non-spin-only Mn3Sn generators."""
    # 0.200_Mn3Sn, convention-oriented operation rows 3, 9, 15, and 20.
    return [
        {
            "spin_rotation": C3_ORIENTED,
            "real_rotation": [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": IDENTITY,
            "real_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
            "real_rotation": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
        },
        {
            "spin_rotation": [[1, 0, 0], [1, -1, 0], [0, 0, -1]],
            "real_rotation": [[-1, 0, 0], [-1, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
    ]


def _mn3sn_spin_only_mirror():
    return {
        "spin_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        "real_rotation": IDENTITY,
        "translation": [0, 0, 0],
    }


def identify_mn3sn():
    """Identify Mn3Sn in the SSG-convention oriented setting."""
    generators = _mn3sn_nssg_generators()
    # Convention-oriented operation row 2: coplanar spin-only mirror.
    generators.append(_mn3sn_spin_only_mirror())
    return get_spin_space_group_from_operations(
        generators,
        spin_frame="oriented",
    )


def identify_mn3sn_with_explicit_spin_only_semantics():
    """Supply Mn3Sn spin-only information separately from its nSSG generators."""
    return get_spin_space_group_from_operations(
        _mn3sn_nssg_generators(),
        spin_configuration="coplanar",
        # For coplanar order this is the spin-plane normal, not a spin axis.
        spin_only_direction=[0, 0, 1],
        spin_frame="oriented",
    )


def identify_mn3sn_convention_cartesian():
    """Identify Mn3Sn with convention-cell operations and Cartesian spin."""
    root_three_over_two = sqrt(3) / 2
    generators = [
        {
            "spin_rotation": [
                [-0.5, -root_three_over_two, 0],
                [root_three_over_two, -0.5, 0],
                [0, 0, 1],
            ],
            "real_rotation": [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": IDENTITY,
            "real_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": [
                [0.5, -root_three_over_two, 0],
                [-root_three_over_two, -0.5, 0],
                [0, 0, -1],
            ],
            "real_rotation": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
        },
        {
            "spin_rotation": [
                [0.5, root_three_over_two, 0],
                [root_three_over_two, -0.5, 0],
                [0, 0, -1],
            ],
            "real_rotation": [[-1, 0, 0], [-1, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
        _mn3sn_spin_only_mirror(),
    ]
    return get_spin_space_group_from_operations(
        generators,
        real_space_lattice=MN3SN_CONVENTION_LATTICE,
    )


def identify_mn3sn_magnetic_primitive_oriented():
    """Identify Mn3Sn in its magnetic-primitive oriented setting."""
    generators = _mn3sn_nssg_generators()
    generators.append(_mn3sn_spin_only_mirror())
    return get_spin_space_group_from_operations(
        generators,
        spin_frame="oriented",
        real_space_lattice=MN3SN_MAGNETIC_PRIMITIVE_LATTICE,
    )


def identify_mnte():
    """Return collinear MnTe, OSSG 194.164.1.1.L, from four generators.

    ``ssg.ops`` contains the 96-operation finite representative used by the
    core. The physical collinear group also contains the continuous C-infinity-v
    spin-only component and therefore has infinitely many operations.
    """
    # 0.800_MnTe, convention-oriented nSSG rows 2, 5, 8, and 10.
    generators = [
        {
            "spin_rotation": MINUS_IDENTITY,
            "real_rotation": [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": MINUS_IDENTITY,
            "real_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": IDENTITY,
            "real_rotation": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
        },
        {
            "spin_rotation": MINUS_IDENTITY,
            "real_rotation": [[-1, 0, 0], [-1, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
    ]
    return get_spin_space_group_from_operations(
        generators,
        spin_configuration="collinear",
        # The collinear direction is expressed in the same oriented basis.
        spin_only_direction=[1, 1, 0],
        spin_frame="oriented",
    )


def identify_crse():
    """Return noncoplanar CrSe, OSSG 194.149.3.3, from six generators."""
    # 2.35_CrSe, convention-oriented operation rows 19, 32, 55, 64, 2, and 3.
    generators = [
        {
            "spin_rotation": [[1, -1, 0], [0, -1, 0], [0, 0, -1]],
            "real_rotation": [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": MINUS_IDENTITY,
            "real_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            "real_rotation": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
        },
        {
            "spin_rotation": MINUS_IDENTITY,
            "real_rotation": [[-1, 0, 0], [-1, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0.5],
        },
        {
            "spin_rotation": C3_ORIENTED,
            "real_rotation": IDENTITY,
            "translation": [0, 1 / 3, 0],
        },
        {
            "spin_rotation": C3_ORIENTED,
            "real_rotation": IDENTITY,
            "translation": [1 / 3, 0, 0],
        },
    ]
    # Identity-only spin-only content is inferred as noncoplanar.
    return get_spin_space_group_from_operations(
        generators,
        spin_frame="oriented",
    )


def main():
    examples = {
        "Mn3Sn (spin-only operation included)": identify_mn3sn(),
        "Mn3Sn (spin-only semantics supplied separately)": (
            identify_mn3sn_with_explicit_spin_only_semantics()
        ),
        "Mn3Sn (convention Cartesian spin frame)": (
            identify_mn3sn_convention_cartesian()
        ),
        "Mn3Sn (magnetic-primitive oriented setting)": (
            identify_mn3sn_magnetic_primitive_oriented()
        ),
        "MnTe": identify_mnte(),
        "CrSe": identify_crse(),
    }
    for material, ssg in examples.items():
        operation_count = str(len(ssg.ops))
        if ssg.conf == "Collinear":
            operation_count += (
                " finite representative operations; "
                "full collinear group is infinite (spin-only C-infinity-v)"
            )
        print(
            f"{material}: index={ssg.index}, conf={ssg.conf}, "
            f"operations={operation_count}, G0={ssg.G0_num}, L0={ssg.L0_num}"
        )


if __name__ == "__main__":
    main()
