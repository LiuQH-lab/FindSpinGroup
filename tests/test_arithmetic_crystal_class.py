import pytest
import spglib

from findspingroup.core.identify_symmetry_from_ops import (
    get_arithmetic_crystal_class_from_ops,
)


@pytest.mark.parametrize(
    ("space_group_number", "hall_number", "expected_acc"),
    [
        (187, 481, "-6m2P"),
        (188, 482, "-6m2P"),
        (189, 483, "-62mP"),
        (190, 484, "-62mP"),
    ],
)
def test_hexagonal_d3h_acc_retains_space_group_orientation(
    space_group_number,
    hall_number,
    expected_acc,
):
    symmetry = spglib.get_symmetry_from_database(hall_number)
    operations = list(zip(symmetry["rotations"], symmetry["translations"]))

    acc_symbol, _, _, _ = get_arithmetic_crystal_class_from_ops(operations)

    assert spglib.get_spacegroup_type(hall_number).number == space_group_number
    assert acc_symbol == expected_acc
