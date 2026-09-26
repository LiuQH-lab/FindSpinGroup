import numpy as np
import pytest
import spglib

from findspingroup.core.identify_symmetry_from_ops import (
    get_magnetic_space_group_from_operations,
)
from findspingroup.data.MSGMPG_DB import MPG_SYMBOL_TO_NUM, OG_NUM_TO_MPG


def test_og_mpg_labels_match_their_mpg_numbers():
    mismatches = {
        og_number: (
            record["pointgroup_label"],
            record["pointgroup_no"],
            MPG_SYMBOL_TO_NUM.get(record["pointgroup_label"]),
        )
        for og_number, record in OG_NUM_TO_MPG.items()
        if MPG_SYMBOL_TO_NUM.get(record["pointgroup_label"])
        != record["pointgroup_no"]
    }

    assert mismatches == {}


def test_effective_inversion_group_is_identified_as_minus_one():
    identity = np.eye(3)
    zero = np.zeros(3)
    info = get_magnetic_space_group_from_operations(
        [
            [1, identity, zero],
            [1, -identity, zero],
        ]
    )

    assert info["msg_bns_symbol"] == "P-1"
    assert info["mpg_num"] == "2.1.3"
    assert info["mpg_symbol"] == "-1"


@pytest.mark.parametrize("uni_number", [2, 4, 276, 917, 933, 1141, 1283, 1451])
@pytest.mark.parametrize(
    "origin",
    [[0.0, 0.0, 0.0], [0.137, 0.271, 0.0], [0.317219, 0.072831, 0.192731]],
)
def test_msg_identification_preserves_arbitrary_origin(uni_number, origin):
    database = spglib.get_magnetic_symmetry_from_database(uni_number)
    shift = np.asarray(origin)
    operations = [
        [1 - 2 * int(time_reversal), rotation,
         (translation + shift - rotation @ shift) % 1.0]
        for rotation, translation, time_reversal in zip(
            database["rotations"], database["translations"], database["time_reversals"]
        )
    ]

    info = get_magnetic_space_group_from_operations(operations)

    assert info is not None
    assert info["msg_int_num"] == uni_number


def test_inversion_near_origin_does_not_create_overlapping_probe_atoms():
    operations = [
        [1, np.eye(3), np.zeros(3)],
        [1, -np.eye(3), np.array([0.0, 0.0024, -0.0024])],
    ]
    original_translation = operations[1][2].copy()

    info = get_magnetic_space_group_from_operations(operations)

    assert info["msg_bns_symbol"] == "P-1"
    np.testing.assert_array_equal(operations[1][2], original_translation)


@pytest.mark.parametrize("probe", [[0.171587, 0.2775421, 0.7373887], [0.319341, 0.431907, 0.619283]])
def test_msg_identification_does_not_depend_on_a_synthetic_probe(probe):
    translation = (2 * np.asarray(probe) + [0.0, 0.0024, -0.0024]) % 1
    info = get_magnetic_space_group_from_operations(
        [[1, np.eye(3), np.zeros(3)], [1, -np.eye(3), translation]]
    )

    assert info["msg_bns_symbol"] == "P-1"


def test_msg_integer_rotation_conversion_does_not_truncate_roundoff():
    noisy_inversion = -np.eye(3) * (1.0 - 1e-12)
    info = get_magnetic_space_group_from_operations(
        [[1, np.eye(3), np.zeros(3)], [1, noisy_inversion, [0.137, 0.271, 0.0]]]
    )

    assert info["msg_bns_symbol"] == "P-1"
