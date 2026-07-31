import numpy as np

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
