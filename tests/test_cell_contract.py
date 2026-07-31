import numpy as np
import pytest

from findspingroup.structure import CrystalCell


@pytest.mark.parametrize(
    ("occupancies", "elements", "moments"),
    [
        ([1.0], ["Fe", "Fe"], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        ([1.0, 1.0], ["Fe"], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        ([1.0, 1.0], ["Fe", "Fe"], [[1.0, 0.0, 0.0]]),
    ],
)
def test_crystal_cell_rejects_mismatched_per_site_array_lengths(
    occupancies,
    elements,
    moments,
):
    with pytest.raises(ValueError, match="per-site arrays must have identical lengths"):
        CrystalCell(
            np.eye(3),
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            occupancies,
            elements,
            moments,
        )


def test_crystal_cell_preserves_all_sites_when_lengths_match():
    cell = CrystalCell(
        np.eye(3),
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        [1.0, 1.0],
        ["Fe", "Fe"],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )

    assert len(cell.positions) == 2
    assert len(cell.elements) == 2
    assert len(cell.moments) == 2
