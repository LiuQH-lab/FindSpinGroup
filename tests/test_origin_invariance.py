"""Material regressions contributed in GitHub issue #36."""

from pathlib import Path

import numpy as np
import pytest

from findspingroup import find_spin_group, find_spin_group_basic


@pytest.mark.parametrize(
    ("material", "index", "bns"),
    [("FeBr3", "189.174.1.1.L", "189.221"),
     ("Mn2P2S3Se3", "157.143.1.1.L", "157.53")],
)
@pytest.mark.parametrize("full", [False, True])
def test_phase_and_ahc_are_invariant_under_origin_shift(tmp_path, material, index, bns, full):
    source = Path(__file__).parent / "data" / "issue36" / f"{material}_2x2_original.vasp"
    lines = source.read_text().splitlines()
    atom_count = sum(map(int, lines[6].split()))
    shift = np.array([0.137, 0.271, 0.0])
    for i in range(8, 8 + atom_count):
        position = (np.asarray(lines[i].split(), dtype=float) + shift) % 1.0
        lines[i] = " ".join(f"{value:.12f}" for value in position)
    shifted = tmp_path / source.name
    shifted.write_text("\n".join(lines) + "\n")

    for path in (source, shifted):
        if full:
            result = find_spin_group(str(path))
            assert result.index == index
            assert result.msg_bns_number == bns
            assert result.magnetic_phase == "AFM(Altermagnet)"
            assert result.ahc_w_soc == "No"
            assert result.AHE_wSOC["is_zero"] is True
        else:
            result = find_spin_group_basic(str(path))
            assert result["index"] == index
            assert result["msg_bns_number"] == bns
            assert result["magnetic_phase"] == "AFM(Altermagnet)"
            assert result["properties"]["ahc_w_soc"] == "No"


@pytest.mark.parametrize(
    ("filename", "bns"),
    [("1.353_SmNiO3", "36.178"), ("1.380_Sr2FeO3Cl", "113.273"),
     ("1.387_Sr2FeO3F", "111.257"), ("2.20_UAs", "134.481")],
)
def test_msg_identification_keeps_probe_collision_regressions(filename, bns):
    result = find_spin_group_basic(f"tests/testset/mcif_241130_no2186/{filename}.mcif")

    assert result["msg_bns_number"] == bns
    assert result["properties"]["ahc_w_soc"] == "No"
