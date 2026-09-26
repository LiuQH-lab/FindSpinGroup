import numpy as np

from findspingroup.structure.group import solve_spin_constraint_from_stacked


def _coefficient(text: str) -> float:
    return float(text.removesuffix("*Sz"))


def test_near_collinear_c2v_constraint_keeps_spin_axis():
    spin_rotations = [
        np.eye(3),
        np.array(
            [
                [-0.9999, 0.0145, 0.0],
                [0.0145, 0.9999, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.9999, -0.0073, 0.0072],
                [-0.0073, -0.0107, 0.9999],
                [0.0072, 0.9999, 0.0108],
            ]
        ),
        np.array(
            [
                [-0.9999, 0.0078, 0.0079],
                [0.0078, -0.0118, 0.9999],
                [0.0079, 0.9999, 0.0118],
            ]
        ),
    ]
    stacked = np.vstack([rotation - np.eye(3) for rotation in spin_rotations])

    spin_splitting, constraint = solve_spin_constraint_from_stacked(stacked)

    assert spin_splitting == "spin splitting"
    assert constraint[2] == "Sz"
    assert np.isclose(_coefficient(constraint[0]), 0.00749, atol=1e-5)
    assert np.isclose(_coefficient(constraint[1]), 0.98876, atol=1e-5)


def test_rank_one_constraint_uses_stable_coordinate_parameters():
    stacked = np.array([[1.0, 0.0, 0.0]])

    spin_splitting, constraint = solve_spin_constraint_from_stacked(stacked)

    assert spin_splitting == "spin splitting"
    assert constraint == ["0", "Sy", "Sz"]


def test_full_rank_constraint_forbids_spin_polarization():
    stacked = np.eye(3)

    spin_splitting, constraint = solve_spin_constraint_from_stacked(stacked)

    assert spin_splitting == "no spin splitting"
    assert constraint == ["0", "0", "0"]


def test_near_collinear_fim_path_and_exact_query_share_the_operation_budget():
    from findspingroup import find_spin_group

    result = find_spin_group("tests/testset/mcif_241130_no2186/0.270_Tb2MnNiO6.mcif")
    assert result.conf == "Collinear"
    assert result.magnetic_phase == "FiM"
    for kpoint in ([0.,0.,0.], [.173,.319,.431]):
        query = result.analyze_kpoint_spin_polarization(kpoint)
        assert query["without_soc"]["dimension"] == 1
        audit = query.audit["without_soc"]["numerical_audit"]
        assert audit["max_operation_residual"] <= audit["threshold"]
    assert "\u0393 ***^^^" in result.KPOINTS
    assert "| GP ***^^^" in result.KPOINTS
