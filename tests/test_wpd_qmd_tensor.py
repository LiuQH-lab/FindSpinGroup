import numpy as np

from spintensor import build_bcd_extra_constraints, solve_bcd, solve_imd, solve_qmd

from findspingroup import find_spin_group
from findspingroup.tensor_constraints import solve_wpd_qmd


def _projector(basis: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((basis.shape[0], basis.shape[0]))
    u, singular_values, _vh = np.linalg.svd(basis, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > tol))
    orthonormal = u[:, :rank]
    return orthonormal @ orthonormal.T


def _nullspace(matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > tol))
    return vh[rank:].T


def _representative_t_odd_operations():
    identity = np.eye(3)
    spin_reflection = np.diag([-1.0, 1.0, 1.0])
    spatial_inversion = -np.eye(3)
    return [[identity, identity], [spin_reflection, spatial_inversion]]


def _rotation_z(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_wpd_qmd_has_the_eight_dimensional_mixed_symmetry_space():
    solution = solve_wpd_qmd([[np.eye(3), np.eye(3)]])
    _constraints, basis, _relations, _components = solution

    assert basis.shape == (27, 8)
    for column in basis.T:
        tensor = column.reshape(3, 3, 3)
        assert np.allclose(tensor, tensor.swapaxes(1, 2), atol=1e-10)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert np.isclose(
                        tensor[i, j, k] + tensor[j, k, i] + tensor[k, i, j],
                        0.0,
                        atol=1e-10,
                    )


def test_wpd_qmd_matches_the_cyclic_projection_of_legacy_qmd():
    operations = _representative_t_odd_operations()
    qmd_basis = solve_qmd(operations)[1]
    wpd_basis = solve_wpd_qmd(operations)[1]
    cyclic = build_bcd_extra_constraints(3)

    projected_qmd_basis = qmd_basis @ _nullspace(cyclic @ qmd_basis)

    assert np.allclose(
        _projector(wpd_basis),
        _projector(projected_qmd_basis),
        atol=1e-9,
    )


def test_legacy_qmd_decomposes_into_wpd_and_fully_symmetric_imd_sectors():
    operations = _representative_t_odd_operations()
    qmd_dimension = solve_qmd(operations)[1].shape[1]
    wpd_dimension = solve_wpd_qmd(operations)[1].shape[1]
    imd_dimension = solve_imd(operations)[1].shape[1]

    assert qmd_dimension == wpd_dimension + imd_dimension


def test_wpd_qmd_is_time_odd_while_bcd_is_time_even():
    identity = np.eye(3)
    spin_reflection = np.diag([-1.0, 1.0, 1.0])
    operations = [[identity, identity], [spin_reflection, identity]]

    assert solve_wpd_qmd(operations)[1].shape[1] == 0
    assert solve_bcd(operations)[1].shape[1] == 8


def test_threefold_rotation_constrains_but_does_not_forbid_wpd_qmd():
    identity = np.eye(3)
    c3 = _rotation_z(2.0 * np.pi / 3.0)
    operations = [[identity, identity], [identity, c3], [identity, c3 @ c3]]

    assert solve_wpd_qmd(operations)[1].shape[1] == 2


def test_real_material_qmd_space_is_the_direct_sum_of_wpd_and_symmetric_sectors():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    for prefix in ("", "MSG"):
        qmd = np.asarray(getattr(result, f"{prefix}QMDTensor")["nullspace_basis"])
        wpd = np.asarray(getattr(result, f"{prefix}WPDQMDTensor")["nullspace_basis"])
        symmetric = np.asarray(getattr(result, f"{prefix}IMDTensor")["nullspace_basis"])
        qmd_projector = _projector(qmd)
        wpd_projector = _projector(wpd)
        symmetric_projector = _projector(symmetric)

        assert np.allclose(qmd_projector, wpd_projector + symmetric_projector, atol=1e-9)
        assert np.allclose(wpd_projector @ symmetric_projector, 0.0, atol=1e-9)
