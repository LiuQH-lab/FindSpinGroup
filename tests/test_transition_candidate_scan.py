import importlib

import numpy as np


symmetry_module = importlib.import_module(
    "findspingroup.core.identify_symmetry_from_ops"
)


def _reference_search(basis_matrices, linear_system, residual_tol):
    best_candidate = None
    best_key = None

    def consider(candidate_matrix):
        nonlocal best_candidate, best_key
        metrics = symmetry_module._score_transition_candidate(
            candidate_matrix, linear_system
        )
        if metrics is None:
            return
        metrics["passes_residual_tol"] = metrics["residual"] <= residual_tol
        key = (
            not metrics["passes_residual_tol"],
            *symmetry_module._candidate_sort_key(metrics),
        )
        if best_candidate is None or key < best_key:
            best_candidate = metrics
            best_key = key

    for coeffs in symmetry_module._iter_transition_coefficients(len(basis_matrices)):
        candidate = sum(
            (coefficient * basis for coefficient, basis in zip(coeffs, basis_matrices)),
            np.zeros((3, 3), dtype=float),
        )
        consider(candidate)

    if (
        best_candidate is None
        or not best_candidate["passes_residual_tol"]
        or best_candidate["sigma_min"] < 1e-8
    ):
        rng = np.random.default_rng(0)
        for _ in range(max(64, 16 * len(basis_matrices))):
            coeffs = rng.normal(size=len(basis_matrices))
            candidate = sum(
                (
                    coefficient * basis
                    for coefficient, basis in zip(coeffs, basis_matrices)
                ),
                np.zeros((3, 3), dtype=float),
            )
            consider(candidate)

    return best_candidate


def _assert_same_candidate(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "matrix":
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


def test_batched_transition_tolerance_scan_matches_independent_searches():
    basis_matrices = [
        np.eye(3),
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
        np.diag([1.0, -2.0, 0.25]),
    ]
    linear_system = np.arange(45, dtype=float).reshape(5, 9) / 37.0
    residual_tolerances = [1e-10, 0.5, 5.0]

    expected = [
        _reference_search(basis_matrices, linear_system, tolerance)
        for tolerance in residual_tolerances
    ]
    actual = symmetry_module._search_transition_candidate(
        basis_matrices,
        linear_system,
        residual_tol=residual_tolerances,
    )

    for actual_candidate, expected_candidate in zip(actual, expected):
        _assert_same_candidate(actual_candidate, expected_candidate)


def test_batched_transition_tolerance_scan_scores_each_candidate_once(monkeypatch):
    basis_matrices = [np.eye(3), np.diag([1.0, -1.0, 0.5])]
    linear_system = np.zeros((2, 9), dtype=float)
    residual_tolerances = [1e-8, 1e-6, 1e-4, 1e-2]
    original_score = symmetry_module._score_transition_candidate
    score_calls = 0

    def counted_score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(
        symmetry_module,
        "_score_transition_candidate",
        counted_score,
    )
    symmetry_module._search_transition_candidate(
        basis_matrices,
        linear_system,
        residual_tol=residual_tolerances,
    )
    batched_calls = score_calls

    score_calls = 0
    for tolerance in residual_tolerances:
        symmetry_module._search_transition_candidate(
            basis_matrices,
            linear_system,
            residual_tol=tolerance,
        )

    assert score_calls == batched_calls * len(residual_tolerances)
