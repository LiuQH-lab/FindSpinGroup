import numpy as np

from findspingroup.find_spin_group import (
    _gspg_pair_closure,
    _gspg_pair_key,
    _safe_classify_spin_texture_config,
    _serialize_tensor_solution,
    _validated_gspg_constraint_generators,
    _validated_spin_texture_constraint_generators,
)
from findspingroup.spin_splitting import radical_text


def _c4_pairs():
    operations = []
    for power in range(4):
        angle = power * np.pi / 2.0
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        operations.append([rotation, rotation])
    return operations


def test_validated_constraint_generators_close_to_exact_full_group():
    operations = _c4_pairs()
    generators = _validated_gspg_constraint_generators(operations, tol=1e-8)

    assert len(generators) < len(operations)
    assert _gspg_pair_closure(generators, tol=1e-8, limit=64) == {
        _gspg_pair_key(operation) for operation in operations
    }


def test_invalid_preferred_generators_are_replaced_by_verified_subset():
    operations = _c4_pairs()
    identity_only = [operations[0]]

    generators = _validated_gspg_constraint_generators(
        operations,
        preferred_generators=identity_only,
        tol=1e-8,
    )

    assert len(generators) < len(operations)
    assert _gspg_pair_key(generators[0]) != _gspg_pair_key(identity_only[0])
    assert _gspg_pair_closure(generators, tol=1e-8, limit=64) == {
        _gspg_pair_key(operation) for operation in operations
    }


def test_tensor_serialization_preserves_full_group_constraint_shape():
    solution = (
        np.zeros((18, 9)),
        np.eye(9),
        [],
        [],
    )

    payload = _serialize_tensor_solution(
        solution,
        operations_count=4,
        solver_operations_count=2,
    )

    assert payload["operations_count"] == 4
    assert payload["constraint_shape"] == [36, 9]


def test_spin_texture_generators_preserve_keyed_spin_real_semantics():
    spin_mirror = np.diag([1.0, 1.0, -1.0])
    real_twofold = np.diag([-1.0, -1.0, 1.0])
    full_operations = [
        {"spin_rotation": np.eye(3), "real_rotation": np.eye(3)},
        {"spin_rotation": spin_mirror, "real_rotation": real_twofold},
    ]
    generators = _validated_spin_texture_constraint_generators(
        full_operations,
        tol=1e-8,
    )

    assert all(isinstance(operation, dict) for operation in generators)
    full_result = _safe_classify_spin_texture_config(full_operations, source="full")
    generator_result = _safe_classify_spin_texture_config(generators, source="generators")
    assert full_result["basis"] == generator_result["basis"]
    assert full_result["momentum_space_spin_configuration"] == "coplanar"


def test_radical_text_prefers_stable_simplest_form_within_tolerance():
    values = [
        np.sqrt(3.0) / 2.0,
        np.nextafter(np.sqrt(3.0) / 2.0, 0.0),
        np.nextafter(np.sqrt(3.0) / 2.0, 1.0),
    ]

    assert {
        radical_text(
            value,
            zero_tol=1e-8,
            max_radicand=12,
            max_denominator=24,
            max_multiplier=12,
        )
        for value in values
    } == {"sqrt(3)/2"}
