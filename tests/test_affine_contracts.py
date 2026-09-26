from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import spglib

from findspingroup.core.identify_spin_space_group import _build_magnetic_atom_preservation_checker
from findspingroup.core.tolerances import Tolerances
from findspingroup.structure import AtomicSite, CrystalCell, SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.structure.group import _deduplicate_spin_space_ops
from findspingroup.utils.matrix_utils import getNormInf, normalize_vector_to_zero, reduce_computed_mod1
from findspingroup.utils.international_symbol import _closure_from_generators, _op_key


@pytest.mark.parametrize("shift", [5e-5, -5e-5, 5e-10, -5e-10])
def test_mod1_algebra_preserves_small_affine_displacements(shift):
    op = SpinSpaceGroupOperation(np.eye(3), -np.eye(3), [shift, 0, 0])
    identity = SpinSpaceGroupOperation.identity()
    for result in [op @ identity, identity @ op, op.inv()]:
        assert getNormInf(result.translation, op.translation) < 1e-15
    assert getNormInf((op @ op.inv()).translation, np.zeros(3)) < 1e-15


def test_nofrac_transport_and_spin_only_preserve_integer_lifts():
    spin_flip = np.diag([-1, -1, 1])
    group = SpinSpaceGroup([SpinSpaceGroupOperation.identity(),
                           SpinSpaceGroupOperation(spin_flip, np.eye(3), [0.5, 0, 0])])
    lifted = group.transform(np.diag([2, 1, 1]), np.zeros(3), frac=False)
    spin_translation = next(op for op in lifted.ops if np.allclose(op.spin_rotation, spin_flip))
    np.testing.assert_array_equal(spin_translation.translation, [1, 0, 0])
    assert len(lifted.sog) == 1
    assert len(lifted.transform_spin(np.eye(3)).sog) == 1


def test_representative_dedup_checks_boundaries_without_collapsing_lifts():
    ops = [SpinSpaceGroupOperation(np.eye(3), np.eye(3), t)
           for t in ([0.499e-5, 0, 0], [0.501e-5, 0, 0], [1.00000499, 0, 0])]
    assert len(_deduplicate_spin_space_ops(ops, tol=1e-6)) == 2
    assert len(_deduplicate_spin_space_ops(list(reversed(ops)), tol=1e-6)) == 2


def test_representative_dedup_matches_exhaustive_absolute_comparison():
    rng = np.random.default_rng(208)
    tol = 0.02
    ops = []
    for _ in range(25):
        spin = np.eye(3) + rng.normal(scale=0.01, size=(3, 3))
        real = np.eye(3) + rng.normal(scale=0.01, size=(3, 3))
        translation = rng.integers(0, 3, size=3) + rng.normal(scale=0.01, size=3)
        for shift in (0, 0.005, 0.015):
            ops.append(SpinSpaceGroupOperation(spin + shift, real + shift, translation + shift))
    expected = []
    for op in ops:
        if not any(all(np.max(np.abs(op[i] - old[i])) <= tol for i in (0, 1))
                   and np.max(np.abs(op[2] - old[2])) < tol for old in expected):
            expected.append(op)
    actual = _deduplicate_spin_space_ops(ops, tol=tol)
    assert [id(op) for op in actual] == [id(op) for op in expected]


def test_tight_site_tolerance_does_not_alias_cached_actions():
    atom = AtomicSite([0, 0, 0], [0, 0, 1], 1, "Fe", lattice_matrix=np.eye(3) * 30)
    preserves = _build_magnetic_atom_preservation_checker([atom], Tolerances(space=1e-9))
    assert preserves(np.eye(3), np.eye(3), np.zeros(3))
    assert not preserves(np.eye(3), np.eye(3), np.array([4e-9, 0, 0]))


def test_computed_modulo_cleans_roundoff_but_not_resolved_displacements():
    values = np.array([1 - 5e-10, 5e-10, -5e-10])
    np.testing.assert_array_equal(reduce_computed_mod1(values), np.mod(values, 1))
    np.testing.assert_array_equal(reduce_computed_mod1([1 - np.finfo(float).eps, 1, 0]), np.zeros(3))


def test_symbol_generator_closure_preserves_legacy_budget_without_mutating_inputs():
    ops = [SpinSpaceGroupOperation(np.eye(3), np.eye(3), t)
           for t in ([0, 0, 0], [0.333333, 0, 0], [0.666667, 0, 0])]
    before = [op.translation.copy() for op in ops]
    # Independent oracle for the pre-refactor symbol route. This intentionally
    # checks compatibility, not an exact abstract group order for noisy inputs.
    generator = ops[1]
    inverse_real = np.linalg.inv(generator.rotation)
    inverse = SpinSpaceGroupOperation(np.linalg.inv(generator.spin_rotation), inverse_real,
                                     normalize_vector_to_zero(-inverse_real @ generator.translation, atol=1e-4))
    identity = SpinSpaceGroupOperation.identity()
    expected = {_op_key(identity)}
    queue = [identity]
    for current in queue:
        for word in (generator, inverse):
            product = SpinSpaceGroupOperation(
                current.spin_rotation @ word.spin_rotation,
                current.rotation @ word.rotation,
                normalize_vector_to_zero(current.rotation @ word.translation + current.translation, atol=1e-4),
            )
            key = _op_key(product)
            if key not in expected:
                expected.add(key)
                queue.append(product)
    assert _closure_from_generators([generator]) == expected
    for old, op in zip(before, ops):
        np.testing.assert_array_equal(old, op.translation)


def test_symbol_generator_closure_completes_partial_database_representatives():
    generator = SpinSpaceGroupOperation(np.eye(3), np.eye(3), [0.25, 0, 0])
    assert len(_closure_from_generators([generator])) == 4


def test_scif_parent_transform_display_snaps_near_integer_shifts(monkeypatch):
    module = importlib.import_module("findspingroup.find_spin_group")
    dataset = SimpleNamespace(number=194, international="P6_3/mmc", transformation_matrix=np.eye(3),
                              origin_shift=np.array([1 - 4e-6, 4e-6, -4e-6]))
    monkeypatch.setattr(module, "get_symmetry_dataset", lambda *a, **kw: dataset)
    cell = CrystalCell(np.eye(3), [[0, 0, 0]], [1], ["Fe"], [[0, 0, 1]])
    parent, _, _ = module._identify_parent_space_group_for_export_cell(cell, symprec=0.02)
    assert parent["child_transform_Pp_abc"] == "a,b,c;0,0,0"


def test_mn3sn_parent_setting_transports_to_database_operations(monkeypatch):
    module = importlib.import_module("findspingroup.find_spin_group")
    original = module._identify_parent_space_group_for_export_cell
    checked = []

    def verify(cell, **kwargs):
        result = original(cell, **kwargs)
        dataset = result[2]
        database = spglib.get_symmetry_from_database(dataset.hall_number)
        p, origin = dataset.transformation_matrix, dataset.origin_shift
        assert len(dataset.rotations) == len(database["rotations"])
        for rotation, translation in zip(dataset.rotations, dataset.translations):
            moved_rotation = p @ rotation @ np.linalg.inv(p)
            moved_translation = p @ translation + (np.eye(3) - moved_rotation) @ origin
            assert any(np.allclose(moved_rotation, r, atol=1e-8, rtol=0)
                       and getNormInf(moved_translation, t) < 1e-8
                       for r, t in zip(database["rotations"], database["translations"]))
        checked.append(dataset.hall_number)
        return result

    monkeypatch.setattr(module, "_identify_parent_space_group_for_export_cell", verify)
    result = module.find_spin_group("tests/testset/mcif_241130_no2186/0.199_Mn3Sn.mcif")
    assert result.index == "194.11.1.1.P"
    assert checked and set(checked) == {488}


@pytest.mark.parametrize("case", ["2.63_DyCrO3", "2.64_DyCrO3"])
def test_g_type_symbol_retains_nontrivial_axis_translation_after_setting_transport(case):
    module = importlib.import_module("findspingroup.find_spin_group")
    result = module.find_spin_group(f"tests/testset/mcif_241130_no2186/{case}.mcif")
    assert result.index == "11.2.2.6"
    # A modulo-one intermediate must not introduce a spurious off-axis integer
    # lift: it would hide the twofold spin translation from the g-type symbol.
    assert ": (1,1,2_{010})" in result.convention_ssg_international_linear
    assert ": (2_{001},1,1)" in result.magnetic_primitive_ssg_international_linear
