import importlib

import numpy as np
import pytest

from findspingroup.structure import SpinSpaceGroup, SpinSpaceGroupOperation


find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")


def _operation(real_rotation, translation=(0.0, 0.0, 0.0)):
    return SpinSpaceGroupOperation(
        np.asarray(real_rotation, dtype=float),
        np.asarray(real_rotation, dtype=float),
        np.asarray(translation, dtype=float),
    )


def _c2_operations(*, centered=False):
    identity = _operation(np.eye(3))
    twofold = _operation(np.diag([-1.0, -1.0, 1.0]))
    if not centered:
        return [identity, twofold]
    centering = _operation(np.eye(3), (0.5, 0.5, 0.0))
    centered_twofold = centering @ twofold
    return [identity, centering, twofold, centered_twofold]


def _view_payload(ssg, generator_ops, *, generator_ops_complete):
    return find_spin_group_module._build_operation_view_set(
        ssg,
        ops_payload=find_spin_group_module._serialize_ssg_operation_matrices(
            list(ssg.ops)
        ),
        seitz_latex=[f"op-{index}" for index in range(len(ssg.ops))],
        setting_label="test",
        spin_frame="cartesian",
        generator_ops=generator_ops,
        generator_ops_complete=generator_ops_complete,
    )


def test_operation_view_reuses_closure_validated_transformed_generators(monkeypatch):
    ssg = SpinSpaceGroup(_c2_operations(), tol=1e-8)
    preferred = [ssg.ops[1]]

    def unexpected(_ssg):
        raise AssertionError("complete transformed generators were reidentified")

    monkeypatch.setattr(
        find_spin_group_module,
        "_symbol_generator_ops_for_current_basis",
        unexpected,
    )

    views = _view_payload(ssg, preferred, generator_ops_complete=True)

    assert views["views"]["generators"]["indices"] == [2]


def test_operation_view_trusts_complete_generator_flag_without_runtime_closure(monkeypatch):
    ssg = SpinSpaceGroup(_c2_operations(centered=True), tol=1e-8)
    preferred = [next(op for op in ssg.ops if np.allclose(op.rotation, np.diag([-1, -1, 1]))) ]

    def unexpected(_ssg):
        raise AssertionError("centering-aware generators were reidentified")

    monkeypatch.setattr(
        find_spin_group_module,
        "_symbol_generator_ops_for_current_basis",
        unexpected,
    )

    views = _view_payload(ssg, preferred, generator_ops_complete=True)

    # This tests reuse only: the caller must establish completeness. A supplied
    # flag is not an independent proof of closure for this synthetic fixture.
    assert views["views"]["generators"]["operation_count"] == 1


def test_operation_view_reidentifies_generators_for_incomplete_current_setting(
    monkeypatch,
):
    ssg = SpinSpaceGroup(_c2_operations(), tol=1e-8)
    outside_current_setting = _operation(np.diag([1.0, -1.0, -1.0]))
    calls = 0

    def fallback(view_ssg):
        nonlocal calls
        calls += 1
        return [view_ssg.ops[1]]

    monkeypatch.setattr(
        find_spin_group_module,
        "_symbol_generator_ops_for_current_basis",
        fallback,
    )

    views = _view_payload(
        ssg,
        [outside_current_setting],
        generator_ops_complete=False,
    )

    assert calls == 1
    assert views["views"]["generators"]["indices"] == [2]


def _generated_public_indices(views, *, include_spin_only=False):
    ops = views["all"]["ops"]
    spin = np.asarray([op["spin_rotation"] for op in ops])
    real = np.asarray([op["real_rotation"] for op in ops])
    trans = np.asarray([op["translation"] for op in ops])
    generators = [index - 1 for index in views["generators"]["indices"]]
    if include_spin_only:
        generators.extend(np.flatnonzero((np.max(abs(real - np.eye(3)), axis=(1, 2)) < 1e-7)
                                         & (np.max(abs(trans - np.rint(trans)), axis=1) < 1e-7)).tolist())
    identity = np.flatnonzero((np.max(abs(spin - np.eye(3)), axis=(1, 2)) < 1e-7)
                             & (np.max(abs(real - np.eye(3)), axis=(1, 2)) < 1e-7)
                             & (np.max(abs(trans - np.rint(trans)), axis=1) < 1e-7))
    assert len(identity) == 1
    reached = {int(identity[0])}
    queue = list(reached)
    for i in queue:
        for j in generators:
            translation = real[i] @ trans[j] + trans[i]
            delta = trans - translation
            matches = np.flatnonzero((np.max(abs(spin - spin[i] @ spin[j]), axis=(1, 2)) < 1e-7)
                                    & (np.max(abs(real - real[i] @ real[j]), axis=(1, 2)) < 1e-7)
                                    & (np.max(abs(delta - np.rint(delta)), axis=1) < 1e-7))
            assert len(matches) == 1
            k = int(matches[0])
            if k not in reached:
                reached.add(k)
                queue.append(k)
    return reached


@pytest.mark.parametrize("case", ["1.367_Pu2O3", "1.648_Nd2O3"])
def test_collinear_centered_generators_close_in_all_public_settings(case):
    result = find_spin_group_module.find_spin_group(f"tests/testset/mcif_241130_no2186/{case}.mcif")
    assert result.index == "12.12.2.1.L"
    assert result.conf == "Collinear"
    assert len(result.operation_views) == 6
    for setting, payload in result.operation_views.items():
        views = payload["views"]
        assert len(_generated_public_indices(views)) == len(views["all"]["ops"]), setting


def test_collinear_presentation_does_not_remove_spin_only_constraints(monkeypatch):
    original = find_spin_group_module._spin_texture_config_from_ossg_convention
    checked = []

    def verify(ssg, cell, **kwargs):
        ops = kwargs["generator_ops"]
        assert any(not any(np.allclose(op.spin_rotation, sign * np.eye(3), atol=1e-7, rtol=0)
                           for sign in (-1, 1)) for op in ops)
        checked.append(len(ops))
        return original(ssg, cell, **kwargs)

    monkeypatch.setattr(find_spin_group_module, "_spin_texture_config_from_ossg_convention", verify)
    result = find_spin_group_module.find_spin_group("tests/testset/mcif_241130_no2186/1.367_Pu2O3.mcif")
    assert result.conf == "Collinear"
    assert checked


def test_collinear_generator_image_uses_axis_action_not_determinant():
    mirror = SpinSpaceGroupOperation(np.diag([-1, 1, 1]), np.eye(3), [0.5, 0.5, 0])
    reversal = SpinSpaceGroupOperation(np.diag([1, -1, -1]), np.eye(3), [0, 0.5, 0.5])
    spin_only = SpinSpaceGroupOperation(mirror.spin_rotation, np.eye(3), [0, 0, 0])
    projected = find_spin_group_module._collinear_presentation_generators(
        [mirror, reversal, spin_only], np.array([[0], [0], [1]]), tol=1e-8)
    assert len(projected) == 2
    np.testing.assert_array_equal(projected[0].spin_rotation, np.eye(3))
    np.testing.assert_array_equal(projected[1].spin_rotation, -np.eye(3))
    np.testing.assert_array_equal(mirror.spin_rotation, np.diag([-1, 1, 1]))
    np.testing.assert_array_equal(projected[0].translation, mirror.translation)


def test_collinear_generator_image_rejects_a_different_spin_line():
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    op = SpinSpaceGroupOperation(rotation, np.eye(3), [0, 0, 0])
    with pytest.raises(ValueError, match="spin line"):
        find_spin_group_module._collinear_presentation_generators([op], [0, 0, 1], tol=1e-8)


def test_generator_transport_includes_implicit_source_lattice_even_for_p1():
    generators = find_spin_group_module._transform_operation_generators(
        [], np.diag([0.5, 1, 1]), np.zeros(3), tol=1e-8)
    assert len(generators) == 1
    np.testing.assert_array_equal(generators[0].spin_rotation, np.eye(3))
    np.testing.assert_array_equal(generators[0].rotation, np.eye(3))
    np.testing.assert_array_equal(generators[0].translation, [0.5, 0, 0])
    assert find_spin_group_module._transform_operation_generators(
        [], np.eye(3), np.zeros(3), tol=1e-8) == []


@pytest.mark.parametrize("case", ["1.0.7_LuFe2O4", "2.2_Sr2F2Fe2OS2", "2.49_La2O2Fe2OSe2", "2.56_La2O2Fe2OS2"])
def test_input_generator_transport_keeps_source_translation_lattice(case):
    result = find_spin_group_module.find_spin_group(f"tests/testset/mcif_241130_no2186/{case}.mcif")
    for setting, payload in result.operation_views.items():
        views = payload["views"]
        assert len(_generated_public_indices(views, include_spin_only=True)) == len(views["all"]["ops"]), setting
