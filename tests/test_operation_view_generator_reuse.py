import importlib

import numpy as np

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


def test_operation_view_generator_closure_includes_implicit_centering(monkeypatch):
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
