import json
from types import SimpleNamespace

import numpy as np
import pytest

from findspingroup import (
    KPointSpinPolarizationAnalyzer,
    analyze_kpoint_spin_polarization,
    find_spin_group,
)
from findspingroup.batch_mcif import _build_export_root
from findspingroup.structure.group import SpinSpaceGroup


def _ssg_op(spin, real=None, translation=None):
    return {
        "spin_rotation": np.asarray(spin, dtype=float).tolist(),
        "real_rotation": np.asarray(np.eye(3) if real is None else real, dtype=float).tolist(),
        "translation": np.asarray(
            np.zeros(3) if translation is None else translation,
            dtype=float,
        ).tolist(),
    }


def _synthetic_result(
    ssg_operations,
    *,
    msg_operations=None,
    input_to_acc=None,
    quasi_2d=None,
):
    if msg_operations is None:
        msg_operations = [[1, np.eye(3), np.zeros(3)]]
    return SimpleNamespace(
        acc_primitive_ssg_ops=ssg_operations,
        acc_primitive_msg_ops=msg_operations,
        acc_primitive_magnetic_cell=(
            np.eye(3),
            np.zeros((1, 3)),
            np.ones(1, dtype=int),
            np.zeros((1, 3)),
        ),
        T_input_to_acc_primitive=(
            np.eye(3) if input_to_acc is None else np.asarray(input_to_acc, dtype=float),
            np.zeros(3),
        ),
        tolerances={"matrix_tol": 0.01},
        ssg_little_group_ops=None,
        spin_polarizations_acc_cartesian=None,
        msg_little_group_ops=None,
        msg_spin_polarizations_acc_cartesian=None,
        quasi_2d=quasi_2d,
        fsg_version="test",
        index="test",
        msg_bns_number="test",
    )


@pytest.fixture(scope="module")
def vcl2_result():
    return find_spin_group("src/findspingroup/examples/1.237_VCl2.mcif", components=())


def test_existing_acc_kpoint_constraints_are_reused_without_drift(vcl2_result):
    original_contract = (
        vcl2_result.index,
        vcl2_result.KPOINTS,
        list(vcl2_result.spin_polarizations_acc_cartesian),
        list(vcl2_result.msg_spin_polarizations_acc_cartesian),
    )
    analyzer = vcl2_result.prepare_kpoint_spin_polarization_analyzer()
    acc_ssg = SpinSpaceGroup(list(vcl2_result.acc_primitive_ssg_ops), tol=0.01)

    for index, kpoint in enumerate(acc_ssg.kpoints_primitive):
        query = analyzer.query(kpoint, kpoint_setting="acc_primitive")
        assert (
            query["without_soc"]["constraint"]
            == vcl2_result.spin_polarizations_acc_cartesian[index]
        )
        assert (
            query["with_soc"]["constraint"]
            == vcl2_result.msg_spin_polarizations_acc_cartesian[index]
        )
        assert query.audit["without_soc"]["source"] == "precomputed_kspace"
        assert query.audit["with_soc"]["source"] == "precomputed_kspace"

    validation = analyzer.validate_precomputed_constraints()
    assert validation["without_soc_precomputed_kpoints"] == len(acc_ssg.kpoints_primitive)
    assert validation["with_soc_precomputed_kpoints"] == len(acc_ssg.kpoints_primitive)

    assert original_contract == (
        vcl2_result.index,
        vcl2_result.KPOINTS,
        vcl2_result.spin_polarizations_acc_cartesian,
        vcl2_result.msg_spin_polarizations_acc_cartesian,
    )


def test_serialized_result_reuses_numerically_equivalent_constraint_formatting():
    result = find_spin_group(
        "tests/testset/mcif_241130_no2186/0.1002_SrZn2Fe16O27.mcif",
        components=(),
    )
    serialized = json.loads(json.dumps(_build_export_root(result)))
    analyzer = KPointSpinPolarizationAnalyzer.from_result(SimpleNamespace(**serialized))

    validation = analyzer.validate_precomputed_constraints()

    assert validation["without_soc_precomputed_kpoints"] == len(
        serialized["spin_polarizations_acc_cartesian"]
    )


def test_inconsistent_precomputed_spin_subspace_is_rejected():
    c2z = np.diag([-1.0, -1.0, 1.0])
    operations = [_ssg_op(np.eye(3)), _ssg_op(c2z)]
    result = _synthetic_result(operations)
    result.ssg_little_group_ops = [operations]
    result.spin_polarizations_acc_cartesian = [["Sx", "0", "0"]]
    analyzer = KPointSpinPolarizationAnalyzer.from_result(result)

    with pytest.raises(RuntimeError, match="spin subspace disagrees"):
        analyzer.validate_precomputed_constraints()


def test_vcl2_special_mk_submanifold_is_not_treated_as_generic_plane(vcl2_result):
    analyzer = vcl2_result.prepare_kpoint_spin_polarization_analyzer()

    special_mk = analyzer.query([0.25, 0.5, 0.0], kpoint_setting="acc_primitive")
    generic_plane = analyzer.query([0.25, 0.125, 0.0], kpoint_setting="acc_primitive")

    assert special_mk["without_soc"]["allowed"] is False
    assert special_mk["with_soc"]["allowed"] is True
    assert generic_plane["without_soc"]["allowed"] is True
    assert (
        special_mk.audit["without_soc"]["little_group_operation_indices"]
        != generic_plane.audit["without_soc"]["little_group_operation_indices"]
    )


def test_acc_reciprocal_lattice_shifts_give_the_same_constraint(vcl2_result):
    analyzer = vcl2_result.prepare_kpoint_spin_polarization_analyzer()
    reference = analyzer.query([0.25, 0.125, 0.0], kpoint_setting="acc_primitive")
    shifted = analyzer.query([1.25, -1.875, 3.0], kpoint_setting="acc_primitive")

    assert shifted.audit["kpoint"]["acc_primitive_reduced"] == pytest.approx(
        [0.25, 0.125, 0.0]
    )
    assert shifted["without_soc"] == reference["without_soc"]
    assert shifted["with_soc"] == reference["with_soc"]


def test_input_and_acc_kpoint_settings_use_the_reciprocal_transform():
    result = _synthetic_result(
        [_ssg_op(np.eye(3))],
        input_to_acc=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    analyzer = KPointSpinPolarizationAnalyzer.from_result(result)

    from_acc = analyzer.query([0.25, 0.2, 0.3], kpoint_setting="acc_primitive")
    from_input = analyzer.query([0.2, 0.25, 0.3], kpoint_setting="input")

    assert from_input.audit["kpoint"]["acc_primitive_reduced"] == pytest.approx(
        [0.25, 0.2, 0.3]
    )
    assert from_input["without_soc"] == from_acc["without_soc"]
    assert from_input["with_soc"] == from_acc["with_soc"]


def test_input_kpoint_setting_rejects_ambiguous_supercell_folding():
    analyzer = KPointSpinPolarizationAnalyzer.from_result(
        _synthetic_result(
            [_ssg_op(np.eye(3))],
            input_to_acc=np.diag([2.0, 1.0, 1.0]),
        )
    )

    default_acc_query = analyzer.query([0.25, 0.2, 0.3])
    assert default_acc_query["kpoint_setting"] == "acc_primitive"

    with pytest.raises(ValueError, match="folds multiple ACC-primitive k points"):
        analyzer.query([0.25, 0.2, 0.3], kpoint_setting="input")


def test_analyzer_accepts_json_style_ssg_operation_sequences():
    identity_sequence = [np.eye(3).tolist(), np.eye(3).tolist(), [0.0, 0.0, 0.0]]
    result = _synthetic_result([identity_sequence])

    query = KPointSpinPolarizationAnalyzer.from_result(result).query([0.2, 0.3, 0.4])

    assert query["without_soc"]["dimension"] == 3


def test_kpoint_tolerance_controls_little_group_membership_only():
    c2z = np.diag([-1.0, -1.0, 1.0])
    real_rotation = np.diag([-1.0, 1.0, -1.0])
    result = _synthetic_result(
        [
            _ssg_op(np.eye(3)),
            _ssg_op(c2z, real=real_rotation),
        ]
    )
    analyzer = KPointSpinPolarizationAnalyzer.from_result(result)

    inside = analyzer.query([4e-6, 0.123, 0.0], kpoint_tol=1e-5)
    outside = analyzer.query([6e-6, 0.123, 0.0], kpoint_tol=1e-5)

    assert inside["without_soc"]["dimension"] == 1
    assert inside["without_soc"]["direction"] == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert outside["without_soc"]["dimension"] == 3
    assert (
        inside.audit["without_soc"]["constraint_tol"]
        == outside.audit["without_soc"]["constraint_tol"]
    )
    assert inside.audit["without_soc"]["membership_audit"]["stability"] == "near_boundary"


@pytest.mark.parametrize(
    ("spin_rotation", "expected_dimension"),
    [
        (np.eye(3), 3),
        (np.diag([1.0, 1.0, -1.0]), 2),
        (np.diag([-1.0, -1.0, 1.0]), 1),
        (-np.eye(3), 0),
    ],
)
def test_spin_constraint_subspace_dimensions(spin_rotation, expected_dimension):
    result = _synthetic_result([_ssg_op(spin_rotation)])
    query = KPointSpinPolarizationAnalyzer.from_result(result).query([0.0, 0.0, 0.0])

    assert query["without_soc"]["dimension"] == expected_dimension
    assert (
        len(query.audit["without_soc"]["basis_acc_primitive_cartesian"])
        == expected_dimension
    )


@pytest.mark.parametrize(("vacuum_axis", "vacuum_axis_index"), [("a", 0), ("b", 1), ("c", 2)])
def test_quasi2d_accepts_in_plane_k_and_rejects_out_of_plane_k(
    vacuum_axis,
    vacuum_axis_index,
):
    result = _synthetic_result(
        [_ssg_op(np.eye(3))],
        quasi_2d={
            "dimension": "2d",
            "vacuum_axis_input": vacuum_axis,
            "vacuum_axis_index": vacuum_axis_index,
        },
    )
    analyzer = KPointSpinPolarizationAnalyzer.from_result(result)
    in_plane_kpoint = np.array([0.2, 0.3, 0.4])
    in_plane_kpoint[vacuum_axis_index] = 0.0

    in_plane = analyzer.query(in_plane_kpoint)
    assert in_plane.audit["calculation_mode"] == "quasi2d"
    assert in_plane.audit["quasi2d_plane"]["distance_to_plane"] == 0.0
    equivalent_kpoint = in_plane_kpoint.copy()
    equivalent_kpoint[vacuum_axis_index] = 1.0
    equivalent_plane = analyzer.query(equivalent_kpoint)
    assert equivalent_plane["without_soc"] == in_plane["without_soc"]
    assert equivalent_plane["with_soc"] == in_plane["with_soc"]

    out_of_plane_kpoint = in_plane_kpoint.copy()
    out_of_plane_kpoint[vacuum_axis_index] = 1e-3
    with pytest.raises(ValueError, match="outside the quasi-2D reciprocal plane"):
        analyzer.query(out_of_plane_kpoint, kpoint_tol=1e-5)


def test_quasi2d_plane_is_checked_in_the_input_reciprocal_setting():
    input_to_acc = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = _synthetic_result(
        [_ssg_op(np.eye(3))],
        input_to_acc=input_to_acc,
        quasi_2d={
            "dimension": "2d",
            "vacuum_axis_input": "a",
            "vacuum_axis_index": 0,
        },
    )
    analyzer = KPointSpinPolarizationAnalyzer.from_result(result)

    assert analyzer.query([0.3, 0.0, 0.2]).audit["calculation_mode"] == "quasi2d"
    with pytest.raises(ValueError, match="outside the quasi-2D reciprocal plane"):
        analyzer.query([0.0, 0.3, 0.2])


def test_real_quasi2d_generic_point_matches_existing_display_payload():
    result = find_spin_group(
        "tests/testset/V2Se2O_2d.mcif",
        calculation_mode="quasi2d",
        vacuum_axis="c",
        components=(),
    )
    existing = result.quasi_2d["generic_point_2d"]
    query = result.analyze_kpoint_spin_polarization(
        existing["k_acc_primitive"],
        kpoint_setting="acc_primitive",
    )

    assert query.audit["calculation_mode"] == "quasi2d"
    assert (
        query["without_soc"]["constraint"]
        == existing["spin_polarizations"]
    )
    assert (
        query["with_soc"]["constraint"]
        == existing["msg_spin_polarization_2d"]
    )


def test_query_rejects_unresolved_quasi2d_setting():
    result = _synthetic_result(
        [_ssg_op(np.eye(3))],
        quasi_2d={
            "calculation_mode": "quasi2d",
            "dimension": "3d_or_unknown",
            "status": "ambiguous",
        },
    )

    with pytest.raises(ValueError, match="did not resolve"):
        KPointSpinPolarizationAnalyzer.from_result(result).query([0.2, 0.3, 0.0])


def test_query_rejects_invalid_kpoint_and_tolerance():
    analyzer = KPointSpinPolarizationAnalyzer.from_result(
        _synthetic_result([_ssg_op(np.eye(3))])
    )

    with pytest.raises(ValueError, match="exactly three"):
        analyzer.query([0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        analyzer.query([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="finite positive"):
        analyzer.query([0.0, 0.0, 0.0], kpoint_tol=0.0)
    with pytest.raises(ValueError, match="smaller than 0.5"):
        analyzer.query([0.0, 0.0, 0.0], kpoint_tol=0.5)
    with pytest.raises(ValueError, match="constraint_tol must be a finite positive"):
        KPointSpinPolarizationAnalyzer.from_result(analyzer.result, constraint_tol=0.0)


def test_top_level_api_accepts_an_existing_result(vcl2_result):
    result_fields_before = set(vcl2_result.to_dict())
    query = analyze_kpoint_spin_polarization(
        vcl2_result,
        [0.25, 0.5, 0.0],
        kpoint_setting="acc_primitive",
    )

    assert query.audit["ssg_index"] == "164.149.6.1.P"
    assert query["without_soc"]["allowed"] is False
    assert query["with_soc"]["allowed"] is True
    assert set(vcl2_result.to_dict()) == result_fields_before


def test_default_result_is_compact_and_keeps_diagnostics_in_audit(vcl2_result):
    query = vcl2_result.analyze_kpoint_spin_polarization(
        [0.25, 0.5, 0.0],
        kpoint_tol=2e-6,
    )

    assert set(query) == {
        "kpoint",
        "kpoint_setting",
        "kpoint_tol",
        "spin_frame",
        "without_soc",
        "with_soc",
    }
    assert query["kpoint"] == [0.25, 0.5, 0.0]
    assert query["kpoint_tol"] == 2e-6
    assert query["spin_frame"] == "acc_primitive_cartesian"
    assert set(query["without_soc"]) == {
        "allowed",
        "dimension",
        "constraint",
        "direction",
    }
    assert "little_group_operation_indices" not in repr(query)
    assert "little_group_operation_indices" in query.audit["without_soc"]
    assert "audit" not in query.to_dict()
    assert query.to_dict(include_audit=True)["audit"]["status"] == "ok"


def test_top_level_api_accepts_a_structure_file():
    query = analyze_kpoint_spin_polarization(
        "src/findspingroup/examples/1.237_VCl2.mcif",
        [0.25, 0.5, 0.0],
    )

    assert query.audit["ssg_index"] == "164.149.6.1.P"
    assert query["kpoint_setting"] == "acc_primitive"
    assert query["without_soc"]["allowed"] is False
