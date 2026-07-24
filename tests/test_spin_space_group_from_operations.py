import numpy as np
import pytest

from findspingroup import find_spin_group, get_spin_space_group_from_operations
from findspingroup.structure import SpinSpaceGroup, SpinSpaceGroupOperation
from examples.spin_space_group_from_operations import (
    identify_crse,
    identify_mn3sn,
    identify_mn3sn_convention_cartesian,
    identify_mn3sn_magnetic_primitive_oriented,
    identify_mn3sn_with_explicit_spin_only_semantics,
    identify_mnte,
)


IDENTITY = np.eye(3)
ZERO = np.zeros(3)


def _operation(spin_rotation=IDENTITY, real_rotation=IDENTITY, translation=ZERO):
    return SpinSpaceGroupOperation(spin_rotation, real_rotation, translation)


@pytest.mark.parametrize(
    ("configuration", "expected_configuration", "expected_order", "expected_index"),
    [
        (None, "Noncoplanar", 1, "1.1.1.1"),
        ("coplanar", "Coplanar", 2, "1.1.1.1.P"),
        ("collinear", "Collinear", 4, "1.1.1.1.L"),
    ],
)
def test_factory_materializes_internal_spin_only_contract(
    configuration,
    expected_configuration,
    expected_order,
    expected_index,
):
    kwargs = {}
    if configuration is not None:
        kwargs["spin_configuration"] = configuration
        kwargs["spin_only_direction"] = [0, 0, 1]

    group = get_spin_space_group_from_operations(
        [_operation()],
        spin_frame="oriented",
        **kwargs,
    )

    assert isinstance(group, SpinSpaceGroup)
    assert group.conf == expected_configuration
    assert len(group.sog) == expected_order
    assert group.index == expected_index
    assert group.G0_num == 1
    assert group.L0_num == 1


def test_repository_operation_input_examples():
    expected = {
        identify_mn3sn: ("194.11.1.1.P", "Coplanar", 48),
        identify_mn3sn_with_explicit_spin_only_semantics: (
            "194.11.1.1.P",
            "Coplanar",
            48,
        ),
        identify_mn3sn_convention_cartesian: (
            "194.11.1.1.P",
            "Coplanar",
            48,
        ),
        identify_mn3sn_magnetic_primitive_oriented: (
            "194.11.1.1.P",
            "Coplanar",
            48,
        ),
        identify_mnte: ("194.164.1.1.L", "Collinear", 96),
        identify_crse: ("194.149.3.3", "Noncoplanar", 216),
    }

    for factory, (index, configuration, operation_count) in expected.items():
        group = factory()
        assert group.index == index
        assert group.conf == configuration
        assert len(group.ops) == operation_count


def test_factory_infers_collinear_axis_from_noncanonical_spin_only_generators():
    angle = 2 * np.pi / 3
    c3z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    mirror_xz = np.diag([1, -1, 1])

    group = get_spin_space_group_from_operations(
        [_operation(c3z), _operation(mirror_xz)],
        spin_frame="oriented",
    )

    assert group.conf == "Collinear"
    assert len(group.sog) == 4
    assert np.allclose(np.abs(group.collinear_axis), [0, 0, 1], atol=1e-8)
    assert group.index == "1.1.1.1.L"


def test_factory_closes_affine_generators_and_matches_complete_operation_input():
    inversion = _operation(IDENTITY, -IDENTITY, [0.5, 0, 0])
    generated = get_spin_space_group_from_operations(
        [inversion],
        spin_frame="oriented",
    )
    complete = get_spin_space_group_from_operations(
        [_operation(), inversion],
        spin_frame="oriented",
    )

    assert len(generated.ops) == 2
    assert generated.index == complete.index
    assert all(
        any(left.is_same_with(right, atol=1e-8) for right in complete.ops)
        for left in generated.ops
    )


def test_factory_closes_spin_translation_generator_modulo_collinear_spin_only_group():
    spin_translation = _operation(-IDENTITY, IDENTITY, [0.5, 0, 0])

    group = get_spin_space_group_from_operations(
        [spin_translation],
        spin_configuration="collinear",
        spin_only_direction=[0, 0, 1],
        spin_frame="oriented",
    )

    assert group.conf == "Collinear"
    assert group.index == "1.1.2.1.L"
    assert len(group.ops) == 8
    assert group.ik == 2
    assert group.msg_bns_num == "1.3"


def test_factory_accepts_serialized_operation_dicts():
    group = get_spin_space_group_from_operations(
        [
            {
                "spin_rotation": IDENTITY.tolist(),
                "real_rotation": IDENTITY.tolist(),
                "translation": ZERO.tolist(),
            }
        ],
        spin_frame="oriented",
    )

    assert group.index == "1.1.1.1"


@pytest.mark.parametrize("translation", ([1, 0, 0], [-1e-8, 0, 0]))
def test_factory_rejects_translations_outside_mod_one_cell(translation):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        get_spin_space_group_from_operations(
            [_operation(translation=translation)]
        )


def test_factory_rejects_explicit_configuration_conflicting_with_spin_only_group():
    spin_plane_mirror = np.diag([1, 1, -1])

    with pytest.raises(ValueError, match="conflicts"):
        get_spin_space_group_from_operations(
            [_operation(), _operation(spin_plane_mirror)],
            spin_configuration="collinear",
            spin_only_direction=[0, 0, 1],
            spin_frame="oriented",
        )


def test_factory_requires_direction_when_explicit_configuration_cannot_infer_it():
    with pytest.raises(ValueError, match="spin_only_direction"):
        get_spin_space_group_from_operations(
            [_operation()],
            spin_configuration="collinear",
            spin_frame="oriented",
        )


def test_factory_accepts_finite_spin_rotations_in_a_nonorthogonal_basis():
    basis = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    basis_inverse = np.linalg.inv(basis)
    angle = 2 * np.pi / 3
    c3_cartesian = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    mirror_cartesian = np.diag([1, -1, 1])
    c3_nonorthogonal = basis_inverse @ c3_cartesian @ basis
    mirror_nonorthogonal = basis_inverse @ mirror_cartesian @ basis

    group = get_spin_space_group_from_operations(
        [_operation(c3_nonorthogonal), _operation(mirror_nonorthogonal)],
        spin_frame="oriented",
    )

    assert group.conf == "Collinear"
    assert group.index == "1.1.1.1.L"


def test_factory_rejects_spin_matrices_without_finite_invariant_metric():
    with pytest.raises(ValueError, match="determinant"):
        get_spin_space_group_from_operations(
            [_operation(np.diag([2, 1, 1]))],
            spin_frame="oriented",
        )


def _hexagonal_frame_case():
    lattice = np.array(
        [
            [2.0, 0.0, 0.0],
            [-1.0, np.sqrt(3.0), 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    metric = lattice @ lattice.T
    setting_to_cartesian = np.linalg.cholesky(metric).T
    real_threefold = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    spin_threefold_cartesian = (
        setting_to_cartesian
        @ real_threefold
        @ np.linalg.inv(setting_to_cartesian)
    )
    return lattice, metric, real_threefold, spin_threefold_cartesian


def test_factory_cartesian_frame_matches_oriented_frame_for_generators():
    lattice, metric, real_threefold, spin_threefold_cartesian = (
        _hexagonal_frame_case()
    )
    cartesian = get_spin_space_group_from_operations(
        [_operation(spin_threefold_cartesian, real_threefold)],
        spin_frame="cartesian",
        real_space_lattice=lattice,
        real_space_metric=metric,
    )
    oriented = get_spin_space_group_from_operations(
        [_operation(real_threefold, real_threefold)],
        spin_frame="oriented",
        real_space_metric=metric,
    )

    assert cartesian.input_spin_frame == "cartesian"
    assert cartesian.spin_settings == "oriented"
    assert cartesian.index == oriented.index
    assert cartesian.msg_bns_num == oriented.msg_bns_num
    assert len(cartesian.ops) == len(oriented.ops)
    assert all(
        any(left.is_same_with(right, atol=1e-8) for right in oriented.ops)
        for left in cartesian.ops
    )


def test_factory_cartesian_frame_uses_default_relative_frame_from_metric():
    _lattice, metric, real_threefold, spin_threefold_cartesian = (
        _hexagonal_frame_case()
    )
    cartesian = get_spin_space_group_from_operations(
        [_operation(spin_threefold_cartesian, real_threefold)],
        spin_frame="cartesian",
        real_space_metric=metric,
    )
    oriented = get_spin_space_group_from_operations(
        [_operation(real_threefold, real_threefold)],
        spin_frame="oriented",
        real_space_metric=metric,
    )

    assert cartesian.index == oriented.index
    assert cartesian.msg_bns_num == oriented.msg_bns_num


def test_factory_cartesian_frame_matches_oriented_frame_for_complete_operations():
    lattice, metric, real_threefold, spin_threefold_cartesian = (
        _hexagonal_frame_case()
    )
    cartesian_generator = _operation(
        spin_threefold_cartesian,
        real_threefold,
    )
    oriented_generator = _operation(real_threefold, real_threefold)
    cartesian_complete = [
        _operation(),
        cartesian_generator,
        cartesian_generator @ cartesian_generator,
    ]
    oriented_complete = [
        _operation(),
        oriented_generator,
        oriented_generator @ oriented_generator,
    ]

    cartesian = get_spin_space_group_from_operations(
        cartesian_complete,
        spin_frame="cartesian",
        real_space_lattice=lattice,
        real_space_metric=metric,
    )
    oriented = get_spin_space_group_from_operations(
        oriented_complete,
        spin_frame="oriented",
        real_space_metric=metric,
    )

    assert cartesian.index == oriented.index
    assert cartesian.msg_bns_num == oriented.msg_bns_num


def test_factory_cartesian_frame_is_invariant_under_common_rigid_rotation():
    lattice, metric, real_threefold, spin_threefold_cartesian = (
        _hexagonal_frame_case()
    )
    angle = 0.37
    rigid_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_lattice = lattice @ rigid_rotation.T
    rotated_spin = (
        rigid_rotation
        @ spin_threefold_cartesian
        @ rigid_rotation.T
    )

    cartesian = get_spin_space_group_from_operations(
        [_operation(rotated_spin, real_threefold)],
        spin_frame="cartesian",
        real_space_lattice=rotated_lattice,
        real_space_metric=metric,
    )
    oriented = get_spin_space_group_from_operations(
        [_operation(real_threefold, real_threefold)],
        spin_frame="oriented",
        real_space_metric=metric,
    )

    assert cartesian.index == oriented.index
    assert cartesian.msg_bns_num == oriented.msg_bns_num


def test_factory_cartesian_frame_handles_permuted_setting_lattice():
    lattice = np.array(
        [
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 4.0, 0.0],
        ]
    )
    metric = lattice @ lattice.T
    setting_to_cartesian = lattice.T
    real_twofold = np.diag([-1.0, -1.0, 1.0])
    spin_twofold_cartesian = (
        setting_to_cartesian
        @ real_twofold
        @ np.linalg.inv(setting_to_cartesian)
    )

    cartesian = get_spin_space_group_from_operations(
        [_operation(spin_twofold_cartesian, real_twofold)],
        spin_frame="cartesian",
        real_space_lattice=lattice,
        real_space_metric=metric,
    )
    oriented = get_spin_space_group_from_operations(
        [_operation(real_twofold, real_twofold)],
        spin_frame="oriented",
        real_space_metric=metric,
    )

    assert cartesian.index == oriented.index
    assert cartesian.msg_bns_num == oriented.msg_bns_num


def test_factory_cartesian_frame_transforms_explicit_spin_only_direction():
    lattice = np.diag([2.0, 3.0, 4.0])
    metric = lattice @ lattice.T
    cartesian = get_spin_space_group_from_operations(
        [_operation()],
        spin_configuration="collinear",
        spin_only_direction=[1, 0, 0],
        spin_frame="cartesian",
        real_space_lattice=lattice,
        real_space_metric=metric,
    )
    oriented = get_spin_space_group_from_operations(
        [_operation()],
        spin_configuration="collinear",
        spin_only_direction=[1, 0, 0],
        spin_frame="oriented",
        real_space_metric=metric,
    )

    assert cartesian.index == oriented.index
    assert np.allclose(cartesian.collinear_axis, oriented.collinear_axis)


def test_factory_default_cartesian_frame_requires_real_space_geometry():
    with pytest.raises(ValueError, match="real_space_lattice or real_space_metric"):
        get_spin_space_group_from_operations([_operation()])


def test_factory_defaults_to_cartesian_frame():
    group = get_spin_space_group_from_operations(
        [_operation()],
        real_space_lattice=np.eye(3),
    )

    assert group.input_spin_frame == "cartesian"
    assert group.spin_settings == "oriented"


def test_factory_rejects_unknown_spin_frame():
    with pytest.raises(ValueError, match="spin_frame must be one of"):
        get_spin_space_group_from_operations(
            [_operation()],
            spin_frame="laboratory",
        )


def test_factory_rejects_inconsistent_lattice_and_metric():
    with pytest.raises(ValueError, match="inconsistent"):
        get_spin_space_group_from_operations(
            [_operation()],
            real_space_lattice=np.eye(3),
            real_space_metric=2 * np.eye(3),
        )


def test_factory_roundtrips_material_operation_views_across_settings_and_frames():
    cases = [
        ("src/findspingroup/examples/0.200_Mn3Sn.mcif", None),
        (
            "src/findspingroup/examples/0.800_MnTe.mcif",
            {
                "configuration": "collinear",
                "cartesian_direction": [0.5, np.sqrt(3.0) / 2.0, 0.0],
                "oriented_direction": [1.0, 1.0, 0.0],
            },
        ),
        ("src/findspingroup/examples/CoNb3S6_tripleQ.mcif", None),
    ]

    for path, spin_semantics in cases:
        result = find_spin_group(path, components=["operation_views"])
        cell_details = {
            "convention": result.convention_cell_detail,
            "magnetic_primitive": result.acc_primitive_magnetic_cell_detail,
            "input": result.input_cell_detail,
        }
        for setting_key, setting_payload in result.operation_views.items():
            setting_name, spin_frame = setting_key.rsplit("_", 1)
            lattice = np.asarray(
                cell_details[setting_name]["lattice"],
                dtype=float,
            )
            metric = lattice @ lattice.T
            all_operations = setting_payload["views"]["all"]["ops"]
            generator_indices = list(
                setting_payload["views"]["generators"]["indices"]
            )
            generator_indices.extend(
                setting_payload["views"]
                .get("spin_translations", {})
                .get("indices", [])
            )
            generator_indices = list(dict.fromkeys(generator_indices))
            generator_operations = [
                all_operations[index - 1] for index in generator_indices
            ]
            kwargs = {}
            if spin_semantics is not None:
                kwargs = {
                    "spin_configuration": spin_semantics["configuration"],
                    "spin_only_direction": spin_semantics[
                        f"{spin_frame}_direction"
                    ],
                }

            for supplied_operations in (all_operations, generator_operations):
                rebuilt = get_spin_space_group_from_operations(
                    supplied_operations,
                    spin_frame=spin_frame,
                    real_space_lattice=lattice,
                    real_space_metric=metric,
                    tol=0.01,
                    **kwargs,
                )
                assert rebuilt.index == result.index
                assert rebuilt.conf == result.conf
                assert rebuilt.msg_bns_num == result.msg_bns_number


def test_factory_roundtrips_high_order_coplanar_operation_input():
    result = find_spin_group(
        "tests/testset/mcif_241130_no2186/1.0.41_RbNiCl3.mcif",
        components=["operation_views"],
    )
    setting = result.operation_views["magnetic_primitive_oriented"]
    operations = setting["views"]["all"]["ops"]
    generator_indices = list(setting["views"]["generators"]["indices"])
    generator_indices.extend(setting["views"]["spin_translations"]["indices"])
    generator_indices = list(dict.fromkeys(generator_indices))
    generators = [operations[index - 1] for index in generator_indices]
    lattice = np.asarray(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        dtype=float,
    )

    rebuilt = get_spin_space_group_from_operations(
        operations,
        spin_frame="oriented",
        real_space_lattice=lattice,
        tol=0.01,
    )
    rebuilt_from_generators = get_spin_space_group_from_operations(
        generators,
        spin_frame="oriented",
        real_space_lattice=lattice,
        tol=0.01,
    )

    assert rebuilt.index == result.index
    assert rebuilt.conf == "Coplanar"
    assert rebuilt.msg_bns_num == result.msg_bns_number
    assert rebuilt_from_generators.index == result.index
    assert rebuilt_from_generators.conf == "Coplanar"
    assert rebuilt_from_generators.msg_bns_num == result.msg_bns_number
