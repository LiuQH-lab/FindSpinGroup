from pathlib import Path
from types import SimpleNamespace

import json
import importlib
import sys
import numpy as np
import pytest

import findspingroup.core.identify_symmetry_from_ops as identify_symmetry_from_ops_module
import findspingroup.core.identify_spin_space_group as identify_spin_space_group_module
import findspingroup.structure.cell as cell_module
import findspingroup.structure.group as group_module
from findspingroup.data.acc_aligned_p_index_loader import (
    _spin_texture_config_record,
    get_pair_id_for_ssg_label,
    get_ssg_conventional_kpoint_symbols_for_label,
    get_spin_texture_config_for_ssg_label,
    get_spin_texture_config_id_for_ssg_label,
)
from findspingroup import (
    find_spin_group,
    find_spin_group_acc_primitive,
    find_spin_group_basic,
    find_spin_group_from_data,
    find_spin_group_input_ssg,
    find_spin_group_poscar_ssg,
    write_poscar_ssg_symmetry_dat,
    write_ssg_operation_matrices,
)
from findspingroup.find_spin_group import _expand_magnetic_indices_by_sg_orbit
from findspingroup.spin_splitting import (
    _append_basis_remainder_ascii,
    _append_basis_remainder_latex,
    basis_expression_to_latex,
    canonicalize_nullspace,
    classify_public_spin_texture_config,
    combine_spin_texture_basis_expression,
    combine_spin_texture_basis_span,
    operation_pairs_from_gspg_ops,
    spin_texture_basis_latex,
)
from findspingroup.core.identify_index.functions import (
    find_stand_gen_maps,
    is_matrix_equal,
    make_4d_matrix,
    map_transformation,
)
from findspingroup.core.identify_index.functions.get_stand_trans import get_stand_trans
from findspingroup.core.identify_index.functions.find_ssg_reduce import (
    find_ssg_transformation,
)
from findspingroup.core.identify_spin_space_group import (
    MagneticToleranceDegeneracyError,
    NONMAGNETIC_MTOL_ERROR,
    UNSTABLE_MTOL_ERROR,
    _candidate_directions_from_moments,
    _complete_ssg_ops_by_closure,
    _select_identify_pg_candidate,
    dedup_moments_with_tol,
    get_pg,
    identify_spin_space_group,
    identify_spin_space_group_result,
    _classify_moment_configuration,
    _build_pg_candidates,
    _configuration_compatibility,
    _collinear_residual,
    _configuration_details,
    _coplanar_residual,
)
from findspingroup.core.tolerances import DEFAULT_TOL, Tolerances
from findspingroup.core.identify_symmetry_from_ops import (
    analyze_transition_matrix_problem,
    deduplicate_matrix_pairs,
    find_transition_matrix_deterministic,
    get_magnetic_space_group_from_operations,
    identify_point_group,
)
from findspingroup.core import Molecule, PointGroupAnalyzer
from findspingroup.core.pg_analyzer import SymmOp, generate_full_symmops
from findspingroup.ferroelectric import (
    build_domain_reversal_coset_analysis,
    build_ferroelectric_switching_payload,
    build_parent_standard_supercell_domain_coset_analysis,
    build_vector_constraints_by_symmetry_payload,
)
from findspingroup.find_spin_group import (
    SCIF_CELL_MODE_INPUT_IDENTIFIED,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
    SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED,
    _classify_quasi2d_spin_texture_config,
    _canonicalize_input_to_standard_setting,
    _nonmagnetic_space_group_real_space_ops_in_cell_basis,
    audit_spatial_transform_effect,
    classify_magnetic_phase,
    get_magnetic_phase,
    _build_candidate_transform_chen_pp_abcs_hex_spatial_cubic_spin_from_identify,
    _ossg_oriented_spin_frame_ssg,
    _spin_transform_to_in_lattice,
    _spin_transform_to_oriented_abc,
    _build_msg_little_group_payload,
    _get_magnetic_little_group,
    _primitive_msg_ops_from_ssg,
    _tensor_ops_wo_soc,
    _translations_equivalent_mod_pure_translations,
    combine_parametric_solutions,
)
from findspingroup.io import (
    parse_cif_file,
    parse_poscar_file,
    parse_scif_metadata,
    parse_scif_text,
)
from findspingroup.io.scif_generator import write_scif_spin_only
from findspingroup.quasi2d import (
    prepare_quasi2d_input_cell,
    resolve_quasi2d_preprocessing,
)
from findspingroup.structure import SpinSpaceGroup
from findspingroup.structure.cell import (
    AtomicSite,
    CrystalCell,
    SpaceToleranceDegeneracyError,
    change_cell_settings,
)
from findspingroup.version import __version__
from findspingroup.structure.group import (
    SpinSpaceGroupOperation,
    _deduplicate_spin_space_ops,
    _resolve_point_group_info,
    op_key,
)
from findspingroup.utils.international_symbol import (
    _compose_setting_transform as _compose_symbol_setting_transform,
    _default_centering_vectors,
    _find_real_operation,
    _parse_sg_generator_ops,
    _select_preferred_primitive_translation_match,
    _select_preferred_translation_match,
    _transport_standard_generators_to_current_basis,
    build_international_symbol,
)
from findspingroup.utils.space_group_flags import (
    msg_parent_space_group_info,
    space_group_polar_axis_basis,
    space_group_is_centrosymmetric,
    space_group_is_chiral,
    space_group_is_polar,
)
from findspingroup.utils.seitz_symbol import describe_point_operation, describe_spin_space_operation
from findspingroup.utils import general_positions_to_matrix

find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")


def _serialize_gspg_pairs(ops):
    return [
        [
            np.asarray(spin_rotation, dtype=float).tolist(),
            np.asarray(space_rotation, dtype=float).tolist(),
        ]
        for spin_rotation, space_rotation in ops
    ]


def _assert_setting_transform_inverse(forward, backward):
    forward_matrix = np.asarray(forward[0], dtype=float)
    forward_shift = np.asarray(forward[1], dtype=float)
    backward_matrix = np.asarray(backward[0], dtype=float)
    backward_shift = np.asarray(backward[1], dtype=float)

    assert forward_matrix.shape == (3, 3)
    assert forward_shift.shape == (3,)
    assert backward_matrix.shape == (3, 3)
    assert backward_shift.shape == (3,)
    assert np.allclose(backward_matrix @ forward_matrix, np.eye(3), atol=1e-8)

    residual_shift = backward_matrix @ forward_shift + backward_shift
    residual_shift = residual_shift - np.round(residual_shift)
    assert np.allclose(residual_shift, np.zeros(3), atol=1e-8)


def _assert_setting_transform_chain(first, second, target):
    first_matrix = np.asarray(first[0], dtype=float)
    first_shift = np.asarray(first[1], dtype=float)
    second_matrix = np.asarray(second[0], dtype=float)
    second_shift = np.asarray(second[1], dtype=float)
    target_matrix = np.asarray(target[0], dtype=float)
    target_shift = np.asarray(target[1], dtype=float)

    chained_matrix = second_matrix @ first_matrix
    chained_shift = second_matrix @ first_shift + second_shift
    assert np.allclose(chained_matrix, target_matrix, atol=1e-8)
    residual_shift = chained_shift - target_shift
    residual_shift = residual_shift - np.round(residual_shift)
    assert np.allclose(residual_shift, np.zeros(3), atol=1e-8)


def test_generated_poscar_lattice_vectors_use_nine_decimals():
    cell = cell_module.CrystalCell(
        lattice=np.array(
            [
                [1.2345678912, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.1, 0.2, 3.3333333333],
            ]
        ),
        positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]]),
        occupancies=[1.0, 1.0],
        elements=["Fe", "O"],
        moments=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    expected_lattice = [
        "1.234567891 0.000000000 0.000000000",
        "0.000000000 2.000000000 0.000000000",
        "0.100000000 0.200000000 3.333333333",
    ]

    poscar_outputs = [
        cell.to_poscar("case"),
        find_spin_group_module._cell_to_poscar_in_snapshot_order(
            cell,
            "case",
            site_order=[0, 1],
        ),
        find_spin_group_module._cell_to_poscar_preserving_lattice(cell, "case"),
    ]

    for poscar in poscar_outputs:
        assert poscar.splitlines()[2:5] == expected_lattice


def test_find_spin_group_basic_skips_tensor_and_scif_generation(monkeypatch):
    def _unexpected(*args, **kwargs):
        raise AssertionError("unexpected heavy-route call")

    monkeypatch.setattr(find_spin_group_module, "_compute_tensor_outputs", _unexpected)
    monkeypatch.setattr(find_spin_group_module, "generate_scif", _unexpected)

    payload = find_spin_group_basic("examples/0.800_MnTe.mcif")
    expected_spin_texture_config = get_spin_texture_config_for_ssg_label("194.164.1.1.L")

    assert payload["index"] == "194.164.1.1.L"
    assert payload["conf"] == "Collinear"
    assert payload["acc_symbol"] == "6/mmmP"
    assert payload["spin_texture_config_no_soc"]["spin_texture_type"] == "g-wave"
    assert payload["spin_texture_config_no_soc"]["source"] == "ossg_unit_cartesian_generators"
    assert payload["spin_texture_config_no_soc"]["basis_setting"] == "ossg_unit_cartesian"
    assert payload["spin_texture_config_soc"]["source"] == "ossg_unit_cartesian_msg_ops"
    assert payload["spin_texture_config_soc"]["basis_setting"] == "ossg_unit_cartesian"
    assert payload["spin_texture_config_database"] == expected_spin_texture_config
    assert "id" not in payload["spin_texture_config_database"]
    assert payload["quasi_2d"] is None
    assert payload["g0_number"] == 194
    assert payload["l0_number"] == 164
    assert payload["space_group_number"] == 194
    assert payload["nsspg"] == "-1"
    assert payload["sspg"] == "∞/mm"
    assert "msg_symbol" in payload


def test_configuration_details_matches_candidate_residual_minimum():
    moments = np.array(
        [
            [1.0, 0.0, 0.25],
            [0.0, 1.0, -0.5],
            [-1.0, 0.5, 0.75],
            [0.5, -1.0, -0.25],
            [0.25, 0.75, 1.0],
            [-0.75, -0.25, -1.0],
        ]
    )

    details = _configuration_details(moments, mtol=0.02)
    candidates = _candidate_directions_from_moments(moments)

    assert details["collinear_residual"] == pytest.approx(
        min(_collinear_residual(moments, direction) for direction in candidates)
    )
    assert details["coplanar_residual"] == pytest.approx(
        min(_coplanar_residual(moments, direction) for direction in candidates)
    )


def test_configuration_details_handles_large_noncollinear_moment_sets():
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    moments = np.column_stack(
        [
            np.cos(angles),
            np.sin(angles),
            0.5 * np.cos(5.0 * angles) + 0.25 * np.sin(7.0 * angles),
        ]
    )

    details = _configuration_details(moments, mtol=0.02)

    assert details["configuration"] == "Noncoplanar"
    assert details["collinear_residual"] > 0.02
    assert details["coplanar_residual"] > 0.02


def test_bucketed_spin_space_operation_dedup_matches_naive_tolerant_dedup():
    identity = np.eye(3)
    c2z = np.diag([-1.0, -1.0, 1.0])
    mirror_xy = np.diag([1.0, 1.0, -1.0])
    base_ops = [
        SpinSpaceGroupOperation(identity, identity, np.zeros(3)),
        SpinSpaceGroupOperation(c2z, c2z, np.array([0.5, 0.0, 0.0])),
        SpinSpaceGroupOperation(mirror_xy, c2z, np.array([0.0, 0.5, 0.0])),
    ]
    near_duplicates = [
        SpinSpaceGroupOperation(
            op[0] + 1e-8 * np.eye(3),
            op[1] - 1e-8 * np.eye(3),
            op[2] + np.array([1e-8, 0.0, 0.0]),
        )
        for op in base_ops
    ]
    ops = base_ops + near_duplicates

    expected = []
    for op in sorted(ops, key=op_key):
        if any(op.is_same_with(existing, atol=1e-5) for existing in expected):
            continue
        expected.append(op)

    actual = _deduplicate_spin_space_ops(ops, tol=1e-5, sort=True)

    assert len(actual) == len(expected)
    assert all(
        actual_op.is_same_with(expected_op, atol=1e-5)
        for actual_op, expected_op in zip(actual, expected)
    )


def test_cli_basic_mode_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["findspingroup", "examples/0.800_MnTe.mcif", "--mode", "basic"],
    )

    import findspingroup.cli as cli_module

    cli_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["index"] == "194.164.1.1.L"
    assert payload["conf"] == "Collinear"
    assert payload["acc_symbol"] == "6/mmmP"


def test_find_spin_group_acc_primitive_skips_tensor_and_scif_generation(monkeypatch):
    def _unexpected(*args, **kwargs):
        raise AssertionError("unexpected heavy-route call")

    monkeypatch.setattr(find_spin_group_module, "_compute_tensor_outputs", _unexpected)
    monkeypatch.setattr(find_spin_group_module, "generate_scif", _unexpected)

    payload = find_spin_group_acc_primitive("examples/0.800_MnTe.mcif")
    expected_spin_texture_config = get_spin_texture_config_for_ssg_label("194.164.1.1.L")

    assert payload["index"] == "194.164.1.1.L"
    assert payload["conf"] == "Collinear"
    assert payload["acc_symbol"] == "6/mmmP"
    assert payload["spin_texture_config_database"] == expected_spin_texture_config
    assert "id" not in payload["spin_texture_config_database"]
    assert payload["spin_texture_config_no_soc"]["source"] == "ossg_unit_cartesian_generators"
    assert payload["spin_texture_config_no_soc"]["basis_setting"] == "ossg_unit_cartesian"
    assert payload["spin_texture_config_no_soc"]["spin_texture_type"] == expected_spin_texture_config["spin_texture_type"]
    assert payload["spin_texture_config_soc"]["source"] == "ossg_unit_cartesian_msg_ops"
    assert payload["acc_primitive_cell_setting"] == "acc_primitive"
    assert payload["acc_primitive_cell_detail"] is not None
    assert payload["acc_primitive_poscar"]
    assert payload["acc_primitive_ssg_operation_matrices"]
    assert payload["acc_primitive_ssg_ops_cartesian"]
    assert payload["acc_primitive_ssg_seitz_cartesian"]
    assert payload["acc_primitive_ssg_seitz_latex_cartesian"]
    assert payload["acc_primitive_ssg_ops_oriented"]
    assert payload["acc_primitive_ssg_seitz_oriented"]
    assert payload["acc_primitive_ssg_seitz_latex_oriented"]
    assert sorted(payload["operation_views"]) == [
        "magnetic_primitive_cartesian",
        "magnetic_primitive_oriented",
    ]
    assert payload["operation_views"]["magnetic_primitive_cartesian"]["default_view"] == "nssg"
    cartesian_views = payload["operation_views"]["magnetic_primitive_cartesian"]["views"]
    assert cartesian_views["all"]["ops"] == cartesian_views["nssg"]["ops"]
    assert cartesian_views["all"]["seitz_latex"] == cartesian_views["nssg"]["seitz_latex"]
    assert cartesian_views["nssg"]["note"]["spin_only_symbol_hm"] == "∞m"
    assert cartesian_views["nssg"]["note"]["spin_only_symbol_s"] == "C∞v"
    acc_primitive_full_ops = [
        [
            op["spin_rotation"],
            op["real_rotation"],
            op["translation"],
        ]
        for op in payload["acc_primitive_ssg_ops_cartesian"]
    ]
    assert len(cartesian_views["all"]["ops"]) == len(
        SpinSpaceGroup(acc_primitive_full_ops).nssg
    )
    assert len(cartesian_views["all"]["seitz_latex"]) == len(cartesian_views["all"]["ops"])
    assert "ops" not in cartesian_views["generators"]
    assert cartesian_views["generators"]["indices"]
    assert max(cartesian_views["generators"]["indices"]) <= len(
        cartesian_views["all"]["ops"]
    )
    assert "ops" not in cartesian_views["pure_translations"]
    assert cartesian_views["pure_translations"]["indices"]
    assert max(cartesian_views["pure_translations"]["indices"]) <= len(
        cartesian_views["all"]["ops"]
    )
    assert payload["acc_primitive_poscar_spin_frame_ssg_operation_matrices"]
    assert payload["acc_primitive_spin_only_direction_cartesian"] == "1/2,sqrt(3)/2,0"
    assert payload["acc_primitive_spin_only_direction_poscar_spin_frame"] == "1/2,sqrt(3)/2,0"
    assert payload["acc_primitive_wp_chain"]
    assert np.asarray(payload["T_input_to_acc_primitive"][0], dtype=float).shape == (3, 3)
    assert np.asarray(payload["T_input_to_acc_primitive"][1], dtype=float).shape == (3,)
    assert np.asarray(payload["T_acc_primitive_to_G0std"][0], dtype=float).shape == (3, 3)
    assert np.asarray(payload["T_acc_primitive_to_G0std"][1], dtype=float).shape == (3,)
    assert np.asarray(payload["T_acc_primitive_to_L0std"][0], dtype=float).shape == (3, 3)
    assert np.asarray(payload["T_acc_primitive_to_L0std"][1], dtype=float).shape == (3,)


def test_find_spin_group_acc_primitive_oriented_seitz_handles_0427_nonorthogonal_spin_frame():
    payload = find_spin_group_acc_primitive(
        "tests/testset/mcif_241130_no2186/0.427_Sm2Ti2O7.mcif"
    )

    assert payload["index"] == "227.2.1.2"
    assert payload["conf"] == "Noncoplanar"
    assert len(payload["acc_primitive_ssg_ops_oriented"]) == len(
        payload["acc_primitive_ssg_seitz_oriented"]
    )
    assert any("-4" in symbol for symbol in payload["acc_primitive_ssg_seitz_oriented"])


def test_acc_primitive_oriented_seitz_uses_same_spin_frame_as_oriented_ops_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert len(result.acc_primitive_ssg_ops_oriented) == len(
        result.acc_primitive_ssg_seitz_latex_oriented
    )
    assert r"2_{112}" in result.acc_primitive_ssg_seitz_latex_oriented[1]
    assert r"2_{001}" not in result.acc_primitive_ssg_seitz_latex_oriented[1]
    assert r"3^{1}_{110}" in result.acc_primitive_ssg_seitz_latex_oriented[10]
    assert r"3^{2}_{111}" in result.acc_primitive_ssg_seitz_latex_oriented[10]


def test_find_spin_group_acc_primitive_oriented_seitz_uses_acc_lattice_frame_for_324():
    payload = find_spin_group_acc_primitive(
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif"
    )

    assert r"2_{112}" in payload["acc_primitive_ssg_seitz_latex_oriented"][1]
    assert r"2_{001}" not in payload["acc_primitive_ssg_seitz_latex_oriented"][1]


def test_write_ssg_operation_matrices_writes_json(tmp_path):
    payload = find_spin_group_acc_primitive("examples/0.800_MnTe.mcif")
    output_path = tmp_path / "acc_primitive_ops.json"

    write_ssg_operation_matrices(output_path, payload["acc_primitive_ssg_operation_matrices"])

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert loaded
    assert sorted(loaded[0].keys()) == ["index", "real_rotation", "spin_rotation", "translation"]


def test_cli_acc_primitive_mode_prints_json_and_writes_matrix_file(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "ops.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "findspingroup",
            "examples/0.800_MnTe.mcif",
            "--mode",
            "acc-primitive",
            "--write-ssg-matrices",
            str(output_path),
        ],
    )

    import findspingroup.cli as cli_module

    cli_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["index"] == "194.164.1.1.L"
    assert payload["acc_symbol"] == "6/mmmP"
    assert output_path.is_file()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(written) == len(payload["acc_primitive_ssg_operation_matrices"])


def test_cli_without_explicit_file_prefers_mcif_over_poscar_and_runs_basic(monkeypatch, capsys, tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    (tmp_path / "POSCAR").write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")
    (tmp_path / "other.mcif").write_text(Path("examples/0.800_MnTe.mcif").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fsg"])

    import findspingroup.cli as cli_module

    cli_module.main()
    stdout = capsys.readouterr()

    assert "Index: 194.164.1.1.L" in stdout.out
    assert "Spin arithmetic crystal class: 6/mmmP" in stdout.out
    assert "Magnetic phase: AFM(Altermagnet)" in stdout.out
    assert "Using other.mcif" in stdout.err or "Auto-selected structure file: other.mcif" in stdout.err


def test_cli_auto_selects_only_magnetic_structure_candidates(monkeypatch, tmp_path):
    import findspingroup.cli as cli_module

    (tmp_path / "plain.cif").write_text("_cell_length_a 1\n", encoding="utf-8")
    (tmp_path / "plain.poscar").write_text("no magnetic payload\n", encoding="utf-8")
    (tmp_path / "CONTCAR").write_text("last resort\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Path(cli_module._select_structure_file(None)).name == "CONTCAR"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {
                "sample.scif": "_space_group_spin.fsg_cell_setting input\n",
                "sample.mcif": "_atom_site_moment.label Fe1\n",
                "magnetic.cif": "_atom_site_moment.label Fe1\n",
                "POSCAR": "# MAGMOM=1\n",
                "layer.vasp": "# MAGMOM=1\n",
                "CONTCAR": "fallback\n",
            },
            "sample.scif",
        ),
        (
            {
                "sample.mcif": "_atom_site_moment.label Fe1\n",
                "magnetic.cif": "_atom_site_moment.label Fe1\n",
                "POSCAR": "# MAGMOM=1\n",
            },
            "sample.mcif",
        ),
        (
            {
                "magnetic.cif": "_atom_site_spin_moment.axis_u 1\n",
                "POSCAR": "no embedded magmom\n",
                "INCAR": "MAGMOM = 1 -1\n",
            },
            "magnetic.cif",
        ),
        (
            {
                "POSCAR": "no embedded magmom\n",
                "INCAR": "MAGMOM = 1 -1\n",
                "layer.poscar": "# MAGMOM=1 -1\n",
            },
            "POSCAR",
        ),
        (
            {
                "POSCAR": "# MAGMOM=1 -1\n",
                "layer.vasp": "# MAGMOM=1 -1\n",
                "CONTCAR": "fallback\n",
            },
            "POSCAR",
        ),
        (
            {
                "layer.vasp": "# MAGMOM=1 -1\n",
                "CONTCAR": "fallback\n",
            },
            "layer.vasp",
        ),
    ],
)
def test_cli_auto_select_priority(monkeypatch, tmp_path, files, expected):
    import findspingroup.cli as cli_module

    for filename, content in files.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Path(cli_module._select_structure_file(None)).name == expected


def test_cli_write_outputs_input_ssg_bundle_into_current_directory(monkeypatch, capsys, tmp_path):
    source = Path("tests/testset/mcif_241130_no2186/0.396_MnPtGa.mcif")
    target = tmp_path / source.name
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fsg", "-w", target.name])

    import findspingroup.cli as cli_module

    cli_module.main()
    stdout = json.loads(capsys.readouterr().out)

    assert stdout["written_files"] == [
        "ssg_symm.json",
        "input_poscar.vasp",
        "magnetic_primitive_poscar.vasp",
    ]
    assert stdout["summary"]["input_ssg_index"] == "63.12.1.2.P2"
    assert (tmp_path / "ssg_symm.json").is_file()
    assert (tmp_path / "input_poscar.vasp").is_file()
    assert (tmp_path / "magnetic_primitive_poscar.vasp").is_file()


def test_cli_all_show_filters_full_route_fields(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fsg", "--all", "--show", "msg_symbol", "examples/0.800_MnTe.mcif"],
    )

    import findspingroup.cli as cli_module

    cli_module.main()
    assert capsys.readouterr().out.strip() == "Cmcm"


def test_cli_show_field_aliases_full_route(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    class _FakeResult:
        def to_dict(self):
            return {
                "KPOINTS": "KPOINTS text\n",
                "acc_primitive_magnetic_cell_poscar": "POSCAR text\n",
            }

    monkeypatch.setattr(cli_module, "find_spin_group", lambda *_args, **_kwargs: _FakeResult())
    monkeypatch.setattr(sys, "argv", ["fsg", "--all", "--show", "kpoints", "dummy.mcif"])

    cli_module.main()

    assert capsys.readouterr().out == "KPOINTS text\n"


def test_cli_show_formats_dict_fields_readably(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    class _FakeResult:
        def to_dict(self):
            return {
                "spin_texture_config_no_soc": {
                    "spin_texture_type": "d-wave",
                    "momentum_space_spin_configuration": "coplanar",
                    "order": 2,
                    "basis": ["C1*kx*sigma_y"],
                }
            }

    monkeypatch.setattr(cli_module, "find_spin_group", lambda *_args, **_kwargs: _FakeResult())
    monkeypatch.setattr(sys, "argv", ["fsg", "--all", "--show", "spin-texture-no-soc", "dummy.mcif"])

    cli_module.main()

    output = capsys.readouterr().out
    assert "spin_texture_type: d-wave" in output
    assert "momentum_space_spin_configuration: coplanar" in output
    assert "basis:" in output
    assert "1. C1*kx*sigma_y" in output


def test_cli_show_json_keeps_machine_readable_payload(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    class _FakeResult:
        def to_dict(self):
            return {"spin_texture_config_no_soc": {"basis": ["C1*kx*sigma_y"]}}

    monkeypatch.setattr(cli_module, "find_spin_group", lambda *_args, **_kwargs: _FakeResult())
    monkeypatch.setattr(
        sys,
        "argv",
        ["fsg", "--all", "--json", "--show", "spin-texture-no-soc", "dummy.mcif"],
    )

    cli_module.main()

    assert json.loads(capsys.readouterr().out) == {"basis": ["C1*kx*sigma_y"]}


def test_cli_write_scif_and_poscar_kpoints_use_full_route_once(monkeypatch, capsys, tmp_path):
    import findspingroup.cli as cli_module

    calls = []

    class _FakeResult:
        index = "1.1.1.1.P1"
        conf = "Collinear"
        phase = "AFM"
        magnetic_phase = "AFM"
        KPOINTS_setting = "acc_primitive"
        KPOINTS_real_space_setting = "acc_primitive"
        acc_primitive_magnetic_cell_poscar = "POSCAR text\n"
        KPOINTS = "KPOINTS text\n"

        def to_scif(self, *, cell_mode):
            return f"SCIF {cell_mode}\n"

    def _fake_find_spin_group(path, **kwargs):
        calls.append((path, kwargs))
        return _FakeResult()

    scif_path = tmp_path / "structure.scif"
    vasp_dir = tmp_path / "vasp"
    monkeypatch.setattr(cli_module, "find_spin_group", _fake_find_spin_group)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsg",
            "--write-scif",
            str(scif_path),
            "--scif-cell-mode",
            "magnetic_primitive_oriented",
            "--write-poscar-kpoints",
            str(vasp_dir),
            "dummy.mcif",
        ],
    )

    cli_module.main()

    stdout = json.loads(capsys.readouterr().out)
    assert len(calls) == 1
    assert scif_path.read_text(encoding="utf-8") == "SCIF magnetic_primitive_oriented\n"
    assert (vasp_dir / "POSCAR").read_text(encoding="utf-8") == "POSCAR text\n"
    assert (vasp_dir / "KPOINTS").read_text(encoding="utf-8") == "KPOINTS text\n"
    assert stdout["written_files"] == [str(scif_path), str(vasp_dir / "POSCAR"), str(vasp_dir / "KPOINTS")]
    assert stdout["summary"]["index"] == "1.1.1.1.P1"
    assert calls[0][1]["poscar_allow_incar_magmom"] is True
    assert calls[0][1]["poscar_prefer_incar_magmom"] is True


def test_cli_rejects_show_with_artifact_writer(monkeypatch, capsys, tmp_path):
    import findspingroup.cli as cli_module

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsg", "--write-scif", str(tmp_path / "out.scif"), "--show", "index", "dummy.mcif"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 1
    assert "Write-artifact flags cannot be combined" in capsys.readouterr().err


def test_cli_accepts_hyphen_tolerance_aliases_and_forwards_full_route(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    captured = {}

    class _FakeResult:
        def to_dict(self):
            return {"ok": True}

    def _fake_find_spin_group(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(cli_module, "find_spin_group", _fake_find_spin_group)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsg",
            "--all",
            "--space-tol",
            "0.03",
            "--mtol",
            "0.04",
            "--meigtol",
            "0.0001",
            "--matrix-tol",
            "0.02",
            "--parser-atol",
            "0.05",
            "--calculation-mode",
            "quasi2d",
            "--vacuum-axis",
            "b",
            "--spin-texture-basis-max-order",
            "4",
            "dummy.mcif",
        ],
    )

    cli_module.main()

    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert captured == {
        "path": "dummy.mcif",
        "space_tol": pytest.approx(0.03),
        "mtol": pytest.approx(0.04),
        "meigtol": pytest.approx(0.0001),
        "matrix_tol": pytest.approx(0.02),
        "parser_atol": pytest.approx(0.05),
        "calculation_mode": "quasi2d",
        "vacuum_axis": "b",
        "spin_texture_basis_max_order": 4,
        "poscar_allow_incar_magmom": True,
        "poscar_prefer_incar_magmom": True,
    }


def test_cli_default_route_forwards_basic_without_quasi2d_options(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    captured = {}

    def _fake_find_spin_group_basic(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {
            "index": "1.1.1.1.P1",
            "ossg_symbol_linear": "P 1",
            "identify_index_details": {"t_index": 1, "k_index": 1},
            "g0_number": 1,
            "g0_symbol": "P1",
            "l0_number": 1,
            "l0_symbol": "P1",
            "nsspg": "1",
            "sspg": "1",
            "nontrivial_spin_space_point_group_hm": "1",
            "nontrivial_spin_space_point_group_schoenflies": "C1",
            "spin_space_point_group_hm": "1",
            "spin_space_point_group_schoenflies": "C1",
            "conf": "Collinear",
            "magnetic_phase": "AFM",
            "msg_bns_number": "1.1",
            "msg_og_number": "1.1.1",
            "msg_type": 1,
            "msg_symbol": "P1",
            "acc_symbol": "1P",
            "empg": "1",
            "space_group_number": 1,
            "space_group_symbol": "P1",
            "net_moment": 0.0,
            "zero_net_moment_tol": 0.04,
            "properties": {"ss_wo_soc": "No", "ss_w_soc": "No", "ahc_wo_soc": "No", "ahc_w_soc": "No"},
            "is_alter": "",
            "is_som": "",
            "sg_is_polar": False,
            "sg_is_chiral": False,
            "ssg_is_polar": False,
            "ssg_is_chiral": False,
            "msg_is_polar": False,
            "msg_is_chiral": False,
            "spin_texture_config_database": {"spin_texture_type": "forbidden"},
            "spin_texture_config_no_soc": {"spin_texture_type": "s-wave", "basis": ["C1*sigma_z"]},
            "spin_texture_config_soc": {"spin_texture_type": "p-wave", "basis": ["C1*kx*sigma_z"]},
            "vector_constraints_by_symmetry": {
                "sg": {
                    "constraints": {
                        "real_space_t_even_p_odd": {
                            "free_dimension": 0,
                            "allowed_axes": [],
                        }
                    }
                }
            },
            "ferroelectric_switching": {
                "status": "ordered_symmetry_nonpolar",
                "switching_detected": False,
                "polarity_status": "ordered_symmetry_nonpolar",
            },
        }

    monkeypatch.setattr(cli_module, "find_spin_group_basic", _fake_find_spin_group_basic)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsg",
            "--space-tol",
            "0.03",
            "--mtol",
            "0.04",
            "--meigtol",
            "0.0001",
            "--matrix-tol",
            "0.02",
            "--parser-atol",
            "0.05",
            "dummy.mcif",
        ],
    )

    cli_module.main()

    output = capsys.readouterr().out
    assert "OSSG symbol: P 1" in output
    assert "Index: 1.1.1.1.P1" in output
    assert "G0: 1 P1; L0: 1 P1; t_index: 1; k_index: 1" in output
    assert "Spin-space point group: HM=1; Schoenflies=C1" in output
    assert "Nontrivial spin-space point group: HM=1; Schoenflies=C1" in output
    assert "Configuration: Collinear" in output
    assert "Magnetic phase: AFM; altermagnet=No; spin-orbit magnet=No" in output
    assert "Magnetic space group: 1.1 P1 (type 1)" in output
    assert "Spin arithmetic crystal class: 1P" in output
    assert "EMPG: 1" in output
    assert "Net moment: 0.0 (zero tol 0.04)" in output
    assert "Spin texture database" not in output
    assert "Spin texture w/o SOC: wave=s-wave; basis=C1*sigma_z" in output
    assert "Spin texture w/ SOC: wave=p-wave; basis=C1*kx*sigma_z" in output
    assert "Symmetry flags:" not in output
    assert "SG: polar=No, chiral=No; T-even/P-odd real=0D none" in output
    assert "Polar axes:" not in output
    assert "Ferroelectric switching:" not in output
    assert captured == {
        "path": "dummy.mcif",
        "space_tol": pytest.approx(0.03),
        "mtol": pytest.approx(0.04),
        "meigtol": pytest.approx(0.0001),
        "matrix_tol": pytest.approx(0.02),
        "parser_atol": pytest.approx(0.05),
        "spin_texture_basis_max_order": None,
        "poscar_allow_incar_magmom": True,
        "poscar_prefer_incar_magmom": True,
    }
    assert "calculation_mode" not in captured
    assert "vacuum_axis" not in captured


def test_cli_default_route_json_flag_emits_complete_basic_payload(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    def _fake_find_spin_group_basic(path, **kwargs):
        return {"index": "1.1.1.1.P1", "route": "basic", "extra": {"kept": True}}

    monkeypatch.setattr(cli_module, "find_spin_group_basic", _fake_find_spin_group_basic)
    monkeypatch.setattr(sys, "argv", ["fsg", "--json", "dummy.mcif"])

    cli_module.main()

    assert json.loads(capsys.readouterr().out) == {
        "index": "1.1.1.1.P1",
        "route": "basic",
        "extra": {"kept": True},
    }


def test_cli_rejects_negative_spin_texture_basis_max_order(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsg", "--spin-texture-basis-max-order", "-1", "dummy.mcif"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 1
    assert "must be non-negative" in capsys.readouterr().err


def test_cli_legacy_poscar_ssg_dispatches_poscar_route(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    captured = {}

    def _fake_find_spin_group_poscar_ssg(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {"route": "poscar-ssg"}

    def _unexpected_input_ssg(*_args, **_kwargs):
        raise AssertionError("input-ssg route should not be used for --mode poscar-ssg")

    monkeypatch.setattr(cli_module, "find_spin_group_poscar_ssg", _fake_find_spin_group_poscar_ssg)
    monkeypatch.setattr(cli_module, "find_spin_group_input_ssg", _unexpected_input_ssg)
    monkeypatch.setattr(sys, "argv", ["fsg", "--mode", "poscar-ssg", "POSCAR"])

    cli_module.main()

    assert json.loads(capsys.readouterr().out) == {"route": "poscar-ssg"}
    assert captured == {
        "path": "POSCAR",
        "space_tol": pytest.approx(0.02),
        "mtol": pytest.approx(0.02),
        "meigtol": pytest.approx(0.00002),
        "matrix_tol": pytest.approx(0.01),
        "poscar_allow_incar_magmom": True,
        "poscar_prefer_incar_magmom": True,
    }


def test_cli_rejects_quasi2d_options_without_full_route(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsg", "--calculation-mode", "quasi2d", "dummy.mcif"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 1
    assert "only supported by the full route" in capsys.readouterr().err


def test_cli_rejects_legacy_writer_flags_outside_matching_modes(monkeypatch, capsys):
    import findspingroup.cli as cli_module

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsg", "--write-symmetry-dat", "ssg_symm.json", "dummy.mcif"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 1
    assert "`--write-symmetry-dat` is only valid" in capsys.readouterr().err


def test_find_spin_group_poscar_ssg_reports_embedded_magnetic_primitive_case(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    payload = find_spin_group_poscar_ssg(str(poscar_path))
    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(
        poscar_path,
        allow_incar_magmom=False,
        require_embedded_magmom=True,
    )
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="cartesian",
    )
    primitive_cell, _ = input_cell.get_primitive_structure(magnetic=True)
    primitive_identify = identify_spin_space_group_result(
        primitive_cell,
        find_primitive=False,
    )
    primitive_ossg = _ossg_oriented_spin_frame_ssg(primitive_identify.ssg, primitive_cell)

    assert payload["summary"]["is_input_magnetic_primitive"] is True
    assert payload["summary"]["input_ssg_may_be_incomplete"] is False
    assert payload["summary"]["warning"] is None
    assert payload["summary"]["input_ssg_index"] == original.index
    assert payload["summary"]["input_conf"] == original.conf
    assert payload["summary"]["input_magnetic_phase"] == original.magnetic_phase
    assert isinstance(payload["summary"]["input_spin_only_direction"], str)
    assert payload["summary"]["input_ssg_database_symbol"] is not None
    assert payload["summary"]["input_msg_num"] == primitive_ossg.msg_int_num
    assert payload["summary"]["input_msg_bns_number"] == primitive_ossg.msg_bns_num
    assert payload["summary"]["input_msg_symbol"] == primitive_ossg.msg_bns_symbol
    assert payload["summary"]["primitive_ssg_index"] == original.index
    assert payload["summary"]["primitive_msg_num"] == primitive_ossg.msg_int_num
    assert payload["summary"]["primitive_msg_bns_number"] == primitive_ossg.msg_bns_num
    assert payload["input_poscar"] is None
    assert payload["magnetic_primitive_poscar"]
    assert "# MAGMOM=" in payload["magnetic_primitive_poscar"]
    assert payload["ssg"]["ops"]
    assert payload["msg"]["ops"]


def test_find_spin_group_poscar_ssg_warns_for_nonprimitive_input_cell(tmp_path):
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.1000_Fe4O5.mcif")
    conventional_cell = CrystalCell(
        original.acc_conventional_cell_detail["lattice"],
        original.acc_conventional_cell_detail["positions"],
        original.acc_conventional_cell_detail["occupancies"],
        original.acc_conventional_cell_detail["elements"],
        original.acc_conventional_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(conventional_cell.to_poscar("Fe4O5_conventional"), encoding="utf-8")

    payload = find_spin_group_poscar_ssg(str(poscar_path))

    assert payload["summary"]["is_input_magnetic_primitive"] is False
    assert payload["summary"]["input_ssg_may_be_incomplete"] is True
    assert "not a magnetic primitive cell" in payload["summary"]["warning"]
    assert abs(payload["primitive_relation"]["determinant"]) != pytest.approx(1.0)
    assert payload["summary"]["input_conf"] is not None
    assert payload["summary"]["input_magnetic_phase"] is not None
    assert payload["summary"]["input_ssg_index"] is not None
    assert payload["summary"]["input_ssg_database_symbol"] is not None
    assert payload["summary"]["primitive_ssg_index"] is not None
    assert payload["summary"]["primitive_msg_bns_number"] is not None
    assert payload["input_poscar"] is None
    assert payload["magnetic_primitive_poscar"]
    assert "# MAGMOM=" in payload["magnetic_primitive_poscar"]
    assert payload["ssg"]["ops"]
    assert payload["msg"]["ops"]


def test_find_spin_group_poscar_ssg_requires_embedded_magmom(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        "\n".join(original.acc_primitive_magnetic_cell_poscar.splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embedded MAGMOM payload"):
        find_spin_group_poscar_ssg(str(poscar_path))


def test_find_spin_group_input_ssg_emits_input_poscar_for_mcif_input():
    payload = find_spin_group_input_ssg("examples/0.800_MnTe.mcif")

    assert payload["summary"]["input_ssg_index"] == "194.164.1.1.L"
    assert payload["quasi_2d"] is None
    assert payload["ssg"]["spin_frame_setting"] == "cartesian"
    if payload["summary"]["is_input_magnetic_primitive"]:
        assert payload["input_poscar"] is None
    else:
        assert payload["input_poscar"]
        assert "# MAGMOM=" in payload["input_poscar"]
    assert payload["magnetic_primitive_poscar"]
    assert "# MAGMOM=" in payload["magnetic_primitive_poscar"]


def test_find_spin_group_input_ssg_rejects_cif_without_explicit_moments(tmp_path):
    source = Path("tests/testset/structure.cif").read_text(encoding="utf-8")
    marker = "loop_\n_atom_site_moment.label"
    no_moment_cif = source.split(marker, 1)[0].rstrip() + "\n"
    tmp_cif = Path(tmp_path) / "no_moment_structure.cif"
    tmp_cif.write_text(no_moment_cif, encoding="utf-8")

    with pytest.raises(ValueError, match="requires explicit magnetic moments"):
        find_spin_group_input_ssg(str(tmp_cif))


def test_find_spin_group_input_ssg_reports_distinct_primitive_index_for_nonprimitive_mcif():
    payload = find_spin_group_input_ssg("tests/testset/mcif_241130_no2186/0.396_MnPtGa.mcif")

    assert payload["summary"]["is_input_magnetic_primitive"] is False
    assert payload["summary"]["input_ssg_index"] == "63.12.1.2.P2"
    assert payload["summary"]["primitive_ssg_index"] == "194.164.1.2.P2"
    assert payload["summary"]["primitive_msg_bns_number"] == "63.462"
    assert payload["summary"]["input_ssg_index"] != payload["summary"]["primitive_ssg_index"]
    assert payload["input_poscar"]
    assert payload["magnetic_primitive_poscar"]


def test_find_spin_group_input_ssg_magnetic_primitive_poscar_preserves_lattice_setting(tmp_path):
    payload = find_spin_group_input_ssg("tests/testset/mcif_241130_no2186/2.18_Sc2NiMnO6.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(payload["magnetic_primitive_poscar"], encoding="utf-8")

    roundtrip = find_spin_group_input_ssg(str(poscar_path))

    assert payload["summary"]["is_input_magnetic_primitive"] is False
    assert payload["summary"]["primitive_ssg_index"] == "2.2.2.2.P1"
    assert roundtrip["summary"]["is_input_magnetic_primitive"] is True
    assert roundtrip["summary"]["primitive_ssg_index"] == payload["summary"]["primitive_ssg_index"]
    assert roundtrip["summary"]["primitive_msg_bns_number"] == payload["summary"]["primitive_msg_bns_number"]


def test_write_poscar_ssg_symmetry_dat_writes_structured_json(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    payload = find_spin_group_poscar_ssg(str(poscar_path))
    output_path = tmp_path / "ssg_symm.dat"

    write_poscar_ssg_symmetry_dat(output_path, payload)

    text = output_path.read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["format"] == "findspingroup.poscar_ssg.v1"
    assert document["summary"] == payload["summary"]
    assert document["ssg"] == payload["ssg"]
    assert document["msg"] == payload["msg"]
    assert document["primitive_relation"] == payload["primitive_relation"]
    assert text.index('"summary"') < text.index('"ssg"') < text.index('"msg"')
    lines = text.splitlines()
    spin_rotation_line = next(idx for idx, line in enumerate(lines) if '"spin_rotation": [' in line)
    assert lines[spin_rotation_line + 1].strip().startswith("[")
    assert lines[spin_rotation_line + 2].strip().startswith("[")
    assert lines[spin_rotation_line + 3].strip().startswith("[")


def _serialize_effective_mpg_ops(ops):
    return [
        [int(time_reversal), np.asarray(rotation, dtype=float).tolist()]
        for time_reversal, rotation in ops
    ]


def _serialize_msg_ops(ops):
    return [
        [
            int(time_reversal),
            np.asarray(rotation, dtype=float).tolist(),
            np.asarray(translation, dtype=float).tolist(),
        ]
        for time_reversal, rotation, translation in ops
    ]


def _serialize_rotation_ops(ops):
    return [np.asarray(rotation, dtype=float).tolist() for rotation in ops]


def _primitive_magnetic_cell_from_cif(path: str) -> CrystalCell:
    lattice_factors, positions, elements, occupancies, _labels, moments = parse_cif_file(path)
    primitive_cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    return primitive_cell


def _rotation_order(rotation: np.ndarray, *, max_order: int = 12, tol: float = 1e-6) -> int | None:
    power = np.eye(3)
    rotation = np.asarray(rotation, dtype=float)
    for order in range(1, max_order + 1):
        power = power @ rotation
        if np.allclose(power, np.eye(3), atol=tol):
            return order
    return None


def _effective_proper_axis_from_space_rotation(rotation: np.ndarray, *, tol: float = 1e-6) -> np.ndarray | None:
    rotation = np.asarray(rotation, dtype=float)
    effective = rotation if np.linalg.det(rotation) > 0 else -rotation
    eigenvalues, eigenvectors = np.linalg.eig(effective)
    matches = np.isclose(eigenvalues, 1.0, atol=tol)
    if not np.any(matches):
        return None
    axis = eigenvectors[:, matches][:, 0].real
    axis = axis / np.linalg.norm(axis)
    for value in axis:
        if abs(value) > tol:
            if value < 0:
                axis = -axis
            break
    return axis


def _serialize_ssg_ops(ops):
    return [
        [
            np.asarray(op[0], dtype=float).tolist(),
            np.asarray(op[1], dtype=float).tolist(),
            np.asarray(op[2], dtype=float).tolist(),
        ]
        for op in ops
    ]


def test_combine_parametric_solutions_uses_axis_named_single_free_variable_for_z_only():
    rref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert combine_parametric_solutions(rref) == ["0", "0", "Sz"]


def test_combine_parametric_solutions_uses_axis_named_single_free_variable_for_first_nonzero_component():
    # x = 0, y - 0.57 z = 0 -> (0, 0.57*t, t)
    rref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -0.57]])
    assert combine_parametric_solutions(rref) == ["0", "0.57*Sy", "Sy"]


def test_combine_parametric_solutions_snaps_common_sqrt_coefficients():
    rref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -0.57735200925825]])
    assert combine_parametric_solutions(rref) == ["0", "sqrt(3)/3*Sy", "Sy"]


def test_tensor_output_display_snaps_common_sqrt_coefficients():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    relations = result.tensor_outputs["AHE_wSOC"]["relations"]
    components = result.tensor_outputs["AHE_wSOC"]["components"]

    assert relations == [
        r"\sigma_{xz} = -sqrt(3)/3\sigma_{yz} = -\sigma_{zx} = sqrt(3)/3\sigma_{zy}"
    ]
    assert components == [
        ["0", "0", r"-sqrt(3)/3\sigma_{yz}"],
        ["0", "0", r"\sigma_{yz}"],
        [r"sqrt(3)/3\sigma_{yz}", r"-\sigma_{yz}", "0"],
    ]


def test_combine_parametric_solutions_keeps_multi_free_variable_ordering():
    rref = np.array([[1.0, 0.0, 0.0]])
    assert combine_parametric_solutions(rref) == ["0", "Sx", "Sy"]


def test_space_group_is_centrosymmetric_lookup_matches_basic_examples():
    assert space_group_is_centrosymmetric(1) is False
    assert space_group_is_centrosymmetric(2) is True
    assert space_group_is_centrosymmetric(33) is False
    assert space_group_is_centrosymmetric(14) is True
    assert space_group_is_centrosymmetric(110) is False
    assert space_group_is_centrosymmetric(123) is True


def test_space_group_is_polar_lookup_matches_reference_examples():
    for sg in [1, 3, 4, 5, 25, 75, 99, 143, 156, 168, 183]:
        assert space_group_is_polar(sg) is True
    for sg in [2, 14, 16, 47, 62, 123, 195, 207]:
        assert space_group_is_polar(sg) is False


def test_space_group_polar_axis_basis_matches_reference_examples():
    assert space_group_polar_axis_basis(33) == ((0.0, 0.0, 1.0),)
    assert space_group_polar_axis_basis(3) == ((0.0, 1.0, 0.0),)
    assert space_group_polar_axis_basis(1) == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert space_group_polar_axis_basis(14) == ()


def test_space_group_is_chiral_lookup_matches_reference_examples():
    for sg in [1, 3, 4, 5, 16, 75, 143, 168, 195, 207, 214]:
        assert space_group_is_chiral(sg) is True
    for sg in [2, 14, 25, 47, 62, 99, 156, 183]:
        assert space_group_is_chiral(sg) is False


def test_msg_bns_and_og_first_segments_agree_on_centrosymmetric_rule():
    from findspingroup.data import MSGMPG_DB

    for msg_num in MSGMPG_DB.MSG_INT_TO_BNS:
        if msg_num is None:
            continue
        info = msg_parent_space_group_info(msg_num)
        assert space_group_is_centrosymmetric(
            info["bns_parent_space_group_number"]
        ) == space_group_is_centrosymmetric(
            info["og_parent_space_group_number"]
        )
        assert space_group_is_polar(
            info["bns_parent_space_group_number"]
        ) == space_group_is_polar(
            info["og_parent_space_group_number"]
        )
        assert space_group_is_chiral(
            info["bns_parent_space_group_number"]
        ) == space_group_is_chiral(
            info["og_parent_space_group_number"]
        )


def test_find_spin_group_exposes_centrosymmetric_flags_from_identified_numbers():
    noncentro = find_spin_group("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif")
    centro = find_spin_group("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif")

    assert noncentro.input_space_group_number == 33
    assert noncentro.sg_is_centrosymmetric is False
    assert noncentro.ossg_space_group_number == 33
    assert noncentro.ossg_is_centrosymmetric is False
    assert noncentro.msg_parent_space_group_number == 33
    assert noncentro.msg_is_centrosymmetric is False

    assert centro.input_space_group_number == 14
    assert centro.sg_is_centrosymmetric is True
    assert centro.ossg_space_group_number == 14
    assert centro.ossg_is_centrosymmetric is True
    assert centro.msg_parent_space_group_number == 14
    assert centro.msg_is_centrosymmetric is True


def test_find_spin_group_exposes_polar_and_chiral_flags_from_identified_numbers():
    polar_chiral = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    centro_nonpolar = find_spin_group("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif")

    assert polar_chiral.input_space_group_number == 182
    assert polar_chiral.sg_is_polar is False
    assert polar_chiral.sg_is_chiral is True
    assert polar_chiral.ossg_space_group_number == 182
    assert polar_chiral.ossg_is_polar is False
    assert polar_chiral.ossg_is_chiral is True
    assert polar_chiral.msg_parent_space_group_number == 150
    assert polar_chiral.msg_is_polar is False
    assert polar_chiral.msg_is_chiral is True

    assert centro_nonpolar.input_space_group_number == 14
    assert centro_nonpolar.sg_is_polar is False
    assert centro_nonpolar.sg_is_chiral is False
    assert centro_nonpolar.ossg_space_group_number == 14
    assert centro_nonpolar.ossg_is_polar is False
    assert centro_nonpolar.ossg_is_chiral is False
    assert centro_nonpolar.msg_parent_space_group_number == 14
    assert centro_nonpolar.msg_is_polar is False
    assert centro_nonpolar.msg_is_chiral is False


def test_ferroelectric_switching_payload_classifies_parent_ordered_routes():
    induced = build_ferroelectric_switching_payload(
        input_space_group_number=63,
        input_space_group_symbol="Cmcm",
        ssg_space_group_number=36,
        ossg_space_group_number=36,
        msg_parent_space_group_number=4,
        magnetic_phase="FM/FiM",
        magnetic_phase_base="FM/FiM",
        spin_splitting_without_soc="Zeeman",
        is_altermagnet="",
    )
    assert induced["polarity_status"] == "magnetically_induced_polar_candidate"
    assert induced["status"] == "candidate_requires_parent_ordered_coset"
    assert induced["switching_detected"] is None
    assert induced["structural_parent_symmetry"]["is_polar"] is False
    assert induced["parent_selection"]["default"] == "current_ordered_exact_parent"
    assert induced["parent_selection"]["high_temperature_parent_status"] == (
        "not_inferred_from_fsg_inputs"
    )
    assert induced["polarization_test_contract"]["mode"] == "polar_axis_basis_only"
    assert induced["secondary_order_parameter_contract"]["status"] == (
        "pending_definition_and_transport"
    )
    assert induced["secondary_order_parameter_contract"]["default_for_collinear_discussion"] == (
        "neel_vector"
    )
    assert induced["polarization_coupling_contract"]["magnetically_induced_polarization"][
        "current_status_value"
    ] == "magnetically_induced_polar_candidate"
    assert [
        branch["spin_space_operation"]
        for branch in induced["polarization_coupling_contract"]["collinear_spin_space_branches"]
    ] == ["+1", "-1"]
    assert [
        relation["label"]
        for relation in induced["polarization_coupling_contract"]["collinear_relation_classes"]
    ] == [
        "p_and_magnetic_order_reversed",
        "p_reversed_magnetic_order_preserved",
        "p_preserved_magnetic_order_reversed",
    ]
    assert induced["msg_compatibility_rule"]["exchange_only_label"] == (
        "valid_spin_space_operation_not_msg_compatible"
    )
    assert induced["domain_deduplication_contract"]["mode"] == (
        "transformed_magnetic_structure_equivalence"
    )
    assert induced["domain_deduplication_contract"]["domain_level_output"].startswith(
        "after magnetic-structure equivalence"
    )
    assert induced["claim_level_contract"]["current_positive_level"] == (
        "p_reversal_symmetry_candidate"
    )
    assert induced["translation_quotient_contract"]["status"] == (
        "implemented_by_signed_pattern_dedup_for_collinear_output"
    )
    assert induced["domain_relation_output_contract"]["candidate_reversal_domains_scope"] == (
        "P -> -P candidates only"
    )
    assert induced["domain_relation_output_contract"]["internal_descriptor"] == (
        "signed_collinear_magnetic_pattern"
    )
    assert induced["ordered_spin_space_symmetry"]["is_polar"] is True
    assert induced["soc_magnetic_symmetry"]["is_polar"] is True
    assert (
        induced["domain_switching_relation"]["switching_test"]
        == "real_space_coset_representative_maps_P_to_minus_P"
    )
    assert (
        induced["ferroelectric_altermagnet_screening"]["status"]
        == "not_candidate_no_k_dependent_nonrelativistic_spin_splitting"
    )

    nonpolar = build_ferroelectric_switching_payload(
        input_space_group_number=14,
        input_space_group_symbol="P2_1/c",
        ssg_space_group_number=14,
        ossg_space_group_number=14,
        msg_parent_space_group_number=14,
        magnetic_phase="AFM",
        magnetic_phase_base="AFM",
        spin_splitting_without_soc="No",
        is_altermagnet="",
    )
    assert nonpolar["polarity_status"] == "ordered_symmetry_nonpolar"
    assert nonpolar["status"] == "ordered_symmetry_forbids_polarization"
    assert nonpolar["switching_detected"] is False

    switchable_screening = build_ferroelectric_switching_payload(
        input_space_group_number=63,
        input_space_group_symbol="Cmcm",
        ssg_space_group_number=33,
        ossg_space_group_number=33,
        msg_parent_space_group_number=33,
        magnetic_configuration="Collinear",
        magnetic_phase="AFM(Altermagnet)",
        magnetic_phase_base="AFM",
        spin_splitting_without_soc="k-dependent",
        is_altermagnet="(Altermagnet)",
    )
    assert switchable_screening["ferroelectric_altermagnet_screening"]["status"] == "candidate"
    assert (
        switchable_screening["switchable_altermagnet_screening"]["status"]
        == "candidate_requires_p_s_coset_and_barrier_validation"
    )
    assert (
        switchable_screening["domain_reversal_symmetry_screening"]["status"]
        == "requires_parent_ordered_coset_validation"
    )
    assert (
        "test_optional_secondary_descriptor_transform_for_each_surviving_candidate"
        in switchable_screening["domain_reversal_symmetry_screening"]["candidate_operation_tests"]
    )
    assert (
        "which_minus_p_domain_is_selected_by_the_practical_electric_field_path"
        in switchable_screening["post_fsg_path_validation_requirements"]["checks"]
    )
    assert switchable_screening["energy_barrier_workflow"]["status"] == "not_computed_by_findspingroup"

    coplanar_screening = build_ferroelectric_switching_payload(
        input_space_group_number=63,
        input_space_group_symbol="Cmcm",
        ssg_space_group_number=36,
        ossg_space_group_number=36,
        msg_parent_space_group_number=4,
        magnetic_configuration="Coplanar",
    )
    assert coplanar_screening["polarity_status"] == "magnetically_induced_polar_candidate"
    assert coplanar_screening["status"] == "not_evaluated_coplanar_order_collinear_only"
    assert (
        coplanar_screening["domain_reversal_symmetry_screening"]["status"]
        == "not_evaluated_coplanar_order_collinear_only"
    )
    assert coplanar_screening["candidate_reversal_domains"] == []
    assert coplanar_screening["polarization_coupling_contract"]["scope"] == "collinear_only"

    parent_polar_transport = build_ferroelectric_switching_payload(
        input_space_group_number=33,
        input_space_group_symbol="Pna2_1",
        ssg_space_group_number=7,
        ossg_space_group_number=7,
        msg_parent_space_group_number=9,
    )
    assert (
        parent_polar_transport["polarity_status"]
        == "parent_polar_ordered_polar_transport_required"
    )
    assert (
        "parent_to_ordered_polar_axis_coordinate_transport"
        in parent_polar_transport["required_inputs_for_switching_claim"]
    )


def test_domain_reversal_coset_analysis_finds_parent_operation_flipping_polar_axis():
    identity = np.eye(3)
    mirror_x = np.diag([-1.0, 1.0, 1.0])
    mirror_y = np.diag([1.0, -1.0, 1.0])
    twofold_z = np.diag([-1.0, -1.0, 1.0])
    mirror_z = np.diag([1.0, 1.0, -1.0])
    inversion = -np.eye(3)
    twofold_y = np.diag([-1.0, 1.0, -1.0])
    twofold_x = np.diag([1.0, -1.0, -1.0])
    zero = np.zeros(3)

    analysis = build_domain_reversal_coset_analysis(
        parent_ops=[
            [identity, zero],
            [mirror_x, zero],
            [mirror_y, zero],
            [twofold_z, zero],
            [mirror_z, zero],
            [twofold_y, zero],
            [twofold_x, zero],
            [inversion, zero],
        ],
        ordered_ops=[
            [identity, zero],
            [mirror_x, zero],
            [mirror_y, zero],
            [twofold_z, zero],
        ],
        ordered_space_group_number=33,
        parent_space_group_number=47,
        parent_space_group_symbol="Pmmm",
        basis_setting="test_ordered_basis",
        tol=1e-8,
    )

    assert analysis["status"] == "candidate_reversal_domains_found"
    assert analysis["ordered_subset_of_parent"] is True
    assert analysis["left_coset_count"] == 2
    assert analysis["candidate_reversal_domain_count"] == 1
    candidate = analysis["candidate_reversal_domains"][0]
    assert candidate["maps_p_to_minus_p"] is True
    assert candidate["reversed_polar_axes"] == ["c"]
    assert candidate["representative"]["xyzt"] == "x,y,-z"
    assert candidate["representative_class"] == "spin_domain_relation_pending"


def test_generated_parent_standard_supercell_domain_coset_analysis_keeps_1048_full_representatives():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif")
    payload = result.ferroelectric_switching["domain_reversal_symmetry_screening"]
    assert payload["parent_group_source"] == (
        "spglib_standard_parent_lifted_to_ordered_standard_supercell"
    )
    assert payload["parent_action_scope"] == "parent_space_group_mod_ordered_translation_lattice"
    assert payload["parent_operation_count"] == 72
    assert payload["parent_grey_operation_count"] == 144
    assert payload["ordered_operation_count"] == 4
    assert payload["left_coset_count"] == 36
    assert set(payload["left_coset_sizes"]) == {4}
    assert payload["candidate_reversal_domain_count"] == 6
    assert payload["translation_quotient_status"] == (
        "physical_domains_deduplicated_by_signed_collinear_pattern"
    )
    assert payload["physical_reversal_domain_count"] == 6
    first_candidate = payload["candidate_reversal_domains"][0]
    assert [
        branch["spin_branch_relation_label"]
        for branch in first_candidate["collinear_branch_relations"]
    ] == [
        "p_reversed_spin_branch_preserved",
        "p_and_spin_branch_reversed",
    ]
    assert {
        branch["signed_collinear_pattern_relation"]
        for branch in first_candidate["collinear_branch_relations"]
    } == {"signed_pattern_changed"}
    assert {
        branch["representative_class"]
        for branch in first_candidate["collinear_branch_relations"]
    } == {"msg_compatible", "exchange_only"}
    assert payload["deduplicated_reversal_domains"][0]["soc_allowed_exists"] is True
    assert payload["deduplicated_reversal_domains"][0]["exchange_only_exists"] is True
    top_level_payload = result.ferroelectric_switching
    rows = top_level_payload["domain_relation_rows"]
    soc_rows = top_level_payload["soc_domain_relation_rows"]
    assert payload["domain_relation_representative_count"] == 36
    assert len(rows) == 36
    assert len(soc_rows) == 36
    assert top_level_payload["analysis_level"] == (
        "symmetry_only_parent_ordered_and_soc_msg_cosets_screened"
    )
    assert rows[0]["coset_index"] == first_candidate["coset_index"]
    assert rows[0]["coset_operation"].startswith("{+1 || -x,-y+1/6,-z,+1}")
    assert rows[0]["reverses_S"] is False
    assert rows[0]["reverses_P"] is True
    assert rows[1]["coset_operation"].startswith("{-1 || -x,-y+1/6,-z,-1}")
    assert rows[1]["reverses_S"] is True
    assert rows[1]["reverses_P"] is True
    assert any(row["reverses_P"] is False for row in rows)
    assert top_level_payload["domain_relation_text"].splitlines()[:4] == [
        "domain relations:",
        "No. xyzt uvw reverses_S reverses_P",
        "1 -x,-y+1/6,-z,+1 u,v,w N Y",
        "2 -x,-y+1/6,-z,-1 -u,-v,-w Y Y",
    ]
    assert top_level_payload["soc_domain_relation_text"].splitlines()[:4] == [
        "domain relations:",
        "No. xyzt uvw reverses_S reverses_P",
        "1 -x,-y+1/6,-z,+1 u,v,w N Y",
        "2 -x,-y+1/6,-z,-1 -u,-v,-w Y Y",
    ]


def test_soc_polar_branch_uses_full_msg_time_branch_cosets():
    soc_coset_payload = build_parent_standard_supercell_domain_coset_analysis(
        parent_space_group_number=2,
        parent_space_group_symbol="P-1",
        parent_hall_number=2,
        child_basis_in_parent=np.eye(3),
        child_origin_in_parent=np.zeros(3),
        ordered_magnetic_ops=[
            (np.eye(3), np.zeros(3), 1),
            (np.eye(3), np.zeros(3), -1),
        ],
        ordered_space_group_number=1,
        basis_setting="G0std",
        ordered_subgroup_source="soc_magnetic_space_group",
        relation_layer="soc_magnetic",
        subgroup_time_branch_scope="full",
    )

    assert soc_coset_payload["ordered_time_branch_scope"] == "full"
    assert soc_coset_payload["relation_layer"] == "soc_magnetic"
    assert soc_coset_payload["left_coset_count"] == 2
    assert soc_coset_payload["candidate_reversal_domain_count"] == 1

    payload = build_ferroelectric_switching_payload(
        input_space_group_number=2,
        input_space_group_symbol="P-1",
        ssg_space_group_number=2,
        ossg_space_group_number=2,
        msg_parent_space_group_number=1,
        magnetic_configuration="Collinear",
        soc_domain_reversal_coset_analysis=soc_coset_payload,
    )

    assert payload["polarity_status"] == "ordered_symmetry_nonpolar"
    assert payload["domain_relation_text"] is None
    assert (
        payload["soc_domain_reversal_symmetry_screening"]["status"]
        == "candidate_reversal_domains_found"
    )
    assert payload["soc_domain_relation_text"].splitlines()[:3] == [
        "domain relations:",
        "No. xyzt uvw reverses_S reverses_P",
        "1 -x,-y,-z,+1 u,v,w N Y",
    ]


def test_find_spin_group_exposes_conservative_ferroelectric_switching_payload():
    polar = find_spin_group("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif")
    nonpolar = find_spin_group("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif")
    induced = find_spin_group("tests/testset/mcif_241130_no2186/0.1000_Fe4O5.mcif")

    polar_payload = polar.ferroelectric_switching
    constraints = polar.vector_constraints_by_symmetry
    sg_constraint_keys = {
        "real_space_t_even_p_odd",
        "real_space_t_even_p_even",
    }
    spin_constraint_keys = {
        "real_space_t_even_p_odd",
        "real_space_t_odd_p_odd",
        "real_space_t_odd_p_even",
        "spin_space_t_odd_p_even",
        "real_space_t_even_p_even",
        "spin_space_t_even_p_even",
    }
    assert set(constraints) == {"sg", "ossg", "msg"}
    assert set(constraints["sg"]["constraints"]) == sg_constraint_keys
    assert set(constraints["ossg"]["constraints"]) == spin_constraint_keys
    assert set(constraints["msg"]["constraints"]) == spin_constraint_keys
    assert (
        constraints["ossg"]["constraints"]["real_space_t_even_p_odd"]["allowed_axes"]
        == polar_payload["allowed_polar_axes"]
    )
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes_setting"
        ]
        == "G0std"
    )
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes_source"
        ]
        == "space_group_standard_basis_transformed_to_ssg_convention"
    )
    assert polar.to_summary_dict()["vector_constraints_by_symmetry"] == constraints
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_even"][
            "allowed_axes_setting"
        ]
        == constraints["sg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes_setting"
        ]
    )
    assert (
        constraints["ossg"]["constraints"]["real_space_t_even_p_even"][
            "allowed_axes_setting"
        ]
        == constraints["ossg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes"
        ][0]["setting"]
    )
    assert polar_payload["analysis_level"] == "symmetry_only_collinear_switching_not_evaluated"
    assert polar_payload["polarity_status"] == "parent_polar_axis_preserved"
    assert polar_payload["status"].endswith("_collinear_only")
    assert polar_payload["switching_detected"] is None
    assert polar_payload["governing_symmetry"]["source"] == "ossg_real_space_projection"
    assert polar_payload["governing_symmetry"]["space_group_number"] == 33
    assert polar_payload["allowed_polar_axes"] == [
            {
                "label": "c",
                "components": [0.0, 0.0, 1.0],
                "setting": "G0std",
            }
        ]
    assert polar_payload["allowed_polar_axes_source"] == "real_space_operations"
    assert (
        constraints["msg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes_source"
        ]
        == "real_space_operations"
    )
    assert polar_payload["special_coset"]["status"] == "not_promoted_to_switching_claim"
    assert (
        polar_payload["domain_reversal_symmetry_screening"]["status"]
        == polar_payload["status"]
    )
    assert (
        polar_payload["ferroelectric_altermagnet_screening"]["status"]
        == "candidate_k_dependent_spin_splitting_not_flagged_altermagnet"
    )
    assert (
        polar_payload["switchable_altermagnet_screening"]["status"]
        == polar_payload["status"]
    )

    nonpolar_payload = nonpolar.ferroelectric_switching
    assert nonpolar_payload["polarity_status"] == "ordered_symmetry_nonpolar"
    assert nonpolar_payload["status"] == "ordered_symmetry_forbids_polarization"
    assert nonpolar_payload["switching_detected"] is False
    assert nonpolar_payload["allowed_polar_axes"] == []
    assert (
        nonpolar_payload["ferroelectric_altermagnet_screening"]["status"]
        == "not_candidate_ordered_symmetry_nonpolar"
    )

    induced_payload = induced.ferroelectric_switching
    assert induced_payload["polarity_status"] == "magnetically_induced_polar_candidate"
    assert induced_payload["status"] == "not_evaluated_coplanar_order_collinear_only"
    assert induced_payload["switching_detected"] is None
    assert induced_payload["structural_parent_symmetry"]["space_group_number"] == 63
    assert induced_payload["comparison_symmetry"]["input_space_group_number"] == 63
    assert (
        induced_payload["comparison_symmetry"][
            "current_ordered_exact_parent_space_group_number"
        ]
        == 63
    )
    assert induced_payload["domain_switching_relation"]["parent_group"] == (
        "current_ordered_exact_parent"
    )
    assert induced_payload["ordered_spin_space_symmetry"]["space_group_number"] == 36
    assert induced_payload["soc_magnetic_symmetry"]["space_group_number"] == 4
    assert (
        induced_payload["domain_reversal_symmetry_screening"]["status"]
        == "not_evaluated_coplanar_order_collinear_only"
    )
    assert induced_payload["candidate_reversal_domain_status"] == (
        "not_evaluated_coplanar_order_collinear_only"
    )
    assert induced_payload["candidate_reversal_domains"] == []
    assert induced_payload["polarization_coupling_contract"]["input_magnetic_configuration"] == (
        "Coplanar"
    )
    assert (
        induced_payload["ferroelectric_altermagnet_screening"]["status"]
        == "not_candidate_no_k_dependent_nonrelativistic_spin_splitting"
    )


def test_real_space_axial_axes_are_distinct_from_polar_axes():
    inversion_ops = [(-np.eye(3), np.zeros(3))]
    constraints = build_vector_constraints_by_symmetry_payload(
        sg_space_group_number=2,
        sg_real_space_ops=inversion_ops,
        sg_real_space_ops_setting="test_setting",
    )

    assert constraints["sg"]["constraints"]["real_space_t_even_p_odd"]["allowed_axes"] == []
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_even"]["free_dimension"]
        == 3
    )
    assert {
        axis["label"]
        for axis in constraints["sg"]["constraints"]["real_space_t_even_p_even"][
            "allowed_axes"
        ]
    } == {"a", "b", "c"}
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_even"]["constraint"]
        == "det(R) * R * v = v"
    )

    mirror_ops = [(np.diag([1.0, 1.0, -1.0]), np.zeros(3))]
    mirror_constraints = build_vector_constraints_by_symmetry_payload(
        sg_space_group_number=6,
        sg_space_group_symbol="Pm",
        sg_real_space_ops=mirror_ops,
        sg_real_space_ops_setting="test_setting",
    )
    assert mirror_constraints["sg"]["constraints"]["real_space_t_even_p_even"][
        "allowed_axes"
    ] == [
        {
            "label": "c",
            "components": [0.0, 0.0, 1.0],
            "setting": "test_setting",
        }
    ]

    spin_inversion_constraints = build_vector_constraints_by_symmetry_payload(
        ossg_symmetry={
            "source": "test_ossg",
            "space_group_number": 1,
            "space_group_symbol": "P1",
            "is_polar": True,
            "is_centrosymmetric": False,
            "allowed_polar_axes": None,
            "allowed_polar_axes_setting": "test_real_setting",
            "allowed_polar_axes_source": "test",
        },
        ossg_real_space_ops=[(np.eye(3), np.zeros(3))],
        ossg_real_space_ops_setting="test_real_setting",
        ossg_spin_space_ops=[(-np.eye(3), np.eye(3), np.zeros(3))],
        ossg_spin_space_setting="test_spin_setting",
    )
    spin_constraints = spin_inversion_constraints["ossg"]["constraints"]
    assert spin_constraints["real_space_t_odd_p_odd"]["allowed_axes"] == []
    assert spin_constraints["real_space_t_odd_p_even"]["allowed_axes"] == []
    assert spin_constraints["spin_space_t_odd_p_even"]["allowed_axes"] == []
    assert {
        axis["label"]
        for axis in spin_constraints["spin_space_t_even_p_even"]["allowed_axes"]
    } == {"a", "b", "c"}


def test_nonmagnetic_sg_ops_are_database_ops_transformed_to_current_basis(monkeypatch):
    current_to_standard = np.diag([2.0, 1.0, 1.0])
    dataset = SimpleNamespace(
        hall_number=1,
        rotations=[np.eye(3, dtype=int)],
        translations=[np.zeros(3)],
        transformation_matrix=current_to_standard,
    )
    standard_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    standard_translation = np.array([0.5, 0.0, 0.0])
    calls = []

    class FakeCell:
        def to_spglib(self, *, mag):
            assert mag is False
            return "fake-spglib-cell"

    def fake_get_symmetry_dataset(cell, *, symprec):
        assert cell == "fake-spglib-cell"
        assert symprec == DEFAULT_TOL.space
        return dataset

    def fake_get_symmetry_from_database(hall_number):
        calls.append(hall_number)
        return {
            "rotations": [standard_rotation],
            "translations": [standard_translation],
        }

    find_spin_group_module = sys.modules[
        _nonmagnetic_space_group_real_space_ops_in_cell_basis.__module__
    ]
    monkeypatch.setattr(find_spin_group_module, "get_symmetry_dataset", fake_get_symmetry_dataset)
    monkeypatch.setattr(
        find_spin_group_module,
        "get_symmetry_from_database",
        fake_get_symmetry_from_database,
    )

    ops = _nonmagnetic_space_group_real_space_ops_in_cell_basis(
        FakeCell(),
        tol_cfg=DEFAULT_TOL,
    )

    expected_rotation = np.linalg.inv(current_to_standard) @ standard_rotation @ current_to_standard
    expected_translation = np.mod(
        np.linalg.inv(current_to_standard) @ standard_translation,
        1.0,
    )
    assert calls == [1]
    assert len(ops) == 1
    assert np.allclose(ops[0][0], expected_rotation)
    assert np.allclose(ops[0][1], expected_translation)


def test_find_spin_group_basic_exposes_ferroelectric_switching_payload():
    result = find_spin_group_basic("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif")

    payload = result["ferroelectric_switching"]
    constraints = result["vector_constraints_by_symmetry"]
    assert set(constraints) == {"sg", "ossg", "msg"}
    assert (
        constraints["ossg"]["constraints"]["real_space_t_even_p_odd"]["allowed_axes"]
        == payload["allowed_polar_axes"]
    )
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes_setting"
        ]
        == "acc_primitive"
    )
    assert (
        constraints["sg"]["constraints"]["real_space_t_even_p_even"][
            "allowed_axes_setting"
        ]
        == constraints["sg"]["constraints"]["real_space_t_even_p_odd"][
            "allowed_axes_setting"
        ]
    )
    assert (
        constraints["ossg"]["constraints"]["real_space_t_even_p_even"][
            "allowed_axes_setting"
        ]
        == "acc_primitive"
    )
    assert payload["polarity_status"] == "parent_polar_axis_preserved"
    assert payload["comparison_symmetry"]["ssg_space_group_number"] == 33
    assert payload["comparison_symmetry"]["ossg_space_group_number"] is None
    assert payload["ordered_spin_space_symmetry"]["source"] == "ssg_g0_real_space_projection"
    assert constraints["ossg"]["source"] == "ssg_g0_real_space_projection"
    assert (
        payload["ferroelectric_altermagnet_screening"]["status"]
        == "candidate_k_dependent_spin_splitting_not_flagged_altermagnet"
    )


def test_generate_full_symmops_raises_on_runaway_non_group_closure():
    bad_generator = SymmOp(np.diag([1.01, 1.0, 1.0, 1.0]), tol=1e-8)
    with pytest.raises(ValueError, match="maximum generated operation count"):
        generate_full_symmops([bad_generator], tol=1e-8, max_generated_ops=20)


def test_find_spin_group_uses_acc_primitive_ossg_msg_info_for_public_msg_fields():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    internal_msg_info = get_magnetic_space_group_from_operations(result.primitive_msg_ops)

    assert result.msg_num == internal_msg_info["msg_int_num"]
    assert result.msg_type == internal_msg_info["msg_type"]
    assert result.msg_symbol == internal_msg_info["msg_bns_symbol"]


def test_find_spin_group_exposes_used_tolerances_in_result_metadata():
    result = find_spin_group(
        "examples/CoNb3S6_tripleQ.mcif",
        space_tol=0.03,
        mtol=0.05,
        meigtol=1e-4,
        matrix_tol=0.02,
        parser_atol=0.08,
    )

    assert result.tolerances == {
        "space_tol": 0.03,
        "mtol": 0.05,
        "meigtol": 1e-4,
        "matrix_tol": 0.02,
        "parser_atol": 0.08,
    }


def _identify_translation_vectors(details):
    return [
        np.asarray(item[1][1], dtype=float)
        for item in details["translation_maps"]
    ]


def _build_similarity_transformed_generator(standard_generator, transform):
    transform = np.asarray(transform, dtype=float)
    return transform @ np.asarray(standard_generator, dtype=float) @ np.linalg.inv(transform)


def test_analyze_transition_matrix_problem_reports_stable_candidate_for_mirror_group():
    mirror = np.diag([1.0, 1.0, -1.0])
    transform = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.0, 1.3, 0.2],
            [0.1, 0.0, 0.9],
        ],
        dtype=float,
    )
    transformed_mirror = _build_similarity_transformed_generator(mirror, transform)

    analysis = analyze_transition_matrix_problem([transformed_mirror], "m", id=True)

    assert analysis["group_symbol"] == "m"
    assert analysis["expected_null_space_dimension"] == 5
    assert analysis["null_space_dimension"] >= 1
    assert len(analysis["basis_metrics"]) == analysis["null_space_dimension"]
    assert analysis["best_candidate"] is not None
    assert analysis["best_candidate"]["passes_residual_tol"] is True
    assert analysis["best_candidate"]["sigma_min"] > 1e-8


def test_find_transition_matrix_deterministic_is_repeatable_for_near_mirror_generator():
    mirror = np.diag([1.0, 1.0, -1.0])
    transform = np.array(
        [
            [1.0, 0.25, 0.05],
            [0.0, 1.2, 0.15],
            [0.08, -0.03, 0.95],
        ],
        dtype=float,
    )
    transformed_mirror = _build_similarity_transformed_generator(mirror, transform)

    solution_a = find_transition_matrix_deterministic([transformed_mirror], "m", id=True)
    solution_b = find_transition_matrix_deterministic([transformed_mirror], "m", id=True)

    assert np.allclose(solution_a, solution_b, atol=1e-10)
    assert abs(np.linalg.det(solution_a)) > 1e-8


def test_find_spin_group_forwards_parser_atol_to_parse_structure_file(monkeypatch):
    captured = {}
    fake_parsed = (
        np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0], dtype=float),
        [np.zeros(3)],
        ["X"],
        [1.0],
        ["X1"],
        [np.zeros(3)],
    )

    def fake_parse_structure_file(
        filename,
        atol=0.01,
        return_metadata=False,
        **kwargs,
    ):
        captured["filename"] = filename
        captured["atol"] = atol
        captured["return_metadata"] = return_metadata
        captured["parse_kwargs"] = kwargs
        return fake_parsed, {"kind": "fake"}

    def fake_find_spin_group_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        source_metadata=None,
        parser_atol=None,
        input_spin_setting=None,
        calculation_mode=None,
        vacuum_axis=None,
        spin_texture_basis_max_order=None,
        components=None,
    ):
        captured["source_name"] = source_name
        captured["source_metadata"] = source_metadata
        captured["parser_atol"] = parser_atol
        captured["input_spin_setting"] = input_spin_setting
        captured["calculation_mode"] = calculation_mode
        captured["vacuum_axis"] = vacuum_axis
        captured["spin_texture_basis_max_order"] = spin_texture_basis_max_order
        captured["components"] = components
        return {"ok": True}

    monkeypatch.setattr(find_spin_group_module, "parse_structure_file", fake_parse_structure_file)
    monkeypatch.setattr(find_spin_group_module, "_find_spin_group_from_parsed", fake_find_spin_group_from_parsed)

    result = find_spin_group_module.find_spin_group("dummy.scif", parser_atol=0.123)

    assert result == {"ok": True}
    assert captured["filename"] == "dummy.scif"
    assert captured["atol"] == 0.123
    assert captured["return_metadata"] is True
    assert captured["parse_kwargs"] == {
        "poscar_allow_incar_magmom": False,
        "poscar_prefer_incar_magmom": False,
    }
    assert captured["source_name"] == "dummy.scif"
    assert captured["source_metadata"] == {"kind": "fake"}
    assert captured["parser_atol"] == 0.123
    assert captured["input_spin_setting"] == "in_lattice"
    assert captured["calculation_mode"] == "3d"
    assert captured["vacuum_axis"] == "c"
    assert captured["spin_texture_basis_max_order"] is None
    assert captured["components"] is None


def test_find_transition_matrix_deterministic_error_suggests_pg_standardization_direction(monkeypatch):
    mirror = np.diag([1.0, 1.0, -1.0])
    transform = np.array(
        [
            [1.0, 0.25, 0.05],
            [0.0, 1.2, 0.15],
            [0.08, -0.03, 0.95],
        ],
        dtype=float,
    )
    transformed_mirror = _build_similarity_transformed_generator(mirror, transform)

    monkeypatch.setattr(
        identify_symmetry_from_ops_module,
        "_search_transition_candidate",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError) as exc_info:
        find_transition_matrix_deterministic([transformed_mirror], "m", id=True)

    message = str(exc_info.value)
    assert "Unable to find a nonsingular matrix P in the null space." in message
    assert "find_spin_group(..., matrix_tol=...)" in message
    assert "meigtol=..." in message


def test_get_stand_trans_error_suggests_database_coverage_direction():
    with pytest.raises(ValueError) as exc_info:
        get_stand_trans(
            143,
            147,
            2,
            12,
            64,
            (np.identity(3), np.zeros(3)),
            [],
            [],
            tol=0.001,
        )

    message = str(exc_info.value)
    assert "No identify-index reduction record for L0=143, G0=147, it=2, ik=12, iso=64." in message
    assert "database/special-case coverage" in message
    assert "do not tune `space_tol`, `mtol`, `meigtol`, or `matrix_tol` first" in message


def _build_identify_standardization_debug(result):
    details = result.identify_index_details
    reduce_info = find_ssg_transformation(
        details["L0_id"],
        details["G0_id"],
        details["t_index"],
        details["k_index"],
        details["point_group_id"],
        make_4d_matrix(details["transformation_matrix"]),
        tol=0.01,
    )
    standardization_transform = np.linalg.inv(make_4d_matrix(reduce_info["TTM"]))
    transformed_name_maps = map_transformation(
        details["name_maps"],
        standardization_transform,
    )
    transformed_translation_maps = map_transformation(
        details["translation_maps"],
        standardization_transform,
    )
    standard_generator_maps = find_stand_gen_maps(
        transformed_name_maps,
        transformed_translation_maps,
        reduce_info["gen_matrices"],
        reduce_info["cell_size"],
    )
    database_standard_generators = [
        np.asarray(make_4d_matrix(generator), dtype=float)
        for generator in reduce_info["gen_matrices"]
    ]
    return {
        "identify_index_details": details,
        "reduce_info": reduce_info,
        "standardization_transform": np.asarray(
            standardization_transform,
            dtype=float,
        ),
        "transformed_name_maps": [
            {
                "point": np.asarray(item[0], dtype=float),
                "space": np.asarray(item[1], dtype=float),
            }
            for item in transformed_name_maps
        ],
        "transformed_translation_maps": [
            {
                "point": np.asarray(item[0], dtype=float),
                "space": np.asarray(item[1], dtype=float),
            }
            for item in transformed_translation_maps
        ],
        "standard_generator_maps": [
            {
                "point": np.asarray(item[0], dtype=float),
                "space": np.asarray(item[1], dtype=float),
            }
            for item in standard_generator_maps
        ],
        "database_standard_generators": database_standard_generators,
    }


def test_find_spin_group_exposes_main_flow_identify_result_for_collinear_case():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    expected_spin_texture_config = get_spin_texture_config_for_ssg_label("194.164.1.1.L")

    assert result.index == "194.164.1.1.L"
    assert result.conf == "Collinear"
    assert result.spin_part_point_group == "∞/mm"
    assert result.magnetic_phase == "AFM(Altermagnet)"
    assert result.magnetic_phase_base == "AFM"
    assert result.magnetic_phase_modifier == "(Altermagnet)"
    assert result.is_spin_orbit_magnet == ""
    assert result.magnetic_phase_details["classification_rule"] == "default_antiferromagnetic"
    assert result.magnetic_phase_details["is_altermagnet"] is True
    assert result.magnetic_phase_details["is_spin_orbit_magnet"] is False
    assert result.identify_index_details is not None
    assert result.identify_index_details["G0_id"] == 194
    assert result.identify_index_details["L0_id"] == 164
    assert result.identify_index_details["k_index"] == 1
    assert result.identify_index_details["equivalent_map_index"] == 1
    assert result.spin_texture_config_database == expected_spin_texture_config
    assert result.spin_texture_config_no_soc["source"] == "ossg_unit_cartesian_generators"
    assert result.spin_texture_config_no_soc["basis_setting"] == "ossg_unit_cartesian"
    assert result.spin_texture_config_no_soc["spin_texture_type"] == expected_spin_texture_config["spin_texture_type"]
    assert result.spin_texture_config_no_soc["order"] == expected_spin_texture_config["order"]
    assert result.spin_texture_config_no_soc["spin_rank"] == expected_spin_texture_config["spin_rank"]
    assert result.spin_texture_config_soc["source"] == "ossg_unit_cartesian_msg_ops"
    assert result.spin_texture_config_soc["spin_texture_type"] == "d-wave"
    assert "id" not in result.spin_texture_config_database
    assert result.to_summary_dict()["spin_texture_config_database"] == expected_spin_texture_config
    assert result.to_summary_dict()["spin_texture_config_no_soc"] == result.spin_texture_config_no_soc
    assert result.to_summary_dict()["spin_texture_config_soc"] == result.spin_texture_config_soc
    assert result.to_structured_dict()["summary"]["spin_texture_config_database"] == expected_spin_texture_config
    assert (
        result.to_structured_dict()["summary"]["spin_texture_config_no_soc"]
        == result.spin_texture_config_no_soc
    )
    assert result.to_structured_dict()["summary"]["spin_texture_config_soc"] == result.spin_texture_config_soc
    assert result.to_dict()["spin_texture_config_database"] == expected_spin_texture_config
    assert result.to_dict()["spin_texture_config_no_soc"] == result.spin_texture_config_no_soc
    assert result.to_dict()["spin_texture_config_soc"] == result.spin_texture_config_soc


def test_full_result_phase_alias_preserves_altermagnet_modifier():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1008_Sr2ErRuO6.mcif")
    payload = result.to_dict()

    assert result.index == "14.2.1.1.L"
    assert result.phase == "AFM(Altermagnet)\n(SOM)"
    assert result.phase == result.magnetic_phase
    assert payload["phase"] == payload["magnetic_phase"]
    assert payload["is_alter"] == "(Altermagnet)"
    assert result.magnetic_phase_details["spin_splitting_without_soc"] == "k-dependent"


def test_spin_space_group_index_is_lazy_identify_index_not_legacy_shorthand(monkeypatch):
    result = find_spin_group("examples/0.800_MnTe.mcif")
    ssg = SpinSpaceGroup(
        result.input_magnetic_primitive_ssg_ops,
        identify_source_name="examples/0.800_MnTe.mcif",
    )

    repr_text = repr(ssg)

    assert "index" not in ssg.__dict__
    assert "<SpinSpaceGroup #<unidentified> '" in repr_text
    with pytest.raises(TypeError):
        hash(ssg)
    assert ssg.index == result.index
    assert ssg.__dict__["index"] == "194.164.1.1.L"

    def _raise_point_group_map_error(*_args, **_kwargs):
        raise ValueError(
            "Cannot identify point-group map number for point_group=1, generator_numbers=[1]."
        )

    monkeypatch.setattr(
        find_spin_group_module,
        "_identify_ssg_index_details",
        _raise_point_group_map_error,
    )
    failing_ssg = SpinSpaceGroup(
        result.input_magnetic_primitive_ssg_ops,
        identify_source_name="examples/0.800_MnTe.mcif",
    )
    with pytest.raises(ValueError, match="Cannot identify point-group map number"):
        _ = failing_ssg.index
    assert "index" not in failing_ssg.__dict__


def test_find_spin_group_exposes_identify_transformations_for_coplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    details = result.identify_index_details

    assert result.index == "38.6.1.2.P2"
    assert result.conf == "Coplanar"
    assert details["G0_id"] == 38
    assert details["L0_id"] == 6
    assert details["t_index"] == 2
    assert details["k_index"] == 1
    assert details["point_group_id"] == 2
    assert details["equivalent_map_index"] == 2
    assert details["configuration_suffix"] == "P2"

    lattice_transform = details["transformation_matrix"]
    assert np.asarray(lattice_transform[0], dtype=float).shape == (3, 3)
    assert np.asarray(lattice_transform[1], dtype=float).shape == (3,)

    space_transform = details["space_group_transformation"]
    point_transform = np.asarray(details["point_group_transformation"], dtype=float)
    assert np.asarray(space_transform[0], dtype=float).shape == (3, 3)
    assert np.asarray(space_transform[1], dtype=float).shape == (3,)
    assert point_transform.shape == (3, 3)
    assert abs(np.linalg.det(np.asarray(space_transform[0], dtype=float))) > 1e-8
    assert abs(np.linalg.det(point_transform)) > 1e-8

    assert details["name_maps"]
    assert len(details["translation_maps"]) == 3


def test_identify_transformations_send_167_tmptin_to_database_symbol_with_spin_transform():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.67_TmPtIn.mcif")
    details = result.identify_index_details

    space_transform = details["space_group_transformation"]
    point_transform = np.asarray(details["point_group_transformation"], dtype=float)
    database_std_ssg = (
        SpinSpaceGroup(result.convention_ssg_ops)
        .transform(
            np.asarray(space_transform[0], dtype=float),
            np.asarray(space_transform[1], dtype=float),
        )
        .transform_spin(point_transform)
    )

    assert result.index == "25.8.2.1.P3"
    assert database_std_ssg.international_symbol_linear_current_frame == (
        "P 2_{001}|m 2_{100}|m 2_{010}|2 : (2_{001},1,2_{001}) m_{001}|1"
    )


def test_scif_chen_transform_contract_for_167_tmptin_omits_lattice_scaled_spin_rows():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.67_TmPtIn.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    assert metadata["space_group_spin"]["spin_space_group_number_chen"] == "25.8.2.1.P3"
    assert metadata["space_group_spin"]["spin_space_group_name_chen"] == (
        "P 1|m 2_{100}|m 1|2 : (2_{010},1,2_{010}) m_{010}|1"
    )
    assert metadata["space_group_spin"]["transform_Chen_Pp_abcs"] == "1/2b,-2a,c;0,1/2,0;cs,-bs,as"


@pytest.mark.parametrize(
    ("source_path", "expected_index", "expected_suffix"),
    [
        ("tests/testset/mcif_241130_no2186/0.1010_C10H6MnN4O4.mcif", "14.1.1.1.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/0.394_Cu2CdB2O6.mcif", "14.1.1.1.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif", "33.1.1.1.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/0.716_HoCrWO6.mcif", "33.1.1.1.P2", "P2"),
        ("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif", "14.2.2.1.P2", "P2"),
        ("tests/testset/mcif_241130_no2186/1.197_Fe4Si2Sn7O16.mcif", "12.2.2.1.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/1.647_Na2.4Ni2TeO6.mcif", "63.13.2.21.P3", "P3"),
        ("tests/testset/mcif_241130_no2186/2.96_GdMn2Si2.mcif", "139.115.2.1.P3", "P3"),
    ],
)
def test_find_spin_group_uses_excel_backed_suffixes_for_coplanar_d2_identify_branch(
    source_path,
    expected_index,
    expected_suffix,
):
    result = find_spin_group(source_path)
    details = result.identify_index_details

    assert result.conf == "Coplanar"
    assert result.index == expected_index
    assert details["configuration_suffix"] == expected_suffix


def test_build_candidate_transform_chen_pp_abcs_for_324_hex_spatial_cubic_spin():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    metadata = parse_scif_metadata(source_text=result.scif)

    transform_to_input = metadata["space_group_spin"]["transform_to_input_Pp"]
    basis_expr = transform_to_input.split(";", 1)[0]
    basis_matrices, _ = general_positions_to_matrix([basis_expr], variables=("a", "b", "c"))
    current_space_to_input_basis = np.asarray(basis_matrices[0][0], dtype=float)

    candidate = _build_candidate_transform_chen_pp_abcs_hex_spatial_cubic_spin_from_identify(
        current_space_to_input_basis=current_space_to_input_basis,
        identify_point_group_transformation=np.asarray(
            result.identify_index_details["point_group_transformation"],
            dtype=float,
        ),
    )

    assert candidate["from_spatial_setting"] == "current_scif_ssg_convention_oriented_hex"
    assert candidate["to_spatial_setting"] == "chen_hex_spatial"
    assert candidate["to_spin_frame"] == "chen_cubic_spin_basis"
    assert candidate["transform_Chen_Pp_abcs"] == (
        "a,b,c;0,0,0;"
        "4/3as+4/3bs+4/3cs,-1/3as+2/3bs-1/3cs,-2/3as+1/3bs+1/3cs"
    )

    spin_basis_columns = np.asarray(candidate["spin_basis_rows_abcs"], dtype=float).T
    spin_basis_columns_inv = np.linalg.inv(spin_basis_columns)

    current_point_ops = metadata["space_group_symop_spin_operation"]["uvw"][:6]
    point_op_matrices, _ = general_positions_to_matrix(current_point_ops, variables=("u", "v", "w"))
    transformed_point_ops = [
        np.round(spin_basis_columns_inv @ np.asarray(matrix, dtype=float) @ spin_basis_columns, 6)
        for matrix, _ in point_op_matrices
    ]
    expected_point_ops = [
        np.eye(3),
        np.array([[0.0, 0.25, 0.5], [-2.666667, 0.666667, 0.333333], [1.333333, -1.333333, -0.666667]]),
        np.array([[0.0, 0.25, 0.5], [-2.666667, 0.666667, 0.333333], [1.333333, -1.333333, -0.666667]]),
        np.array([[0.0, -0.5, -0.25], [-1.333333, -0.666667, -1.333333], [2.666667, 0.333333, 0.666667]]),
        np.array([[0.0, -0.5, -0.25], [-1.333333, -0.666667, -1.333333], [2.666667, 0.333333, 0.666667]]),
        np.eye(3),
    ]
    for actual, expected in zip(transformed_point_ops, expected_point_ops):
        assert np.allclose(actual, expected, atol=1e-6)

    current_lattice_ops = metadata["space_group_symop_spin_lattice"]["uvw"][:4]
    lattice_op_matrices, _ = general_positions_to_matrix(current_lattice_ops, variables=("u", "v", "w"))
    transformed_lattice_ops = [
        np.round(spin_basis_columns_inv @ np.asarray(matrix, dtype=float) @ spin_basis_columns, 6)
        for matrix, _ in lattice_op_matrices
    ]
    expected_lattice_ops = [
        np.eye(3),
        np.array([[-1.166667, 0.041667, 0.125], [3.333333, -1.833333, -2.5], [-4.0, 1.0, 2.0]]),
        np.array([[-1.166667, 0.041667, 0.125], [3.333333, -1.833333, -2.5], [-4.0, 1.0, 2.0]]),
        np.array([[0.944444, 0.680556, 0.486111], [-2.222222, -1.777778, -0.555556], [3.333333, 1.166667, -0.166667]]),
    ]
    for actual, expected in zip(transformed_lattice_ops, expected_lattice_ops):
        assert np.allclose(actual, expected, atol=1e-6)


def test_find_spin_group_uses_p1_branch_for_parallel_coplanar_order_two_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.317_La0.25Pr0.75Co2P2.mcif")
    details = result.identify_index_details

    assert result.index == "123.123.2.2.P1"
    assert result.conf == "Coplanar"
    assert details["point_group_id"] == 2
    assert details["equivalent_map_index"] == 2
    assert details["configuration_suffix"] == "P1"


@pytest.mark.parametrize(
    ("source_path", "expected_target"),
    [
        ("tests/testset/mcif_241130_no2186/1.302_Ba2CoO4.mcif", "My"),
        ("tests/testset/mcif_241130_no2186/0.425_Na2CoP2O7.mcif", "My"),
        ("tests/testset/mcif_241130_no2186/0.716_HoCrWO6.mcif", "Mx"),
    ],
)
def test_find_spin_group_exposes_total_coplanar_222_spin_transform(source_path, expected_target):
    result = find_spin_group(source_path)
    details = result.identify_index_details

    assert result.conf == "Coplanar"
    assert details["point_group_id"] == 14
    assert details["point_group_transformation_raw"] is not None
    assert details["coplanar_222_q_transform"] is not None
    assert details["coplanar_222_b_transform"] is not None
    assert details["coplanar_222_target_spin_only_matrix"] is not None
    assert details["coplanar_222_target_spin_only_label"] == expected_target

    raw = np.asarray(details["point_group_transformation_raw"], dtype=float)
    q_transform = np.asarray(details["coplanar_222_q_transform"], dtype=float)
    b_transform = np.asarray(details["coplanar_222_b_transform"], dtype=float)
    total = np.asarray(details["point_group_transformation"], dtype=float)
    target_matrix = np.asarray(details["coplanar_222_target_spin_only_matrix"], dtype=float)

    assert np.allclose(total, b_transform @ raw @ q_transform, atol=1e-6)

    mz = np.diag([1.0, 1.0, -1.0])
    assert np.allclose(
        b_transform @ mz @ np.linalg.inv(b_transform),
        target_matrix,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("source_path", "expected_index", "expected_suffix", "expected_phase"),
    [
        ("tests/testset/mcif_241130_no2186/0.1000_Fe4O5.mcif", "36.8.1.2.P2", "P2", "FM/FiM"),
        ("tests/testset/mcif_241130_no2186/0.188_CeMnAsO.mcif", "59.25.1.2.P1", "P1", "AFM"),
        ("tests/testset/mcif_241130_no2186/0.196_Co4Nb2O9.mcif", "165.158.1.2.P1", "P1", "AFM"),
    ],
)
def test_find_spin_group_matches_coplanar_configuration_suffixes_seen_in_batch_regression_scan(
    source_path,
    expected_index,
    expected_suffix,
    expected_phase,
):
    result = find_spin_group(source_path)
    details = result.identify_index_details

    assert result.conf == "Coplanar"
    assert result.index == expected_index
    assert result.magnetic_phase == expected_phase
    assert details["equivalent_map_index"] == 2
    assert details["configuration_suffix"] == expected_suffix


def test_find_spin_group_uses_single_equivalent_map_for_iso_zero_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.333_Mn2Mo3O8.mcif")
    details = result.identify_index_details

    assert result.index == "186.186.1.1.L"
    assert details["point_group_id"] == 0
    assert details["equivalent_map_index"] == 1


def test_ktb3f10_spin_point_group_sentinel_stays_c3v_with_origin_anchor():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.1120_KTb3F10.mcif"
    )
    primitive_cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    non_zero_indices = np.where(np.linalg.norm(primitive_cell.moments, axis=1) > 2e-5)[0]
    filtered_moments = np.array([primitive_cell.moments[i] for i in non_zero_indices])
    filtered_types = np.array([primitive_cell.atom_types[i] for i in non_zero_indices])
    unique_types, unique_moments = dedup_moments_with_tol(filtered_types, filtered_moments, 0.02)
    pg = PointGroupAnalyzer(Molecule(unique_types, unique_moments), tolerance=0.02, eigen_tolerance=2e-5)

    assert pg.sch_symbol == "C3v"
    assert len(pg.get_symmetry_operations()) == 6


def test_point_group_analyzer_relabels_symbol_from_audited_operation_closure():
    moments = np.array(
        [
            [-2.121, 1.832, -0.007],
            [-0.522, -2.753, 0.002],
            [2.651, 0.91, -0.003],
        ],
        dtype=float,
    )

    pg = PointGroupAnalyzer(
        Molecule([1, 1, 1], moments),
        tolerance=0.02,
        eigen_tolerance=2e-7,
    )

    assert pg.heuristic_sch_symbol == "Cs"
    assert pg.sch_symbol == "C3v"
    assert str(pg.get_pointgroup()) == "C3v"
    assert len(pg.get_symmetry_operations()) == 6


def test_ssg_generation_completes_closure_only_after_magnetic_revalidation():
    identity = np.eye(3)
    zero = np.zeros(3)
    mirror_x = np.diag([1.0, -1.0, 1.0])
    angle = np.deg2rad(60.0)
    mirror_60 = np.array(
        [
            [np.cos(2 * angle), np.sin(2 * angle), 0.0],
            [np.sin(2 * angle), -np.cos(2 * angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mag_atoms = [AtomicSite([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1)]
    raw_ops = [
        SpinSpaceGroupOperation(identity, identity, zero),
        SpinSpaceGroupOperation(mirror_x, identity, zero),
        SpinSpaceGroupOperation(mirror_60, identity, zero),
    ]

    closed_ops = _complete_ssg_ops_by_closure(raw_ops, mag_atoms, group_tol=0.02)

    assert len(closed_ops) == 6
    assert _resolve_point_group_info(
        [op[0] for op in closed_ops],
        tol=0.02,
        label="closed spin operations",
    )[-1] == "C3v"


@pytest.mark.parametrize(
    ("path", "expected_symbol"),
    [
        ("tests/testset/mcif_241130_no2186/1.412_Au72Al14Tb14.mcif", "Th"),
        ("tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif", "D3d"),
        ("tests/testset/mcif_241130_no2186/1.798_Tb2O3.mcif", "Th"),
        ("examples/CoNb3S6_tripleQ.mcif", "C3v"),
    ],
)
def test_get_pg_recovers_expected_symbols_for_stable_magnetic_point_sets(path, expected_symbol):
    primitive_cell = _primitive_magnetic_cell_from_cif(path)

    pg_symbol, _pg_operations = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        2e-5,
    )

    assert pg_symbol == expected_symbol


@pytest.mark.parametrize(
    "path",
    [
        "tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif",
        "tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif",
        "tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif",
        "tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif",
        "tests/testset/mcif_241130_no2186/1.412_Au72Al14Tb14.mcif",
        "tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif",
        "tests/testset/mcif_241130_no2186/1.798_Tb2O3.mcif",
        "examples/CoNb3S6_tripleQ.mcif",
    ],
)
def test_get_pg_candidate_is_configuration_compatible_for_magnetic_point_sets(path):
    primitive_cell = _primitive_magnetic_cell_from_cif(path)
    moments = np.asarray(primitive_cell.moments, dtype=float)
    non_zero_moments = moments[
        np.linalg.norm(moments, axis=1) > cell_module.MAGNETIC_PRESENCE_TOL
    ]
    configuration_details = _configuration_details(
        non_zero_moments,
        primitive_cell.tol.moment,
    )

    pg_symbol, pg_operations = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        2e-5,
    )

    assert (
        _configuration_compatibility(
            pg_symbol,
            configuration_details["configuration"],
            pg_operations=pg_operations,
            configuration_details=configuration_details,
            tol=primitive_cell.tol.moment,
        )
        > 0
    )


@pytest.mark.parametrize(
    ("path", "expected_conf", "expected_msg_num", "expected_msg_type"),
    [
        ("tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif", "Coplanar", 199, 3),
        ("tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif", "Coplanar", 82, 1),
        ("tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif", "Coplanar", 82, 1),
        ("tests/testset/mcif_241130_no2186/1.713_CsCr0.98Al0.02F4.mcif", "Coplanar", 332, 4),
        ("tests/testset/mcif_241130_no2186/1.748_TbAuIn.mcif", "Coplanar", 1366, 4),
        ("tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif", "Noncoplanar", 1233, 4),
    ],
)
def test_find_spin_group_recovers_pg_boundary_residual_msg_numbers(
    path,
    expected_conf,
    expected_msg_num,
    expected_msg_type,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.index is not None
    assert result.conf == expected_conf
    assert result.msg_num == expected_msg_num
    assert result.msg_type == expected_msg_type
    assert acc_primitive_ossg.msg_int_num == expected_msg_num
    assert acc_primitive_ossg.msg_type == expected_msg_type


def test_find_spin_group_keeps_conbnb3s6_tripleq_pg_boundary_recoverable():
    primitive_cell = _primitive_magnetic_cell_from_cif("examples/CoNb3S6_tripleQ.mcif")
    default_pg_symbol, _ = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        2e-5,
    )
    loose_pg_symbol, _ = get_pg(
        primitive_cell.moments,
        primitive_cell.atom_types,
        primitive_cell.tol.moment,
        1e-4,
    )

    default_result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    loose_result = find_spin_group("examples/CoNb3S6_tripleQ.mcif", meigtol=1e-4)

    assert default_pg_symbol == "C3v"
    assert loose_pg_symbol == "Td"
    assert default_result.msg_num == loose_result.msg_num == 1257
    assert default_result.msg_type == loose_result.msg_type == 3


def test_little_groups_symbols_recover_for_conbnb3s6_tripleq():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    symbols = ssg.little_groups_symbols

    assert isinstance(symbols, list)
    assert len(symbols) == len(ssg.kpoints_symbol_primitive)
    assert all(symbol != "?" for symbol in symbols)


def test_acc_aligned_runtime_index_exposes_ssg_conventional_kpoints():
    label = "12.6.4.11.P"

    assert get_pair_id_for_ssg_label(label) == "A010_P26"
    assert get_ssg_conventional_kpoint_symbols_for_label(label)[:5] == (
        "A:(1,0,0)",
        "B:(1/2,0,1/2)",
        "C:(1/2,1/2,-1/2)",
        "D:(1/2,1/2,1/2)",
        "E:(1,1/2,0)",
    )


def test_acc_aligned_runtime_index_exposes_spin_texture_config_records():
    label = "115.3.2.17.P"

    assert get_pair_id_for_ssg_label(label) == "A123_P02"
    assert get_spin_texture_config_id_for_ssg_label(label) == "W0043"
    assert get_spin_texture_config_for_ssg_label(label) == {
        "basis": ["C1*((-ky^2*kz + kx^2*kz)*sigma_z) + o(k^3)"],
        "basis_latex": [
            r"C_{1}\left(\left(-k_{y}^{2}k_{z} + k_{x}^{2}k_{z}\right)\,\sigma_{z}\right) + o(k^{3})"
        ],
        "basis_vectors": ["C1*((-ky^2*kz + kx^2*kz)*sigma_z)"],
        "basis_vectors_latex": [
            r"C_{1}\left(\left(-k_{y}^{2}k_{z} + k_{x}^{2}k_{z}\right)\,\sigma_{z}\right)"
        ],
        "momentum_space_spin_configuration": "collinear",
        "nullity": 1,
        "order": 3,
        "spin_rank": 1,
        "spin_texture_type": "f-wave",
    }


def test_kpoint_symbols_use_runtime_index_for_primitive_and_ssg_convention():
    identity = np.eye(3)
    ssg = SpinSpaceGroup([[identity, identity, np.zeros(3)]])
    ssg._input_index = "12.6.4.11.P"
    ssg.__dict__["acc_num"] = 10

    assert ssg.kpoints_symbol_primitive[:3] == (
        "A:(1/2,0,1/2)",
        "B:(0,0,1/2)",
        "C:(1/2,1/2,0)",
    )
    assert ssg.kpoints_symbol_conventional[:5] == (
        "A:(1,0,0)",
        "B:(1/2,0,1/2)",
        "C:(1/2,1/2,-1/2)",
        "D:(1/2,1/2,1/2)",
        "E:(1,1/2,0)",
    )
    assert ssg.kpoints_primitive[0] == (0.5, 0, 0.5)


def test_kpoint_symbols_fall_back_to_acc_convention_without_runtime_index():
    identity = np.eye(3)
    ssg = SpinSpaceGroup([[identity, identity, np.zeros(3)]])
    ssg._input_index = "not.in.runtime.index"
    ssg.__dict__["acc_num"] = 10

    assert ssg.kpoints_symbol_conventional[:3] == (
        "A:(1/2,0,1/2)",
        "B:(0,0,1/2)",
        "C:(1/2,1/2,0)",
    )
    assert ssg.kpoints_conventional[0] == (0.5, 0, 0.5)


def test_little_groups_symbols_use_minus3_not_minus6_for_vcl2_trigonal_cogroups():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.237_VCl2.mcif")
    ssg = SpinSpaceGroup(result.acc_primitive_ssg_ops)

    actual = {
        label: symbol
        for label, symbol in zip(ssg.kpoints_symbol_primitive, ssg.little_groups_symbols)
        if label in {"A:(0,0,1/2)", "Γ:(0,0,0)", "H:(1/3,1/3,1/2)", "K:(1/3,1/3,0)"}
    }

    assert actual["A:(0,0,1/2)"] == "^{m}-3^{m}m^{6/m}1"
    assert actual["Γ:(0,0,0)"] == "^{m}-3^{m}m^{6/m}1"
    assert "-6" not in actual["H:(1/3,1/3,1/2)"]
    assert "-6" not in actual["K:(1/3,1/3,0)"]
    assert "-3" in actual["H:(1/3,1/3,1/2)"]
    assert "-3" in actual["K:(1/3,1/3,0)"]


@pytest.mark.parametrize(
    "matrix",
    [
        np.array(
            [
                [3.833458638e-05, -1.000019165e00, -7.295004068e-05],
                [1.000019167e00, -9.999808303e-01, -5.285192675e-05],
                [1.532712294e-05, -5.886194163e-05, 9.999999980e-01],
            ],
            dtype=float,
        ),
        np.array(
            [
                [-0.49991117, 0.86607668, 0.0],
                [-0.86607668, -0.49991117, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [-5.04334917e-01, 1.66181668e-01, -2.20093585e-03],
                [-4.48690519e00, -5.04325025e-01, 1.98492772e-02],
                [2.22780031e-03, 7.49713550e-04, 9.99990108e-01],
            ],
            dtype=float,
        ),
    ],
)
def test_describe_point_operation_rejects_unverified_noisy_finite_order_guesses(matrix):
    with pytest.raises(ValueError, match="Cannot determine matrix order"):
        describe_point_operation(matrix, tol=1e-4, max_order=120)


def test_describe_spin_space_operation_can_mark_unresolved_without_guessing():
    matrix = np.array(
        [
            [-5.04334917e-01, 1.66181668e-01, -2.20093585e-03],
            [-4.48690519e00, -5.04325025e-01, 1.98492772e-02],
            [2.22780031e-03, 7.49713550e-04, 9.99990108e-01],
        ],
        dtype=float,
    )

    info = describe_spin_space_operation(
        matrix,
        np.eye(3),
        np.zeros(3),
        tol=1e-4,
        max_order=120,
        allow_unresolved=True,
    )

    assert info["spin"]["symbol"] == "?"
    assert info["spin"]["unresolved"] is True
    assert "Cannot determine matrix order" in info["spin"]["unresolved_reason"]
    assert info["symbol"] == "{ ? || 1 | 0,0,0 }"


def test_point_group_resolver_rejects_nonclosed_operation_set():
    rotation_120 = np.array(
        [
            [-0.5, -np.sqrt(3) / 2, 0.0],
            [np.sqrt(3) / 2, -0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="no inverse|is not closed"):
        _resolve_point_group_info(
            [np.eye(3), rotation_120],
            tol=1e-6,
            label="nonclosed test point group",
        )


def test_point_group_resolver_accepts_closed_operation_set():
    rotation_120 = np.array(
        [
            [-0.5, -np.sqrt(3) / 2, 0.0],
            [np.sqrt(3) / 2, -0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_240 = rotation_120 @ rotation_120

    info = _resolve_point_group_info(
        [np.eye(3), rotation_120, rotation_240],
        tol=1e-6,
        label="closed test point group",
    )

    assert info[0] == "3"
    assert info[4] == "C3"


def test_mn3sn_seitz_symbols_do_not_emit_illegal_minus6_power2_tokens():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.199_Mn3Sn.mcif")

    joined = "\n".join(result.g0_standard_ssg_seitz + result.primitive_magnetic_cell_ssg_seitz)

    assert "-6^{2}_{001}" not in joined
    assert "-6^{5}_{001}" in joined


def test_mn3sn_cartesian_standard_spin_axes_prefer_symbolic_components_over_alpha_beta():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.199_Mn3Sn.mcif")

    g0_joined = "\n".join(result.g0_standard_ssg_seitz)
    l0_joined = "\n".join(result.l0_standard_ssg_seitz)

    assert "alpha,beta,0" not in g0_joined
    assert "alpha,beta,0" not in l0_joined
    assert "sqrt(3)/2" in g0_joined
    assert "sqrt(3)/2" in l0_joined


def test_describe_point_operation_requires_physical_tolerance_for_noisy_improper_fourfold():
    matrix = np.array(
        [
            [-0.00195873, -0.9999979, -0.00022571],
            [0.99999823, -0.00195903, 0.00015037],
            [0.00015076, 0.00022567, -0.99999996],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="Cannot determine matrix order"):
        describe_point_operation(matrix, tol=1e-4, max_order=120)

    info = describe_point_operation(matrix, tol=1e-2, max_order=120)

    assert info["hm_symbol"] == "-4"
    assert info["axis_direction"] == (0, 0, 1)
    assert info["symbol"] == "-4^{3}_{001}"


def test_audit_spatial_transform_effect_identity_preserves_real_ops_exactly():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)

    audit = audit_spatial_transform_effect(ssg, np.eye(3), np.zeros(3), tol=1e-6)

    assert audit["source_real_op_count"] == audit["transformed_real_op_count"] == 48
    assert audit["real_ops_exact_same"] is True
    assert audit["real_ops_same_mod_integer"] is True
    assert audit["real_ops_same_mod_pure_translations"] is True
    assert audit["paired_spin_changed_count"] == 0
    assert audit["unmatched_source_indices"] == []


def test_audit_spatial_transform_effect_flags_normalizer_like_real_space_invariance_for_conbnb3s6():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    transform_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    origin_shift = np.array([0.0, 0.5, 0.0], dtype=float)

    audit = audit_spatial_transform_effect(ssg, transform_matrix, origin_shift, tol=1e-6)

    assert audit["source_real_op_count"] == audit["transformed_real_op_count"] == 48
    assert audit["real_ops_exact_same"] is True
    assert audit["real_ops_same_mod_integer"] is True
    assert audit["real_ops_same_mod_pure_translations"] is True
    assert audit["paired_spin_changed_count"] > 0
    assert audit["unmatched_source_indices"] == []


def test_audit_spatial_transform_effect_short_circuits_when_volume_changes():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    transform_matrix = 2.0 * np.eye(3)

    audit = audit_spatial_transform_effect(ssg, transform_matrix, np.zeros(3), tol=1e-6, det_tol=1e-2)

    assert abs(audit["determinant"] - 8.0) < 1e-9
    assert audit["volume_preserving"] is False
    assert audit["can_be_affine_normalizer_equivalent"] is False
    assert audit["real_ops_exact_same"] is False
    assert audit["source_real_op_count"] is None
    assert audit["transformed_to_source"] == []


def test_canonicalize_input_to_standard_setting_returns_cartesianized_input_cell_on_identity_collapse():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "examples/0.800_MnTe.mcif"
    )
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    )
    input_cell_cartesian = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        input_cell.moments_cartesian,
        spin_setting="cartesian",
    )
    result = find_spin_group("examples/0.800_MnTe.mcif")
    target_ssg = SpinSpaceGroup(result.g0_standard_ssg_ops)

    collapsed_cell, collapsed_ssg, collapsed_transform, audit = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        input_cell,
        target_ssg,
        (np.eye(3), np.zeros(3)),
    )

    assert audit["real_ops_exact_same"] is True
    assert collapsed_transform[0].tolist() == np.eye(3).tolist()
    assert collapsed_transform[1].tolist() == np.zeros(3).tolist()
    assert collapsed_cell.spin_setting == "cartesian"
    assert np.allclose(collapsed_cell.moments, input_cell_cartesian.moments, atol=1e-8)
    assert len(collapsed_ssg.ops) == len(target_ssg.ops)


def _changed_basis_conb3s6_tripleq_input():
    lattice = [11.498, 11.498, 11.886, 90, 90, 120]
    _, _, elements, occupancies, _labels, _ = parse_cif_file("examples/CoNb3S6_tripleQ.mcif")
    raw = """
0.333333 0.166700 0.750000 1.000000 2.1772 1.08860 -0.666667
0.833300 0.166633 0.750000 1.000000 -1.08860 1.08860 -0.666667
0.833367 0.666667 0.750000 1.000000 -1.08860 -2.1772 -0.666667
0.166633 0.833300 0.250000 1.000000 -1.08860 1.08860 -0.666667
0.166700 0.333333 0.250000 1.000000 -1.08860 -2.1772 -0.666667
0.666667 0.833367 0.250000 1.000000 2.1772 1.08860 -0.666667
0.666667 0.333333 0.250000 1.000000 0.000000 0.000000 2.000000
0.333333 0.666667 0.750000 1.000000 0.000000 0.000000 2.000000
0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.500000 1.000000 0.000000 0.000000 0.000000
0.500000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000
0.000000 0.500000 0.000000 1.000000 0.000000 0.000000 0.000000
0.500000 0.500000 0.000000 1.000000 0.000000 0.000000 0.000000
0.500000 0.000000 0.500000 1.000000 0.000000 0.000000 0.000000
0.000000 0.500000 0.500000 1.000000 0.000000 0.000000 0.000000
0.500000 0.500000 0.500000 1.000000 0.000000 0.000000 0.000000
0.666667 0.333333 0.994400 1.000000 0.000000 0.000000 0.000000
0.333333 0.666667 0.005600 1.000000 0.000000 0.000000 0.000000
0.333333 0.166700 0.494400 1.000000 0.000000 0.000000 0.000000
0.833300 0.166633 0.494400 1.000000 0.000000 0.000000 0.000000
0.833367 0.666667 0.494400 1.000000 0.000000 0.000000 0.000000
0.166633 0.833300 0.505600 1.000000 0.000000 0.000000 0.000000
0.166700 0.333333 0.505600 1.000000 0.000000 0.000000 0.000000
0.666667 0.833367 0.505600 1.000000 0.000000 0.000000 0.000000
0.333333 0.166700 0.005600 1.000000 0.000000 0.000000 0.000000
0.833300 0.166633 0.005600 1.000000 0.000000 0.000000 0.000000
0.833367 0.666667 0.005600 1.000000 0.000000 0.000000 0.000000
0.166633 0.833300 0.994400 1.000000 0.000000 0.000000 0.000000
0.166700 0.333333 0.994400 1.000000 0.000000 0.000000 0.000000
0.666667 0.833367 0.994400 1.000000 0.000000 0.000000 0.000000
0.666667 0.333333 0.505600 1.000000 0.000000 0.000000 0.000000
0.333333 0.666667 0.494400 1.000000 0.000000 0.000000 0.000000
0.166100 0.000500 0.369400 1.000000 0.000000 0.000000 0.000000
0.999500 0.165600 0.369400 1.000000 0.000000 0.000000 0.000000
0.834400 0.833900 0.369400 1.000000 0.000000 0.000000 0.000000
0.165600 0.999500 0.630600 1.000000 0.000000 0.000000 0.000000
0.000500 0.166100 0.630600 1.000000 0.000000 0.000000 0.000000
0.833900 0.834400 0.630600 1.000000 0.000000 0.000000 0.000000
0.499600 0.165700 0.369400 1.000000 0.000000 0.000000 0.000000
0.834300 0.333900 0.369400 1.000000 0.000000 0.000000 0.000000
0.666100 0.500400 0.369400 1.000000 0.000000 0.000000 0.000000
0.333900 0.834300 0.630600 1.000000 0.000000 0.000000 0.000000
0.165700 0.499600 0.630600 1.000000 0.000000 0.000000 0.000000
0.500400 0.666100 0.630600 1.000000 0.000000 0.000000 0.000000
0.334400 0.833900 0.369400 1.000000 0.000000 0.000000 0.000000
0.166100 0.500500 0.369400 1.000000 0.000000 0.000000 0.000000
0.499500 0.665600 0.369400 1.000000 0.000000 0.000000 0.000000
0.500500 0.166100 0.630600 1.000000 0.000000 0.000000 0.000000
0.833900 0.334400 0.630600 1.000000 0.000000 0.000000 0.000000
0.665600 0.499500 0.630600 1.000000 0.000000 0.000000 0.000000
0.833900 0.999600 0.869400 1.000000 0.000000 0.000000 0.000000
0.000400 0.834300 0.869400 1.000000 0.000000 0.000000 0.000000
0.165700 0.166100 0.869400 1.000000 0.000000 0.000000 0.000000
0.834300 0.000400 0.130600 1.000000 0.000000 0.000000 0.000000
0.999600 0.833900 0.130600 1.000000 0.000000 0.000000 0.000000
0.166100 0.165700 0.130600 1.000000 0.000000 0.000000 0.000000
0.500500 0.834400 0.869400 1.000000 0.000000 0.000000 0.000000
0.165600 0.666100 0.869400 1.000000 0.000000 0.000000 0.000000
0.333900 0.499500 0.869400 1.000000 0.000000 0.000000 0.000000
0.666100 0.165600 0.130600 1.000000 0.000000 0.000000 0.000000
0.834400 0.500500 0.130600 1.000000 0.000000 0.000000 0.000000
0.499500 0.333900 0.130600 1.000000 0.000000 0.000000 0.000000
0.665700 0.166100 0.869400 1.000000 0.000000 0.000000 0.000000
0.833900 0.499600 0.869400 1.000000 0.000000 0.000000 0.000000
0.500400 0.334300 0.869400 1.000000 0.000000 0.000000 0.000000
0.499600 0.833900 0.130600 1.000000 0.000000 0.000000 0.000000
0.166100 0.665700 0.130600 1.000000 0.000000 0.000000 0.000000
0.334300 0.500400 0.130600 1.000000 0.000000 0.000000 0.000000
0.666100 0.000500 0.369400 1.000000 0.000000 0.000000 0.000000
0.999500 0.665600 0.369400 1.000000 0.000000 0.000000 0.000000
0.334400 0.333900 0.369400 1.000000 0.000000 0.000000 0.000000
0.665600 0.999500 0.630600 1.000000 0.000000 0.000000 0.000000
0.000500 0.666100 0.630600 1.000000 0.000000 0.000000 0.000000
0.333900 0.334400 0.630600 1.000000 0.000000 0.000000 0.000000
0.333900 0.999600 0.869400 1.000000 0.000000 0.000000 0.000000
0.000400 0.334300 0.869400 1.000000 0.000000 0.000000 0.000000
0.665700 0.666100 0.869400 1.000000 0.000000 0.000000 0.000000
0.334300 0.000400 0.130600 1.000000 0.000000 0.000000 0.000000
0.999600 0.333900 0.130600 1.000000 0.000000 0.000000 0.000000
0.666100 0.665700 0.130600 1.000000 0.000000 0.000000 0.000000
"""
    rows = [tuple(float(x) for x in line.split()) for line in raw.strip().splitlines()]
    positions = [r[:3] for r in rows]
    occupancies_in = [r[3] for r in rows]
    moments = [r[4:] for r in rows]
    assert all(abs(a - b) < 1e-8 for a, b in zip(occupancies_in, occupancies))
    return lattice, positions, elements, occupancies, moments


def test_changed_basis_conb3s6_tripleq_preserves_msg_after_g0_collapse():
    lattice, positions, elements, occupancies, moments = _changed_basis_conb3s6_tripleq_input()

    result = find_spin_group_from_data(
        "changed_basis_Conb3s6",
        lattice,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert result.index == "182.4.4.2"
    assert result.conf == "Noncoplanar"
    assert result.acc == "6mmP"
    assert result.msg_num == 1257
    assert result.msg_type == 3
    assert result.msg_symbol == "P32'1"
    assert result.msg_acc == "3m1P"
    canonical_symbol = SpinSpaceGroup(result.convention_ssg_ops).international_symbol
    primitive_terms = canonical_symbol["translation_terms_linear"][0].strip("()").split(",")
    assert primitive_terms[0] != "1"
    assert primitive_terms[1] != "1"
    assert primitive_terms[2] == "1"


def test_identify_point_group_recovers_td_for_conbnb3s6_gamma_little_group_spin_part():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)
    gamma_index = ssg.kpoints_symbol_primitive.index("Γ:(0,0,0)")
    little_group = ssg.little_groups[gamma_index]
    spin_part = deduplicate_matrix_pairs([np.array(op[0]) for op in little_group])

    group_symbol, *_ = identify_point_group(spin_part)

    assert group_symbol == "-43m"


def test_classify_moment_configuration_uses_mtol_residual_contract():
    moments = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.019],
        ],
        dtype=float,
    )
    assert _classify_moment_configuration(moments, 0.02) == "Coplanar"
    assert _classify_moment_configuration(moments, 0.006) == "Noncoplanar"


def test_classify_moment_configuration_reports_o3_degeneracy_when_mtol_erases_spin_scale():
    moments = np.array(
        [
            [0.6, 0.0, 0.0],
            [0.0, -0.7, 0.0],
        ],
        dtype=float,
    )

    details = _configuration_details(moments, mtol=1.0)

    assert details["configuration"] == "Nonmagnetic"
    assert details["constraint_rank"] == 0
    assert details["spin_point_group_semantics"] == "O3"
    assert _classify_moment_configuration(moments, 1.0) == "Nonmagnetic"


def test_get_pg_coplanar_candidate_contains_required_spin_mirror():
    moments = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    symbol, operations = get_pg(moments, np.array([1, 1, 1]), mtol=0.02, meigtol=2e-5)

    assert symbol not in {"C1", "C*v", "D*h", "Kh"}
    assert any(np.allclose(op, np.diag([1.0, 1.0, -1.0]), atol=0.02, rtol=0) for op in operations)


@pytest.mark.parametrize(
    ("path", "low_mtol", "mid_mtol", "high_mtol", "expected_low", "expected_mid", "expected_high"),
    [
        (
            "tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif",
            0.019,
            0.02,
            0.041,
            "Coplanar",
            "Coplanar",
            "Collinear",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif",
            0.019,
            0.02,
            0.041,
            "Coplanar",
            "Coplanar",
            "Collinear",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif",
            0.049,
            0.05,
            0.101,
            "Coplanar",
            "Coplanar",
            "Collinear",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.394_Cu2CdB2O6.mcif",
            0.012,
            0.014,
            0.39,
            "Noncoplanar",
            "Coplanar",
            "Collinear",
        ),
    ],
)
def test_real_cases_exhibit_mtol_driven_configuration_boundaries(
    path,
    low_mtol,
    mid_mtol,
    high_mtol,
    expected_low,
    expected_mid,
    expected_high,
):
    primitive_cell = _primitive_magnetic_cell_from_cif(path)
    moments = np.asarray(primitive_cell.moments, dtype=float)
    non_zero = moments[np.linalg.norm(moments, axis=1) > primitive_cell.tol.m_eig]

    assert _classify_moment_configuration(non_zero, low_mtol) == expected_low
    assert _classify_moment_configuration(non_zero, mid_mtol) == expected_mid
    assert _classify_moment_configuration(non_zero, high_mtol) == expected_high


def test_identify_spin_space_group_reports_nonmagnetic_error_when_input_is_effectively_nonmagnetic():
    cell = CrystalCell(
        lattice=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
        positions=[[0.0, 0.0, 0.0]],
        occupancies=[1.0],
        elements=["Fe"],
        moments=[[1e-6, 0.0, 0.0]],
        spin_setting="in_lattice",
    )

    with pytest.raises(MagneticToleranceDegeneracyError, match=NONMAGNETIC_MTOL_ERROR):
        identify_spin_space_group_result(cell, find_primitive=False)


def test_crystal_cell_initializes_net_moment_without_magnetic_moments():
    cell = CrystalCell(
        lattice=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
        positions=[[0.0, 0.0, 0.0]],
        occupancies=[1.0],
        elements=["Fe"],
        moments=None,
    )

    assert cell.net_moment is None


def test_identify_spin_space_group_reports_mtol_induced_o3_degeneracy():
    cell = CrystalCell(
        lattice=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
        positions=[[0.0, 0.0, 0.0]],
        occupancies=[1.0],
        elements=["Fe"],
        moments=[[0.5, 0.0, 0.0]],
        spin_setting="in_lattice",
    )

    with pytest.raises(MagneticToleranceDegeneracyError, match="nonmagnetic O3"):
        identify_spin_space_group_result(
            cell,
            find_primitive=False,
            tol=Tolerances(space=0.02, moment=1.0, m_eig=2e-5, occupancy=0.002, m_matrix_tol=0.01),
        )


def test_find_spin_group_extreme_mtol_reports_spin_constraint_degeneracy():
    with pytest.raises(MagneticToleranceDegeneracyError, match="nonmagnetic O3"):
        find_spin_group("tests/testset/mcif_241130_no2186/1.850_Tb6FeSi2S14.mcif", mtol=5.0)


def test_find_spin_group_extreme_mtol_rejects_real_case_as_semantically_degenerate():
    with pytest.raises(MagneticToleranceDegeneracyError, match="nonmagnetic O3"):
        find_spin_group("tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif", mtol=5.0)


@pytest.mark.parametrize(
    ("path", "expected_index", "expected_msg_num", "expected_msg_type"),
    [
        ("tests/testset/mcif_241130_no2186/1.138_MgV2O4.mcif", "22.1.2.7", 135, 4),
        ("tests/testset/mcif_241130_no2186/1.207_U2Rh2Sn.mcif", "127.2.2.8", 1152, 4),
        ("tests/testset/mcif_241130_no2186/1.501_Ba2CoO2Cu2S2.mcif", "69.65.2.1.L", 97, 4),
    ],
)
def test_find_spin_group_recovers_post_batch_three_residual_regressions(
    path,
    expected_index,
    expected_msg_num,
    expected_msg_type,
):
    result = find_spin_group(path)

    assert result.index == expected_index
    assert result.msg_num == expected_msg_num
    assert result.msg_type == expected_msg_type


def test_find_spin_group_keeps_ktb3f10_out_of_identity_collapse_sentinel():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1120_KTb3F10.mcif")

    assert result.index == "225.69.1.2"
    assert result.identify_index_details["t_index"] == 6
    assert result.identify_index_details["k_index"] == 1
    assert result.identify_index_details["equivalent_map_index"] == 2
    assert result.identify_index_details["point_group_id"] == 32


@pytest.mark.parametrize(
    ("path", "expected_index"),
    [
        ("tests/testset/mcif_241130_no2186/0.1010_C10H6MnN4O4.mcif", "14.1.1.1.P3"),
        ("tests/testset/mcif_241130_no2186/2.96_GdMn2Si2.mcif", "139.115.2.1.P3"),
        ("tests/testset/mcif_241130_no2186/1.647_Na2.4Ni2TeO6.mcif", "63.13.2.21.P3"),
    ],
)
def test_find_spin_group_basic_matches_manual_checked_identify_222_index_changes(
    path,
    expected_index,
):
    payload = find_spin_group_basic(path)

    assert payload["index"] == expected_index


def test_find_spin_group_basic_does_not_fallback_when_identify_database_entry_is_missing():
    with pytest.warns(RuntimeWarning, match="Identify-index database entry unavailable"):
        payload = find_spin_group_basic("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert payload["index"].startswith("not in identify-index database:")
    assert payload["identify_index_details"] is None
    assert payload["phase"]
    assert payload["spin_texture_config_database"] is None
    assert payload["acc_primitive_resolution_audit"]["status"] == "identify_index_unavailable"


def test_find_spin_group_basic_uses_global_acc_primitive_selection():
    payload = find_spin_group_basic("tests/testset/mcif_241130_no2186/1.115_Dy3Ru4Al12.mcif")

    assert payload["index"] == "12.2.2.3"
    assert "acc_primitive_standard_setting" not in payload
    audit = payload["acc_primitive_resolution_audit"]["G0std_transform_selection"]
    assert audit["selected_strategy"] == "nofrac_lattice_shear:r2+=(-2)r0"
    assert all(
        "legacy" not in candidate["strategy"]
        for candidate in audit["rejected_candidates"]
    )


def test_find_spin_group_basic_uses_monoclinic_ac_column_reduction_for_index2_convention():
    payload = find_spin_group_basic("tests/testset/mcif_241130_no2186/2.116_Na3Co2SbO6.mcif")

    assert payload["index"] == "10.2.2.21.P2"
    assert "acc_primitive_standard_setting" not in payload
    audit = payload["acc_primitive_resolution_audit"]["G0std_transform_selection"]
    assert audit["selected_strategy"].startswith("monoclinic_ac_column_reduce:")
    assert "det_factor=2" in audit["selected_strategy"]
    selected_matrix = np.asarray(audit["selected_matrix"], dtype=float)
    convention_to_acc = find_spin_group_module._acc_aligned_convention_to_primitive_transform(
        payload["index"]
    )[0]
    assert np.isclose(abs(np.linalg.det(convention_to_acc @ selected_matrix)), 1.0)


@pytest.mark.parametrize(
    ("index", "primitive_to_database_matrix"),
    [
        (
            "2.1.3.1.P",
            [
                [-7.0 / 3.0, -11.0 / 3.0, 0.0],
                [-2.0, -3.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        ),
        (
            "2.1.4.4.P",
            [
                [8.0, 0.5, 0.0],
                [7.0, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        (
            "2.1.2.1.P2",
            [
                [-2.5, -1.5, 0.0],
                [-2.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        ),
    ],
)
def test_triclinic_column_reduction_handles_spin_translation_supercell(
    index,
    primitive_to_database_matrix,
):
    class _FakeTriclinicSSG:
        G0_num = 2

    candidates = []
    seen = set()
    find_spin_group_module._append_triclinic_column_reduction_candidates(
        candidates,
        seen,
        find_spin_group_module.G0_STANDARD_SETTING,
        _FakeTriclinicSSG(),
        (np.asarray(primitive_to_database_matrix, dtype=float), np.zeros(3)),
        index,
        {
            "space_group_transformation": [
                np.eye(3).tolist(),
                np.zeros(3).tolist(),
            ],
        },
        tol=0.01,
    )

    triclinic_candidates = [
        (name, transform)
        for name, transform in candidates
        if name.startswith("triclinic_column_reduce:")
    ]
    assert triclinic_candidates

    convention_to_acc = find_spin_group_module._acc_aligned_convention_to_primitive_transform(
        index
    )[0]
    assert np.isclose(
        abs(float(np.linalg.det(convention_to_acc @ triclinic_candidates[0][1][0]))),
        1.0,
        atol=1e-8,
    )


def test_find_spin_group_basic_uses_it_ik_to_select_g0std_for_t_type_linear_index():
    payload = find_spin_group_basic("tests/testset/mcif_241130_no2186/0.1017_CePdAl3.mcif")

    assert payload["index"] == "63.38.1.1.L"
    assert payload["it"] == 2
    assert payload["ik"] == 1
    assert "acc_primitive_standard_setting" not in payload
    audit = payload["acc_primitive_resolution_audit"]["G0std_transform_selection"]
    assert audit["selected_strategy"] == "current_integerized"
    assert audit["preferred_standard_setting"] == "G0std"
    assert audit["standard_setting_rule"] == "t_index/k_index"


def test_find_spin_group_basic_uses_it_ik_to_select_l0std_for_k_type_p_index():
    payload = find_spin_group_basic("tests/testset/mcif_241130_no2186/1.455_Mn6Ni16Si7.mcif")

    assert payload["index"] == "69.65.2.2.P1"
    assert payload["it"] == 1
    assert payload["ik"] == 2
    assert "acc_primitive_standard_setting" not in payload
    audit = payload["acc_primitive_resolution_audit"]["L0std_transform_selection"]
    assert audit["selected_strategy"] == "current_integerized"
    assert audit["preferred_standard_setting"] == "L0std"
    assert audit["standard_setting_rule"] == "t_index/k_index"


def test_find_spin_group_basic_reraises_non_database_identify_errors(monkeypatch):
    def _raise_point_group_map_error(*_args, **_kwargs):
        raise ValueError(
            "Cannot identify point-group map number for point_group=1, generator_numbers=[1]."
        )

    monkeypatch.setattr(
        find_spin_group_module,
        "_identify_ssg_index_details",
        _raise_point_group_map_error,
    )

    with pytest.raises(ValueError, match="Cannot identify point-group map number"):
        find_spin_group_basic("examples/0.800_MnTe.mcif")


def test_find_spin_group_acc_primitive_does_not_fallback_when_identify_database_entry_is_missing():
    with pytest.warns(RuntimeWarning, match="Identify-index database entry unavailable"):
        payload = find_spin_group_acc_primitive("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert payload["index"].startswith("not in identify-index database:")
    assert payload["identify_index_details"] is None
    assert payload["acc_primitive_resolution_audit"]["status"] == "identify_index_unavailable"


def test_find_spin_group_does_not_fallback_when_identify_database_entry_is_missing():
    source_name = "tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif"
    with pytest.warns(RuntimeWarning, match="Identify-index database entry unavailable"):
        result = find_spin_group(source_name)

    assert result.index.startswith("not in identify-index database:")
    assert result.identify_index_details is None
    assert result.magnetic_phase
    assert result.acc_primitive_resolution_audit["status"] == "identify_index_unavailable"


def test_g_type_output_ossg_uses_shortest_nonzero_axis_translations():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")

    ssg = SpinSpaceGroup(result.convention_ssg_ops)
    symbol = ssg.international_symbol

    primitive_terms = symbol["translation_terms_linear"][0].strip("()").split(",")
    assert symbol["linear"].endswith(f" : {symbol['translation_terms_linear'][0]}")
    assert primitive_terms[0] != "1"
    assert primitive_terms[1] != "1"
    assert primitive_terms[2] == "1"
    details = symbol["translation_details"][:3]
    assert [detail["label"] for detail in details] == ["t_a", "t_b", "t_c"]
    assert details[0]["spin_symbol"] != "1"
    assert details[1]["spin_symbol"] != "1"
    assert details[2]["spin_symbol"] == "1"
    assert np.allclose(details[0]["vector"], (1.0, 0.0, 0.0))
    assert np.allclose(details[1]["vector"], (0.0, 1.0, 0.0))
    assert np.allclose(details[2]["vector"], (0.0, 0.0, 0.0))


def test_find_spin_group_preserves_historical_identify_index_for_srmnvo4oh():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.165_SrMn(VO4)(OH).mcif")

    assert result.index == "19.4.1.2.P2"
    assert result.identify_index_details["equivalent_map_index"] == 2
    assert result.identify_index_details["configuration_suffix"] == "P2"


def test_find_spin_group_preserves_historical_identify_index_for_ndco2():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.403_NdCo2.mcif")

    assert result.index == "227.227.1.1.L"
    assert result.spin_part_point_group == "∞m"
    assert result.magnetic_phase == "FM/FiM"
    assert result.magnetic_phase_base == "FM/FiM"
    assert result.magnetic_phase_modifier == ""
    assert result.is_spin_orbit_magnet == ""
    assert result.magnetic_phase_details["classification_rule"] == "fm_like_spin_point_group"
    assert result.magnetic_phase_details["fm_like_by_spin_point_group"] is True
    assert result.identify_index_details["equivalent_map_index"] == 1
    assert result.identify_index_details["configuration_suffix"] == "L"


def test_find_spin_group_exposes_compensated_fim_classification_details():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.103_Mn2GeO4.mcif")

    assert result.magnetic_phase == "Compensated FiM"
    assert result.magnetic_phase_base == "Compensated FiM"
    assert result.magnetic_phase_modifier == ""
    assert result.is_spin_orbit_magnet == ""
    assert result.magnetic_phase_details["classification_rule"] == "fm_like_spin_point_group"
    assert result.magnetic_phase_details["zero_net_moment"] is True
    assert result.magnetic_phase_details["zero_net_moment_tol"] == pytest.approx(0.02)
    assert result.tolerances["mtol"] == pytest.approx(0.02)


def test_compensated_fim_zero_net_moment_uses_magnetic_tolerance():
    default_payload = classify_magnetic_phase(
        conf="Collinear",
        full_spin_part_point_group_hm=None,
        full_spin_part_point_group_s="∞m",
        net_moment=1e-3,
        mpg_identifier=None,
        is_ss_gp="spin splitting",
    )
    strict_payload = classify_magnetic_phase(
        conf="Collinear",
        full_spin_part_point_group_hm=None,
        full_spin_part_point_group_s="∞m",
        net_moment=1e-3,
        net_moment_tol=1e-4,
        mpg_identifier=None,
        is_ss_gp="spin splitting",
    )
    relaxed_payload = classify_magnetic_phase(
        conf="Collinear",
        full_spin_part_point_group_hm=None,
        full_spin_part_point_group_s="∞m",
        net_moment=1e-3,
        net_moment_tol=0.02,
        mpg_identifier=None,
        is_ss_gp="spin splitting",
    )

    assert default_payload["base_phase"] == "Compensated FiM"
    assert default_payload["details"]["zero_net_moment"] is True
    assert default_payload["details"]["zero_net_moment_tol"] == pytest.approx(DEFAULT_TOL.moment)
    assert strict_payload["base_phase"] == "FM/FiM"
    assert strict_payload["details"]["zero_net_moment"] is False
    assert strict_payload["details"]["zero_net_moment_tol"] == pytest.approx(1e-4)
    assert relaxed_payload["base_phase"] == "Compensated FiM"
    assert relaxed_payload["details"]["zero_net_moment"] is True
    assert relaxed_payload["details"]["zero_net_moment_tol"] == pytest.approx(0.02)


def test_get_magnetic_phase_accepts_net_moment_tolerance():
    strict_phase = get_magnetic_phase(None, "∞m", 1e-3, None, net_moment_tol=1e-4)
    relaxed_phase = get_magnetic_phase(None, "∞m", 1e-3, None, net_moment_tol=0.02)

    assert strict_phase == "FM/FiM"
    assert relaxed_phase == "Compensated FiM"


def test_get_magnetic_phase_returns_base_phase_when_full_context_is_supplied():
    phase = get_magnetic_phase(
        "∞m",
        "C∞v",
        1e-3,
        None,
        conf="Collinear",
        is_ss_gp="spin splitting",
        net_moment_tol=0.02,
    )

    assert phase == "Compensated FiM"


def test_find_spin_group_basic_reports_classification_tolerances():
    result = find_spin_group_basic(
        "tests/testset/mcif_241130_no2186/0.103_Mn2GeO4.mcif",
        mtol=0.05,
    )

    assert result["magnetic_phase"] == "Compensated FiM"
    assert result["magnetic_phase_base"] == "Compensated FiM"
    assert result["magnetic_phase_details"]["zero_net_moment"] is True
    assert result["magnetic_phase_details"]["zero_net_moment_tol"] == pytest.approx(0.05)
    assert result["net_moment"] == pytest.approx(
        result["magnetic_phase_details"]["net_moment"]
    )
    assert result["zero_net_moment_tol"] == pytest.approx(0.05)
    assert result["msg_type"] in {1, 2, 3, 4}
    assert result["quasi_2d"] is None
    assert result["tolerances"]["mtol"] == pytest.approx(0.05)


def test_find_spin_group_exposes_spin_orbit_magnet_classification_details():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.101_Mn2GeO4.mcif")

    assert result.magnetic_phase == "AFM\n(SOM)"
    assert result.magnetic_phase_base == "AFM"
    assert result.magnetic_phase_modifier == ""
    assert result.is_spin_orbit_magnet == "(SOM)"
    assert result.magnetic_phase_details["classification_rule"] == "afm_with_spin_orbit_magnet"
    assert result.magnetic_phase_details["som_by_mpg"] is True


def test_find_spin_group_exposes_independent_alter_and_spin_orbit_magnet_tags():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1008_Sr2ErRuO6.mcif")

    assert result.magnetic_phase == "AFM(Altermagnet)\n(SOM)"
    assert result.magnetic_phase_base == "AFM"
    assert result.magnetic_phase_modifier == "(Altermagnet)"
    assert result.is_alter == "(Altermagnet)"
    assert result.is_spin_orbit_magnet == "(SOM)"
    assert result.magnetic_phase_details["is_altermagnet"] is True
    assert result.magnetic_phase_details["is_spin_orbit_magnet"] is True


def test_identify_name_generator_matching_uses_pure_translation_cosets():
    target = np.array([0.0, 0.0, 0.0], dtype=float)
    candidate = np.array([0.0006, 0.4994, 0.5], dtype=float)
    pure_translations = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.5, 0.0, 0.5], dtype=float),
        np.array([0.0, 0.5, 0.5], dtype=float),
        np.array([0.5, 0.5, 0.0], dtype=float),
    ]

    assert not _translations_equivalent_mod_pure_translations(
        target,
        candidate,
        [np.zeros(3)],
        tol=1e-3,
    )
    assert _translations_equivalent_mod_pure_translations(
        target,
        candidate,
        pure_translations,
        tol=1e-3,
    )


def test_find_spin_group_recovers_msg_little_group_symbols_after_translation_cleanup():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.118_Ba5Co5ClO13.mcif")

    assert result.msg_little_group_symbols
    assert "Unknown" not in set(result.msg_little_group_symbols)


def test_msg_little_group_uses_reciprocal_rotation_for_centered_primitive_basis():
    result = find_spin_group(
        "tests/testset/mcif_241130_no2186/0.236_CaFe4Al8.mcif",
        mtol=0.002,
    )

    assert result.msg_bns_number == "139.535"
    assert result.msg_little_group_symbols[:5] == [
        "4'/mmm'",
        "4'/mmm'",
        "4'22'",
        "m'm'm",
        "2/m",
    ]


def test_find_spin_group_keeps_maximal_ssg_when_ossg_msg_little_group_is_unindexed():
    result = find_spin_group(
        "tests/testset/mcif_241130_no2186/1.85_alpha-Mn.mcif",
        mtol=0.2,
        matrix_tol=0.02,
        space_tol=0.02,
        meigtol=0.00002,
    )

    assert result.conf == "Noncoplanar"
    assert result.index is not None
    assert len(result.acc_primitive_ssg_ops) == 8
    assert len(result.primitive_msg_ops) == 8
    assert result.msg_num is None
    assert result.msg_symbol is None
    assert None in result.msg_little_group_symbols
    assert len(result.msg_little_group_symbols) == len(result.spin_polarizations)
    assert len(result.msg_spin_polarizations) == len(result.spin_polarizations)


def test_tensor_ops_do_not_depend_on_international_symbol_generation(monkeypatch):
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.1114_CeAlSi.mcif"
    )
    cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    ssg = identify_spin_space_group(cell, find_primitive=False)
    acc_cell = cell.transform(ssg.acc_primitive_trans, ssg.acc_primitive_origin_shift)
    acc_ssg = ssg.transform(ssg.acc_primitive_trans, ssg.acc_primitive_origin_shift)

    def _boom(self, tol=1e-4):
        raise AssertionError("symbol generation should be decoupled from tensor ops")

    monkeypatch.setattr(SpinSpaceGroup, "get_international_symbol", _boom)

    ops_wo_soc = _tensor_ops_wo_soc(acc_ssg, acc_cell)

    assert ops_wo_soc
    assert len(ops_wo_soc) == len(acc_ssg.gspg_ops_raw)


def test_get_magnetic_space_group_from_operations_handles_noisy_fractional_translations():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.403_NdCo2.mcif"
    )
    cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    ssg = identify_spin_space_group(cell, find_primitive=False)
    acc_ssg = ssg.transform(ssg.acc_primitive_trans, ssg.acc_primitive_origin_shift)
    primitive_msg_ops = _primitive_msg_ops_from_ssg(acc_ssg.ops, tol=0.01)
    little_group = _get_magnetic_little_group(
        acc_ssg.kpoints_primitive[0],
        primitive_msg_ops,
        tol=0.01,
    )
    msg_info = get_magnetic_space_group_from_operations(little_group)

    assert msg_info is not None
    assert msg_info["mpg_symbol"] == "1"


@pytest.mark.parametrize(
    "source_path",
    [
        "tests/testset/mcif_241130_no2186/0.1114_CeAlSi.mcif",
        "tests/testset/mcif_241130_no2186/0.37_U3Al2Si3.mcif",
        "tests/testset/mcif_241130_no2186/2.115_Er2CuMn5O12.mcif",
    ],
)
def test_find_spin_group_handles_near_involution_seitz_symbol_noise(source_path):
    result = find_spin_group(source_path)

    assert result.primitive_magnetic_cell_ssg_seitz


def test_find_spin_group_stably_handles_repeated_near_involution_symbol_runs():
    path = "tests/testset/mcif_241130_no2186/0.1114_CeAlSi.mcif"

    for _ in range(8):
        result = find_spin_group(path)
        assert result.index == "109.44.1.2.P2"
        assert result.primitive_magnetic_cell_ssg_seitz


def test_identify_index_transform_can_reach_database_standard_generators():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    debug = _build_identify_standardization_debug(result)

    assert result.index == "38.6.1.2.P2"
    assert debug["reduce_info"]
    assert debug["standard_generator_maps"]
    assert len(debug["standard_generator_maps"]) == len(
        debug["database_standard_generators"]
    )

    for generated, database in zip(
        debug["standard_generator_maps"],
        debug["database_standard_generators"],
    ):
        assert generated["space"].shape == (4, 4)
        assert database.shape == (4, 4)
        assert is_matrix_equal(generated["space"], database, tol=0.001)


def test_find_spin_group_exposes_tensor_outputs():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.tensor_outputs
    assert result.BCDTensor is not None
    assert result.MSGBCDTensor is not None
    assert result.QMDTensor is not None
    assert result.MSGQMDTensor is not None
    assert result.IMDTensor is not None
    assert result.MSGIMDTensor is not None
    assert result.AHE_woSOC is not None
    assert result.AHE_wSOC is not None

    assert result.BCDTensor["free_parameters"] == 2
    assert result.MSGBCDTensor["free_parameters"] == 2
    assert len(result.BCDTensor["relations"]) == 2
    assert len(result.MSGBCDTensor["relations"]) == 2
    assert result.AHE_woSOC["is_zero"] is True
    assert result.AHE_wSOC["free_parameters"] == 1
    assert result.AHE_wSOC["is_zero"] is False
    assert result.MSGQMDTensor["free_parameters"] == 3
    assert result.MSGIMDTensor["free_parameters"] == 1


def test_crse_w_soc_tensor_inputs_match_legacy_magnetic_point_group_behavior():
    result = find_spin_group("tests/testset/mcif_241130_no2186/2.35_CrSe.mcif")

    assert result.index == "194.149.3.3"
    assert result.conf == "Noncoplanar"
    assert result.AHE_wSOC["free_parameters"] == 1
    assert result.MSGBCDTensor["free_parameters"] == 1
    assert result.MSGQMDTensor["free_parameters"] == 2
    assert result.MSGIMDTensor["free_parameters"] == 1


@pytest.mark.parametrize(
    ("source_path", "expected_index"),
    [
        ("tests/testset/mcif_241130_no2186/0.435_Pb5Fe3TiO11Cl.mcif", "123.129.2.1.L"),
        ("tests/testset/mcif_241130_no2186/1.234_Ca2Sr2IrO6.mcif", "2.2.2.2.P1"),
        ("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif", "148.2.2.3"),
    ],
)
def test_find_spin_group_preserves_identify_indices_when_translation_representatives_require_integer_shifts(
    source_path,
    expected_index,
):
    result = find_spin_group(source_path)

    assert result.index == expected_index


def test_find_spin_group_prefers_nontrivial_translation_generators_for_identify_when_available():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.435_Pb5Fe3TiO11Cl.mcif")
    translation_spins = [
        np.asarray(item[0], dtype=float)
        for item in result.identify_index_details["translation_maps"]
    ]

    assert np.allclose(translation_spins[0], -np.eye(3), atol=1e-6)
    assert np.allclose(translation_spins[1], -np.eye(3), atol=1e-6)
    assert np.allclose(translation_spins[2], np.eye(3), atol=1e-6)


@pytest.mark.parametrize(
    "source_path",
    [
        "examples/0.800_MnTe.mcif",
        "tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif",
        "tests/testset/mcif_241130_no2186/1.274_DyFeWO6.mcif",
    ],
)
def test_identify_translation_maps_keep_exact_nofrac_basis_translations(source_path):
    result = find_spin_group(source_path)
    translations = _identify_translation_vectors(result.identify_index_details)

    assert len(translations) == 3
    assert np.allclose(translations[0], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(translations[1], np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(translations[2], np.array([0.0, 0.0, 1.0]), atol=1e-6)


@pytest.mark.parametrize(
    ("source_path", "expected_index"),
    [
        ("tests/testset/mcif_241130_no2186/1.274_DyFeWO6.mcif", "7.1.2.19"),
        ("tests/testset/mcif_241130_no2186/1.828_ZnFe2O4.mcif", "115.1.2.6"),
        ("tests/testset/mcif_241130_no2186/2.108_Tb3NbO7.mcif", "17.1.2.6"),
        ("tests/testset/mcif_241130_no2186/2.75_Sr2Fe3S2O3.mcif", "10.11.4.6"),
        ("tests/testset/mcif_241130_no2186/3.19_CoO.mcif", "134.2.2.2"),
        ("tests/testset/mcif_241130_no2186/3.4_MgCr2O4.mcif", "119.3.4.14"),
    ],
)
def test_find_spin_group_matches_historical_noncoplanar_identify_indices_with_exact_basis_lifts(
    source_path,
    expected_index,
):
    result = find_spin_group(source_path)

    assert result.conf == "Noncoplanar"
    assert result.index == expected_index


def test_find_spin_group_from_data_matches_file_based_flow():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "examples/0.800_MnTe.mcif"
    )
    result = find_spin_group_from_data(
        "0.800_MnTe.mcif",
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert result.index == "194.164.1.1.L"
    assert result.conf == "Collinear"
    assert result.primitive_magnetic_cell_ssg_type == "t"


def test_find_spin_group_exposes_input_space_group_metadata_from_identified_magnetic_primitive():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.11_DyFeO3.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    assert result.input_space_group_number == 62
    assert result.input_space_group_symbol == "Pnma"
    assert result.input_space_group_number != ssg.international_symbol["sg_number"]
    assert result.input_space_group_symbol != ssg.international_symbol["sg_symbol"]


def test_identify_spin_space_group_result_keeps_input_space_group_context_off_ssg():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.11_DyFeO3.mcif"
    )
    cell, _ = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting="in_lattice",
    ).get_primitive_structure(magnetic=True)
    identify_result = identify_spin_space_group_result(cell, find_primitive=False)

    assert identify_result.input_space_group is not None
    assert identify_result.input_space_group.number == 62
    assert identify_result.input_space_group.symbol == "Pnma"
    assert not hasattr(identify_result.ssg, "input_space_group_number")


def test_find_spin_group_exposes_input_space_group_metadata_for_type_k_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif")
    ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    assert result.primitive_magnetic_cell_ssg_type == "k"
    assert result.input_space_group_number == 55
    assert result.input_space_group_symbol == "Pbam"
    assert result.input_space_group_number != ssg.international_symbol["sg_number"]


def test_find_spin_group_from_data_preserves_input_space_group_metadata():
    lattice_factors, positions, elements, occupancies, labels, moments = parse_cif_file(
        "tests/testset/mcif_241130_no2186/0.11_DyFeO3.mcif"
    )
    result = find_spin_group_from_data(
        "0.11_DyFeO3.mcif",
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert result.input_space_group_number == 62
    assert result.input_space_group_symbol == "Pnma"


def test_find_spin_group_exposes_source_structure_metadata_from_mcif():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert result.source_parent_space_group["name_H_M_alt"] == "I m -3"
    assert result.source_parent_space_group["IT_number"] == 204
    assert result.source_parent_space_group["transform_Pp_abc"] == "a,b,c;0,0,0"
    assert result.source_parent_space_group["child_transform_Pp_abc"] == "2a,2b,2c;0,0,0"
    assert result.source_cell_parameter_strings["_cell_length_a"] == "14.88540"
    assert result.source_cell_parameter_strings["_cell_angle_alpha"] == "90.00000"


def test_find_spin_group_reports_quasi2d_diagnostics_without_changing_3d_fields():
    result = find_spin_group(
        "tests/testset/structure_2d_1.cif",
        calculation_mode="quasi2d",
    )

    assert result.index == "164.149.3.1.P"
    assert result.spinsplitting_wo_soc == "k-dependent"
    assert result.quasi_2d["status"] == "explicit"
    assert result.quasi_2d["source"] == "runtime_parameter"
    assert result.quasi_2d["dimension"] == "2d"
    assert result.quasi_2d["vacuum_axis_input"] == "c"
    assert result.quasi_2d["interpretation"] == "in_plane_k_dependent"
    assert result.quasi_2d["spin_splitting_2d"] == "spin splitting"
    assert "spin_splitting_2d_interpretation" not in result.to_dict()
    assert "spin_splitting_2d" not in result.to_dict()
    assert "is_alter_2d" not in result.to_dict()
    assert result.quasi_2d["diagnostic_points"][0]["label"] == "GP"
    assert result.quasi_2d["diagnostic_points"][0]["plane_classification"] == "in_plane"
    assert result.quasi_2d["generic_point_comparison"]["status"] == "compared"
    assert result.quasi_2d["generic_point_comparison"]["k_input_changed"] is True
    assert (
        result.quasi_2d["generic_point_comparison"]["spin_splitting_changed"]
        is False
    )
    assert result.quasi_2d["kpoint_projection_summary"]["by_plane_count"]["mixed"] > 0


def test_quasi2d_auto_mode_does_not_report_not_applicable_magnetic_phase():
    result = find_spin_group(
        "tests/testset/mcif_241130_no2186/0.1008_Sr2ErRuO6.mcif",
        calculation_mode="auto",
    )

    assert result.index == "14.2.1.1.L"
    assert result.magnetic_phase == "AFM(Altermagnet)\n(SOM)"
    assert result.quasi_2d["status"] == "not_applicable"
    assert result.quasi_2d["dimension"] == "3d_or_unknown"
    assert "magnetic_phase" not in result.quasi_2d


def test_quasi2d_preprocessing_resolves_heuristic_axis_before_padding():
    lattice = np.diag([3.0, 3.0, 6.0])
    positions = np.array([[0.0, 0.0, 0.45], [0.5, 0.5, 0.55]])

    mode, axis = resolve_quasi2d_preprocessing(
        lattice,
        positions,
        calculation_mode="quasi2d",
        vacuum_axis=None,
    )
    padded_lattice, _, padding = prepare_quasi2d_input_cell(
        lattice,
        positions,
        calculation_mode=mode,
        vacuum_axis=axis,
    )

    assert mode == "quasi2d"
    assert axis == "c"
    assert padding["applied"] is True
    assert np.isclose(np.linalg.norm(padded_lattice[2]), 20.0)

    auto_mode, auto_axis = resolve_quasi2d_preprocessing(
        lattice,
        positions,
        calculation_mode="auto",
        vacuum_axis=None,
    )
    assert auto_mode == "auto"
    assert auto_axis is None


@pytest.mark.parametrize(
    ("calculation_mode", "vacuum_axis", "message"),
    [
        ("unknown", "c", "Unknown calculation_mode"),
        ("quasi2d", "unknown", "Unknown vacuum_axis"),
    ],
)
def test_quasi2d_preprocessing_rejects_unknown_runtime_parameters(
    calculation_mode,
    vacuum_axis,
    message,
):
    with pytest.raises(ValueError, match=message):
        resolve_quasi2d_preprocessing(
            np.diag([3.0, 3.0, 20.0]),
            np.array([[0.0, 0.0, 0.5]]),
            calculation_mode=calculation_mode,
            vacuum_axis=vacuum_axis,
        )


def test_quasi2d_magnetic_phase_uses_generated_generic_point_after_vacuum_padding():
    result = find_spin_group(
        "tests/testset/mcif_241130_no2186/0.105_ErVO3.mcif",
        calculation_mode="quasi2d",
        vacuum_axis="c",
    )

    quasi_2d = result.quasi_2d
    assert quasi_2d["geometry"]["vacuum_padding"]["applied"] is True
    assert quasi_2d["generic_point_2d"]["label"] == "GP"
    assert quasi_2d["generic_point_2d"]["role"] == "generic_point_2d"
    assert quasi_2d["generic_point_2d"]["source"] == "generated"
    assert quasi_2d["generic_point_2d"]["display_k_symbol"] == "GP:(u,v,0)"
    assert quasi_2d["generic_point_2d"]["display_setting"] == "input_reciprocal"
    assert quasi_2d["generic_point_2d"]["matched_acc_label"] is None
    assert quasi_2d["generic_point_2d"]["spin_splitting"] == "spin splitting"
    assert quasi_2d["generic_point_2d"]["little_group_order"] == 2
    required_display_fields = {
        "role",
        "label",
        "display_k_symbol",
        "display_setting",
        "source",
        "ssg_little_group_symbol_2d",
        "spin_polarizations",
        "msg_little_group_symbol_2d",
        "msg_spin_polarization_2d",
    }
    assert required_display_fields <= set(quasi_2d["display_kpoints"][0])
    assert quasi_2d["generic_point_2d"] == quasi_2d["display_kpoints"][0]
    assert quasi_2d["display_kpoints"][0]["role"] == "generic_point_2d"
    assert quasi_2d["display_kpoints"][0]["display_k_symbol"] == "GP:(u,v,0)"
    assert quasi_2d["display_kpoints"][0]["ssg_little_group_symbol_2d"] is not None
    assert quasi_2d["display_kpoints"][0]["spin_polarizations"] == ["0", "0", "Sz"]
    assert quasi_2d["display_kpoints"][0]["msg_little_group_symbol_2d"] == "1"
    assert quasi_2d["display_kpoints"][0]["msg_spin_polarization_2d"] == ["Sx", "Sy", "Sz"]
    assert quasi_2d["diagnostic_points"][0]["label"] == "GP"
    assert quasi_2d["diagnostic_points"][0]["kind"] == "generated_in_plane_generic_probe"
    assert quasi_2d["diagnostic_points"][0]["plane_classification"] == "in_plane"
    assert quasi_2d["diagnostic_points"][0]["spin_splitting"] == "spin splitting"
    assert quasi_2d["generic_point_comparison"]["gp_2d"]["spin_splitting"] == "spin splitting"
    assert quasi_2d["interpretation"] == "in_plane_k_dependent"
    assert quasi_2d["spin_splitting_2d"] == "spin splitting"
    assert quasi_2d["is_alter_2d"] == "(Altermagnet)"
    assert quasi_2d["magnetic_phase"] == "AFM(Altermagnet)\n(SOM)"


def test_v2se2o_quasi2d_case_study_uses_final_acc_primitive_transform_chain():
    result = find_spin_group(
        "tests/testset/V2Se2O_2d.mcif",
        calculation_mode="quasi2d",
    )

    assert result.index == "123.47.1.1.L"
    assert result.acc == "4/mmmP"
    assert result.spinsplitting_wo_soc == "k-dependent"
    assert result.is_alter == "(Altermagnet)"

    quasi_2d = result.quasi_2d
    assert quasi_2d["status"] == "explicit"
    assert quasi_2d["source"] == "runtime_parameter"
    assert quasi_2d["dimension"] == "2d"
    assert quasi_2d["vacuum_axis_input"] == "c"
    assert quasi_2d["interpretation"] == "in_plane_k_dependent"
    assert quasi_2d["spin_splitting_2d"] == "spin splitting"
    assert quasi_2d["is_alter_2d"] == "(Altermagnet)"
    assert quasi_2d["magnetic_phase"] == "AFM(Altermagnet)\n(SOM)"
    assert quasi_2d["spin_texture_config_no_soc"]["source"] == (
        "quasi2d_ossg_unit_cartesian_in_plane_ops"
    )
    assert quasi_2d["spin_texture_config_no_soc"]["basis_setting"] == (
        "quasi2d_ossg_unit_cartesian_in_plane"
    )
    assert quasi_2d["spin_texture_config_no_soc"]["spin_texture_type"] == "d-wave"
    assert quasi_2d["spin_texture_config_no_soc"]["in_plane_k_axes"] == ["a", "b"]
    assert quasi_2d["spin_texture_config_no_soc"]["operation_audit"][
        "plane_preserving_operation_count"
    ] > 0
    assert quasi_2d["spin_texture_config_no_soc"]["operation_audit"][
        "non_plane_preserving_operation_count"
    ] == 0
    assert quasi_2d["spin_texture_config_soc"]["source"] == (
        "quasi2d_ossg_unit_cartesian_in_plane_msg_ops"
    )
    assert quasi_2d["spin_texture_config_basis"]["setting"] == (
        "quasi2d_ossg_unit_cartesian_in_plane"
    )
    assert "magnetic_phase_base" not in quasi_2d
    assert not hasattr(result, "magnetic_phase_2d")
    assert quasi_2d["KPOINTS"].startswith(result.KPOINTS.splitlines()[0])
    assert "Line-mode\nReciprocal" in quasi_2d["KPOINTS"]
    assert "  0.000000   0.000000   0.500000 ! Z" not in quasi_2d["KPOINTS"]
    assert "  0.417000   0.237000   0.000000 ! D" in quasi_2d["KPOINTS"]
    kpoints_2d = [
        row
        for row in quasi_2d["kpoints"]
        if row["kind"] == "acc_table" and row["plane_classification"] == "in_plane"
    ]
    assert [row["label"] for row in kpoints_2d] == ["Γ", "M", "X", "Δ", "Σ", "Y", "D"]
    assert [row["k_symbol_2d"] for row in kpoints_2d] == [
        "Γ:(0,0,0)",
        "M:(1/2,1/2,0)",
        "X:(0,1/2,0)",
        "Δ:(0,v,0)",
        "Σ:(u,u,0)",
        "Y:(u,1/2,0)",
        "D:(u,v,0)",
    ]
    for key in (
        "ssg_little_group_symbol_2d",
        "msg_little_group_symbol_2d",
        "msg_spin_polarization_2d",
        "ssg_little_group_ops_2d",
        "ssg_little_group_seitz_latex_2d",
        "msg_little_group_ops_2d",
        "msg_little_group_seitz_latex_2d",
    ):
        assert len(quasi_2d[key]) == len(kpoints_2d)
    assert quasi_2d["msg_little_group_symbol_2d"][0] == result.msg_little_group_symbols[1]
    assert quasi_2d["msg_spin_polarization_2d"][0] == result.msg_spin_polarizations[1]
    assert quasi_2d["diagnostic_points"][0]["label"] == "GP"
    assert quasi_2d["diagnostic_points"][0]["k_symbol_2d"] == "GP:(0.237,0.371,0)"
    assert quasi_2d["diagnostic_points"][0]["plane_classification"] == "in_plane"
    assert quasi_2d["diagnostic_points"][0]["spin_splitting"] == "spin splitting"
    assert quasi_2d["diagnostic_points"][0]["matched_acc_label"] == "D"
    assert quasi_2d["diagnostic_points"][0]["matched_acc_k_symbol_2d"] == "D:(u,v,0)"
    assert quasi_2d["generic_point_2d"]["label"] == "GP"
    assert quasi_2d["generic_point_2d"]["role"] == "generic_point_2d"
    assert quasi_2d["generic_point_2d"]["source"] == "acc_table"
    assert quasi_2d["generic_point_2d"]["display_k_symbol"] == "GP:(u,v,0)"
    assert quasi_2d["generic_point_2d"]["input_k_symbol"] == "GP:(u,v,0)"
    assert quasi_2d["generic_point_2d"]["acc_primitive_k_symbol"] == "D:(u,v,0)"
    assert quasi_2d["generic_point_2d"]["matched_acc_label"] == "D"
    assert quasi_2d["generic_point_2d"]["spin_splitting"] == "spin splitting"
    assert quasi_2d["display_kpoints"][0]["role"] == "generic_point_2d"
    assert quasi_2d["display_kpoints"][0]["source"] == "acc_table"
    assert quasi_2d["display_kpoints"][0]["display_k_symbol"] == "GP:(u,v,0)"
    required_display_fields = {
        "role",
        "label",
        "display_k_symbol",
        "display_setting",
        "source",
        "ssg_little_group_symbol_2d",
        "spin_polarizations",
        "msg_little_group_symbol_2d",
        "msg_spin_polarization_2d",
    }
    assert quasi_2d["generic_point_2d"] == quasi_2d["display_kpoints"][0]
    assert required_display_fields <= set(quasi_2d["display_kpoints"][0])
    assert required_display_fields <= set(quasi_2d["display_kpoints"][1])
    assert quasi_2d["display_kpoints"][0]["spin_polarizations"] == ["Sx", "0", "0"]
    assert quasi_2d["display_kpoints"][0]["msg_spin_polarization_2d"] == ["Sx", "Sy", "0"]
    assert quasi_2d["display_kpoints"][1]["role"] == "path_point"
    assert quasi_2d["display_kpoints"][1]["source"] == "acc_table"
    assert quasi_2d["display_kpoints"][1]["msg_spin_polarization_2d"] == result.msg_spin_polarizations[1]
    assert quasi_2d["generic_point_comparison"]["gp_3d"]["label"] == "GP"
    assert quasi_2d["generic_point_comparison"]["gp_3d"]["plane_classification"] == "mixed"
    assert quasi_2d["generic_point_comparison"]["gp_2d"]["label"] == "GP"
    assert quasi_2d["generic_point_comparison"]["gp_2d"]["k_symbol_2d"] == "GP:(0.237,0.371,0)"
    assert quasi_2d["generic_point_comparison"]["gp_2d"]["plane_classification"] == "in_plane"
    assert quasi_2d["generic_point_comparison"]["k_input_changed"] is True
    assert quasi_2d["generic_point_comparison"]["spin_splitting_changed"] is False
    assert quasi_2d["generic_point_comparison"]["summary"] == "k_changed_spin_splitting_same"

    _assert_setting_transform_inverse(result.T_input_to_G0std, result.T_G0std_to_input)
    _assert_setting_transform_inverse(result.T_input_to_L0std, result.T_L0std_to_input)
    _assert_setting_transform_inverse(
        result.T_input_to_acc_primitive,
        result.T_acc_primitive_to_input,
    )
    _assert_setting_transform_inverse(
        result.T_acc_primitive_to_G0std,
        result.T_G0std_to_acc_primitive,
    )
    _assert_setting_transform_inverse(
        result.T_acc_primitive_to_L0std,
        result.T_L0std_to_acc_primitive,
    )
    _assert_setting_transform_chain(
        result.T_input_to_G0std,
        result.T_G0std_to_acc_primitive,
        result.T_input_to_acc_primitive,
    )
    _assert_setting_transform_chain(
        result.T_input_to_L0std,
        result.T_L0std_to_acc_primitive,
        result.T_input_to_acc_primitive,
    )
    _assert_setting_transform_chain(
        result.T_input_to_convention,
        result.T_convention_to_acc_primitive,
        result.T_input_to_acc_primitive,
    )


def test_quasi2d_calculation_mode_parameter_overrides_geometry_heuristic():
    source_path = next(Path("tests/testset/quasi2d_small/2CrI3-1").glob("*.mcif"))
    result = find_spin_group(
        str(source_path),
        calculation_mode="quasi2d",
        vacuum_axis="c",
    )

    assert result.index == "149.149.1.1.L"
    assert result.quasi_2d["calculation_mode"] == "quasi2d"
    assert result.quasi_2d["status"] == "explicit"
    assert result.quasi_2d["source"] == "runtime_parameter"
    assert result.quasi_2d["dimension"] == "2d"
    assert result.quasi_2d["vacuum_axis_input"] == "c"
    assert result.quasi_2d["spin_splitting_2d"] == "spin splitting"
    assert result.quasi_2d["interpretation"] == "in_plane_k_dependent"
    assert result.quasi_2d["spin_texture_config_no_soc"]["basis_setting"] == (
        "quasi2d_ossg_unit_cartesian_in_plane"
    )
    assert result.quasi_2d["spin_texture_config_no_soc"]["in_plane_k_axes"] == ["a", "b"]
    assert result.quasi_2d["generic_point_comparison"]["k_input_changed"] is True
    assert result.quasi_2d["generic_point_comparison"]["spin_splitting_changed"] is False


def test_quasi2d_spin_texture_config_honors_explicit_a_vacuum_axis():
    result = find_spin_group(
        "tests/testset/V2Se2O_2d.mcif",
        calculation_mode="quasi2d",
        vacuum_axis="a",
    )

    quasi_2d = result.quasi_2d
    assert quasi_2d["vacuum_axis_input"] == "a"
    assert quasi_2d["spin_texture_config_basis"]["in_plane_k_axes"] == ["b", "c"]
    assert quasi_2d["spin_texture_config_no_soc"]["in_plane_k_axes"] == ["b", "c"]
    assert quasi_2d["spin_texture_config_no_soc"]["k_variable_labels"] == {
        "ky": "input reciprocal b*",
        "kz": "input reciprocal c*",
    }
    basis_text = " ".join(quasi_2d["spin_texture_config_no_soc"]["basis"])
    assert "kx" not in basis_text
    basis_latex = " ".join(quasi_2d["spin_texture_config_no_soc"]["basis_latex"])
    assert "k_{x}" not in basis_latex
    assert quasi_2d["spin_texture_config_no_soc"]["operation_audit"][
        "non_plane_preserving_operation_count"
    ] == 0


def test_quasi2d_spin_texture_honors_full_route_component_selection():
    result = find_spin_group(
        "tests/testset/V2Se2O_2d.mcif",
        calculation_mode="quasi2d",
        vacuum_axis="c",
        components=(),
    )

    quasi_2d = result.quasi_2d
    assert quasi_2d["spin_texture_config_no_soc"] is None
    assert quasi_2d["spin_texture_config_soc"] is None
    assert quasi_2d["spin_texture_config_basis"] is None
    assert quasi_2d["magnetic_phase"] == "AFM(Altermagnet)\n(SOM)"
    assert quasi_2d["display_kpoints"][0]["role"] == "generic_point_2d"


def test_quasi2d_little_group_symbols_do_not_materialize_all_3d_symbols(monkeypatch):
    def _unexpected_all_symbols(_self):
        raise AssertionError("quasi-2D should identify only displayed in-plane symbols")

    monkeypatch.setattr(
        group_module.SpinSpaceGroup,
        "little_groups_symbols",
        property(_unexpected_all_symbols),
    )

    result = find_spin_group(
        "tests/testset/V2Se2O_2d.mcif",
        calculation_mode="quasi2d",
        vacuum_axis="c",
        components=(),
    )

    symbols = result.quasi_2d["ssg_little_group_symbol_2d"]
    assert symbols
    assert all(symbol is not None for symbol in symbols)


def test_quasi2d_spin_texture_config_preserves_free_variable_names():
    payload = _classify_quasi2d_spin_texture_config(
        [
            {
                "Q": -np.eye(2),
                "S": -np.eye(3),
            }
        ],
        source="test",
        operation_audit={
            "input_operation_count": 1,
            "plane_preserving_operation_count": 1,
            "non_plane_preserving_operation_count": 0,
            "skipped_operation_count": 0,
            "max_kept_plane_residual": 0.0,
            "max_skipped_plane_residual": None,
            "plane_residual_tol": 1e-8,
        },
        in_plane_axes=["b", "c"],
        k_names=("ky", "kz"),
        calibration_atol_limit=1e-8,
        relax_without_reference=False,
    )

    assert payload is not None
    assert payload["in_plane_k_axes"] == ["b", "c"]
    assert payload["k_variable_labels"] == {
        "ky": "input reciprocal b*",
        "kz": "input reciprocal c*",
    }
    basis_text = " ".join(payload["basis"])
    assert "kx" not in basis_text
    assert "ky" in basis_text
    assert "kz" in basis_text
    basis_latex = " ".join(payload["basis_latex"])
    assert "k_{x}" not in basis_latex
    assert "k_{y}" in basis_latex
    assert "k_{z}" in basis_latex


def test_quasi2d_kpoint_plane_uses_raw_vacuum_component_for_forced_axis():
    result = find_spin_group(
        "tests/testset/quasi2d_small/1AgVP2S6-1/"
        "5.1.1.1,5.5.1.1.L,a,m1_p0_p0_p0_p0_m1_p0_p0_p0_p0_p1_"
        "p0_p0_p0_p0_p1,1,1,[[0,0,0]],epY-mpY-spY-ahcN.mcif",
        calculation_mode="quasi2d",
        vacuum_axis="a",
    )

    quasi_2d = result.quasi_2d
    rows = {
        row["label"]: row
        for row in quasi_2d["kpoints"]
        if row["kind"] == "acc_table"
    }
    assert rows["Γ"]["plane_classification"] == "in_plane"
    assert rows["Y"]["plane_classification"] == "in_plane"
    assert rows["U"]["plane_classification"] == "mixed"
    assert rows["R"]["plane_classification"] == "mixed"
    assert rows["U"]["vacuum_component_distance_to_integer"] == 0.0
    assert rows["U"]["vacuum_component_distance_to_zero"] > 2.9
    assert rows["U"]["k_input_reciprocal_raw"][0] < -2.9
    assert quasi_2d["kpoint_projection_summary"]["by_plane_count"]["in_plane"] == 2
    assert quasi_2d["KPOINTS_status"] == "ok"
    assert quasi_2d["KPOINTS_error"] is None
    assert "Y ***" in quasi_2d["KPOINTS"]
    assert "Γ ***" in quasi_2d["KPOINTS"]
    assert "| GP ***" in quasi_2d["KPOINTS"]


def test_kpoints_mark_spin_splitting_with_and_without_soc():
    matcher = group_module.BrillouinZoneMatcher(
        [
            ("GAMMA", "(0,0,0)", (True, False)),
            ("X", "(1/2,0,0)", (False, True)),
            ("GP", "(u,0,0)", (True, True)),
        ]
    )
    kpoints_text = group_module.write_kpoints(
        {
            "point_coords": {
                "GAMMA": [0.0, 0.0, 0.0],
                "X": [0.5, 0.0, 0.0],
            },
            "path": [("GAMMA", "X")],
        },
        matcher,
    )

    assert "(*** for spin splitting w/o SOC; ^^^ for spin splitting w/ SOC)" in kpoints_text
    assert "Γ ***" in kpoints_text
    assert "X ^^^" in kpoints_text
    assert "| GP ***^^^" in kpoints_text


def test_quasi2d_input_padding_expands_vacuum_axis_without_stretching_slab():
    lattice = np.diag([3.0, 3.0, 5.0])
    positions = np.array(
        [
            [0.0, 0.0, 0.45],
            [0.5, 0.5, 0.55],
        ],
        dtype=float,
    )

    padded_lattice, padded_positions, padding = prepare_quasi2d_input_cell(
        lattice,
        positions,
        calculation_mode="quasi2d",
        vacuum_axis="c",
    )

    assert padding["status"] == "expanded"
    assert padding["applied"] is True
    assert np.allclose(padded_lattice[0], lattice[0])
    assert np.allclose(padded_lattice[1], lattice[1])
    assert np.isclose(np.linalg.norm(padded_lattice[2]), 20.0)
    assert np.allclose(padded_positions[:, 2], [0.4875, 0.5125])
    original_span = (positions[1, 2] - positions[0, 2]) * np.linalg.norm(lattice[2])
    padded_span = (
        (padded_positions[1, 2] - padded_positions[0, 2])
        * np.linalg.norm(padded_lattice[2])
    )
    assert np.isclose(padded_span, original_span)
    assert np.isclose(padding["final_vacuum_gap_length"], 19.5)


def test_quasi2d_input_padding_is_not_used_for_3d_mode():
    lattice = np.diag([3.0, 3.0, 5.0])
    positions = np.array([[0.0, 0.0, 0.45]], dtype=float)

    padded_lattice, padded_positions, padding = prepare_quasi2d_input_cell(
        lattice,
        positions,
        calculation_mode="3d",
        vacuum_axis="c",
    )

    assert padding is None
    assert np.allclose(padded_lattice, lattice)
    assert np.allclose(padded_positions, positions)


def test_quasi2d_input_tags_do_not_control_runtime_mode(tmp_path):
    source_path = next(Path("tests/testset/quasi2d_small/2CrI3-1").glob("*.mcif"))
    tagged_path = tmp_path / "tagged_CrI3.mcif"
    tagged_path.write_text(
        source_path.read_text(encoding="utf-8")
        + "\n"
        + '_space_group_spin.fsg_calculation_mode "quasi2d"\n'
        + '_space_group_spin.fsg_vacuum_axis_input "c"\n',
        encoding="utf-8",
    )

    result = find_spin_group(str(tagged_path))

    assert result.index == "149.149.1.1.L"
    assert result.quasi_2d is None
    assert "spin_splitting_2d_interpretation" not in result.to_dict()
    assert "spin_splitting_2d" not in result.to_dict()
    assert "is_alter_2d" not in result.to_dict()
    assert not hasattr(result, "magnetic_phase_2d")


def test_find_spin_group_exposes_input_setting_payload_for_magnetic_primitive_poscar(tmp_path):
    source_result = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(source_result.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    result = find_spin_group(str(poscar_path))

    assert result.input_cell_detail is not None
    assert not hasattr(result, "input_cell_detail_oriented")
    assert result.input_ssg_may_be_incomplete is False
    assert result.input_setting_warning is None
    assert result.input_wp_chain is not None
    assert result.input_ssg_ops_spin_cartesian
    assert result.input_ssg_seitz_latex_spin_cartesian
    assert result.input_ssg_ops_spin_oriented
    assert result.input_ssg_seitz_latex_spin_oriented
    assert len(result.input_ssg_ops_spin_cartesian) == len(
        result.input_ssg_seitz_latex_spin_cartesian
    )
    assert len(result.input_ssg_ops_spin_oriented) == len(
        result.input_ssg_seitz_latex_spin_oriented
    )
    assert isinstance(result.input_spin_only_direction_spin_cartesian, str)
    assert isinstance(result.input_spin_only_direction_spin_oriented, str)

    _assert_setting_transform_inverse(result.T_input_to_G0std, result.T_G0std_to_input)
    _assert_setting_transform_inverse(result.T_input_to_L0std, result.T_L0std_to_input)
    _assert_setting_transform_inverse(
        result.T_input_to_acc_primitive,
        result.T_acc_primitive_to_input,
    )
    _assert_setting_transform_inverse(
        result.T_input_to_convention,
        result.T_convention_to_input,
    )


def test_find_spin_group_input_setting_payload_allows_nonprimitive_when_ssg_matches_true_setting():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert result.input_cell_detail is not None
    assert result.input_ssg_may_be_incomplete is False
    assert not hasattr(result, "input_ssg_index")
    assert result.input_setting_warning is None
    assert result.input_wp_chain is not None
    assert result.input_ssg_ops_spin_cartesian
    assert result.input_ssg_seitz_latex_spin_cartesian
    assert result.input_ssg_ops_spin_oriented
    assert result.input_ssg_seitz_latex_spin_oriented
    assert len(result.input_ssg_ops_spin_cartesian) == len(
        result.input_ssg_seitz_latex_spin_cartesian
    )
    assert len(result.input_ssg_ops_spin_oriented) == len(
        result.input_ssg_seitz_latex_spin_oriented
    )
    _assert_setting_transform_inverse(result.T_input_to_G0std, result.T_G0std_to_input)
    _assert_setting_transform_inverse(result.T_input_to_L0std, result.T_L0std_to_input)
    _assert_setting_transform_inverse(
        result.T_input_to_acc_primitive,
        result.T_acc_primitive_to_input,
    )
    _assert_setting_transform_inverse(
        result.T_input_to_convention,
        result.T_convention_to_input,
    )
    input_identified_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_INPUT_IDENTIFIED)
    assert f'_space_group_spin.number_Chen_Liu  "{result.index}"' in input_identified_scif
    assert "_space_group_spin.fsg_G0_number  148" in input_identified_scif
    assert "_space_group_spin.fsg_L0_number  2" in input_identified_scif
    assert "_space_group_spin.fsg_it  3" in input_identified_scif
    assert "_space_group_spin.fsg_ik  4" in input_identified_scif
    assert '_space_group_spin.fsg_spin_space_point_group_name  "23"' in input_identified_scif
    assert '_space_group_spin.fsg_magnetic_phase  "AFM(SOM)"' in input_identified_scif
    assert '_space_group_spin.fsg_spin_arithmetic_crystal_class_symbol  "-3R"' in input_identified_scif


def test_find_spin_group_reuses_transformed_primitive_ssg_for_compatible_input_supercell(monkeypatch):
    find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")
    original_identify = find_spin_group_module.identify_spin_space_group_result
    calls = []

    def counting_identify(*args, **kwargs):
        calls.append(len(args[0].positions))
        return original_identify(*args, **kwargs)

    monkeypatch.setattr(
        find_spin_group_module,
        "identify_spin_space_group_result",
        counting_identify,
    )

    result = find_spin_group("tests/testset/mcif_241130_no2186/1.31_MnO.mcif")

    assert calls == [4]
    assert result.input_setting_warning is None
    assert result.input_ssg_may_be_incomplete is False
    assert len(result.input_ssg_ops_spin_cartesian) == 1536


def test_find_spin_group_input_setting_payload_warns_when_input_ssg_differs_from_true_setting(monkeypatch):
    find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")
    original_identify = find_spin_group_module.identify_spin_space_group_result
    calls = []

    def counting_identify(*args, **kwargs):
        calls.append(len(args[0].positions))
        return original_identify(*args, **kwargs)

    monkeypatch.setattr(
        find_spin_group_module,
        "identify_spin_space_group_result",
        counting_identify,
    )

    result = find_spin_group("tests/testset/mcif_241130_no2186/0.396_MnPtGa.mcif")

    assert calls == [6]
    assert result.input_cell_detail is not None
    assert result.input_ssg_may_be_incomplete is True
    assert result.input_setting_warning == (
        "Input-cell SSG differs from the magnetic-primitive SSG transformed "
        "to the input setting; input_ssg_index=63.12.1.2.P2."
    )
    assert result.input_wp_chain is None
    assert result.input_ssg_ops_spin_cartesian
    assert result.input_ssg_seitz_latex_spin_cartesian
    assert result.input_ssg_ops_spin_oriented
    assert result.input_ssg_seitz_latex_spin_oriented

    input_identified_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_INPUT_IDENTIFIED)
    assert '_space_group_spin.number_Chen_Liu  "63.12.1.2.P2"' in input_identified_scif
    assert (
        '_space_group_spin.fsg_input_setting_warning  '
        '"Input-cell SSG differs from the magnetic-primitive SSG transformed '
        'to the input setting; input_ssg_index=63.12.1.2.P2."'
    ) in input_identified_scif
    assert input_identified_scif.index("# repo-local FINDSPINGROUP extensions") < input_identified_scif.index(
        "_space_group_spin.fsg_input_setting_warning"
    ) < input_identified_scif.index("_space_group_spin.fsg_oriented_spin_space_group_name_linear")
    for suppressed_tag in [
        "_space_group_spin.fsg_G0_number",
        "_space_group_spin.fsg_L0_number",
        "_space_group_spin.fsg_it",
        "_space_group_spin.fsg_ik",
        "_space_group_spin.fsg_spin_space_point_group_name",
        "_space_group_spin.fsg_magnetic_phase",
        "_space_group_spin.fsg_spin_arithmetic_crystal_class_symbol",
        "_space_group_spin.fsg_magnetic_arithmetic_crystal_class_symbol",
    ]:
        assert suppressed_tag not in input_identified_scif
    with pytest.raises(ValueError, match="Unsupported scif output cell_mode: input"):
        result.to_scif(cell_mode="input")


def test_input_setting_oriented_seitz_uses_cartesian_order_with_input_spin_frame():
    with pytest.warns(RuntimeWarning, match="Identify-index database entry unavailable"):
        result = find_spin_group("tests/testset/mcif_241130_no2186/1.669_KFe(PO3F)2.mcif")

    assert result.identify_index_details is None
    assert result.index.startswith("not in identify-index database:")
    assert (
        "No identify-index reduction record for L0=143, G0=147, it=2, ik=12, iso=64"
        in result.index
    )


@pytest.mark.parametrize(
    (
        "path",
        "expected_index",
        "expected_convention_setting",
        "expected_acc",
        "expected_msg_num",
        "expected_msg_symbol",
        "expected_label",
        "expected_is_self_automorphism",
        "expected_convention_count",
        "expected_acc_conventional_count",
        "expected_acc_primitive_count",
    ),
    [
        (
            "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
            "148.2.4.1",
            "G0std",
            "-3R",
            1247,
            "R-3",
            "setting_change",
            False,
            240,
            240,
            80,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.456_RbFeO2.mcif",
            "227.216.1.1.L",
            "G0std",
            "m-3mF",
            1216,
            "I4_1'/a'm'd",
            "self_automorphism",
            True,
            400,
            400,
            100,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.458_CsFeO2.mcif",
            "227.216.1.1.L",
            "G0std",
            "m-3mF",
            1216,
            "I4_1'/a'm'd",
            "self_automorphism",
            True,
            400,
            400,
            100,
        ),
        (
            "examples/CoNb3S6_tripleQ.mcif",
            "182.4.4.2",
            "G0std",
            "6mmP",
            1257,
            "P32'1",
            "self_automorphism",
            True,
            80,
            80,
            80,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.712_VNb3S6.mcif",
            "182.149.1.1.L",
            "G0std",
            "6/mmmP",
            131,
            "C2'2'2_1",
            "self_automorphism",
            True,
            20,
            20,
            20,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif",
            "29.4.1.1.L",
            "G0std",
            "mmmP",
            200,
            "Pca'2_1'",
            "self_automorphism",
            True,
            36,
            36,
            36,
        ),
        (
            "tests/testset/mcif_241130_no2186/0.2_Cd2Os2O7.mcif",
            "227.2.1.2",
            "G0std",
            "m-3mF",
            1633,
            "Fd-3m'",
            "self_automorphism",
            True,
            88,
            88,
            22,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
            "6.6.2.4.L",
            "L0std",
            "2/mP",
            35,
            "P_cc",
            "self_automorphism",
            True,
            64,
            64,
            64,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.347_CuFeO2.mcif",
            "13.15.2.1.L",
            "L0std",
            "2/mC",
            98,
            "C_a2/c",
            "self_automorphism",
            True,
            32,
            32,
            16,
        ),
        (
            "tests/testset/mcif_241130_no2186/1.348_CuFeO2.mcif",
            "12.12.2.1.L",
            "L0std",
            "2/mC",
            97,
            "C_c2/c",
            "self_automorphism",
            True,
            80,
            80,
            40,
        ),
    ],
)
def test_find_spin_group_exposes_convention_to_acc_conventional_chain_for_representative_cases(
    path,
    expected_index,
    expected_convention_setting,
    expected_acc,
    expected_msg_num,
    expected_msg_symbol,
    expected_label,
    expected_is_self_automorphism,
    expected_convention_count,
    expected_acc_conventional_count,
    expected_acc_primitive_count,
):
    result = find_spin_group(path)

    assert result.index == expected_index
    assert result.convention_ssg_setting == expected_convention_setting
    assert result.acc == expected_acc
    assert result.msg_num == expected_msg_num
    assert result.msg_symbol == expected_msg_symbol

    assert result.acc_conventional_cell_setting == "acc_conventional"
    assert result.acc_conventional_ssg_setting == "acc_conventional"
    assert result.selected_standard_setting == expected_convention_setting
    assert result.T_selected_standard_to_acc_conventional_label == expected_label
    assert (
        result.T_selected_standard_to_acc_conventional_is_self_automorphism
        is expected_is_self_automorphism
    )

    assert len(result.convention_cell_detail["positions"]) == expected_convention_count
    assert len(result.acc_conventional_cell_detail["positions"]) == expected_acc_conventional_count
    assert len(result.acc_primitive_magnetic_cell_detail["positions"]) == expected_acc_primitive_count

    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[0], dtype=float),
        np.asarray(result.T_selected_standard_to_acc_conventional[0], dtype=float),
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[1], dtype=float),
        np.asarray(result.T_selected_standard_to_acc_conventional[1], dtype=float),
        atol=1e-8,
    )
    assert np.asarray(result.T_convention_to_acc_conventional[0], dtype=float).shape == (3, 3)
    assert np.asarray(result.T_convention_to_acc_conventional[1], dtype=float).shape == (3,)


@pytest.mark.parametrize(
    ("path", "expected_matrix", "expected_shift", "expected_label", "expected_setting"),
    [
        (
            "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
            np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
            np.zeros(3),
            "setting_change",
            "G0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/0.456_RbFeO2.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "G0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "L0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.347_CuFeO2.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "L0std",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.348_CuFeO2.mcif",
            np.eye(3),
            np.zeros(3),
            "self_automorphism",
            "L0std",
        ),
    ],
)
def test_find_spin_group_exposes_expected_representative_convention_to_acc_conventional_transforms(
    path,
    expected_matrix,
    expected_shift,
    expected_label,
    expected_setting,
):
    result = find_spin_group(path)

    assert result.convention_ssg_setting == expected_setting
    assert result.selected_standard_setting == expected_setting
    assert result.T_selected_standard_to_acc_conventional_label == expected_label
    assert np.allclose(
        np.asarray(result.T_selected_standard_to_acc_conventional[0], dtype=float),
        expected_matrix,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_selected_standard_to_acc_conventional[1], dtype=float),
        expected_shift,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[0], dtype=float),
        expected_matrix,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_conventional[1], dtype=float),
        expected_shift,
        atol=1e-8,
    )


def test_current_basis_symbol_builder_transports_r_centering_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")
    payload = build_international_symbol(SpinSpaceGroup(result.g0_standard_ssg_ops), basis_mode="current")

    centering_vectors = {
        item["label"]: np.asarray(item["vector"], dtype=float)
        for item in payload["translation_details"]
        if item["label"].startswith("b_")
    }

    assert np.allclose(centering_vectors["b_1"], np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0]), atol=1e-6)
    assert np.allclose(centering_vectors["b_2"], np.array([1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0]), atol=1e-6)


def test_current_basis_symbol_builder_keeps_p_translation_targets_for_conb3s6():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")
    payload = build_international_symbol(SpinSpaceGroup(result.g0_standard_ssg_ops), basis_mode="current")

    translation_vectors = {
        item["label"]: np.asarray(item["vector"], dtype=float)
        for item in payload["translation_details"]
        if item["label"].startswith("t_")
    }

    assert np.allclose(translation_vectors["t_a"], np.array([0.5, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(translation_vectors["t_b"], np.array([0.0, 0.5, 0.0]), atol=1e-6)
    assert np.allclose(translation_vectors["t_c"], np.array([0.0, 0.0, 0.0]), atol=1e-6)


def _current_basis_symbol_context(result):
    if result.convention_ssg_setting == "G0std":
        ops = result.g0_standard_ssg_ops
        ssg = SpinSpaceGroup(ops)
        sg_num = int(ssg.G0_num)
        bravais = ssg.G0_symbol[0]
        current_to_standard, current_to_standard_shift = _compose_symbol_setting_transform(
            np.asarray(ssg.transformation_to_G0std, dtype=float),
            np.asarray(ssg.origin_shift_to_G0std, dtype=float),
            np.asarray(ssg.transformation_to_G0std_id, dtype=float),
            np.asarray(ssg.origin_shift_to_G0std_id, dtype=float),
        )
    else:
        ops = result.l0_standard_ssg_ops
        ssg = SpinSpaceGroup(ops)
        sg_num = int(ssg.L0_num)
        bravais = ssg.L0_symbol[0]
        current_to_standard = np.asarray(ssg.transformation_to_L0std, dtype=float)
        current_to_standard_shift = np.asarray(ssg.origin_shift_to_L0std, dtype=float)

    named_ops, centering_trans = _parse_sg_generator_ops(sg_num)
    if not centering_trans:
        centering_trans = [vec for _, vec in _default_centering_vectors(bravais)]
    named_ops_cur, centering_cur = _transport_standard_generators_to_current_basis(
        named_ops,
        centering_trans,
        current_to_standard,
        current_to_standard_shift,
    )
    return ssg, named_ops_cur, centering_cur


@pytest.mark.parametrize(
    "path",
    [
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
        "examples/CoNb3S6_tripleQ.mcif",
        "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
        "tests/testset/mcif_241130_no2186/1.347_CuFeO2.mcif",
        "tests/testset/mcif_241130_no2186/1.348_CuFeO2.mcif",
    ],
)
def test_current_basis_symbol_builder_matches_standard_named_generators_after_transport(path):
    result = find_spin_group(path)
    ssg, named_ops_cur, _ = _current_basis_symbol_context(result)

    for rotation, translation in named_ops_cur:
        assert _find_real_operation(ssg.nssg, rotation, translation, tol=1e-4) is not None


@pytest.mark.parametrize(
    "path",
    [
        "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif",
        "examples/CoNb3S6_tripleQ.mcif",
    ],
)
def test_current_basis_symbol_builder_matches_required_translation_targets_after_transport(path):
    result = find_spin_group(path)
    ssg, _, centering_cur = _current_basis_symbol_context(result)

    for axis in range(3):
        assert (
            _select_preferred_primitive_translation_match(
                ssg.nssg,
                axis,
                tol=1e-4,
                identity_real_ops=ssg.identity_real_nssg_ops,
            )
            is not None
        )

    for vec in centering_cur:
        assert (
            _select_preferred_translation_match(
                ssg.nssg,
                vec,
                tol=1e-4,
                identity_real_ops=ssg.identity_real_nssg_ops,
            )
            is not None
        )


@pytest.mark.parametrize(
    ("path", "expected_type", "expected_setting"),
    [
        ("examples/0.800_MnTe.mcif", "t", "G0std"),
        ("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif", "k", "L0std"),
        ("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif", "g", "G0std"),
    ],
)
def test_find_spin_group_exposes_convention_selected_standard_payloads(
    path,
    expected_type,
    expected_setting,
):
    result = find_spin_group(path)
    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)

    assert result.primitive_magnetic_cell_ssg_type == expected_type
    assert result.convention_cell_setting == expected_setting
    assert result.convention_ssg_setting == expected_setting
    assert result.selected_standard_setting == expected_setting
    assert result.convention_ssg_spin_frame_setting == "ossg_oriented_spin_frame"

    assert result.convention_ssg_seitz == public_ossg.seitz_symbols
    assert result.convention_ssg_seitz_latex == public_ossg.seitz_symbols_latex
    assert (
        result.convention_ssg_international_linear
        == public_ossg.international_symbol_linear_current_frame
    )
    assert (
        result.convention_ssg_international_latex
        == public_ossg.international_symbol_latex_current_frame
    )


def test_input_to_standard_transforms_remain_nontrivial_for_1048():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif")

    g0_forward = np.asarray(result.T_input_to_G0std[0], dtype=float)
    g0_shift = np.asarray(result.T_input_to_G0std[1], dtype=float)
    l0_forward = np.asarray(result.T_input_to_L0std[0], dtype=float)
    l0_shift = np.asarray(result.T_input_to_L0std[1], dtype=float)

    assert not np.allclose(g0_forward, np.eye(3), atol=1e-8)
    assert not np.allclose(l0_forward, np.eye(3), atol=1e-8)
    assert not np.allclose(g0_shift, np.zeros(3), atol=1e-8)
    assert not np.allclose(l0_shift, np.zeros(3), atol=1e-8)


def test_input_to_standard_transforms_remain_nontrivial_for_324():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    g0_forward = np.asarray(result.T_input_to_G0std[0], dtype=float)
    g0_shift = np.asarray(result.T_input_to_G0std[1], dtype=float)
    l0_forward = np.asarray(result.T_input_to_L0std[0], dtype=float)
    l0_shift = np.asarray(result.T_input_to_L0std[1], dtype=float)

    assert not np.allclose(g0_forward, np.eye(3), atol=1e-8)
    assert not np.allclose(l0_forward, np.eye(3), atol=1e-8)
    assert np.allclose(
        g0_forward,
        np.array(
            [
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
                [-4.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0],
            ]
        ),
        atol=1e-8,
    )
    assert np.allclose(g0_shift, np.zeros(3), atol=1e-8)
    assert np.allclose(l0_shift, np.zeros(3), atol=1e-8)


def test_find_spin_group_exposes_acc_primitive_aliases_and_setting_tags():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.primitive_magnetic_cell_setting == "acc_primitive"
    assert result.acc_primitive_magnetic_cell_setting == "acc_primitive"
    assert result.primitive_magnetic_cell == result.acc_primitive_magnetic_cell
    assert result.primitive_magnetic_cell_detail == result.acc_primitive_magnetic_cell_detail
    assert result.primitive_magnetic_cell_poscar == result.acc_primitive_magnetic_cell_poscar

    assert result.primitive_magnetic_cell_ssg_setting == "acc_primitive"
    assert result.acc_primitive_ssg_setting == "acc_primitive"
    assert result.primitive_magnetic_cell_ssg_ops == result.acc_primitive_ssg_ops
    assert result.primitive_magnetic_cell_ssg_seitz == result.acc_primitive_ssg_seitz
    assert (
        result.primitive_magnetic_cell_ssg_international_linear
        == result.acc_primitive_ssg_international_linear
    )
    assert (
        result.primitive_magnetic_cell_ssg_international_latex
        == result.acc_primitive_ssg_international_latex
    )

    assert result.KPOINTS_setting == "acc_primitive"
    assert result.KPOINTS_real_space_setting == "acc_primitive"
    assert result.spin_polarizations_setting == "acc_primitive_poscar_spin_frame"
    assert result.spin_polarizations_real_space_setting == "acc_primitive"
    assert result.spin_polarizations_spin_frame == "acc_primitive_poscar_spin_frame"
    assert result.spin_polarizations_acc_cartesian_setting == "acc_primitive_cartesian"
    assert result.spin_polarizations == result.spin_polarizations_acc_poscar_spin_frame
    assert result.spin_polarizations_acc_poscar_spin_frame_setting == "acc_primitive_poscar_spin_frame"
    assert result.spin_polarizations_acc_poscar_spin_frame is not None
    assert (
        result.real_cartesian_to_spin_frame
        == result.acc_primitive_real_cartesian_to_poscar_spin_frame
    )
    assert (
        result.spin_frame_to_real_cartesian
        == result.poscar_spin_frame_to_acc_primitive_real_cartesian
    )


def test_find_spin_group_exposes_input_and_public_magnetic_primitive_layers():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    assert result.input_magnetic_primitive_cell_setting == "input_magnetic_primitive"
    assert result.magnetic_primitive_cell_setting == "acc_primitive"
    assert result.acc_primitive_magnetic_cell_setting == "acc_primitive"

    assert result.magnetic_primitive_cell == result.acc_primitive_magnetic_cell
    assert result.magnetic_primitive_cell_detail == result.acc_primitive_magnetic_cell_detail
    assert result.magnetic_primitive_cell_poscar == result.acc_primitive_magnetic_cell_poscar

    assert result.primitive_magnetic_cell == result.magnetic_primitive_cell
    assert result.primitive_magnetic_cell_detail == result.magnetic_primitive_cell_detail
    assert result.primitive_magnetic_cell_poscar == result.magnetic_primitive_cell_poscar

    assert result.input_magnetic_primitive_cell_detail != result.magnetic_primitive_cell_detail
    assert result.input_magnetic_primitive_cell_poscar != result.magnetic_primitive_cell_poscar

    assert result.magnetic_primitive_ssg_ops == result.acc_primitive_ssg_ops
    assert result.magnetic_primitive_ssg_international_linear == result.acc_primitive_ssg_international_linear
    assert result.primitive_magnetic_cell_ssg_ops == result.magnetic_primitive_ssg_ops


def test_scif_transform_tags_use_basis_relation_contract():
    result = find_spin_group("tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif")

    default_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED)
    assert (
        "_space_group_spin.fsg_transform_to_input_Pp  "
        "'2/3a+1/3b-4/3c,-1/3a-2/3b-4/3c,-1/3a+1/3b-4/3c;0,0,0'"
    ) in default_scif
    assert (
        "_space_group_spin.fsg_transform_to_magnetic_primitive_Pp  "
        "'-1/3a-2/3b-1/3c,-1/3a+1/3b-1/3c,2/3a+1/3b-1/3c;0,0,0'"
    ) in default_scif
    assert (
        "_space_group_spin.fsg_transform_to_L0std_Pp  "
        "'-c,-2/3a-1/3b+1/3c,1/3a+2/3b+1/3c;0,0,0'"
    ) in default_scif
    assert "_space_group_spin.fsg_transform_to_G0std_Pp  'a,b,c;0,0,0'" in default_scif
    assert '_space_group_spin.fsg_spin_arithmetic_crystal_class_symbol  "-3R"' in default_scif
    assert '_space_group_spin.fsg_magnetic_arithmetic_crystal_class_symbol  "-3R"' in default_scif

    with pytest.raises(ValueError, match="Unsupported scif output cell_mode: input"):
        result.to_scif(cell_mode="input")

    input_identified_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_INPUT_IDENTIFIED)
    assert "_space_group_spin.fsg_transform_to_input_Pp  'a,b,c;0,0,0'" in input_identified_scif
    assert "_space_group_spin.fsg_input_setting_warning" not in input_identified_scif
    assert "_space_group_spin.fsg_G0_number  148" in input_identified_scif

    magnetic_primitive_scif = result.to_scif(cell_mode=SCIF_CELL_MODE_MAGNETIC_PRIMITIVE)
    assert (
        "_space_group_spin.fsg_transform_to_magnetic_primitive_Pp  'a,b,c;0,0,0'"
        in magnetic_primitive_scif
    )

    assert result.primitive_msg_ops_setting == "acc_primitive"
    assert result.acc_primitive_msg_ops_setting == "acc_primitive"
    assert result.primitive_msg_ops == result.acc_primitive_msg_ops
    assert result.msg_spin_polarizations_setting == "acc_primitive_poscar_spin_frame"
    assert result.msg_spin_polarizations_real_space_setting == "acc_primitive"
    assert result.msg_spin_polarizations_spin_frame == "acc_primitive_poscar_spin_frame"
    assert result.msg_spin_polarizations_acc_cartesian_setting == "acc_primitive_cartesian"
    assert result.msg_spin_polarizations == result.msg_spin_polarizations_acc_poscar_spin_frame
    assert result.symbol_calibration_tol == result.acc_primitive_ssg_symbol_calibration_tol
    assert result.convention_ssg_symbol_calibration_tol is not None
    assert result.primitive_magnetic_cell_ssg_seitz_descriptions
    assert result.acc_primitive_ssg_seitz_descriptions
    assert result.acc_primitive_wp_chain
    assert result.acc_primitive_ssg_ops_cartesian
    assert result.acc_primitive_ssg_seitz_cartesian
    assert result.acc_primitive_ssg_seitz_latex_cartesian
    assert result.acc_primitive_ssg_ops_oriented
    assert result.acc_primitive_ssg_seitz_oriented
    assert result.acc_primitive_ssg_seitz_latex_oriented
    assert len(result.acc_primitive_ssg_ops_cartesian) == len(result.acc_primitive_ssg_seitz_cartesian)
    assert len(result.acc_primitive_ssg_ops_oriented) == len(result.acc_primitive_ssg_seitz_oriented)
    for seitz_symbols in (
        result.acc_primitive_ssg_seitz_cartesian,
        result.acc_primitive_ssg_seitz_latex_cartesian,
        result.acc_primitive_ssg_seitz_oriented,
        result.acc_primitive_ssg_seitz_latex_oriented,
        result.convention_ssg_seitz,
        result.convention_ssg_seitz_latex,
    ):
        assert seitz_symbols
        assert "tau" not in seitz_symbols[0]
        assert "0,0,0" in seitz_symbols[0]
    assert result.acc_primitive_spin_only_direction_cartesian is not None
    assert result.acc_primitive_spin_only_direction_poscar_spin_frame is not None
    assert result.g0_standard_ssg_seitz_descriptions
    assert result.l0_standard_ssg_seitz_descriptions
    assert result.convention_ssg_seitz_descriptions


def test_find_spin_group_exposes_msg_acc_for_conb3s6_tripleq():
    result = find_spin_group("examples/CoNb3S6_tripleQ.mcif")

    assert result.acc == "6mmP"
    assert result.msg_acc == "3m1P"


def test_mag_symmetry_result_exposes_core_group_identifiers():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    payload = result.to_dict()

    assert result.fsg_version == __version__
    assert payload["fsg_version"] == __version__
    assert next(iter(payload)) == "fsg_version"
    assert result.G0_symbol == "P6_3/mmc"
    assert result.G0_num == 194
    assert result.L0_symbol == "P-3m1"
    assert result.L0_num == 164
    assert result.it == 2
    assert result.ik == 1
    assert result.spin_part_point_group == "∞/mm"
    assert result.SSPG_symbol_hm == "∞/mm"
    assert result.SSPG_symbol_s == "D∞h"
    assert result.input_space_group_symbol == "P6_3/mmc"
    assert result.input_space_group_number == 194
    assert payload["G0_symbol"] == "P6_3/mmc"
    assert payload["L0_num"] == 164
    assert payload["spin_part_point_group"] == "∞/mm"
    assert payload["SSPG_symbol_hm"] == "∞/mm"
    assert payload["SSPG_symbol_s"] == "D∞h"
    assert payload["input_space_group_number"] == 194


def test_mag_symmetry_result_exposes_compact_operation_views():
    result = find_spin_group("examples/0.800_MnTe.mcif")

    assert sorted(result.operation_views) == [
        "convention_cartesian",
        "convention_oriented",
        "input_cartesian",
        "input_oriented",
        "magnetic_primitive_cartesian",
        "magnetic_primitive_oriented",
    ]
    convention_all = result.operation_views["convention_oriented"]["views"]["all"]
    assert len(convention_all["ops"]) == len(convention_all["seitz_latex"])
    assert convention_all["indices"] == list(range(1, len(convention_all["ops"]) + 1))

    for setting_key in (
        "convention_oriented",
        "magnetic_primitive_oriented",
        "input_oriented",
    ):
        all_view = result.operation_views[setting_key]["views"]["all"]
        generator_view = result.operation_views[setting_key]["views"]["generators"]
        assert "ops" not in generator_view
        assert "seitz_latex" not in generator_view
        assert generator_view["indices"]
        assert len(generator_view["indices"]) == len(set(generator_view["indices"]))
        assert max(generator_view["indices"]) <= len(all_view["ops"])

    convention_nssg = SpinSpaceGroup(result.convention_nssg_ops)
    symbol_payload = convention_nssg.get_international_symbol(basis_mode="current")
    assert symbol_payload["generator_operations"]
    assert len(symbol_payload["generator_operations"]) == len(
        result.operation_views["convention_oriented"]["views"]["generators"]["indices"]
    )

    assert result.operation_views["convention_oriented"]["default_view"] == "nssg"
    assert "nssg_collinear" not in result.operation_views["convention_oriented"]["views"]
    nssg_view = result.operation_views["convention_oriented"]["views"]["nssg"]
    assert nssg_view["ops"] == convention_all["ops"]
    assert nssg_view["seitz_latex"] == convention_all["seitz_latex"]
    assert nssg_view["indices"] == convention_all["indices"]
    assert nssg_view["note"]["type"] == "collinear"
    assert nssg_view["note"]["spin_only_symbol_hm"] == "∞m"
    assert nssg_view["note"]["spin_only_symbol_s"] == "C∞v"
    assert nssg_view["note"]["spin_frame"] == "oriented"
    assert nssg_view["operation_count"] == len(SpinSpaceGroup(result.convention_ssg_ops).nssg)

    msg_view = result.operation_views["convention_oriented"]["views"]["msg"]
    assert msg_view["label"] == "MSG operations"
    assert "ops" in msg_view
    assert len(msg_view["ops"]) == len(msg_view["seitz_latex"])
    assert msg_view["operation_count"] == len(SpinSpaceGroup(result.convention_ssg_ops).msg_ops)
    assert msg_view["indices"] == list(range(1, msg_view["operation_count"] + 1))
    assert sorted(msg_view["ops"][0]) == [
        "index",
        "real_rotation",
        "spin_rotation",
        "time_reversal",
        "translation",
    ]
    for row in msg_view["ops"]:
        real_rotation = np.asarray(row["real_rotation"], dtype=float)
        spin_rotation = np.asarray(row["spin_rotation"], dtype=float)
        expected_spin_rotation = (
            int(row["time_reversal"])
            * float(np.linalg.det(real_rotation))
            * real_rotation
        )
        assert np.allclose(spin_rotation, expected_spin_rotation, atol=1e-6)
    for cartesian_key, oriented_key in (
        ("convention_cartesian", "convention_oriented"),
        ("magnetic_primitive_cartesian", "magnetic_primitive_oriented"),
        ("input_cartesian", "input_oriented"),
    ):
        assert (
            result.operation_views[cartesian_key]["views"]["msg"]["operation_count"]
            == result.operation_views[oriented_key]["views"]["msg"]["operation_count"]
        )

    spin_translations = result.operation_views["magnetic_primitive_cartesian"]["views"][
        "spin_translations"
    ]
    assert "ops" not in spin_translations
    assert spin_translations["indices"]


def test_operation_views_expose_msg_indices_when_msg_is_all_subset():
    result = find_spin_group("examples/0.200_Mn3Sn.mcif")

    msg_view = result.operation_views["convention_oriented"]["views"]["msg"]
    assert msg_view["label"] == "MSG operations"
    assert "ops" not in msg_view
    assert msg_view["indices"] == [1, 10, 14, 20, 32, 36, 42, 47]
    assert msg_view["operation_count"] == 8
    assert result.operation_views["convention_cartesian"]["views"]["msg"]["indices"] == (
        msg_view["indices"]
    )
    assert result.operation_views["convention_cartesian"]["views"]["msg"]["operation_count"] == (
        msg_view["operation_count"]
    )


def test_mag_symmetry_result_exposes_structured_output_contract():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    structured = result.to_structured_dict()

    assert sorted(structured) == [
        "artifacts",
        "cells",
        "groups",
        "legacy",
        "properties",
        "summary",
        "transforms",
    ]
    assert not hasattr(result, "structured")

    assert structured["summary"]["index"] == result.index
    assert structured["summary"]["conf"] == result.conf
    assert structured["summary"]["phase"] == result.magnetic_phase
    assert structured["summary"]["acc"] == result.acc

    assert structured["groups"]["input_space_group"] == {
        "number": result.input_space_group_number,
        "symbol": result.input_space_group_symbol,
        "basis_or_setting": result.input_space_group_basis_or_setting,
        "is_centrosymmetric": result.sg_is_centrosymmetric,
        "is_polar": result.sg_is_polar,
        "is_chiral": result.sg_is_chiral,
    }
    assert structured["groups"]["msg"]["num"] == result.msg_num
    assert structured["groups"]["msg"]["type"] == result.msg_type
    assert structured["groups"]["msg"]["bns_number"] == result.msg_bns_number
    assert structured["groups"]["ssg_by_cell"]["acc_primitive"]["ops"] == result.acc_primitive_ssg_ops

    assert structured["cells"]["acc_primitive"]["setting"] == "acc_primitive"
    assert structured["cells"]["acc_primitive"]["detail"] == result.acc_primitive_magnetic_cell_detail
    assert structured["cells"]["database_standard"]["selected"] == result.selected_standard_setting

    assert structured["transforms"]["input_to_acc_primitive"] == result.T_input_to_acc_primitive
    assert (
        structured["transforms"]["audit"]["acc_primitive_resolution"]
        == result.acc_primitive_resolution_audit
    )

    assert structured["properties"]["magnetic_phase"]["details"] == result.magnetic_phase_details
    assert structured["properties"]["quasi_2d"] == result.quasi_2d
    assert structured["properties"]["magnetic_site"] == result.magnetic_site_summary

    assert structured["artifacts"]["scif"]["default"] == result.scif
    assert structured["artifacts"]["kpoints"]["acc_primitive"]["text"] == result.KPOINTS
    assert structured["artifacts"]["poscar"]["acc_primitive"] == result.acc_primitive_magnetic_cell_poscar
    assert structured["legacy"]["index"] == result.index
    assert "structured" not in structured["legacy"]


def test_find_spin_group_exposes_explicit_gspg_payload_for_coplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    oriented_ssg = SpinSpaceGroup(result.convention_ssg_ops)

    assert repr(oriented_ssg.gspg) == result.gspg_symbol_linear
    assert oriented_ssg.gspg.empg_symbol == result.gspg_effective_mpg_symbol
    assert result.gspg_ops == _serialize_gspg_pairs(oriented_ssg.gspg.ops)
    assert result.gspg_raw_ops == _serialize_gspg_pairs(oriented_ssg.gspg.raw_ops)
    assert result.gspg_symbol_linear == "1|m 2_{001}|m 2_{001}|2 m|1"
    assert result.gspg_collinear_axis is None
    expected_direction = find_spin_group_module._format_spin_only_direction(
        oriented_ssg.sog_direction
    )
    assert f"Spin-only direction: {expected_direction}" in result.gspg_text
    assert "spin only:\n1 x,y,z,+1,u,v,w\n2 x,y,z,-1,-u,v,w" in result.gspg_text


def test_gspg_text_uses_stensor_compatible_xyz_uvw_separator():
    rows = [
        {
            "index": 10,
            "xyzt": "-x,y,-z,-1",
            "uvw": "-u,v,-2u+w",
        }
    ]

    assert find_spin_group_module._format_gspg_xyz_uvw_text(rows) == [
        "10 -x,y,-z,-1,-u,v,-2u+w"
    ]


def test_find_spin_group_reports_collinear_gspg_as_nssg_times_spin_only():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    oriented_ssg = SpinSpaceGroup(result.convention_ssg_ops)
    expected_nssg_point_ops = deduplicate_matrix_pairs(
        [[op[0], op[1]] for op in oriented_ssg.nssg],
        tol=oriented_ssg.tol,
    )

    assert result.conf == "Collinear"
    assert repr(oriented_ssg.gspg) == result.gspg_symbol_linear
    assert oriented_ssg.gspg.empg_symbol == result.gspg_effective_mpg_symbol
    assert result.gspg_ops == _serialize_gspg_pairs(expected_nssg_point_ops)
    assert result.gspg_raw_ops == _serialize_gspg_pairs(oriented_ssg.gspg.raw_ops)
    assert len(result.gspg_ops) < len(oriented_ssg.gspg.raw_ops)
    assert result.gspg_symbol_linear == "-1|6/ -1|m 1|m -1|m ∞_{110}m|1"
    assert result.gspg_collinear_axis == pytest.approx(oriented_ssg.gspg.collinear_axis.tolist())


def test_find_spin_group_exposes_gspg_xyz_uvw_and_spin_only_exports_for_collinear_case():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    oriented_ssg = SpinSpaceGroup(result.convention_ssg_ops)

    assert result.conf == "Collinear"
    assert result.gspg_collinear_axis == pytest.approx(oriented_ssg.gspg.collinear_axis.tolist())
    assert len(result.gspg_raw_ops) >= len(result.gspg_ops)
    assert result.gspg_spin_only_ops
    assert result.gspg_spin_only_ops_xyz_uvw
    assert all("xyzt" in item and "uvw" in item for item in result.gspg_ops_xyz_uvw)
    assert all("xyzt" in item and "uvw" in item for item in result.gspg_raw_ops_xyz_uvw)
    assert all("xyzt" in item and "uvw" in item for item in result.gspg_spin_only_ops_xyz_uvw)
    assert len(result.gspg_raw_ops_xyz_uvw) == len(result.gspg_raw_ops)


def test_find_spin_group_exposes_gspg_text_payload_from_public_ossg():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    summary_gspg = result.to_summary_dict()["gspg"]
    expected_direction = find_spin_group_module._format_spin_only_direction(
        result.gspg_collinear_axis
    )

    assert isinstance(result.gspg_text, str)
    assert result.gspg_text.splitlines()[:6] == [
        f"GSPG linear symbol: {result.gspg_symbol_linear}",
        f"Spin-space point group symbol: {result.SSPG_symbol_hm} ({result.SSPG_symbol_s})",
        f"Effective MPG: {result.gspg_effective_mpg_symbol}",
        f"Real-space setting: {result.convention_ssg_setting}",
        "Spin-frame setting: oriented",
        f"Spin-only direction: {expected_direction}",
    ]
    assert summary_gspg["spin_space_point_group_symbol_hm"] == result.SSPG_symbol_hm
    assert summary_gspg["spin_space_point_group_symbol_s"] == result.SSPG_symbol_s
    assert "Spin-only component:" not in result.gspg_text
    assert "nPG symbol:" not in result.gspg_text
    assert "\ngenerators (excluding spin-only):\n" in result.gspg_text
    assert result.gspg_generator_indices
    assert len(result.gspg_generator_ops) == len(result.gspg_generator_ops_xyz_uvw)
    assert len(result.gspg_generator_ops) == len(result.gspg_generator_indices)
    assert "\noperations:\n1 x,y,z,+1,u,v,w" in result.gspg_text
    assert "xyzt=" not in result.gspg_text
    assert f"\nspin only:\nCollinear direction: {expected_direction}" in result.gspg_text
    assert summary_gspg["text"] == result.gspg_text


def test_spin_splitting_numeric_classifies_gspg_generators_with_spin_only():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    operations = operation_pairs_from_gspg_ops(
        list(result.gspg_generator_ops) + list(result.gspg_spin_only_ops)
    )

    payload = classify_public_spin_texture_config(
        operations,
        source="gspg_generators",
    )

    assert payload["source"] == "gspg_generators"
    assert payload["spin_texture_type"] == "g-wave"
    assert payload["order"] == 4
    assert payload["spin_rank"] == 1
    assert payload["momentum_space_spin_configuration"] == "collinear"
    assert payload["basis_latex"]
    assert r"\sigma" in payload["basis_latex"][0]


def test_spin_texture_basis_by_order_is_opt_in():
    identity = np.eye(3)
    operations = [
        {
            "real_rotation": identity,
            "spin_rotation": identity,
        }
    ]

    leading = classify_public_spin_texture_config(operations, source="test")
    assert leading["order"] == 0
    assert leading["basis"][0].endswith(" + o(1)")
    assert "basis_by_order" not in leading

    through_second = classify_public_spin_texture_config(
        operations,
        source="test",
        basis_orders_through=2,
    )
    assert [payload["order"] for payload in through_second["basis_by_order"]] == [0, 1, 2]
    assert through_second["basis_by_order"][1]["basis"][0].endswith(" + o(k)")
    assert through_second["basis_by_order"][2]["basis"][0].endswith(" + o(k^2)")


def test_spin_texture_basis_by_order_keeps_forbidden_orders():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    operations = operation_pairs_from_gspg_ops(
        list(result.gspg_generator_ops) + list(result.gspg_spin_only_ops)
    )

    payload = classify_public_spin_texture_config(
        operations,
        source="gspg_generators",
        basis_orders_through=2,
    )

    assert payload["spin_texture_type"] == "forbidden"
    assert payload["order"] is None
    assert [order_payload["order"] for order_payload in payload["basis_by_order"]] == [0, 1, 2]
    assert [order_payload["nullity"] for order_payload in payload["basis_by_order"]] == [0, 0, 0]
    assert all(not order_payload["basis"] for order_payload in payload["basis_by_order"])


def test_spin_texture_basis_expression_latex_formatter():
    assert combine_spin_texture_basis_expression(
        "C1*((kx*ky)*sigma_x + (ky*kz)*sigma_x)"
    ) == "C1*((kx*ky + ky*kz)*sigma_x)"
    assert combine_spin_texture_basis_expression(
        "C1*((kx*ky)*sigma_x - (ky*kz)*sigma_x)"
    ) == "C1*((kx*ky - ky*kz)*sigma_x)"
    assert basis_expression_to_latex(
        "C1*((kx*ky*kz)*sigma_x - (sqrt(3)/3*kx*ky*kz)*sigma_z)"
    ) == (
        r"C_{1}\left(k_{x}k_{y}k_{z}\,\sigma_{x} - "
        r"\frac{\sqrt{3}}{3}k_{x}k_{y}k_{z}\,\sigma_{z}\right)"
    )
    assert basis_expression_to_latex(
        "C1*((-2/3*ky^3)*sigma_z + (kx*ky^2)*sigma_z)"
    ) == (
        r"C_{1}\left(\left(-\frac{2}{3}k_{y}^{3} + "
        r"k_{x}k_{y}^{2}\right)\,\sigma_{z}\right)"
    )
    suffixed_expression = (
        "C1*((kx*ky)*sigma_x + (ky*kz)*sigma_x) + o(k^2) + o(k^2)"
    )
    assert combine_spin_texture_basis_expression(suffixed_expression) == (
        "C1*((kx*ky + ky*kz)*sigma_x) + o(k^2)"
    )
    assert _append_basis_remainder_ascii([suffixed_expression], 2) == [
        "C1*((kx*ky + ky*kz)*sigma_x) + o(k^2)"
    ]
    assert _append_basis_remainder_ascii(
        ["C1*((kx*ky)*sigma_x + (ky*kz)*sigma_x) + o(k)"],
        2,
    ) == ["C1*((kx*ky + ky*kz)*sigma_x) + o(k^2)"]
    suffixed_latex = spin_texture_basis_latex([suffixed_expression])
    assert _append_basis_remainder_latex(suffixed_latex, 2) == [
        r"C_{1}\left(\left(k_{x}k_{y} + k_{y}k_{z}\right)\,\sigma_{x}\right) + o(k^{2})"
    ]
    assert _append_basis_remainder_latex(
        [
            r"C_{1}\left(\left(k_{x}k_{y} + k_{y}k_{z}\right)\,\sigma_{x}\right) + o(k)"
        ],
        2,
    ) == [
        r"C_{1}\left(\left(k_{x}k_{y} + k_{y}k_{z}\right)\,\sigma_{x}\right) + o(k^{2})"
    ]
    assert combine_spin_texture_basis_span(
        [
            "C1*((ky*kz)*sigma_z)",
            "C2*((kx*ky)*sigma_z)",
        ]
    ) == ["(C1*ky*kz + C2*kx*ky)*sigma_z"]


def test_spin_texture_runtime_record_keeps_single_remainder_after_grouping():
    record = _spin_texture_config_record(
        {
            "spin_texture_type": "d-wave",
            "momentum_space_spin_configuration": "collinear",
            "spin_rank": 1,
            "nullity": 1,
            "order": 2,
            "basis": [
                "C1*((kx*ky)*sigma_x + (ky*kz)*sigma_x) + o(k^2)",
            ],
        }
    )

    assert record["basis"] == ["C1*((kx*ky + ky*kz)*sigma_x) + o(k^2)"]
    assert record["basis_latex"] == [
        r"C_{1}\left(\left(k_{x}k_{y} + k_{y}k_{z}\right)\,\sigma_{x}\right) + o(k^{2})"
    ]
    assert record["basis_vectors"] == ["C1*((kx*ky + ky*kz)*sigma_x)"]

    multi_basis_record = _spin_texture_config_record(
        {
            "spin_texture_type": "d-wave",
            "momentum_space_spin_configuration": "collinear",
            "spin_rank": 1,
            "nullity": 2,
            "order": 2,
            "basis": [
                "C1*((ky*kz)*sigma_z)",
                "C2*((kx*ky)*sigma_z)",
            ],
        }
    )

    assert multi_basis_record["nullity"] == 2
    assert multi_basis_record["basis"] == [
        "(C1*ky*kz + C2*kx*ky)*sigma_z + o(k^2)"
    ]
    assert multi_basis_record["basis_latex"] == [
        r"\left(C_{1}k_{y}k_{z} + C_{2}k_{x}k_{y}\right)\,\sigma_{z} + o(k^{2})"
    ]
    assert multi_basis_record["basis_vectors"] == [
        "C1*((ky*kz)*sigma_z)",
        "C2*((kx*ky)*sigma_z)",
    ]


def test_spin_texture_canonical_nullspace_does_not_amplify_near_zero_pivots():
    basis = np.array(
        [
            [1e-7, 0.0],
            [1.0, 0.0],
            [0.0, 1e-7],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    canonical = canonicalize_nullspace(basis, zero_tol=1e-8)

    assert len(canonical) == 2
    assert all(np.max(np.abs(vector)) == pytest.approx(1.0) for vector in canonical)
    assert all(np.all(np.abs(vector) <= 1.0) for vector in canonical)
    assert np.allclose(canonical[0], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(canonical[1], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("mcif_path", "expected_index"),
    [
        ("tests/testset/mcif_241130_no2186/0.1035_PbNi1.76Mg0.24V2O8.mcif", "110.43.1.1.L"),
        ("tests/testset/mcif_241130_no2186/0.799_Sr2Co2O5.mcif", "46.28.2.1.L"),
    ],
)
def test_spin_texture_config_uses_ossg_unit_cartesian_generators(mcif_path, expected_index):
    result = find_spin_group(mcif_path)
    expected_spin_texture_config = get_spin_texture_config_for_ssg_label(expected_index)

    assert result.index == expected_index
    assert result.spin_texture_config_no_soc["source"] == "ossg_unit_cartesian_generators"
    assert result.spin_texture_config_no_soc["basis_setting"] == "ossg_unit_cartesian"
    for key in (
        "spin_texture_type",
        "order",
        "nullity",
        "spin_rank",
        "momentum_space_spin_configuration",
    ):
        assert result.spin_texture_config_no_soc[key] == expected_spin_texture_config[key]


@pytest.mark.parametrize(
    "path",
    [
        "examples/0.800_MnTe.mcif",
        "tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif",
        "tests/testset/mcif_241130_no2186/0.1005_Mn3RhGe.mcif",
    ],
)
def test_find_spin_group_gspg_generators_generate_public_gspg(path):
    result = find_spin_group(path)

    target_keys = {
        find_spin_group_module._gspg_pair_key(op)
        for op in result.gspg_ops
    }
    assert all(
        not np.allclose(op[1], np.eye(3), atol=0.02, rtol=0)
        for op in result.gspg_generator_ops
    )
    closure_keys = find_spin_group_module._gspg_pair_closure(
        result.gspg_generator_ops + result.gspg_spin_only_ops,
        tol=0.02,
        limit=4096,
    )
    generator_lines = result.gspg_text.split("\ngenerators (excluding spin-only):\n", 1)[1].split(
        "\n\noperations:",
        1,
    )[0].splitlines()

    assert result.gspg_generator_indices
    assert target_keys.issubset(closure_keys)
    assert len(generator_lines) == len(result.gspg_generator_indices)
    assert all(
        result.gspg_ops_xyz_uvw[index - 1]["index"] == index
        for index in result.gspg_generator_indices
    )


def test_find_spin_group_gspg_text_uses_identity_spin_only_for_noncoplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1005_Mn3RhGe.mcif")

    assert result.conf == "Noncoplanar"
    assert result.gspg_spin_only_component_symbol_s == "C1"
    assert "Spin-only direction: None" in result.gspg_text
    assert result.gspg_text.rsplit("spin only:\n", 1)[1].splitlines() == [
        "1 x,y,z,+1,u,v,w"
    ]


@pytest.mark.parametrize(
    ("path", "expected_linear"),
    [
        ("tests/testset/mcif_241130_no2186/0.454_PrScSb.mcif", "∞_{001}/mm|1"),
        ("tests/testset/mcif_241130_no2186/0.1001_PbMn2Ni6Te3O18.mcif", "∞_{001}m|1"),
    ],
)
def test_find_spin_group_distinguishes_collinear_spin_only_cinfv_vs_dinfh(
    path,
    expected_linear,
):
    result = find_spin_group(path)

    assert result.conf == "Collinear"
    assert expected_linear in result.gspg_symbol_linear


def test_find_spin_group_keeps_explicit_public_gspg_ops_for_coplanar_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1004_CsO2.mcif")

    assert result.conf == "Coplanar"
    assert result.gspg_ops == result.gspg_raw_ops


@pytest.mark.parametrize(
    ("path", "expected_type"),
    [
        ("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif", "k"),
        ("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif", "g"),
    ],
)
def test_find_spin_group_uses_componentized_gspg_symbol_for_type_k_and_g(path, expected_type):
    result = find_spin_group(path)

    assert result.primitive_magnetic_cell_ssg_type == expected_type
    assert result.gspg_symbol_linear is not None


def test_find_spin_group_uses_gspg_r_eq_i_spin_only_for_collinear_type_k_case():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif")

    assert result.conf == "Collinear"
    assert result.gspg_symbol_linear == "1|m ∞_{001}/mm|1"


def test_find_spin_group_uses_oriented_path_for_public_type_g_gspg_symbol():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif")

    assert result.gspg_symbol_linear == "3^{2}_{001}|-3 -1|1"


def test_find_spin_group_public_gspg_is_derived_from_public_ossg():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif")
    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)

    assert result.convention_ssg_international_linear == public_ossg.international_symbol_linear_current_frame
    assert result.convention_ssg_international_latex == public_ossg.international_symbol_latex_current_frame
    assert result.gspg_symbol_linear == public_ossg.gspg.symbol_linear


def test_find_spin_group_exposes_poscar_spin_frame_transform_and_polarizations():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    forward = np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float)
    backward = np.asarray(result.poscar_spin_frame_to_acc_primitive_real_cartesian, dtype=float)

    assert forward.shape == (3, 3)
    assert backward.shape == (3, 3)
    assert np.allclose(forward @ backward, np.eye(3), atol=1e-8)
    assert np.allclose(backward @ forward, np.eye(3), atol=1e-8)

    assert np.allclose(forward, np.eye(3), atol=1e-8)

    expected_poscar_spin_polarizations = SpinSpaceGroup(result.acc_primitive_ssg_ops).transform_spin(
        forward
    ).spin_polarizations
    assert result.spin_polarizations == expected_poscar_spin_polarizations
    assert result.spin_polarizations_acc_poscar_spin_frame == expected_poscar_spin_polarizations
    assert result.spin_polarizations_acc_poscar_spin_frame == result.spin_polarizations_acc_cartesian


def test_acc_primitive_poscar_preserves_core_acc_primitive_lattice_and_moments(tmp_path):
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(result.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    lattice, positions, elements, _occupancies, _labels, moments = parse_poscar_file(poscar_path)
    expected_lattice = np.asarray(result.acc_primitive_magnetic_cell_detail["lattice"], dtype=float)
    expected_positions = np.asarray(result.acc_primitive_magnetic_cell_detail["positions"], dtype=float)
    expected_moments = np.asarray(result.acc_primitive_magnetic_cell_detail["moments"], dtype=float)

    assert np.allclose(lattice, expected_lattice, atol=1e-6)
    assert elements == result.acc_primitive_magnetic_cell_detail["elements"]
    assert np.allclose(positions, expected_positions, atol=1e-8)
    assert np.allclose(moments, expected_moments, atol=1e-8)
    assert result.acc_primitive_magnetic_cell[2] == result.acc_primitive_magnetic_cell_detail["type_ids"]
    assert np.allclose(
        np.asarray(result.acc_primitive_magnetic_cell[1], dtype=float),
        expected_positions,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.acc_primitive_magnetic_cell[3], dtype=float),
        expected_moments,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float),
        np.eye(3),
        atol=1e-8,
    )


def test_acc_primitive_is_built_from_identified_index_p_map_and_roundtrips(tmp_path):
    result = find_spin_group("src/findspingroup/examples/1.237_VCl2.mcif")

    assert result.index == "164.149.6.1.P"
    assert result.msg_bns_number == "159.64"
    assert len(result.acc_primitive_magnetic_cell_detail["positions"]) == 18
    assert np.allclose(
        np.asarray(result.T_convention_to_acc_primitive[0], dtype=float),
        np.asarray([[-1, 2, 0], [-2, 1, 0], [0, 0, 1]], dtype=float),
        atol=1e-8,
    )

    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(result.acc_primitive_magnetic_cell_poscar, encoding="utf-8")
    lattice, positions, elements, occupancies, _labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        lattice,
        positions,
        elements,
        occupancies,
        moments,
        input_spin_setting="cartesian",
    )

    assert roundtrip.index == result.index
    assert roundtrip.msg_bns_number == result.msg_bns_number


def _normalize_origin_shift_for_test(vector):
    normalized = np.mod(np.asarray(vector, dtype=float), 1.0)
    normalized[np.isclose(normalized, 0.0, atol=1e-8)] = 0.0
    normalized[np.isclose(normalized, 1.0, atol=1e-8)] = 0.0
    return normalized


@pytest.mark.parametrize(
    ("path", "expected_strategy"),
    [
        (
            "tests/testset/mcif_241130_no2186/1.67_TmPtIn.mcif",
            "identify_space_transform_current_to_database_after_current",
        ),
        (
            "tests/testset/mcif_241130_no2186/1.75_BiMn2O5.mcif",
            "identify_space_transform_current_to_database_after_current",
        ),
        (
            "tests/testset/mcif_241130_no2186/2.101_TbSbTe.mcif",
            "identify_space_transform_current_to_database_after_current",
        ),
        (
            "tests/testset/mcif_241130_no2186/2.102_TbSbTe.mcif",
            "identify_space_transform_current_to_database_after_current",
        ),
    ],
)
def test_acc_primitive_database_gauge_transform_scales_origin_shift_and_roundtrips(
    tmp_path,
    path,
    expected_strategy,
):
    result = find_spin_group(path)
    audit = result.acc_primitive_resolution_audit["G0std_transform_selection"]

    assert audit["selected_strategy"] == expected_strategy

    current_candidate = next(
        candidate
        for candidate in audit["rejected_candidates"]
        if candidate["strategy"] == "current_integerized"
    )
    current_matrix = np.asarray(current_candidate["matrix"], dtype=float)
    current_shift = np.asarray(current_candidate["origin_shift"], dtype=float)
    identify_matrix = np.asarray(
        result.identify_index_details["space_group_transformation"][0],
        dtype=float,
    )
    identify_shift = np.asarray(
        result.identify_index_details["space_group_transformation"][1],
        dtype=float,
    )
    identify_matrix = np.linalg.inv(identify_matrix)
    identify_shift = -identify_matrix @ identify_shift

    expected_matrix = identify_matrix @ current_matrix
    expected_shift = _normalize_origin_shift_for_test(
        identify_matrix @ current_shift + identify_shift
    )

    assert np.allclose(
        np.asarray(audit["selected_matrix"], dtype=float),
        expected_matrix,
        atol=1e-8,
    )
    assert np.allclose(
        _normalize_origin_shift_for_test(audit["selected_origin_shift"]),
        expected_shift,
        atol=1e-8,
    )

    input_primitive_lattice = np.asarray(
        result.input_magnetic_primitive_cell_detail["lattice"],
        dtype=float,
    )
    acc_primitive_lattice = np.asarray(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        dtype=float,
    )
    lattice_relation = acc_primitive_lattice @ np.linalg.inv(input_primitive_lattice)
    rounded_relation = np.rint(lattice_relation)
    assert np.allclose(lattice_relation, rounded_relation, atol=1e-6)
    assert abs(round(np.linalg.det(rounded_relation))) == 1

    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(result.acc_primitive_magnetic_cell_poscar, encoding="utf-8")
    lattice, positions, elements, occupancies, _labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        lattice,
        positions,
        elements,
        occupancies,
        moments,
        input_spin_setting="cartesian",
    )

    assert roundtrip.index == result.index


def test_acc_primitive_lightweight_route_uses_identified_index_p_map():
    payload = find_spin_group_acc_primitive("src/findspingroup/examples/1.237_VCl2.mcif")

    assert payload["index"] == "164.149.6.1.P"
    assert len(payload["acc_primitive_cell_detail"]["positions"]) == 18
    assert np.allclose(
        np.asarray(payload["T_input_to_acc_primitive"][0], dtype=float),
        np.asarray([[-1, 2, 0], [-2, 1, 0], [0, 0, 1]], dtype=float),
        atol=1e-8,
    )


def test_acc_primitive_g0std_matrix_selection_uses_nofrac_lattice_shear_not_legacy_fallback():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.115_Dy3Ru4Al12.mcif")

    assert result.index == "12.2.2.3"
    audit = result.acc_primitive_resolution_audit["G0std_transform_selection"]
    assert audit["selected_strategy"] == "nofrac_lattice_shear:r2+=(-2)r0"
    assert all(
        "legacy" not in candidate["strategy"]
        for candidate in audit["rejected_candidates"]
    )
    assert np.allclose(
        np.asarray(audit["selected_matrix"], dtype=float),
        np.asarray([[0.0, -0.5, 0.5], [1.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
        atol=1e-8,
    )

    identity_translations = {
        tuple(np.round(np.mod(np.asarray(op[2], dtype=float), 1.0), 6))
        for op in result.convention_ssg_ops
        if np.allclose(np.asarray(op[1], dtype=float), np.eye(3), atol=1e-8)
    }
    assert (0.0, 0.5, 0.5) not in identity_translations
    assert (0.5, 0.0, 0.5) not in identity_translations


def test_acc_primitive_l0std_uses_direct_composed_identify_p_transform():
    payload = find_spin_group_acc_primitive("tests/testset/mcif_241130_no2186/1.367_Pu2O3.mcif")

    assert payload["index"] == "12.12.2.1.L"
    assert len(payload["acc_primitive_cell_detail"]["positions"]) == 10
    assert len(payload["acc_primitive_ssg_operation_matrices"]) == 32


def test_find_spin_group_exposes_convention_nssg_views():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)
    expected_nssg = SpinSpaceGroup(public_ossg.nssg)

    assert _serialize_ssg_ops(result.convention_nssg_ops) == _serialize_ssg_ops(expected_nssg.ops)
    assert result.convention_nssg_seitz == expected_nssg.seitz_symbols
    assert result.convention_nssg_seitz_latex == expected_nssg.seitz_symbols_latex

    nssg_view = result.operation_views["convention_oriented"]["views"]["nssg"]
    if "ops" in nssg_view:
        assert nssg_view["ops"] == result.operation_views["convention_oriented"]["views"]["all"]["ops"]
        nssg_ops_from_view = [
            [
                op["spin_rotation"],
                op["real_rotation"],
                op["translation"],
            ]
            for op in nssg_view["ops"]
        ]
    else:
        all_ops = list(public_ossg.ops)
        nssg_ops_from_view = [
            [
                np.asarray(op[0], dtype=float).tolist(),
                np.asarray(op[1], dtype=float).tolist(),
                np.asarray(op[2], dtype=float).tolist(),
            ]
            for op in (all_ops[index - 1] for index in nssg_view["indices"])
        ]
    assert nssg_ops_from_view == _serialize_ssg_ops(expected_nssg.ops)


@pytest.mark.parametrize(
    ("path", "expected_direction"),
    [
        ("examples/0.800_MnTe.mcif", "sqrt(2)/2,sqrt(2)/2,0"),
        ("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif", "0,0,1"),
        ("examples/CoNb3S6_tripleQ.mcif", ""),
    ],
)
def test_find_spin_group_exposes_convention_spin_only_direction(path, expected_direction):
    result = find_spin_group(path)
    assert result.convention_spin_only_direction == expected_direction


def test_find_spin_group_exposes_convention_spin_only_direction_cartesian():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.977_NdPdIn.mcif")
    nonzero_moments = [
        np.asarray(moment, dtype=float)
        for moment in result.convention_cell_detail["moments"]
        if np.linalg.norm(moment) > 1e-8
    ]
    expected_direction = nonzero_moments[0] / np.linalg.norm(nonzero_moments[0])

    assert result.convention_spin_only_direction_cartesian == "-0.383837,0.664825,0.640841"
    assert np.allclose(
        np.fromstring(result.convention_spin_only_direction_cartesian, sep=","),
        expected_direction,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("path", "expect_identity_rotation", "expect_changed"),
    [
        ("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif", True, False),
        ("examples/0.800_MnTe.mcif", True, False),
        ("tests/testset/mcif_241130_no2186/1.317_La0.25Pr0.75Co2P2.mcif", True, False),
    ],
)
def test_poscar_spin_frame_projection_behaves_consistently_across_representative_cases(
    path,
    expect_identity_rotation,
    expect_changed,
):
    result = find_spin_group(path)

    forward = np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float)
    projected = SpinSpaceGroup(result.acc_primitive_ssg_ops).transform_spin(forward).spin_polarizations

    assert np.allclose(forward, np.eye(3), atol=1e-8) is expect_identity_rotation
    assert result.spin_polarizations == projected
    assert projected == result.spin_polarizations_acc_poscar_spin_frame
    assert (projected != result.spin_polarizations_acc_cartesian) is expect_changed


def test_find_spin_group_exposes_msg_little_groups_and_wp_chain():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.26_TmAgGe.mcif")

    assert result.primitive_msg_ops
    assert result.msg_little_group_symbols
    assert result.msg_spin_polarizations
    assert result.msg_spin_polarizations_acc_poscar_spin_frame is not None
    assert (
        result.msg_spin_polarizations_acc_poscar_spin_frame_setting
        == "acc_primitive_poscar_spin_frame"
    )
    assert len(result.msg_little_group_symbols) == len(result.spin_polarizations)
    assert len(result.msg_spin_polarizations) == len(result.spin_polarizations)
    assert len(result.ssg_little_group_ops) == len(result.spin_polarizations)
    assert len(result.ssg_little_group_seitz_latex) == len(result.ssg_little_group_ops)
    assert len(result.msg_little_group_ops) == len(result.spin_polarizations)
    assert len(result.msg_little_group_seitz_latex) == len(result.msg_little_group_ops)
    for ops, seitz_latex in zip(result.ssg_little_group_ops, result.ssg_little_group_seitz_latex):
        assert len(ops) == len(seitz_latex)
        assert all({"index", "spin_rotation", "real_rotation", "translation"} <= set(op) for op in ops)
    for ops, seitz_latex in zip(result.msg_little_group_ops, result.msg_little_group_seitz_latex):
        assert len(ops) == len(seitz_latex)
        assert all({"index", "time_reversal", "real_rotation", "translation"} <= set(op) for op in ops)
    assert result.wp_chain
    assert all(len(row) == 7 for row in result.wp_chain)
    assert [row[0] for row in result.wp_chain] == ["Ag", "Ag", "Ge", "Ge", "Tm", "Tm"]
    assert result.g0_standard_cell["elements"] == (
        ["Ag"] * 6 + ["Ge"] * 6 + ["Tm"] * 6
    )
    assert [row[0] for row in result.acc_primitive_wp_chain] == [
        "Ag",
        "Ag",
        "Ge",
        "Ge",
        "Tm",
        "Tm",
    ]
    assert result.acc_primitive_magnetic_cell_detail["elements"] == (
        ["Ag"] * 3 + ["Ge"] * 3 + ["Tm"] * 3
    )


def test_wp_chain_uses_crystallographic_orbits_for_sg_multiplicity():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.197_Fe4Si2Sn7O16.mcif")

    fe_rows = [row for row in result.wp_chain if row[0] == "Fe"]
    assert [row[1] for row in fe_rows].count("12f") == 2
    assert "4f" not in {row[1] for row in fe_rows}
    assert "8f" not in {row[1] for row in fe_rows}
    assert {row[3] for row in fe_rows if row[1] == "12f"} == {"4d(0)", "8f(2)"}
    assert {row[5] for row in fe_rows if row[1] == "12f"} == {"4d(0)", "8f(3)"}
    assert {row[3] for row in fe_rows if row[1] == "4a"} == {"4a"}

    acc_fe_rows = [row for row in result.acc_primitive_wp_chain if row[0] == "Fe"]
    assert {row[3] for row in acc_fe_rows if row[1] == "6f"} == {"2d(0)", "4f(2)"}
    assert {row[5] for row in acc_fe_rows if row[1] == "6f"} == {"2d(0)", "4f(3)"}
    assert {row[3] for row in acc_fe_rows if row[1] == "2a"} == {"2a"}
    assert result.magnetic_site_summary["status"] == "ok"
    assert result.magnetic_site_summary["setting"] == "acc_primitive"
    assert result.magnetic_site_summary["cell_expansion"] == 2
    assert result.magnetic_site_summary["magnetic_atom_count"] == 6
    assert result.magnetic_site_summary["nonzero_moment_atom_count"] == 4
    assert result.magnetic_site_summary["zero_moment_magnetic_atom_count"] == 2
    assert result.magnetic_site_summary["zero_moment_magnetic_atom_indices"] == [6, 7]
    assert result.magnetic_site_summary["n_magnetic_orbits_sg"] == 1
    assert result.magnetic_site_summary["n_magnetic_orbits_ssg"] == 2
    assert result.magnetic_site_summary["n_magnetic_orbits_msg"] == 2
    assert result.magnetic_site_summary["max_magnetic_site_dof_ssg"] == 2
    assert result.magnetic_site_summary["max_magnetic_site_dof_msg"] == 3
    assert result.magnetic_site_summary["total_magnetic_site_dof_ssg"] == 2
    assert result.magnetic_site_summary["total_magnetic_site_dof_msg"] == 3
    magnetic_wp_rows = result.magnetic_site_summary["magnetic_wp_dof_rows"]
    assert magnetic_wp_rows
    assert {row["ssg_site_dof"] for row in magnetic_wp_rows} == {0, 2}
    assert {row["msg_site_dof"] for row in magnetic_wp_rows} == {0, 3}
    assert {
        row["ssg_wyckoff_with_dof"] for row in magnetic_wp_rows
    } == {"2d(0)", "4f(2)"}
    assert {
        row["msg_wyckoff_with_dof"] for row in magnetic_wp_rows
    } == {"2d(0)", "4f(3)"}

    for scif_text in result.scif_outputs.values():
        metadata = parse_scif_metadata(source_text=scif_text)
        atom_labels = set(metadata["raw_scif_tags"]["_atom_site_label"])
        spin_moments = metadata["atom_site_spin_moment"]
        spin_labels = spin_moments["label"]
        assert set(spin_labels) <= atom_labels
        assert spin_labels == ["Fe1", "Fe3"]
        zero_rows = [
            index
            for index, components in enumerate(
                zip(
                    spin_moments["axis_u"],
                    spin_moments["axis_v"],
                    spin_moments["axis_w"],
                )
            )
            if np.linalg.norm(np.asarray(components, dtype=float)) < 1e-12
        ]
        assert len(zero_rows) == 1
        zero_index = zero_rows[0]
        assert spin_labels[zero_index].startswith("Fe")
        assert spin_moments["symmform_uvw"][zero_index] == "0,0,0"
        assert spin_moments["symmform_rel_uvw"][zero_index] == "0,0,0"
        assert float(spin_moments["magnitude"][zero_index]) == pytest.approx(0.0)


def test_scif_includes_zero_moment_sites_from_magnetic_parent_sg_orbit():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.353_SmNiO3.mcif")

    assert result.magnetic_site_summary["zero_moment_magnetic_atom_count"] == 4
    for cell_mode, scif_text in result.scif_outputs.items():
        metadata = parse_scif_metadata(source_text=scif_text)
        atom_labels = set(metadata["raw_scif_tags"]["_atom_site_label"])
        spin_moments = metadata["atom_site_spin_moment"]
        spin_labels = spin_moments["label"]
        assert set(spin_labels) <= atom_labels, cell_mode
        assert spin_labels == ["Ni1", "Sm1", "Sm2"], cell_mode
        assert len(spin_labels) == 3, cell_mode

        zero_rows = [
            index
            for index, components in enumerate(
                zip(
                    spin_moments["axis_u"],
                    spin_moments["axis_v"],
                    spin_moments["axis_w"],
                )
            )
            if np.linalg.norm(np.asarray(components, dtype=float)) < 1e-12
        ]
        assert len(zero_rows) == 1, cell_mode
        zero_index = zero_rows[0]
        assert spin_labels[zero_index].startswith("Sm"), cell_mode
        assert spin_moments["symmform_uvw"][zero_index] == "0,0,0", cell_mode
        assert spin_moments["symmform_rel_uvw"][zero_index] == "0,0,0", cell_mode
        assert float(spin_moments["magnitude"][zero_index]) == pytest.approx(0.0)

    lattice, positions, elements, occupancies, _labels, moments = parse_scif_text(result.scif)
    roundtrip = find_spin_group_from_data(
        "1.353_SmNiO3.scif",
        lattice,
        positions,
        elements,
        occupancies,
        moments,
    )
    assert roundtrip.index == result.index
    assert roundtrip.conf == result.conf
    assert roundtrip.magnetic_site_summary["zero_moment_magnetic_atom_count"] == 4


def test_magnetic_site_selection_expands_to_parent_sg_orbit():
    class Dataset:
        crystallographic_orbits = [0, 0, 1, 1, 1, 2]

    expanded, audit = _expand_magnetic_indices_by_sg_orbit(
        Dataset(),
        magnetic_indices=[1, 4],
        site_count=6,
    )

    assert expanded == [0, 1, 2, 3, 4]
    assert audit == {
        "mode": "sg_orbit_closure_of_nonzero_moment_sites",
        "source_nonzero_moment_indices": [1, 4],
        "included_zero_moment_indices": [0, 2, 3],
        "parent_sg_orbit_labels": [0, 1],
    }


def test_magnetic_site_summary_handles_close_same_element_sites():
    expected = {
        "tests/testset/mcif_241130_no2186/0.1002_SrZn2Fe16O27.mcif": {
            "index": "194.2.1.2",
            "orbits": 7,
            "total_dof": 9,
        },
        "tests/testset/mcif_241130_no2186/0.1003_SrCo2Fe16O27.mcif": {
            "index": "63.2.1.10",
            "orbits": 17,
            "total_dof": 38,
        },
    }

    for path, case_expected in expected.items():
        result = find_spin_group(path)
        summary = result.magnetic_site_summary

        assert result.index == case_expected["index"]
        assert summary["status"] == "ok"
        assert summary["n_magnetic_orbits_ssg"] == case_expected["orbits"]
        assert summary["n_magnetic_orbits_msg"] == case_expected["orbits"]
        assert summary["total_magnetic_site_dof_ssg"] == case_expected["total_dof"]
        assert summary["total_magnetic_site_dof_msg"] == case_expected["total_dof"]
        assert summary["magnetic_wp_dof_rows"]


def test_spin_space_group_exposes_class_level_msg_ops_for_0200_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")

    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)
    msg_ops = public_ossg.msg_ops

    assert msg_ops
    assert _serialize_ssg_ops(msg_ops) == _serialize_ssg_ops(public_ossg.magnetic_space_group_ops)
    assert all(op.is_magnetic_space_group_operation(public_ossg.tol) for op in msg_ops)
    assert _serialize_msg_ops(_primitive_msg_ops_from_ssg(public_ossg.ops, tol=public_ossg.tol)) == (
        _serialize_msg_ops(_primitive_msg_ops_from_ssg(msg_ops, tol=public_ossg.tol))
    )


def test_spin_space_group_msg_info_is_lazy_and_cached(monkeypatch):
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    public_ossg = SpinSpaceGroup(result.convention_ssg_ops)

    captured = {"count": 0}
    original = group_module.get_magnetic_space_group_from_operations

    def wrapped(operations):
        captured["count"] += 1
        return original(operations)

    monkeypatch.setattr(group_module, "get_magnetic_space_group_from_operations", wrapped)

    assert captured["count"] == 0
    assert "msg_info" not in public_ossg.__dict__

    first = public_ossg.msg_info

    assert captured["count"] == 1
    assert public_ossg.magnetic_space_group_info == first
    assert public_ossg.msg_int_num == 562
    assert public_ossg.msg_bns_num == "63.464"
    assert public_ossg.msg_bns_symbol == "Cm'cm'"
    assert public_ossg.msg_og_num == "63.8.518"
    assert public_ossg.msg_og_symbol == "Cm'cm'"
    assert public_ossg.msg_type == 3
    assert public_ossg.mpg_num == "8.4.27"
    assert public_ossg.mpg_symbol == "m'mm'"

    second = public_ossg.msg_info

    assert captured["count"] == 1
    assert first == second


def test_magnetic_time_reversal_uses_axial_vector_det_rule():
    improper_rotation = np.diag([1.0, -1.0, 1.0])
    op_without_time_reversal = group_module.SpinSpaceGroupOperation(
        -improper_rotation,
        improper_rotation,
        np.zeros(3),
    )
    op_with_time_reversal = group_module.SpinSpaceGroupOperation(
        improper_rotation,
        improper_rotation,
        np.zeros(3),
    )

    assert op_without_time_reversal.magnetic_time_reversal() == 1
    assert op_with_time_reversal.magnetic_time_reversal() == -1


def test_in_lattice_spin_frame_is_not_the_same_as_oriented_abc_for_0200_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )

    in_lattice = _spin_transform_to_in_lattice(acc_cell)
    oriented_abc = _spin_transform_to_oriented_abc(acc_cell)

    assert not np.allclose(in_lattice, oriented_abc, atol=1e-8)


def test_acc_primitive_msg_ops_are_derived_from_acc_primitive_ossg_for_0200_mn3sn():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.200_Mn3Sn.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_ssg = SpinSpaceGroup(result.acc_primitive_ssg_ops)
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(acc_ssg, acc_cell)

    assert _serialize_msg_ops(result.acc_primitive_msg_ops) == _serialize_msg_ops(
        _primitive_msg_ops_from_ssg(acc_primitive_ossg.msg_ops, tol=acc_primitive_ossg.tol)
    )
    assert result.acc_primitive_msg_ops_spin_frame_setting == "ossg_oriented_spin_frame"


def test_acc_primitive_ossg_reconstructs_collinear_msg_for_0712_vnb3s6():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.712_VNb3S6.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert result.acc_primitive_msg_ops_spin_frame_setting == "ossg_oriented_spin_frame"
    assert acc_primitive_ossg.msg_int_num == 131
    assert acc_primitive_ossg.msg_bns_num == "20.33"
    assert acc_primitive_ossg.msg_bns_symbol == "C2'2'2_1"


def test_acc_primitive_ossg_reconstructs_hex_collinear_msg_for_0800_mnte():
    result = find_spin_group("examples/0.800_MnTe.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert result.acc_primitive_msg_ops_spin_frame_setting == "ossg_oriented_spin_frame"
    assert acc_primitive_ossg.msg_int_num == 555
    assert acc_primitive_ossg.msg_bns_num == "63.457"
    assert acc_primitive_ossg.msg_bns_symbol == "Cmcm"


@pytest.mark.parametrize(
    ("path", "expected_bns_num", "expected_bns_symbol"),
    [
        ("tests/testset/mcif_241130_no2186/0.691_CaCo1.86As2.mcif", "126.386", "P_I4/nnc"),
        ("tests/testset/mcif_241130_no2186/0.454_PrScSb.mcif", "128.410", "P_I4/mnc"),
    ],
)
def test_acc_primitive_ossg_reconstructs_tetragonal_type4_collinear_msg_representatives(
    path,
    expected_bns_num,
    expected_bns_symbol,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.gspg.spin_only_symbol_s == "D∞h"
    assert acc_primitive_ossg.msg_bns_num == expected_bns_num
    assert acc_primitive_ossg.msg_bns_symbol == expected_bns_symbol


@pytest.mark.parametrize(
    ("path", "expected_bns_num", "expected_bns_symbol"),
    [
        ("tests/testset/mcif_241130_no2186/0.1073_Cr2CoAl.mcif", "119.319", "I-4m'2'"),
        ("tests/testset/mcif_241130_no2186/0.229_Ba2MnSi2O7.mcif", "113.267", "P-42_1m"),
        ("tests/testset/mcif_241130_no2186/0.802_CuFeS2.mcif", "122.333", "I-42d"),
        ("tests/testset/mcif_241130_no2186/0.826_MnTeLi0.003.mcif", "12.62", "C2'/m'"),
        ("tests/testset/mcif_241130_no2186/1.188_CeRh2Si2.mcif", "64.480", "C_Amca"),
        ("tests/testset/mcif_241130_no2186/1.33_ErAuGe.mcif", "33.154", "P_Cna2_1"),
        ("tests/testset/mcif_241130_no2186/0.19_MnTiO3.mcif", "148.19", "R-3'"),
        ("tests/testset/mcif_241130_no2186/0.1001_PbMn2Ni6Te3O18.mcif", "176.146", "P6_3/m'"),
        ("tests/testset/mcif_241130_no2186/0.35_Cu2OSeO3.mcif", "146.10", "R3"),
    ],
)
def test_acc_primitive_ossg_reconstructs_high_order_collinear_msg_representatives(
    path,
    expected_bns_num,
    expected_bns_symbol,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.msg_bns_num == expected_bns_num
    assert acc_primitive_ossg.msg_bns_symbol == expected_bns_symbol


@pytest.mark.parametrize(
    ("path", "expected_msg_int_num", "expected_msg_type"),
    [
        ("tests/testset/mcif_241130_no2186/1.0.57_NdAlGe.mcif", 325, 3),
        ("tests/testset/mcif_241130_no2186/1.646_Na2Ni2TeO6.mcif", 332, 4),
        ("tests/testset/mcif_241130_no2186/1.738_TbNiAl.mcif", 345, 4),
    ],
)
def test_metric_aware_collinear_geometry_recovers_last_residual_msg_numbers(
    path,
    expected_msg_int_num,
    expected_msg_type,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.real_space_metric is not None
    assert acc_primitive_ossg.msg_int_num == expected_msg_int_num
    assert acc_primitive_ossg.msg_type == expected_msg_type


@pytest.mark.parametrize(
    ("path", "expected_order"),
    [
        ("tests/testset/mcif_241130_no2186/0.1073_Cr2CoAl.mcif", 4),
        ("tests/testset/mcif_241130_no2186/0.229_Ba2MnSi2O7.mcif", 4),
        ("tests/testset/mcif_241130_no2186/0.802_CuFeS2.mcif", 4),
        ("tests/testset/mcif_241130_no2186/0.826_MnTeLi0.003.mcif", 2),
        ("tests/testset/mcif_241130_no2186/1.188_CeRh2Si2.mcif", 2),
        ("tests/testset/mcif_241130_no2186/1.33_ErAuGe.mcif", 2),
    ],
)
def test_collinear_spin_promotion_order_uses_effective_proper_rotations(
    path,
    expected_order,
):
    result = find_spin_group(path)
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Collinear"
    assert acc_primitive_ossg.collinear_spin_promotion_order == expected_order


def test_acc_primitive_ossg_recovers_type4_noncollinear_msg_for_1412_au72al14tb14():
    result = find_spin_group("tests/testset/mcif_241130_no2186/1.412_Au72Al14Tb14.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    assert result.conf == "Noncoplanar"
    assert acc_primitive_ossg.msg_bns_num == "201.21"
    assert acc_primitive_ossg.msg_bns_symbol == "P_In-3"
    assert acc_primitive_ossg.msg_type == 4


def test_high_order_effective_axis_stays_aligned_with_collinear_axis_for_01073_cr2coal():
    result = find_spin_group("tests/testset/mcif_241130_no2186/0.1073_Cr2CoAl.mcif")
    acc_cell = CrystalCell(
        result.acc_primitive_magnetic_cell_detail["lattice"],
        result.acc_primitive_magnetic_cell_detail["positions"],
        result.acc_primitive_magnetic_cell_detail["occupancies"],
        result.acc_primitive_magnetic_cell_detail["elements"],
        result.acc_primitive_magnetic_cell_detail["moments"],
        spin_setting="in_lattice",
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(SpinSpaceGroup(result.acc_primitive_ssg_ops), acc_cell)

    axes = []
    for op in acc_primitive_ossg.ops:
        order = _rotation_order(np.asarray(op[1], dtype=float), tol=1e-6)
        if order != 4:
            continue
        axis = _effective_proper_axis_from_space_rotation(np.asarray(op[1], dtype=float), tol=1e-4)
        if axis is None:
            continue
        if np.allclose(axis, acc_primitive_ossg.collinear_axis, atol=1e-4) or np.allclose(
            axis, -np.asarray(acc_primitive_ossg.collinear_axis), atol=1e-4
        ):
            axes.append(axis)

    assert result.conf == "Collinear"
    expected_axis = np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0], dtype=float)
    assert np.allclose(acc_primitive_ossg.collinear_axis, expected_axis, atol=1e-4) or np.allclose(
        acc_primitive_ossg.collinear_axis,
        -expected_axis,
        atol=1e-4,
    )
    assert axes


@pytest.mark.parametrize(
    ("path", "expect_identity_rotation", "expect_changed"),
    [
        ("tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif", True, False),
        ("tests/testset/mcif_241130_no2186/0.13_Ca3Co2-xMnxO6.mcif", True, False),
        ("examples/0.800_MnTe.mcif", True, False),
    ],
)
def test_msg_spin_polarizations_poscar_projection_behaves_consistently_across_representative_cases(
    path,
    expect_identity_rotation,
    expect_changed,
):
    result = find_spin_group(path)

    forward = np.asarray(result.acc_primitive_real_cartesian_to_poscar_spin_frame, dtype=float)

    assert np.allclose(forward, np.eye(3), atol=1e-8) is expect_identity_rotation
    assert result.msg_spin_polarizations == result.msg_spin_polarizations_acc_poscar_spin_frame
    assert (
        result.msg_spin_polarizations_acc_poscar_spin_frame
        != result.msg_spin_polarizations_acc_cartesian
    ) is expect_changed

    encoded = json.dumps(result.to_dict(), default=str)
    assert '"msg_spin_polarizations_acc_poscar_spin_frame"' in encoded
    assert '"gspg_symbol_linear"' in encoded
    assert '"gspg_ops_xyz_uvw"' in encoded
    assert '"gspg_raw_ops_xyz_uvw"' in encoded
    assert '"gspg_spin_only_ops_xyz_uvw"' in encoded
    assert '"gspg_text"' in encoded
    assert '"gspg_collinear_axis"' in encoded


def test_space_tolerance_site_collapse_reports_semantic_error():
    magnetic_cell = (
        np.eye(3),
        [np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])],
        [1, 1],
        [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])],
    )

    with pytest.raises(SpaceToleranceDegeneracyError, match="space_tol"):
        change_cell_settings(
            magnetic_cell,
            np.eye(3),
            np.zeros(3),
            eps=0.1,
            moment_eps=0.02,
        )


def test_change_cell_settings_uses_unimodular_fast_path_without_supercell_enumeration(monkeypatch):
    def _fail_generic_path(*args, **kwargs):
        raise AssertionError("generic supercell enumeration should not run")

    monkeypatch.setattr(cell_module, "find_cell_border", _fail_generic_path)
    lattice = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.2, 3.0, 0.0],
            [0.1, 0.4, 4.0],
        ]
    )
    positions = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
    ]
    atom_types = [2, 1]
    moments = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    transformation = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ],
        dtype=float,
    )
    origin_shift = np.array([0.25, -0.125, 0.5])

    new_lattice, new_positions, new_types, new_moments = change_cell_settings(
        (lattice, positions, atom_types, moments),
        transformation,
        origin_shift,
        eps=1e-6,
    )

    expected_lattice = np.linalg.inv(transformation).T @ lattice
    expected_order = [1, 0]
    expected_positions = np.mod(np.asarray(positions) @ transformation.T + origin_shift, 1.0)[expected_order]
    assert np.allclose(new_lattice, expected_lattice, atol=1e-8)
    assert np.allclose(np.asarray(new_positions), expected_positions, atol=1e-8)
    assert new_types == [atom_types[index] for index in expected_order]
    assert all(
        np.allclose(actual, moments[index], atol=1e-8)
        for actual, index in zip(new_moments, expected_order)
    )


def test_scif_spin_only_direction_requires_single_vector():
    with pytest.raises(ValueError, match="single 3-vector"):
        write_scif_spin_only("Coplanar", np.eye(3))


def test_spin_only_inversion_has_no_unique_coplanar_direction():
    ops = [
        [np.eye(3), np.eye(3), np.zeros(3)],
        [-np.eye(3), np.eye(3), np.zeros(3)],
    ]

    with pytest.raises(SpaceToleranceDegeneracyError, match="non-unique coplanar"):
        _ = SpinSpaceGroup(ops).conf
