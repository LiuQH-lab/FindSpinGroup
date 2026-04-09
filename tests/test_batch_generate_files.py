import json
from pathlib import Path

import pytest
import numpy as np

from findspingroup import find_spin_group, find_spin_group_from_data
from findspingroup import batch_mcif
from findspingroup import batch_poscar_roundtrip
from findspingroup import batch_scif_roundtrip
from findspingroup.batch_mcif import run_mcif_batch, run_mcif_batch_with_auto_baseline
from findspingroup.batch_poscar_roundtrip import run_poscar_roundtrip_batch
from findspingroup.batch_scif_roundtrip import run_scif_roundtrip_batch
from findspingroup.io import parse_scif_file, parse_scif_text
from findspingroup.version import __version__


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "batch_smoke_manifest.txt"
BASELINE_PATH = PROJECT_ROOT / "tests" / "baselines" / "mcif_batch_smoke_baseline.json"


def _load_manifest_entries() -> list[str]:
    entries = []
    for raw_line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _baseline_cases() -> dict[str, dict]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


MANIFEST_ENTRIES = _load_manifest_entries()


class _FakeResult:
    def __init__(self, summary: dict, tensor_outputs: dict | None = None):
        self._summary = summary
        self.tensor_outputs = tensor_outputs or {}
        self.magnetic_phase = summary.get("phase", "phase:fake")

    def to_summary_dict(self) -> dict:
        return dict(self._summary)

    def to_dict(self) -> dict:
        return dict(self._summary)

    def properties_summary(self) -> dict:
        return dict(self._summary.get("properties", {}))


class _FakeRoundtripBatchResult:
    def __init__(
        self,
        *,
        index: str,
        conf: str,
        occupancies: list[float] | None = None,
        scif: str = "data_test\n",
        acc_primitive_magnetic_cell_poscar: str = "fake poscar\n",
        g0_standard_cell: dict | None = None,
        T_convention_to_acc_primitive=None,
    ):
        self.index = index
        self.conf = conf
        self.scif = scif
        self.acc_primitive_magnetic_cell_poscar = acc_primitive_magnetic_cell_poscar
        self.g0_standard_cell = g0_standard_cell
        self.T_convention_to_acc_primitive = T_convention_to_acc_primitive
        self.acc_primitive_magnetic_cell_detail = {
            "occupancies": list(occupancies or []),
        }


def _fake_summary(index: str) -> dict:
    return {
        "index": index,
        "conf": f"conf:{index}",
        "phase": f"phase:{index}",
        "acc": f"acc:{index}",
        "properties": {"ss_w_soc": f"prop:{index}"},
        "gspg": {
            "output_mode": f"mode:{index}",
            "effective_mpg_symbol": f"empg:{index}",
            "symbol_mode": f"symbol_mode:{index}",
            "npg_symbol_s": f"npg:{index}",
            "spin_only_component_symbol_s": f"ss:{index}",
            "tentative_symbol_s": f"tentative:{index}",
        },
    }


def _fake_find_spin_group_factory(
    *,
    ok_by_name: dict[str, dict],
    errors_by_name: dict[str, Exception] | None = None,
    tensor_by_name: dict[str, dict] | None = None,
):
    errors_by_name = errors_by_name or {}
    tensor_by_name = tensor_by_name or {}

    def fake_find_spin_group(path: str, **kwargs):
        file_name = Path(path).name
        if file_name in errors_by_name:
            raise errors_by_name[file_name]
        if file_name not in ok_by_name:
            raise AssertionError(f"Unexpected fake case: {file_name}")
        return _FakeResult(ok_by_name[file_name], tensor_by_name.get(file_name))

    return fake_find_spin_group


def _fake_tensor_summary(tag: str) -> dict:
    return {
        "AHE_woSOC": {"free_parameters": 0, "is_zero": True, "relations": []},
        "AHE_wSOC": {"free_parameters": 1, "is_zero": False, "relations": [tag]},
    }


def _roundtrip_index_from_scif_text(tmp_path: Path, file_name: str, scif_text: str):
    scif_path = tmp_path / file_name
    scif_path.write_text(scif_text, encoding="utf-8")
    lattice_factors, positions, elements, occupancies, labels, moments = parse_scif_file(scif_path)
    return find_spin_group_from_data(
        str(scif_path),
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )


def test_parse_scif_text_matches_file_based_parser(tmp_path):
    original = find_spin_group(str((PROJECT_ROOT / MANIFEST_ENTRIES[0]).resolve()))
    scif_path = tmp_path / "sample.scif"
    scif_path.write_text(original.scif, encoding="utf-8")

    from_file = parse_scif_file(scif_path)
    from_text = parse_scif_text(original.scif)

    assert len(from_file) == len(from_text) == 6
    assert np.allclose(from_file[0], from_text[0])
    assert all(a == b for a, b in zip(from_file[2], from_text[2]))
    assert all(a == b for a, b in zip(from_file[3], from_text[3]))
    assert all(a == b for a, b in zip(from_file[4], from_text[4]))
    assert all(np.allclose(a, b) for a, b in zip(from_file[5], from_text[5]))


def _write_fake_mcif(tmp_path: Path, file_name: str) -> Path:
    path = tmp_path / file_name
    path.write_text("# fake mcif\n", encoding="utf-8")
    return path


def test_discover_mcif_files_ignores_appledouble_sidecars(tmp_path):
    real_file = _write_fake_mcif(tmp_path, "real_case.mcif")
    _write_fake_mcif(tmp_path, "._real_case.mcif")

    nested = tmp_path / "nested"
    nested.mkdir()
    nested_real = _write_fake_mcif(nested, "nested_case.mcif")
    _write_fake_mcif(nested, "._nested_case.mcif")

    discovered = batch_mcif._discover_mcif_files([str(tmp_path)], recursive=True)

    assert discovered == [real_file.resolve(), nested_real.resolve()]


def test_batch_smoke_manifest_matches_baseline_keys():
    assert sorted(MANIFEST_ENTRIES) == sorted(_baseline_cases())


@pytest.mark.parametrize("relative_path", MANIFEST_ENTRIES)
def test_find_spin_group_mcif_smoke_baseline(relative_path):
    expected = _baseline_cases()[relative_path]
    source_path = (PROJECT_ROOT / relative_path).resolve()

    result = find_spin_group(str(source_path))

    assert expected["status"] == "ok"
    assert result.to_summary_dict() == expected["result"]


@pytest.mark.parametrize("relative_path", MANIFEST_ENTRIES)
def test_scif_roundtrip_smoke_manifest_preserves_index(tmp_path, relative_path):
    source_path = (PROJECT_ROOT / relative_path).resolve()
    original = find_spin_group(str(source_path))
    scif_text = original.scif

    roundtrip = _roundtrip_index_from_scif_text(
        tmp_path,
        f"{source_path.stem}.scif",
        scif_text,
    )

    assert roundtrip.index == original.index


def test_run_mcif_batch_matches_smoke_baseline(tmp_path):
    files = [(PROJECT_ROOT / relative_path).resolve() for relative_path in MANIFEST_ENTRIES]

    summary = run_mcif_batch(
        files,
        tmp_path,
        baseline_path=BASELINE_PATH,
        fail_on_mismatch=True,
        quiet=True,
    )

    assert summary["success_count"] == len(files)
    assert summary["error_count"] == 0
    assert summary["comparison"]["missing_in_baseline_count"] == 0
    assert summary["comparison"]["mismatch_count"] == 0
    assert summary["comparison"]["tensor_summary_backfill_count"] == 0
    assert summary["exit_code"] == 0


def test_run_scif_roundtrip_batch_smoke_manifest(tmp_path):
    files = [(PROJECT_ROOT / relative_path).resolve() for relative_path in MANIFEST_ENTRIES]

    summary = run_scif_roundtrip_batch(
        files,
        tmp_path,
        quiet=True,
    )

    assert summary["processed_cases"] == len(files)
    assert summary["success_count"] == len(files)
    assert summary["mismatch_count"] == 0
    assert summary["error_count"] == 0
    assert summary["exit_code"] == 0
    assert summary["save_scif"] is True
    assert summary["scif_output_dir"] == str(tmp_path / "scif")

    mismatches = json.loads((tmp_path / "mismatches.json").read_text(encoding="utf-8"))
    errors = json.loads((tmp_path / "errors_by_file.json").read_text(encoding="utf-8"))

    assert mismatches == []
    assert errors == {}
    assert (tmp_path / "scif" / "0.800_MnTe.scif").exists()


def test_run_scif_roundtrip_batch_marks_fractional_occupancy_mismatches(monkeypatch, tmp_path):
    file_path = _write_fake_mcif(tmp_path, "fractional_case.mcif")
    original = _FakeRoundtripBatchResult(
        index="142.88.1.2.P2",
        conf="Coplanar",
        occupancies=[0.08, 0.92, 1.0],
    )
    roundtrip = _FakeRoundtripBatchResult(
        index="134.141.2.2.P2",
        conf="Coplanar",
    )

    monkeypatch.setattr(batch_scif_roundtrip, "find_spin_group", lambda path, **kwargs: original)
    monkeypatch.setattr(
        batch_scif_roundtrip,
        "_roundtrip_from_scif_text",
        lambda **kwargs: roundtrip,
    )

    summary = run_scif_roundtrip_batch(
        [file_path],
        tmp_path / "scif_batch",
        quiet=True,
    )

    mismatches = json.loads((tmp_path / "scif_batch" / "mismatches.json").read_text(encoding="utf-8"))
    record = json.loads((tmp_path / "scif_batch" / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert summary["fractional_occupancy_case_count"] == 1
    assert summary["fractional_occupancy_mismatch_count"] == 1
    assert summary["fractional_occupancy_error_count"] == 0
    assert mismatches[0]["source_has_fractional_occupancy"] is True
    assert mismatches[0]["source_occupancy_values"] == [0.08, 0.92, 1.0]
    assert mismatches[0]["source_fractional_occupancy_values"] == [0.08, 0.92]
    assert mismatches[0]["source_fractional_occupancy_site_count"] == 2
    assert record["source_has_fractional_occupancy"] is True


def test_run_poscar_roundtrip_batch_marks_fractional_occupancy_errors(monkeypatch, tmp_path):
    file_path = _write_fake_mcif(tmp_path, "fractional_case.mcif")
    original = _FakeRoundtripBatchResult(
        index="142.88.1.2.P2",
        conf="Coplanar",
        occupancies=[0.08, 0.92, 1.0],
        acc_primitive_magnetic_cell_poscar="fake poscar\n",
    )

    monkeypatch.setattr(batch_poscar_roundtrip, "find_spin_group", lambda path, **kwargs: original)

    def _raise_roundtrip(**kwargs):
        raise RuntimeError("synthetic poscar roundtrip failure")

    monkeypatch.setattr(batch_poscar_roundtrip, "_roundtrip_from_poscar_text", _raise_roundtrip)

    summary = run_poscar_roundtrip_batch(
        [file_path],
        tmp_path / "poscar_batch",
        quiet=True,
    )

    errors = json.loads((tmp_path / "poscar_batch" / "errors_by_file.json").read_text(encoding="utf-8"))
    record = json.loads((tmp_path / "poscar_batch" / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    case_id = batch_mcif._normalize_case_id(file_path)

    assert summary["fractional_occupancy_case_count"] == 1
    assert summary["fractional_occupancy_mismatch_count"] == 0
    assert summary["fractional_occupancy_error_count"] == 1
    assert errors[case_id]["source_has_fractional_occupancy"] is True
    assert errors[case_id]["source_occupancy_values"] == [0.08, 0.92, 1.0]
    assert errors[case_id]["source_fractional_occupancy_values"] == [0.08, 0.92]
    assert errors[case_id]["source_fractional_occupancy_site_count"] == 2
    assert record["source_has_fractional_occupancy"] is True


def test_run_poscar_roundtrip_batch_supports_g0_cptrans_candidate_source_mode(monkeypatch, tmp_path):
    file_path = _write_fake_mcif(tmp_path, "g0_candidate_case.mcif")
    original = _FakeRoundtripBatchResult(
        index="148.2.4.1",
        conf="Noncoplanar",
        occupancies=[1.0],
        g0_standard_cell={
            "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "positions": [[0.0, 0.0, 0.0]],
            "occupancies": [1.0],
            "elements": ["Fe"],
            "moments": [[1.0, 0.0, 0.0]],
            "type_ids": [1],
        },
        T_convention_to_acc_primitive=(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        ),
    )

    monkeypatch.setattr(batch_poscar_roundtrip, "find_spin_group", lambda path, **kwargs: original)
    captured = {}

    def _roundtrip(**kwargs):
        captured.update(kwargs)
        return original

    monkeypatch.setattr(batch_poscar_roundtrip, "_roundtrip_from_poscar_text", _roundtrip)

    summary = run_poscar_roundtrip_batch(
        [file_path],
        tmp_path / "poscar_batch",
        source_mode=batch_poscar_roundtrip.POSCAR_SOURCE_G0_CPTRANS_CANDIDATE,
        quiet=True,
    )

    assert summary["success_count"] == 1
    assert summary["input_format"] == "repo_generated_g0_cptrans_candidate_magnetic_poscar"
    assert summary["source_mode"] == batch_poscar_roundtrip.POSCAR_SOURCE_G0_CPTRANS_CANDIDATE
    assert captured["source_name"].endswith("g0_candidate_case.mcif::g0_cptrans_candidate_magnetic.POSCAR")
    assert "# MAGMOM=" in captured["poscar_text"]


def test_run_mcif_batch_writes_baseline_meta_and_export_txt(tmp_path):
    files = [(PROJECT_ROOT / MANIFEST_ENTRIES[0]).resolve()]

    summary = run_mcif_batch(
        files,
        tmp_path,
        export_fields=["index", "properties.ss_w_soc"],
        export_txt_path=tmp_path / "selected.txt",
        quiet=True,
    )

    export_lines = (tmp_path / "selected.txt").read_text(encoding="utf-8").splitlines()

    assert summary["success_count"] == 1
    assert summary["tolerances"] == {
        "space_tol": 0.02,
        "mtol": 0.02,
        "meigtol": 0.00002,
        "matrix_tol": 0.01,
    }
    assert summary["started_at"]
    assert summary["finished_at"]
    assert summary["package_version"] == __version__
    assert not (tmp_path / "baseline.meta.json").exists()
    assert export_lines == [
        '0.800_MnTe.mcif: {"index": "194.164.1.1.L", "properties.ss_w_soc": "Yes"}'
    ]

    summary_record = json.loads((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    full_record = json.loads((tmp_path / "full_results.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert "group_identifiers" in summary_record
    assert summary_record["group_identifiers"]["msg_bns_number"] == "63.457"
    assert summary_record["group_identifiers"]["msg_og_number"] == "63.1.511"
    assert "sg_has_real_space_inversion" in summary_record["group_identifiers"]
    assert "ossg_has_real_space_inversion" in summary_record["group_identifiers"]
    assert "msg_has_real_space_inversion" in summary_record["group_identifiers"]
    assert full_record["result"]["msg_bns_number"] == "63.457"
    assert full_record["result"]["msg_og_number"] == "63.1.511"
    assert summary["full_results_jsonl"].endswith("full_results.jsonl")


def test_run_mcif_batch_export_tolerates_missing_identify_index_details(monkeypatch, tmp_path):
    source_file = _write_fake_mcif(tmp_path, "missing_identify.mcif")
    summary_payload = _fake_summary("fake.index")
    summary_payload["identify_index_details"] = None
    summary_payload["spin_part_point_group"] = "3m"

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(ok_by_name={"missing_identify.mcif": summary_payload}),
    )

    summary = run_mcif_batch(
        [source_file],
        tmp_path / "batch",
        export_fields=[
            "index",
            "identify_index_details.G0_id",
            "identify_index_details.L0_id",
            "spin_part_point_group",
        ],
        export_txt_path=tmp_path / "batch" / "selected.txt",
        quiet=True,
    )

    export_lines = (tmp_path / "batch" / "selected.txt").read_text(encoding="utf-8").splitlines()

    assert summary["success_count"] == 1
    assert summary["error_count"] == 0
    assert export_lines == [
        'missing_identify.mcif: {"identify_index_details.G0_id": null, "identify_index_details.L0_id": null, "index": "fake.index", "spin_part_point_group": "3m"}'
    ]


def test_run_mcif_batch_does_not_count_export_failures_as_success(monkeypatch, tmp_path):
    source_file = _write_fake_mcif(tmp_path, "export_fail.mcif")

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(ok_by_name={"export_fail.mcif": _fake_summary("fake.index")}),
    )

    def raise_export_error(result, selectors):
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr(batch_mcif, "_build_export_content", raise_export_error)

    summary = run_mcif_batch(
        [source_file],
        tmp_path / "batch",
        export_fields=["index"],
        export_txt_path=tmp_path / "batch" / "selected.txt",
        quiet=True,
    )

    case_id = batch_mcif._normalize_case_id(source_file)
    errors = json.loads((tmp_path / "batch" / "errors_by_file.json").read_text(encoding="utf-8"))
    records = (tmp_path / "batch" / "records.jsonl").read_text(encoding="utf-8").splitlines()

    assert summary["processed_cases"] == 1
    assert summary["success_count"] == 0
    assert summary["error_count"] == 1
    assert errors[case_id]["message"] == "synthetic export failure"
    assert json.loads(records[0])["status"] == "error"


def test_run_mcif_batch_exports_new_setting_alias_fields(tmp_path):
    files = [(PROJECT_ROOT / MANIFEST_ENTRIES[0]).resolve()]

    run_mcif_batch(
        files,
        tmp_path,
        export_fields=[
            "primitive_magnetic_cell_setting",
            "spin_polarizations_setting",
            "msg_spin_polarizations_setting",
            "spin_polarizations_acc_poscar_spin_frame_setting",
            "msg_spin_polarizations_acc_poscar_spin_frame_setting",
            "gspg_output_mode",
            "gspg_effective_mpg_symbol",
            "gspg_symbol_linear",
            "gspg_spin_only_symbol_s",
        ],
        export_txt_path=tmp_path / "settings.txt",
        quiet=True,
    )

    export_lines = (tmp_path / "settings.txt").read_text(encoding="utf-8").splitlines()

    assert len(export_lines) == 1
    file_name, payload = export_lines[0].split(": ", 1)
    assert file_name == "0.800_MnTe.mcif"
    assert json.loads(payload) == {
        "primitive_magnetic_cell_setting": "acc_primitive",
        "spin_polarizations_setting": "acc_primitive_poscar_spin_frame",
        "msg_spin_polarizations_setting": "acc_primitive_poscar_spin_frame",
        "spin_polarizations_acc_poscar_spin_frame_setting": "acc_primitive_poscar_spin_frame",
        "msg_spin_polarizations_acc_poscar_spin_frame_setting": "acc_primitive_poscar_spin_frame",
        "gspg_output_mode": "reduced_point_part_with_spin_only_annotation",
        "gspg_effective_mpg_symbol": "6/mmm1'",
        "gspg_symbol_linear": "-1|6_{3}/ -1|m 1|m -1|m ∞_{110}m|1",
        "gspg_spin_only_symbol_s": "C∞v",
    }


def test_run_mcif_batch_exports_convention_selected_fields(tmp_path):
    files = [(PROJECT_ROOT / "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif").resolve()]

    run_mcif_batch(
        files,
        tmp_path,
        export_fields=[
            "primitive_magnetic_cell_ssg_type",
            "convention_cell_setting",
            "convention_ssg_setting",
            "T_input_to_convention",
            "T_convention_to_acc_primitive",
        ],
        export_txt_path=tmp_path / "convention.txt",
        quiet=True,
    )

    export_lines = (tmp_path / "convention.txt").read_text(encoding="utf-8").splitlines()

    assert len(export_lines) == 1
    file_name, payload = export_lines[0].split(": ", 1)
    assert file_name == "1.325_PrMn2O5.mcif"
    decoded = json.loads(payload)
    assert decoded["primitive_magnetic_cell_ssg_type"] == "k"
    assert decoded["convention_cell_setting"] == "L0std"
    assert decoded["convention_ssg_setting"] == "L0std"
    assert len(decoded["T_input_to_convention"]) == 2
    assert len(decoded["T_convention_to_acc_primitive"]) == 2


@pytest.mark.parametrize(
    ("relative_path", "expected_gspg"),
    [
        (
            "tests/testset/mcif_241130_no2186/1.325_PrMn2O5.mcif",
            {
                "output_mode": "reduced_point_part_with_spin_only_annotation",
                "effective_mpg_symbol": "m1'",
                "symbol_linear": "1|m ∞_{001}/mm|1",
                "point_part_linear": "1|m",
                "spin_only_part_linear": "∞_{001}/mm|1",
                "real_space_setting": "L0std",
                "spin_frame_setting": "ossg_oriented_spin_frame",
                "symbol_mode": "point_part_and_spin_only",
                "npg_symbol_s": "Ci",
                "spin_only_component_symbol_s": "D∞h",
                "tentative_symbol_s": None,
            },
        ),
        (
            "tests/testset/mcif_241130_no2186/1.498_Cu6(SiO3)6(H2O)6.mcif",
            {
                "output_mode": "explicit_ops",
                "effective_mpg_symbol": "-31'",
                "symbol_linear": "3^{2}_{001}|-3 -1|1",
                "point_part_linear": "3^{2}_{001}|-3",
                "spin_only_part_linear": "-1|1",
                "real_space_setting": "G0std",
                "spin_frame_setting": "ossg_oriented_spin_frame",
                "symbol_mode": "point_part_and_spin_only",
                "npg_symbol_s": "S6",
                "spin_only_component_symbol_s": "Ci",
                "tentative_symbol_s": None,
            },
        ),
    ],
)
def test_find_spin_group_batch_summary_covers_gspg_fallback_cases(relative_path, expected_gspg):
    result = find_spin_group(str((PROJECT_ROOT / relative_path).resolve()))

    assert result.to_summary_dict()["gspg"] == expected_gspg


def test_auto_baseline_creates_then_reuses_matching_tolerance_baseline(tmp_path):
    files = [(PROJECT_ROOT / MANIFEST_ENTRIES[0]).resolve()]
    baseline_root = tmp_path / "baseline_store"

    first_summary = run_mcif_batch_with_auto_baseline(
        files,
        tmp_path / "run1",
        baseline_root=baseline_root,
        suite_name="standard_smoke",
        quiet=True,
    )
    first_meta = json.loads(
        (baseline_root / "standard_smoke" / "space_0.02__mtol_0.02__meigtol_2e-05__matrix_0.01" / "baseline.meta.json")
        .read_text(encoding="utf-8")
    )

    second_summary = run_mcif_batch_with_auto_baseline(
        files,
        tmp_path / "run2",
        baseline_root=baseline_root,
        suite_name="standard_smoke",
        quiet=True,
    )

    assert first_summary["auto_baseline"]["action"] == "created"
    assert first_summary["package_version"] == __version__
    assert first_meta["package_version"] == __version__
    assert first_meta["created_at"]
    assert first_meta["created_at_epoch"]
    assert first_meta["updated_at"] is None
    assert first_meta["updated_at_epoch"] is None
    assert first_meta["suite_name"] == "standard_smoke"
    assert first_meta["tolerances"] == {
        "space_tol": 0.02,
        "mtol": 0.02,
        "meigtol": 0.00002,
        "matrix_tol": 0.01,
    }
    assert second_summary["auto_baseline"]["action"] == "used_existing"
    assert second_summary["comparison"]["mismatch_count"] == 0
    assert second_summary["comparison"]["missing_in_baseline_count"] == 0
    assert second_summary["exit_code"] == 0


def test_auto_baseline_creates_even_when_first_run_has_errors(monkeypatch, tmp_path):
    file_a = _write_fake_mcif(tmp_path, "a.mcif")
    file_b = _write_fake_mcif(tmp_path, "b.mcif")
    baseline_root = tmp_path / "baseline_store"

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(
            ok_by_name={"a.mcif": _fake_summary("IDX_A")},
            errors_by_name={"b.mcif": RuntimeError("synthetic known error")},
        ),
    )

    summary = run_mcif_batch_with_auto_baseline(
        [file_a, file_b],
        tmp_path / "run1",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )
    auto_paths = batch_mcif._resolve_auto_baseline_paths(
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        space_tol=0.02,
        mtol=0.02,
        meigtol=0.00002,
        matrix_tol=0.01,
    )
    stored_baseline = json.loads(auto_paths["baseline_json"].read_text(encoding="utf-8"))
    case_a = batch_mcif._normalize_case_id(file_a)
    case_b = batch_mcif._normalize_case_id(file_b)
    stored_meta = json.loads(auto_paths["baseline_meta"].read_text(encoding="utf-8"))

    assert summary["auto_baseline"]["action"] == "created"
    assert summary["error_count"] == 1
    assert summary["exit_code"] == 0
    assert stored_meta["package_version"] == __version__
    assert stored_meta["updated_at"] is None
    assert stored_baseline[case_a]["status"] == "ok"
    assert stored_baseline[case_b]["status"] == "error"


def test_auto_baseline_updates_full_snapshot_for_error_repairs_and_new_files(
    monkeypatch, tmp_path
):
    file_a = _write_fake_mcif(tmp_path, "a.mcif")
    file_b = _write_fake_mcif(tmp_path, "b.mcif")
    file_c = _write_fake_mcif(tmp_path, "c.mcif")
    file_d = _write_fake_mcif(tmp_path, "d.mcif")
    baseline_root = tmp_path / "baseline_store"

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(
            ok_by_name={
                "a.mcif": _fake_summary("IDX_A"),
                "c.mcif": _fake_summary("IDX_C"),
            },
            errors_by_name={"b.mcif": RuntimeError("synthetic known error")},
        ),
    )
    run_mcif_batch_with_auto_baseline(
        [file_a, file_b, file_c],
        tmp_path / "run1",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(
            ok_by_name={
                "a.mcif": _fake_summary("IDX_A"),
                "b.mcif": _fake_summary("IDX_B_FIXED"),
                "d.mcif": _fake_summary("IDX_D"),
            },
        ),
    )
    summary = run_mcif_batch_with_auto_baseline(
        [file_a, file_b, file_d],
        tmp_path / "run2",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )
    auto_paths = batch_mcif._resolve_auto_baseline_paths(
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        space_tol=0.02,
        mtol=0.02,
        meigtol=0.00002,
        matrix_tol=0.01,
    )
    stored_baseline = json.loads(auto_paths["baseline_json"].read_text(encoding="utf-8"))
    stored_meta = json.loads(auto_paths["baseline_meta"].read_text(encoding="utf-8"))
    case_a = batch_mcif._normalize_case_id(file_a)
    case_b = batch_mcif._normalize_case_id(file_b)
    case_c = batch_mcif._normalize_case_id(file_c)
    case_d = batch_mcif._normalize_case_id(file_d)

    assert summary["auto_baseline"]["action"] == "updated"
    assert summary["comparison"]["protected_ok_mismatch_count"] == 0
    assert summary["comparison"]["error_to_ok_update_count"] == 1
    assert summary["comparison"]["new_case_count"] == 1
    assert summary["comparison"]["baseline_only_case_count"] == 1
    assert summary["exit_code"] == 0
    assert stored_meta["package_version"] == __version__
    assert stored_meta["created_at"]
    assert stored_meta["updated_at"]
    assert stored_meta["updated_at_epoch"] is not None
    assert stored_baseline[case_a]["result"] == _fake_summary("IDX_A")
    assert stored_baseline[case_b]["result"] == _fake_summary("IDX_B_FIXED")
    assert stored_baseline[case_c]["result"] == _fake_summary("IDX_C")
    assert stored_baseline[case_d]["result"] == _fake_summary("IDX_D")


def test_auto_baseline_blocks_overwrite_when_previous_ok_case_changes(monkeypatch, tmp_path):
    file_a = _write_fake_mcif(tmp_path, "a.mcif")
    baseline_root = tmp_path / "baseline_store"

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(ok_by_name={"a.mcif": _fake_summary("IDX_A")}),
    )
    run_mcif_batch_with_auto_baseline(
        [file_a],
        tmp_path / "run1",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(
            ok_by_name={"a.mcif": _fake_summary("IDX_A_CHANGED")}
        ),
    )
    summary = run_mcif_batch_with_auto_baseline(
        [file_a],
        tmp_path / "run2",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )
    auto_paths = batch_mcif._resolve_auto_baseline_paths(
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        space_tol=0.02,
        mtol=0.02,
        meigtol=0.00002,
        matrix_tol=0.01,
    )
    stored_baseline = json.loads(auto_paths["baseline_json"].read_text(encoding="utf-8"))
    case_a = batch_mcif._normalize_case_id(file_a)

    assert summary["auto_baseline"]["action"] == "blocked_by_ok_mismatch"
    assert summary["comparison"]["protected_ok_mismatch_count"] == 1
    assert summary["comparison"]["protected_ok_mismatches"][0]["differences"] == [
        {
            "field": "result.acc",
            "expected": "acc:IDX_A",
            "actual": "acc:IDX_A_CHANGED",
        },
        {
            "field": "result.conf",
            "expected": "conf:IDX_A",
            "actual": "conf:IDX_A_CHANGED",
        },
        {
            "field": "result.gspg.effective_mpg_symbol",
            "expected": "empg:IDX_A",
            "actual": "empg:IDX_A_CHANGED",
        },
        {
            "field": "result.gspg.npg_symbol_s",
            "expected": "npg:IDX_A",
            "actual": "npg:IDX_A_CHANGED",
        },
        {
            "field": "result.gspg.output_mode",
            "expected": "mode:IDX_A",
            "actual": "mode:IDX_A_CHANGED",
        },
        {
            "field": "result.gspg.spin_only_component_symbol_s",
            "expected": "ss:IDX_A",
            "actual": "ss:IDX_A_CHANGED",
        },
        {
            "field": "result.gspg.symbol_mode",
            "expected": "symbol_mode:IDX_A",
            "actual": "symbol_mode:IDX_A_CHANGED",
        },
        {
            "field": "result.gspg.tentative_symbol_s",
            "expected": "tentative:IDX_A",
            "actual": "tentative:IDX_A_CHANGED",
        },
        {
            "field": "result.index",
            "expected": "IDX_A",
            "actual": "IDX_A_CHANGED",
        },
        {
            "field": "result.phase",
            "expected": "phase:IDX_A",
            "actual": "phase:IDX_A_CHANGED",
        },
        {
            "field": "result.properties.ss_w_soc",
            "expected": "prop:IDX_A",
            "actual": "prop:IDX_A_CHANGED",
        },
    ]
    assert summary["exit_code"] == 1
    assert stored_baseline[case_a]["result"] == _fake_summary("IDX_A")


def test_auto_baseline_backfills_missing_tensor_summary_without_counting_mismatch(
    monkeypatch, tmp_path
):
    file_a = _write_fake_mcif(tmp_path, "a.mcif")
    baseline_root = tmp_path / "baseline_store"

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(ok_by_name={"a.mcif": _fake_summary("IDX_A")}),
    )
    run_mcif_batch_with_auto_baseline(
        [file_a],
        tmp_path / "run1",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )
    auto_paths = batch_mcif._resolve_auto_baseline_paths(
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        space_tol=0.02,
        mtol=0.02,
        meigtol=0.00002,
        matrix_tol=0.01,
    )
    stored_baseline = json.loads(auto_paths["baseline_json"].read_text(encoding="utf-8"))
    case_a = batch_mcif._normalize_case_id(file_a)
    stored_baseline[case_a].pop("tensor_summary", None)
    auto_paths["baseline_json"].write_text(
        json.dumps(stored_baseline, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        batch_mcif,
        "find_spin_group",
        _fake_find_spin_group_factory(
            ok_by_name={"a.mcif": _fake_summary("IDX_A")},
            tensor_by_name={"a.mcif": _fake_tensor_summary("A")},
        ),
    )
    summary = run_mcif_batch_with_auto_baseline(
        [file_a],
        tmp_path / "run2",
        baseline_root=baseline_root,
        suite_name="synthetic_suite",
        quiet=True,
    )
    updated_baseline = json.loads(auto_paths["baseline_json"].read_text(encoding="utf-8"))
    updated_meta = json.loads(auto_paths["baseline_meta"].read_text(encoding="utf-8"))

    assert summary["auto_baseline"]["action"] == "updated"
    assert summary["comparison"]["protected_ok_mismatch_count"] == 0
    assert summary["comparison"]["tensor_summary_backfill_count"] == 1
    assert updated_meta["package_version"] == __version__
    assert updated_meta["updated_at"]
    assert updated_meta["updated_at_epoch"] is not None
    assert updated_baseline[case_a]["tensor_summary"] == {
        "AHE_woSOC": {"free_parameters": 0, "is_zero": True, "relations_count": 0},
        "AHE_wSOC": {"free_parameters": 1, "is_zero": False, "relations_count": 1},
    }


def test_run_mcif_batch_writes_flat_error_artifacts(monkeypatch, tmp_path):
    source_file = (PROJECT_ROOT / MANIFEST_ENTRIES[0]).resolve()

    def raise_error(*args, **kwargs):
        raise RuntimeError("synthetic batch failure")

    monkeypatch.setattr(batch_mcif, "find_spin_group", raise_error)

    summary = run_mcif_batch([source_file], tmp_path, quiet=True)

    tagged_name = batch_mcif._tagged_artifact_name(source_file.name, summary["run_tag"])
    error_json = tmp_path / "error_json" / f"{tagged_name}.json"
    error_copy = tmp_path / "error_set" / tagged_name

    assert summary["error_count"] == 1
    assert summary["run_tag"].startswith(f"run_v{__version__}_")
    assert error_json.exists()
    assert error_copy.exists()
    assert not (tmp_path / "ok").exists()
    assert not (tmp_path / "error").exists()
