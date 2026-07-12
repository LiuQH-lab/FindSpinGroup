from pathlib import Path
from types import SimpleNamespace

from scripts import fsg_workflow


def test_release_test_submits_profile_quasi2d_stage(monkeypatch):
    profile = {
        "release_quasi2d_dataset": "quasi2d_inputversion_share",
    }
    config = {"profiles": {"share": profile}}
    snapshot = fsg_workflow.Snapshot(
        archive=Path("snapshot.tar.gz"),
        root_name="snapshot",
        remote_repo_root="/remote/snapshot",
    )
    mcif_calls = []
    roundtrip_calls = []

    monkeypatch.setattr(fsg_workflow, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        fsg_workflow,
        "_prepare_remote_snapshot",
        lambda _profile, dry_run: snapshot,
    )
    monkeypatch.setattr(
        fsg_workflow,
        "_submit_mcif_stage",
        lambda *args, **kwargs: mcif_calls.append(kwargs),
    )
    monkeypatch.setattr(
        fsg_workflow,
        "_submit_roundtrip_stage",
        lambda *args, **kwargs: roundtrip_calls.append(kwargs),
    )

    fsg_workflow.command_release_test(
        SimpleNamespace(
            profile_config=Path("profiles.toml"),
            profile="share",
            workers=56,
            roundtrip_workers=24,
            quasi2d_workers=40,
            quasi2d_dataset=None,
            run_local_tests=False,
            execute=False,
            tag="prepr",
        )
    )

    assert [call["dataset_name"] for call in mcif_calls] == [
        "mcif_260414_no2241_basic",
        "mcif_260414_no2241_full",
        "quasi2d_inputversion_share",
    ]
    quasi2d_call = mcif_calls[-1]
    assert quasi2d_call["route"] == "full"
    assert quasi2d_call["calculation_mode"] == "quasi2d"
    assert quasi2d_call["vacuum_axis"] is None
    assert quasi2d_call["workers"] == 40
    assert quasi2d_call["limit"] is None
    assert quasi2d_call["selected_export"] is True
    assert quasi2d_call["export_fields_override"] == fsg_workflow.QUASI2D_EXPORT_FIELDS
    assert quasi2d_call["tag"] == "prepr"
    assert [call["kind"] for call in roundtrip_calls] == ["poscar", "scif"]
    assert all(call["workers"] == 24 for call in roundtrip_calls)
