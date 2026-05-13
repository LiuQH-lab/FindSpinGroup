import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_export_mcif_results_module():
    module_path = PROJECT_ROOT / "scripts" / "export_mcif_results_to_excel.py"
    spec = importlib.util.spec_from_file_location("export_mcif_results_to_excel", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_excel_export_rows_include_quasi2d_fields():
    exporter = _load_export_mcif_results_module()
    row = exporter._row_from_serialized_result_record(
        {
            "case_id": "quasi.mcif",
            "file_name": "quasi.mcif",
            "status": "ok",
            "duration_seconds": 1.25,
            "result": {
                "index": "123.47.1.1.L",
                "conf": "Collinear",
                "phase": "AFM",
                "acc": "4/mmmP",
                "quasi_2d": {
                    "calculation_mode": "quasi2d",
                    "dimension": "2d",
                    "status": "heuristic",
                    "source": "heuristic",
                    "vacuum_axis_input": "c",
                    "spin_splitting_2d": "spin splitting",
                    "interpretation": "in_plane_k_dependent",
                    "is_alter_2d": "(Altermagnet)",
                    "magnetic_phase": "AFM(Altermagnet)",
                    "diagnostic_points": [
                        {
                            "label": "GP",
                            "k_symbol_2d": "GP:(0.237,0.371,0)",
                            "k_input_reciprocal": [0.237, 0.371, 0.0],
                            "k_acc_primitive": [0.237, 0.371, 0.0],
                            "spin_splitting": "spin splitting",
                            "spin_polarizations": ["Sx", "0", "0"],
                        }
                    ],
                    "kpoints": [
                        {
                            "label": "Γ",
                            "k_symbol_2d": "Γ:(0,0,0)",
                            "plane_classification": "in_plane",
                            "spin_splitting": "no spin splitting",
                        },
                    ],
                    "kpoint_projection_summary": {
                        "source": "acc_table",
                        "total": 2,
                        "by_plane_count": {"in_plane": 1, "mixed": 1},
                    },
                    "generic_point_comparison": {
                        "status": "compared",
                        "gp_3d": {
                            "label": "GP",
                            "k_symbol_2d": "GP:(0.417,0.237,0.371)",
                            "k_input_reciprocal": [0.417, 0.237, 0.371],
                            "k_acc_primitive": [0.417, 0.237, 0.371],
                            "plane_classification": "mixed",
                            "spin_splitting": "no spin splitting",
                            "spin_polarizations": [],
                        },
                        "gp_2d": {
                            "label": "GP",
                            "k_symbol_2d": "GP:(0.237,0.371,0)",
                            "k_input_reciprocal": [0.237, 0.371, 0.0],
                            "k_acc_primitive": [0.237, 0.371, 0.0],
                            "plane_classification": "in_plane",
                            "spin_splitting": "spin splitting",
                            "spin_polarizations": ["Sx", "0", "0"],
                        },
                        "k_input_delta_wrapped": [-0.18, 0.134, -0.371],
                        "k_acc_delta_wrapped": [-0.18, 0.134, -0.371],
                        "k_input_changed": True,
                        "spin_splitting_changed": True,
                        "spin_polarization_changed": True,
                        "summary": "k_changed_spin_splitting_changed",
                    },
                },
            },
        }
    )

    assert row["calculation_mode"] == "quasi2d"
    assert row["dimension"] == "2d"
    assert row["quasi2d_status"] == "heuristic"
    assert row["quasi2d_source"] == "heuristic"
    assert row["vacuum_axis_input"] == "c"
    assert row["spin_splitting_2d"] == "spin splitting"
    assert row["spin_splitting_2d_interpretation"] == "in_plane_k_dependent"
    assert row["is_alter_2d"] == "(Altermagnet)"
    assert row["quasi2d_magnetic_phase"] == "AFM(Altermagnet)"
    assert row["quasi2d_gp_label"] == "GP"
    assert row["quasi2d_gp_symbol"] == "GP:(0.237,0.371,0)"
    assert row["quasi2d_gp_spin_splitting"] == "spin splitting"
    assert row["quasi2d_3d_gp_symbol"] == "GP:(0.417,0.237,0.371)"
    assert row["quasi2d_3d_gp_k_input"] == [0.417, 0.237, 0.371]
    assert row["quasi2d_2d_gp_symbol"] == "GP:(0.237,0.371,0)"
    assert row["quasi2d_2d_gp_k_input"] == [0.237, 0.371, 0.0]
    assert row["quasi2d_gp_k_input_changed"] is True
    assert row["quasi2d_gp_spin_splitting_changed"] is True
    assert row["quasi2d_gp_comparison_summary"] == "k_changed_spin_splitting_changed"
    assert row["quasi2d_kpoint_projection_summary"]["by_plane_count"]["mixed"] == 1
    assert row["quasi2d_kpoints"] == [
        {
            "label": "Γ",
            "k_symbol_2d": "Γ:(0,0,0)",
            "plane": "in_plane",
            "spin_splitting": "no spin splitting",
        }
    ]
