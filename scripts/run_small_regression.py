#!/usr/bin/env python3
"""Run a compact FindSpinGroup regression matrix.

The script is a thin orchestration layer over ``findspingroup.batch_mcif``.
It does not implement an alternate route: each suite runs the same batch entry
point used by local and cluster validation, then checks the resulting
``summary.json`` and ``errors_by_file.json`` against the suite contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


CORE_EXPORT_FIELDS = [
    "index",
    "phase",
    "properties.ss_w_soc",
    "properties.ss_wo_soc",
    "properties.ahc_w_soc",
    "properties.ahc_wo_soc",
    "properties.is_alter",
    "properties.is_spin_orbit_magnet",
]


QUASI2D_EXPORT_FIELDS = [
    "index",
    "phase",
    "properties.ss_wo_soc",
    "quasi_2d.status",
    "quasi_2d.source",
    "quasi_2d.vacuum_axis_input",
    "quasi_2d.magnetic_phase",
    "quasi_2d.diagnostic_points.0.k_symbol_2d",
    "quasi_2d.generic_point_comparison.summary",
    "quasi_2d.generic_point_comparison.spin_splitting_changed",
    "quasi_2d.kpoints",
]


FERROELECTRIC_EXPORT_FIELDS = [
    "index",
    "phase",
    "ferroelectric_switching.polarity_status",
    "ferroelectric_switching.status",
    "ferroelectric_switching.switching_detected",
    "ferroelectric_switching.domain_reversal_symmetry_screening.status",
    "ferroelectric_switching.ferroelectric_altermagnet_screening.status",
    "ferroelectric_switching.switchable_altermagnet_screening.status",
    "ferroelectric_switching.post_fsg_path_validation_requirements.status",
    "ferroelectric_switching.energy_barrier_workflow.status",
    "ferroelectric_switching.structural_parent_symmetry.space_group_number",
    "ferroelectric_switching.ordered_spin_space_symmetry.space_group_number",
    "ferroelectric_switching.soc_magnetic_symmetry.space_group_number",
]


@dataclass(frozen=True)
class Suite:
    name: str
    route: str
    calculation_mode: str
    inputs: tuple[str, ...]
    manifest: str | None = None
    vacuum_axis: str = "c"
    export_fields: tuple[str, ...] = tuple(CORE_EXPORT_FIELDS)
    expected_error_substrings: tuple[str, ...] = ()


SUITES = {
    "basic3d": Suite(
        name="basic3d",
        route="basic",
        calculation_mode="3d",
        manifest="tests/testset/regression_small/3d_core_manifest.txt",
        inputs=(),
        expected_error_substrings=("1.669_KFe(PO3F)2.mcif",),
    ),
    "full3d": Suite(
        name="full3d",
        route="full",
        calculation_mode="3d",
        manifest="tests/testset/regression_small/3d_core_manifest.txt",
        inputs=(),
        expected_error_substrings=("1.669_KFe(PO3F)2.mcif",),
    ),
    "quasi2d": Suite(
        name="quasi2d",
        route="full",
        calculation_mode="quasi2d",
        inputs=("tests/testset/quasi2d_small",),
        export_fields=tuple(QUASI2D_EXPORT_FIELDS),
    ),
    "ferroelectric3d": Suite(
        name="ferroelectric3d",
        route="full",
        calculation_mode="3d",
        manifest="tests/testset/ferroelectric_switching_small/3d_manifest.txt",
        inputs=(),
        export_fields=tuple(FERROELECTRIC_EXPORT_FIELDS),
    ),
}


def _json_load(path: Path) -> object:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_command(python: str, suite: Suite, output_dir: Path, quiet: bool) -> list[str]:
    command = [
        python,
        "-m",
        "findspingroup.batch_mcif",
        "--output-dir",
        str(output_dir),
        "--route",
        suite.route,
        "--calculation-mode",
        suite.calculation_mode,
        "--vacuum-axis",
        suite.vacuum_axis,
        "--export-txt",
        "selected.txt",
    ]
    for field in suite.export_fields:
        command.extend(["--export-field", field])
    if suite.manifest is not None:
        command.extend(["--manifest", suite.manifest])
    command.extend(suite.inputs)
    if quiet:
        command.append("--quiet")
    return command


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


def _validate_errors(errors_by_file: object, suite: Suite) -> dict[str, object]:
    errors = errors_by_file if isinstance(errors_by_file, dict) else {}
    expected = set(suite.expected_error_substrings)
    seen_expected = set()
    unexpected = []
    for case_id in sorted(errors):
        matched = False
        for token in expected:
            if token in case_id:
                seen_expected.add(token)
                matched = True
                break
        if not matched:
            unexpected.append(case_id)
    return {
        "expected_error_substrings": sorted(expected),
        "seen_expected_error_substrings": sorted(seen_expected),
        "missing_expected_error_substrings": sorted(expected - seen_expected),
        "unexpected_error_cases": unexpected,
    }


def _run_suite(python: str, suite: Suite, run_root: Path, quiet: bool) -> dict[str, object]:
    output_dir = run_root / suite.name
    command = _build_command(python, suite, output_dir, quiet)
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    finished = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    summary = _json_load(output_dir / "summary.json")
    errors_by_file = _json_load(output_dir / "errors_by_file.json")
    error_validation = _validate_errors(errors_by_file, suite)
    passed = (
        completed.returncode == 0
        and isinstance(summary, dict)
        and not error_validation["unexpected_error_cases"]
        and not error_validation["missing_expected_error_substrings"]
    )
    return {
        "name": suite.name,
        "passed": passed,
        "started_at": started,
        "finished_at": finished,
        "command": command,
        "returncode": completed.returncode,
        "output_dir": str(output_dir),
        "summary": summary,
        "error_validation": error_validation,
        "stdout_tail": completed.stdout.splitlines()[-80:],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(SUITES),
        help="Suite to run. Can be passed more than once. Defaults to all suites.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/findspingroup_small_regression"),
        help="Root directory where a timestamped run directory will be created.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke findspingroup.batch_mcif.",
    )
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to batch runs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    suite_names = args.suite or sorted(SUITES)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / f"run_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    payload = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "run_root": str(run_root),
        "suites": [],
    }
    for suite_name in suite_names:
        suite_result = _run_suite(args.python, SUITES[suite_name], run_root, args.quiet)
        payload["suites"].append(suite_result)
        (run_root / f"{suite_name}_driver_summary.json").write_text(
            json.dumps(suite_result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    payload["passed"] = all(suite["passed"] for suite in payload["suites"])
    summary_path = run_root / "small_regression_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
