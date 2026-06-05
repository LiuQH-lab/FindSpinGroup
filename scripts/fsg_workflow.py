#!/usr/bin/env python3
"""Privacy-aware FindSpinGroup workflow commands.

This script is intentionally profile-driven.  Do not put private SSH aliases,
cluster paths, partitions, QOS names, or interpreter paths in this file.  Put
them in `.fsg-batch-profiles.local.toml`, which is ignored by git.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPO_ROOT / ".fsg-batch-profiles.local.toml"


@dataclass(frozen=True)
class Snapshot:
    archive: Path
    root_name: str
    remote_repo_root: str


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _run(args: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    if dry_run:
        print(" ".join(_q(arg) for arg in args))
        return
    subprocess.run(args, check=True, env=env)


def _run_shell(script: str, *, dry_run: bool) -> None:
    script = textwrap.dedent(script).strip()
    if dry_run:
        print(script)
        return
    subprocess.run(["bash", "-lc", script], check=True)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing profile config: {path}\n"
            "Copy fsg-batch-profiles.example.toml to "
            ".fsg-batch-profiles.local.toml and fill in local values."
        )
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _profile(config: dict[str, Any], name: str) -> dict[str, Any]:
    profiles = config.get("profiles") or {}
    if name not in profiles:
        raise SystemExit(f"Unknown profile {name!r}. Available: {', '.join(sorted(profiles))}")
    return profiles[name]


def _dataset(config: dict[str, Any], name: str) -> dict[str, Any]:
    datasets = config.get("datasets") or {}
    if name not in datasets:
        raise SystemExit(f"Unknown dataset {name!r}. Available: {', '.join(sorted(datasets))}")
    return datasets[name]


def _sbatch_args(profile: dict[str, Any], *, key: str = "sbatch_args") -> str:
    args = profile.get(key)
    if args is None and key != "sbatch_args":
        args = profile.get("sbatch_args")
    args = args or []
    if isinstance(args, str):
        return args
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _remote_output_root(profile: dict[str, Any], dataset: dict[str, Any]) -> str:
    if dataset.get("remote_output_root"):
        return str(dataset["remote_output_root"])
    output_root_name = dataset.get("output_root_name")
    output_base = profile.get("remote_output_base")
    if output_root_name and output_base:
        return f"{str(output_base).rstrip('/')}/{output_root_name}"
    if profile.get("remote_output_root"):
        return str(profile["remote_output_root"])
    raise SystemExit(
        "Profile must define remote_output_base or remote_output_root, "
        "or the dataset must define remote_output_root."
    )


def _remote_input_dir(remote_repo_root: str, dataset: dict[str, Any]) -> str:
    if dataset.get("remote_input_dir"):
        return str(dataset["remote_input_dir"])
    input_subdir = dataset.get("input_subdir")
    if not input_subdir:
        raise SystemExit("Dataset must define remote_input_dir or input_subdir.")
    return f"{remote_repo_root.rstrip('/')}/{input_subdir}"


def _require_profile_keys(profile: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if not profile.get(key)]
    if missing:
        raise SystemExit(f"Profile is missing required keys: {', '.join(missing)}")


def _build_snapshot(*, dry_run: bool) -> tuple[Path | None, str]:
    build_script = REPO_ROOT / "scripts" / "build_batch_snapshot.sh"
    if dry_run:
        print(f"{_q(build_script)}")
        return None, "<snapshot-root-name>"
    completed = subprocess.run([str(build_script)], check=True, text=True, stdout=subprocess.PIPE)
    print(completed.stdout, end="")
    archive: Path | None = None
    root_name: str | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("Snapshot file : "):
            archive = Path(line.removeprefix("Snapshot file : ").strip())
        elif line.startswith("Snapshot root : "):
            root_name = line.removeprefix("Snapshot root : ").strip()
    if archive is None or root_name is None:
        raise SystemExit("Could not parse snapshot build output.")
    return archive, root_name


def _prepare_remote_snapshot(
    profile: dict[str, Any],
    *,
    dry_run: bool,
) -> Snapshot:
    _require_profile_keys(
        profile,
        ("server_alias", "remote_shared_repo", "remote_snapshot_root"),
    )
    archive, root_name = _build_snapshot(dry_run=dry_run)
    remote_snapshot_root = str(profile["remote_snapshot_root"]).rstrip("/")
    remote_repo_root = f"{remote_snapshot_root}/{root_name}"
    remote_tarball = f"{remote_snapshot_root}/{Path(str(archive)).name if archive else '<snapshot.tar.gz>'}"
    server_alias = str(profile["server_alias"])
    remote_tmp_dir = str(profile.get("remote_tmp_dir") or f"{remote_snapshot_root}/tmp")
    remote_shared_repo = str(profile["remote_shared_repo"]).rstrip("/")
    python_bin = str(profile.get("python_bin") or "./.venv/bin/python")

    _run(["ssh", server_alias, f"mkdir -p {_q(remote_snapshot_root)} {_q(remote_tmp_dir)}"], dry_run=dry_run)
    if archive is not None:
        _run(["scp", str(archive), f"{server_alias}:{remote_tarball}"], dry_run=dry_run)

    prepare_script = f"""
    set -euo pipefail
    export TMPDIR={_q(remote_tmp_dir)}
    cd {_q(remote_snapshot_root)}
    rm -rf {_q(remote_repo_root)}
    tar --warning=no-unknown-keyword -xzf {_q(remote_tarball)}
    cd {_q(remote_repo_root)}
    ln -sfn {_q(remote_shared_repo + "/.venv")} .venv
    ln -sfn {_q(remote_shared_repo + "/batch_baselines")} batch_baselines
    test -d src/findspingroup
    PYTHONPATH="$PWD/src" {_q(python_bin)} -c "import spintensor; import findspingroup; print('snapshot_ok')"
    """
    _run(["ssh", server_alias, prepare_script], dry_run=dry_run)
    return Snapshot(
        archive=archive or Path("<snapshot.tar.gz>"),
        root_name=root_name,
        remote_repo_root=remote_repo_root,
    )


def _remote_env_prefix(
    profile: dict[str, Any],
    *,
    workers: int | None,
    sbatch_args_key: str = "sbatch_args",
) -> str:
    env = {
        "SBATCH_ARGS": _sbatch_args(profile, key=sbatch_args_key),
        "PYTHON_BIN": str(profile.get("python_bin") or ""),
        "BATCH_WORKERS": str(workers or profile.get("workers") or 1),
    }
    return " ".join(f"{key}={_q(value)}" for key, value in env.items() if value != "")


def _submit_remote_stage(
    profile: dict[str, Any],
    snapshot: Snapshot,
    *,
    command: str,
    dry_run: bool,
) -> None:
    server_alias = str(profile["server_alias"])
    remote_script = f"""
    set -euo pipefail
    cd {_q(snapshot.remote_repo_root)}
    {command}
    """
    _run(["ssh", server_alias, remote_script], dry_run=dry_run)


def _submit_mcif_stage(
    config: dict[str, Any],
    profile: dict[str, Any],
    snapshot: Snapshot,
    *,
    dataset_name: str,
    route: str,
    calculation_mode: str,
    vacuum_axis: str | None,
    workers: int | None,
    limit: int | None,
    runtime_export: bool,
    selected_export: bool,
    export_fields_override: str | None,
    tag: str | None,
    dry_run: bool,
    baseline_suite_override: str | None = None,
) -> None:
    dataset = _dataset(config, dataset_name)
    input_dir = _remote_input_dir(snapshot.remote_repo_root, dataset)
    output_root = _remote_output_root(profile, dataset)
    if tag:
        output_root = f"{output_root}_{tag}"
    baseline_suite = str(baseline_suite_override or dataset.get("baseline_suite") or dataset_name)
    baseline_root = f"{snapshot.remote_repo_root}/batch_baselines"
    export_txt = "selected.txt" if selected_export else ""
    if export_fields_override is not None:
        export_fields = export_fields_override
    elif not selected_export:
        export_fields = ""
    else:
        export_fields = (
            "index,magnetic_phase,acc_symbol"
            if route == "basic"
            else "index,phase,properties.ss_w_soc"
        )
    env_prefix = _remote_env_prefix(profile, workers=workers)
    command = (
        f"{env_prefix} "
        f"BATCH_ROUTE={_q(route)} "
        f"CALCULATION_MODE={_q(calculation_mode)} "
        f"VACUUM_AXIS={_q(vacuum_axis or '')} "
        f"LIMIT={_q(limit or '')} "
        f"EXPORT_RUNTIME_ROWS={_q('1' if runtime_export else '0')} "
        f"EXPORT_TXT={_q(export_txt)} "
        f"EXPORT_FIELDS={_q(export_fields)} "
        f"bash scripts/submit_batch_mcif.sh "
        f"{_q(input_dir)} {_q(output_root)} {_q(baseline_suite)} {_q(baseline_root)}"
    )
    _submit_remote_stage(profile, snapshot, command=command, dry_run=dry_run)


def _submit_roundtrip_stage(
    config: dict[str, Any],
    profile: dict[str, Any],
    snapshot: Snapshot,
    *,
    dataset_name: str,
    kind: str,
    workers: int | None,
    tag: str | None,
    dry_run: bool,
) -> None:
    dataset = _dataset(config, dataset_name)
    input_dir = _remote_input_dir(snapshot.remote_repo_root, dataset)
    output_root = _remote_output_root(profile, dataset)
    if tag:
        output_root = f"{output_root}_{tag}"
    env_prefix = _remote_env_prefix(
        profile,
        workers=workers,
        sbatch_args_key="roundtrip_sbatch_args",
    )
    if kind == "poscar":
        wrapper = "scripts/submit_batch_poscar_roundtrip.sh"
        extra = "SOURCE_MODE=acc_primitive COMPARE_MODE=basic"
    elif kind == "scif":
        wrapper = "scripts/submit_batch_scif_roundtrip.sh"
        extra = ""
    else:
        raise ValueError(kind)
    command = f"{env_prefix} {extra} bash {wrapper} {_q(input_dir)} {_q(output_root)}"
    _submit_remote_stage(profile, snapshot, command=command, dry_run=dry_run)


def command_batch_test(args: argparse.Namespace) -> None:
    config = _load_config(args.profile_config)
    profile = _profile(config, args.profile)
    snapshot = _prepare_remote_snapshot(profile, dry_run=not args.execute)
    _submit_mcif_stage(
        config,
        profile,
        snapshot,
        dataset_name=args.dataset,
        route=args.route,
        calculation_mode=args.calculation_mode,
        vacuum_axis=None,
        workers=args.workers,
        limit=None,
        runtime_export=False,
        selected_export=False,
        export_fields_override=None,
        baseline_suite_override=None,
        tag=args.tag,
        dry_run=not args.execute,
    )
    print("Batch-test submission prepared. Inspect summary.json and core comparison after completion.")


def command_quasi2d_test(args: argparse.Namespace) -> None:
    config = _load_config(args.profile_config)
    profile = _profile(config, args.profile)
    if args.axis_sweep and args.vacuum_axis:
        raise SystemExit("--axis-sweep cannot be combined with --vacuum-axis.")
    snapshot = _prepare_remote_snapshot(profile, dry_run=not args.execute)
    axes = ["a", "b", "c"] if args.axis_sweep else [args.vacuum_axis]
    export_fields = (
        "index,phase,quasi_2d.status,quasi_2d.source,"
        "quasi_2d.vacuum_axis_input,quasi_2d.magnetic_phase,"
        "quasi_2d.spin_splitting_2d,quasi_2d.interpretation,"
        "quasi_2d.generic_point_comparison.summary,"
        "quasi_2d.generic_point_comparison.spin_splitting_changed,"
        "quasi_2d.spin_texture_config_no_soc.spin_texture_type,"
        "quasi_2d.spin_texture_config_no_soc.momentum_space_spin_configuration,"
        "quasi_2d.spin_texture_config_no_soc.operation_audit.non_plane_preserving_operation_count,"
        "quasi_2d.spin_texture_config_soc.spin_texture_type,"
        "quasi_2d.spin_texture_config_soc.momentum_space_spin_configuration,"
        "quasi_2d.spin_texture_config_soc.operation_audit.non_plane_preserving_operation_count"
    )
    dataset = _dataset(config, args.dataset)
    base_baseline_suite = str(dataset.get("baseline_suite") or args.dataset)
    for axis in axes:
        axis_tag = args.tag
        baseline_suite_override = None
        if args.axis_sweep:
            axis_tag = f"{args.tag}_axis_{axis}" if args.tag else f"axis_{axis}"
            baseline_suite_override = f"{base_baseline_suite}_axis_{axis}"
        _submit_mcif_stage(
            config,
            profile,
            snapshot,
            dataset_name=args.dataset,
            route="full",
            calculation_mode="quasi2d",
            vacuum_axis=axis,
            workers=args.workers,
            limit=args.limit,
            runtime_export=False,
            selected_export=True,
            export_fields_override=export_fields,
            baseline_suite_override=baseline_suite_override,
            tag=axis_tag,
            dry_run=not args.execute,
        )
    print(
        "Quasi2D-test submission prepared. Inspect summary.json, errors_by_file.json, "
        "selected.txt, and quasi_2d aggregate counts after completion."
    )


def command_release_test(args: argparse.Namespace) -> None:
    config = _load_config(args.profile_config)
    profile = _profile(config, args.profile)
    roundtrip_workers = args.roundtrip_workers or args.workers
    if args.run_local_tests:
        pytest_cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            "tests/test_find_spin_group.py",
            "tests/test_poscar_parser.py",
            "tests/test_scif_parser.py",
            "tests/test_export_mcif_results.py",
            "-q",
        ]
        _run(pytest_cmd, dry_run=not args.execute)
    snapshot = _prepare_remote_snapshot(profile, dry_run=not args.execute)
    _submit_mcif_stage(
        config,
        profile,
        snapshot,
        dataset_name="mcif_260414_no2241_basic",
        route="basic",
        calculation_mode="3d",
        vacuum_axis=None,
        workers=args.workers,
        limit=None,
        runtime_export=False,
        selected_export=False,
        export_fields_override=None,
        baseline_suite_override=None,
        tag=args.tag,
        dry_run=not args.execute,
    )
    _submit_mcif_stage(
        config,
        profile,
        snapshot,
        dataset_name="mcif_260414_no2241_full",
        route="full",
        calculation_mode="3d",
        vacuum_axis=None,
        workers=args.workers,
        limit=None,
        runtime_export=False,
        selected_export=False,
        export_fields_override=None,
        baseline_suite_override=None,
        tag=args.tag,
        dry_run=not args.execute,
    )
    _submit_roundtrip_stage(
        config,
        profile,
        snapshot,
        dataset_name="poscar_roundtrip_2241",
        kind="poscar",
        workers=roundtrip_workers,
        tag=args.tag,
        dry_run=not args.execute,
    )
    _submit_roundtrip_stage(
        config,
        profile,
        snapshot,
        dataset_name="scif_roundtrip_2241",
        kind="scif",
        workers=roundtrip_workers,
        tag=args.tag,
        dry_run=not args.execute,
    )
    print("Release-test submissions prepared. Do not tag or push until all stages are clean.")


def command_export(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    runtime_jsonl = run_dir / "full_results.jsonl"
    if not runtime_jsonl.exists():
        raise SystemExit(f"Missing runtime results: {runtime_jsonl}")
    output = args.output
    if output is None:
        suffix = "xlsx" if args.format == "xlsx" else "jsonl"
        output = run_dir / f"{args.preset}.{suffix}"
    output = output.resolve()
    if args.preset == "ferroelectric-switching":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_ferroelectric_switching_ops.py"),
            "--runtime-jsonl",
            str(runtime_jsonl),
        ]
        if args.format == "xlsx":
            cmd += ["--output-xlsx", str(output)]
        elif args.format == "csv":
            cmd += ["--output-csv", str(output)]
        else:
            cmd += ["--output-jsonl", str(output)]
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_mcif_results_to_excel.py"),
            "--runtime-jsonl",
            str(runtime_jsonl),
        ]
        if args.format == "xlsx":
            cmd += ["--output-xlsx", str(output)]
        else:
            cmd += ["--output-jsonl", str(output)]
    _run(cmd, dry_run=not args.execute)
    print(f"Export prepared: {output}")


def command_install_smoke(args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        smoke_dir = args.output_dir.resolve()
    elif args.execute:
        smoke_dir = Path(tempfile.mkdtemp(prefix="fsg-install-smoke-")).resolve()
    else:
        smoke_dir = Path("/tmp/fsg-install-smoke")
    venv_dir = smoke_dir / ".venv"
    python = args.python
    package = args.package
    commands = [
        [python, "-m", "venv", str(venv_dir)],
        [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip"],
        [str(venv_dir / "bin" / "python"), "-m", "pip", "install", package],
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            "import findspingroup; from findspingroup.version import __version__; print(__version__)",
        ],
        [str(venv_dir / "bin" / "fsg"), "--help"],
        [str(venv_dir / "bin" / "findspingroup"), "--help"],
    ]
    for cmd in commands:
        _run(cmd, dry_run=not args.execute)
    print(f"Install smoke prepared: {smoke_dir}")


def add_common_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="Profile name from local profile config.")
    parser.add_argument(
        "--profile-config",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help="Local gitignored profile TOML.",
    )
    parser.add_argument("--workers", type=int, help="Override profile worker count.")
    parser.add_argument("--tag", help="Short run tag suffix for output roots.")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default prints a dry-run plan.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("batch-test", help="Submit a normal full-dataset change validation.")
    add_common_profile_args(batch)
    batch.add_argument("--dataset", default="mcif_260414_no2241")
    batch.add_argument("--route", choices=("basic", "full"), default="full")
    batch.add_argument("--calculation-mode", default="3d")
    batch.set_defaults(func=command_batch_test)

    quasi2d = subparsers.add_parser("quasi2d-test", help="Submit the quasi-2D inputversion validation.")
    add_common_profile_args(quasi2d)
    quasi2d.add_argument("--dataset", default="quasi2d_inputversion")
    quasi2d.add_argument(
        "--vacuum-axis",
        choices=("a", "b", "c"),
        help="Optional explicit vacuum axis. Omit to use runtime auto/heuristic handling.",
    )
    quasi2d.add_argument(
        "--axis-sweep",
        action="store_true",
        help="Submit separate explicit a/b/c vacuum-axis runs for 2D guard diagnostics.",
    )
    quasi2d.add_argument(
        "--limit",
        type=int,
        help="Limit the number of resolved .mcif inputs for quick 2D diagnostics.",
    )
    quasi2d.set_defaults(func=command_quasi2d_test)

    release = subparsers.add_parser("release-test", help="Submit the pre-push release validation suite.")
    add_common_profile_args(release)
    release.add_argument(
        "--roundtrip-workers",
        type=int,
        help="Override roundtrip worker count. Defaults to --workers/profile workers.",
    )
    release.add_argument("--no-local-tests", dest="run_local_tests", action="store_false")
    release.set_defaults(func=command_release_test, run_local_tests=True)

    export = subparsers.add_parser("export", help="Generate business exports from an existing full run.")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument(
        "--preset",
        choices=("standard-excel", "magnetic-site", "ferroelectric-switching", "symmetry-property"),
        default="standard-excel",
    )
    export.add_argument("--format", choices=("xlsx", "jsonl", "csv"), default="xlsx")
    export.add_argument("--output", type=Path)
    export.add_argument("--execute", action="store_true", help="Actually run export. Default prints command.")
    export.set_defaults(func=command_export)

    smoke = subparsers.add_parser("install-smoke", help="Create a fresh venv, pip install, and check CLI entry points.")
    smoke.add_argument("--python", default="python3")
    smoke.add_argument("--package", default="findspingroup")
    smoke.add_argument("--output-dir", type=Path)
    smoke.add_argument("--execute", action="store_true", help="Actually run smoke. Default prints commands.")
    smoke.set_defaults(func=command_install_smoke)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
