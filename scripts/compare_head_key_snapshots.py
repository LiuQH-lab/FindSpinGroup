#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

DEFAULT_CASES = [
    ("file", "examples/CoNb3S6_tripleQ.mcif", "CoNb3S6_tripleQ_example"),
    ("file", "tests/testset/mcif_241130_no2186/3.24_CaFe3Ti4O12.mcif", "3.24_CaFe3Ti4O12"),
    ("file", "tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif", "1.0.48_MnSe2"),
    ("file", "tests/testset/mcif_241130_no2186/0.712_VNb3S6.mcif", "0.712_VNb3S6"),
    ("file", "tests/testset/mcif_241130_no2186/0.2_Cd2Os2O7.mcif", "0.2_Cd2Os2O7"),
    ("changed_basis_conb3s6", "", "changed_basis_Conb3S6_tripleQ"),
]

DEFAULT_FIELDS = [
    "index",
    "conf",
    "acc",
    "msg_acc",
    "msg_num",
    "msg_type",
    "msg_symbol",
    "convention_ssg_international_linear",
    "input_space_group_symbol",
    "input_space_group_number",
    "KPOINTS",
    "spin_polarizations",
    "msg_spin_polarizations",
    "T_input_to_G0std",
    "T_input_to_L0std",
    "T_input_to_acc_primitive",
    "T_convention_to_acc_primitive",
]

OVERLAY_HEAD_FILES = [
    "src/findspingroup/core/identify_spin_space_group.py",
    "src/findspingroup/core/identify_symmetry_from_ops.py",
    "src/findspingroup/core/pg_analyzer.py",
    "src/findspingroup/core/properties.py",
    "src/findspingroup/find_spin_group.py",
    "src/findspingroup/io/cif_parser.py",
    "src/findspingroup/io/scif_generator.py",
    "src/findspingroup/structure/cell.py",
    "src/findspingroup/structure/group.py",
    "src/findspingroup/utils/international_symbol.py",
    "src/findspingroup/utils/matrix_utils.py",
    "src/findspingroup/utils/seitz_symbol.py",
]

SNAPSHOT_SCRIPT = r"""
import json
import os
import sys
import importlib

repo_root = os.path.abspath(sys.argv[1])
case_kind = sys.argv[2]
case_path = sys.argv[3]
changed_basis_json = sys.argv[4]
fields = json.loads(sys.argv[5])

for path in [repo_root, os.path.join(repo_root, "src")]:
    if path not in sys.path:
        sys.path.insert(0, path)

fsg_mod = importlib.import_module("findspingroup.find_spin_group")
fsg_mod.generate_scif = lambda *args, **kwargs: ""
find_spin_group = fsg_mod.find_spin_group
find_spin_group_from_data = fsg_mod.find_spin_group_from_data

if case_kind == "file":
    result = find_spin_group(os.path.join(repo_root, case_path))
elif case_kind == "changed_basis_conb3s6":
    payload = json.loads(changed_basis_json)
    result = find_spin_group_from_data(
        "changed_basis_Conb3s6",
        payload["lattice"],
        payload["positions"],
        payload["elements"],
        payload["occupancies"],
        payload["moments"],
    )
else:
    raise ValueError(case_kind)

snapshot = {field: getattr(result, field, None) for field in fields}
print(json.dumps(snapshot, ensure_ascii=False))
"""


def _changed_basis_payload() -> dict:
    current_src = str(REPO_ROOT / "src")
    current_root = str(REPO_ROOT)
    for path in [current_root, current_src]:
        if path not in sys.path:
            sys.path.insert(0, path)
    from tests.test_find_spin_group import _changed_basis_conb3s6_tripleq_input

    lattice, positions, elements, occupancies, moments = _changed_basis_conb3s6_tripleq_input()
    return {
        "lattice": lattice,
        "positions": positions,
        "elements": elements,
        "occupancies": occupancies,
        "moments": moments,
    }


def _build_overlay_from_ref(ref: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="fsg_snapshot_compare."))
    for folder in ["src", "tests", "examples"]:
        shutil.copytree(REPO_ROOT / folder, tmpdir / folder)

    for rel_path in OVERLAY_HEAD_FILES:
        target = tmpdir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        target.write_bytes(blob)

    return tmpdir


def _run_snapshot(
    repo_root: Path,
    case_kind: str,
    case_path: str,
    changed_basis_payload: dict,
    fields: list[str],
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo_root / "src"),
            str(repo_root),
        ]
    )
    cmd = [
        str(PYTHON_BIN),
        "-c",
        SNAPSHOT_SCRIPT,
        str(repo_root),
        case_kind,
        case_path,
        json.dumps(changed_basis_payload),
        json.dumps(fields),
    ]
    cp = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            "Snapshot command failed for "
            f"repo_root={repo_root} case={case_kind}:{case_path}\n"
            f"STDERR:\n{cp.stderr}"
        )
    return json.loads(cp.stdout.strip().splitlines()[-1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare key MagSymmetryResult fields between a git ref and the current "
            "working tree on a fixed sentinel case set."
        )
    )
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="Git ref used as the old snapshot baseline. Default: HEAD.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path to write the full compare payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overlay_root = _build_overlay_from_ref(args.base_ref)
    changed_basis_payload = _changed_basis_payload()

    try:
        summary = {}
        for case_kind, case_path, label in DEFAULT_CASES:
            old = _run_snapshot(
                overlay_root,
                case_kind,
                case_path,
                changed_basis_payload,
                DEFAULT_FIELDS,
            )
            current = _run_snapshot(
                REPO_ROOT,
                case_kind,
                case_path,
                changed_basis_payload,
                DEFAULT_FIELDS,
            )
            diffs = {}
            for field in DEFAULT_FIELDS:
                if old.get(field) != current.get(field):
                    diffs[field] = {
                        "old": old.get(field),
                        "current": current.get(field),
                    }
            summary[label] = {
                "same": not diffs,
                "diffs": diffs,
            }
    finally:
        shutil.rmtree(overlay_root, ignore_errors=True)

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
