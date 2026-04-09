#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from findspingroup import find_spin_group


DEFAULT_CASE_IDS = [
    "0.250",
    "1.0.20",
    "1.201",
    "1.450",
    "1.858",
    "1.859",
]

PREFERRED_DATA_ROOTS = [
    REPO_ROOT / "tests" / "testset" / "mcif_241130_no2186",
    REPO_ROOT / "tests" / "testset" / "new_251010_no2186_p82",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export project-standard acc primitive magnetic POSCAR files for one or more "
            "FindSpinGroup mcif cases."
        )
    )
    parser.add_argument(
        "cases",
        nargs="*",
        default=DEFAULT_CASE_IDS,
        help=(
            "Case ids like '1.450' or direct .mcif paths. Defaults to the six requested "
            "representative cases."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "magnetic_primitive_poscars"),
        help="Directory where exported POSCAR files will be written.",
    )
    return parser.parse_args()


def _resolve_case_path(case: str) -> Path:
    candidate = Path(case)
    if candidate.suffix.lower() == ".mcif":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"mcif file not found: {candidate}")
        return candidate

    case_prefix = case if case.endswith("_") else f"{case}_"
    for root in PREFERRED_DATA_ROOTS:
        matches = sorted(root.glob(f"{case_prefix}*.mcif"))
        if matches:
            return matches[0]

    fallback_matches = []
    for path in sorted((REPO_ROOT / "tests" / "testset").rglob(f"{case_prefix}*.mcif")):
        normalized = path.as_posix()
        if "/skipped" in normalized or "fsg_test_log" in normalized:
            continue
        fallback_matches.append(path)

    if not fallback_matches:
        raise FileNotFoundError(f"No mcif case found for '{case}'.")
    if len(fallback_matches) > 1:
        match_list = ", ".join(str(path.relative_to(REPO_ROOT)) for path in fallback_matches)
        raise ValueError(
            f"Multiple non-skipped matches found for '{case}'. Please pass the exact path. Matches: {match_list}"
        )
    return fallback_matches[0]


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        source_path = _resolve_case_path(case)
        result = find_spin_group(str(source_path))
        output_path = output_dir / f"{source_path.stem}.magnetic_primitive.POSCAR"
        output_path.write_text(result.acc_primitive_magnetic_cell_poscar + "\n", encoding="utf-8")
        print(
            f"Wrote {output_path.relative_to(REPO_ROOT)} "
            f"from {source_path.relative_to(REPO_ROOT)} "
            f"(index {result.index}, {result.conf})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
