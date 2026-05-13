#!/usr/bin/env python3
"""Run a focused tolerance scan over known FindSpinGroup sensitive cases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from findspingroup.find_spin_group import NumpyEncoder, find_spin_group, find_spin_group_basic


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROFILES: dict[str, dict[str, float]] = {
    "default": {
        "space_tol": 0.02,
        "mtol": 0.02,
        "meigtol": 0.00002,
        "matrix_tol": 0.01,
        "parser_atol": 0.02,
    },
    "tight_space_matrix": {
        "space_tol": 0.01,
        "mtol": 0.02,
        "meigtol": 0.00002,
        "matrix_tol": 0.005,
        "parser_atol": 0.02,
    },
    "loose_space_matrix": {
        "space_tol": 0.03,
        "mtol": 0.02,
        "meigtol": 0.00002,
        "matrix_tol": 0.02,
        "parser_atol": 0.02,
    },
    "tight_mtol": {
        "space_tol": 0.02,
        "mtol": 0.01,
        "meigtol": 0.00002,
        "matrix_tol": 0.01,
        "parser_atol": 0.02,
    },
    "loose_mtol": {
        "space_tol": 0.02,
        "mtol": 0.05,
        "meigtol": 0.00002,
        "matrix_tol": 0.01,
        "parser_atol": 0.02,
    },
    "very_loose_mtol": {
        "space_tol": 0.02,
        "mtol": 0.1,
        "meigtol": 0.00002,
        "matrix_tol": 0.01,
        "parser_atol": 0.02,
    },
    "tight_meigtol": {
        "space_tol": 0.02,
        "mtol": 0.02,
        "meigtol": 0.000002,
        "matrix_tol": 0.01,
        "parser_atol": 0.02,
    },
    "loose_meigtol": {
        "space_tol": 0.02,
        "mtol": 0.02,
        "meigtol": 0.0002,
        "matrix_tol": 0.01,
        "parser_atol": 0.02,
    },
}


DEFAULT_CASES: list[dict[str, str]] = [
    {
        "path": "tests/testset/mcif_241130_no2186/0.120_LiFe(SO4)2.mcif",
        "reason": "mtol boundary: coplanar to collinear at loose mtol",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/0.122_Li2Mn(SO4)2.mcif",
        "reason": "mtol boundary: coplanar to collinear at loose mtol",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/0.1060_C3H6MnO6.mcif",
        "reason": "mtol boundary near 0.05",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/0.394_Cu2CdB2O6.mcif",
        "reason": "mtol boundary: noncoplanar/coplanar/collinear",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/0.1120_KTb3F10.mcif",
        "reason": "spin point-group tolerance sentinel",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/0.199_Mn3Sn.mcif",
        "reason": "Seitz-symbol and hexagonal spin-frame tolerance sentinel",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/0.427_Sm2Ti2O7.mcif",
        "reason": "nonorthogonal spin-frame / SCIF-sensitive sentinel",
    },
    {
        "path": "tests/testset/mcif_241130_no2186/2.116_Na3Co2SbO6.mcif",
        "reason": "ACC-P identify transform / monoclinic reduction sentinel",
    },
    {
        "path": "examples/2.35_CrSe.mcif",
        "reason": "GSPG explicit-ops smoke baseline sentinel",
    },
    {
        "path": "examples/CoNb3S6_tripleQ.mcif",
        "reason": "changed-basis and quasi-2D-adjacent route sentinel",
    },
]


KEY_FIELDS = (
    "index",
    "conf",
    "phase",
    "acc",
    "acc_symbol",
    "msg_num",
    "msg_symbol",
    "gspg_effective_mpg_symbol",
)


def _read_case_file(path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        cases.append({"path": parts[0], "reason": parts[1] if len(parts) > 1 else "user-provided"})
    return cases


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "to_summary_dict"):
        return payload.to_summary_dict()
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    if isinstance(payload, dict):
        return payload
    return {"repr": repr(payload)}


def _extract_record_fields(payload: dict[str, Any]) -> dict[str, Any]:
    fields = {key: payload.get(key) for key in KEY_FIELDS if key in payload}
    properties = payload.get("properties")
    if isinstance(properties, dict):
        fields["properties"] = {
            key: properties.get(key)
            for key in (
                "ss_w_soc",
                "ss_wo_soc",
                "ahc_w_soc",
                "ahc_wo_soc",
                "is_alter",
                "is_spin_orbit_magnet",
            )
            if key in properties
        }
    magnetic_phase = payload.get("magnetic_phase")
    if magnetic_phase is not None:
        fields["magnetic_phase"] = magnetic_phase
    return fields


def _run_one(path: Path, profile: dict[str, float], route: str) -> dict[str, Any]:
    if not path.is_absolute():
        path = REPO_ROOT / path
    kwargs = {
        "space_tol": profile["space_tol"],
        "mtol": profile["mtol"],
        "meigtol": profile["meigtol"],
        "matrix_tol": profile["matrix_tol"],
        "parser_atol": profile["parser_atol"],
    }
    try:
        if route == "basic":
            payload = find_spin_group_basic(str(path), **kwargs)
        else:
            payload = find_spin_group(str(path), **kwargs)
        return {
            "status": "ok",
            "fields": _extract_record_fields(_normalize_payload(payload)),
        }
    except Exception as exc:  # noqa: BLE001 - scanner records diagnostics by design.
        return {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _classify_case(case_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = case_records.get("default")
    if baseline is None:
        return {"summary": "unresolved", "changes": ["missing default profile"]}

    changes: list[dict[str, Any]] = []
    for name, record in case_records.items():
        if name == "default":
            continue
        if record.get("status") != baseline.get("status"):
            changes.append(
                {
                    "profile": name,
                    "type": "status_changed",
                    "default": baseline.get("status"),
                    "current": record.get("status"),
                }
            )
            continue
        if record.get("status") != "ok":
            if record.get("error") != baseline.get("error"):
                changes.append(
                    {
                        "profile": name,
                        "type": "error_changed",
                        "default": baseline.get("error"),
                        "current": record.get("error"),
                    }
                )
            continue
        default_fields = baseline.get("fields", {})
        current_fields = record.get("fields", {})
        for key in sorted(set(default_fields) | set(current_fields)):
            if default_fields.get(key) == current_fields.get(key):
                continue
            change_type = "field_changed"
            if key == "index":
                change_type = "index_changed"
            elif key in {"conf", "phase", "magnetic_phase", "properties"}:
                change_type = "classification_changed"
            changes.append(
                {
                    "profile": name,
                    "type": change_type,
                    "field": key,
                    "default": default_fields.get(key),
                    "current": current_fields.get(key),
                }
            )

    if not changes:
        summary = "stable"
    elif any(change["type"] in {"status_changed", "error_changed", "index_changed"} for change in changes):
        summary = "index_or_status_sensitive"
    elif any(change["type"] == "classification_changed" for change in changes):
        summary = "classification_boundary"
    else:
        summary = "field_sensitive"
    return {"summary": summary, "changes": changes}


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FindSpinGroup Tolerance Scan",
        "",
        f"Created: `{payload['created_at']}`",
        f"Route: `{payload['route']}`",
        "",
        "## Profiles",
        "",
        "| Profile | space_tol | mtol | meigtol | matrix_tol | parser_atol |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, profile in payload["profiles"].items():
        lines.append(
            "| {name} | {space_tol:g} | {mtol:g} | {meigtol:g} | {matrix_tol:g} | {parser_atol:g} |".format(
                name=name,
                **profile,
            )
        )
    lines.extend(
        [
            "",
            "## Case Summary",
            "",
            "| Case | Classification | Reason | Changes |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for case in payload["cases"]:
        classification = payload["classification"][case["path"]]
        lines.append(
            f"| `{case['path']}` | {classification['summary']} | {case['reason']} | "
            f"{len(classification['changes'])} |"
        )
    lines.extend(["", "## Changes", ""])
    for case in payload["cases"]:
        classification = payload["classification"][case["path"]]
        if not classification["changes"]:
            continue
        lines.append(f"### `{case['path']}`")
        lines.append("")
        for change in classification["changes"]:
            field = change.get("field", "<status>")
            lines.append(
                f"- `{change['profile']}` `{change['type']}` `{field}`: "
                f"`{change.get('default')}` -> `{change.get('current')}`"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        choices=["basic", "full"],
        default="basic",
        help="Route to scan. Basic is the default because it is fast and covers index/classification.",
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        help="Optional case list. Each non-comment line is 'path [reason...]'.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(DEFAULT_PROFILES),
        help="Profile name to include. Defaults to all built-in profiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/findspingroup_tolerance_scan"),
        help="Directory for tolerance_scan.json and tolerance_scan.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _read_case_file(args.case_file) if args.case_file else list(DEFAULT_CASES)
    profile_names = args.profile or list(DEFAULT_PROFILES)
    profiles = {name: DEFAULT_PROFILES[name] for name in profile_names}
    if "default" not in profiles:
        profiles = {"default": DEFAULT_PROFILES["default"], **profiles}

    records: dict[str, dict[str, Any]] = {}
    classification: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_path = Path(case["path"])
        case_records = {
            profile_name: _run_one(case_path, profile, args.route)
            for profile_name, profile in profiles.items()
        }
        records[case["path"]] = case_records
        classification[case["path"]] = _classify_case(case_records)

    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "route": args.route,
        "profiles": profiles,
        "cases": cases,
        "records": records,
        "classification": classification,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tolerance_scan.json"
    md_path = args.output_dir / "tolerance_scan.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    _write_markdown(md_path, payload)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
