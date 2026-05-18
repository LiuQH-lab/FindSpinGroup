#!/usr/bin/env python3
"""Select representative 3D cases for ferroelectric-switching diagnostics.

The selector consumes a FindSpinGroup batch ``records.jsonl`` file and groups
successful 3D records by the symmetry relation relevant to the current
ferroelectric-switching payload:

* structural parent/current nonmagnetic SG has nontrivial cosets over the
  ordered spin-space real-space projection and at least one coset representative
  maps the ordered polar axis P to -P
* nontrivial parent/ordered cosets without a P-reversal representative
* current polar/trivial-coset controls such as the Gu-table MAGNDATA entries
* ordered nonpolar controls

The output manifest is intentionally a small review set, not a new baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from findspingroup.utils.space_group_flags import space_group_polar_axis_labels


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    REPO_ROOT
    / "output"
    / "mcif_241130_no2186_run"
    / "run_v0.13.16_20260413_221533"
    / "records.jsonl"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "testset"
    / "ferroelectric_switching_small"
    / "3d_manifest.txt"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "output"
    / "ferroelectric_switching_small"
    / "3d_case_selection.md"
)


def _case_path(record: dict[str, Any]) -> str:
    case_id = str(record.get("case_id") or "")
    if case_id.startswith("tests/testset/"):
        return case_id
    file_name = record.get("file_name")
    if file_name:
        return f"tests/testset/mcif_241130_no2186/{file_name}"
    return case_id


def _result(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    return result if isinstance(result, dict) else record


def _ferroelectric(record: dict[str, Any]) -> dict[str, Any]:
    payload = _result(record).get("ferroelectric_switching")
    return payload if isinstance(payload, dict) else {}


def _domain_screen(record: dict[str, Any]) -> dict[str, Any]:
    payload = _ferroelectric(record).get("domain_reversal_symmetry_screening")
    return payload if isinstance(payload, dict) else {}


def _group_identifiers(record: dict[str, Any]) -> dict[str, Any]:
    result = _result(record)
    group = result.get("group_identifiers") or record.get("group_identifiers")
    return group if isinstance(group, dict) else {}


def _material_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[1]


def _axis_labels(space_group_number: int | None) -> list[str] | None:
    return space_group_polar_axis_labels(space_group_number)


def _category(record: dict[str, Any]) -> str:
    fe = _ferroelectric(record)
    screen = _domain_screen(record)
    screen_status = screen.get("status")
    polarity_status = fe.get("polarity_status")
    left_coset_count = screen.get("left_coset_count")
    p_reversal_count = screen.get("candidate_reversal_domain_count") or 0

    if screen_status == "candidate_reversal_domains_found" and p_reversal_count:
        return "parent_ordered_coset_p_reversal_candidate"
    if (
        isinstance(left_coset_count, int)
        and left_coset_count > 1
        and screen_status == "no_parent_ordered_coset_maps_p_to_minus_p"
    ):
        return "nontrivial_parent_ordered_coset_no_p_reversal_control"
    if polarity_status == "parent_polar_axis_preserved" or (
        isinstance(left_coset_count, int)
        and left_coset_count == 1
        and polarity_status
        in {
            "parent_polar_axis_preserved",
            "parent_polar_ordered_polar_transport_required",
        }
    ):
        return "current_polar_or_trivial_coset_control"
    if polarity_status == "ordered_symmetry_nonpolar":
        return "ordered_nonpolar_control"

    group = _group_identifiers(record)
    sg_is_polar = group.get("sg_is_polar")
    ossg_is_polar = group.get("ossg_is_polar")
    msg_is_polar = group.get("msg_is_polar")
    sg_num = group.get("sg_num")
    ossg_num = group.get("ossg_space_group_number")

    if sg_is_polar is False and ossg_is_polar is True and msg_is_polar is True:
        return "nonpolar_parent_to_polar_ordered__msg_polar"
    if sg_is_polar is False and ossg_is_polar is True and msg_is_polar is False:
        return "nonpolar_parent_to_polar_ordered__msg_nonpolar"
    if sg_is_polar is True and ossg_is_polar is True:
        if sg_num == ossg_num and _axis_labels(sg_num) == _axis_labels(ossg_num):
            return "polar_parent_axis_preserved_control"
        return "polar_parent_ordered_polar__axis_transport_required"
    if sg_is_polar is False and ossg_is_polar is False:
        return "nonpolar_parent_ordered_nonpolar_control"
    return "other_or_incomplete_symmetry"


def _record_row(record: dict[str, Any], category: str) -> dict[str, Any]:
    result = _result(record)
    group = _group_identifiers(record)
    fe = _ferroelectric(record)
    screen = _domain_screen(record)
    file_name = str(record.get("file_name") or Path(_case_path(record)).name)
    sg_num = (
        screen.get("parent_space_group_number")
        or (fe.get("source_parent_space_group") or {}).get("space_group_number")
        or group.get("sg_num")
    )
    ossg_num = (
        screen.get("ordered_space_group_number")
        or group.get("ossg_space_group_number")
        or result.get("G0_num")
    )
    msg_parent = group.get("msg_parent_space_group_number")
    candidates = screen.get("candidate_reversal_domains") or []
    first_candidate = candidates[0] if candidates else {}
    representative = first_candidate.get("representative") or {}
    return {
        "category": category,
        "material": _material_name(file_name),
        "file": _case_path(record),
        "index": group.get("index") or result.get("index"),
        "phase": result.get("phase") or group.get("phase"),
        "polarity_status": fe.get("polarity_status"),
        "domain_status": screen.get("status") or fe.get("candidate_reversal_domain_status"),
        "sg": sg_num,
        "sg_axes": _axis_labels(sg_num),
        "ossg": ossg_num,
        "ossg_axes": _axis_labels(ossg_num),
        "msg_parent": msg_parent,
        "msg_parent_axes": _axis_labels(msg_parent),
        "msg_symbol": group.get("msg_symbol"),
        "left_cosets": screen.get("left_coset_count"),
        "p_reversal_count": screen.get("candidate_reversal_domain_count"),
        "first_reversal_op": representative.get("xyzt"),
        "first_reversal_axes": first_candidate.get("reversed_polar_axes"),
        "first_reversal_msg_compatible": first_candidate.get("msg_compatible"),
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("status") == "ok":
                records.append(record)
    return records


def _select(records: list[dict[str, Any]], max_per_category: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = _category(record)
        if category == "other_or_incomplete_symmetry":
            continue
        buckets[category].append(_record_row(record, category))

    priority = [
        "parent_ordered_coset_p_reversal_candidate",
        "nontrivial_parent_ordered_coset_no_p_reversal_control",
        "current_polar_or_trivial_coset_control",
        "ordered_nonpolar_control",
        "nonpolar_parent_to_polar_ordered__msg_polar",
        "nonpolar_parent_to_polar_ordered__msg_nonpolar",
        "polar_parent_ordered_polar__axis_transport_required",
        "polar_parent_axis_preserved_control",
        "nonpolar_parent_ordered_nonpolar_control",
    ]
    selected = []
    for category in priority:
        selected.extend(buckets.get(category, [])[:max_per_category])
    return selected


def _write_manifest(path: Path, selected: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representative 3D ferroelectric-switching diagnostic cases.",
        "# Generated by scripts/select_ferroelectric_switching_cases.py.",
    ]
    for row in selected:
        case_path = (REPO_ROOT / row["file"]).resolve()
        lines.append(os.path.relpath(case_path, path.parent.resolve()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, selected: list[dict[str, Any]], records_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        records_label = records_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        records_label = records_path.as_posix()
    lines = [
        "# Ferroelectric Switching 3D Case Selection",
        "",
        f"Source records: `{records_label}`",
        "",
        "This small set is selected from successful 3D batch records to cover the",
        "SG(parent/current nonmagnetic) / OSSG_real(ordered) coset relations used",
        "by the ferroelectric-switching payload. It is a review and regression aid,",
        "not a replacement for the full 2185-case batch.",
        "",
        "| Category | Material | File | Index | Phase | Polarity | Domain screen | Parent SG axes | Ordered SG axes | Cosets | P-reversal ops | First reversal op | Reversed axes | MSG-compatible |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected:
        lines.append(
            "| {category} | {material} | `{file}` | `{index}` | {phase} | "
            "{polarity_status} | {domain_status} | {sg} {sg_axes} | "
            "{ossg} {ossg_axes} | {left_cosets} | {p_reversal_count} | "
            "`{first_reversal_op}` | {first_reversal_axes} | "
            "{first_reversal_msg_compatible} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-per-category", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = _load_records(args.records)
    selected = _select(records, max_per_category=args.max_per_category)
    _write_manifest(args.manifest, selected)
    _write_report(args.report, selected, args.records)
    print(f"selected={len(selected)}")
    print(f"manifest={args.manifest}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
