from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from findspingroup import batch_mcif
from findspingroup.find_spin_group import find_spin_group
from findspingroup.structure.group import SpinSpaceGroup


def _stringify(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _compact_wp_chain(wp_chain: Any, *, limit: int = 24) -> str | None:
    if not wp_chain:
        return None
    items: list[str] = []
    for row in wp_chain[:limit]:
        try:
            element, wp_sg, idx_sg, wp_ssg, idx_ssg, wp_msg, idx_msg = row
            items.append(
                f"{element}:{wp_sg}[{idx_sg}]->{wp_ssg}[{idx_ssg}]->{wp_msg}[{idx_msg}]"
            )
        except Exception:
            items.append(str(row))
    if len(wp_chain) > limit:
        items.append(f"...(+{len(wp_chain) - limit} more)")
    return " | ".join(items)


def _row_from_result(file_path: Path, result) -> dict[str, Any]:
    identify = result.identify_index_details or {}
    primitive_ssg = SpinSpaceGroup(result.primitive_magnetic_cell_ssg_ops)

    return {
        "case_id": batch_mcif._normalize_case_id(file_path),
        "file_name": file_path.name,
        "status": "ok",
        "index": result.index,
        "conf": result.conf,
        "phase": result.magnetic_phase,
        "acc": result.acc,
        "msg_acc": result.msg_acc,
        "G0_id": identify.get("G0_id"),
        "L0_id": identify.get("L0_id"),
        "t_index": identify.get("t_index"),
        "k_index": identify.get("k_index"),
        "nsspg_hm": primitive_ssg.n_spin_part_point_group_symbol_hm,
        "nsspg_symbol": primitive_ssg.n_spin_part_point_group_symbol_s,
        "sspg_hm": primitive_ssg.spin_part_point_group_symbol_hm,
        "sspg_symbol": primitive_ssg.spin_part_point_group_symbol_s,
        "ssg_type": result.primitive_magnetic_cell_ssg_type,
        "spin_only_direction": result.convention_spin_only_direction,
        "ossg_symbol": result.convention_ssg_international_linear,
        "primitive_ssg_symbol": result.primitive_magnetic_cell_ssg_international_linear,
        "sg_symbol": result.input_space_group_symbol,
        "sg_num": result.input_space_group_number,
        "sg_has_real_space_inversion": result.sg_has_real_space_inversion,
        "sg_is_polar": result.sg_is_polar,
        "sg_is_chiral": result.sg_is_chiral,
        "ossg_space_group_number": result.ossg_space_group_number,
        "ossg_has_real_space_inversion": result.ossg_has_real_space_inversion,
        "ossg_is_polar": result.ossg_is_polar,
        "ossg_is_chiral": result.ossg_is_chiral,
        "msg_symbol": result.msg_symbol,
        "msg_num": result.msg_num,
        "msg_type": result.msg_type,
        "msg_bns_number": result.msg_bns_number,
        "msg_og_number": result.msg_og_number,
        "msg_parent_space_group_number": result.msg_parent_space_group_number,
        "msg_has_real_space_inversion": result.msg_has_real_space_inversion,
        "msg_is_polar": result.msg_is_polar,
        "msg_is_chiral": result.msg_is_chiral,
        "wyckoff_split": _compact_wp_chain(result.wp_chain),
    }


def _row_from_serialized_result_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("result") or {}
    identify = payload.get("identify_index_details") or {}
    primitive_ops = payload.get("primitive_magnetic_cell_ssg_ops") or []
    primitive_ssg = SpinSpaceGroup(primitive_ops) if primitive_ops else None

    return {
        "case_id": record.get("case_id"),
        "file_name": record.get("file_name"),
        "status": record.get("status", "ok"),
        "index": payload.get("index"),
        "conf": payload.get("conf"),
        "phase": payload.get("phase"),
        "acc": payload.get("acc"),
        "msg_acc": payload.get("msg_acc"),
        "G0_id": identify.get("G0_id"),
        "L0_id": identify.get("L0_id"),
        "t_index": identify.get("t_index"),
        "k_index": identify.get("k_index"),
        "nsspg_hm": (
            primitive_ssg.n_spin_part_point_group_symbol_hm if primitive_ssg is not None else None
        ),
        "nsspg_symbol": (
            primitive_ssg.n_spin_part_point_group_symbol_s if primitive_ssg is not None else None
        ),
        "sspg_hm": primitive_ssg.spin_part_point_group_symbol_hm if primitive_ssg is not None else None,
        "sspg_symbol": primitive_ssg.spin_part_point_group_symbol_s if primitive_ssg is not None else None,
        "ssg_type": payload.get("primitive_magnetic_cell_ssg_type"),
        "spin_only_direction": payload.get("convention_spin_only_direction"),
        "ossg_symbol": payload.get("convention_ssg_international_linear"),
        "primitive_ssg_symbol": payload.get("primitive_magnetic_cell_ssg_international_linear"),
        "sg_symbol": payload.get("input_space_group_symbol"),
        "sg_num": payload.get("input_space_group_number"),
        "sg_has_real_space_inversion": payload.get("sg_has_real_space_inversion"),
        "sg_is_polar": payload.get("sg_is_polar"),
        "sg_is_chiral": payload.get("sg_is_chiral"),
        "ossg_space_group_number": payload.get("ossg_space_group_number"),
        "ossg_has_real_space_inversion": payload.get("ossg_has_real_space_inversion"),
        "ossg_is_polar": payload.get("ossg_is_polar"),
        "ossg_is_chiral": payload.get("ossg_is_chiral"),
        "msg_symbol": payload.get("msg_symbol"),
        "msg_num": payload.get("msg_num"),
        "msg_type": payload.get("msg_type"),
        "msg_bns_number": payload.get("msg_bns_number"),
        "msg_og_number": payload.get("msg_og_number"),
        "msg_parent_space_group_number": payload.get("msg_parent_space_group_number"),
        "msg_has_real_space_inversion": payload.get("msg_has_real_space_inversion"),
        "msg_is_polar": payload.get("msg_is_polar"),
        "msg_is_chiral": payload.get("msg_is_chiral"),
        "wyckoff_split": _compact_wp_chain(payload.get("wp_chain")),
        "error_type": record.get("error", {}).get("type"),
        "error_message": record.get("error", {}).get("message"),
    }


def _row_from_error(file_path: Path, exc: Exception) -> dict[str, Any]:
    return {
        "case_id": batch_mcif._normalize_case_id(file_path),
        "file_name": file_path.name,
        "status": "error",
        "index": None,
        "conf": None,
        "phase": None,
        "acc": None,
        "msg_acc": None,
        "G0_id": None,
        "L0_id": None,
        "t_index": None,
        "k_index": None,
        "nsspg_hm": None,
        "nsspg_symbol": None,
        "sspg_hm": None,
        "sspg_symbol": None,
        "ssg_type": None,
        "spin_only_direction": None,
        "ossg_symbol": None,
        "primitive_ssg_symbol": None,
        "sg_symbol": None,
        "sg_num": None,
        "sg_has_real_space_inversion": None,
        "sg_is_polar": None,
        "sg_is_chiral": None,
        "ossg_space_group_number": None,
        "ossg_has_real_space_inversion": None,
        "ossg_is_polar": None,
        "ossg_is_chiral": None,
        "msg_symbol": None,
        "msg_num": None,
        "msg_type": None,
        "msg_bns_number": None,
        "msg_og_number": None,
        "msg_parent_space_group_number": None,
        "msg_has_real_space_inversion": None,
        "msg_is_polar": None,
        "msg_is_chiral": None,
        "wyckoff_split": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


COLUMNS = [
    "case_id",
    "file_name",
    "status",
    "index",
    "conf",
    "phase",
    "acc",
    "msg_acc",
    "G0_id",
    "L0_id",
    "t_index",
    "k_index",
    "nsspg_hm",
    "nsspg_symbol",
    "sspg_hm",
    "sspg_symbol",
    "ssg_type",
    "spin_only_direction",
    "ossg_symbol",
    "primitive_ssg_symbol",
    "sg_symbol",
    "sg_num",
    "sg_has_real_space_inversion",
    "sg_is_polar",
    "sg_is_chiral",
    "ossg_space_group_number",
    "ossg_has_real_space_inversion",
    "ossg_is_polar",
    "ossg_is_chiral",
    "msg_symbol",
    "msg_num",
    "msg_type",
    "msg_bns_number",
    "msg_og_number",
    "msg_parent_space_group_number",
    "msg_has_real_space_inversion",
    "msg_is_polar",
    "msg_is_chiral",
    "wyckoff_split",
    "error_type",
    "error_message",
]


def _write_workbook(rows: list[dict[str, Any]], output_xlsx: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "records"
    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in rows:
        ws.append([_stringify(row.get(column)) for column in COLUMNS])

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for index, column in enumerate(COLUMNS, start=1):
        max_len = len(column)
        for row in ws.iter_rows(min_row=2, min_col=index, max_col=index):
            value = row[0].value
            if value is None:
                continue
            max_len = max(max_len, len(str(value)))
        ws.column_dimensions[get_column_letter(index)].width = min(max_len + 2, 60)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_xlsx)


def _write_jsonl(rows: list[dict[str, Any]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run find_spin_group over mcif files and export compact Excel rows.")
    parser.add_argument("inputs", nargs="*", help="Input .mcif files or directories")
    parser.add_argument("--runtime-jsonl", type=Path, help="Read rows from batch full_results.jsonl instead of re-running.")
    parser.add_argument("--output-xlsx", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--space-tol", type=float, default=0.02)
    parser.add_argument("--mtol", type=float, default=0.02)
    parser.add_argument("--meigtol", type=float, default=0.00002)
    parser.add_argument("--matrix-tol", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_xlsx is None and args.output_jsonl is None:
        raise ValueError("Provide at least one of --output-xlsx or --output-jsonl.")
    rows: list[dict[str, Any]] = []
    if args.runtime_jsonl is not None:
        records = [
            json.loads(line)
            for line in args.runtime_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit is not None:
            records = records[: args.limit]
        for index, record in enumerate(records, start=1):
            rows.append(_row_from_serialized_result_record(record))
            print(f"[{index}/{len(records)}] {record.get('status', 'ok').upper():5} {record.get('file_name')}")
    else:
        if not args.inputs:
            raise ValueError("Provide input files/directories unless --runtime-jsonl is used.")
        files = batch_mcif._discover_mcif_files(args.inputs, recursive=not args.non_recursive)
        files = batch_mcif._dedupe_sorted(files)
        if args.limit is not None:
            files = files[: args.limit]

        for index, file_path in enumerate(files, start=1):
            try:
                result = find_spin_group(
                    str(file_path),
                    space_tol=args.space_tol,
                    mtol=args.mtol,
                    meigtol=args.meigtol,
                    matrix_tol=args.matrix_tol,
                )
                rows.append(_row_from_result(file_path, result))
                print(f"[{index}/{len(files)}] OK    {file_path.name} -> {result.index}")
            except Exception as exc:
                rows.append(_row_from_error(file_path, exc))
                print(f"[{index}/{len(files)}] ERROR {file_path.name} -> {type(exc).__name__}: {exc}")

    if args.output_jsonl is not None:
        _write_jsonl(rows, args.output_jsonl)
        print(args.output_jsonl)
    if args.output_xlsx is not None:
        _write_workbook(rows, args.output_xlsx)
        print(args.output_xlsx)


if __name__ == "__main__":
    main()
