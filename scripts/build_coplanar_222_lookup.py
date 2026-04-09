from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / "SSG_磁维度分辨的所有数据_v0924.xlsx"
OUTPUT_TXT_PATH = ROOT / "src" / "findspingroup" / "core" / "identify-py-222" / "output" / "output.txt"
JSON_PATH = ROOT / "src" / "findspingroup" / "core" / "identify_index" / "data" / "coplanar_222_lookup_v0924.json"

SHEET_CONFIGS = {
    "T1 Coplanar": {"spin_col": 48, "index_col": 52},
    "T2 Coplanar": {"spin_col": 48, "index_col": 51},
    "T3 Coplanar": {"spin_col": 58, "index_col": 61},
}


def _normalize_triplet(values: list[int] | tuple[int, ...]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def _parse_int_triplet(text: str) -> list[int]:
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list-like triplet, got {text!r}")
    return [int(item) for item in value]


def _parse_flat_matrix(text: str) -> list[list[int]]:
    numbers = [int(value) for value in re.findall(r"-?\d+", text)]
    if len(numbers) != 9:
        raise ValueError(f"Expected 9 integers in spin-only matrix, got {numbers!r}")
    return [numbers[0:3], numbers[3:6], numbers[6:9]]


def _direction_from_mirror(matrix: list[list[int]]) -> list[int]:
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    for idx, value in enumerate(eigenvalues):
        if np.isclose(value, -1.0, atol=1e-8):
            direction = np.asarray(eigenvectors[:, idx].real, dtype=float)
            break
    else:
        raise ValueError(f"Cannot find -1 eigenvector for mirror matrix {matrix!r}")

    direction /= np.linalg.norm(direction)
    snapped = np.round(direction).astype(int)
    if not np.all(np.isin(snapped, (-1, 0, 1))):
        raise ValueError(f"Cannot snap mirror direction {direction.tolist()} to axis")
    if np.count_nonzero(snapped) != 1:
        raise ValueError(f"Expected axis-aligned mirror direction, got {snapped.tolist()}")
    return snapped.tolist()


def _parse_output_mapping() -> dict[tuple[str, str, int], int]:
    blocks = [block.strip() for block in OUTPUT_TXT_PATH.read_text(encoding="utf-8").split("****") if block.strip()]
    mapping: dict[tuple[str, str, int], int] = {}
    for block in blocks:
        normalized = " ".join(line.strip() for line in block.splitlines())
        parts = normalized.split(";", 5)
        if len(parts) < 5:
            raise ValueError(f"Cannot parse output block: {block!r}")
        lg = parts[0].replace(" ", "")
        ttk = parts[1].replace(" ", "")
        map_id = int(parts[2])
        old_map_num = int(parts[4])
        key = (lg, ttk, map_id)
        if key in mapping:
            raise ValueError(f"Duplicate output mapping for {key!r}")
        mapping[key] = old_map_num
    return mapping


def build_payload() -> dict[str, object]:
    wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True, data_only=True)
    output_mapping = _parse_output_mapping()

    by_old_map_num: dict[str, dict[str, object]] = {}
    source_rows: list[dict[str, object]] = []
    groups: set[str] = set()
    for sheet_name, sheet_cfg in SHEET_CONFIGS.items():
        ws = wb[sheet_name]
        current_lg = None
        current_ttk = None
        for row_index, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            if row[1] is not None:
                current_lg = str(row[1]).replace(" ", "")
            if row[2] is not None:
                current_ttk = str(row[2]).replace(" ", "")
            if current_lg is None or current_ttk is None:
                continue
            if str(row[5]).strip() != "222":
                continue

            map_id = int(row[3])
            lookup_key = (current_lg, current_ttk, map_id)
            if lookup_key not in output_mapping:
                raise ValueError(f"Missing identify-py-222 output mapping for {lookup_key!r}")

            old_map_num = output_mapping[lookup_key]
            lg = _parse_int_triplet(current_lg)
            ttk = _parse_int_triplet(current_ttk)
            spin_only_matrix = _parse_flat_matrix(str(row[sheet_cfg["spin_col"] - 1]))
            spin_only_direction = _direction_from_mirror(spin_only_matrix)
            final_index = str(row[sheet_cfg["index_col"] - 1]).strip()
            payload_key = f"{current_lg}|{current_ttk}|{old_map_num}"
            groups.add(f"{current_lg}|{current_ttk}")

            entry = {
                "sheet_name": sheet_name,
                "lg": lg,
                "ttk": ttk,
                "map_id": map_id,
                "old_map_num": old_map_num,
                "point_group_id": int(row[4]),
                "spin_only_matrix": spin_only_matrix,
                "spin_only_direction": spin_only_direction,
                "final_index": final_index,
                "configuration_suffix": final_index.split(".")[-1],
                "source_excel_row": row_index,
            }
            if payload_key in by_old_map_num:
                raise ValueError(f"Duplicate (lg, ttk, old_map_num) lookup key {payload_key!r}")
            by_old_map_num[payload_key] = entry
            source_rows.append(entry)

    suffix_counts: dict[str, int] = defaultdict(int)
    for entry in source_rows:
        suffix_counts[str(entry["configuration_suffix"])] += 1

    return {
        "source_excel": EXCEL_PATH.name,
        "source_sheets": list(SHEET_CONFIGS.keys()),
        "filter": {"F": 222},
        "row_count": len(source_rows),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "groups": sorted(groups),
        "by_old_map_num": by_old_map_num,
    }


def main() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(JSON_PATH)
    print(f"rows={payload['row_count']}")


if __name__ == "__main__":
    main()
