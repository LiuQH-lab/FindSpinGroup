from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


COPLANAR_222_MAP_NUMBER_CONTRACT = "identify_core_map_num_is_0924_num_of_mapping"
COPLANAR_222_LOOKUP_FILE = "coplanar_222_lookup_v0924.json"


def _data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / COPLANAR_222_LOOKUP_FILE


@lru_cache(maxsize=1)
def load_coplanar_222_lookup() -> dict[str, Any]:
    path = _data_path()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("map_number_contract") != COPLANAR_222_MAP_NUMBER_CONTRACT:
        raise ValueError(
            f"Unexpected Coplanar+D2 lookup contract in {path}: "
            f"{payload.get('map_number_contract')!r}. Expected {COPLANAR_222_MAP_NUMBER_CONTRACT!r}."
        )
    return payload


def has_coplanar_222_lookup_group(
    lg: list[int] | tuple[int, int],
    ttk: list[int] | tuple[int, int, int],
) -> bool:
    payload = load_coplanar_222_lookup()
    group_key = f"{_normalize_triplet(lg)}|{_normalize_triplet(ttk)}"
    return group_key in payload["groups"]


def _normalize_triplet(triplet: list[int] | tuple[int, ...]) -> str:
    return "[" + ",".join(str(int(value)) for value in triplet) + "]"


def build_coplanar_222_lookup_key(
    lg: list[int] | tuple[int, int],
    ttk: list[int] | tuple[int, int, int],
    map_num: int,
) -> str:
    return f"{_normalize_triplet(lg)}|{_normalize_triplet(ttk)}|{int(map_num)}"


def get_coplanar_222_lookup_entry(
    lg: list[int] | tuple[int, int],
    ttk: list[int] | tuple[int, int, int],
    map_num: int,
) -> dict[str, Any] | None:
    payload = load_coplanar_222_lookup()
    key = build_coplanar_222_lookup_key(lg, ttk, map_num)
    return payload["by_map_num"].get(key)
