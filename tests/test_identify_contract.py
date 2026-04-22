import json
import sqlite3

from findspingroup.core.identify_index.contract_222 import (
    COPLANAR_222_MAP_NUMBER_CONTRACT,
    load_coplanar_222_lookup,
)
import findspingroup.core.identify_index.databases.query_ssg_map as ssg_map_query
import findspingroup.core.identify_index.databases.query_ssg_reduce as ssg_reduce_query


def test_coplanar_222_lookup_uses_fixed_map_number_contract():
    payload = load_coplanar_222_lookup()

    assert payload["map_number_contract"] == COPLANAR_222_MAP_NUMBER_CONTRACT
    assert "by_map_num" in payload
    assert "by_old_map_num" not in payload


def test_coplanar_222_lookup_map_num_directly_matches_manual_checked_rows():
    rows = load_coplanar_222_lookup()["by_map_num"]

    assert rows["[1,14]|[4,4,1]|3"]["final_index"] == "14.1.1.1.P3"
    assert rows["[1,33]|[4,4,1]|2"]["final_index"] == "33.1.1.1.P2"
    assert rows["[13,63]|[4,2,2]|9"]["final_index"] == "63.13.2.21.P3"


def _create_ssg_map_db(path, *, include_record: bool) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE ssg_map (
            id INTEGER,
            L0_id INTEGER,
            G0_id INTEGER,
            it INTEGER,
            ik INTEGER,
            num INTEGER,
            isonum INTEGER,
            transformation_matrix TEXT,
            all_maps TEXT,
            transformation_maps TEXT,
            old_num TEXT,
            old_trans_1 TEXT,
            old_trans_2 TEXT
        )
        """
    )
    if include_record:
        conn.execute(
            "INSERT INTO ssg_map VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                13,
                63,
                2,
                2,
                1,
                14,
                json.dumps([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]]),
                json.dumps([[1, 1]]),
                json.dumps([]),
                json.dumps(1),
                json.dumps([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                json.dumps([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ),
        )
    conn.commit()
    conn.close()


def _create_ssg_reduce_db(path, *, include_record: bool) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE ssg_reduce (
            id INTEGER,
            L0_id INTEGER,
            G0_id INTEGER,
            it INTEGER,
            ik INTEGER,
            cell_size INTEGER,
            isonum TEXT,
            transformation_matrix TEXT,
            gen_matrix TEXT,
            TTM TEXT
        )
        """
    )
    if include_record:
        conn.execute(
            "INSERT INTO ssg_reduce VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                13,
                63,
                2,
                2,
                1,
                json.dumps([14]),
                json.dumps([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]]),
                json.dumps([]),
                json.dumps([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            ),
        )
    conn.commit()
    conn.close()


def test_coplanar_222_database_queries_do_not_fallback_to_legacy_db(monkeypatch, tmp_path):
    legacy_map_db = tmp_path / "legacy_map.db"
    contract_map_db = tmp_path / "contract_map.db"
    legacy_reduce_db = tmp_path / "legacy_reduce.db"
    contract_reduce_db = tmp_path / "contract_reduce.db"
    _create_ssg_map_db(legacy_map_db, include_record=True)
    _create_ssg_map_db(contract_map_db, include_record=False)
    _create_ssg_reduce_db(legacy_reduce_db, include_record=True)
    _create_ssg_reduce_db(contract_reduce_db, include_record=False)

    monkeypatch.setattr(ssg_map_query, "db_path", str(legacy_map_db))
    monkeypatch.setattr(ssg_map_query, "db_path_222", str(contract_map_db))
    monkeypatch.setattr(ssg_reduce_query, "db_path", str(legacy_reduce_db))
    monkeypatch.setattr(ssg_reduce_query, "db_path_222", str(contract_reduce_db))

    assert ssg_map_query.find_ssg_map(13, 63, 2, 2, 14, use_222_contract=False)
    assert ssg_reduce_query.find_ssg_reduce(13, 63, 2, 2, 14, use_222_contract=False)
    assert ssg_map_query.find_ssg_map(13, 63, 2, 2, 14, use_222_contract=True) == []
    assert ssg_reduce_query.find_ssg_reduce(13, 63, 2, 2, 14, use_222_contract=True) == []
