from findspingroup.core.identify_index.databases.query_ssg_map import find_ssg_map


def test_identify_20260501_ssg_map_fixes_126_7_1_3_4_type1_partner_order():
    records = find_ssg_map(7, 126, 8, 1, 30)
    mapping = {
        tuple(tuple(item) for item in record["all_maps"]): record["old_num"]
        for record in records
    }

    assert mapping[((1, 1),)] == 4
    assert mapping[((5, 1),)] == 3
