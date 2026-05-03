from findspingroup.core.identify_index.databases.query_ssg_map import find_ssg_map


def test_identify_20260430_ssg_map_fixes_103_3_2_23_24_partner_order():
    records = find_ssg_map(3, 103, 4, 2, 30)
    mapping = {
        tuple(tuple(item) for item in record["all_maps"]): record["old_num"]
        for record in records
    }

    assert mapping[((1, 1),)] == 24
    assert mapping[((5, 1),)] == 23
