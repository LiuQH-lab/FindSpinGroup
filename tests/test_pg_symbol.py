from findspingroup.core.identify_symmetry_from_ops import (
    generate_non_crystallographic_point_groups,
    identify_point_group,
)
from findspingroup.data.PG_SYMBOL import PG_SCH_TO_ID_INDEX
from findspingroup.structure.group import _matrix_group_closure


def test_combined_schoenflies_point_group_symbols_are_available_as_aliases():
    combined_index = PG_SCH_TO_ID_INDEX["C9i=S18"]

    assert PG_SCH_TO_ID_INDEX["C9i"] == combined_index
    assert PG_SCH_TO_ID_INDEX["S18"] == combined_index


def test_identified_s18_point_group_maps_to_identify_index_id():
    generators = generate_non_crystallographic_point_groups("-9")
    operations = _matrix_group_closure(generators, tol=1e-8, limit=64)
    *_, symbol = identify_point_group(operations, _id=True)

    assert symbol == "S18"
    assert PG_SCH_TO_ID_INDEX[symbol] == PG_SCH_TO_ID_INDEX["C9i=S18"]
