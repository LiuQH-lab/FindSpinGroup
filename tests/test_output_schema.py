from findspingroup.output_schema import (
    EXPORT_ROW_COLUMNS,
    LAYER_ACC_PRIMITIVE,
    LAYER_CONVENTION,
    LAYER_DATABASE_STANDARD,
    MAGNETIC_ORBIT_EXPORT_COLUMNS,
    PUBLIC_OUTPUT_LAYERS,
    QUASI2D_EXPORT_COLUMNS,
    complete_export_row,
)


def test_public_output_layer_names_are_unique_and_keep_compatibility_aliases():
    layer_names = [layer.name for layer in PUBLIC_OUTPUT_LAYERS]
    prefixes = [layer.preferred_prefix for layer in PUBLIC_OUTPUT_LAYERS]

    assert len(layer_names) == len(set(layer_names))
    assert len(prefixes) == len(set(prefixes))
    assert LAYER_ACC_PRIMITIVE.preferred_prefix == "acc_primitive"
    assert "primitive_magnetic_cell" in LAYER_ACC_PRIMITIVE.legacy_prefixes
    assert "magnetic_primitive" in LAYER_ACC_PRIMITIVE.legacy_prefixes
    assert LAYER_DATABASE_STANDARD.preferred_prefix == "database_standard"
    assert "G0std" in LAYER_DATABASE_STANDARD.legacy_prefixes
    assert LAYER_CONVENTION.preferred_prefix == "convention"


def test_export_row_schema_has_no_duplicate_columns_and_completes_rows():
    assert len(EXPORT_ROW_COLUMNS) == len(set(EXPORT_ROW_COLUMNS))
    assert len(MAGNETIC_ORBIT_EXPORT_COLUMNS) == len(set(MAGNETIC_ORBIT_EXPORT_COLUMNS))
    assert len(QUASI2D_EXPORT_COLUMNS) == len(set(QUASI2D_EXPORT_COLUMNS))
    assert "quasi2d_status" not in EXPORT_ROW_COLUMNS
    assert "quasi2d_status" in QUASI2D_EXPORT_COLUMNS

    completed = complete_export_row({"case_id": "sample", "status": "ok"})

    assert completed["case_id"] == "sample"
    assert completed["status"] == "ok"
    assert completed["magnetic_site_sg_primitive_to_magnetic_primitive_cell_expansion"] is None
    assert "magnetic_site_cell_expansion" not in completed
    for column in EXPORT_ROW_COLUMNS:
        assert column in completed
