# SummaryResult

Returned by:

```python
result = find_spin_group("path/to/structure.mcif")
summary = result.to_summary_dict()
```

Use `SummaryResult` when you ran the full route but only need a compact,
display-friendly object.

## Shape

```python
{
    "index": str,
    "conf": str,
    "phase": str,
    "acc": str,
    "properties": BasicProperties,
    "gspg": GSPGSummary,
    "spin_texture_config_database": dict | None,
    "spin_texture_config_no_soc": dict | None,
    "spin_texture_config_soc": dict | None,
    "vector_constraints_by_symmetry": dict,
    "ferroelectric_switching": dict,
}
```

## Main Fields

`index`
Final identified OSSG index.

`conf`
Magnetic configuration class.

`phase`
Magnetic phase classification.

`acc`
Spin arithmetic crystal class.

`properties`
Physical-property summary. It uses the same compact shape documented in
[BasicResult](basic-result.md#properties).

## `gspg`

Compact GSPG symbol and operation summary.

`effective_mpg_symbol`
Effective magnetic point-group symbol.

`npg_symbol_s`
Nontrivial point-group symbol.

`output_mode`
GSPG output mode.

`point_part_linear`
Linear point-part text.

`real_space_setting`
Real-space setting used by the GSPG payload.

`spin_frame_setting`
Spin-frame setting used by the GSPG payload.

`spin_only_component_symbol_s`
Spin-only component symbol.

`spin_only_part_linear`
Linear spin-only-part text.

`spin_space_point_group_symbol_hm`
Spin-space point-group symbol in Hermann-Mauguin notation.

`spin_space_point_group_symbol_s`
Spin-space point-group symbol in Schoenflies notation.

`symbol_linear`
Linear GSPG symbol.

`symbol_mode`
Symbol mode.

`tentative_symbol_s`
Tentative Schoenflies-style symbol.

`text`
Compact GSPG text payload.

## Spin Texture And Vector Constraints

`spin_texture_config_database`
Spin-texture configuration summary loaded from the runtime database.

`spin_texture_config_no_soc`
Spin-texture configuration from the non-SOC route.

`spin_texture_config_soc`
Spin-texture configuration from the SOC-compatible route.

`vector_constraints_by_symmetry`
Vector constraints grouped by `sg`, `ossg`, and `msg`.

## Ferroelectric Switching

`ferroelectric_switching`
Symmetry-only ferroelectric-switching payload. Read `status`,
`switching_detected`, `analysis_level`, `polarity_status`, and
`allowed_polar_axes` first.
