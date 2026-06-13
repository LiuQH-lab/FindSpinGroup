# Payload Definitions

These reusable payloads appear inside `BasicResult`, `SummaryResult`,
`StructuredResult`, and `InputSSGResult`.

Use this page when a schema says a field is an `SSGPayload`, `MSGOperation`,
`CellPayload`, or another nested object.

## `SSGPayload`

Spin space group operations in one named cell setting.

```python
{
    "setting": str,
    "spin_frame_setting": str | None,
    "ops": list[SSGOperation] | None,
    "seitz": list[str] | None,
    "seitz_latex": list[str] | None,
    "seitz_descriptions": list[SeitzDescription] | None,
    "international_linear": str | None,
    "international_latex": str | None,
    "symbol_calibration_tol": float | None,
    "type": object | None,
}
```

`setting`
Real-space cell setting used by the operation matrices and translations.

`spin_frame_setting`
Spin-frame setting used by `spin_rotation`. `None` means the payload does not
need a separate spin-frame label or the value was not recorded for that route.

`ops`
One-based list of spin space group operation matrices. See `SSGOperation`.

`seitz`
Linear Seitz-style symbols for the same operations in `ops`.

`seitz_latex`
LaTeX Seitz-style symbols for the same operations in `ops`.

`seitz_descriptions`
Structured display metadata behind the Seitz symbols. See `SeitzDescription`.

`international_linear`
Linear international-style SSG symbol for this payload when available.

`international_latex`
LaTeX version of `international_linear` when available.

`symbol_calibration_tol`
Tolerance used while formatting symbolic operation labels.

`type`
SSG type or route-specific type label when available.

## `SSGOperation`

One spin space group operation.

```python
{
    "index": int,
    "spin_rotation": list[list[float]],
    "real_rotation": list[list[float]],
    "translation": list[float],
}
```

`index`
One-based operation index. The same index aligns with `seitz`,
`seitz_latex`, and `seitz_descriptions` when those lists are present.

`spin_rotation`
`3 x 3` spin-space rotation matrix in the payload's spin frame.

`real_rotation`
`3 x 3` real-space rotation matrix in the payload's real-space setting.

`translation`
Length-3 translation vector in fractional coordinates of the payload's
real-space setting.

## `MSGPayload`

Magnetic space group operations in one named cell setting.

```python
{
    "setting": str,
    "spin_frame_setting": str | None,
    "ops": list[MSGOperation] | None,
}
```

`setting`
Real-space cell setting used by the MSG operation matrices and translations.

`spin_frame_setting`
Spin-frame label used by the route that produced the oriented MSG payload.

`ops`
One-based list of magnetic space group operations. See `MSGOperation`.

## `MSGOperation`

One magnetic space group operation.

```python
{
    "index": int,
    "time_reversal": int,
    "real_rotation": list[list[float]],
    "translation": list[float],
}
```

`index`
One-based operation index.

`time_reversal`
`0` for an ordinary operation and `1` for an anti-time-reversal operation.

`real_rotation`
`3 x 3` real-space rotation matrix in the payload's real-space setting.

`translation`
Length-3 translation vector in fractional coordinates of the payload's
real-space setting.

## `SeitzDescription`

Structured description used to build the operation display symbols.

```python
{
    "spin": PointOperationDescription,
    "real": PointOperationDescription,
    "translation_symbol": str,
    "translation_symbol_latex": str,
    "symbol": str,
    "symbol_latex": str,
}
```

`spin`
Point-operation description for the spin part.

`real`
Point-operation description for the real-space part.

`translation_symbol`
Linear translation label.

`translation_symbol_latex`
LaTeX translation label.

`symbol`
Full linear SSG operation symbol, combining spin, real-space, and translation
parts.

`symbol_latex`
LaTeX version of `symbol`.

## `PointOperationDescription`

Point-operation metadata inside `SeitzDescription.spin` and
`SeitzDescription.real`.

```python
{
    "hm_symbol": str,
    "rotation_power": int | None,
    "axis_vector": tuple[float, float, float] | None,
    "axis_kind": str | None,
    "axis_direction": tuple[int, int, int] | None,
    "axis_parameter_values": tuple[float, float, float] | None,
    "axis_subscript_linear": str | None,
    "axis_subscript_latex": str | None,
    "symbol": str,
    "symbol_latex": str,
}
```

`hm_symbol`
Hermann-Mauguin-like point-operation token, such as `1`, `-1`, `2`, `m`, or a
higher-fold rotation token.

`rotation_power`
Power applied to the operation token. `None` means no explicit power is needed.

`axis_vector`
Normalized axis vector used for formatting when an axis exists.

`axis_kind`
How the axis was formatted. Common values are `direction`, `symbolic`, and
`parameter`.

`axis_direction`
Integer direction label when `axis_kind == "direction"`.

`axis_parameter_values`
Numeric axis parameters when the axis is not represented as a simple integer
direction.

`axis_subscript_linear`
Linear axis subscript used in the display symbol.

`axis_subscript_latex`
LaTeX axis subscript used in the display symbol.

`symbol`
Linear point-operation symbol after canonical axis formatting.

`symbol_latex`
LaTeX point-operation symbol after canonical axis formatting.

## `CellPayload`

Named cell payloads appear under `StructuredResult["cells"]`.

```python
{
    "setting": str | None,
    "cell": object | None,
    "detail": dict | None,
    "wp_chain": object | None,
}
```

`setting`
Cell-setting label.

`cell`
Serialized cell object for that setting. It contains the lattice, positions,
species, occupancies, and magnetic moments as produced by the route.

`detail`
Diagnostic and display metadata for the cell construction.

`wp_chain`
Wyckoff-position chain information when that route produced it.

Some cell entries use a smaller shape. For example, `cells.input` contains
`detail` and `wp_chain`, while `cells.database_standard` contains `selected`,
`cell`, `g0_standard`, `l0_standard`, and `wp_chain`.

## `TransformPayload`

Named transforms appear under `StructuredResult["transforms"]` and
`InputSSGResult["primitive_relation"]`.

Most public transform fields are matrix-like objects or serialized nested lists
that map one named cell setting to another. The field name gives the direction,
for example `input_to_acc_primitive` or `convention_to_input`.

`raw`
Raw standard-setting transforms before public cleanup.

`audit`
Diagnostic route-audit payloads. These are useful when debugging setting
selection or transform construction, but they are not the recommended first
fields for ordinary result display.

## `SpinPolarizationPayload`

Spin-polarization outputs appear under `properties.spin_splitting` and
`groups.little_groups`.

```python
{
    "values": object,
    "setting": str | None,
    "real_space_setting": str | None,
    "spin_frame": str | None,
    "acc_cartesian": object | None,
    "acc_poscar_spin_frame": object | None,
}
```

`values`
Spin-polarization values produced by the route.

`setting`
Cell setting for the reported polarization values.

`real_space_setting`
Real-space setting associated with the values.

`spin_frame`
Spin-frame label associated with the values.

`acc_cartesian`
ACC Cartesian-frame representation when available.

`acc_poscar_spin_frame`
ACC POSCAR spin-frame representation when available.
