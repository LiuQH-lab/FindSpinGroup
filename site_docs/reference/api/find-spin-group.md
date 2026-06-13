# `find_spin_group`

Full analysis route.

Use this function when you need the full `MagSymmetryResult`, generated
artifacts, operation payloads, tensor outputs, quasi-2D diagnostics, or route
audits.

## Signature

```python
find_spin_group(
    cif: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    parser_atol: float = 0.02,
    calculation_mode: str | None = "3d",
    vacuum_axis: str | None = "c",
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
    spin_texture_basis_max_order: int | None = None,
) -> MagSymmetryResult
```

## Parameters

`cif`
Path to the input structure file.

`space_tol`, `mtol`, `meigtol`, `matrix_tol`, `parser_atol`
Tolerance controls. See
[find_spin_group_basic](find-spin-group-basic.md#parameters) for their basic
roles.

`calculation_mode`
Controls additive quasi-2D diagnostics. The default `"3d"` runs ordinary 3D
identification. Values such as `"quasi2d"`, `"2d"`, `"slab"`, or `"layer"`
request quasi-2D interpretation data in addition to the base 3D result.

`vacuum_axis`
Input-cell axis normal to the intended slab plane. This parameter is interpreted
only when quasi-2D diagnostics are requested.

`poscar_allow_incar_magmom`, `poscar_prefer_incar_magmom`
POSCAR / INCAR magnetic-moment controls.

`spin_texture_basis_max_order`
When set, include `basis_by_order` entries for computed spin-texture
configuration fields from order 0 through this order. The default `None` emits
only the leading allowed basis.

## Returns

Returns a [MagSymmetryResult](../magsymmetry-result.md) object.

Read it through these accessors:

```python
result.to_summary_dict()      # SummaryResult
result.to_structured_dict()   # StructuredResult
result.to_scif(...)           # SCIF text
result.to_dict()              # raw compatibility dictionary
```

## Returned Fields

### `result.to_summary_dict()`

Returns [SummaryResult](../result-schemas/summary-result.md).

Use it for compact display after running the full route.

Important fields:

`index`
Final identified OSSG index.

`phase`
Magnetic phase classification.

`acc`
Spin arithmetic crystal class.

`properties`
Compact physical-property summary.

`gspg`
Compact GSPG symbol and operation summary.

### `result.to_structured_dict()`

Returns [StructuredResult](../result-schemas/structured-result.md).

Use it for full programmatic integrations.

Top-level fields:

`summary`
High-level identifiers, phase fields, spin-texture fields, tolerances, and
source metadata.

`groups`
Input SG, G0, L0, OSSG, MSG, SSG operation payloads, MSG operation payloads,
and little-group outputs.

For operation list items, read
[SSGOperation](../result-schemas/payload-definitions.md#ssgoperation) and
[MSGOperation](../result-schemas/payload-definitions.md#msgoperation).

`cells`
Input, input magnetic primitive, database standard, convention, ACC primitive,
and ACC conventional cell payloads.

For the nested cell object shape, read
[CellPayload](../result-schemas/payload-definitions.md#cellpayload).

`transforms`
Setting transforms and route audits.

For transform fields, read
[TransformPayload](../result-schemas/payload-definitions.md#transformpayload).

`properties`
Magnetic phase, spin splitting, AHC, tensors, magnetic site, quasi-2D, vector
constraints, and ferroelectric-switching outputs.

`artifacts`
Generated POSCAR, SCIF, and KPOINTS text.

### Direct attributes

Common direct attributes include `index`,
`convention_ssg_international_linear`, `magnetic_phase`, `msg_symbol`,
`msg_bns_number`, `scif`, `acc_primitive_magnetic_cell_poscar`, `KPOINTS`, and
`quasi_2d`.

## Example

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

print(result.index)
print(result.convention_ssg_international_linear)
print(result.magnetic_phase)

summary = result.to_summary_dict()
structured = result.to_structured_dict()
```

## Notes

`to_structured_dict()` is the recommended complete output for new integrations.
`to_dict()` exposes the raw compatibility surface and may include legacy names
or diagnostic details that are not ideal as a new public contract.
