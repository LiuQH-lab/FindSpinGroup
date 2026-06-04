# Python API

This page documents the main Python entry points. Each section is organized by
function, because users usually start from the function they call.

## `find_spin_group_basic(...)`

Use this function when you need a compact identification result.

```python
find_spin_group_basic(
    cif: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    parser_atol: float = 0.02,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> dict
```

### Parameters

`cif`
Path to the input structure file.

`space_tol`
Tolerance for real-space position and lattice matching.

`mtol`
Tolerance for magnetic-moment matching and zero-net-moment classification.

`meigtol`
Tolerance for point-group eigenvalue decisions.

`matrix_tol`
Tolerance for point-group standardization and transform matrix checks.

`parser_atol`
Tolerance used while expanding and validating CIF or SCIF input.

`poscar_allow_incar_magmom`
Allow POSCAR-like Python calls to read magnetic moments from a sibling `INCAR`.
The default is `False`.

`poscar_prefer_incar_magmom`
Prefer the sibling `INCAR` `MAGMOM` over embedded POSCAR moment data when both
are available. This is used only when `poscar_allow_incar_magmom=True`.

### Returns

Returns a JSON-serializable `dict` with the compact identification result.

Read these fields first:

`index`
Identified OSSG index.

`g0_symbol`, `g0_number`
Real-space group component of the identified SSG.

`l0_symbol`, `l0_number`
Spin-lattice group component of the identified SSG.

`it`, `ik`
Translation and k-index components of the identified SSG label.

`conf`
Magnetic configuration class.

`magnetic_phase`
High-level magnetic phase classification.

`msg_symbol`, `msg_bns_number`, `msg_og_number`
Corresponding MSG identifiers.

`properties`
Compact physical-property summary.

See [Output Fields](output-fields.md) for the full field list.

### Example

```python
from findspingroup import example_path, find_spin_group_basic

summary = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print(summary["index"])
print(summary["magnetic_phase"])
print(summary["msg_bns_number"], summary["msg_symbol"])
```

Expected output:

```text
194.164.1.1.L
AFM(Altermagnet)
63.457 Cmcm
```

## `find_spin_group(...)`

Use this function when you need the full analysis result.

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
) -> MagSymmetryResult
```

### Parameters

The tolerance and POSCAR parameters have the same meaning as in
`find_spin_group_basic(...)`.

`calculation_mode`
Controls additive quasi-2D diagnostics. The default `"3d"` runs ordinary 3D
identification. Values such as `"quasi2d"`, `"2d"`, `"slab"`, or `"layer"`
request quasi-2D interpretation data in addition to the base 3D result.

`vacuum_axis`
Input-cell axis normal to the intended slab plane. It is interpreted only for
quasi-2D diagnostics.

### Returns

Returns a `MagSymmetryResult` object.

Use these accessors first:

`result.to_summary_dict()`
Compact human-facing summary.

`result.to_structured_dict()`
Full result grouped by summary, groups, cells, transforms, properties, and
artifacts.

`result.to_scif(cell_mode=...)`
Generated SCIF text for the requested cell mode.

`result.to_dict()`
Raw compatibility dictionary. New integrations should prefer
`to_structured_dict()` unless they need legacy field names.

### Example

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

print(result.index)
print(result.convention_ssg_international_linear)
print(result.magnetic_phase)

summary = result.to_summary_dict()
structured = result.to_structured_dict()
```

## `find_spin_group_input_ssg(...)`

Use this function when you need SSG and MSG operations in the input-cell
setting.

```python
find_spin_group_input_ssg(
    structure_file: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
) -> dict
```

### Parameters

`structure_file`
Path to the input structure file.

The tolerance and POSCAR parameters have the same meaning as in
`find_spin_group_basic(...)`.

### Returns

Returns a dictionary with input-cell operation payloads and summary metadata.

Top-level fields:

`summary`
Input-cell and primitive-side identifiers.

`ssg`
Input-cell SSG operations and setting metadata.

`msg`
Input-cell oriented MSG operations and setting metadata.

`primitive_relation`
Relation between the input cell and the input magnetic primitive cell.

`input_poscar`
POSCAR text for non-POSCAR inputs when available.

`magnetic_primitive_poscar`
POSCAR text for the input magnetic primitive cell when useful.

### Example

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("path/to/structure.mcif")

print(payload["summary"]["input_ssg_index"])
print(payload["summary"]["primitive_ssg_index"])
print(payload["ssg"]["ops"])
```

## Advanced Entry Points

`find_spin_group_acc_primitive(...)`
Specialized route for ACC primitive payloads.

`find_spin_group_poscar_ssg(...)`
Specialized POSCAR-facing SSG route.

`find_spin_group_from_data(...)`, `find_spin_group_basic_from_data(...)`, and
`find_spin_group_acc_primitive_from_data(...)`
Data-array variants for integrations that already have parsed lattice,
positions, elements, occupancies, labels, and moments.

These functions are intentionally separate from the main guide because most
users should start with the three main routes above.
