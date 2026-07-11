# `find_spin_group`

Run full analysis when the task requires explicit operations, cell settings,
tensor/site constraints, quasi-2D interpretation, generated artifacts, or route
diagnostics.

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

## When To Use It

Use full analysis for at least one specific product:

- SSG or MSG operations in a named setting;
- input, primitive, convention, or ACC cell data;
- SCIF, matched POSCAR/KPOINTS, or GSPG text;
- tensor-component constraints;
- magnetic-site or Wyckoff-chain analysis;
- quasi-2D interpretation;
- route, transform, or identify-index diagnostics.

For ordinary identification and screening, use
[`find_spin_group_basic`](find-spin-group-basic.md); it is faster and easier to
consume.

## Parameters

### Input and numerical controls

`cif`, `space_tol`, `mtol`, `meigtol`, `matrix_tol`, `parser_atol`, and the two
POSCAR/INCAR controls have the same meaning as in
[`find_spin_group_basic`](find-spin-group-basic.md#parameters).

### Full-analysis controls

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `calculation_mode` | `"3d"` | Ordinary 3D analysis. `"quasi2d"`/`"2d"`/`"slab"`/`"layer"` request slab interpretation and additive quasi-2D outputs. |
| `vacuum_axis` | `"c"` | Input-cell axis normal to the slab plane. Used only by quasi-2D interpretation. |
| `spin_texture_basis_max_order` | `None` | Set the spin-texture search/output ceiling and include per-order bases through that degree. |

The quasi-2D workflow can regularize or extend insufficient vacuum along the
selected axis before its interpretation path. Inspect the returned quasi-2D
diagnostics; do not assume it merely relabels the unchanged input lattice.

## Return Value

Returns [`MagSymmetryResult`](../magsymmetry-result.md), not a dictionary.

### Recommended accessors

| Need | Accessor |
| --- | --- |
| Compact display after full analysis | `result.to_summary_dict()` |
| Complete Python result grouped by meaning | `result.to_structured_dict()` |
| SCIF in an explicit setting | `result.to_scif(cell_mode=...)` |
| Existing raw/legacy integration | `result.to_dict()` |

`to_summary_dict()` is intentionally compact and does not repeat every direct
attribute. Use `to_structured_dict()` to locate complete MSG/cell/operation
information by semantic layer, but remember that this Python view retains
operation/domain objects and is not directly JSON serializable.

### Structured top level

```python
structured = result.to_structured_dict()

structured["summary"]
structured["groups"]
structured["cells"]
structured["transforms"]
structured["properties"]
structured["artifacts"]
```

The nested contract is documented in
[`StructuredResult`](../result-schemas/structured-result.md).

## Example: Ask For Specific Products

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

print(result.index)
print(result.msg_bns_number, result.msg_symbol)
print(result.magnetic_phase)

structured = result.to_structured_dict()
operation_views = result.operation_views
properties = structured["properties"]

scif = result.to_scif(cell_mode="ssg_convention_oriented")
poscar = result.acc_primitive_magnetic_cell_poscar
kpoints = result.KPOINTS
```

The generated ACC-primitive POSCAR and KPOINTS use the same real-space setting
and should be kept paired.

## Direct Attributes

Direct attributes remain convenient for interactive use. Common examples are:

- `index`, `magnetic_phase`, `msg_bns_number`, `msg_symbol`;
- `operation_views`;
- `scif`, `scif_outputs`;
- `acc_primitive_magnetic_cell_poscar`, `KPOINTS`;
- `magnetic_site_summary`, `vector_constraints_by_symmetry`;
- `quasi_2d`.

For interactive Python work, prefer the structured view because each payload
carries semantic setting information more clearly than the flat raw attribute
surface. For JSON integrations, prefer the basic route or a purpose-built
serialized operation payload.
