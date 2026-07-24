# Python API: Start With The Scientific Task

FindSpinGroup has three primary file-based functions and one operation-only
constructor. Most structure-analysis users need only the first function.

## Main Functions

| Scientific task | Function | Return | Read first |
| --- | --- | --- | --- |
| Identify and screen a magnetic structure | `find_spin_group_basic(...)` | `BasicResult` dictionary | `index`, `magnetic_phase`, MSG fields, `properties` |
| Inspect operations, settings, tensors, sites, or generated artifacts | `find_spin_group(...)` | `MagSymmetryResult` | a specific attribute or `to_structured_dict()` Python view |
| Export operations in the user-supplied cell | `find_spin_group_input_ssg(...)` | `InputSSGResult` dictionary | primitive-cell warning, then `ssg`/`msg` |
| Identify an SSG from complete operations or generators | `get_spin_space_group_from_operations(...)` | `SpinSpaceGroup` | `index`, `conf`, G0/L0, ACC, point groups |

Detailed pages:

- [`find_spin_group_basic`](api/find-spin-group-basic.md)
- [`find_spin_group`](api/find-spin-group.md)
- [`find_spin_group_input_ssg`](api/find-spin-group-input-ssg.md)
- [`get_spin_space_group_from_operations`](api/get-spin-space-group-from-operations.md)

## Recommended First Call

```python
from findspingroup import find_spin_group_basic

result = find_spin_group_basic("structure.mcif")

print(result["index"])
print(result["msg_bns_number"], result["msg_symbol"])
print(result["conf"], result["magnetic_phase"])
print(result["properties"])
```

This answers the ordinary identification and symmetry-permission questions
without constructing the full operation/cell/artifact result.

## Common Parameters

All main file-based functions share the central symmetry tolerances.

| Parameter | Default | Role | Change when |
| --- | ---: | --- | --- |
| `space_tol` | `0.02` | Shared spatial matching and symmetry-detection tolerance | Known positional noise makes the result unstable |
| `mtol` | `0.02 μB` | Magnetic-moment matching and zero-net-moment threshold | Known moment uncertainty/noise requires a sensitivity study |
| `meigtol` | `2e-5` | Spin point-group eigenvalue decisions | An explicit numerical point-group diagnostic identifies it |
| `matrix_tol` | `0.01` | Matrix/standardization/transform comparisons | An explicit transform or matrix diagnostic identifies it |

`find_spin_group_basic(...)` and `find_spin_group(...)` also accept:

| Parameter | Role |
| --- | --- |
| `parser_atol` | Parser-side moment-consistency tolerance, primarily for expanded SCIF sites |
| `spin_texture_basis_max_order` | Sets the requested/order-search ceiling and emits `basis_by_order` through that degree |
| `poscar_allow_incar_magmom` | Allows magnetic moments from a sibling `INCAR` |
| `poscar_prefer_incar_magmom` | Prefers sibling `INCAR` moments to embedded POSCAR moments |

For reliability guidance and the effect of increasing/decreasing each value,
read [Parameters and Tolerances](../guide/reliability-and-tolerances.md).

## Full Result: Use Semantic Accessors

```python
from findspingroup import find_spin_group

result = find_spin_group("structure.mcif")

summary = result.to_summary_dict()
structured = result.to_structured_dict()
scif = result.to_scif(cell_mode="ssg_convention_oriented")
```

Use `to_structured_dict()` to navigate the complete result in Python. It
separates:

- `summary`: high-level identifiers and source information;
- `groups`: SG/SSG/OSSG/MSG operations and symbols;
- `cells`: input, primitive, convention, and ACC cells;
- `transforms`: relations between settings and audits;
- `properties`: physical constraints, tensors, sites, and quasi-2D data;
- `artifacts`: generated SCIF, POSCAR, and KPOINTS.

The structured view still retains Python operation/domain objects and is not
directly JSON serializable. For a stable JSON-facing integration, start with
`find_spin_group_basic(...)`; use purpose-built `operation_views` or
`find_spin_group_input_ssg(...)` when operation matrices are required.

`to_dict()` is a raw compatibility surface. It is useful when maintaining an
older integration, but it should not be the starting contract for a new one.

## Input-Cell Operations: Guard Before Consuming

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("structure.mcif")
summary = payload["summary"]

if summary["input_ssg_may_be_incomplete"]:
    print(summary["warning"])

input_operations = payload["ssg"]["ops"]
```

When the supplied cell is not magnetic primitive, the input-cell operation set
can be incomplete relative to the primitive reference. In that case,
`input_ssg_index` and `primitive_ssg_index` need not be the same label.

## Parsed-Data Variants

Use `find_spin_group_from_data(...)`, `find_spin_group_basic_from_data(...)`, or
`find_spin_group_acc_primitive_from_data(...)` only when another application
already owns parsed lattice, position, species, occupancy, and moment arrays.
The file-based functions are safer for ordinary use because they also preserve
input-format metadata and spin-frame conventions.

## Operation-Only Identification

Use `get_spin_space_group_from_operations(...)` when another program already
owns SSG operations or generators and no atomic structure is available:

```python
from findspingroup import get_spin_space_group_from_operations

ssg = get_spin_space_group_from_operations(
    generators,
    spin_configuration="collinear",
    spin_only_direction=[0, 0, 1],
    spin_frame="oriented",
)

print(ssg.index, ssg.G0_num, ssg.L0_num)
print(ssg.acc, ssg.n_spin_part_point_group_symbol_s)
```

This route performs finite affine closure, completes the finite spin-only
representative used by the core, and identifies the database/convention index.
It does not construct a fictitious magnetic crystal. Consequently it cannot
produce atom-, cell-, Wyckoff-, net-moment-, SCIF-, POSCAR-, or
material-classification results.

For SOC/MSG analysis, spin and real rotations must be expressed in one common
physical coordinate representation. See the
[operation-only API contract](api/get-spin-space-group-from-operations.md)
before using `ssg.msg_info`.

## Interpretation Rule

> `Yes` or `allowed` means that the analyzed symmetry does not force a response
> to vanish. It does not predict a nonzero magnitude. `No` or `forbidden`
> means that the response is symmetry-forbidden under the stated no-SOC or SOC
> model.

See [Interpret Your Result](../guide/understanding-results.md) before turning a
field into a physical claim.
