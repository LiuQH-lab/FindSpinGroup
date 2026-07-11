# `find_spin_group_input_ssg`

Export SSG and MSG operations in the **user-supplied cell setting** for
interoperability with another program.

This is not a competing canonical identification route. If the supplied cell
is not magnetic primitive, its operation list can be incomplete relative to the
primitive-cell symmetry.

## Signature

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

## Safe Use Pattern

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("structure.mcif")
summary = payload["summary"]

if summary["input_ssg_may_be_incomplete"]:
    print(summary["warning"])
    print("input-cell label:", summary["input_ssg_index"])
    print("primitive reference:", summary["primitive_ssg_index"])

ssg_operations = payload["ssg"]["ops"]
msg_operations = payload["msg"]["ops"]
```

Read the warning before consuming the operations. For a non-primitive input,
`input_ssg_index` can be the label of an incomplete input-cell subgroup and can
differ from `primitive_ssg_index`.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `structure_file` | required | Magnetic structure path. |
| `space_tol` | `0.02` | Shared spatial matching and symmetry-detection tolerance. |
| `mtol` | `0.02 μB` | Magnetic-moment matching and zero-moment tolerance. |
| `meigtol` | `2e-5` | Spin point-group eigenvalue tolerance. |
| `matrix_tol` | `0.01` | Matrix/standardization tolerance. |
| `poscar_allow_incar_magmom` | `False` | Allow a sibling `INCAR` to provide `MAGMOM`. |
| `poscar_prefer_incar_magmom` | `False` | Prefer sibling `INCAR` moments over embedded POSCAR moments. |

## Return Value

Returns [`InputSSGResult`](../result-schemas/input-ssg-result.md):

```python
{
    "summary": dict,
    "ssg": {"setting": str, "spin_frame_setting": str | None, "ops": list},
    "msg": {"setting": str, "spin_frame_setting": str | None, "ops": list},
    "primitive_relation": dict,
    "input_poscar": str | None,
    "magnetic_primitive_poscar": str | None,
    "quasi_2d": None,
}
```

### Summary guard fields

| Field | Meaning |
| --- | --- |
| `is_input_magnetic_primitive` | Whether the supplied cell is already magnetic primitive. |
| `input_ssg_may_be_incomplete` | Whether the input-cell operation set can omit primitive-cell symmetries. |
| `warning` | Human-readable consequence and recommended interpretation. |
| `input_ssg_index` | Label found from the input-cell operation set. |
| `primitive_ssg_index` | Complete magnetic-primitive reference label. |

### Operation convention

An SSG operation acts as

```text
r' = R r + t
m' = S m
```

where `real_rotation = R` and fractional `translation = t` use the payload's
real-space `setting`, while `spin_rotation = S` uses its
`spin_frame_setting`.

An MSG operation stores `real_rotation`, fractional `translation`, and
`time_reversal`:

- `time_reversal = +1`: ordinary operation;
- `time_reversal = -1`: operation containing time reversal.

For axial magnetic moments, the associated spin action follows the route's MSG
convention based on `time_reversal * det(R) * R`.

### Primitive relation

`T_input_to_input_magnetic_primitive` maps input fractional coordinates to the
input magnetic primitive cell (modulo lattice translations). The absolute
determinant is the input-cell volume multiple relative to the magnetic
primitive cell; a value near one indicates that the input is magnetic
primitive.

## CLI Equivalent

```bash
fsg -w structure.mcif
```

This writes:

- `ssg_symm.json`;
- `magnetic_primitive_poscar.vasp`;
- `input_poscar.vasp` when a distinct input-cell POSCAR representation is
  useful.

The command writes into the current directory; move existing files first if
they must not be replaced.
