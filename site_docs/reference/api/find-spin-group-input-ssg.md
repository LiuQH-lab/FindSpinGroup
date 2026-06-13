# `find_spin_group_input_ssg`

Input-cell operation route.

Use this function when a downstream tool needs SSG or MSG operations in the
input-cell setting.

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

## Parameters

`structure_file`
Path to the input structure file.

`space_tol`, `mtol`, `meigtol`, `matrix_tol`
Tolerance controls.

`poscar_allow_incar_magmom`, `poscar_prefer_incar_magmom`
POSCAR / INCAR magnetic-moment controls.

## Returns

Returns an [InputSSGResult](../result-schemas/input-ssg-result.md) dictionary.

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

## Returned Fields

### `summary`

`input_ssg_index`
Identified SSG index for the input-cell setting.

`primitive_ssg_index`
Identified SSG index for the input magnetic primitive cell.

`input_conf`
Input-cell magnetic configuration class.

`input_magnetic_phase`
Magnetic phase classification for the input-cell result.

`input_msg_bns_number`, `input_msg_symbol`
Input-cell MSG identifiers.

`is_input_magnetic_primitive`
Whether the supplied input cell is already magnetic primitive.

`input_ssg_may_be_incomplete`
Whether input-cell operations may miss primitive-cell symmetry operations.

`warning`
Explanation when the input-cell payload needs caution.

### `ssg`

Input-cell SSG operation payload.

`setting`
Real-space setting.

`spin_frame_setting`
Spin-frame setting.

`ops`
Input-cell SSG operations.

Each item is an
[SSGOperation](../result-schemas/payload-definitions.md#ssgoperation).

### `msg`

Input-cell oriented MSG operation payload.

`setting`
Real-space setting.

`spin_frame_setting`
Spin-frame setting.

`ops`
Input-cell oriented MSG operations.

Each item is an
[MSGOperation](../result-schemas/payload-definitions.md#msgoperation).

### `primitive_relation`

`T_input_to_input_magnetic_primitive`
Transform from input cell to input magnetic primitive cell.

See [TransformPayload](../result-schemas/payload-definitions.md#transformpayload).

`determinant`
Determinant of the transform.

## Example

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("path/to/structure.mcif")

print(payload["summary"]["input_ssg_index"])
print(payload["summary"]["primitive_ssg_index"])
print(payload["ssg"]["ops"])
```

## Notes

The CLI equivalent is:

```bash
fsg -w path/to/structure.mcif
```

This writes `ssg_symm.json`. It also writes `magnetic_primitive_poscar.vasp`;
`input_poscar.vasp` is written only when a non-POSCAR input cell is distinct
from the magnetic primitive cell.
