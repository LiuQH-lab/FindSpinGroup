# InputSSGResult

Returned by:

```python
payload = find_spin_group_input_ssg("path/to/structure.mcif")
```

Also written by:

```bash
fsg -w path/to/structure.mcif
```

Use `InputSSGResult` when a downstream tool needs SSG or MSG operations in the
input-cell setting.

## Shape

```python
{
    "summary": {
        "input_ssg_index": str,
        "primitive_ssg_index": str,
        "input_conf": str,
        "input_spin_only_direction": object | None,
        "input_magnetic_phase": str,
        "input_ssg_database_symbol": str | None,
        "input_msg_num": int | None,
        "input_msg_bns_number": str | None,
        "input_msg_symbol": str | None,
        "primitive_msg_num": int | None,
        "primitive_msg_bns_number": str | None,
        "is_input_magnetic_primitive": bool,
        "input_ssg_may_be_incomplete": bool,
        "warning": str | None,
    },
    "ssg": {"setting": str, "spin_frame_setting": str | None, "ops": list},
    "msg": {"setting": str, "spin_frame_setting": str | None, "ops": list},
    "primitive_relation": {
        "T_input_to_input_magnetic_primitive": object,
        "determinant": float,
    },
    "input_poscar": str | None,
    "magnetic_primitive_poscar": str | None,
    "quasi_2d": None,
}
```

## `summary`

`input_ssg_index`
Identified SSG index for the input-cell setting.

`primitive_ssg_index`
Identified SSG index for the input magnetic primitive cell.

`input_conf`
Input-cell magnetic configuration class.

`input_spin_only_direction`
Spin-only direction for the input-cell result when applicable.

`input_magnetic_phase`
Magnetic phase classification for the input-cell result.

`input_ssg_database_symbol`
Database SSG symbol for the input-cell result.

`input_msg_num`, `input_msg_bns_number`, `input_msg_symbol`
Input-cell MSG identifiers.

`primitive_msg_num`, `primitive_msg_bns_number`
Primitive-side MSG identifiers.

`is_input_magnetic_primitive`
Whether the supplied input cell is already magnetic primitive.

`input_ssg_may_be_incomplete`
Whether input-cell operations may miss primitive-cell symmetry operations.

`warning`
Explanation when the input cell is not magnetic primitive or when the
input-cell payload needs caution.

## Operation Payloads

`ssg.setting`
Real-space setting of the input-cell SSG payload.

`ssg.spin_frame_setting`
Spin-frame setting of the input-cell SSG payload.

`ssg.ops`
Input-cell SSG operations.

Each item is an [SSGOperation](payload-definitions.md#ssgoperation).

`msg.setting`
Real-space setting of the input-cell oriented MSG payload.

`msg.spin_frame_setting`
Spin-frame setting of the input-cell oriented MSG payload.

`msg.ops`
Input-cell oriented MSG operations.

Each item is an [MSGOperation](payload-definitions.md#msgoperation).

## Primitive Relation And Files

`primitive_relation.T_input_to_input_magnetic_primitive`
Transform from input cell to input magnetic primitive cell.

See [TransformPayload](payload-definitions.md#transformpayload).

`primitive_relation.determinant`
Determinant of the transform.

`input_poscar`
POSCAR text for a non-POSCAR input cell when the input cell is distinct from
the magnetic primitive cell. It is omitted when it would duplicate the magnetic
primitive POSCAR.

`magnetic_primitive_poscar`
POSCAR text for the magnetic primitive cell.

`quasi_2d`
Reserved for route compatibility. The input-SSG route does not add quasi-2D
diagnostics.
