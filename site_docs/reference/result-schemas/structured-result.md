# StructuredResult

Returned by:

```python
result = find_spin_group("path/to/structure.mcif")
structured = result.to_structured_dict()
```

Use `StructuredResult` for full programmatic integrations.

## Shape

```python
{
    "summary": SummaryLayer,
    "groups": GroupsLayer,
    "cells": CellsLayer,
    "transforms": TransformsLayer,
    "properties": PropertiesLayer,
    "artifacts": ArtifactsLayer,
    "legacy": dict,
}
```

`legacy` is the raw compatibility dictionary. New integrations should prefer the
structured layers.

## `summary`

High-level identifiers, magnetic phase, spin-texture fields, tolerances, and
source metadata.

```python
{
    "index": str,
    "conf": str,
    "phase": str,
    "phase_base": str,
    "phase_modifier": str,
    "acc": str,
    "msg_acc": str | None,
    "spin_texture_config_database": dict | None,
    "spin_texture_config_no_soc": dict | None,
    "spin_texture_config_soc": dict | None,
    "is_alter": str,
    "is_spin_orbit_magnet": str,
    "tolerances": dict,
    "source": {
        "metadata": dict | None,
        "parent_space_group": dict | None,
        "cell_parameter_strings": dict | None,
    },
}
```

## `groups`

Group identifiers, symbols, operation payloads, and little-group outputs.

```python
{
    "input_space_group": dict,
    "G0": dict,
    "L0": dict,
    "spin_point_group": dict,
    "gspg": dict,
    "ossg": dict,
    "msg": dict,
    "ssg_by_cell": dict,
    "msg_by_cell": dict,
    "little_groups": dict,
}
```

`groups.input_space_group`
Input real-space group information: `number`, `symbol`, `basis_or_setting`,
`is_centrosymmetric`, `is_polar`, and `is_chiral`.

`groups.G0`
Real-space group component: `number`, `symbol`.

`groups.L0`
Spin-lattice group component: `number`, `symbol`, `t_index`, `k_index`.

`groups.spin_point_group`
Spin-point-group symbols: `hm`, `symbol`, `full_hm`.

`groups.gspg`
Compact GSPG summary. See [SummaryResult](summary-result.md#gspg).

`groups.ossg`
Public convention OSSG information: `space_group_number`, `symbol_linear`,
`symbol_latex`, `is_centrosymmetric`, `is_polar`, `is_chiral`,
`real_space_setting`, `spin_frame_setting`, `spin_only_direction`, and
`spin_only_direction_cartesian`.

`groups.msg`
Corresponding MSG information: `num`, `type`, `symbol`, `bns_number`,
`og_number`, `parent_space_group_number`, `is_centrosymmetric`, `is_polar`,
and `is_chiral`.

### `groups.ssg_by_cell`

SSG operation payloads grouped by cell setting.

```python
{
    "input_magnetic_primitive": SSGPayload,
    "acc_primitive": SSGPayload,
    "acc_conventional": SSGPayload,
    "convention": SSGPayload,
    "g0_standard": SSGPayload,
    "l0_standard": SSGPayload,
}
```

Each `SSGPayload` may contain `setting`, `spin_frame_setting`, `ops`, `seitz`,
`seitz_latex`, `seitz_descriptions`, `international_linear`,
`international_latex`, `symbol_calibration_tol`, and `type`.

See [SSGPayload](payload-definitions.md#ssgpayload),
[SSGOperation](payload-definitions.md#ssgoperation), and
[SeitzDescription](payload-definitions.md#seitzdescription) for the next layer
below this field.

### `groups.msg_by_cell`

MSG operation payloads grouped by cell setting.

```python
{
    "acc_primitive": {"setting": str, "spin_frame_setting": str | None, "ops": list},
    "magnetic_primitive": {"setting": str, "spin_frame_setting": str | None, "ops": list},
}
```

See [MSGPayload](payload-definitions.md#msgpayload) and
[MSGOperation](payload-definitions.md#msgoperation) for the operation shape.

### `groups.little_groups`

Little-group operations and spin-polarization constraints:
`ssg_ops`, `ssg_seitz_latex`, `msg_ops`, `msg_seitz_latex`, `msg_symbols`, and
`msg_spin_polarizations`.

See [SpinPolarizationPayload](payload-definitions.md#spinpolarizationpayload)
for `msg_spin_polarizations`.

## `cells`

Cell payloads grouped by setting.

```python
{
    "input": {"detail": dict, "wp_chain": object},
    "input_magnetic_primitive": {"setting": str, "cell": object, "detail": dict},
    "database_standard": {
        "selected": str,
        "cell": object,
        "g0_standard": object,
        "l0_standard": object,
        "wp_chain": object,
    },
    "convention": {"setting": str, "cell": object, "detail": dict},
    "acc_primitive": {"setting": str, "cell": object, "detail": dict, "wp_chain": object},
    "acc_conventional": {"setting": str, "cell": object, "detail": dict},
}
```

Most entries follow [CellPayload](payload-definitions.md#cellpayload).

## `transforms`

Transforms connect named cell settings. Each transform is a matrix-plus-origin
mapping unless otherwise noted.

Fields include `input_to_input_magnetic_primitive`, `input_to_acc_primitive`,
`input_to_G0std`, `input_to_L0std`, `input_to_convention`,
`G0std_to_acc_primitive`, `L0std_to_acc_primitive`,
`acc_primitive_to_G0std`, `acc_primitive_to_L0std`,
`acc_primitive_to_input`, `convention_to_input`,
`convention_to_acc_primitive`, `convention_to_acc_conventional`,
`selected_standard_to_acc_conventional`, `raw`, and `audit`.

`raw`
Raw standard-setting transforms before public cleanup.

`audit`
Diagnostic route-audit payloads for transform and ACC primitive resolution.

See [TransformPayload](payload-definitions.md#transformpayload) for how to read
matrix-like transform fields and route-audit fields.

## `properties`

Physical-property, tensor, magnetic-site, quasi-2D, vector-constraint, and
ferroelectric-switching outputs.

`properties.magnetic_phase`
Structured magnetic-phase information: `phase`, `base`, `modifier`,
`spin_orbit_magnet_tag`, and `details`.

`properties.spin_splitting`
Spin-splitting outputs: `with_soc`, `without_soc`, `is_alter`,
`is_spin_orbit_magnet`, and `spin_polarizations`.

See [SpinPolarizationPayload](payload-definitions.md#spinpolarizationpayload)
for `spin_polarizations`.

`properties.ahc`
Anomalous Hall conductivity constraints: `with_soc`, `without_soc`.

`properties.tensors`
Symmetry-constrained tensor outputs: `AHE_wSOC`, `AHE_woSOC`, `BCDTensor`,
`IMDTensor`, `QMDTensor`, `MSGBCDTensor`, `MSGIMDTensor`, and `MSGQMDTensor`.

`properties.magnetic_site`
Magnetic-site and Wyckoff degree-of-freedom summary. Important fields include
`status`, `setting`, `magnetic_atom_count`, `nonzero_moment_atom_count`,
`zero_moment_magnetic_atom_count`, `n_magnetic_orbits_sg`,
`n_magnetic_orbits_ssg`, `n_magnetic_orbits_msg`,
`max_magnetic_site_dof_ssg`, `max_magnetic_site_dof_msg`,
`total_magnetic_site_dof_ssg`, `total_magnetic_site_dof_msg`, and
`magnetic_wp_dof_rows`.

`properties.quasi_2d`
Additive quasi-2D diagnostic payload when quasi-2D mode is requested. It is
`None` in ordinary 3D mode.

`properties.vector_constraints_by_symmetry`
Vector constraints grouped by `sg`, `ossg`, and `msg`.

`properties.ferroelectric_switching`
Symmetry-only ferroelectric-switching diagnostic payload. Read `status`,
`switching_detected`, `analysis_level`, `polarity_status`,
`allowed_polar_axes`, `allowed_polar_axes_setting`,
`candidate_reversal_domains`, and `domain_relation_rows` first.

## `artifacts`

Generated text artifacts.

```python
{
    "poscar": {
        "input_magnetic_primitive": str | None,
        "acc_primitive": str | None,
    },
    "scif": {
        "default": str,
        "by_mode": dict[str, str],
        "modes": list[str],
    },
    "kpoints": {
        "acc_primitive": {
            "text": str,
            "setting": str,
            "real_space_setting": str,
        },
    },
}
```
