# CLI Show Fields

`--show FIELD` reads fields from the payload produced by the selected route.
There is no separate CLI allow-list: any key that exists in the returned payload
can be read, and nested dictionaries can be addressed with dot paths.

Plain `--show` prints a readable representation. Use `--json --show FIELD` for
machine-readable JSON.

The lists below are ordered by practical usefulness for users: summary and
physics-facing fields first, generated artifacts next, and route/debug/legacy
fields last.

For the physical meaning of recurring terms such as OSSG, MSG, ACC primitive,
operation views, spin texture, Wyckoff chains, and vector constraints, see
[Concepts](../guide/concepts.md).

## Built-In Aliases

| Alias | Resolved field | Route |
| --- | --- | --- |
| `kpoints`, `kpoints_text` | `KPOINTS` | full |
| `poscar`, `primitive_poscar`, `primitive-poscar` | `acc_primitive_magnetic_cell_poscar` | full |
| `acc_primitive_poscar`, `acc-primitive-poscar` | `acc_primitive_magnetic_cell_poscar` | full |
| `scif_default`, `default_scif` | `scif` | full |
| `gspg` | `gspg_text` | full |
| `operation-views`, `ops` | `operation_views` | full |
| `wp-chain`, `wyckoff-chain` | `wp_chain` | full |
| `spin-texture-no-soc` | `spin_texture_config_no_soc` | basic or full |
| `spin-texture-soc` | `spin_texture_config_soc` | basic or full |

## Basic Route: find_spin_group_basic(...)

Available from:

```bash
fsg structure.mcif --show FIELD
```

### 1. Core Identification

These are the most useful fields for identifying the OSSG, ordinary space
group, MSG, and spin point groups.

```text
index
ossg_symbol_linear
conf
space_group_number
space_group_symbol
msg_symbol
msg_type
msg_bns_number
msg_og_number
empg
acc_symbol
g0_number
g0_symbol
l0_number
l0_symbol
it
ik
sspg
nsspg
spin_space_point_group_hm
spin_space_point_group_schoenflies
nontrivial_spin_space_point_group_hm
nontrivial_spin_space_point_group_schoenflies
```

### 2. Magnetic Phase And Material Properties

These fields are useful for screening and user-facing summaries.

```text
phase
magnetic_phase
magnetic_phase_base
magnetic_phase_modifier
magnetic_phase_details
net_moment
zero_net_moment_tol
properties
is_alter
is_som
spin_texture_config_database
spin_texture_config_no_soc
spin_texture_config_soc
```

Common nested fields:

```text
properties.ss_wo_soc
properties.ss_w_soc
properties.ahc_wo_soc
properties.ahc_w_soc
spin_texture_config_no_soc.spin_texture_type
spin_texture_config_no_soc.momentum_space_spin_configuration
spin_texture_config_no_soc.order
spin_texture_config_no_soc.basis
spin_texture_config_no_soc.basis_latex
spin_texture_config_soc.spin_texture_type
spin_texture_config_soc.momentum_space_spin_configuration
spin_texture_config_soc.order
spin_texture_config_soc.basis
spin_texture_config_soc.basis_latex
```

### 3. Symmetry-Derived Constraints

These fields are useful when checking polar/chiral constraints, vector
constraints, quasi-2D results, or ferroelectric-switching diagnostics.

```text
sg_is_polar
sg_is_chiral
ssg_is_polar
ssg_is_chiral
msg_is_polar
msg_is_chiral
vector_constraints_by_symmetry
ferroelectric_switching
quasi_2d
```

Common nested fields:

```text
vector_constraints_by_symmetry.sg
vector_constraints_by_symmetry.ossg
vector_constraints_by_symmetry.msg
```

### 4. Transforms, Route Audit, And Tolerances

These are mostly diagnostic fields for validating the route and reproducing
settings.

```text
identify_index_details
acc_primitive_resolution_audit
T_input_to_acc_primitive
T_selected_standard_to_acc_primitive
tolerances
```

Common nested fields:

```text
identify_index_details.G0_id
identify_index_details.L0_id
identify_index_details.t_index
identify_index_details.k_index
tolerances.space
tolerances.moment
tolerances.matrix
```

## Full Route: MagSymmetryResult.to_dict()

Available from:

```bash
fsg --all structure.mcif --show FIELD
```

### 1. Core Identification

The most useful full-route summary fields.

```text
index
conf
magnetic_phase
magnetic_phase_base
magnetic_phase_modifier
magnetic_phase_spin_orbit_magnet
magnetic_phase_details
acc
msg_acc
G0_symbol
G0_num
L0_symbol
L0_num
it
ik
SSPG_symbol_hm
SSPG_symbol_s
input_space_group_number
input_space_group_symbol
sg_is_centrosymmetric
sg_is_polar
sg_is_chiral
ossg_space_group_number
ossg_is_centrosymmetric
ossg_is_polar
ossg_is_chiral
msg_num
msg_type
msg_symbol
msg_bns_number
msg_og_number
msg_parent_space_group_number
msg_is_centrosymmetric
msg_is_polar
msg_is_chiral
```

### 2. Physics And Materials Properties

Useful for screening, UI cards, and tabular exports.

```text
spin_texture_config_database
spin_texture_config_no_soc
spin_texture_config_soc
spinsplitting_w_soc
spinsplitting_wo_soc
ahc_w_soc
ahc_wo_soc
is_alter
is_spin_orbit_magnet
vector_constraints_by_symmetry
ferroelectric_switching
magnetic_site_summary
wp_chain
acc_primitive_wp_chain
quasi_2d
```

Common nested fields:

```text
spin_texture_config_no_soc.spin_texture_type
spin_texture_config_no_soc.momentum_space_spin_configuration
spin_texture_config_no_soc.order
spin_texture_config_no_soc.basis
spin_texture_config_no_soc.basis_latex
spin_texture_config_soc.spin_texture_type
spin_texture_config_soc.momentum_space_spin_configuration
spin_texture_config_soc.order
spin_texture_config_soc.basis
spin_texture_config_soc.basis_latex
vector_constraints_by_symmetry.sg
vector_constraints_by_symmetry.ossg
vector_constraints_by_symmetry.msg
magnetic_site_summary.status
magnetic_site_summary.setting
magnetic_site_summary.cell_expansion
magnetic_site_summary.magnetic_atom_count
magnetic_site_summary.nonzero_moment_atom_count
magnetic_site_summary.zero_moment_magnetic_atom_count
magnetic_site_summary.n_magnetic_orbits_sg
magnetic_site_summary.n_magnetic_orbits_ssg
magnetic_site_summary.n_magnetic_orbits_msg
magnetic_site_summary.max_magnetic_site_dof_ssg
magnetic_site_summary.max_magnetic_site_dof_msg
magnetic_site_summary.total_magnetic_site_dof_ssg
magnetic_site_summary.total_magnetic_site_dof_msg
magnetic_site_summary.magnetic_wp_dof_rows
```

### 3. Generated User Artifacts

These fields contain text or structured data users often want to save or display.

```text
scif
scif_outputs
scif_cell_modes
KPOINTS
KPOINTS_setting
KPOINTS_real_space_setting
acc_primitive_magnetic_cell_poscar
acc_primitive_magnetic_cell
acc_primitive_magnetic_cell_setting
acc_primitive_magnetic_cell_detail
acc_conventional_cell
acc_conventional_cell_setting
acc_conventional_cell_detail
convention_cell
convention_cell_setting
convention_cell_detail
gspg_text
gspg_symbol_linear
gspg_symbol_latex
gspg_effective_mpg_symbol
gspg_npg_symbol_s
gspg_point_part_linear
gspg_output_mode
gspg_real_space_setting
gspg_spin_frame_setting
gspg_symbol_mode
gspg_tentative_symbol_s
```

SCIF modes:

```text
scif_outputs.ssg_convention_cartesian
scif_outputs.ssg_convention_oriented
scif_outputs.database_standard_cartesian
scif_outputs.database_standard_oriented
scif_outputs.magnetic_primitive_cartesian
scif_outputs.magnetic_primitive_oriented
scif_outputs.input_cartesian
scif_outputs.input_oriented
```

### 4. Operation Views

Preferred operation output for UI and users. This is more structured and easier
to consume than the legacy operation lists.

```text
operation_views
convention_nssg_ops
convention_nssg_seitz
convention_nssg_seitz_latex
convention_ssg_ops
convention_ssg_setting
convention_ssg_spin_frame_setting
convention_ssg_seitz
convention_ssg_seitz_latex
convention_ssg_seitz_descriptions
convention_ssg_international_linear
convention_ssg_international_latex
convention_ssg_symbol_calibration_tol
convention_spin_only_direction
convention_spin_only_direction_cartesian
```

Operation-view settings:

```text
operation_views.convention_cartesian
operation_views.convention_oriented
operation_views.magnetic_primitive_cartesian
operation_views.magnetic_primitive_oriented
operation_views.input_cartesian
operation_views.input_oriented
```

Common views under each setting:

```text
operation_views.<setting>.default_view
operation_views.<setting>.views.all
operation_views.<setting>.views.nssg
operation_views.<setting>.views.generators
operation_views.<setting>.views.pure_translations
operation_views.<setting>.views.spin_translations
operation_views.<setting>.views.l0_operations
operation_views.<setting>.views.msg
```

### 5. Spin Polarization, Little Groups, And K-Path Payloads

Useful for band-structure spin polarization and little-group analysis.

```text
spin_polarizations
spin_polarizations_setting
spin_polarizations_real_space_setting
spin_polarizations_spin_frame
spin_polarizations_acc_cartesian
spin_polarizations_acc_cartesian_setting
spin_polarizations_acc_poscar_spin_frame
spin_polarizations_acc_poscar_spin_frame_setting
ssg_little_group_ops
ssg_little_group_seitz_latex
msg_little_group_ops
msg_little_group_seitz_latex
msg_little_group_symbols
msg_spin_polarizations
msg_spin_polarizations_setting
msg_spin_polarizations_real_space_setting
msg_spin_polarizations_spin_frame
msg_spin_polarizations_acc_cartesian
msg_spin_polarizations_acc_cartesian_setting
msg_spin_polarizations_acc_poscar_spin_frame
msg_spin_polarizations_acc_poscar_spin_frame_setting
```

### 6. Tensor Outputs

Useful when the caller needs explicit tensor constraints and equation payloads.

```text
tensor_outputs
AHE_woSOC
AHE_wSOC
BCDTensor
MSGBCDTensor
QMDTensor
MSGQMDTensor
IMDTensor
MSGIMDTensor
```

Common nested fields:

```text
tensor_outputs.AHE_woSOC
tensor_outputs.AHE_wSOC
tensor_outputs.BCDTensor
tensor_outputs.MSGBCDTensor
tensor_outputs.QMDTensor
tensor_outputs.MSGQMDTensor
tensor_outputs.IMDTensor
tensor_outputs.MSGIMDTensor
```

### 7. Cells And Cell Details

Useful when reconstructing settings or comparing generated cells. Some names
are compatibility aliases retained from earlier versions.

```text
input_cell_detail
input_magnetic_primitive_cell
input_magnetic_primitive_cell_setting
input_magnetic_primitive_cell_poscar
input_magnetic_primitive_cell_detail
magnetic_primitive_cell
magnetic_primitive_cell_setting
magnetic_primitive_cell_poscar
magnetic_primitive_cell_detail
primitive_magnetic_cell
primitive_magnetic_cell_setting
primitive_magnetic_cell_poscar
primitive_magnetic_cell_detail
acc_primitive_magnetic_cell
acc_primitive_magnetic_cell_setting
acc_primitive_magnetic_cell_poscar
acc_primitive_magnetic_cell_detail
acc_conventional_cell
acc_conventional_cell_setting
acc_conventional_cell_detail
g0_standard_cell
l0_standard_cell
convention_cell
convention_cell_setting
convention_cell_detail
ssg_std_cell
input_space_group_basis_or_setting
source_structure_metadata
source_parent_space_group
source_cell_parameter_strings
```

### 8. SSG Operation Lists By Setting

Raw or legacy operation payloads. Prefer `operation_views` for new integrations,
but these remain useful for debugging and compatibility.

```text
input_ssg_ops
input_magnetic_primitive_ssg_ops
input_magnetic_primitive_ssg_setting
input_magnetic_primitive_ssg_seitz
input_magnetic_primitive_ssg_seitz_latex
input_magnetic_primitive_ssg_seitz_descriptions
input_magnetic_primitive_ssg_international_linear
input_magnetic_primitive_ssg_international_latex
input_magnetic_primitive_ssg_symbol_calibration_tol
input_magnetic_primitive_ssg_type
magnetic_primitive_ssg_ops
magnetic_primitive_ssg_setting
magnetic_primitive_ssg_seitz
magnetic_primitive_ssg_seitz_latex
magnetic_primitive_ssg_seitz_descriptions
magnetic_primitive_ssg_international_linear
magnetic_primitive_ssg_international_latex
magnetic_primitive_ssg_symbol_calibration_tol
magnetic_primitive_ssg_type
primitive_magnetic_cell_ssg_ops
primitive_magnetic_cell_ssg_setting
primitive_magnetic_cell_ssg_seitz
primitive_magnetic_cell_ssg_seitz_latex
primitive_magnetic_cell_ssg_seitz_descriptions
primitive_magnetic_cell_ssg_international_linear
primitive_magnetic_cell_ssg_international_latex
primitive_magnetic_cell_ssg_symbol_calibration_tol
primitive_magnetic_cell_ssg_type
acc_primitive_ssg_ops
acc_primitive_ssg_setting
acc_primitive_ssg_seitz
acc_primitive_ssg_seitz_latex
acc_primitive_ssg_seitz_descriptions
acc_primitive_ssg_international_linear
acc_primitive_ssg_international_latex
acc_primitive_ssg_symbol_calibration_tol
acc_primitive_ssg_ops_cartesian
acc_primitive_ssg_seitz_cartesian
acc_primitive_ssg_seitz_latex_cartesian
acc_primitive_ssg_ops_oriented
acc_primitive_ssg_seitz_oriented
acc_primitive_ssg_seitz_latex_oriented
acc_primitive_spin_only_direction_cartesian
acc_primitive_spin_only_direction_poscar_spin_frame
acc_conventional_ssg_ops
acc_conventional_ssg_setting
acc_conventional_ssg_seitz
acc_conventional_ssg_seitz_latex
acc_conventional_ssg_seitz_descriptions
acc_conventional_ssg_international_linear
acc_conventional_ssg_international_latex
acc_conventional_ssg_symbol_calibration_tol
g0_standard_ssg_ops
g0_standard_ssg_seitz
g0_standard_ssg_seitz_latex
g0_standard_ssg_seitz_descriptions
l0_standard_ssg_ops
l0_standard_ssg_seitz
l0_standard_ssg_seitz_latex
l0_standard_ssg_seitz_descriptions
input_ssg_ops_spin_cartesian
input_ssg_seitz_latex_spin_cartesian
input_ssg_ops_spin_oriented
input_ssg_seitz_latex_spin_oriented
input_wp_chain
input_spin_only_direction_spin_cartesian
input_spin_only_direction_spin_oriented
input_ssg_may_be_incomplete
input_setting_warning
```

### 9. MSG And GSPG Operation Payloads

Useful for low-level operation export and GSPG text/debugging.

```text
spin_only
spin_part_point_group
gspg_ops
gspg_raw_ops
gspg_ops_xyz_uvw
gspg_raw_ops_xyz_uvw
gspg_generator_indices
gspg_generator_ops
gspg_generator_ops_xyz_uvw
gspg_spin_only_ops
gspg_spin_only_ops_xyz_uvw
gspg_collinear_axis
gspg_spin_only_component_symbol_s
gspg_spin_only_part_linear
magnetic_primitive_msg_ops
magnetic_primitive_msg_ops_setting
magnetic_primitive_msg_ops_spin_frame_setting
primitive_msg_ops
primitive_msg_ops_setting
primitive_msg_ops_spin_frame_setting
acc_primitive_msg_ops
acc_primitive_msg_ops_setting
acc_primitive_msg_ops_spin_frame_setting
```

### 10. Coordinate Transforms And Spin Frames

Mostly diagnostic, but important for reproducing output settings.

```text
T_input_to_ssg_std
T_input_to_mag_primitive
T_input_to_input_magnetic_primitive
T_input_to_acc_primitive
T_input_to_G0std
T_input_to_G0std_ops_nofrac
T_G0std_to_primitive
T_G0std_to_acc_primitive
T_acc_primitive_to_G0std
T_input_to_L0std
T_L0std_to_primitive
T_L0std_to_acc_primitive
T_acc_primitive_to_L0std
T_input_to_convention
T_G0std_to_input
T_L0std_to_input
T_acc_primitive_to_input
T_convention_to_input
T_convention_to_acc_primitive
T_convention_to_acc_conventional
T_convention_to_acc_conventional_is_convention_self_automorphism
T_convention_to_acc_conventional_label
T_convention_to_acc_conventional_audit
selected_standard_setting
T_selected_standard_to_acc_conventional
T_selected_standard_to_acc_conventional_is_self_automorphism
T_selected_standard_to_acc_conventional_label
T_selected_standard_to_acc_conventional_audit
raw_T_input_to_G0std
raw_T_input_to_L0std
acc_primitive_real_cartesian_to_poscar_spin_frame
poscar_spin_frame_to_acc_primitive_real_cartesian
real_cartesian_to_spin_frame
spin_frame_to_real_cartesian
```

### 11. Route Audit, Version, And Tolerances

Diagnostic fields useful for validating route choices and reproducing runs.

```text
fsg_version
identify_index_details
acc_primitive_resolution_audit
g0std_axis_collapse_audit
tolerances
symbol_calibration_tol
```

## Input-SSG Writer Payload

`-w/--write` does not combine with `--show`; it writes the input-SSG payload to
files. The same payload is available from Python through
`find_spin_group_input_ssg(...)`.

Top-level input-SSG payload fields:

```text
summary
ssg
msg
primitive_relation
quasi_2d
input_poscar
magnetic_primitive_poscar
```
