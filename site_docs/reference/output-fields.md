# Output Fields

This page documents public output fields in prose blocks rather than a large
table. Fields are grouped by the function or accessor that returns them.

## `find_spin_group_basic(...)`

Returns a compact JSON-serializable dictionary.

`index`
Final identified OSSG index.

`identify_index_details`
Diagnostic details from the identify-index route. Use this for debugging and
validation rather than normal result display.

`g0_symbol`, `g0_number`
Real-space group symbol and number used by the identified SSG.

`l0_symbol`, `l0_number`
Spin-lattice group symbol and number.

`it`, `ik`
Translation and k-index components.

`nsspg`, `sspg`
Nontrivial and full spin-part point-group symbols.

`acc_symbol`
Arithmetic crystal class symbol.

`space_group_symbol`, `space_group_number`
Input real-space group information.

`msg_symbol`, `msg_type`, `msg_bns_number`, `msg_og_number`
Corresponding MSG identifiers.

`empg`
Effective magnetic point-group symbol.

`conf`
Magnetic configuration class.

`phase`, `magnetic_phase`
Magnetic phase classification. `phase` is the compact export alias and
`magnetic_phase` is the explicit field name.

`magnetic_phase_base`, `magnetic_phase_modifier`, `magnetic_phase_details`
Structured details behind the magnetic-phase label.

`properties`
Physical-property summary, including spin splitting, anomalous Hall
constraints, and altermagnet / spin-orbit-magnet tags.

`is_alter`, `is_som`
Altermagnet and spin-orbit-magnet flags.

`sg_is_polar`, `sg_is_chiral`
Polar and chiral flags for the input space group.

`ssg_is_polar`, `ssg_is_chiral`
Polar and chiral flags for the SSG real-space group.

`msg_is_polar`, `msg_is_chiral`
Polar and chiral flags for the corresponding MSG.

`quasi_2d`
`None` for the basic public 3D route.

`ferroelectric_switching`
Symmetry-only ferroelectric-switching payload.

`tolerances`
Effective tolerance values used by the route.

## `MagSymmetryResult.to_summary_dict()`

Returns a compact dictionary from the full route.

`index`
Final identified OSSG index.

`conf`
Magnetic configuration class.

`phase`
Magnetic phase classification.

`acc`
Arithmetic crystal class.

`properties`
Physical-property summary from `properties_summary()`.

`gspg`
GSPG summary from `gspg_summary()`, including compact text output.

`polar_axes_by_symmetry`
Polar-axis information inferred from symmetry.

`ferroelectric_switching`
Symmetry-only ferroelectric-switching payload.

## `MagSymmetryResult.to_structured_dict()`

Returns the full result grouped by semantic layer.

`summary`
High-level identifiers, phase fields, tolerances, and source metadata.

`groups`
Input SG, G0, L0, spin point group, GSPG, OSSG, MSG, SSG operation payloads by
cell, MSG operation payloads by cell, and little-group outputs.

`cells`
Input, input magnetic primitive, database standard, convention, ACC primitive,
and ACC conventional cell data.

`transforms`
Setting transforms between cell layers and route audit details.

`properties`
Magnetic phase details, spin splitting, anomalous Hall constraints, tensor
outputs, magnetic-site summaries, quasi-2D diagnostics, polar-axis data, and
ferroelectric-switching payloads.

`artifacts`
Generated POSCAR, SCIF, and KPOINTS text.

`legacy`
Raw attribute dictionary for compatibility. New integrations should avoid
depending on undocumented legacy keys when a structured field exists.

## `find_spin_group_input_ssg(...)`

Returns a dictionary focused on the input-cell setting.

`summary`
Input-cell and primitive-side identifiers.

Important `summary` fields:

`input_ssg_index`
Identified SSG index for the input-cell setting.

`primitive_ssg_index`
Identified SSG index for the input magnetic primitive cell.

`input_conf`
Input-cell magnetic configuration class.

`input_spin_only_direction`
Spin-only direction when applicable.

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
Explanation when the input cell is not magnetic primitive.

`ssg`
Input-cell SSG operations and setting metadata.

`msg`
Input-cell oriented MSG operations and setting metadata.

`primitive_relation`
Transform relation from input cell to input magnetic primitive cell.

`input_poscar`
POSCAR text for non-POSCAR inputs when available.

`magnetic_primitive_poscar`
POSCAR text when the input cell is not magnetic primitive.

## Batch Export Columns

Batch exports use stable column names for screening and regression comparison.

Metadata columns:
`source_fsg_version`, `source_run_tag`, `source_route`.

Case columns:
`case_id`, `file_name`, `status`, `duration_seconds`, `index`, `conf`,
`phase`, `acc`, `msg_acc`.

Identify columns:
`G0_id`, `L0_id`, `t_index`, `k_index`.

Symbol columns:
`ossg_symbol`, `primitive_ssg_symbol`, `sg_symbol`, `sg_num`,
`msg_symbol`, `msg_num`, `msg_type`, `msg_bns_number`, `msg_og_number`, and
polar or chiral flags.

Property columns:
`spin_splitting_with_soc`, `spin_splitting_without_soc`, `ahc_with_soc`,
`ahc_without_soc`, `is_altermagnet`, `is_spin_orbit_magnet`.

Error columns:
`error_type`, `error_message`.

Quasi-2D exports add fields such as `quasi2d_status`, `vacuum_axis_input`,
`spin_splitting_2d`, `quasi2d_kpoint_projection_summary`, and
`quasi2d_kpoints`.
