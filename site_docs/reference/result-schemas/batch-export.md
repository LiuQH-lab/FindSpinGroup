# BatchExport

Batch export columns are used by screening and regression comparison routes.

## Metadata Columns

`source_fsg_version`
FindSpinGroup version used to generate the row.

`source_run_tag`
Run tag.

`source_route`
Route used to generate the row.

## Case Columns

`case_id`
Stable case identifier.

`file_name`
Input file name.

`status`
Case status.

`duration_seconds`
Runtime for the case.

`index`
Identified OSSG index.

`conf`
Magnetic configuration class.

`phase`
Magnetic phase classification.

`acc`
Spin arithmetic crystal class.

`msg_acc`
MSG spin arithmetic crystal class when available.

## Identify Columns

`G0_id`
G0 group identifier.

`L0_id`
L0 group identifier.

`t_index`
Translation-index component.

`k_index`
k-index component.

## Spin Point-Group Columns

`nsspg_hm`
Nontrivial spin-part point group in Hermann-Mauguin notation.

`nsspg_symbol`
Nontrivial spin-part point-group symbol.

`sspg_hm`
Full spin-part point group in Hermann-Mauguin notation.

`sspg_symbol`
Full spin-part point-group symbol.

`ssg_type`
SSG type when available.

`spin_only_direction`
Spin-only direction when applicable.

## Symbol Columns

`ossg_symbol`
Public OSSG symbol.

`primitive_ssg_symbol`
Primitive-cell SSG symbol.

`sg_symbol`, `sg_num`
Input space-group symbol and number.

`sg_is_centrosymmetric`, `sg_is_polar`, `sg_is_chiral`
Input space-group flags.

`ossg_space_group_number`
Real-space group number of the OSSG projection.

`ossg_is_centrosymmetric`, `ossg_is_polar`, `ossg_is_chiral`
OSSG real-space projection flags.

`msg_symbol`, `msg_num`, `msg_type`, `msg_bns_number`, `msg_og_number`
MSG identifiers.

`msg_parent_space_group_number`
Parent space-group number of the MSG.

`msg_is_centrosymmetric`, `msg_is_polar`, `msg_is_chiral`
MSG flags.

## Property Columns

`spin_splitting_with_soc`
Spin-splitting conclusion with SOC.

`spin_splitting_without_soc`
Spin-splitting conclusion without SOC.

`ahc_with_soc`
AHC constraint with SOC.

`ahc_without_soc`
AHC constraint without SOC.

`is_altermagnet`
Altermagnet flag.

`is_spin_orbit_magnet`
Spin-orbit-magnet flag.

## Wyckoff And Magnetic-Site Columns

`wyckoff_split`
Wyckoff splitting summary.

`acc_primitive_wyckoff_split`
ACC primitive Wyckoff splitting summary.

`magnetic_site_status`
Status of magnetic-site analysis.

`magnetic_site_setting`
Setting used for magnetic-site output.

`magnetic_site_sg_primitive_to_magnetic_primitive_cell_expansion`
Expansion between SG primitive and magnetic primitive cells.

`magnetic_atom_count`
Number of magnetic atoms.

`nonzero_moment_atom_count`
Number of magnetic atoms with nonzero moment.

`zero_moment_magnetic_atom_count`
Number of selected magnetic atoms with zero moment.

`magnetic_atom_selection_mode`
Mode used to select magnetic atoms.

`number_of_magnetic_orbits_sg`
Number of magnetic orbits under SG.

`number_of_magnetic_orbits_ssg`
Number of magnetic orbits under SSG.

`number_of_magnetic_orbits_msg`
Number of magnetic orbits under MSG.

`max_magnetic_site_dof_ssg`, `max_magnetic_site_dof_msg`
Maximum magnetic-site degrees of freedom under SSG and MSG.

`total_magnetic_site_dof_ssg`, `total_magnetic_site_dof_msg`
Total magnetic-site degrees of freedom under SSG and MSG.

`magnetic_wyckoff_dof_summary`
Compact magnetic Wyckoff degree-of-freedom summary.

## Error Columns

`error_type`
Exception or error category.

`error_message`
Human-readable error message.

## Quasi-2D Columns

`quasi2d_status`
Status of quasi-2D diagnostics.

`quasi2d_source`
Source of quasi-2D interpretation.

`vacuum_axis_input`
Input-cell axis treated as the vacuum direction.

`spin_splitting_2d`
2D spin-splitting conclusion.

`spin_splitting_2d_interpretation`
Controlled interpretation text for the 2D spin-splitting conclusion.

`is_alter_2d`
2D altermagnet flag.

`quasi2d_magnetic_phase`
2D magnetic phase label.

`quasi2d_gp_label`, `quasi2d_gp_symbol`
Generic-point label and symbol used in quasi-2D diagnostics.

`quasi2d_gp_k_input`, `quasi2d_gp_k_acc`
Generic-point k coordinates in input and ACC settings.

`quasi2d_gp_spin_splitting`
Spin splitting at the quasi-2D generic point.

`quasi2d_gp_spin_polarizations`
Spin polarizations at the quasi-2D generic point.

`quasi2d_kpoint_projection_summary`
Summary of k-point projection into in-plane, out-of-plane, mixed, and unknown
classes.

`quasi2d_kpoints`
Compact quasi-2D k-point rows.
