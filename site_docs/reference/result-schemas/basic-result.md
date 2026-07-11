# BasicResult

Returned by:

```python
summary = find_spin_group_basic("path/to/structure.mcif")
```

Also used by the default CLI route:

```bash
fsg path/to/structure.mcif
```

Use `BasicResult` for compact identification and screening.

## Selected Shape

The following block shows the main fields, not every compatibility or
diagnostic key. Use the sections below and the live dictionary keys when
maintaining a complete integration.

```python
{
    "index": str,
    "ossg_symbol_linear": str,
    "g0_symbol": str,
    "g0_number": int,
    "l0_symbol": str,
    "l0_number": int,
    "it": int,
    "ik": int,
    "nsspg": str,
    "sspg": str,
    "nontrivial_spin_space_point_group_hm": str,
    "nontrivial_spin_space_point_group_schoenflies": str,
    "spin_space_point_group_hm": str,
    "spin_space_point_group_schoenflies": str,
    "acc_symbol": str,
    "space_group_symbol": str,
    "space_group_number": int,
    "msg_symbol": str | None,
    "msg_type": int | None,
    "msg_bns_number": str | None,
    "msg_og_number": str | None,
    "conf": str,
    "phase": str,
    "magnetic_phase": str,
    "properties": BasicProperties,
    "tolerances": Tolerances,
    "identify_index_details": dict,
    "acc_primitive_resolution_audit": dict,
    "ferroelectric_switching": dict,
}
```

## Read First

`index`
Final identified oriented spin space group index. Example:
`194.164.1.1.L`.

`ossg_symbol_linear`
Linear oriented spin space group symbol in the selected convention setting.

`magnetic_phase`
High-level magnetic phase classification. Example: `AFM(Altermagnet)`.

`magnetic_phase_family`
Primary symmetry family, `FM-class` or `AFM-class`.

`magnetic_atom_orbit_count_ssg`
Magnetic-atom orbit count under the complete SSG in the magnetic primitive
cell. This count refines `FM-class` into FM, FiM, or compensated FiM.

`msg_bns_number`
BNS number of the corresponding magnetic space group.

`msg_symbol`
BNS symbol of the corresponding magnetic space group.

`conf`
Magnetic configuration class, such as `Collinear`, `Coplanar`, or
`Noncoplanar`.

## SSG Components

`g0_symbol`, `g0_number`
Real-space group component of the identified SSG.

`l0_symbol`, `l0_number`
Spin-lattice group component of the identified SSG.

`it`
Translation-index component of the identified SSG label.

`ik`
k-index component of the identified SSG label.

`nsspg`
Nontrivial spin-space point-group HM symbol.

`sspg`
Full spin-space point-group HM symbol.

`nontrivial_spin_space_point_group_hm`,
`nontrivial_spin_space_point_group_schoenflies`
Nontrivial spin-space point group in HM and Schoenflies notation.

`spin_space_point_group_hm`, `spin_space_point_group_schoenflies`
Full spin-space point group in HM and Schoenflies notation.

`acc_symbol`
Spin arithmetic crystal class symbol.

`empg`
Effective magnetic point-group symbol.

## Spatial SG And MSG

`space_group_symbol`, `space_group_number`
Ordinary real-space group symbol and number detected on the route's magnetic
cell after moment directions are ignored. Do not assume that this label alone
describes the unreduced input-cell setting.

`sg_is_polar`, `sg_is_chiral`
Polar and chiral flags for the input space group.

`ssg_is_polar`, `ssg_is_chiral`
Polar and chiral flags for the real-space projection of the identified SSG.

`msg_symbol`, `msg_type`, `msg_bns_number`, `msg_og_number`
Corresponding magnetic space group identifiers.

`msg_is_polar`, `msg_is_chiral`
Polar and chiral flags for the corresponding MSG.

## Magnetic Phase

`phase`
Compact alias for `magnetic_phase`.

`magnetic_phase_base`
Base magnetic phase before modifiers.

`magnetic_phase_modifier`
Modifier appended to the base phase.

`magnetic_phase_details`
Classifier evidence and diagnostic details. Important keys include `conf`,
`net_moment`, `zero_net_moment`, `classification_rule`, `base_phase`,
`symmetry_family`, `order_type_status`, `magnetic_atom_orbit_count_ssg`,
`magnetic_atom_orbit_analysis`, `modifier`, `spin_splitting_without_soc`,
`is_altermagnet`, and `is_spin_orbit_magnet`.

`net_moment`
Magnitude of the magnetic-cell moment sum used by the classifier, in μB.

`zero_net_moment_tol`
Tolerance used for zero-net-moment decisions.

## Properties

`properties`
Compact physical-property summary.

```python
{
    "ss_w_soc": str,
    "ss_wo_soc": str,
    "ahc_w_soc": str,
    "ahc_wo_soc": str,
    "is_alter": str,
    "is_spin_orbit_magnet": str,
    "magnetic_phase_family": str,
    "magnetic_phase_base": str,
    "magnetic_phase_modifier": str,
}
```

`is_alter`
Display tag for altermagnet classification. Empty when not applicable.

`is_som`
Display tag for spin-orbit-magnet classification. Empty when not applicable.

## Spin Texture And Vector Constraints

`spin_texture_config_database`
Spin-texture configuration summary loaded from the runtime database.

`spin_texture_config_no_soc`
Spin-texture configuration from the non-SOC route.

`spin_texture_config_soc`
Spin-texture configuration from the SOC-compatible route.

`vector_constraints_by_symmetry`
Vector constraints grouped by `sg`, `ossg`, and `msg`.

## Transforms And Diagnostics

`T_input_to_acc_primitive`
Transform from input cell to ACC primitive magnetic cell.

`T_selected_standard_to_acc_primitive`
Transform from selected database-standard setting to ACC primitive magnetic
cell.

`identify_index_details`
Diagnostic payload from the identify-index route.

`acc_primitive_resolution_audit`
Diagnostic payload describing how the ACC primitive route was resolved.

`tolerances`
Effective core tolerance values: `space_tol`, `mtol`, `meigtol`, and
`matrix_tol`. `parser_atol` is a call/input-parser setting and is not serialized
inside this BasicResult mapping.

`quasi_2d`
`None` for the basic public 3D route.

`ferroelectric_switching`
Symmetry-only ferroelectric-switching payload.
