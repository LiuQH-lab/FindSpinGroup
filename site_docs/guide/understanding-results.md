# Understanding Results

Read FindSpinGroup output in this order: identified OSSG, corresponding MSG,
magnetic phase, settings, and generated artifacts.

## Identified OSSG

The primary identifier is `index`.

Example:

```text
194.164.1.1.L
```

This is the final identified oriented spin space group index used by the
FindSpinGroup route. It is not only the raw tuple of `G0`, `L0`, `it`, and `ik`
components, although those components are also available in the outputs.

Common fields:

`index`
The final identified OSSG index.

`G0_symbol`, `G0_num`
The real-space group component used by the identified spin space group.

`L0_symbol`, `L0_num`
The spin-lattice group component.

`it`, `ik`
Translation and k-index components in the SSG identification.

`conf`
Magnetic configuration class, such as `Collinear`, `Coplanar`, or
`Noncoplanar`.

## Corresponding MSG

FindSpinGroup also reports the magnetic space group associated with the
identified result.

Common fields:

`msg_symbol`
The BNS magnetic space group symbol.

`msg_bns_number`
The BNS number.

`msg_og_number`
The OG number when available.

`msg_type`
The MSG type.

## Magnetic Phase And Physical Flags

The most useful high-level field is `magnetic_phase`.

Example:

```text
AFM(Altermagnet)
```

Related fields:

`is_alter`
Altermagnet tag when the route identifies altermagnetic behavior.

`is_spin_orbit_magnet`
Spin-orbit-magnet tag when detected.

`properties`
Compact physical-property summary. In the full result, related fields include
spin-splitting and anomalous-Hall-conductivity constraints with and without
spin-orbit coupling.

## Settings And Operation Outputs

A magnetic structure can be represented in several cell settings. Always check
which setting an operation or generated file belongs to before using it in a
downstream program.

`input`
The structure as supplied by the user.

`input_magnetic_primitive`
The magnetic primitive reduction of the input structure.

`database_standard`
The selected `G0std` or `L0std` database-standard setting used by
identify-index.

`convention`
The public convention setting used for OSSG/GSPG presentation.

`acc_primitive`
The ACC magnetic primitive setting used for downstream POSCAR and Brillouin-zone
outputs.

`acc_conventional`
The ACC conventional setting.

Use `result.to_structured_dict()` to see the full result grouped by these
semantic layers.

## Generated Artifacts

The full route can generate text artifacts for downstream workflows.

`result.scif`
Default generated SCIF text.

`result.to_scif(cell_mode=...)`
Generated SCIF text for an explicit output mode.

`result.acc_primitive_magnetic_cell_poscar`
POSCAR text in the ACC primitive magnetic cell.

`result.KPOINTS`
KPOINTS text with spin-polarization labels.

`result.gspg_text`
Compact GSPG text and operation payload.

## Diagnostics

Diagnostic fields explain how the result was produced. They are useful for
debugging, validation, and advanced workflows, but most users should read the
summary fields first.

Examples include `tolerances`, `identify_index_details`, ACC primitive
resolution audits, transform audits, `quasi_2d`, and
`ferroelectric_switching`.
