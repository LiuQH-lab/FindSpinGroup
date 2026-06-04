# Troubleshooting

This page answers common user questions that come up before reading the full
reference.

## Which Function Should I Use?

Use `find_spin_group_basic(...)` for the identified OSSG, MSG, and magnetic
phase.

Use `find_spin_group(...)` for generated artifacts, operation payloads, quasi-2D
diagnostics, tensor outputs, and audit information.

Use `find_spin_group_input_ssg(...)` when another program needs input-cell SSG
or MSG operations.

## Why Is The Full Output So Large?

`find_spin_group(...)` returns `MagSymmetryResult`, which contains summary
fields, multiple cell settings, operation outputs, generated files, physical
properties, and diagnostics.

Use `result.to_summary_dict()` first. Use `result.to_structured_dict()` when
you need the full result grouped by meaning. Use `result.to_dict()` only when
you need the raw compatibility surface.

## Why Do I Get A Missing Magnetic Moment Error?

FindSpinGroup identifies magnetic symmetry. The input must contain magnetic
moments in a supported format.

For POSCAR-like inputs, add embedded `MAGMOM` information or use the CLI from a
VASP directory with a sibling `INCAR` that contains `MAGMOM`.

## Why Does CLI POSCAR Behavior Differ From Python?

The CLI is optimized for VASP working directories and prefers a sibling `INCAR`
`MAGMOM` when present.

Python calls read only the POSCAR content unless you opt into INCAR reading with
`poscar_allow_incar_magmom=True`.

## Does Quasi-2D Change The 3D SSG?

No. Quasi-2D analysis is an additive diagnostic layer. The ordinary 3D
identification remains the base result.

## Which Operation Setting Should I Use?

Use convention-setting outputs for public OSSG symbols and display.

Use input-cell outputs when a downstream tool expects the user-supplied cell.

Use ACC primitive outputs for downstream POSCAR and Brillouin-zone workflows.

Use database-standard outputs only when you need identify-index setting
diagnostics or exact database-standard SCIF export.

## Should I Change Tolerances?

Start with the defaults. Tighten tolerances only when your input structure is
numerically clean enough. Loosen tolerances only when small structural or
magnetic-moment noise prevents a physically expected match.
