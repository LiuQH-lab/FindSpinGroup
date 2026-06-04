# MagSymmetryResult

`find_spin_group(...)` returns a `MagSymmetryResult` object. The object contains
the full analysis result, generated artifacts, and diagnostics.

## Recommended Accessors

### `to_summary_dict()`

Returns a compact dictionary with the most useful human-facing fields.

Main contents:

`index`
Identified OSSG index.

`conf`
Magnetic configuration class.

`phase`
Magnetic phase classification.

`acc`
Arithmetic crystal class.

`properties`
Physical-property summary.

`gspg`
Compact GSPG summary and text payload.

`polar_axes_by_symmetry`
Polar-axis information inferred by symmetry.

`ferroelectric_switching`
Symmetry-only ferroelectric switching payload.

### `to_structured_dict()`

Returns the full result grouped by semantic layer. This is the preferred full
output for new integrations.

Top-level groups:

`summary`
Identifiers, phase, tolerances, and source metadata.

`groups`
Input SG, G0, L0, spin point group, GSPG, OSSG, MSG, SSG-by-cell, MSG-by-cell,
and little-group payloads.

`cells`
Input, input magnetic primitive, database standard, convention, ACC primitive,
and ACC conventional cells.

`transforms`
Setting transforms and route audits.

`properties`
Magnetic phase, spin splitting, AHC, tensors, magnetic-site summary, quasi-2D,
polar-axis, and ferroelectric-switching outputs.

`artifacts`
Generated POSCAR, SCIF, and KPOINTS text.

`legacy`
Raw attribute dictionary for compatibility with older callers.

### `to_scif(cell_mode=...)`

Returns generated SCIF text for the requested cell mode. See
[SCIF Export](scif.md).

### `to_dict()`

Returns the raw attribute dictionary. This is useful for compatibility and
debugging, but it is not the recommended starting point for new integrations.
Prefer `to_summary_dict()` or `to_structured_dict()`.

## Common Direct Attributes

`index`
Final identified OSSG index.

`convention_ssg_international_linear`
Public convention OSSG symbol.

`magnetic_phase`
High-level magnetic phase label.

`msg_symbol`, `msg_bns_number`, `msg_og_number`
Corresponding MSG identifiers.

`acc`
Arithmetic crystal class.

`scif`
Default SCIF text.

`scif_outputs`
SCIF text keyed by cell mode.

`acc_primitive_magnetic_cell_poscar`
Generated POSCAR text in ACC primitive setting.

`KPOINTS`
Generated KPOINTS text.

`quasi_2d`
Additive quasi-2D diagnostics when requested; otherwise `None` in ordinary 3D
mode.
