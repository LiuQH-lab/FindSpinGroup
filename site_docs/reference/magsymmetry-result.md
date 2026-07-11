# MagSymmetryResult Reference

`find_spin_group(...)` returns a `MagSymmetryResult` object. The object contains
the full analysis result, generated artifacts, and diagnostic details.

Use the accessor methods below before reaching for raw attributes.

## `to_summary_dict`

### Signature

```python
result.to_summary_dict() -> dict
```

### Purpose

Return a compact, display-friendly summary from the full route.

### Returns

Type: `dict`

Main fields:

`index`
Identified OSSG index.

`conf`
Magnetic configuration class.

`phase`
Magnetic phase classification.

`acc`
Spin arithmetic crystal class.

`properties`
Physical-property summary.

`gspg`
GSPG summary and compact text payload.

`spin_texture_config_database`
Spin-texture configuration summary loaded from the runtime database.

`spin_texture_config_no_soc`
Spin-texture configuration from the non-SOC route.

`spin_texture_config_soc`
Spin-texture configuration from the SOC-compatible route.

`vector_constraints_by_symmetry`
Symmetry constraints on vector quantities.

`ferroelectric_switching`
Symmetry-only ferroelectric-switching payload.

## `to_structured_dict`

### Signature

```python
result.to_structured_dict() -> dict
```

### Purpose

Return the full result grouped by semantic layer so Python callers do not have
to infer meaning from flat legacy attribute names.

This is a navigation view, not a fully serialized contract: operation lists,
cells, arrays, and the `legacy` layer can contain Python/NumPy domain objects.
Do not pass the whole result directly to `json.dumps()`.

### Returns

Type: `dict`

Top-level fields:

`summary`
Identifiers, phase fields, spin-texture fields, tolerances, and source
metadata.

`groups`
Input SG, G0, L0, spin point group, GSPG, OSSG, MSG, SSG-by-cell payloads,
MSG-by-cell payloads, and little-group outputs.

`cells`
Input, input magnetic primitive, database standard, convention, ACC primitive,
and ACC conventional cells.

`transforms`
Setting transforms and route audit details.

`properties`
Magnetic phase details, spin splitting, AHC constraints, tensors, magnetic-site
summary, quasi-2D diagnostics, vector constraints, and ferroelectric-switching
outputs.

`artifacts`
Generated POSCAR, SCIF, and KPOINTS text.

`legacy`
Raw attribute dictionary for compatibility with older callers.

The nested field tree is documented in
[StructuredResult](result-schemas/structured-result.md). Use that page when you
need the next layer below `summary`, `groups`, `cells`, `transforms`,
`properties`, or `artifacts`.

## `to_scif`

### Signature

```python
result.to_scif(cell_mode: str = "ssg_convention_oriented") -> str
```

### Purpose

Return generated SCIF text for an explicit cell mode.

See [SCIF Export](scif.md) for all cell modes and aliases.

## `to_dict`

### Signature

```python
result.to_dict() -> dict
```

### Purpose

Return the raw attribute dictionary. This is useful for compatibility and
debugging, but it is not the recommended starting point for new integrations.
Use `to_summary_dict()` for compact display and `to_structured_dict()` for
semantic navigation of the complete Python result.

## Common Direct Attributes

Direct attributes are still available for interactive work and compatibility.
The most commonly used ones are:

`index`
Final identified OSSG index.

`convention_ssg_international_linear`
Public convention OSSG symbol.

`magnetic_phase`
High-level magnetic phase label.

`magnetic_phase_family`, `magnetic_phase_base`
Primary FM/AFM symmetry family and its refined order type.

`magnetic_atom_orbit_count_ssg`
Number of complete-SSG magnetic-atom orbits in the magnetic primitive cell.

`msg_symbol`, `msg_bns_number`, `msg_og_number`
Corresponding MSG identifiers.

`acc`
Spin arithmetic crystal class.

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
