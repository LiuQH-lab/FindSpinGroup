# Input Formats

FindSpinGroup requires a magnetic structure. The input must contain magnetic
moments in a supported form.

## Supported Files

`.mcif`
Magnetic CIF input. This is the most direct format for magnetic structures.

`.cif`
Ordinary CIF input when magnetic-moment tags are present in a supported form.

`.scif`
Repo-generated spinCIF-style files. Generated SCIF files can be parsed back
through the public input path.

POSCAR-like files
Supported names and suffixes include `POSCAR`, `CONTCAR`, `.vasp`, and
`.poscar`.

## Magnetic Moments

Magnetic moments must be present. Without moments, the program cannot identify
magnetic symmetry.

CIF, mCIF, and SCIF moments are converted into the route's Cartesian input-cell
frame before operation export.

POSCAR moments are treated as Cartesian.

## POSCAR And INCAR Behavior

For POSCAR-like inputs, the CLI is optimized for VASP working directories. It
prefers a sibling `INCAR` `MAGMOM` when available.

Direct Python calls read only the POSCAR content unless INCAR reading is
explicitly enabled with:

```python
find_spin_group(
    "POSCAR",
    poscar_allow_incar_magmom=True,
    poscar_prefer_incar_magmom=True,
)
```

This difference keeps scripted Python calls reproducible while preserving the
convenient CLI behavior for VASP directories.

## CLI Auto-Selection

When no input file is given, the CLI searches the current directory and selects
a readable structure file in this order:

1. SCIF
2. mCIF
3. CIF files with magnetic-moment tags
4. `POSCAR` with sibling `INCAR` `MAGMOM`
5. `POSCAR` with embedded `MAGMOM`
6. `.vasp` or `.poscar` files with embedded `MAGMOM`
7. `CONTCAR`

When multiple candidates are found, the CLI reports which file it selected.
