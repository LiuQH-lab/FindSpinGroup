# Input Formats

FindSpinGroup requires a magnetic structure. The input must contain magnetic
moments in a supported form.

## Pre-Run Checklist

Before interpreting a result, verify:

1. every intended magnetic site has a moment or is represented through the
   input format's magnetic-site/orbit information;
2. moment components use the coordinate convention expected by the format;
3. occupancies and labels distinguish crystallographically distinct sites;
4. the lattice/coordinates represent the magnetic structure you intend to
   classify, not only its nonmagnetic parent;
5. for POSCAR workflows, you know whether moments came from the file or a
   sibling `INCAR`.

## Supported Files

`.mcif`
Magnetic CIF input. This is the most direct format for magnetic structures.

`.cif`
Ordinary CIF input when magnetic-moment tags are present in a supported form.

`.scif`
Repo-generated spinCIF-style files. Generated SCIF files can be parsed back
through the public input path.

POSCAR-like files
Supported names and suffixes include `POSCAR`, `CONTCAR`, `.vasp`, `.poscar`,
and explicit ordinary filenames without a CIF/SCIF suffix. In other words,
when a file path is supplied explicitly, `.cif` and `.mcif` are parsed as CIF,
`.scif` is parsed as SCIF, and other names are treated as POSCAR-like input.

## Magnetic Moments

Magnetic moments must be present. Without moments, the program cannot identify
magnetic symmetry.

CIF, mCIF, and SCIF moments are converted into the route's Cartesian input-cell
frame before operation export.

POSCAR moments are treated as Cartesian.

The moment scale is interpreted in μB for tolerance and net-moment reporting.
Check the source format carefully when moments were originally supplied in a
crystallographic/lattice basis; a correct numerical triplet in the wrong frame
can produce a plausible but incorrect symmetry result.

For a first diagnostic, print the parsed classification context rather than
immediately changing tolerances:

```bash
fsg structure.mcif \
  --show conf \
  --show net_moment \
  --show magnetic_phase_details
```

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

For reproducible calculations, state the source of the moments explicitly. In
a Python integration, prefer embedded POSCAR moments unless reading the sibling
`INCAR` is an intentional part of the input contract.

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

Automatic discovery intentionally does not scan arbitrary ordinary filenames;
pass such POSCAR-like files explicitly:

```bash
fsg magnetic_input
```

When multiple candidates are found, the CLI reports which file it selected.

Auto-selection is convenient interactively. Scripts and published workflows
should pass an explicit path so that a new file in the working directory cannot
silently change the selected input.

## Input Validity Versus Physical Validity

A successfully parsed file is not automatically a physically reliable model.
FindSpinGroup analyzes the ordered moments it receives. It cannot decide
whether the structure is energetically stable, whether domains are populated,
or whether a refined small moment is significant. Those judgments belong to
the input provenance and the accompanying tolerance-sensitivity analysis.
