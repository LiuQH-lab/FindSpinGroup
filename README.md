# FINDSPINGROUP

`findspingroup` is a Python toolkit, command-line program, and
[web application](https://app.findspingroup.com) for identifying and inspecting
oriented spin space group (OSSG) symmetry in magnetic crystal structures.

It is designed for research workflows involving the interplay between
exchange-driven magnetic geometry and spin-orbit coupling, which are described
by spin space group (SSG) and magnetic space group (MSG) frameworks,
respectively.

Given a magnetic structure, FINDSPINGROUP identifies the OSSG, derives the
corresponding MSG, and organizes crystallographic and physical information
needed to analyze the material with and without spin-orbit coupling.

Main outputs can include:

- OSSG information and corresponding MSG information in matched settings;
- spin Wyckoff positions and Wyckoff splitting from space group (SG) to OSSG and MSG;
- spin Brillouin zones, high-symmetry k points, and symmetry-allowed spin-polarization components;
- magnetic-phase classification, including unconventional cases such as altermagnets and spin-orbit magnets;
- symmetry constraints on anomalous Hall conductivity, nonlinear tensors, and related physical responses;
- chiral and polar group information with and without spin-orbit coupling;
- `.scif` files for downstream spin-group-based tensor analysis and data exchange;
- magnetic primitive-cell POSCAR files in relevant coordinate conventions;
- KPOINTS files labeled with symmetry-allowed spin-polarization components.

## Installation

```bash
pip install findspingroup
```

Python `>= 3.11` is required.

## Quick Start

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

print(result.index)
print(result.convention_ssg_international_linear)
print(result.magnetic_phase)
```

## Command Line

After installation, the package provides the `fsg` command.

Print a lightweight summary for a file:

```bash
fsg path/to/structure.mcif
```

Input: a supported magnetic structure file, such as `.mcif`, `.scif`, or a
POSCAR-like file with embedded `MAGMOM`.
Output: a lightweight JSON summary printed to stdout.

Write input-cell SSG / MSG operations and POSCAR helper files:

```bash
fsg -w path/to/structure.mcif
```

Input: a supported magnetic structure file. Output: files written in the current
directory:

- `ssg_symm.json`
- `input_poscar.vasp`, for non-POSCAR inputs
- `magnetic_primitive_poscar.vasp`, when the input cell is not magnetic primitive

Use help to inspect the current command-line options:

```bash
fsg --help
```

## Python APIs

### Full Analysis

```python
from findspingroup import find_spin_group

result = find_spin_group("path/to/structure.mcif")

print(result.index)
print(result.acc)
print(result.convention_ssg_international_linear)
print(result.magnetic_phase)
print(result.gspg_text["symbol"])
print(result.to_scif(cell_mode="ssg_convention_oriented"))
```

`find_spin_group(...)` returns a `MagSymmetryResult` object with the full
analysis result. See the
[usage documentation](https://findspingroup.readthedocs.io/en/latest/usage/)
for the main `MagSymmetryResult` attributes and route-specific outputs.

SCIF export modes are documented at
[SCIF](https://findspingroup.readthedocs.io/en/latest/scif/). Current modes are
`ssg_convention_oriented`, `ssg_convention_cartesian`,
`database_standard_oriented`, `database_standard_cartesian`,
`magnetic_primitive_oriented`, `magnetic_primitive_cartesian`,
`input_oriented`, and `input_cartesian`. Legacy aliases include
`magnetic_primitive`, `input_identified`, and `database_standard`.

`database_standard_*` SCIF modes emit operations in the selected SSG database
standard setting: `G0std` for non-type-k results and `L0std` for type-k results.
The emitted file records its real-space setting and spin-frame setting in
repo-local `_space_group_spin.fsg_*` tags.

### Quasi-2D Diagnostics

The default API mode is ordinary 3D identification. To add quasi-2D
spin-splitting diagnostics without changing the base SSG/MSG/ACC route, request
the mode explicitly:

```python
result = find_spin_group(
    "path/to/slab.mcif",
    calculation_mode="quasi2d",
    vacuum_axis="c",
)

print(result.quasi_2d["spin_splitting_2d"])
print(result.quasi_2d["kpoints"])
```

In 3D mode, `result.quasi_2d` is `None`. `vacuum_axis` names the input-cell
axis normal to the intended 2D slab and is only interpreted for quasi-2D
diagnostics.

### Lightweight Basic Summary

```python
from findspingroup import find_spin_group_basic

summary = find_spin_group_basic("path/to/structure.mcif")
print(summary["index"])
print(summary["magnetic_phase"])
```

This route avoids expensive downstream outputs that are not needed for simple
identification.

### Input-Cell SSG Operations

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("path/to/structure.mcif")

print(payload["summary"])
print(payload["ssg"]["ops"])
print(payload["msg"]["ops"])
```

This route returns SSG operations in the **input cell setting**. If the input
cell is not already magnetic primitive, those input-cell operations may be fewer
than the full symmetry operations of the magnetic primitive cell. In that case,
the payload includes primitive-side identifiers and a warning.

## Supported Inputs

`findspingroup` supports:

- `.cif`
- `.mcif`
- repo-generated `.scif`
- POSCAR-like files with embedded magnetic moments

Input notes:

- Magnetic inputs must contain explicit magnetic moments.
- POSCAR inputs must include an embedded `MAGMOM` payload, for example a trailing
  `# MAGMOM=...` line.
- The input-SSG POSCAR route does not read `INCAR`.
- POSCAR moments are treated as Cartesian.
- CIF, mCIF, and SCIF moments are converted into the route's Cartesian input-cell
  frame before operation export.

## Tolerances

The main APIs accept the same basic tolerance controls:

```python
find_spin_group(
    "path/to/structure.mcif",
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
)
```

`space_tol` controls real-space matching, `mtol` controls magnetic-moment
matching and zero-net-moment classification thresholds, `meigtol` controls
point-group eigenvalue decisions, and `matrix_tol` controls point-group
standardization matrix checks. Use tighter tolerances only when the input
structure is numerically clean enough to support them.

## License

This project is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
