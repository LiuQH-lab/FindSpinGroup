# FindSpinGroup

FindSpinGroup is a Python package and command-line program for identifying
oriented spin space group symmetry in magnetic crystal structures.

Given a structure with magnetic moments, FindSpinGroup identifies the oriented
spin space group (OSSG), reports the corresponding magnetic space group (MSG),
classifies the magnetic phase, and can generate symmetry data for
first-principles, high-throughput, and web-application workflows.

The project also powers the public web application:

[https://app.findspingroup.com](https://app.findspingroup.com)

## Installation

```bash
pip install findspingroup
```

FindSpinGroup requires Python 3.11 or newer.

For development from a local checkout:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Run a compact identification from the command line:

```bash
fsg path/to/structure.mcif --show index --show magnetic_phase --show msg_symbol
```

Use the Python API when the result will be consumed by another program:

```python
from findspingroup import example_path, find_spin_group_basic

summary = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print(summary["index"])
print(summary["magnetic_phase"])
print(summary["msg_bns_number"], summary["msg_symbol"])
```

Expected output:

```text
194.164.1.1.L
AFM(Altermagnet)
63.457 Cmcm
```

## Main Routes

Most users should start with the default CLI command or
`find_spin_group_basic(...)`. This route returns the identified OSSG,
corresponding MSG, magnetic phase, and compact property flags.

Use the full route, `fsg --all` or `find_spin_group(...)`, when you need the
complete result object, generated artifacts, quasi-2D diagnostics, tensor
analysis, or route-specific audit information.

Use `fsg -w` when you want input-setting symmetry helper files written into the
current directory. This route is intended for downstream tools that consume
explicit operation files.

The legacy `--mode` selector is still available for compatibility, but new CLI
usage should prefer the default route, `--all`, and `-w`.

## Main Outputs

Depending on the route, outputs can include:

- identified OSSG index and symbols;
- corresponding MSG identifiers;
- magnetic-phase and spin-splitting classification;
- SSG and MSG operations in named settings;
- SCIF, POSCAR, KPOINTS, and GSPG text artifacts;
- magnetic-site, tensor, quasi-2D, and ferroelectric-switching diagnostics.

## Supported Inputs

FindSpinGroup supports:

- magnetic CIF / mCIF files;
- ordinary CIF files when magnetic information is available in supported tags;
- repo-generated SCIF files;
- POSCAR-like files, including `POSCAR`, `CONTCAR`, `.vasp`, and `.poscar`.

Magnetic moments must be present. For POSCAR-like files, the command-line tool
is optimized for VASP working directories and prefers a sibling `INCAR` MAGMOM
when available. Direct Python calls read only the POSCAR content unless INCAR
reading is explicitly enabled, which keeps scripted calls reproducible.

When no input file is given, the CLI auto-selects from the current directory in
this order: SCIF, mCIF, CIF files with magnetic-moment tags, `POSCAR` with
sibling `INCAR` MAGMOM, `POSCAR` with embedded MAGMOM, `.vasp` / `.poscar`
files with embedded MAGMOM, and finally `CONTCAR`.

## Quasi-2D Calculations

The default calculation mode is three-dimensional. Quasi-2D analysis is an
explicit full-route diagnostic:

```bash
fsg --all --calculation-mode quasi2d --vacuum-axis c path/to/slab.mcif
```

The ordinary 3D symmetry result remains the base result. Quasi-2D data is added
only when the quasi-2D calculation mode is requested.

## Documentation

The README is only the entry point. Detailed documentation is split into:

- Guide: how to choose a route, run examples, and understand results;
- Reference: exact Python functions, return objects, CLI flags, and output
  fields;
- Advanced Notes: settings, diagnostics, and validation terminology.

Local source files live under [`site_docs/`](site_docs/index.md) and are
published through Read the Docs.

## Development

Run focused tests with:

```bash
PYTHONPATH=src python -m pytest
```

For changes that affect symmetry identification, generated cells, route output,
or public fields, use focused local regression tests first and then run the
agreed project batch validations.

## License

FindSpinGroup is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
