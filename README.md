# FindSpinGroup

FindSpinGroup is a Python package and command-line program for identifying
oriented spin space group symmetry in magnetic crystal structures.

It is designed for research workflows where the same magnetic structure must be
understood in both spin-space-group and magnetic-space-group language. Given a
magnetic crystal structure, FindSpinGroup identifies the spin symmetry,
constructs standard and ACC primitive settings, and prepares symmetry data for
first-principles, high-throughput, and web-application workflows.

The project also powers the public web application:

[https://app.findspingroup.com](https://app.findspingroup.com)

## What It Does

FindSpinGroup takes a magnetic structure and organizes its symmetry information
across the main settings used by the project:

- the input structure and its magnetic primitive cell;
- the database standard setting used for spin-space-group identification;
- the public convention setting used for symbols and operation display;
- the ACC primitive cell used for downstream Brillouin-zone and POSCAR output.

The full route can also prepare SCIF, POSCAR, KPOINTS, Wyckoff-splitting,
magnetic-site, quasi-2D, tensor, and ferroelectric-switching data. Detailed
field-level output contracts are intentionally kept in the documentation rather
than in this README.

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
fsg path/to/structure.mcif
```

Run the full analysis route:

```bash
fsg --all path/to/structure.mcif
```

Use the Python API when the result will be consumed by another program:

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))
summary = result.to_summary_dict()
```

## Choosing A Route

Most users should start with the default CLI command or
`find_spin_group_basic(...)`. It is the fast identification route and returns a
compact summary.

Use the full route, `fsg --all` or `find_spin_group(...)`, when you need the
complete result object, generated artifacts, quasi-2D diagnostics, tensor
analysis, or route-specific audit information.

Use `fsg -w` when you want input-setting symmetry helper files written into the
current directory. This route is intended for downstream tools that consume
explicit operation files.

The legacy `--mode` selector is still available for compatibility, but new CLI
usage should prefer the default route, `--all`, and `-w`.

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

## Batch And Validation Tools

Batch entry points are included for regression validation and high-throughput
screening:

```bash
fsg-batch --help
findspingroup-batch --help
```

Repository scripts under `scripts/` provide additional project-maintenance
workflows, such as batch submission, tolerance scans, POSCAR roundtrip checks,
and Excel exports. These scripts are mainly intended for project development
and database validation.

## Documentation

The README is only the entry point. User-facing documentation lives in `site_docs/` and is published through Read the Docs. Development notes, batch logs, and local validation records should stay outside tracked documentation.

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
