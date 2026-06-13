# Getting Started

This page gets you from installation to a first identified spin space group.

## Install

Install from PyPI:

```bash
pip install findspingroup
```

FindSpinGroup requires Python 3.11 or newer.

For development from a local checkout:

```bash
git clone https://github.com/LiuQH-lab/FindSpinGroup.git
cd FindSpinGroup
python -m pip install -e ".[dev]"
```

## First Python Example

Use the bundled MnTe example to verify that the package works:

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

This quick route returns a JSON-serializable dictionary with the main OSSG, MSG,
and magnetic-phase fields.

## First CLI Example

For your own magnetic structure file:

```bash
fsg path/to/structure.mcif
```

The default `fsg` route runs the compact identification path and prints a short
human-readable summary. Use `--show` for selected fields:

```bash
fsg path/to/structure.mcif --show index --show magnetic_phase --show msg_symbol
```

Use `--json` when another program needs the complete compact JSON payload.

## When The First Example Fails

The most common cause is missing magnetic moments. FindSpinGroup identifies
magnetic symmetry, so `.mcif`, `.cif`, `.scif`, and POSCAR-like inputs must
provide magnetic-moment information in a supported form.

POSCAR-like inputs are special. The CLI is optimized for VASP working
directories and can prefer a sibling `INCAR` `MAGMOM` when present. Direct
Python calls read only the POSCAR content unless INCAR reading is explicitly
enabled.

See [Input Formats](input-formats.md) and
[Troubleshooting](troubleshooting.md) for details.
