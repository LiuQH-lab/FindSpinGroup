# Choosing an API

Most users should choose one of three routes.

## Quick Identification

Use `find_spin_group_basic(...)` in Python or `fsg file.mcif` on the command
line when you need the identified OSSG, corresponding MSG, and magnetic-phase
summary.

Choose this route when:

- you are screening many structures;
- you only need labels and summary classification;
- you want a JSON-serializable dictionary;
- you do not need generated SCIF, POSCAR, KPOINTS, or operation text.

Python:

```python
from findspingroup import find_spin_group_basic

summary = find_spin_group_basic("path/to/structure.mcif")
print(summary["index"])
print(summary["magnetic_phase"])
```

CLI:

```bash
fsg path/to/structure.mcif
```

The default CLI output is a short summary. Use `--show` for selected fields and
`--json` for the complete compact JSON payload.

## Full Analysis

Use `find_spin_group(...)` in Python or `fsg --all file.mcif` when you need the
full `MagSymmetryResult`.

Choose this route when:

- you need generated SCIF text;
- you need POSCAR or KPOINTS output;
- you need SSG or MSG operations in named settings;
- you need tensor, magnetic-site, quasi-2D, or ferroelectric-switching outputs;
- you need route diagnostics for debugging or validation.

Python:

```python
from findspingroup import find_spin_group

result = find_spin_group("path/to/structure.mcif")
summary = result.to_summary_dict()
structured = result.to_structured_dict()
```

Prefer `to_summary_dict()` for a compact human-facing summary and
`to_structured_dict()` when another program needs the full result grouped by
meaning. Treat `to_dict()` as the raw compatibility surface.

## Input-Cell Operations

Use `find_spin_group_input_ssg(...)` in Python or `fsg -w file.mcif` when a
downstream tool needs explicit operations in the input-cell setting.

Choose this route when:

- you want operations in the cell supplied by the user;
- you want `ssg_symm.json` written by the CLI;
- you need the relation between the input cell and the input magnetic primitive
  cell;
- you need helper POSCAR text for the magnetic primitive cell.

CLI:

```bash
fsg -w path/to/structure.mcif
```

This writes `ssg_symm.json` and `magnetic_primitive_poscar.vasp`.
`input_poscar.vasp` is written only when a non-POSCAR input cell is distinct
from the magnetic primitive cell.

## Advanced Routes

The package also exposes `find_spin_group_acc_primitive(...)`,
`find_spin_group_poscar_ssg(...)`, and `*_from_data` variants. These are useful
for specialized integration and development workflows, but they are not the
recommended starting points for new users.
