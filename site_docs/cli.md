# CLI

The package installs command-line entry points for single-case and batch use.

Single-case entry points:

- `fsg`
- `findspingroup`
- `findspin`

Use help to inspect the current interface:

```bash
fsg --help
```

## Lightweight Summary

Input: a supported magnetic structure file.

```bash
fsg path/to/structure.mcif
```

Output: a lightweight JSON summary printed to stdout.

Use `--show` to print selected fields:

```bash
fsg path/to/structure.mcif --show index --show magnetic_phase
```

Use `--all` for the full `MagSymmetryResult` payload:

```bash
fsg --all path/to/structure.mcif
```

## Write Input-Cell SSG Files

Input: a supported magnetic structure file.

```bash
fsg -w path/to/structure.mcif
```

Output: files written in the current directory:

- `ssg_symm.json`
- `input_poscar.vasp`, for non-POSCAR inputs
- `magnetic_primitive_poscar.vasp`, when the input cell is not magnetic primitive

## Quasi-2D Diagnostics

The command-line default is ordinary 3D identification. Request quasi-2D
diagnostics explicitly:

```bash
fsg --all --calculation-mode quasi2d --vacuum-axis c path/to/slab.mcif
```

`--vacuum-axis` names the input-cell axis normal to the intended 2D slab and is
interpreted only for quasi-2D diagnostics.

## Batch CLI

Batch entry points:

- `fsg-batch`
- `findspingroup-batch`
- `findspin-batch`

Show help:

```bash
fsg-batch --help
```

Minimal batch example:

```bash
fsg-batch tests/testset/mcif_241130_no2186 \
  --output-dir /tmp/findspingroup_batch_smoke \
  --limit 5
```

Run a lightweight 3D batch:

```bash
fsg-batch tests/testset/mcif_241130_no2186 \
  --route basic \
  --output-dir /tmp/findspingroup_basic \
  --export-txt selected.txt \
  --export-field index \
  --export-field phase
```

Run a full quasi-2D batch and export compact diagnostic fields:

```bash
fsg-batch tests/testset/quasi2d_small \
  --route full \
  --calculation-mode quasi2d \
  --vacuum-axis c \
  --output-dir /tmp/findspingroup_quasi2d \
  --export-txt selected_2d.txt \
  --export-field index \
  --export-field quasi_2d.status \
  --export-field quasi_2d.kpoints \
  --export-field quasi_2d.generic_point_comparison.summary
```

Common tolerance flags are `--space-tol`, `--mtol`, `--meigtol`, and
`--matrix-tol`. The single-case and batch CLIs also keep the older underscore
spellings, such as `--space_tol` and `--matrix_tol`, as compatibility aliases.
