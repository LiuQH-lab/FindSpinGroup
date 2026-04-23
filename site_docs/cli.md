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

## Write Input-Cell SSG Files

Input: a supported magnetic structure file.

```bash
fsg -w path/to/structure.mcif
```

Output: files written in the current directory:

- `ssg_symm.json`
- `input_poscar.vasp`, for non-POSCAR inputs
- `magnetic_primitive_poscar.vasp`, when the input cell is not magnetic primitive

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
