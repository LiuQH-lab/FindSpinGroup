# CLI Reference

The package installs single-case and batch command-line entry points.

## Single-Case Commands

Equivalent entry points:

`fsg`
Primary short command.

`findspingroup`
Long package-name command.

`findspin`
Compatibility command.

## Default Route

```bash
fsg path/to/structure.mcif
```

Runs the compact identification route and prints JSON to stdout.

Use `--show` to print selected fields:

```bash
fsg path/to/structure.mcif --show index --show magnetic_phase
```

`--show` accepts dot paths when the payload is nested.

## Full Route

```bash
fsg --all path/to/structure.mcif
```

Runs the full `MagSymmetryResult` route and prints the serialized payload.

## Write Input-Cell Files

```bash
fsg -w path/to/structure.mcif
```

Runs the input-SSG route and writes files in the current directory.

Written files:

`ssg_symm.json`
Input-cell SSG and MSG operation payload.

`input_poscar.vasp`
Written for non-POSCAR inputs when useful.

`magnetic_primitive_poscar.vasp`
Written when the input cell is not already magnetic primitive.

## Quasi-2D Diagnostics

```bash
fsg --all --calculation-mode quasi2d --vacuum-axis c path/to/slab.mcif
```

`--calculation-mode` requests additive quasi-2D diagnostics.

`--vacuum-axis` names the input-cell axis normal to the slab plane and is
interpreted only for quasi-2D diagnostics.

## Tolerance Flags

`--space-tol`, `--space_tol`
Spatial tolerance.

`--mtol`
Magnetic-moment tolerance.

`--meigtol`
Point-group eigenvalue tolerance.

`--matrix-tol`, `--matrix_tol`
Point-group standardization tolerance.

`--parser-atol`, `--parser_atol`
CIF and SCIF parser expansion tolerance.

## Legacy Mode Selector

`--mode {full,basic,acc-primitive,poscar-ssg,input-ssg}`
Legacy route selector. New CLI usage should prefer the default route, `--all`,
and `-w`.

## Batch Commands

Equivalent entry points:

`fsg-batch`

`findspingroup-batch`

`findspin-batch`

Minimal example:

```bash
fsg-batch tests/testset/mcif_241130_no2186 \
  --output-dir /tmp/findspingroup_batch_smoke \
  --limit 5
```

Export selected fields:

```bash
fsg-batch tests/testset/mcif_241130_no2186 \
  --route basic \
  --output-dir /tmp/findspingroup_basic \
  --export-txt selected.txt \
  --export-field index \
  --export-field phase
```
