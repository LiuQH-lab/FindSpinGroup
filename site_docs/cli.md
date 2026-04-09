# CLI

The package installs command-line entry points for single-case and batch use.

## Single-case CLI

Installed entry points include:

- `findspingroup`
- `findspin`
- `fsg`

Use `--help` to inspect the current interface:

```bash
findspingroup --help
```

## Batch CLI

Installed batch entry points include:

- `findspingroup-batch`
- `findspin-batch`
- `fsg-batch`

Show help:

```bash
findspingroup-batch --help
```

## Minimal batch example

```bash
findspingroup-batch tests/testset/mcif_241130_no2186 \
  --output-dir /tmp/findspingroup_batch_smoke \
  --limit 5
```

Common options:

- `--output-dir`
- `--baseline`
- `--auto-baseline`
- `--space-tol`
- `--mtol`
- `--meigtol`
- `--matrix-tol`
- `--export-field`
- `--export-txt`
