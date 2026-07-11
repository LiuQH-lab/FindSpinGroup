# CLI By Task

`fsg` is the main single-structure command. It uses progressive disclosure:

1. `fsg FILE` prints the short scientific answer;
2. `--details` expands the human-readable answer;
3. `--show` selects an exact field;
4. `--json` produces machine-readable quick-analysis output;
5. `--full` computes operations, cells, tensors, sites, and artifacts.

Check that the executable and manual version match:

```bash
fsg --version
```

If `--version` itself is unavailable, the executable predates this CLI. Check
which `fsg` is on `PATH`, then install the matching package/manual version.

## Identify A Magnetic Structure

```bash
fsg structure.mcif
```

The default output answers:

- which OSSG was identified;
- which MSG is compatible with SOC;
- whether the moments are collinear/coplanar/noncoplanar;
- the magnetic-phase classification and net moment;
- whether spin splitting and AHC are symmetry-allowed;
- the leading no-SOC and SOC spin-texture wave types.

It intentionally omits G0/L0 decomposition, point-group notation variants,
full basis polynomials, and vector-constraint tables.

## Expand The Human-Readable Summary

```bash
fsg structure.mcif --details
```

Use this when you intentionally want the group components, spin point groups,
ACC/EMPG labels, complete leading spin-texture expressions, and vector
constraints.

## Show Only What You Need

```bash
fsg structure.mcif \
  --show index \
  --show magnetic_phase \
  --show msg_bns_number \
  --show msg_symbol
```

One selected scalar prints directly. Multiple fields print labeled sections.
Nested dictionaries use dot paths:

```bash
fsg structure.mcif --show properties.ss_wo_soc
fsg structure.mcif --show magnetic_phase_details.is_altermagnet
```

Useful quick-analysis fields:

| Question | Field |
| --- | --- |
| OSSG identity | `index`, `ossg_symbol_linear` |
| SOC-compatible MSG | `msg_bns_number`, `msg_symbol`, `msg_type` |
| Moment geometry and phase | `conf`, `magnetic_phase`, `magnetic_phase_details` |
| Spin splitting and AHC | `properties` or its nested fields |
| Leading spin texture | `spin-texture-no-soc`, `spin-texture-soc` |
| Polar/vector constraints | `vector_constraints_by_symmetry` |
| Numerical thresholds | `tolerances` |

The complete inventory is in [CLI Field Guide](cli-show-fields.md).

An unknown or unavailable field is an error and exits nonzero. A field may
require full analysis; retry with `--full` when the error says so.

## Machine-Readable Output

Complete quick-analysis JSON:

```bash
fsg structure.mcif --json
```

One selected field as JSON:

```bash
fsg structure.mcif --json --show properties
```

Results are written to stdout. Auto-selection notices and errors are written to
stderr, so scripts can redirect them independently.

## Full Analysis

Use `--full` (alias: `--all`) only when the requested field/product needs it:

```bash
fsg --full structure.mcif --show operation-views
fsg --full structure.mcif --show magnetic_site_summary
fsg --full slab.mcif --calculation-mode quasi2d --vacuum-axis c --show quasi_2d
```

`--full` requires at least one `--show FIELD`. A bare full result is both too
large for a useful terminal answer and not a stable JSON contract. In Python,
prefer
`MagSymmetryResult.to_structured_dict()` to navigate the complete result by
meaning; it is a Python view rather than a directly JSON-serializable contract.

Common full-only aliases:

| Alias | Product |
| --- | --- |
| `operation-views`, `ops` | Operation-view summary |
| `kpoints` | Generated KPOINTS text |
| `poscar` | ACC-primitive magnetic POSCAR text |
| `scif_default` | Default generated SCIF text |
| `gspg` | Compact GSPG text |
| `wp-chain` | Wyckoff-chain rows |

## Inspect Spin Texture

```bash
fsg structure.mcif --show spin-texture-no-soc
fsg structure.mcif --show spin-texture-soc
```

The output includes the leading wave/order, basis, nullity, spin rank, and
coordinate setting. The free coefficients are not fitted material parameters.

Request per-order bases only when needed:

```bash
fsg structure.mcif \
  --spin-texture-basis-max-order 4 \
  --show spin_texture_config_no_soc.basis_by_order
```

This option sets the search/output ceiling; it is not only a display toggle.

## Generate Matched VASP Inputs

```bash
fsg structure.mcif --write-poscar-kpoints calculation_inputs
```

Writes:

- `calculation_inputs/POSCAR`;
- `calculation_inputs/KPOINTS`.

Both use the same ACC-primitive real-space setting. Existing same-named files
are replaced, so choose a fresh output directory or preserve earlier files.

## Export SCIF

```bash
fsg structure.mcif --write-scif output.scif
```

The default setting is `ssg_convention_oriented`. Choose another explicit
cell/spin-frame mode when required:

```bash
fsg structure.mcif \
  --write-scif magnetic_primitive.scif \
  --scif-cell-mode magnetic_primitive_oriented
```

Use [SCIF Export](scif.md) to choose among convention, magnetic-primitive,
database-standard, and input settings. An existing output file is replaced.

## Export Input-Cell Operations

```bash
fsg -w structure.mcif
```

Writes `ssg_symm.json` and magnetic-primitive POSCAR text in the current
directory. This specialized route exists for downstream programs that require
the supplied cell setting.

Before using a non-primitive input-cell operation list, inspect the written
summary fields:

- `is_input_magnetic_primitive`;
- `input_ssg_may_be_incomplete`;
- `warning`;
- `primitive_ssg_index`.

Existing same-named files are replaced.

## Quasi-2D Analysis

```bash
fsg --full structure.mcif \
  --calculation-mode quasi2d \
  --vacuum-axis c \
  --show quasi_2d
```

The vacuum axis is an input-cell axis. The quasi-2D workflow can adjust
insufficient vacuum before its interpretation path, so inspect its returned
cell/plane diagnostics rather than assuming a simple label change.

## POSCAR And INCAR

For POSCAR-like inputs, the CLI allows and prefers `MAGMOM` from a sibling
`INCAR`. If both embedded POSCAR moments and `INCAR` moments exist, the CLI uses
the `INCAR` values.

The Python API defaults are deliberately different: sibling `INCAR` reading is
off unless explicitly enabled. See [Input Formats](../guide/input-formats.md).

## Numerical Tolerances

```text
--space-tol 0.02
--mtol 0.02
--meigtol 0.00002
--matrix-tol 0.01
--parser-atol 0.02
```

Do not tune all values together. Read
[Parameters And Reliability](../guide/reliability-and-tolerances.md) for the
physical/numerical effect of each parameter.

## Auto-Select An Input

With no `STRUCTURE` argument, `fsg` searches the current directory in this
priority order:

1. SCIF;
2. mCIF;
3. CIF with magnetic-moment tags;
4. `POSCAR` with sibling `INCAR` `MAGMOM`;
5. `POSCAR` with embedded `MAGMOM`;
6. `.vasp`/`.poscar` with embedded `MAGMOM`;
7. `CONTCAR`.

The selected file is reported on stderr. For reproducible scripts, pass the
path explicitly.

## Batch Analysis

Use `fsg-batch` for multiple structures. Batch runs write JSONL records and can
compare against baselines or export selected fields. The operational batch
workflow is documented separately in the repository's batch guide; do not use
`fsg --full` as though `--full` meant “all files.”

## Legacy Options

`--mode`, `--write-ssg-matrices`, `--write-symmetry-dat`, and
`--ssg-matrix-setting` remain for compatibility. New user documentation uses
quick analysis, `--full`, `--show`, explicit artifact writers, and `-w`.
