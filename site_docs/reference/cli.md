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

Runs the compact identification route and prints a short human-readable summary.

The default summary includes:

- identification: OSSG symbol and index, G0/L0 components, identify
  `t_index`/`k_index`, spin-space point group and nontrivial spin-space point
  group in HM and Schoenflies notation, space group, magnetic space group, spin
  arithmetic crystal class, and EMPG;
- magnetic phase and properties: configuration, phase, net moment, spin
  splitting, AHC, altermagnet / spin-orbit-magnet flags, and no-SOC / SOC spin
  texture wave and basis summaries;
- vector constraints: polar/chiral flags plus compact vector-constraint
  summaries by SG, OSSG/G0, and MSG.

Use `--json` when another program needs the complete compact JSON payload:

```bash
fsg path/to/structure.mcif --json
```

Use `--show` to print selected fields:

```bash
fsg path/to/structure.mcif --show index --show magnetic_phase
```

Plain `--show` prints human-readable output. Scalars are printed directly,
operation views are shown as tables, spin-texture dictionaries are shown as
short labeled blocks, and long text artifacts such as SCIF, POSCAR, KPOINTS,
and GSPG text are printed as text.

Use `--json --show FIELD` when a script needs the selected field as JSON.

`--show` accepts dot paths when the payload is nested. It also accepts built-in
aliases for common fields and artifacts.

Built-in aliases:

| Alias | Resolved field | Route | Output |
| --- | --- | --- | --- |
| `kpoints`, `kpoints_text` | `KPOINTS` | full | KPOINTS text |
| `poscar`, `primitive_poscar`, `primitive-poscar` | `acc_primitive_magnetic_cell_poscar` | full | ACC magnetic primitive POSCAR text |
| `acc_primitive_poscar`, `acc-primitive-poscar` | `acc_primitive_magnetic_cell_poscar` | full | ACC magnetic primitive POSCAR text |
| `scif_default`, `default_scif` | `scif` | full | default SCIF text |
| `gspg` | `gspg_text` | full | GSPG text block |
| `operation-views`, `ops` | `operation_views` | full | operation-view summary table |
| `wp-chain`, `wyckoff-chain` | `wp_chain` | full | Wyckoff-chain rows |
| `spin-texture-no-soc` | `spin_texture_config_no_soc` | basic or full | spin-texture block |
| `spin-texture-soc` | `spin_texture_config_soc` | basic or full | spin-texture block |

For the complete route-specific field list, see
[CLI Show Fields](cli-show-fields.md).

Common direct fields:

| Field | Route | Output |
| --- | --- | --- |
| `index` | basic or full | OSSG index |
| `ossg_symbol_linear` | basic or full | linear OSSG symbol |
| `conf` | basic or full | magnetic configuration class |
| `magnetic_phase` | basic or full | magnetic phase |
| `empg` | basic or full | effective magnetic point group symbol |
| `msg_symbol` | basic or full | BNS MSG symbol |
| `msg_bns_number` | basic or full | BNS MSG number as text |
| `properties` | basic or full | compact property block |
| `vector_constraints_by_symmetry` | basic or full | vector constraints by SG/OSSG/MSG |
| `magnetic_site_summary` | full | magnetic-site orbit and DOF summary |
| `quasi_2d` | full quasi-2D | quasi-2D diagnostics |
| `scif_outputs.<mode>` | full | one SCIF text artifact |
| `operation_views.<setting>.views.<view>` | full | one operation view table |

Examples:

```bash
fsg examples/0.800_MnTe.mcif --show index --show magnetic_phase
```

Output:

```text
## index
194.164.1.1.L

## magnetic_phase
AFM(Altermagnet)
```

Spin-texture example:

```bash
fsg --all examples/0.800_MnTe.mcif --show spin-texture-no-soc
```

Output:

```text
spin_texture_type: g-wave
momentum_space_spin_configuration: collinear
spin_rank: 1
nullity: 1
order: 4
source: ossg_unit_cartesian_generators
basis_setting: ossg_unit_cartesian
basis:
  1. C1*((-sqrt(3)/9*ky^3*kz)*sigma_x + (sqrt(3)/3*kx^2*ky*kz)*sigma_x - (1/3*ky^3*kz)*sigma_y + (kx^2*ky*kz)*sigma_y) + o(k^4)
basis_latex:
  1. C_{1}\left(-\frac{\sqrt{3}}{9}k_{y}^{3}k_{z}\,\sigma_{x} + \frac{\sqrt{3}}{3}k_{x}^{2}k_{y}k_{z}\,\sigma_{x} - \frac{1}{3}k_{y}^{3}k_{z}\,\sigma_{y} + k_{x}^{2}k_{y}k_{z}\,\sigma_{y}\right) + o(k^{4})
classifier_tolerances:
  rtol: 1e-08
  atol: 1e-10
  zero_tol: 1e-08
```

Operation-view summary example:

```bash
fsg --all examples/0.800_MnTe.mcif --show operation-views
```

Output excerpt:

```text
Setting                      | Default | View              | Ops | Label
-----------------------------+---------+-------------------+-----+------------------
convention_cartesian         |         | all               | 24  | All operations
convention_cartesian         | yes     | nssg              | 24  | nSSG operations
convention_cartesian         |         | generators        | 4   | Symbol generators
convention_cartesian         |         | pure_translations | 1   | Pure translations
convention_cartesian         |         | spin_translations | 1   | Spin translations
...
```

Single operation view example:

```bash
fsg --all examples/0.800_MnTe.mcif --show operation_views.convention_oriented.views.msg
```

Output excerpt:

```text
MSG operations
operation_count: 8
indices: 1, 2, 3, 4, 5, 6, 7, 8

No. | Spin Rotation                      | Space Rotation                     | Space Translation | Seitz Symbol
----+------------------------------------+------------------------------------+-------------------+-------------
1   | [[1, 0, 0]; [0, 1, 0]; [0, 0, 1]]  | [[1, 0, 0]; [0, 1, 0]; [0, 0, 1]]  | [0, 0, 0]         | \left\{ 1 \,\middle\|\, 1 \,\middle|\, 0,0,0 \right\}
2   | [[0, 1, 0]; [1, 0, 0]; [0, 0, -1]] | [[0, -1, 0]; [-1, 0, 0]; [0, 0, 1]] | [0, 0, 0]         | \left\{ 2_{110} \,\middle\|\, m_{110} \,\middle|\, 0,0,0 \right\}
...
```

Long text artifact example:

```bash
fsg --all examples/0.800_MnTe.mcif --show kpoints
```

Output excerpt:

```text
Generated by seekpath and findspingroup v0.15.6 (*** for spin splitting w/o SOC; ^^^ for spin splitting w/ SOC)
 40
Line-mode
Reciprocal
  0.000000   0.000000   0.000000 ! Γ
  0.500000   0.000000   0.000000 ! M ^^^     | Σ ^^^
```

## Full Route

```bash
fsg --all path/to/structure.mcif
```

Runs the full `MagSymmetryResult` route and prints the serialized payload.

## Write SCIF

```bash
fsg path/to/structure.mcif --write-scif output.scif
```

Runs the full route once and writes a SCIF file. The default SCIF mode is
`ssg_convention_oriented`.

Choose another available SCIF setting with `--scif-cell-mode`:

```bash
fsg path/to/structure.mcif \
  --write-scif magnetic_primitive.scif \
  --scif-cell-mode magnetic_primitive_oriented
```

Common `--scif-cell-mode` values:

- `ssg_convention_oriented`
- `ssg_convention_cartesian`
- `magnetic_primitive_oriented`
- `magnetic_primitive_cartesian`
- `database_standard_oriented`
- `database_standard_cartesian`
- `input_oriented`
- `input_cartesian`

Legacy aliases `magnetic_primitive`, `database_standard`, and
`input_identified` are also accepted.

## Write POSCAR And KPOINTS

```bash
fsg path/to/structure.mcif --write-poscar-kpoints calc_inputs
```

Runs the full route once and writes:

`calc_inputs/POSCAR`
ACC magnetic primitive POSCAR.

`calc_inputs/KPOINTS`
KPOINTS in the same acc-primitive real-space setting.

`--write-scif` and `--write-poscar-kpoints` can be used in one command; the
structure is still analyzed only once.

## Write Input-Cell Files

```bash
fsg -w path/to/structure.mcif
```

Runs the input-SSG route and writes files in the current directory.

Written files:

`ssg_symm.json`
Input-cell SSG and MSG operation payload.

`input_poscar.vasp`
Written for non-POSCAR inputs when the input cell is distinct from the magnetic
primitive cell.

`magnetic_primitive_poscar.vasp`
Written for the magnetic primitive cell. If the input cell is already magnetic
primitive, `input_poscar.vasp` is omitted to avoid a duplicate.

## Quasi-2D Diagnostics

```bash
fsg --all --calculation-mode quasi2d --vacuum-axis c path/to/slab.mcif
```

`--calculation-mode` requests additive quasi-2D diagnostics.

`--vacuum-axis` names the input-cell axis normal to the slab plane and is
interpreted only for quasi-2D diagnostics.

## Spin-Texture Basis Order

```bash
fsg --all path/to/structure.mcif --spin-texture-basis-max-order 4 \
  --show spin_texture_config_no_soc.basis_by_order
```

By default, the CLI prints only the leading allowed spin-texture basis. Set
`--spin-texture-basis-max-order N` to include `basis_by_order` entries from
order 0 through order `N` in the computed no-SOC and SOC spin-texture
configuration fields. This option is mainly for diagnostics and can increase
runtime on large or high-symmetry cases.

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

`--export-field` accepts the same dot-path style used by `--show`. The selected
fields are written into the text export for quick inspection.

The batch JSONL files keep the complete per-case payloads:

`records.jsonl`
One record per processed input, including status, duration, error metadata, and
compact result fields.

`full_results.jsonl`
Serialized result payloads. Use this when a field is not part of the compact
record schema.

Excel exports are produced by `scripts/export_mcif_results_to_excel.py`. The
stable column contract lives in `src/findspingroup/output_schema.py`; the script
extracts those fields from `full_results.jsonl`, formats nested payloads, and
adds auxiliary sheets such as magnetic-site orbit details when available.

## Python API Equivalents

Use the Python API when another program should consume structured results
directly:

```python
from findspingroup import find_spin_group_basic, find_spin_group

basic = find_spin_group_basic("path/to/structure.mcif")
full = find_spin_group("path/to/structure.mcif")

scif = full.to_scif(cell_mode="ssg_convention_oriented")
poscar = full.acc_primitive_magnetic_cell_poscar
kpoints = full.KPOINTS
```
