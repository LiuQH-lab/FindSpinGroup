# Usage

## Full Analysis: `find_spin_group`

```python
from findspingroup import find_spin_group

result = find_spin_group("path/to/structure.mcif")
```

Main signature:

```python
def find_spin_group(
    cif: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    parser_atol: float = 0.02,
    calculation_mode: str | None = "3d",
    vacuum_axis: str | None = "c",
) -> MagSymmetryResult
```

`find_spin_group(...)` returns a `MagSymmetryResult` object. The `index` field is
the final identified SSG index from the identify-index route, not only the raw
`G0/L0/it/ik` component tuple.

Commonly used `MagSymmetryResult` attributes include:

| Attribute | Meaning |
| --- | --- |
| `index` | Final identified spin-space-group index. |
| `G0_symbol`, `G0_num` | Real-space group symbol and number used by the identified SSG. |
| `L0_symbol`, `L0_num` | Spin-lattice group symbol and number. |
| `it`, `ik` | Translation and k-index components used in the SSG identification. |
| `conf` | Magnetic configuration class, such as collinear, coplanar, or noncoplanar. |
| `magnetic_phase` | Magnetic-phase classification, including altermagnet / spin-orbit-magnet tags when detected. |
| `acc` | Arithmetic crystal class. |
| `selected_standard_setting` | Database standard setting selected by the identify-index route, either `G0std` or `L0std`. |
| `convention_ssg_setting` | Public convention setting used for the main OSSG operation output. |
| `msg_num`, `msg_symbol` | Corresponding MSG number and BNS symbol. |
| `msg_bns_number`, `msg_og_number` | MSG BNS and OG numbers. |
| `convention_ssg_international_linear` | OSSG-facing international-style symbol in the public convention setting. |
| `convention_ssg_ops` | Public convention SSG operations. |
| `acc_primitive_magnetic_cell_detail` | ACC primitive magnetic cell used by POSCAR and ACC-setting outputs. |
| `acc_primitive_ssg_ops` | SSG operations transformed with the ACC primitive cell. |
| `gspg_text` | Compact text payload for the public GSPG symbol and operations. |
| `KPOINTS` | Generated KPOINTS text with spin-polarization labels. |
| `spin_polarizations` | Symmetry-allowed spin-polarization components at sampled k points. |
| `msg_spin_polarizations` | MSG-derived spin-polarization constraints. |
| `quasi_2d` | Additive quasi-2D diagnostics when `calculation_mode="quasi2d"`; otherwise `None` in the public 3D contract. |
| `scif` | Generated `.scif` text. |
| `scif_outputs` | Generated `.scif` text keyed by cell mode. |
| `scif_cell_modes` | Available `.scif` output modes. |
| `tensor_outputs` | Symmetry-constrained tensor-analysis outputs. |
| `tolerances` | Effective tolerance values used by the route. |

Example:

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

print("index:", result.index)
print("OSSG symbol:", result.convention_ssg_international_linear)
print("phase:", result.magnetic_phase)
print("KPOINTS setting:", result.KPOINTS_setting)
```

For exploratory scripting, `result.to_dict()` exposes the current attribute
dictionary.

SCIF output modes:

| Mode | Meaning |
| --- | --- |
| `ssg_convention_oriented` | Default public OSSG convention output. |
| `ssg_convention_cartesian` | Public OSSG convention output with Cartesian spin components. |
| `database_standard_oriented` | Selected raw database standard setting, with spin components tied to the emitted lattice basis. |
| `database_standard_cartesian` | Selected raw database standard setting, with Cartesian spin components. |
| `magnetic_primitive_oriented` | ACC magnetic primitive-cell output, with spin components tied to the emitted lattice basis. |
| `magnetic_primitive_cartesian` | ACC magnetic primitive-cell output, with Cartesian spin components. |
| `input_oriented` | Direct input-cell identified output, with spin components tied to the input lattice basis. |
| `input_cartesian` | Direct input-cell identified output, with Cartesian spin components. |

Legacy aliases map to explicit modes: `database_standard` maps to
`database_standard_oriented`, `magnetic_primitive` maps to
`magnetic_primitive_oriented`, and `input_identified` maps to `input_oriented`.
There is no separate `input` SCIF mode.

`database_standard_*` emits operations in the selected SSG database standard
setting before final-convention canonicalization: `G0std` for non-type-k
results and `L0std` for type-k results. Generated SCIF files include repo-local
`_space_group_spin.fsg_scif_real_space_setting` and
`_space_group_spin.fsg_scif_spin_frame_setting` tags so the emitted real-space
basis and spin-frame convention are explicit.

## Quasi-2D Diagnostics

The base public route is three-dimensional. Quasi-2D analysis is an additive
diagnostic layer for slab-like inputs:

```python
result = find_spin_group(
    "path/to/slab.mcif",
    calculation_mode="quasi2d",
    vacuum_axis="c",
)
```

In default 3D mode, `result.quasi_2d` is `None` and the ordinary 3D `KPOINTS`
table is unchanged. In quasi-2D mode, the base `index`, SSG, MSG, ACC, and
ordinary `KPOINTS` stay on the same 3D route, while `result.quasi_2d` adds:

| Field | Meaning |
| --- | --- |
| `status`, `source`, `dimension` | How the 2D interpretation was supplied or inferred. |
| `vacuum_axis_input` | Input-cell axis normal to the intended slab plane. |
| `kpoints` | Compact KPOINTS-style rows filtered to 2D-relevant in-plane entries. |
| `kpoint_projection_summary` | Counts and labels for in-plane, out-of-plane, mixed, and unknown rows. |
| `diagnostic_points` | Generated in-plane diagnostic rows such as `GP:(u,v,0)`. |
| `generic_point_comparison` | Comparison between the ordinary 3D `GP` and the in-plane quasi-2D `GP`. |
| `spin_splitting_2d`, `interpretation` | Quasi-2D spin-splitting conclusion and controlled interpretation string. |

`vacuum_axis` is interpreted only for quasi-2D diagnostics.

## ACC And GSPG Output

The public ACC primitive cell is derived through the same paired transform route
used for the SSG: final convention -> selected database standard ->
identify-index `P` -> ACC primitive. Cell and SSG transforms are kept paired.
Useful attributes include `acc_primitive_magnetic_cell_detail`,
`acc_primitive_magnetic_cell_poscar`, `acc_primitive_ssg_ops`,
`acc_primitive_msg_ops`, and `T_convention_to_acc_primitive`.

`gspg_text` is a compact text payload built from the same public OSSG source as
the existing public GSPG fields. It includes the symbol, effective MPG, spin-only
part, real-space setting, spin-frame setting, and formatted operation strings.

## Lightweight Summary: `find_spin_group_basic`

```python
from findspingroup import find_spin_group_basic

summary = find_spin_group_basic("path/to/structure.mcif")
```

This route returns a JSON-serializable `dict` and avoids expensive downstream
outputs that are not needed for quick identification.

Returned dictionary keys:

| Key | Meaning |
| --- | --- |
| `index` | Final identified SSG index. |
| `g0_symbol`, `g0_number` | Real-space group symbol and number. |
| `l0_symbol`, `l0_number` | Spin-lattice group symbol and number. |
| `it`, `ik` | Translation and k-index components. |
| `nsspg`, `sspg` | Nontrivial and full spin-part point-group symbols. |
| `acc_symbol` | Arithmetic crystal class symbol. |
| `space_group_symbol`, `space_group_number` | Input real-space group information. |
| `msg_symbol`, `msg_bns_number`, `msg_og_number` | Corresponding MSG identifiers. |
| `empg` | Effective magnetic point-group symbol. |
| `conf` | Magnetic configuration class. |
| `magnetic_phase` | Magnetic-phase classification. |
| `is_alter`, `is_som` | Altermagnet and spin-orbit-magnet flags. |
| `sg_is_polar`, `sg_is_chiral` | Polar/chiral flags for the input SG. |
| `ssg_is_polar`, `ssg_is_chiral` | Polar/chiral flags for the SSG real-space group. |
| `msg_is_polar`, `msg_is_chiral` | Polar/chiral flags for the MSG. |

Example:

```python
summary = find_spin_group_basic("path/to/structure.mcif")
print(summary["index"])
print(summary["magnetic_phase"])
```

## Input-Cell Operations: `find_spin_group_input_ssg`

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("path/to/structure.mcif")
```

This route returns SSG and MSG operations in the input cell setting. If the input
cell is not already magnetic primitive, the input-cell operations may be fewer
than the full magnetic-primitive symmetry operations. The payload therefore also
contains primitive-side identifiers and a warning.

Top-level payload keys:

| Key | Meaning |
| --- | --- |
| `summary` | Input-cell and primitive-side identifiers. |
| `ssg` | Input-cell SSG operations and setting metadata. |
| `msg` | Input-cell oriented MSG operations and setting metadata. |
| `primitive_relation` | Transform from input cell to input magnetic primitive cell. |
| `input_poscar` | POSCAR text for non-POSCAR inputs. |
| `magnetic_primitive_poscar` | POSCAR text when the input cell is not magnetic primitive. |

Important `summary` keys:

| Key | Meaning |
| --- | --- |
| `input_ssg_index` | Identified SSG index for the input cell setting. |
| `primitive_ssg_index` | Identified SSG index for the input magnetic primitive cell. |
| `input_conf` | Input-cell magnetic configuration class. |
| `input_spin_only_direction` | Spin-only direction when applicable. |
| `input_magnetic_phase` | Magnetic-phase classification for the input-cell result. |
| `input_ssg_database_symbol` | Database SSG symbol for the input-cell result. |
| `input_msg_num`, `input_msg_bns_number`, `input_msg_symbol` | Input-cell MSG identifiers. |
| `primitive_msg_num`, `primitive_msg_bns_number` | Primitive-side MSG identifiers. |
| `is_input_magnetic_primitive` | Whether the supplied input cell is already magnetic primitive. |
| `input_ssg_may_be_incomplete` | Whether input-cell operations may miss primitive-cell symmetry. |
| `warning` | Explanation when the input cell is not magnetic primitive. |

Example:

```python
payload = find_spin_group_input_ssg("path/to/structure.mcif")

print(payload["summary"]["input_ssg_index"])
print(payload["summary"]["primitive_ssg_index"])
print(payload["ssg"]["ops"])
print(payload["msg"]["ops"])
```

## Supported Input Formats

`findspingroup` supports:

- standard `.cif`
- magnetic `.mcif`
- repo-generated `.scif`
- POSCAR-like files with embedded magnetic moments

Input notes:

- Magnetic inputs must contain explicit magnetic moments.
- POSCAR inputs must include an embedded `MAGMOM` payload, for example a trailing
  `# MAGMOM=...` line.
- The input-SSG POSCAR route does not read `INCAR`.
- POSCAR moments are treated as Cartesian.
- CIF, mCIF, and SCIF moments are converted into the route's Cartesian input-cell
  frame before operation export.

## Tolerances

The main APIs accept the same basic tolerance controls:

```python
find_spin_group(
    "path/to/structure.mcif",
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
)
```

Tolerance roles:

| Parameter | Main role |
| --- | --- |
| `space_tol` | Real-space position and lattice matching. |
| `mtol` | Magnetic-moment matching and zero-net-moment / compensated-FiM classification threshold. |
| `meigtol` | Point-group eigenvalue decisions. |
| `matrix_tol` | Point-group standardization matrix checks. |
| `parser_atol` | CIF/SCIF expansion and parser consistency tolerance. |

Use tighter tolerances only when the input structure is numerically clean enough
to support them.
