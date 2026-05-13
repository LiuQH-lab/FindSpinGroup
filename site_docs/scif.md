# SCIF

`findspingroup` can export a repo-generated spinCIF-style `.scif` snapshot from
the `MagSymmetryResult`, and can also parse that generated `.scif` back through
the same public input path.

## Export

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

scif_text = result.scif
assert scif_text == result.to_scif(cell_mode="ssg_convention_oriented")
```

Available export modes:

```python
scif_convention = result.to_scif(cell_mode="ssg_convention_oriented")
scif_standard = result.to_scif(cell_mode="database_standard_oriented")
scif_primitive = result.to_scif(cell_mode="magnetic_primitive_oriented")
scif_input_direct = result.to_scif(cell_mode="input_oriented")
```

| Mode | Meaning |
| --- | --- |
| `ssg_convention_oriented` | Default public OSSG convention output. |
| `ssg_convention_cartesian` | Public OSSG convention output with Cartesian spin components. |
| `database_standard_oriented` | Raw selected SSG database standard setting, with spin components tied to the emitted lattice basis. |
| `database_standard_cartesian` | Raw selected SSG database standard setting, with Cartesian spin components. |
| `magnetic_primitive_oriented` | ACC magnetic primitive-cell output, with spin components tied to the emitted lattice basis. |
| `magnetic_primitive_cartesian` | ACC magnetic primitive-cell output, with Cartesian spin components. |
| `input_oriented` | Direct input-cell identified output, with spin components tied to the input lattice basis. |
| `input_cartesian` | Direct input-cell identified output, with Cartesian spin components. |

Legacy aliases are kept for compatibility: `database_standard`,
`magnetic_primitive`, and `input_identified` map to the corresponding oriented
modes. There is no separate `input` SCIF mode.

## Database-standard output

`database_standard_*` writes the symmetry operations in the selected database
standard setting used by identify-index:

- non-type-k results use the raw `G0std` setting
- type-k results use the raw `L0std` setting
- the setting is taken before any final-convention canonicalization

Generated SCIF text records the choice with repo-local tags:

```text
_space_group_spin.fsg_scif_real_space_setting  "G0std"
_space_group_spin.fsg_scif_spin_frame_setting  "oriented"
```

For `*_oriented` modes, spin components are tied to the emitted lattice basis.
For `*_cartesian` modes, spin components are Cartesian. The transform written in
`_space_group_spin.transform_spinframe_P_abc` records the spin-frame basis used
by the file.

## Current generator rules

- repo-local FINDSPINGROUP metadata is emitted under CIF-legal
  `_space_group_spin.fsg_*` tags
- symmetry-operation and transform coefficients keep full precision by default
- values near simple fractions or common square-root forms may be written as
  symbolic expressions such as `1/3`, `2/3`, or `sqrt(6)/3`
- `_space_group_spin.number_Chen_Liu` is the SSG number for the exported SCIF
  mode; for `input_identified`, it is the directly identified input-cell number
- repo-generated `.scif` files can be parsed back with `find_spin_group(...)`
  and are regression-tested to preserve the identified `index`
- in quasi-2D mode, generated SCIF may include additive repo-local tags such as
  `_space_group_spin.fsg_calculation_mode`,
  `_space_group_spin.fsg_vacuum_axis_input`, and
  `_space_group_spin.fsg_spin_splitting_2d_interpretation`; these tags do not
  change the SSG index encoded by the file
