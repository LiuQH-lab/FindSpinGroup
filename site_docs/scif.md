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
scif_primitive = result.to_scif(cell_mode="magnetic_primitive")
scif_input_direct = result.to_scif(cell_mode="input_identified")
```

`ssg_convention_oriented` is the default public OSSG convention output.
`magnetic_primitive` exports the magnetic primitive cell. `input_identified`
exports the result of identifying the supplied input cell directly; if the input
cell misses symmetry relative to the magnetic-primitive SSG transformed back to
the input setting, the SCIF includes
`_space_group_spin.fsg_input_setting_warning`.

There is no separate `input` SCIF mode. When the input cell is already the same
setting as the OSSG convention output, `ssg_convention_oriented` covers that
case without duplicating an alias.

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
