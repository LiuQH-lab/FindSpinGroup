# SCIF

`findspingroup` can export a repo-generated spinCIF-style `.scif` snapshot from
the `MagSymmetryResult`, and can also parse that generated `.scif` back through
the same public input path.

## Export

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

scif_text = result.scif
assert scif_text == result.to_scif(profile="legacy", cell_mode="g0std_oriented")
```

Additional export variants:

```python
scif_working = result.to_scif(profile="spincif_working", cell_mode="g0std_oriented")
scif_primitive = result.to_scif(profile="legacy", cell_mode="magnetic_primitive")
```

## Current generator rules

- repo-local FINDSPINGROUP metadata is emitted under CIF-legal
  `_space_group_spin.fsg_*` tags
- symmetry-operation and transform coefficients keep full precision by default
- values near simple fractions or common square-root forms may be written as
  symbolic expressions such as `1/3`, `2/3`, or `sqrt(6)/3`
- repo-generated `.scif` files can be parsed back with `find_spin_group(...)`
  and are regression-tested to preserve the identified `index`
