# SCIF Export

`find_spin_group(...)` can generate SCIF text from the full result.

## `to_scif(...)`

```python
result.to_scif(cell_mode="ssg_convention_oriented") -> str
```

### Parameters

`cell_mode`
Selects the cell and spin-frame convention used by the generated SCIF text.

### Returns

Returns SCIF text as a string.

## Default Export

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

scif_text = result.scif
assert scif_text == result.to_scif(cell_mode="ssg_convention_oriented")
```

## Cell Modes

`ssg_convention_oriented`
Default public OSSG convention output.

`ssg_convention_cartesian`
Public OSSG convention output with Cartesian spin components.

`database_standard_oriented`
Selected raw database-standard setting with spin components tied to the emitted
lattice basis.

`database_standard_cartesian`
Selected raw database-standard setting with Cartesian spin components.

`magnetic_primitive_oriented`
ACC magnetic primitive-cell output with spin components tied to the emitted
lattice basis.

`magnetic_primitive_cartesian`
ACC magnetic primitive-cell output with Cartesian spin components.

`input_oriented`
Direct input-cell identified output with spin components tied to the input
lattice basis.

`input_cartesian`
Direct input-cell identified output with Cartesian spin components.

## Legacy Aliases

`database_standard`
Alias for `database_standard_oriented`.

`magnetic_primitive`
Alias for `magnetic_primitive_oriented`.

`input_identified`
Alias for `input_oriented`.

There is no separate `input` SCIF mode.

## Database-Standard Output

`database_standard_*` writes operations in the selected SSG database-standard
setting before final-convention canonicalization.

Non-type-k results use the raw `G0std` setting.

Type-k results use the raw `L0std` setting.

Generated SCIF text records the emitted real-space basis and spin-frame
convention with repo-local `_space_group_spin.fsg_*` tags.

## Magnetic Sites With Zero Moment

The `_atom_site_spin_moment` loop includes every site orbit that belongs to a
magnetic crystallographic orbit of the parent nonmagnetic space group. The
selection starts from sites with nonzero moments and closes their parent-space-
group crystallographic orbits. Consequently, a spin-space-group orbit whose
reported moment is zero is still written when it split from the same parent
space-group orbit as a nonzero-moment site.

For such a row, `axis_u`, `axis_v`, `axis_w`, and `magnitude` describe the
reported zero moment. The `symmform_uvw` and `symmform_rel_uvw` fields retain
the site-symmetry constraint: `0,0,0` means the zero moment is symmetry-forced,
while a nonzero symbolic form with zero magnitude means symmetry permits a
moment but the reported model sets it to zero.

Zero-moment sites from unrelated parent-space-group orbits are not inferred to
be magnetic. An input format must preserve an explicit magnetic-site marker to
distinguish such a site from an ordinary nonmagnetic site.
