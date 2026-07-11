# Concepts

This page is a glossary for concepts that appear repeatedly in FindSpinGroup
output. New users should read [Interpret Your Result](understanding-results.md)
first, then return here for terminology. Use it as the companion page for field references such as
[CLI Show Fields](../reference/cli-show-fields.md) and the result-schema pages.

The field-reference pages should stay short: a field entry should say what the
field contains, which route provides it, and which setting it uses. Longer
physical definitions belong here.

## Routes And Result Layers

FindSpinGroup has several routes. They share the same symmetry language but
return different amounts of data.

`find_spin_group_basic(...)`
Identifies the oriented spin space group (OSSG), corresponding magnetic space
group (MSG), spin point groups, magnetic phase, and compact physical flags. It
does not generate all user artifacts.

`find_spin_group(...)`
Runs the full route. In addition to the basic result, it generates cells,
operation views, SCIF text, POSCAR/KPOINTS text, little-group payloads,
Wyckoff-chain summaries, tensor outputs, and detailed diagnostics.

`find_spin_group_input_ssg(...)`
Uses an input magnetic structure as the setting for input-cell operations and
writeout workflows. It is useful when the caller wants operation data tied to a
specific input cell rather than only the public convention setting.

`calculation_mode="3d"`
The default. The input is treated as a three-dimensional magnetic crystal.

`calculation_mode="quasi2d"`
Runs the quasi-2D interpretation path. The code identifies a vacuum axis,
regularizes insufficient vacuum when necessary, builds a 2D compact k-point
view, and recomputes 2D spin-texture constraints in the corresponding in-plane
momentum variables.

## Group Objects

`SG`
The ordinary crystallographic space group of the structure after magnetic
moments are ignored. SG fields are useful for structural symmetry, polarity,
chirality, and parent real-space constraints.

`SSG`
Spin space group. An SSG operation combines a spin rotation, a real-space
rotation, and a real-space translation. In FindSpinGroup serialized operations
use:

```python
{
    "spin_rotation": [[...], [...], [...]],
    "real_rotation": [[...], [...], [...]],
    "translation": [...],
}
```

`OSSG`
Oriented spin space group. This is the main group identified by
FindSpinGroup. The output `index` is the OSSG index.

`G0`
The real-space group component used by the identified OSSG.

`L0`
The spin-lattice subgroup component used by the identified OSSG.

`MSG`
Magnetic space group corresponding to the magnetic structure under
spin-orbit-coupled magnetic symmetry. MSG fields use BNS and OG notation where
available.

`GSPG`
Generalized spin point group. It is the point-group-level spin-space symmetry
presentation. Its text output is meant for compact display and inspection, not
for reconstructing the full real-space operation set.

`SSPG`
Spin-space point group. It records the point part of the spin-space symmetry.

`nSSPG`
Nontrivial spin-space point group. This removes the trivial spin-only component
from the spin-space point-group discussion, so it highlights the nontrivial
coupling between spin and real-space operations.

## Cells And Settings

Many fields are meaningful only with their cell setting. Always check the
setting before comparing operations, coordinates, or POSCAR output.

`input`
The structure as supplied by the user.

`magnetic primitive`
The magnetic primitive cell found from the input magnetic structure.

`SSG convention`
The public convention setting used for OSSG, GSPG, operation views, and many
display fields. Depending on the group, this can be tied to a `G0std` or
`L0std` standard setting.

`ACC conventional`
The conventional cell of the spin arithmetic crystal class (ACC).

`ACC primitive`
The magnetic primitive cell aligned to the ACC convention. This is the main
setting for generated POSCAR/KPOINTS artifacts.

`oriented spin frame`
The spin frame used by OSSG-oriented spin-space operations. It is a symmetry
frame, not necessarily the same as the input Cartesian frame.

`Cartesian frame`
The real Cartesian frame associated with the current lattice setting.

`OSSG unit Cartesian`
The unit Cartesian frame used for spin-texture constraints. The first lattice
direction is mapped to `x`, the second direction lies in the `xy` plane, and
`z` is the right-handed cross product direction.

## Operation Views

`operation_views`
The preferred structured operation payload for UI and downstream consumers.
Each setting has a `default_view` and a set of named views.

`all`
The ordinary complete operation list for that setting.

`nssg`
The nontrivial SSG presentation. For collinear cases this is the preferred
display view because it avoids exposing auxiliary finite spin-only operations
that are used internally only for computation.

`generators`
A compact generator set supplied by the core. UI code should not infer
generators from the full operation list.

`pure_translations`
Operations with identity spin rotation, identity real-space rotation, and a
possibly nonzero translation.

`spin_translations`
Operations with identity real-space rotation. Pure translations are included as
the identity-spin subset.

`l0_operations`
Operations belonging to the L0 subgroup when that view is meaningful.

`msg`
The MSG-compatible operation view. In collinear cases this uses the spin part
obtained in the MSG construction, not the auxiliary spin-only presentation used
for OSSG bookkeeping.

`seitz_latex`
LaTeX Seitz symbols aligned one-to-one with `ops`. The list length should match
the operation list length.

## Magnetic Configuration And Phase

`conf`
The magnetic configuration class, such as `Collinear`, `Coplanar`, or
`Noncoplanar`.

`phase`
A compact legacy phase label.

`magnetic_phase`
The user-facing magnetic phase summary. It can include tags such as
altermagnetism or spin-orbit magnet behavior.

`magnetic_phase_family`
The article-level FM/AFM dichotomy. `AFM-class` means the spin-space point
group forces the net spin magnetization to zero; `FM-class` means it permits a
net spin magnetization.

`magnetic_atom_orbit_count_ssg`
Number of magnetic-atom orbits in the magnetic primitive cell under the
complete SSG. This is not the number of `G0` Wyckoff orbits. Within `FM-class`,
one orbit gives `FM`, whereas multiple orbits give `FiM` or `Compensated FiM`
depending on the actual net moment.

`net_moment`
The magnitude of the vector sum of magnetic moments in the analyzed magnetic
cell, reported as a scalar in μB.

`zero_net_moment_tol`
The tolerance used to decide whether the net moment is treated as zero for
magnetic phase classification.

`compensated FiM`
An `FM-class` configuration with multiple SSG magnetic-atom orbits whose net
moment is treated as zero under the configured magnetic-moment tolerance.

`FM-class (zero-moment undetermined)`
The boundary case in which the SSG permits net spin magnetization, the magnetic
atoms form one SSG orbit, but the supplied moment sum is zero. It is not labelled
as compensated ferrimagnetism without multiple inequivalent magnetic orbits.

## K Points And Spin Texture

`KPOINTS`
Generated KPOINTS text in the ACC primitive magnetic cell.

`***`
Marks a k path where spin splitting is allowed without spin-orbit coupling.

`^^^`
Marks a k path where spin splitting is allowed with spin-orbit coupling.

`spin_texture_config_database`
The database-provided spin-texture payload tied to the identified SSG label.

`spin_texture_config_no_soc`
The runtime spin-texture constraint without spin-orbit coupling. It uses the
full spin-space symmetry in the OSSG unit Cartesian frame.

`spin_texture_config_soc`
The runtime spin-texture constraint with spin-orbit coupling. It uses the
MSG-compatible symmetry in the OSSG unit Cartesian frame.

`spin_texture_type`
The leading wave type, such as `s-wave`, `p-wave`, or `d-wave`. For example,
`s-wave` means the leading allowed term is order zero in momentum.

`basis`
Symbolic basis functions for the leading allowed spin texture. Higher-order
terms can be omitted and represented with a small-order suffix in display text.

`basis_latex`
LaTeX versions of the same basis functions.

For quasi-2D, the code does not simply edit the 3D leading term. It recomputes
the spin-texture constraint in the two momentum variables that remain after the
vacuum-axis direction is removed.

## Wyckoff Chains And Magnetic Sites

`wp_chain`
A compact Wyckoff splitting chain linking SG, SSG, and MSG site labels.

`magnetic_site_summary`
A summary of magnetic-site orbits and magnetic-site degrees of freedom in the
chosen output setting.

`magnetic atom`
An atom that is part of a symmetry-related magnetic orbit. A site can count as
magnetic even if its refined moment is zero, when it belongs to an orbit split
from magnetic sites by the magnetic symmetry.

`magnetic site DOF`
The number of independent spin components allowed by the site symmetry of a
magnetic orbit.

`cell_expansion`
The volume ratio from the nonmagnetic SG primitive cell to the magnetic
primitive cell used by the magnetic-site summary.

## Vector Constraints

`vector_constraints_by_symmetry`
Symmetry-allowed vector subspaces grouped by SG, OSSG, and MSG. These fields
are useful for polar-axis analysis and time/parity-classified vector responses.

`real-space vector constraint`
A constraint on an ordinary polar or axial vector in real space.

`spin-space vector constraint`
A constraint on a vector in spin space.

`T-even` and `T-odd`
Whether the vector response is even or odd under time reversal.

`P-even` and `P-odd`
Whether the vector response is even or odd under spatial inversion.

`polar_axes_by_symmetry`
Allowed polar-axis directions by symmetry. These are written in the documented
setting of the corresponding field, not necessarily in the input cell.

## Ferroelectric Switching

FindSpinGroup treats ferroelectric-switching output as a symmetry analysis, not
as an energy-barrier calculation.

For collinear cases, switching analysis distinguishes whether a coset operation
reverses the polarization `P`, reverses the magnetic order direction `S`, or
reverses both. This gives symmetry-allowed switching candidates; it does not by
itself prove the kinetic path or energy barrier.

`magnetically induced polarization`
The nonmagnetic SG is nonpolar, but magnetic ordering lowers symmetry so that a
polar axis becomes allowed.

`magnetically controlled polarization`
Magnetic order changes which polarization states are symmetry-related or
switchable. The operation table can separate `P` reversal, `S` reversal, and
joint `P/S` reversal.

## Output Artifacts

`SCIF`
Spin-CIF-like text generated in a selected setting. It is intended for
structured symmetry export.

`POSCAR`
Generated VASP POSCAR text, usually in the ACC primitive magnetic cell.

`KPOINTS`
Generated VASP KPOINTS text, paired with the generated POSCAR setting.

`gspg_text`
Compact text for GSPG inspection, including the effective MPG, spin-space point
group symbol, setting labels, and selected operations.
