# `get_spin_space_group_from_operations`

Identify a spin space group directly from a complete operation set or a
generator subset, without constructing a witness crystal.

## Signature

```python
get_spin_space_group_from_operations(
    operations,
    *,
    spin_configuration: str | None = None,
    spin_only_direction=None,
    spin_frame: str = "cartesian",
    tol: float | Tolerances = 1e-6,
    identify_tol: float = 1e-3,
    max_operations: int = 4096,
    real_space_lattice=None,
    real_space_metric=None,
    spin_metric=None,
) -> SpinSpaceGroup
```

## Operation Format

Each operation is ordered as:

```python
[spin_rotation, real_rotation, translation]
```

It may be a `SpinSpaceGroupOperation`, a three-item sequence, or:

```python
{
    "spin_rotation": [[...], [...], [...]],
    "real_rotation": [[...], [...], [...]],
    "translation": [tx, ty, tz],
}
```

The real rotation must be integral and unimodular in the current cell basis.
Every translation component must already lie in `[0, 1)`. Values outside this
range are rejected rather than silently reduced because an unreduced
translation often signals a setting error.

There is no default real-space convention or cell setting. The operation
matrices define whichever current basis the caller supplies, such as SSG
convention, magnetic primitive, or input. The frame default is
`spin_frame="cartesian"`, which requires `real_space_lattice` or
`real_space_metric`. Use `spin_frame="oriented"` explicitly when the spin
matrices are already expressed in the current real-space setting basis.

## Spin-Only Semantics

`spin_configuration` accepts `collinear`, `coplanar`, or `noncoplanar`.

- For collinear input, `spin_only_direction` is the magnetic axis.
- For coplanar input, it is the normal to the spin plane.
- It is not defined for noncoplanar input.

If the type is omitted, the function closes the supplied finite operations and
classifies their pure spin-only subgroup from its common fixed spin subspace.
An identity-only subgroup is interpreted as noncoplanar. This is an explicit
operation-only API convention: identity alone cannot distinguish a genuinely
noncoplanar group from a collinear/coplanar input whose continuous spin-only
component was omitted.

External input does not need to contain FindSpinGroup's internal C2v
collinear representative. The external spin-only subgroup is used to identify
and validate the physical axis or plane, then the result is normalized to the
finite representation expected by existing `SpinSpaceGroup` analysis methods.
If the input already contains the complete internal spin-only representative,
its exact operation matrices are retained.

For collinear groups, `len(ssg.ops)` reports the size of this finite internal
representative. It is not the order of the physical spin space group. The
physical group contains the continuous spin-only component
\(C_{\infty v}\) and consequently has infinitely many operations.

## Spin Frame

`spin_frame` accepts:

- `cartesian` (default): spin rotations and `spin_only_direction` are expressed
  in the same Cartesian coordinate system as the supplied lattice vectors.
- `oriented`: spin rotations and `spin_only_direction` are expressed in the
  current real-space setting basis.

For Cartesian input with only `real_space_metric`, FindSpinGroup uses its
default relative frame: `m_x` is parallel to lattice vector a, `m_y` is the
positive component of b perpendicular to `m_x`, and
`m_z = m_x cross m_y`.

Supply `real_space_lattice`, whose rows are a, b, and c, when the Cartesian spin
matrices use an existing global Cartesian frame rather than that default
placement. This preserves the lattice's actual global orientation and
handedness. When both lattice and metric are provided, the metric must equal:

```python
real_space_metric = real_space_lattice @ real_space_lattice.T
```

The function converts Cartesian spin data once at the input boundary and
returns the group in the oriented setting representation used by existing
OSSG and MSG analysis. This conversion does not run inside group closure.

## Closure And Identification

The function:

1. validates matrices and strict mod-1 translations;
2. closes the supplied operations or generators as a finite affine group;
3. derives or validates the finite invariant spin metric;
4. identifies the spin configuration and axis or plane;
5. retains an already complete spin-only representation, or otherwise closes
   the nontrivial quotient and restores the canonical finite component;
6. constructs `SpinSpaceGroup` and resolves its database/convention `index`.

`max_operations` is a safety bound. Exceeding it raises an error instead of
returning a truncated group.

A generator input must generate the complete affine group under the current
cell translations. A symbol-level generator list may omit centering
translations because they are implicit in the lattice symbol; this API receives
no such symbol, so those centering or spin-translation generators must be
included explicitly.

## Available Analysis

The returned object supports operation-derived `SpinSpaceGroup` properties,
including:

- `index`, `conf`, and spin-only direction;
- G0/L0 symbols and numbers;
- `it`, `ik`, spin translations, and pure translations;
- nontrivial spin point-group and GSPG information;
- arithmetic crystal class and standard k-point templates;
- international SSG symbols and operation transforms;
- MSG operations and symbols when the coordinate-frame condition below is
  satisfied.

The route has no atoms or physical cell. It therefore cannot provide Wyckoff
or magnetic-site data, net moment, FM/FiM/AFM classification, structure-based
tensor values, SCIF/POSCAR, or other material artifacts.

## Coordinate-Frame Requirement For MSG

SOC/MSG classification compares spin and axial real-space actions. Select the
frame that describes the supplied spin matrices. For Cartesian input, provide
either the real-space metric to select the default relative frame or the full
lattice when the matrices use that lattice's existing global Cartesian frame.
FindSpinGroup then performs the comparison in one common oriented
representation.

`msg_ops` remains available when magnetic operations can be classified.
The downstream BNS/OG standardizer may return `msg_info=None` for a
nonstandard or incomplete current cell even when another setting of the same
OSSG is identifiable; this is a setting-standardization limitation, not a
change of OSSG index.

## Examples

The executable repository example
`examples/spin_space_group_from_operations.py` covers:

- coplanar Mn3Sn with an explicit spin-only mirror;
- the same Mn3Sn nSSG generators with spin-only semantics supplied separately;
- Mn3Sn in convention Cartesian and magnetic-primitive oriented settings;
- collinear MnTe with an explicit collinear direction;
- noncoplanar CrSe.

Minimal spin-translation example:

```python
import numpy as np
from findspingroup import get_spin_space_group_from_operations

spin_translation_generator = {
    "spin_rotation": (-np.eye(3)).tolist(),
    "real_rotation": np.eye(3).tolist(),
    "translation": [0.5, 0, 0],
}

ssg = get_spin_space_group_from_operations(
    [spin_translation_generator],
    spin_configuration="collinear",
    spin_only_direction=[0, 0, 1],
    spin_frame="oriented",
)

assert ssg.index == "1.1.2.1.L"
assert ssg.ik == 2
```
