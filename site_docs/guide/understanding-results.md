# Interpret Your Result

This is the physics-facing guide to the main outputs. Complete field trees live
in [Result Schemas](../reference/result-schemas/index.md); diagnostic transforms
and internal route details should not be the first layer a user reads.

## First Separate The Three Output Surfaces

FindSpinGroup exposes related information through three intentionally different
surfaces:

| Surface | Produced by | Phase field | Use |
| --- | --- | --- | --- |
| `BasicResult` | `find_spin_group_basic(...)` | `magnetic_phase` | Screening and normal user scripts |
| `SummaryResult` | `result.to_summary_dict()` | `phase` | Compact summary after full analysis |
| `MagSymmetryResult` | `find_spin_group(...)` | `result.magnetic_phase` | Interactive full analysis |

Do not assume that raw attribute names and compact dictionary names are
identical. Choose one documented surface for an integration.

## The Eight Results Most Users Need

| Result | Physical question | Meaning | Does not tell you |
| --- | --- | --- | --- |
| `index` | Which OSSG was identified? | Database index of the oriented nonrelativistic spin-space symmetry. | Energy stability or response magnitude. |
| `ossg_symbol_linear` | How is that OSSG written symbolically? | Compact symbol in the reported convention. | Operations in the input cell. |
| `msg_bns_number`, `msg_symbol` | What symmetry is compatible with SOC? | Corresponding MSG in BNS notation. | SOC strength. |
| `conf` | What is the geometry of the moments? | Collinear, coplanar, or noncoplanar. | FM versus AFM by itself. |
| `magnetic_phase` | How is the supplied order classified? | Symmetry/net-moment rule-based label. | Ground state or transition temperature. |
| `properties.ss_*` | Is spin splitting symmetry-allowed? | Permission/character with and without SOC. | Splitting energy. |
| `properties.ahc_*` | Is AHC symmetry-allowed? | Whether symmetry forces AHC to zero. | Conductivity magnitude or sign. |
| `spin_texture_config_*` | What leading momentum dependence is allowed? | Lowest-order `k`-polynomial spin texture found in the search. | A fitted material Hamiltonian. |

## OSSG And MSG Answer Different Limits

### OSSG: nonrelativistic spin-space symmetry

An SSG operation can combine a spin rotation with a real-space operation. In
the nonrelativistic limit, spin and real space are not required to rotate
together. The oriented SSG (OSSG) fixes the orientation/setting used for the
identified public result.

Read:

- `index` for the canonical identifier;
- `ossg_symbol_linear` for a compact symbol;
- `g0_symbol`, `g0_number`, `l0_symbol`, and `l0_number` only when you need the
  group decomposition;
- operation views from full analysis when you need matrices.

`it` and `ik` are identification-index components. They are essential for exact
database identity, but they are not normally the first quantities to discuss in
a materials interpretation.

### MSG: SOC-compatible magnetic symmetry

With spin-orbit coupling, spin transformations are tied to real-space
transformations. `msg_bns_number` and `msg_symbol` identify the corresponding
magnetic space group in BNS notation.

Use the MSG or SOC-labeled outputs for SOC-constrained responses. Use OSSG/no-SOC
outputs for the nonrelativistic spin-space-symmetry limit. A difference between
the two is physically meaningful; it is not a duplicate calculation.

## Moment Geometry Is Not Magnetic Phase

`conf` reports only the geometry of the nonzero ordered moments:

- `Collinear`: all moments lie along one spin-space axis;
- `Coplanar`: moments span a plane;
- `Noncoplanar`: moments span three spin-space directions.

`magnetic_phase` combines symmetry information, the spin point group, the net
moment, and classification rules. For example, a collinear structure can be
FM-like, AFM-like, or a compensated ferrimagnet.

`net_moment` is the magnitude of the vector sum of moments in the analyzed
magnetic cell. The zero/nonzero decision uses `zero_net_moment_tol`, which is
derived from `mtol`. A phase label near that threshold is tolerance-sensitive.

The reliable boolean classifier evidence is stored under
`magnetic_phase_details`, including `is_altermagnet` and
`is_spin_orbit_magnet`. Top-level `is_alter` and `is_som` are display strings
and may be empty rather than `False`.

## Spin Splitting

The compact `properties` dictionary contains:

```python
{
    "ss_wo_soc": "k-dependent" | "Zeeman" | "No",
    "ss_w_soc": "Yes" | "No",
    ...
}
```

Interpretation:

| Value | Meaning |
| --- | --- |
| no-SOC `k-dependent` | Nonrelativistic spin splitting is symmetry-allowed and momentum-dependent. |
| no-SOC `Zeeman` | FM/FiM-like order permits a momentum-independent exchange/Zeeman-like splitting. |
| no-SOC `No` | The analyzed nonrelativistic symmetry forbids spin splitting. |
| SOC `Yes` | The SOC-compatible symmetry permits spin splitting. |
| SOC `No` | The SOC-compatible symmetry forces the relevant splitting to vanish. |

These values classify symmetry permission or character. They do not calculate
band energies, Fermi-surface locations, or the size of a splitting. SOC
`Yes` also does not mean every k point must split; little-group symmetry can
still protect degeneracies at special points or lines.

## Anomalous Hall Conductivity

`properties.ahc_wo_soc` and `properties.ahc_w_soc` are symmetry filters:

- `Yes`: an anomalous Hall response is symmetry-allowed;
- `No`: symmetry forces it to vanish in that model;
- an error/unknown value: the needed symmetry classification was unavailable.

Even when AHC is allowed, the actual value can be zero or small because of the
electronic structure, Fermi level, domains, or cancellations. FindSpinGroup
does not compute Berry curvature or conductivity.

## Spin Texture

The main fields are:

- `spin_texture_config_database`: reference classification associated with the
  identified SSG label;
- `spin_texture_config_no_soc`: runtime constraint from OSSG operations;
- `spin_texture_config_soc`: runtime constraint from MSG-compatible operations.

A spin-texture basis describes symmetry-allowed terms in a model of the form
`d(k) · sigma`. Read one record as follows:

| Field | Meaning |
| --- | --- |
| `spin_texture_type` | Leading wave value: `s-wave`, `p-wave`, `d-wave`, `f-wave`, `g-wave`, ... |
| `order` | Polynomial order in momentum of the leading allowed term. |
| `basis` | Display expression for the allowed span, including free coefficients. |
| `basis_vectors` | Independent null-space vectors when provided. |
| `nullity` | Dimension of the allowed coefficient space. |
| `spin_rank` | Dimension spanned by allowed spin directions. |
| `momentum_space_spin_configuration` | Collinear, coplanar, or noncoplanar geometry of the allowed spin texture. |
| `basis_setting` | Coordinate/spin frame in which `kx`, `ky`, `kz` and `sigma` are defined. |

The returned values `s-wave`, `p-wave`, `d-wave`, `f-wave`, `g-wave`, ...
denote polynomial orders 0, 1, 2, 3, 4, ... in this spin-texture expansion.
They are not claims about the angular-momentum character of an electronic
orbital.

The coefficients `C1`, `C2`, ... are free parameters. Symmetry fixes the
allowed functional form, not their material-specific values. A suffix such as
`o(k^4)` indicates omitted higher-order terms in the display convention.

`spin_texture_type="forbidden"` means that no allowed term was found through
the maximum order searched. The default runtime search is finite (normally
through order 6), so it is not a proof that every possible higher-order term is
forbidden.

The usual runtime basis uses the reported OSSG unit-Cartesian setting. Do not
silently interpret `kx`, `ky`, and `kz` as the original input reciprocal axes.

### Relating the unit-Cartesian basis to crystallographic axes

The regular 3D spin-texture frame is constructed from the **convention cell**
lattice vectors `a`, `b`, and `c`:

```text
e_x = a / |a|
e_y = normalize(b - e_x (e_x · b))
e_z = e_x × e_y
```

Let `A = [a b c]`, `E = [e_x e_y e_z]`, and `D = E^T A`. Then convention-cell
fractional direct components `f` and unit-Cartesian direct components `u` are
related by

```text
u = D f
```

while reciprocal components transform contragrediently:

```text
k_unit = D^(-T) k_convention
```

Spin-vector components expressed in the convention lattice basis transform as
`s_unit = D s_convention`; Cartesian spin vectors instead use `s_unit = E^T
s_cartesian`. For an input-cell vector, first use the full result's
`transforms.input_to_convention`, then apply the convention-to-unit construction
above. Keep the real-space and spin-frame conventions separate when the payload
reports different settings.

This construction makes a basis expression actionable, but the free `C_i`
coefficients still require a material model or fit.

## Vector Constraints

`vector_constraints_by_symmetry` reports allowed vector subspaces under
different symmetry descriptions:

- `sg`: structural space-group constraint;
- `ossg`: nonrelativistic spin-space-symmetry constraint;
- `msg`: SOC-compatible magnetic-space-group constraint.

A `free_dimension` of 0, 1, 2, or 3 means that the allowed vector subspace is
zero-dimensional, axial, planar, or unrestricted, respectively. `allowed_axes`
gives a basis/direction in the field's documented setting. T/P labels describe
time-reversal and inversion parity of the response class.

Always carry the setting with a direction. `[0, 0, 1]` is meaningless if the
reader does not know whether it refers to the input cell, convention cell,
Cartesian frame, or spin frame.

## Tensors, Magnetic Sites, And Switching

Full analysis adds deeper constraints:

**Tensor outputs.** Symmetry-allowed component relations and free parameters.
They do not provide numerical material coefficients.

**`magnetic_site_summary`.** Magnetic orbits and site-symmetry degrees of
freedom. A zero reported moment can still belong to a magnetic
parent-space-group orbit.

**`ferroelectric_switching`.** Symmetry relations between candidate
polarization/magnetic-order domains. It does not calculate a switching path,
barrier, or kinetics.

## Cells, Settings, And Generated Files

Full analysis can contain the input cell, magnetic primitive cell,
database-standard cell, convention cell, ACC primitive cell, and ACC
conventional cell. These are not interchangeable labels for the same arrays.

Practical rules:

1. Keep operations and coordinates in the same real-space setting.
2. Keep spin matrices/components in the same spin-frame setting.
3. Keep generated ACC-primitive POSCAR and KPOINTS together.
4. Use convention outputs for public OSSG presentation.
5. Use input-setting outputs only when a downstream code explicitly requires
   the supplied cell.
6. Use database-standard settings for database/route validation, not by
   default for DFT input.

## A Result Is Numerically Sensitive When...

- small tolerance changes alter the OSSG or MSG;
- the net moment is close to `zero_net_moment_tol`;
- nearly parallel/coplanar moments sit close to a geometry boundary;
- the structure is only approximately symmetric;
- parser expansion reports inconsistent moments on an equivalent site.

In these cases, report the sensitivity and inspect the structure. Do not choose
a tolerance only because it produces the expected label. Follow
[Parameters and Reliability](reliability-and-tolerances.md).
