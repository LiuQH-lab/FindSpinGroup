# Parameters, Tolerances, And Reliability

The safest first run uses the defaults. Tolerances are part of the physical and
numerical definition of a result; they are not generic knobs to turn until a
desired label appears.

## User-Facing Functional Parameters

These parameters change what analysis is requested rather than how numerical
equivalence is judged.

| Parameter | Default | Use when |
| --- | --- | --- |
| `calculation_mode` | `"3d"` | Set to `"quasi2d"` only for an intended slab/layer interpretation. |
| `vacuum_axis` | `"c"` | Name the input-cell axis normal to the slab plane in quasi-2D analysis. |
| `spin_texture_basis_max_order` | `None` | Request basis records through a chosen polynomial order; mainly diagnostic. |
| `poscar_allow_incar_magmom` | `False` in Python | Permit a sibling `INCAR` to provide `MAGMOM`. |
| `poscar_prefer_incar_magmom` | `False` in Python | Prefer sibling `INCAR` moments over embedded POSCAR moments. |

The CLI enables and prefers sibling `INCAR` `MAGMOM` for POSCAR-like inputs.
Python does not, unless explicitly requested, so that a file-path call remains
reproducible.

## Numerical Tolerances

| Parameter | Default | Compares or controls | Increasing it may... | Decreasing it may... |
| --- | ---: | --- | --- | --- |
| `space_tol` | `0.02 Å` | Spatial symmetry detection and lattice-aware magnetic-site matching | Accept larger positional residuals | Reject smaller deviations from an operation's site permutation |
| `mtol` | `0.02 μB` | Magnetic-moment equivalence, magnetic-site splitting, and zero-net-moment decisions | Treat distinct/small moments as equivalent or zero | Split noisy moments and lower magnetic symmetry |
| `meigtol` | `0.00002` | Numerical eigenvalue decisions in spin point-group classification | Accept less exact eigenvalue relations | Reject relations affected by floating-point/input noise |
| `matrix_tol` | `0.01` | Point-group matrices, standardization, and transform consistency | Accept less exact matrices/transforms | Reject numerically noisy but physically intended operations |
| `parser_atol` | `0.02` | Parser-side consistency of expanded moments, especially SCIF equivalent sites | Accept larger moment discrepancies during expansion | Reject smaller parser/rounding discrepancies |

`mtol` deserves special attention: it affects both symmetry matching and the
threshold used to classify the net moment as zero. A change in `mtol` can
therefore change both the OSSG and `magnetic_phase`.

Lattice-aware magnetic-site matching uses the shortest periodic Cartesian
distance, not the largest fractional-coordinate difference. For a row-vector
lattice `L`, the distance between fractional positions `x` and `y` is
`min_n ||(x-y-n) L||`, where `n` is an integer lattice vector. Componentwise
wrapping alone need not find this distance in a skew cell. An accepted operation
must admit a one-to-one match of the sites with compatible elements, occupancies
and moments.

This does not turn every internal threshold into an Å tolerance. The legacy
standalone `AtomicSite` comparison without a lattice is fractional; operation-only
input cannot infer a physical length scale. Cell-transform deduplication,
parser expansion, matrix fitting and downstream numerical rank decisions still
have separate contracts. In particular, `parser_atol` is not a replacement for
`space_tol`.

Core tolerance values must be finite. `space`, `m_eig` and `m_matrix_tol` in
`Tolerances` must be positive; `moment` and `occupancy` may be zero. A closed
comparison allows floating-point roundoff at the boundary, not an additional
relative physical tolerance. A fixed candidate's acceptance is monotone in its
tolerance, but the final identified group need not be: primitive-cell reduction,
moment clustering and candidate selection can also change.

## Keep Three Error Budgets Separate

1. **Input equivalence:** positional lengths, moment differences and occupancies
   decide which approximate input sites can be related by a symmetry operation.
2. **Numerical representation:** matrix fitting, affine transformations and
   nullspace rank deal with the accepted group representation. Their thresholds
   are not magnetic-moment errors and should not be enlarged to obtain a target
   label.
3. **Presentation:** fractional snapping and compact symbolic expressions make
   output readable. They must not be fed back as silently altered operations.

MSG identification uses the supplied operations directly. Its translations are
reduced modulo lattice integers, not rounded to nearby small-denominator
fractions. Computational mod-1 operation multiplication, inversion and setting
transport preserve resolved small translations; cleanup there
is limited to machine roundoff, not a physical equivalence tolerance.

`transform(..., frac=False)` has a different purpose: internal G0/nofrac
representations retain explicit integer translation lifts. The displayed G0
cell need not be a magnetic translation cell. Reducing those lifts modulo its
integer axes can destroy spin-translation information. Representative-list
deduplication and spin-only selection therefore retain literal translations.
A genuine L0 cell, in contrast, has spin-identity lattice translations.

For g-type symbol translation factors, a stored representative need not lie
exactly on the displayed axis: it can differ from an axial translation by a
known spin-identity lattice period. Symbol selection carries that period through
setting changes and solves for the shortest positive axial representative.
It keeps the original operation for generator reuse. The tracked period may
be a sublattice of the full primitive translation lattice; explicit centering
operations are not discarded. This bookkeeping does not change the public
operation tables or redefine spin-only membership.

Named real generators and centering operations are also matched modulo the
tracked spin-identity period, not automatically modulo the displayed G0 axes.
Integer translations outside this period can carry different spin rotations.
If a t/g-type named generator cannot be matched, its spin partner is unresolved
(`?`), not silently reported as identity. L0 generators in a k-type symbol have
identity spin by definition.

Setting transport retains the complete affine map. If an intermediate cell is
`x_mid = A x + a` and the final standard cell is `x_std = P x + p`, the second
step is `(P A^-1, p - P A^-1 a)`. Its origin shift is not generally zero.
Named generators displayed in the original current setting are transported
back using the full `(P,p)`, not the intermediate-to-standard map. Snapping
each transformed translation separately can destroy their common origin and
must not be used as a substitute for this coordinate transformation.

This is not a blanket removal of all legacy numerical policies. Symbol closure
and identify-index preprocessing retain separate canonicalization budgets;
their revision requires their own group-representation validation.
The legacy symbol closure's `1e-4` translation cleanup remains local
to generator selection and does not overwrite the supplied numerical operations.

Collinear `operation_views` use a finite `+/-I` nSSG presentation, with the
continuous spin-only group described separately. Presentation generators are
the images of full generators under `U*n = chi(U)*n`, `U_display = chi(U)*I`;
`chi(U)` is not generally `det(U)`. Filtering out spin-only surrogate matrices
instead of taking this image can lose required generators. These presentation
generators must not replace the full spin-only constraints in physical texture
or tensor calculations. When transporting a finite mod-1 generator list to a
different cell, the source cell's implicit unit translations must also be
transported: their images can be nontrivial translations of the target cell.

An arbitrary-k query similarly preserves its supplied k point modulo reciprocal
lattice integers before applying `kpoint_tol`. The k-point tolerance is expressed
in ACC-primitive reciprocal fractional coordinates; it is not a direct-space
distance or a moment tolerance.

## Does Identification Symmetrize The Crystal?

FindSpinGroup does not globally idealize the lattice or move all atoms onto an
ideal symmetric structure. Both magnetic and nonmagnetic primitive-cell
extraction request `no_idealize=True` from spglib. Subsequent cell-setting
changes transform the cell and its operations together.

For `x_new = P x_old + p`, the row-vector lattice is
`L_new = P^-T L_old`. The integer-matrix fast path only certifies that the
map is a unimodular reindexing; it does not replace the supplied `P` by its
rounded matrix. It preserves each source atom's identity and removes only
periodic copies of that same atom. Two distinct nearby sites must not be
merged by a change of coordinates. Resolved small origin shifts are retained;
the final modulo-one coordinate cleanup is limited to machine roundoff.
The requested volume ratio must yield a numerically resolved integral atom
count, otherwise the transform is rejected. That count is necessary, not
sufficient: for contractions or mixed cells, the new unit translations must
induce compatible, unambiguous site permutations of the original structure.
The permutations must be bijective and commute. Inconsistent moments or an
ambiguous site identity produce diagnostics, not a substitute cell basis.

General cell-transform position comparisons use periodic Cartesian distances
in the lattice's length unit (`eps` in `change_cell_settings`), with a separate
Cartesian moment-vector budget (`moment_eps`). `CrystalCell.transform` supplies
its `Tolerances.space` and `Tolerances.moment`. A volume determinant does not
scale either error budget. Pure expansions retain source atom identities and
identify periodic copies by lattice cosets, even when distinct input sites are
closer than the physical matching tolerance.

`CrystalCell` moments in `in_lattice` are absolute components along the three
normalized lattice directions. Their Cartesian conversion uses the actual
lattice orientation and handedness, not just the cell angles. A cell change
keeps the physical Cartesian vector fixed and re-expresses its components in
the target frame. Magnetic presence and contraction residuals use the physical
vector norm. Identification without primitive-cell reduction also converts
these components to Cartesian before fitting spin rotations. These normalized
moment components are distinct from the relative spin coordinates used in
oriented SCIF operation matrices.

SCIF reconstruction uses the declared spin frame. If its basis rows in lattice
coordinates are `A`, the spin basis in the file's canonical Cartesian frame is
`B = L.T @ A.T`. Atom moments are absolute unit-direction components, while
`uvw` operations act on relative coordinates: the absolute-component action is
`D U D^-1`, with `D` containing the lengths of the columns of `B`.
`parse_scif_file(..., return_metadata=True)` reports `spin_setting`: the default
`a,b,c` frame returns `in_lattice` components; other declared frames are converted
to Cartesian. A nonidentity legacy matrix-only frame without an explicit `abc`
declaration is rejected rather than guessing its row/column convention.

Machine-readable affine expressions retain resolved small coefficients and
translations. The default precision is 15 decimal places; a fraction or radical
is substituted only if its residual fits that precision or machine roundoff.
The writer does not independently approximate every translation with a bounded
denominator or discard terms below `1e-3`. Atomic boundary cleanup is likewise
machine-scale, followed by the configured coordinate serialization precision.
Readable GSPG affine rows use the same precise default because they may be
reused as tensor-analysis input. These rules do not change separate Seitz-label
display tolerances or globally idealize asymmetric representatives.

Numerical operation-list serialization retains the stored arrays without a
six-decimal display rounding or an implicit translation reduction. Reusable
spin-only direction strings likewise retain resolved small components in their
declared frame. A scale-free SCIF collinear direction may be written as small
integers only when the ratios agree within machine roundoff. This formatting
does not change a Cartesian direction into a lattice direction or vice versa.

CIF/SCIF expansion compares periodic positions in the physical cell metric,
using `position_atol` in cell length units (the file-facing analysis routes pass
`space_tol`). `parser_atol` bounds the norm of a physical moment-vector
difference, not each coordinate separately. Direct parsers expose it as `atol`.
Occupancy matching uses a separate absolute, dimensionless `occupancy_atol`.
An absent moment record is unspecified; an explicitly supplied zero moment is
checked like any other observation. Inconsistent specified images of one site
raise a diagnostic rather than silently keeping whichever image appeared first.
The parser does not average positions/moments or project them onto symmetry.

Site-orbit and site-stabilizer construction also uses `cell.tol.space` in
physical length units. Nearby same-element sites remain distinct: each image
uses its nearest compatible site, and the resulting action must be bijective.
Missing images, ambiguous nearest ties and overlapping orbits are errors,
not reasons to silently omit constraints. Spin-rank tolerance is independent
of this geometric matching budget.

MSG membership compares `S` with `theta * det(R) * R` in the same oriented
spin/real basis. When its metric `G` is known, the residual is the spectral norm
of `B * (S - theta * det(R) * R) * B^-1`, with `B.T * B = G`: the largest error
on a physical unit vector. This criterion is invariant under a change of basis.
The metric must be finite, symmetric and positive definite. Without geometry,
the operation-level helper retains an explicit dimensionless component budget;
it does not claim physical frame invariance. Neither mode adds an implicit
relative tolerance.

When transporting an already validated metric, the implementation transports a
physical frame `B` with `B.T @ B = G` and forms the new Gram matrix. This avoids
spurious asymmetry from cancellation in a highly sheared direct congruence
product. It does not loosen the check on a user-supplied asymmetric metric or
change the physical quadratic form.

Spin-polarization permission, subspace dimension and readable equations come
from one numerical kernel. Exact duplicate operation blocks and machine-zero
blocks do not change its weighting. The SVD uses the RMS of distinct nonzero
blocks; the proposed basis is then checked against **every** original operation
with a maximum unit-vector action residual. If that per-operation budget fails,
the result is diagnosed as unresolved instead of increasing the tolerance or
dropping a constraint. This also avoids an unnecessarily large square SVD array.
Readable coefficients retain small resolved components instead of using the
rank threshold as a formatting cutoff. Arbitrary-k query audits include the
threshold, singular values, separation status, frame conditioning, roundoff
allowance and maximum full-operation residual. These are numerical stability
diagnostics, not statistical confidence or experimental uncertainty estimates.

Site constraints use the physical Cartesian kernel and explicitly transform it
to the output spin frame. The DOF and the printed equations describe the same
subspace. Absolute SCIF moment components and relative `uvw` coordinates use
the lengths of the **declared spin basis**, which need not be the real lattice
lengths. Stable parameter pivots avoid magnifying a small leading component into
an artificially large coefficient. Vector constraints likewise test the full
allowed subspace, not only the particular vectors chosen to display a basis.

An accepted finite spin representation must contain identity, be orthogonal in
its physical frame, and close simultaneously within its representation budget.
Individual finite matrix orders
alone do not prove closure. If a fitted representation is projected to an exact
point group, its operations must still preserve the supplied magnetic sites
within their physical error budgets. This does not project the crystal or its
moments. Candidate matrix algebra uses its own dimensionless tolerance, not
the positional tolerance. Rounded lookup keys only select candidates; they
are not certificates of operation equality or physical preservation.

Collinear MSG promotion tests the physical action on the common spin axis,
including its norm. A cosine-only test can hide transverse errors. Domain
comparisons require compatible bijective site permutations and use physical
signed magnetic moments. A domain's SOC axis is calculated from its complete
MSG in the actual child basis, not copied from another standard setting.
Symmetry permits a domain relation; it does not determine an energy barrier or
establish experimentally switchable ferroelectricity.

### Spin-Texture Constraints And Recovery

Spin-texture polynomials are solved in OSSG unit Cartesian coordinates, or in
the explicitly declared quasi-2D variables. Numerical polynomial tolerances
measure monomial-coefficient action residuals, not magnetic moments in μB.
Small supplied coefficients are not deleted before solving. Generators may be
used to obtain a kernel, but the raw kernel and its readable expression are
checked against all supplied operations. An expression that loses a resolved
term is reformatted at higher precision; this does not change the accepted rank.
`constraint_validation` records the norm, threshold, dimensions and largest
operation residual. Requested higher orders are validated too. A `forbidden`
search result is bounded by the recorded maximum searched order.

If a database comparison invokes bounded recovery, `calibration` retains the
strict primary result, strict full-operation result, reference, attempts and
selected result. Matching a reference type is not sufficient: the selected
basis must satisfy the full constraints at the explicit recovery budget. The
database is not a license to relax tolerances indefinitely. Quasi-2D recovery
records the same evidence and leaves the reference absent when none exists.
ASCII and LaTeX describe the same accepted coefficients; converting to LaTeX
does not introduce an independent low-precision approximation.

### Accepted-Model Diagnostics

`magnetic_phase_details.accepted_group_audit` reuses the identifier's physical
residual evaluation. It records the accepted spin-representation adjustment,
position/moment/occupancy budgets and units, and the identifier cell and frame.
Its operation/site indices refer to internal identifier lists, not to the
reordered public Wyckoff table. The accompanying source-site coordinates and
lattice specify that context. The field is absent or `None` when no identifier
context was supplied.

`magnetic_phase_details.net_moment_decision` records the strict comparison
`abs(net_moment) < zero_net_moment_tol`, its signed margin, and the ratio to the
threshold. A positive margin is inside the zero-moment criterion. Equality is
not inside it; a zero threshold has no finite ratio. This is diagnostic evidence,
not a new uncertainty band or a changed FM/FiM definition. Near-threshold inputs
can remain sensitive to reconstruction and should be examined with a one-parameter
scan rather than silently snapped to the preferred classification.

Several separate operations should not be confused with crystal idealization:

- Magnetic primitive reduction groups moment vectors within `mtol` and can
  reuse a representative vector for an equivalent site type.
- Numerical spin-group projection adjusts an operation representation and
  verifies its action on the magnetic sites; it is not a lattice refinement.
- SCIF stores asymmetric representatives and operations. Expanding them on
  readback reconstructs symmetry-related sites, so small deviations in a noisy
  input need not be preserved atom by atom.
- Quasi-2D preprocessing may extend the selected vacuum direction, as described
  below. This is an explicit geometry preprocessing step.

A change of origin or a valid change of basis must preserve the physical
symmetry conclusions. Affine translations at an arbitrary origin need not be
simple rational fractions; they must not be rounded merely to obtain a more
familiar-looking group operation.

## When Should I Change A Tolerance?

Change one only when you can name the numerical problem it addresses.

Reasonable examples:

- a relaxed structure contains known small positional noise around an exact
  parent symmetry;
- reported magnetic moments differ only by known refinement/rounding noise;
- a generated SCIF round trip reports a small same-site moment inconsistency;
- a classification lies close to a documented point-group numerical boundary.

Poor reasons:

- “the default did not produce the group I expected”;
- “a larger tolerance gives a more symmetric answer”;
- changing several tolerances together without identifying which comparison
  failed.

## A Minimal Sensitivity Check

For noisy or nearly symmetric input, vary one physically relevant tolerance
around the default while keeping all others fixed.

```python
from findspingroup import find_spin_group_basic

for space_tol in (0.01, 0.02, 0.03):
    result = find_spin_group_basic(
        "structure.mcif",
        space_tol=space_tol,
        mtol=0.02,
    )
    print(
        space_tol,
        result["index"],
        result["msg_bns_number"],
        result["magnetic_phase"],
    )
```

For moment sensitivity:

```python
for mtol in (0.01, 0.02, 0.03):
    result = find_spin_group_basic("structure.mcif", mtol=mtol)
    print(
        mtol,
        result["index"],
        result["net_moment"],
        result["zero_net_moment_tol"],
        result["magnetic_phase"],
    )
```

Interpretation:

- stable labels across a reasonable interval support a robust classification;
- a change at a clearly identifiable structural/moment scale can be physically
  meaningful;
- irregular changes or route failures call for input and diagnostic inspection;
- a result should not be selected solely because it matches prior expectation.

## Quasi-2D Parameters

Quasi-2D analysis is an interpretation workflow, not merely a shorter k-path.
Specify the intended normal axis:

```bash
fsg --full structure.mcif \
  --calculation-mode quasi2d \
  --vacuum-axis c \
  --show quasi_2d
```

The workflow can regularize/extend insufficient vacuum along the selected
input axis before the quasi-2D identification path. Read the returned
diagnostics rather than assuming the quasi-2D cell is byte-for-byte identical
to the original 3D input.

## Spin-Texture Search Order

By default, the public output emphasizes the leading allowed term. Set
`spin_texture_basis_max_order=N` to request `basis_by_order` through degree
`N`:

```bash
fsg structure.mcif \
  --spin-texture-basis-max-order 4 \
  --show spin_texture_config_no_soc.basis_by_order
```

This can increase runtime. If a configuration is reported as `forbidden`, the
claim is bounded by the maximum order actually searched (normally degree 6 in
the default runtime classifier), not every possible order.

## What To Record In A Paper Or Dataset

Record at least:

- FindSpinGroup version;
- input file or a permanent input identifier;
- `space_tol`, `mtol`, `meigtol`, and `matrix_tol`;
- `parser_atol` when non-default or when parser expansion matters;
- calculation mode and vacuum axis for quasi-2D work;
- whether POSCAR moments came from embedded data or a sibling `INCAR`;
- the cell and spin-frame setting of exported operations or basis functions;
- any observed tolerance sensitivity.

The quick-analysis dictionary reports the effective core tolerances under
`tolerances`. Record `parser_atol` from the call configuration separately when
using a surface that does not serialize it.
