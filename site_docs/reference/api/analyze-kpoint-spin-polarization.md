# Arbitrary-k Spin Polarization

Determine the spin-vector subspace allowed by symmetry at one exact numerical
k point. The calculation uses the actual SSG and MSG little groups rather than
assigning the point to the first matching high-symmetry label.

## One-shot use

```python
from findspingroup import analyze_kpoint_spin_polarization

query = analyze_kpoint_spin_polarization(
    "structure.mcif",
    [0.25, 0.5, 0.0],
    kpoint_setting="acc_primitive",
    kpoint_tol=1e-5,
)

print(query)
```

The default result is deliberately compact:

```python
{
    "kpoint": [0.25, 0.5, 0.0],
    "kpoint_setting": "acc_primitive",
    "kpoint_tol": 1e-5,
    "spin_frame": "acc_primitive_cartesian",
    "without_soc": {
        "allowed": False,
        "dimension": 0,
        "constraint": ["0", "0", "0"],
        "direction": None,
    },
    "with_soc": {
        "allowed": True,
        "dimension": 1,
        "constraint": ["-sqrt(3)/3*Sy", "Sy", "0"],
        "direction": [-0.5, 0.8660254038, 0.0],
    },
}
```

`direction` is populated only for a one-dimensional allowed subspace. It is a
canonical axis representative: symmetry does not determine its sign or
magnitude.

The structure input accepts the same mCIF, CIF, SCIF, and POSCAR-like files as
`find_spin_group(...)`. POSCAR/INCAR moment options and the central structure
tolerances can be passed to the one-shot function as keyword arguments.

## Reuse a full result

When the structure has already been analyzed, do not identify it again:

```python
from findspingroup import find_spin_group

result = find_spin_group("structure.mcif")

query = result.analyze_kpoint_spin_polarization(
    [0.25, 0.5, 0.0],
    kpoint_setting="acc_primitive",
)
```

For several points, retain one analyzer so its operation arrays and little-group
constraint cache are reused:

```python
analyzer = result.prepare_kpoint_spin_polarization_analyzer()
queries = analyzer.query_many(
    [[0, 0, 0], [0.25, 0.5, 0], [0.25, 0.125, 0]],
    kpoint_setting="acc_primitive",
)
```

## Coordinate contract

`kpoint_setting` accepts:

- `acc_primitive`: reciprocal fractional coordinates of the returned
  ACC-primitive magnetic cell and generated KPOINTS (the default);
- `input`: reciprocal fractional coordinates of the supplied magnetic cell,
  only when its reciprocal lattice is related one-to-one to the ACC-primitive
  reciprocal lattice by a unimodular setting change.

If the supplied structure is a magnetic supercell, one input-cell k point folds
multiple distinct ACC-primitive k points. Their spin constraints need not be the
same, so `input` is rejected instead of selecting an arbitrary unfolded branch.
Use the generated ACC-primitive KPOINTS coordinates, or unfold the target band
to an ACC-primitive k point before querying.

Equivalence is tested modulo integer ACC-primitive reciprocal-lattice
translations. `kpoint_tol` controls
only little-group membership; it does not replace the spin-constraint rank
tolerance or any structure-identification tolerance.

For `calculation_mode="quasi2d"`, the selected input-cell vacuum component must
be zero modulo an integer within `kpoint_tol`. An out-of-plane point raises an
error rather than being silently projected into the plane.

## Interpretation

Symmetry generally determines a subspace, not one material-specific vector:

| `dimension` | Meaning |
| ---: | --- |
| 0 | spin polarization is symmetry-forced to zero |
| 1 | one direction is allowed; sign and magnitude remain undetermined |
| 2 | a plane of directions is allowed |
| 3 | the direction is unconstrained by the little group |

The public `constraint` and `direction` fields are expressed in the spin frame
named by the top-level `spin_frame` field.

The no-SOC result uses the OSSG little group. The SOC result uses the MSG little
group and axial-vector action in Cartesian space. These are symmetry-allowed
spin-expectation constraints under the current FSG model; they do not select a
band, determine a magnitude, or resolve how an antiunitary symmetry acts inside
a degenerate band subspace.

This API answers one exact point. It must not be used to infer the behavior of
an entire path from one midpoint. A standard VASP KPOINTS file is also not a
self-contained symmetry source: it can supply coordinates only when paired
with the magnetic structure and an explicit coordinate setting.

## Diagnostics

The detailed coordinate transforms, little-group operation indices, numerical
projectors, residuals, and precomputed-row matching data are intentionally not
part of the default mapping or `print(query)` output. They remain available for
validation and debugging:

```python
print(query.audit["without_soc"]["little_group_operation_indices"])
detailed = query.to_dict(include_audit=True)
```

## Existing-result reuse

`query.audit["without_soc"]["source"]="precomputed_kspace"` (and the
corresponding SOC field)
means an existing ACC k-space row has exactly the
same little-group operation signature and its readable constraint was reused.
`query.audit[...]["source"]="computed_little_group"` means that signature was absent from the
precomputed rows, so the three-component nullspace was solved once and cached.
ACC labels are display metadata and are not used as the scientific decision.
`query.audit[...]["precomputed_constraint_match"]` distinguishes exact readable
agreement from a numerically equivalent subspace reconstructed from a rounded
serialized result.

`query.audit[...]["membership_audit"]` reports the largest included and smallest
excluded reciprocal residual. Treat `stability="near_boundary"` as a reason to
repeat the query with a physically justified tolerance range.
