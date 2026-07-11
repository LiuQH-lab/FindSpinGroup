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
| `space_tol` | `0.02` | Shared atomic-site matching and spatial symmetry detection | Merge slightly displaced sites/operations and produce a higher apparent spatial symmetry | Split approximately equivalent sites and reduce the identified symmetry |
| `mtol` | `0.02 μB` | Magnetic-moment equivalence, magnetic-site splitting, and zero-net-moment decisions | Treat distinct/small moments as equivalent or zero | Split noisy moments and lower magnetic symmetry |
| `meigtol` | `0.00002` | Numerical eigenvalue decisions in spin point-group classification | Accept less exact eigenvalue relations | Reject relations affected by floating-point/input noise |
| `matrix_tol` | `0.01` | Point-group matrices, standardization, and transform consistency | Accept less exact matrices/transforms | Reject numerically noisy but physically intended operations |
| `parser_atol` | `0.02` | Parser-side consistency of expanded moments, especially SCIF equivalent sites | Accept larger moment discrepancies during expansion | Reject smaller parser/rounding discrepancies |

`mtol` deserves special attention: it affects both symmetry matching and the
threshold used to classify the net moment as zero. A change in `mtol` can
therefore change both the OSSG and `magnetic_phase`.

`space_tol` is passed through both Å-scale symmetry detection and internal
position-equivalence logic. Because those internal comparisons do not yet form
one uniform public unit contract, report the numerical value and describe it as
the shared spatial tolerance rather than attaching one unit to every use.

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
