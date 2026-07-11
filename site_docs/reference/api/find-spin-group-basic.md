# `find_spin_group_basic`

Run the recommended quick analysis for identification, screening, and the main
physics-facing symmetry constraints.

## Signature

```python
find_spin_group_basic(
    cif: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
    parser_atol: float = 0.02,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
    spin_texture_basis_max_order: int | None = None,
) -> dict
```

## When To Use It

Use this function when you need:

- the OSSG index and symbol;
- the corresponding SOC-compatible MSG;
- moment geometry and magnetic-phase classification;
- symmetry permission for spin splitting and AHC;
- leading no-SOC and SOC spin textures;
- compact vector/polar/chiral constraints for screening.

Use [`find_spin_group`](find-spin-group.md) instead for operation matrices,
multiple cell settings, tensor details, magnetic-site analysis, quasi-2D
diagnostics, or generated SCIF/POSCAR/KPOINTS.

## Parameters

### Input

| Parameter | Meaning |
| --- | --- |
| `cif` | Path to mCIF, magnetic CIF, FindSpinGroup SCIF, or POSCAR-like input containing magnetic moments. The historical parameter name does not restrict the input to CIF. |

### Numerical controls

| Parameter | Default | Controls | User guidance |
| --- | ---: | --- | --- |
| `space_tol` | `0.02` | Shared spatial matching and space-group detection tolerance | Keep default first; scan a small range for noisy/nearly symmetric structures |
| `mtol` | `0.02 μB` | Moment equivalence, magnetic symmetry, and zero-net-moment classification | Most physically important tolerance; changing it can change both group and phase |
| `meigtol` | `2e-5` | Spin point-group eigenvalue decisions | Advanced diagnostic, not a normal first adjustment |
| `matrix_tol` | `0.01` | Matrix equivalence, standardization, and transform checks | Change only for an identified matrix/transform numerical issue |
| `parser_atol` | `0.02` | Parser-side consistency of expanded moments, especially SCIF same-site checks | Tune for a parser-expansion error, not to force a symmetry label |

Read [Parameters and Reliability](../../guide/reliability-and-tolerances.md) for
directional effects and a sensitivity-study recipe.

### Input-source and spin-texture controls

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `poscar_allow_incar_magmom` | `False` | Allow a sibling `INCAR` to provide `MAGMOM`. |
| `poscar_prefer_incar_magmom` | `False` | Prefer sibling `INCAR` moments when embedded POSCAR moments also exist. Effective only when INCAR reading is allowed. |
| `spin_texture_basis_max_order` | `None` | Set the spin-texture search/output ceiling and include `basis_by_order` from degree 0 through this degree. The default searches the normal classifier range and returns only the leading basis. |

## Return Value

Returns a JSON-serializable
[`BasicResult`](../result-schemas/basic-result.md) dictionary.

Read these fields first:

```python
{
    "index": str,
    "ossg_symbol_linear": str,
    "msg_bns_number": str | None,
    "msg_symbol": str | None,
    "conf": str,
    "magnetic_phase": str,
    "properties": dict,
    "spin_texture_config_no_soc": dict,
    "spin_texture_config_soc": dict,
    "magnetic_phase_details": dict,
    "tolerances": dict,
    ...
}
```

`tolerances` on this surface reports the effective core identification values
(`space_tol`, `mtol`, `meigtol`, and `matrix_tol`). Record a non-default
`parser_atol` from the call configuration separately.

For physical interpretation, read
[Interpret Your Result](../../guide/understanding-results.md). For every field,
read the [BasicResult schema](../result-schemas/basic-result.md).

## Example

```python
from findspingroup import example_path, find_spin_group_basic

result = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print("OSSG:", result["index"])
print("MSG:", result["msg_bns_number"], result["msg_symbol"])
print("order:", result["conf"], result["magnetic_phase"])
print("spin splitting:", result["properties"]["ss_wo_soc"], result["properties"]["ss_w_soc"])
print("AHC:", result["properties"]["ahc_wo_soc"], result["properties"]["ahc_w_soc"])
```

Expected core output:

```text
OSSG: 194.164.1.1.L
MSG: 63.457 Cmcm
order: Collinear AFM(Altermagnet)
spin splitting: k-dependent Yes
AHC: No No
```

`Yes` here means symmetry-allowed, not a calculated nonzero magnitude.
