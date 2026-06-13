# `find_spin_group_basic`

Compact identification route.

Use this function when you need the identified OSSG, corresponding MSG,
magnetic phase, and compact property flags.

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

## Parameters

`cif`
Path to the input structure file. Supported inputs include mCIF, CIF with
magnetic-moment data, repo-generated SCIF, and POSCAR-like files.

`space_tol`
Tolerance for real-space position and lattice matching.

`mtol`
Tolerance for magnetic-moment matching and zero-net-moment classification.

`meigtol`
Tolerance for point-group eigenvalue decisions.

`matrix_tol`
Tolerance for point-group standardization and transform matrix checks.

`parser_atol`
Tolerance used while expanding and validating CIF or SCIF input.

`poscar_allow_incar_magmom`
Allow a POSCAR-like Python call to read magnetic moments from a sibling `INCAR`.
The default is `False`.

`poscar_prefer_incar_magmom`
Prefer the sibling `INCAR` `MAGMOM` over embedded POSCAR magnetic moments when
both are available. This only has an effect when
`poscar_allow_incar_magmom=True`.

`spin_texture_basis_max_order`
When set, include `basis_by_order` entries for computed spin-texture
configuration fields from order 0 through this order. The default `None` emits
only the leading allowed basis.

## Returns

Returns a [BasicResult](../result-schemas/basic-result.md) dictionary.

```python
{
    "index": str,
    "magnetic_phase": str,
    "msg_bns_number": str | None,
    "msg_symbol": str | None,
    "conf": str,
    "properties": dict,
    "tolerances": dict,
    ...
}
```

## Returned Fields

### Main result

`index`
Final identified oriented spin space group index.

`magnetic_phase`
High-level magnetic phase classification.

`msg_bns_number`, `msg_symbol`
Corresponding magnetic space group identifiers.

`conf`
Magnetic configuration class.

### SSG components

`g0_symbol`, `g0_number`
Real-space group component of the identified SSG.

`l0_symbol`, `l0_number`
Spin-lattice group component of the identified SSG.

`it`, `ik`
Translation-index and k-index components.

`nsspg`, `sspg`
Nontrivial and full spin-part point-group symbols.

`acc_symbol`
Spin arithmetic crystal class symbol.

### Physical properties

`properties`
Compact property summary: spin splitting with and without SOC, AHC constraints,
and altermagnet / spin-orbit-magnet display tags.

`spin_texture_config_database`, `spin_texture_config_no_soc`, `spin_texture_config_soc`
Spin-texture configuration summaries. The database field is the runtime-index
reference; the no-SOC and SOC fields are computed from OSSG and MSG symmetry.

`vector_constraints_by_symmetry`
Vector constraints grouped by `sg`, `ossg`, and `msg`.

### Diagnostics

`identify_index_details`
Identify-index route details.

`acc_primitive_resolution_audit`
ACC primitive resolution details.

`tolerances`
Effective tolerance values.

## Example

```python
from findspingroup import example_path, find_spin_group_basic

summary = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print(summary["index"])
print(summary["magnetic_phase"])
print(summary["msg_bns_number"], summary["msg_symbol"])
```

Expected output:

```text
194.164.1.1.L
AFM(Altermagnet)
63.457 Cmcm
```

## Notes

Use `find_spin_group_basic(...)` for compact identification. Use
`find_spin_group(...)` when you need generated SCIF, POSCAR, KPOINTS, operation
payloads, quasi-2D diagnostics, tensor outputs, or detailed route audits.
