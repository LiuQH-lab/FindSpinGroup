# Choose A Workflow

Choose from the scientific task, not from an internal object name.

## Decision Table

| Your task | CLI | Python | Cost and output |
| --- | --- | --- | --- |
| Identify OSSG/MSG, magnetic phase, spin splitting, AHC, and leading spin texture | `fsg FILE` | `find_spin_group_basic(FILE)` | Quick analysis; compact dictionary |
| Screen many files with stable JSON fields | `fsg FILE --json` or `fsg-batch` | loop over `find_spin_group_basic` | Quick analysis; machine-readable |
| Inspect SSG/MSG operations or cell settings | `fsg --full FILE --show operation-views` | `find_spin_group(FILE)` | Full analysis |
| Generate matched VASP POSCAR and KPOINTS | `fsg FILE --write-poscar-kpoints DIR` | attributes on `find_spin_group(FILE)` | Full analysis; writes files |
| Export SCIF in a chosen setting | `fsg FILE --write-scif OUT` | `result.to_scif(...)` | Full analysis |
| Analyze a slab as quasi-2D | `fsg --full --calculation-mode quasi2d --vacuum-axis c FILE --show quasi_2d` | `find_spin_group(..., calculation_mode="quasi2d")` | Full analysis with quasi-2D interpretation payload |
| Export operations in exactly the input cell | `fsg -w FILE` | `find_spin_group_input_ssg(FILE)` | Specialized operation export |
| Identify a group from operation matrices or generators | none | `get_spin_space_group_from_operations(...)` | Operation-only `SpinSpaceGroup`; no material artifacts |

## Quick Analysis: The Default

Use quick analysis for most identification and screening work:

```bash
fsg structure.mcif
```

```python
from findspingroup import find_spin_group_basic

result = find_spin_group_basic("structure.mcif")
```

Read these fields first:

```python
result["index"]
result["msg_bns_number"], result["msg_symbol"]
result["conf"], result["magnetic_phase"]
result["properties"]
result["spin_texture_config_no_soc"]
result["spin_texture_config_soc"]
```

Quick analysis is enough when you do not need explicit operation matrices,
multiple cell settings, generated structure files, tensor components, or
detailed route audits.

## Full Analysis: Ask For A Specific Product

```python
from findspingroup import find_spin_group

result = find_spin_group("structure.mcif")
```

Do not begin a new integration by recursively exploring `result.to_dict()`.
Choose the narrowest accessor for your task:

| Need | Use |
| --- | --- |
| Compact full-route summary | `result.to_summary_dict()` |
| Complete Python result grouped by meaning | `result.to_structured_dict()` |
| SCIF text in an explicit setting | `result.to_scif(cell_mode=...)` |
| Matched ACC-primitive VASP inputs | `result.acc_primitive_magnetic_cell_poscar` and `result.KPOINTS` |
| Raw compatibility fields | `result.to_dict()` only when maintaining an existing integration |

The structured result separates `summary`, `groups`, `cells`, `transforms`,
`properties`, and `artifacts`. This prevents a cell transform or diagnostic
field from being mistaken for a physics result. It is a Python navigation view,
not a directly JSON-serializable contract.

## Input-Cell Operation Export: Check The Warning

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("structure.mcif")
summary = payload["summary"]

print(summary["is_input_magnetic_primitive"])
print(summary["warning"])
```

If the input cell is not magnetic primitive, its operation list can represent
an incomplete subgroup of the primitive-cell symmetry. Before consuming
`payload["ssg"]["ops"]`, read:

- `is_input_magnetic_primitive`;
- `input_ssg_may_be_incomplete`;
- `warning`;
- both `input_ssg_index` and `primitive_ssg_index`.

Use this route because a downstream program requires the input setting, not as
a shortcut to the canonical OSSG identification.

## Operations Without A Structure

Use `get_spin_space_group_from_operations(...)` only when operations are the
primary input. It accepts a complete finite operation set or generators,
closes them modulo the declared or inferred spin-only subgroup, and returns an
identified `SpinSpaceGroup`.

This route can derive group-theoretic information such as index,
configuration, G0/L0, translational indices, spin point groups, ACC, standard
k-point templates, and operation-based symbols. It cannot infer atoms,
Wyckoff splitting, net moment, magnetic phase, structure-dependent tensors, or
files such as SCIF/POSCAR.

MSG interpretation additionally requires spin and real rotations in one common
physical coordinate representation. Merely labeling two independently
oriented settings as "oriented" does not establish their relative frame. A
Cartesian spin matrix paired with a fractional real-space matrix is sufficient
for OSSG identification but not by itself for a physically meaningful SOC/MSG
embedding.

## File Path Or Parsed Arrays?

The three functions above accept a structure-file path. Use the corresponding
`*_from_data` variants only when another program already owns the lattice,
positions, species, occupancies, and moments as arrays. The same tolerance,
coordinate-frame, and magnetic-moment assumptions still apply.

## Common Mistakes

- Running full analysis only to read `index` and `magnetic_phase`.
- Treating `to_dict()` as a curated stable schema.
- Mixing operations from one setting with coordinates from another.
- Treating an input-cell subgroup as the primitive-cell OSSG.
- Interpreting symmetry-allowed AHC or spin splitting as a calculated nonzero
  magnitude.
