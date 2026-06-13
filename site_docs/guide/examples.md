# Examples

The package includes example magnetic structures that can be accessed with
`example_path(...)`.

## List The Example Path

```python
from findspingroup import example_path

print(example_path("0.800_MnTe.mcif"))
```

Bundled example names include:

- `0.200_Mn3Sn.mcif`
- `0.800_MnTe.mcif`
- `1.0.48_MnSe2.mcif`
- `1.237_VCl2.mcif`
- `2.116_Na3Co2SbO6.mcif`
- `2.35_CrSe.mcif`
- `3.24_CaFe3Ti4O12.mcif`
- `CoNb3S6_tripleQ.mcif`
- `Fe.mcif`
- `MnO.mcif`
- `V2Te2O_input.mcif`

## Read The Main Labels

```python
from findspingroup import example_path, find_spin_group_basic

summary = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print("OSSG:", summary["index"])
print("phase:", summary["magnetic_phase"])
print("MSG:", summary["msg_bns_number"], summary["msg_symbol"])
```

Expected output:

```text
OSSG: 194.164.1.1.L
phase: AFM(Altermagnet)
MSG: 63.457 Cmcm
```

## Use The Full Result

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

print(result.index)
print(result.convention_ssg_international_linear)
print(result.magnetic_phase)

summary = result.to_summary_dict()
structured = result.to_structured_dict()
```

Use `summary` for a compact result and `structured` when you need the full
result grouped by summary, groups, cells, transforms, properties, and artifacts.

## Export SCIF

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))

scif_text = result.to_scif(cell_mode="ssg_convention_oriented")
```

Use [SCIF Export](../reference/scif.md) for all available `cell_mode` values.

## Input-Cell Operation Files

From the CLI:

```bash
fsg -w path/to/structure.mcif
```

This writes `ssg_symm.json` and `magnetic_primitive_poscar.vasp` in the current
directory. `input_poscar.vasp` is written only when a non-POSCAR input cell is
distinct from the magnetic primitive cell.

From Python:

```python
from findspingroup import find_spin_group_input_ssg

payload = find_spin_group_input_ssg("path/to/structure.mcif")

print(payload["summary"]["input_ssg_index"])
print(payload["ssg"]["ops"])
```

## Quasi-2D Diagnostics

`V2Te2O_input.mcif` is a bundled slab-style example. Quasi-2D diagnostics are
requested explicitly and do not replace the base 3D identification:

```python
from findspingroup import example_path, find_spin_group

path = example_path("V2Te2O_input.mcif")
result = find_spin_group(path, calculation_mode="quasi2d", vacuum_axis="c")

print(result.index)
print(result.quasi_2d["spin_splitting_2d"])
print(result.quasi_2d["generic_point_comparison"]["summary"])
```
