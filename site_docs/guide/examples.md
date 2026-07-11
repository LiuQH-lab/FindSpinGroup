# Examples By Research Question

The bundled structures are teaching cases. Each example follows the same
pattern: question, command, key result, interpretation, and limitation.

Use `example_path(...)` in Python so examples work after package installation:

```python
from findspingroup import example_path

path = example_path("0.800_MnTe.mcif")
```

## Is MnTe Classified As An Altermagnet?

### Question

What are the OSSG/MSG of the supplied collinear MnTe order, and does its
nonrelativistic symmetry permit momentum-dependent spin splitting?

### Python

```python
from findspingroup import example_path, find_spin_group_basic

result = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print(result["index"])
print(result["conf"], result["magnetic_phase"])
print(result["msg_bns_number"], result["msg_symbol"])
print(result["properties"])
```

### Key result

```text
194.164.1.1.L
Collinear AFM(Altermagnet)
63.457 Cmcm
ss_wo_soc = k-dependent
```

### Interpretation

The supplied order is collinear and AFM-like under the classifier, with
symmetry-allowed momentum-dependent splitting without SOC. That combination
produces the `(Altermagnet)` tag.

### Limitation

This does not calculate the band splitting, Néel temperature, domain state, or
energetic stability of the supplied order.

## What Does “AHC Allowed” Mean For Ferromagnetic Fe?

### Question

Does the SOC-compatible symmetry allow anomalous Hall conductivity?

### Python

```python
from findspingroup import example_path, find_spin_group_basic

result = find_spin_group_basic(example_path("Fe.mcif"))
print(result["conf"], result["magnetic_phase"])
print(result["properties"]["ahc_w_soc"])
print(result["properties"]["ss_wo_soc"])
```

### Key result

```text
Collinear FM/FiM
AHC with SOC: Yes
spin splitting without SOC: Zeeman
```

### Interpretation

The symmetry does not force the SOC anomalous Hall response to vanish. The
FM-like order also permits a momentum-independent exchange/Zeeman-like
splitting in the no-SOC classification.

### Limitation

`Yes` is not a conductivity calculation. The magnitude and sign require
electronic-structure/Berry-curvature information.

## Coplanar Mn3Sn: Moment Geometry Versus Phase

### Question

Can a nearly compensated noncollinear antiferromagnet be distinguished from a
collinear altermagnet, and what changes with SOC?

### Python

```python
from findspingroup import example_path, find_spin_group_basic

result = find_spin_group_basic(example_path("0.200_Mn3Sn.mcif"))

print(result["conf"])
print(result["magnetic_phase"])
print(result["net_moment"])
print(result["properties"])
```

### Key result

```text
Coplanar
AFM (SOM)
net moment approximately 0 μB
AHC with SOC: Yes
```

### Interpretation

`Coplanar` describes the rank/geometry of the moment directions; `AFM (SOM)` is
the separate symmetry classifier result. SOC-compatible symmetry permits AHC
even though the analyzed net moment is numerically zero.

### Limitation

The zero-net-moment decision depends on `mtol`, and symmetry permission does not
specify which experimental domain is populated.

## Quasi-2D Spin Splitting In V2Te2O

### Question

What spin-splitting constraint remains when the supplied structure is
interpreted as a layer normal to input axis `c`?

### Python

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(
    example_path("V2Te2O_input.mcif"),
    calculation_mode="quasi2d",
    vacuum_axis="c",
)

print(result.index)
print(result.quasi_2d["status"])
print(result.quasi_2d["interpretation"])
print(result.quasi_2d["spin_splitting_2d"])
```

### Key result

```text
123.47.1.1.L
interpretation = in_plane_k_dependent
spin_splitting_2d = spin splitting
```

### Interpretation

The quasi-2D path recomputes the allowed in-plane behavior in the two remaining
momentum variables and reports momentum-dependent in-plane splitting.

### Limitation

The workflow can regularize/extend insufficient vacuum along the selected axis.
Inspect `geometry`, `vacuum_axis_input`, and reciprocal-transform diagnostics;
do not treat quasi-2D output as a simple filter on an unchanged 3D cell.

## Export Matched VASP Inputs

### Question

How do I obtain a POSCAR and KPOINTS that use the same symmetry setting?

### CLI

```bash
fsg structure.mcif --write-poscar-kpoints calculation_inputs
```

### Result

```text
calculation_inputs/POSCAR
calculation_inputs/KPOINTS
```

Both use the ACC primitive real-space setting. Keep them paired. Existing files
with those names are replaced.
