# FINDSPINGROUP

`findspingroup` is a Python toolkit, command-line program, and
[web application](https://app.findspingroup.com) for identifying and inspecting
oriented spin space group (OSSG) symmetry in magnetic crystal structures.

It is designed for research workflows involving the interplay between
exchange-driven magnetic geometry and spin-orbit coupling, which are described
by spin space group (SSG) and magnetic space group (MSG) frameworks,
respectively.

Given a magnetic structure, FINDSPINGROUP identifies the OSSG, derives the
corresponding MSG, and organizes crystallographic and physical information
needed to analyze the material with and without spin-orbit coupling.

## Main outputs

Main outputs can include:

- OSSG information and corresponding MSG information in matched settings;
- spin Wyckoff positions and Wyckoff splitting from space group (SG) to OSSG and MSG;
- spin Brillouin zones, high-symmetry k points, and symmetry-allowed spin-polarization components;
- magnetic-phase classification, including unconventional cases such as altermagnets and spin-orbit magnets;
- symmetry constraints on anomalous Hall conductivity, nonlinear tensors, and related physical responses;
- chiral and polar group information with and without spin-orbit coupling;
- `.scif` files for downstream spin-group-based tensor analysis and data exchange;
- magnetic primitive-cell POSCAR files in relevant coordinate conventions;
- KPOINTS files labeled with symmetry-allowed spin-polarization components.

## Quick start

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))
print(result.index)
print(result.convention_ssg_international_linear)
print(result.magnetic_phase)
```

## Next steps

- See [Installation](installation.md) for package installation options.
- See [Usage](usage.md) for `MagSymmetryResult`, basic summaries, and input-SSG outputs.
- See [Examples](examples.md) for packaged sample inputs.
- See [SCIF](scif.md) for `.scif` export and roundtrip notes.
- See [CLI](cli.md) for command-line entry points.
