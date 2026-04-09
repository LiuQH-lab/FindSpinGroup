# findspingroup

`findspingroup` is a Python toolkit for identifying and analyzing spin space
group symmetry in magnetic crystal structures.

It is designed for:

- magnetic symmetry identification from `.cif`, `.mcif`, and generated `.scif`
- standardized primitive and conventional setting construction
- symmetry-aware downstream workflows such as tensor analysis and k-point work
- high-throughput screening on large magnetic structure sets

## What you get

Given a supported structure file, `findspingroup` can return:

- the identified spin-space-group `index`
- the arithmetic crystal class `acc`
- MSG number and symbol
- public convention symbols such as `convention_ssg_international_linear`
- symmetry-derived outputs such as `.scif`, tensor summaries, and k-path data

## Quick start

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("0.800_MnTe.mcif"))
print(result.index)
print(result.acc)
print(result.convention_ssg_international_linear)
```

## Next steps

- See [Installation](installation.md) for package installation options.
- See [Usage](usage.md) for the main Python API.
- See [Examples](examples.md) for packaged sample inputs.
- See [SCIF](scif.md) for `.scif` export and roundtrip notes.
- See [CLI](cli.md) for command-line entry points.
