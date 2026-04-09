# Examples

The package includes a small set of example inputs that can be accessed after
installation.

## Access packaged examples

```python
from findspingroup import example_path

print(example_path("0.800_MnTe.mcif"))
```

## Example names currently bundled with the wheel

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

## Minimal example

```python
from findspingroup import example_path, find_spin_group

path = example_path("2.35_CrSe.mcif")
result = find_spin_group(path)

print(result.index)
print(result.acc)
```
