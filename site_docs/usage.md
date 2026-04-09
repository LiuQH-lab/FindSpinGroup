# Usage

## Main Python entry point

```python
from findspingroup import find_spin_group

result = find_spin_group("path/to/structure.mcif")
```

Main signature:

```python
def find_spin_group(
    cif: str,
    space_tol: float = 0.02,
    mtol: float = 0.02,
    meigtol: float = 0.00002,
    matrix_tol: float = 0.01,
) -> MagSymmetryResult
```

## Supported input formats

`findspingroup` currently supports:

- standard `.cif`
- magnetic `.mcif`
- repo-generated `.scif`
- a narrow first-version magnetic `POSCAR` contract used by this project

## Common result fields

Commonly used attributes on the returned `MagSymmetryResult` include:

- `index`
- `acc`
- `conf`
- `msg_num`
- `msg_symbol`
- `convention_ssg_international_linear`
- `gspg_symbol_linear`
- `scif`

Example:

```python
from findspingroup import example_path, find_spin_group

result = find_spin_group(example_path("1.237_VCl2.mcif"))

print("index:", result.index)
print("acc:", result.acc)
print("msg:", result.msg_symbol)
print("convention:", result.convention_ssg_international_linear)
```

## Notes

- The result object exposes many more fields than those listed here.
- For scripting and downstream workflows, users are expected to read the fields
  they need directly from the returned object.
