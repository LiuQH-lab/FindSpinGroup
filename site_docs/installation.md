# Installation

## From PyPI

```bash
pip install findspingroup
```

Python requirement:

- `>= 3.11`

## From source

```bash
git clone https://github.com/LiuQH-lab/FindSpinGroup.git
cd FindSpinGroup
pip install -e .
```

## Verify the install

```bash
python - <<'PY'
from findspingroup import example_path, find_spin_group
result = find_spin_group(example_path("0.800_MnTe.mcif"))
print(result.index)
print(result.convention_ssg_international_linear)
PY
```

If the install is working, this command should print a valid `index` and a
convention-space-group symbol without raising an exception.
