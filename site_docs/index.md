# FindSpinGroup

**From a magnetic structure to its spin-space symmetry and symmetry-allowed
physical responses.**

FindSpinGroup identifies the oriented spin space group (OSSG) of a supplied
magnetic configuration, its spin-orbit-compatible magnetic space group (MSG),
and symmetry constraints on spin splitting, anomalous Hall conductivity, spin
texture, polar axes, magnetic sites, and related quantities.

> FindSpinGroup analyzes symmetry. It does not calculate energies, transition
> temperatures, band splittings, conductivity magnitudes, or magnetic ground
> states.

## Start In One Minute

```bash
python -m pip install --upgrade findspingroup
fsg path/to/structure.mcif
```

Or from Python:

```python
from findspingroup import find_spin_group_basic

result = find_spin_group_basic("path/to/structure.mcif")

print(result["index"])
print(result["magnetic_phase"])
print(result["msg_bns_number"], result["msg_symbol"])
print(result["properties"])
```

## Read The Result In This Order

| Read first | Question answered | Scientific meaning |
| --- | --- | --- |
| `index`, `ossg_symbol_linear` | What is the nonrelativistic spin-space symmetry? | OSSG symmetry of the supplied ordered magnetic configuration. |
| `msg_bns_number`, `msg_symbol` | What symmetry is compatible with SOC? | MSG after spin and real-space rotations are locked. |
| `conf` | How are the moments geometrically arranged? | Collinear, coplanar, or noncoplanar; not an FM/AFM label. |
| `magnetic_phase` | How is the magnetic configuration classified? | Rule-based, tolerance-dependent phase label. |
| `properties` | Which responses are symmetry-allowed? | Spin splitting and AHC permission with and without SOC. |
| `spin_texture_config_*` | What is the leading momentum dependence? | Lowest-order symmetry-allowed spin-texture polynomial in the reported frame. |

`allowed` means “not forced to vanish by the analyzed symmetry.” It does not
mean that a response is necessarily large or experimentally observable.

## Choose By Your Goal

| I want to... | Start here |
| --- | --- |
| identify one material and understand the main output | [Quickstart](guide/getting-started.md) |
| understand the physical meaning and limitations | [Interpret Your Result](guide/understanding-results.md) |
| choose between quick, full, and input-cell analysis | [Choose a Workflow](guide/choosing-an-api.md) |
| decide whether to change tolerances | [Parameters and Reliability](guide/reliability-and-tolerances.md) |
| run commands or export files | [CLI by Task](reference/cli.md) |
| integrate FindSpinGroup in Python | [Python API](reference/python-api.md) |
| inspect every field and nested payload | [Result Schemas](reference/result-schemas/index.md) |

## Three Analysis Levels

### Quick analysis

Use `fsg FILE` or `find_spin_group_basic(FILE)` for identification, screening,
and the main physics-facing constraints. This is the right starting point for
most users.

### Full analysis

Use `fsg --full FILE --show FIELD` or `find_spin_group(FILE)` when you need
cells, operations, tensor constraints, magnetic-site analysis,
SCIF/POSCAR/KPOINTS, quasi-2D diagnostics, or route audits. The CLI requires a
selected field so a huge raw result cannot bury the requested information.

### Input-cell operation export

Use `fsg -w FILE` or `find_spin_group_input_ssg(FILE)` when a downstream code
needs SSG and MSG operations expressed in the user-supplied cell. Check the
returned warning before using operations from a non-primitive input cell.

## Before You Publish A Result

1. Record the FindSpinGroup version and all non-default parameters.
2. Confirm that the input magnetic moments and their coordinate convention are
   correct.
3. Check tolerance sensitivity for noisy, nearly symmetric, or very small
   moments.
4. Keep every operation, coordinate, spin-texture basis, and generated file
   attached to its reported cell/spin-frame setting.
5. Describe symmetry outputs as allowed or forbidden constraints, not computed
   response magnitudes.
