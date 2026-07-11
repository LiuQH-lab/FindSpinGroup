# FindSpinGroup

FindSpinGroup takes a crystal structure with magnetic moments and identifies:

1. its **oriented spin space group (OSSG)** in the nonrelativistic limit;
2. the corresponding **magnetic space group (MSG)** when spin-orbit coupling
   locks spin and real space; and
3. symmetry constraints on spin splitting, anomalous Hall conductivity, spin
   texture, polar axes, magnetic sites, and related observables.

These are symmetry statements about the magnetic configuration supplied by the
user. FindSpinGroup does **not** calculate an energy scale, transition
temperature, response magnitude, or whether that configuration is the material's
thermodynamic ground state.

Web application: [app.findspingroup.com](https://app.findspingroup.com)

## What Can It Tell Me?

| Scientific question | Main output | Interpretation |
| --- | --- | --- |
| What is the nonrelativistic magnetic symmetry? | OSSG `index` and symbol | Spin and real-space rotations may act independently. |
| What symmetry remains with SOC? | MSG BNS number and symbol | Spin rotations are locked to real-space operations. |
| Are the moments collinear, coplanar, or noncoplanar? | `conf` | Geometry of the supplied ordered moments; not an FM/AFM label. |
| Is the configuration FM-like, AFM-like, or altermagnetic? | `magnetic_phase` | Rule-based classification from symmetry and net moment. |
| Is spin splitting or AHC allowed? | `properties` | Allowed/forbidden by symmetry, not a predicted magnitude. |
| What is the leading spin texture? | `spin_texture_config_*` | Lowest-order symmetry-allowed momentum polynomial in a documented frame. |

## Install

```bash
python -m pip install --upgrade findspingroup
fsg --version
```

FindSpinGroup requires Python 3.11 or newer. To use the current source tree:

```bash
git clone https://github.com/LiuQH-lab/FindSpinGroup.git
cd FindSpinGroup
python -m pip install -e ".[dev]"
```

If `fsg --version` or the options below are unavailable, the executable on your
`PATH` predates this interface. Install the current source or use the manual
version matching the installed package.

## Sixty-Second Start

Analyze a magnetic CIF, SCIF, or POSCAR-like input containing magnetic moments:

```bash
fsg path/to/structure.mcif
```

The default output is deliberately short; its core lines are:

```text
FindSpinGroup result
OSSG: 194.164.1.1.L
MSG with SOC: 63.457 Cmcm
Magnetic order: Collinear; AFM(Altermagnet)
Spin splitting: without SOC k-dependent; with SOC allowed
AHC: without SOC forbidden; with SOC forbidden
Leading spin texture: without SOC g-wave; with SOC d-wave
```

Here, `allowed` means that symmetry does not force the response to vanish. It
does not guarantee a measurable nonzero value.

Ask for only the fields you need:

```bash
fsg structure.mcif --show index --show magnetic_phase --show msg_bns_number
fsg structure.mcif --show properties
fsg structure.mcif --show spin-texture-no-soc
```

Use `--details` for the expanded human-readable symmetry summary, and `--json`
for machine-readable quick-analysis output.

## Python: Choose One Main Function

```python
from findspingroup import find_spin_group_basic

summary = find_spin_group_basic("path/to/structure.mcif")

print(summary["index"])
print(summary["magnetic_phase"])
print(summary["msg_bns_number"], summary["msg_symbol"])
print(summary["properties"])
```

| Goal | Python | CLI | Return value |
| --- | --- | --- | --- |
| Identify symmetry and screen physical constraints | `find_spin_group_basic(...)` | `fsg FILE` | JSON-serializable dictionary |
| Inspect cells, operations, tensors, sites, or generated artifacts | `find_spin_group(...)` | `fsg --full FILE --show FIELD` | `MagSymmetryResult` |
| Export operations in the user-supplied cell | `find_spin_group_input_ssg(...)` | `fsg -w FILE` | Input-cell operation dictionary |

For full analysis, start with the structured accessors instead of the raw
attribute dictionary:

```python
from findspingroup import find_spin_group

result = find_spin_group("path/to/structure.mcif")
summary = result.to_summary_dict()       # compact full-route summary
structured = result.to_structured_dict() # groups, cells, properties, artifacts
```

`to_structured_dict()` is a semantic Python view and retains operation/domain
objects; it is not a directly JSON-serializable contract. Use the basic route
or a purpose-built operation export for machine-readable integration.

## Parameters: Start With The Defaults

Most users should change no numerical tolerances on the first run.

| Parameter | Default | What it controls | Why changing it matters |
| --- | ---: | --- | --- |
| `space_tol` | `0.02` | Shared spatial matching and symmetry-detection tolerance | Can change the identified spatial symmetry. |
| `mtol` | `0.02 μB` | Equivalence of magnetic moments | Can change the SSG and the zero-net-moment phase decision. |
| `meigtol` | `2e-5` | Spin point-group eigenvalue decisions | Advanced numerical diagnostic. |
| `matrix_tol` | `0.01` | Standardization and matrix comparisons | Advanced transform/point-group diagnostic. |
| `parser_atol` | `0.02` | Parser-side moment consistency checks | Tune only for a documented parser-expansion error. |

If a result changes under small, physically reasonable tolerance variations,
treat the classification as numerically sensitive and inspect the input rather
than reporting a single label without qualification.

`space_tol` is shared across symmetry detection and internal position
comparisons; the current route does not expose one uniform unit interpretation
for every use of that value.

## Important Output Boundaries

- `conf` describes the geometry of the moments. `Collinear` does not by itself
  mean ferromagnetic or antiferromagnetic.
- `magnetic_phase` is a symmetry- and tolerance-dependent classification of the
  supplied configuration, not a phase-diagram calculation.
- `properties.ss_*` and `properties.ahc_*` report symmetry permission. They do
  not calculate band splittings or Hall conductivity.
- Spin-texture bases use the reported `basis_setting`; they are not automatically
  expressed along the input-file axes.
- POSCAR and KPOINTS generated together use the same ACC primitive real-space
  setting and should be kept paired.
- Quasi-2D analysis is an explicit interpretation workflow; specify the intended
  vacuum axis and inspect its diagnostic output.

## Supported Inputs

- magnetic CIF / mCIF;
- CIF with supported magnetic-moment tags;
- FindSpinGroup-generated SCIF;
- POSCAR-like files with embedded magnetic moments, or CLI use in a VASP
  directory with a sibling `INCAR` containing `MAGMOM`.

The Python API does not read sibling `INCAR` files unless explicitly requested;
this keeps scripted calls reproducible. See the
[input guide](https://findspingroup.readthedocs.io/en/latest/guide/input-formats/)
for the exact behavior.

## Documentation

Start with:

- [Quickstart](https://findspingroup.readthedocs.io/en/latest/guide/getting-started/)
- [How to interpret the result](https://findspingroup.readthedocs.io/en/latest/guide/understanding-results/)
- [Choose a workflow or API](https://findspingroup.readthedocs.io/en/latest/guide/choosing-an-api/)
- [Parameters, tolerances, and reliability](https://findspingroup.readthedocs.io/en/latest/guide/reliability-and-tolerances/)
- [CLI by task](https://findspingroup.readthedocs.io/en/latest/reference/cli/)
- [Python API](https://findspingroup.readthedocs.io/en/latest/reference/python-api/)

The complete manual is published on
[Read the Docs](https://findspingroup.readthedocs.io/). Detailed schemas and
diagnostic fields are kept in Reference so that they do not obscure the main
scientific workflow.

AI agents and tool-using models should start with the dedicated
[FindSpinGroup AI Agent Guide](https://github.com/LiuQH-lab/FindSpinGroup/blob/main/FSG_AGENT_GUIDE.md),
which provides a compact route-selection and scientific-interpretation
protocol rather than another human tutorial.

## Citation And Reproducibility

The repository does not yet provide formal citation metadata. Until a project
citation is published, report the FindSpinGroup version, repository URL, input
structure provenance, all non-default parameters, and the setting/frame of any
exported operations or basis functions.

## Development

```bash
python -m pip install -e ".[dev,docs]"
PYTHONPATH=src python -m pytest
mkdocs build --strict
```

Changes to symmetry identification, public fields, cells, or generated
artifacts should also be checked with the repository's focused and batch
regression workflows.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
