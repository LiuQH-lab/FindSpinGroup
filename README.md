# FindSpinGroup

FindSpinGroup takes a crystal structure with magnetic moments and identifies:

1. its **oriented spin space group (OSSG)** in the nonrelativistic limit;
2. the corresponding **magnetic space group (MSG)** when spin-orbit coupling
   locks spin and real space; and
3. symmetry constraints on spin splitting, anomalous Hall conductivity, spin
   texture, polar axes, magnetic sites, and related observables.

Web application: [app.findspingroup.com](https://app.findspingroup.com)
Visualization: [view.findspingroup.com](https://view.findspingroup.com)
SSG Database: [database.findspingroup.com](https://database.findspingroup.com)

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
pip install --upgrade findspingroup
fsg --version
```

FindSpinGroup requires Python 3.11 or newer.


## Quick Start

Analyze a magnetic CIF, SCIF, or POSCAR input containing magnetic moments (including #MAGMOM= ... in POSCAR):

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
diagnostic fields are kept in Reference.

AI agents and tool-using models should start with the dedicated
[FindSpinGroup AI Agent Guide](https://github.com/LiuQH-lab/FindSpinGroup/blob/main/FSG_AGENT_GUIDE.md),
which provides a compact route-selection and scientific-interpretation
protocol rather than another human tutorial.

## How to cite FindSpinGroup

If FindSpinGroup contributes to published work, cite:

> Y. Yu, X. Chen, Y. Zhu, Y. Li, R. Xiong, J. Li, Y. Liu, and Q. Liu,
> "Identifying Oriented Spin Space Groups and Related Physical Properties
> Using an Online Platform FINDSPINGROUP," arXiv:2604.21397 (2026).
> [https://doi.org/10.48550/arXiv.2604.21397](https://doi.org/10.48550/arXiv.2604.21397)

```bibtex
@article{Yu2026FindSpinGroup,
  title   = {Identifying Oriented Spin Space Groups and Related Physical Properties Using an Online Platform FINDSPINGROUP},
  author  = {Yu, Yutong and Chen, Xiaobing and Zhu, Yanzhou and Li, Yuhui and Xiong, Renzheng and Li, Jiayu and Liu, Yuntian and Liu, Qihang},
  journal = {arXiv preprint arXiv:2604.21397},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.21397},
  url     = {https://arxiv.org/abs/2604.21397}
}
```


## License

Apache License 2.0. See [`LICENSE`](LICENSE).
