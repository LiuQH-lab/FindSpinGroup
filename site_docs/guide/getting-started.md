# Quickstart

This page takes you from installation to a result you can interpret. It uses
MnTe only as a verification case; the same commands accept your own magnetic
structure.

## 1. Install And Check The Version

```bash
python -m pip install --upgrade findspingroup
fsg --version
```

FindSpinGroup requires Python 3.11 or newer. The installed version should match
the documentation version you are reading.

If `fsg --version` is not recognized, the executable on `PATH` predates the
current interface. Install the current source checkout or switch to the stable
manual that matches the installed package before following later commands.

## 2. Run One Magnetic Structure

Your file must contain magnetic moments.

Copy/paste verification with the bundled MnTe example:

```bash
MNTE="$(python -c 'from findspingroup import example_path; print(example_path("0.800_MnTe.mcif"))')"
fsg "$MNTE"
```

Then run your own structure:

```bash
fsg path/to/structure.mcif
```

The quick analysis prints a short scientific summary. For the bundled MnTe
example, the important lines are:

```text
OSSG: 194.164.1.1.L
MSG with SOC: 63.457 Cmcm
Magnetic order: Collinear; AFM(Altermagnet)
Spin splitting: without SOC k-dependent; with SOC allowed
AHC: without SOC forbidden; with SOC forbidden
Leading spin texture: without SOC g-wave; with SOC d-wave
```

## 3. Understand Those Six Lines

**OSSG.** The oriented spin space group of the supplied magnetic configuration
in the nonrelativistic spin-space-symmetry description.

**MSG with SOC.** The magnetic space group compatible with spin-orbit coupling,
written here by BNS number and symbol.

**Collinear.** All nonzero ordered moments lie along one spin-space axis. This
says nothing by itself about whether the order is FM or AFM.

**AFM(Altermagnet).** FindSpinGroup's rule-based classification of this supplied
configuration. It is derived from symmetry, moment geometry, net moment, and
the selected tolerance; it is not an energetic ground-state calculation.

**Spin splitting.** `k-dependent` without SOC means symmetry permits
momentum-dependent nonrelativistic spin splitting. `allowed` with SOC means the
SOC-compatible symmetry does not force all splitting to vanish. Neither
statement gives an energy magnitude.

**AHC: forbidden.** The analyzed symmetry forces the anomalous Hall response to
vanish in the stated no-SOC or SOC model. Conversely, `allowed` would not
guarantee a nonzero conductivity.

**Leading spin texture.** The `g-wave`/`d-wave` labels identify the lowest
momentum-polynomial order allowed by the corresponding no-SOC/SOC symmetry.
They do not describe orbital angular momentum or determine the free material
coefficients. See [Interpret Your Result](understanding-results.md#spin-texture).

## 4. Ask For The Next Layer, Not Everything

Show a few scalar fields:

```bash
fsg structure.mcif \
  --show index \
  --show magnetic_phase \
  --show msg_bns_number \
  --show msg_symbol
```

Inspect physics-facing flags:

```bash
fsg structure.mcif --show properties
```

Inspect the leading spin texture without SOC:

```bash
fsg structure.mcif --show spin-texture-no-soc
```

Use `--details` when you intentionally want G0/L0 components, point groups,
the full basis expressions, and vector constraints. Use `--json` when another
program will consume the quick-analysis dictionary.

## 5. Run The Same Analysis In Python

```python
from findspingroup import example_path, find_spin_group_basic

result = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print("OSSG:", result["index"])
print("MSG:", result["msg_bns_number"], result["msg_symbol"])
print("order:", result["conf"], result["magnetic_phase"])
print("responses:", result["properties"])
```

Expected core output:

```text
OSSG: 194.164.1.1.L
MSG: 63.457 Cmcm
order: Collinear AFM(Altermagnet)
responses: {'ss_w_soc': 'Yes', 'ss_wo_soc': 'k-dependent', ...}
```

The Python dictionary keeps machine-facing labels such as `Yes`, `No`,
`k-dependent`, and `Zeeman`. The **default CLI summary** translates `Yes`/`No`
into `allowed`/`forbidden` for readability; `--details`, `--show`, and JSON keep
the underlying labels.

## 6. Decide Whether You Need Full Analysis

Stay with quick analysis for identification and screening. Move to full
analysis only when you need operations, cells, tensors, magnetic sites, or
generated files:

```bash
fsg --full structure.mcif --show operation-views
fsg structure.mcif --write-poscar-kpoints calculation_inputs
fsg structure.mcif --write-scif structure.scif
```

In Python:

```python
from findspingroup import find_spin_group

result = find_spin_group("structure.mcif")
summary = result.to_summary_dict()
structured = result.to_structured_dict()
```

## If The First Run Fails

The most common cause is missing or incorrectly interpreted magnetic moments.
Check [Input Formats](input-formats.md) first. If the identified group is
unexpected, do not immediately loosen all tolerances; follow
[Parameters and Reliability](reliability-and-tolerances.md).
