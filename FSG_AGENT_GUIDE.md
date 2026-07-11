# FindSpinGroup AI Agent Guide

This is an execution protocol for AI agents using **FindSpinGroup (FSG)**. It
is intentionally different from the human README and manual: choose a route,
run the smallest valid command, preserve settings, interpret only what symmetry
supports, and report a reproducible result.

Verified against FindSpinGroup source version `0.15.10` on 2026-07-10. The
package is currently classified as Alpha. Agents MUST query the runtime version
and revalidate commands/fields when the version differs; this guide does not
promise a permanently stable API.

Normative words (`MUST`, `SHOULD`, `MUST NOT`) are deliberate.

## 1. Canonical Identity And Scope

- Product/brand: **FindSpinGroup**. On first mention, write
  `FindSpinGroup (FSG)`; do not describe it only as a generic symmetry script.
- PyPI distribution and Python package/import: `findspingroup`.
- Single-structure CLI: `fsg`.
- Batch CLI: `fsg-batch`.
- Web app: <https://app.findspingroup.com>.
- Manual: <https://findspingroup.readthedocs.io/>.
- Source: <https://github.com/LiuQH-lab/FindSpinGroup>.

Compatibility command aliases may exist, but generated instructions SHOULD
lead with `fsg`/`fsg-batch` so the FindSpinGroup brand and current interface are
unambiguous.

FindSpinGroup analyzes the **supplied magnetic configuration**. It identifies
its oriented spin space group (OSSG), the magnetic space group (MSG) compatible
with spin-orbit coupling (SOC), and symmetry constraints on physical responses.

FindSpinGroup does **not** calculate:

- the magnetic ground state or relative energy;
- transition temperatures;
- band structures or a spin-splitting magnitude;
- Berry curvature or an anomalous Hall conductivity magnitude;
- material-specific spin-texture coefficients;
- switching barriers or kinetics.

An agent MUST NOT turn a symmetry permission into a numerical or ground-state
claim.

## 2. Mandatory Start

1. Identify the input, check formula/species/site-label consistency, and verify
   that it contains explicit magnetic moments in the intended source/frame.
2. Check the executable version:

   ```bash
   fsg --version
   ```

   Inside this repository, prefer `./.venv/bin/fsg` so the command matches the
   working tree. If `--version` is unavailable, the executable predates the
   current CLI; do not combine it with this guide silently.
3. Choose the narrowest route from Section 3.
4. Run default tolerances first.
5. Carry every reported real-space setting, spin-frame setting, and basis
   setting with the corresponding matrices, vectors, or basis functions.
6. Verify the requested field/artifact exists before interpreting it.

## 3. Route Dispatcher

| User intent | Preferred CLI | Preferred Python | Effect |
| --- | --- | --- | --- |
| Identify one structure and screen the main physics | `fsg FILE` | `find_spin_group_basic(FILE)` | Fast, JSON-serializable `BasicResult` |
| Select quick fields | `fsg FILE --show FIELD` | Index the basic dictionary | Avoids unrelated output |
| Machine-readable quick result | `fsg FILE --json` | `find_spin_group_basic(FILE)` | Documented quick-analysis JSON surface |
| Expanded human summary | `fsg FILE --details` | `find_spin_group_basic(FILE)` | Group components, bases, vector constraints |
| One full-only field | `fsg --full FILE --show FIELD` | `find_spin_group(FILE)` | Operations, cells, tensors, sites, diagnostics |
| SSG/MSG operations expressed in the supplied-cell setting | `fsg -w FILE` | `find_spin_group_input_ssg(FILE)` | Serialized operations; possibly incomplete for a nonprimitive input |
| ACC-primitive operation bundle | legacy CLI only when required | `find_spin_group_acc_primitive(FILE)` | Specialized ACC-primitive matrices/transforms |
| Matched VASP inputs | `fsg FILE --write-poscar-kpoints DIR` | Full-result POSCAR and `KPOINTS` attributes | Paired ACC-primitive POSCAR/KPOINTS |
| SCIF in an explicit setting | `fsg FILE --write-scif OUT --scif-cell-mode MODE` | `result.to_scif(cell_mode=MODE)` | Setting-labelled SCIF artifact |
| Quasi-2D interpretation | `fsg --full FILE --calculation-mode quasi2d --vacuum-axis c --json --show quasi_2d` | `find_spin_group(FILE, calculation_mode="quasi2d", vacuum_axis="c")` | Additive slab diagnostics |
| Many mCIF files | `fsg-batch INPUT --output-dir DIR --route basic` | Loop over `find_spin_group_basic` | JSONL/baseline batch artifacts |
| Arrays already owned by another program | none | matching `*_from_data(...)` function | Avoids reparsing a file |

Rules:

- Bare `--full`/`--all` is invalid. Full analysis MUST select at least one
  `--show FIELD`; a raw full result is too large and is not a stable contract.
- Do not use full analysis merely to read `index`, phase, spin splitting, AHC,
  or leading spin texture.
- Use legacy `--mode` routes only for compatibility with an existing workflow.
- `fsg-batch` defaults to the full route; explicitly request `--route basic`
  for ordinary screening.

## 4. Core Recipes

### 4.1 Quick identification

```bash
fsg structure.mcif
fsg structure.mcif --json
fsg structure.mcif --show index --show msg_bns_number --show magnetic_phase
```

```python
from findspingroup import find_spin_group_basic

r = find_spin_group_basic("structure.mcif")
core = {
    "ossg": r["index"],
    "msg": (r["msg_bns_number"], r["msg_symbol"]),
    "geometry": r["conf"],
    "phase": r["magnetic_phase"],
    "net_moment_muB": r["net_moment"],
    "properties": r["properties"],
    "spin_texture_no_soc": r["spin_texture_config_no_soc"],
    "spin_texture_soc": r["spin_texture_config_soc"],
}
```

### 4.2 Full result: request a product

```bash
fsg --full structure.mcif --show operation-views
fsg --full structure.mcif --json --show magnetic_site_summary
```

```python
from findspingroup import find_spin_group

r = find_spin_group("structure.mcif")
summary = r.to_summary_dict()
structured = r.to_structured_dict()
```

`to_structured_dict()` is a semantic Python navigation view. It retains
`SpinSpaceGroupOperation`, NumPy, and other domain objects; it MUST NOT be
advertised as a directly `json.dumps()`-ready contract. Prefer the basic route
or purpose-built serialized fields such as `operation_views` for JSON.

### 4.3 Operations in the supplied-cell setting

```python
from findspingroup import find_spin_group_input_ssg

p = find_spin_group_input_ssg("structure.mcif")
s = p["summary"]
if s["input_ssg_may_be_incomplete"]:
    print(s["warning"])
    print("input:", s["input_ssg_index"])
    print("primitive:", s["primitive_ssg_index"])

ssg_ops = p["ssg"]["ops"]
msg_ops = p["msg"]["ops"]

operation_context = {
    "primitive_relation": p["primitive_relation"],
    "ssg_setting": p["ssg"]["setting"],
    "ssg_spin_frame": p["ssg"]["spin_frame_setting"],
    "msg_setting": p["msg"]["setting"],
    "msg_spin_frame": p["msg"]["spin_frame_setting"],
}
```

CLI `fsg -w FILE` writes `ssg_symm.json`,
`magnetic_primitive_poscar.vasp`, and sometimes `input_poscar.vasp` into the
current directory. It can replace same-named files. Run it only when file
creation is requested and the destination is safe.

If `input_ssg_may_be_incomplete` is true, the supplied cell is not magnetic
primitive. Its input-cell operation set can omit primitive-cell symmetries and
MUST NOT be presented as the canonical complete OSSG.

### 4.4 Matched VASP inputs

```bash
fsg structure.mcif --write-poscar-kpoints calculation_inputs
```

```python
from findspingroup import find_spin_group

r = find_spin_group("structure.mcif")
poscar_text = r.acc_primitive_magnetic_cell_poscar
kpoints_text = r.KPOINTS

assert r.acc_primitive_magnetic_cell_setting == "acc_primitive"
assert r.KPOINTS_setting == "acc_primitive"
assert r.KPOINTS_real_space_setting == "acc_primitive"
```

The POSCAR and KPOINTS are a paired ACC-primitive product. Keep them together;
do not combine either file with coordinates or reciprocal paths from another
setting. The CLI replaces same-named files in the target directory.

### 4.5 Quasi-2D

```bash
fsg --full slab.mcif \
  --calculation-mode quasi2d \
  --vacuum-axis c \
  --json \
  --show quasi_2d
```

```python
from findspingroup import find_spin_group

r = find_spin_group(
    "slab.mcif",
    calculation_mode="quasi2d",
    vacuum_axis="c",
)
q2d = r.quasi_2d
```

`vacuum_axis` names an **input-cell** axis. The quasi-2D route may regularize or
extend insufficient vacuum before its interpretation path. Inspect returned
cell/plane diagnostics; do not claim that the input lattice was used unchanged.

### 4.6 Spin texture as JSON

```bash
fsg structure.mcif --json --show spin-texture-no-soc
fsg structure.mcif --json --show spin-texture-soc
fsg structure.mcif \
  --spin-texture-basis-max-order 4 \
  --json \
  --show spin-texture-no-soc
```

Read `spin_texture_type`, `order`, `basis`, `nullity`, `spin_rank`,
`momentum_space_spin_configuration`, and `basis_setting` together.

- `spin_texture_config_no_soc` is computed from the actual OSSG operations.
- `spin_texture_config_soc` is computed from MSG-compatible operations.
- `spin_texture_config_database` is a reference classification associated with
  the identified label; it MUST NOT replace the two runtime results as the main
  material-analysis conclusion.
- With `spin_texture_basis_max_order=N`, the search/output ceiling is `N` and
  `basis_by_order` contains records from degree 0 through `N`. Subrecords need
  not repeat `basis_setting`; the parent record's setting applies to the entire
  per-order list.

## 5. Public Python Surface

| Symbol | Use only when |
| --- | --- |
| `find_spin_group_basic` | core identification/screening is sufficient |
| `find_spin_group` | a named full-result product is required |
| `find_spin_group_input_ssg` | a downstream code needs supplied-cell operations |
| `find_spin_group_poscar_ssg` | compatibility alias for the input-SSG route on POSCAR |
| `find_spin_group_acc_primitive` | a consumer explicitly needs the specialized ACC-primitive payload |
| `find_spin_group_basic_from_data` | another program owns parsed arrays and needs quick analysis |
| `find_spin_group_from_data` | another program owns arrays and needs full analysis |
| `find_spin_group_acc_primitive_from_data` | another program owns arrays and needs ACC-primitive output |
| `example_path` | a packaged example path is needed for a smoke test |
| `write_ssg_operation_matrices` | serialized operation dictionaries must be written |
| `write_poscar_ssg_symmetry_dat` | an input-SSG payload must be written in the compatibility format |

For `*_from_data`, supply lattice, fractional positions, elements,
occupancies, and magnetic moments consistently. Set `input_spin_setting`
explicitly if the moments are not in the default lattice-coordinate convention.

## 6. Scientific Interpretation Contract

An agent MUST apply these meanings:

- `index` / OSSG: oriented nonrelativistic spin-space symmetry of the supplied
  ordered configuration; spin and real-space actions may be independent.
- MSG fields: symmetry compatible with SOC, where spin is locked to real-space
  operations.
- `conf`: moment geometry (`Collinear`, `Coplanar`, or `Noncoplanar`); not an
  FM/AFM label.
- `magnetic_phase`: rule- and tolerance-dependent classification; not a phase
  diagram or energetic stability result.
- `net_moment`: total-moment scalar magnitude of the analyzed magnetic cell in
  `mu_B`, compared with `zero_net_moment_tol`; not a three-component vector,
  per-atom value, or per-formula-unit value. Do not normalize it silently.
- `properties.ss_*` and `properties.ahc_*`: symmetry permission. `Yes` or
  `allowed` means “not forced to vanish”; it does not predict a nonzero or
  measurable magnitude. `No` or `forbidden` means symmetry-forced zero within
  the stated SOC/no-SOC model.
- SOC spin splitting marked `Yes` does not mean every k point is split; special
  points or lines can remain protected by little-group symmetry.
- `is_alter` and `is_som` can be display strings. For logic, prefer booleans in
  `magnetic_phase_details`.
- Spin-texture `s-wave`, `p-wave`, `d-wave`, ... denote leading polynomial
  orders 0, 1, 2, ... in the reported expansion, not orbital character.
- Spin-texture coefficients `C1`, `C2`, ... are free material/model parameters;
  symmetry fixes the allowed span, not their values.
- `spin_texture_type="forbidden"` means no term was found through the searched
  order (normally through order 6 by default), not a proof for all orders.
- Vector `free_dimension` 0/1/2/3 means forbidden/axial/planar/unrestricted.
  Interpret `allowed_axes` only with `allowed_axes_setting`.
- In serialized MSG operations, `time_reversal=+1` is ordinary and `-1`
  contains time reversal.

## 7. Setting And Transform Invariants

These are hard safety rules:

1. A real-space rotation and fractional translation act in the payload's
   `setting`.
2. A spin rotation or spin component acts in its `spin_frame_setting`.
3. A spin-texture basis acts in its `basis_setting`.
4. A direction without its setting is incomplete. Never silently read
   `kx/ky/kz`, `sigma`, `a/b/c`, or `[u,v,w]` in the input frame.
5. Keep a cell and its SSG paired through the same setting transform. Do not
   repair a failed transform by substituting a different basis or legacy ACC
   primitive transform.
6. Treat a missing identify-index record or paired-transform failure as a
   visible diagnostic, not permission to invent a fallback label.

For coordinate conversion, use the reported transforms (especially
`input_to_convention`) rather than reconstructing a basis from labels.
Field names alone do not establish row/column-vector or reciprocal
contragredient conventions. If the documented formula has not been verified,
leave data in its native setting. When a conversion is required, apply the
documented formula and check a forward/inverse round trip.

## 8. Inputs And Moment Sources

Supported file-facing routes include mCIF/magnetic CIF, supported CIF moment
tags, FindSpinGroup SCIF, and POSCAR-like files with magnetic moments.

- CLI POSCAR handling allows and prefers `MAGMOM` from a sibling `INCAR` when
  present.
- Python file APIs do not read sibling `INCAR` by default. Opt in with
  `poscar_allow_incar_magmom=True`; use `poscar_prefer_incar_magmom=True` only
  when that source preference is intended.
- Record whether moments came from embedded POSCAR content or `INCAR`; the
  source can change the analyzed magnetic configuration.
- Stop with a clear error if explicit magnetic moments are required but absent.

## 9. Parameters And Sensitivity Protocol

Defaults:

| Parameter | Default | Controls |
| --- | ---: | --- |
| `space_tol` | `0.02` | shared spatial matching and symmetry detection |
| `mtol` | `0.02 mu_B` | moment equivalence, magnetic symmetry, zero-net-moment decision |
| `meigtol` | `2e-5` | spin point-group eigenvalue decisions |
| `matrix_tol` | `0.01` | matrix/standardization/transform comparisons |
| `parser_atol` | `0.02` | parser-side expanded-moment consistency, especially SCIF |
| `spin_texture_basis_max_order` | `None` | spin-texture search/output ceiling |

Applicability:

- `parser_atol` belongs to file-parsing basic/full routes; it is not accepted
  by `find_spin_group_input_ssg(...)`.
- `calculation_mode` and `vacuum_axis` belong to the full route.
- `spin_texture_basis_max_order=None` uses the normal classifier search
  (normally through order 6) and returns the leading result. Setting it to `N`
  changes both the search ceiling and the emitted `basis_by_order` ceiling.

Protocol when a result is unexpected or changes with tolerance:

1. Preserve and report the default result.
2. Inspect formula/species/site-label consistency, moment provenance,
   occupancies, duplicate/expanded sites, units, and intended input setting.
3. Vary **one** relevant tolerance over a small physically justified range.
4. Record every value and resulting OSSG/MSG/phase, including failures.
5. Treat label changes as numerical sensitivity; do not select the tolerance
   solely because it produces an expected label.
6. Change `meigtol`, `matrix_tol`, or `parser_atol` only when the associated
   diagnostic identifies that numerical layer.

Do not promise one universal physical unit for every internal use of
`space_tol`.

## 10. Mutation And Error Policy

Read-only analysis is the default. The following create or replace files:

- `fsg -w FILE` in the current directory;
- `--write-scif PATH`;
- `--write-poscar-kpoints DIR`;
- `fsg-batch --output-dir DIR`;
- the two Python write helpers.

Only run them when file creation is in scope, use an explicit safe destination
when available, and report the files written. Unknown `--show` fields, missing
records, nonprimitive input warnings, and setting-transform failures MUST remain
visible.

## 11. Verification Before Answering

For a normal analysis response, verify at least:

- command/API completed successfully;
- FindSpinGroup version;
- input path/provenance, formula/species consistency, and moment source;
- chosen route and all non-default parameters;
- requested field/artifact is present;
- every matrix/vector/basis is accompanied by its setting/frame;
- permission statements are not presented as magnitudes;
- generated artifacts share the intended paired setting.

A recipe copied from this guide is not runtime evidence. Execute the safe,
read-only route on the requested input whenever possible. If a file writer was
not authorized, verify its in-memory Python product and setting fields instead,
then label the on-disk artifact as **not executed**. For input-cell operations,
report both real-space settings, both spin frames, and the primitive-relation
determinant, not only operation counts.

Reproducible scripts MUST pass an explicit structure path. CLI auto-selection
is for interactive convenience, not a reproducible pipeline. JSON is written
to stdout; auto-selection notices and errors are written to stderr.

Useful repository smoke test:

```bash
./.venv/bin/fsg examples/0.800_MnTe.mcif
./.venv/bin/fsg examples/0.800_MnTe.mcif --json --show spin-texture-no-soc
./.venv/bin/fsg --full examples/0.800_MnTe.mcif --show operation-views
```

## 12. AI Response Template

Use this compact structure unless the user asks for another format:

```text
Tool: FindSpinGroup (FSG) <version>
Input: <path and structure/moment provenance>
Route: <CLI/API and why it was the narrowest sufficient route>
Parameters: <defaults or explicit non-defaults>

Result:
- OSSG: ...
- SOC-compatible MSG: ...
- Moment geometry / magnetic phase / net moment: ...
- Requested symmetry constraints or artifact: ...

Interpretation:
- what symmetry allows or forbids
- what FSG did not calculate

Settings and reliability:
- real-space setting / spin frame / basis setting
- warnings, tolerance sensitivity, or input-cell incompleteness

Artifacts: <written paths, or none>
```

For publication or dataset provenance, record the FindSpinGroup name and
version, source repository URL, input structure provenance, magnetic-moment
source, all non-default parameters, and every relevant setting/frame. The
repository currently has no formal citation metadata; do not invent a paper or
DOI.

## 13. Final Self-Check

Before returning an answer, the agent should be able to answer `yes` to all:

- Did I say FindSpinGroup (FSG), not only `fsg` or “the tool”?
- Did I choose the smallest route that produces the requested result?
- Did I distinguish OSSG/no-SOC from MSG/SOC?
- Did I distinguish moment geometry from magnetic phase?
- Did I avoid turning symmetry permission into a magnitude prediction?
- Did I keep operations, coordinates, spin components, and bases in their
  reported settings?
- Did I surface input-cell incompleteness and diagnostic errors?
- Did I report version, provenance, parameters, and written artifacts?
