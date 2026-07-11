# Troubleshooting By Symptom

Start from the symptom and inspect the narrowest relevant output. Avoid changing
several tolerances at once.

## “No Magnetic Moments Were Found”

FindSpinGroup requires a magnetic configuration.

- For mCIF/CIF/SCIF, confirm that supported magnetic-moment fields are present.
- For POSCAR-like input, add embedded magnetic moments or run the CLI in a VASP
  directory with a sibling `INCAR` containing `MAGMOM`.
- Remember that Python does not read sibling `INCAR` files unless
  `poscar_allow_incar_magmom=True`.

See [Input Formats](input-formats.md).

## The Identified Group Is Unexpected

First inspect the classification context:

```bash
fsg structure.mcif \
  --show index \
  --show conf \
  --show net_moment \
  --show tolerances \
  --show identify_index_details
```

Then check:

1. moment coordinate frame and units;
2. whether the file contains the intended magnetic cell/configuration;
3. near-duplicate atomic positions or occupancies;
4. moments close to zero or close to a collinear/coplanar boundary;
5. sensitivity to one justified tolerance at a time.

Do not loosen every tolerance until the expected label appears. Follow
[Parameters And Reliability](reliability-and-tolerances.md).

## The Magnetic Phase Looks Wrong

`magnetic_phase` is FindSpinGroup's rule-based symmetry classification, not a
literature-name lookup. Inspect:

```bash
fsg structure.mcif --show magnetic_phase_details
```

Pay particular attention to:

- `classification_rule`;
- `net_moment` and `zero_net_moment_tol`;
- `fm_like_by_spin_point_group`;
- `spin_splitting_without_soc`;
- the boolean altermagnet/SOM evidence.

A net moment near the `mtol`-derived threshold is intrinsically sensitive.

## The Output Is Too Large

Use the default quick summary or select fields:

```bash
fsg structure.mcif
fsg structure.mcif --show properties
fsg --full structure.mcif --show operation-views
```

`--details` intentionally expands the quick summary. `--full` requires at
least one `--show FIELD`; this prevents a huge raw compatibility payload from
burying the result you actually need.

In Python, start with `find_spin_group_basic(...)`. After full analysis, use
`to_summary_dict()` or `to_structured_dict()` rather than printing
`to_dict()`.

## A `--show` Field Is Unknown Or Unavailable

The command exits nonzero so scripts cannot silently accept `None`.

- Check spelling and case in the [CLI Field Guide](../reference/cli-show-fields.md).
- Add `--full` for a full-only field such as `operation_views` or generated
  artifacts.
- Do not mix BasicResult names with `SummaryResult` or raw full-result names.

## POSCAR Results Differ Between CLI And Python

The CLI allows and prefers sibling `INCAR` `MAGMOM`. Python file-path calls use
embedded POSCAR moments unless explicitly opted into INCAR reading.

To reproduce CLI behavior in Python:

```python
find_spin_group_basic(
    "POSCAR",
    poscar_allow_incar_magmom=True,
    poscar_prefer_incar_magmom=True,
)
```

## Input-Cell And Primitive OSSG Labels Differ

This can be expected when the supplied cell is not magnetic primitive. For the
input-cell export route, read:

```text
summary.is_input_magnetic_primitive
summary.input_ssg_may_be_incomplete
summary.warning
summary.input_ssg_index
summary.primitive_ssg_index
```

The input-cell operation set can be an incomplete subgroup. Use the primitive
label as the complete reference and the input-cell operations only for a
downstream setting requirement.

## Which Cell Or Operation Setting Should I Use?

- Convention setting: public OSSG symbol/presentation.
- Input setting: only when a downstream tool expects the supplied cell.
- ACC primitive: matched POSCAR/KPOINTS and common downstream reciprocal-space
  workflows.
- Database standard: identify-index validation and exact database-setting
  export.

Never combine coordinates from one setting with operations from another.

## Quasi-2D Results Are Unexpected

Confirm that `vacuum_axis` names the normal axis in the **input cell**:

```bash
fsg --full slab.mcif \
  --calculation-mode quasi2d \
  --vacuum-axis c \
  --show quasi_2d
```

The quasi-2D workflow may regularize/extend insufficient vacuum before its
interpretation path. Inspect plane, axis, and cell diagnostics rather than
assuming it only edits the 3D output label.

## A Parser Expansion Reports Inconsistent Moments

This usually means symmetry expansion produced the same site with incompatible
moments beyond `parser_atol`.

1. Check the SCIF operation/spin-lattice data and moment precision.
2. Regenerate the file from a trusted source when possible.
3. Adjust `parser_atol` only if the discrepancy is known rounding noise.
4. Do not change `space_tol`, `meigtol`, and `matrix_tol` as a substitute for a
   parser-specific problem.

## Generated Files Would Overwrite Existing Files

Current artifact writers replace their target files. Use a fresh output path or
move the existing files before running:

```bash
fsg structure.mcif --write-scif new/output.scif
fsg structure.mcif --write-poscar-kpoints new/calc_inputs
```
