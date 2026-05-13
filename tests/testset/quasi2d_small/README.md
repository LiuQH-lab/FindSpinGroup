# Quasi-2D Small Case Set

Small PER-41 validation set for additive quasi-2D diagnostics. The files are
copied from the recovered 2D database output version, plus the V2Se2O case
study. Keep original long file names because they encode the source database
labels (`ep`, `mp`, `sp`, `ahc`) used during manual inspection.

Coverage:

- `V2Se2O`: current case study for the final ACC primitive transform chain.
- `1VBr2-1`: positive/negative/MP contrast in a simple halide family.
- `1VSe2-1`: TMD-like V material with `spY`.
- `2CrI3-1`: common layered magnetic material family; a heuristic
  counterexample that can be forced through runtime parameters.
- `2MnPS3-1`: MPX3 antiferromagnetic family.
- `1MnBi2Te4-1`: layered AFM topological-material family.
- `1AgVP2S6-1`: symbolic-coordinate/parser regression coverage.
- `1CoH2O2-1`: old non-integer transform stress coverage.
- `1Cr3Te4-1`: old point-group ambiguity/stress coverage.

Intended use:

```bash
PYTHONPATH=src ./.venv/bin/python -m findspingroup.batch_mcif \
  tests/testset/quasi2d_small \
  --output-dir /tmp/fsg_quasi2d_small \
  --calculation-mode quasi2d \
  --vacuum-axis c \
  --export-txt selected_2d.txt \
  --export-field index \
  --export-field phase \
  --export-field properties.ss_wo_soc \
  --export-field quasi_2d.dimension \
  --export-field quasi_2d.status \
  --export-field quasi_2d.source \
  --export-field quasi_2d.vacuum_axis_input \
  --export-field quasi_2d.spin_splitting_2d \
  --export-field quasi_2d.interpretation \
  --export-field quasi_2d.is_alter_2d \
  --export-field quasi_2d.kpoint_projection_summary \
  --export-field quasi_2d.generic_point_comparison.summary \
  --export-field quasi_2d.generic_point_comparison.spin_splitting_changed
```

Manual calculation-mode control is runtime-only:

```bash
PYTHONPATH=src ./.venv/bin/python -m findspingroup.batch_mcif \
  tests/testset/quasi2d_small/2CrI3-1 \
  --output-dir /tmp/fsg_quasi2d_cri3_forced \
  --calculation-mode quasi2d \
  --vacuum-axis c
```

Use `quasi2d` to force the 2D interpretation. Use `3d`/`bulk` to suppress the
additive 2D diagnostics and keep the public result in ordinary 3D mode.
