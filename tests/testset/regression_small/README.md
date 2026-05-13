# Small Regression Set

This directory contains small local regression inputs for route and output
contract work. It is intentionally not a replacement for the 2185-case batch.

## 3D Core Manifest

`3d_core_manifest.txt` is used by `scripts/run_small_regression.py` for both
basic and full 3D route checks.

Expected behavior:

- all listed cases should run except the known identify-index database gap
  `1.669_KFe(PO3F)2`
- output summaries should preserve stable public fields such as `index`,
  `phase`, spin splitting flags, AHC flags, and classification markers

## Quasi-2D Small Set

The quasi-2D suite reuses `tests/testset/quasi2d_small`, which contains
inputversion-derived 2D materials and the `V2Se2O` case study.
