# Ferroelectric Switching Small 3D Set

This directory contains a compact 3D review manifest for the
`ferroelectric_switching` payload. It is selected from the standard 2185-case
MCIF batch and covers:

- nonpolar structural parent to polar ordered spin-space symmetry
- polar structural parent to polar ordered symmetry where axis transport must
  be validated before claiming an axis change
- polar parent axis-preserved controls
- nonpolar ordered-symmetry controls

Regenerate the manifest and report from a batch `records.jsonl` file with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/select_ferroelectric_switching_cases.py \
  --records output/mcif_241130_no2186_run/run_v0.13.16_20260413_221533/records.jsonl
```
