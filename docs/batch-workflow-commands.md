# Batch Workflow Commands

This document defines the FindSpinGroup batch commands used for the current
project workflow.  The commands are profile-driven so private SSH aliases,
cluster paths, partitions, QOS names, and interpreter paths stay in the local
gitignored profile file, not in git history.

Copy `fsg-batch-profiles.example.toml` to `.fsg-batch-profiles.local.toml` and
fill in local execution details before using cluster-backed commands.

## Privacy Contract

Use `profile` names in commands and reports.  Do not put private target names,
absolute cluster paths, local home paths, chat-app temporary paths, or snapshot
paths in README, Linear, PR descriptions, exported business Excel files, or
committed docs.

Internal run directories may contain raw `run_config.json`, `summary.json`, and
runtime logs with absolute paths for debugging.  Public or handoff artifacts
should use only dataset id, run tag, route, version, commit, tolerances, case id,
file name, and business fields.

Before committing workflow-related files, run:

```bash
python scripts/check_privacy_leaks.py --staged
```

## 1. Change Validation

Use this when the request is equivalent to:

- "batch test this change"
- "run the 2241 full validation"
- "check whether this code change drifts results"

Command:

```bash
python scripts/fsg_workflow.py batch-test \
  --profile fast-full-node \
  --dataset mcif_260414_no2241 \
  --route full \
  --calculation-mode 3d \
  --tag <short-change-name> \
  --execute
```

Default intent:

- build a source snapshot from the current worktree
- submit the 2241 full route
- use the profile worker count unless `--workers` is provided
- skip business Excel export during compute
- inspect `summary.json`, `errors_by_file.json`, `records.jsonl`, and
  `full_results.jsonl` after completion

Acceptance:

- processed count matches the dataset
- errors are only known dataset issues
- stable core fields do not drift against the accepted baseline

## 2. Release Validation

Use this when the request is equivalent to:

- "pre-push release batch test"
- "release validation before merge/tag"
- "run the publication/regression suite"

Command:

```bash
python scripts/fsg_workflow.py release-test \
  --profile fast-full-node \
  --tag <version-or-rc-name> \
  --execute
```

Default intent:

- run local unit/smoke tests
- submit 2241 basic route
- submit 2241 full route
- submit 2241 POSCAR ACC primitive roundtrip
- submit 2241 SCIF roundtrip
- use the same profile worker count for route and roundtrip stages unless
  `--workers` or `--roundtrip-workers` overrides it

Keep the 2241 basic and full routes on separate baseline suites.  The compact
basic payload and full runtime payload are intentionally different comparison
contracts.

Do not push, tag, or move a release version until all stages are clean.

Current acceptance notes for the 2241 release suite:

- basic and full route runs are expected to process all 2241 inputs; the current
  known dataset errors are the two malformed DyGa3 records
- POSCAR roundtrip can still show the known fractional-occupancy POSCAR-loss
  mismatches for Sr2Ir/Sn and YBa/Fe/O mixed-occupancy cases
- SCIF roundtrip currently has two tracked non-fractional roundtrip outliers:
  `0.37_U3Al2Si3` and `2.21_TbOOH`
- roundtrip jobs should be submitted with parallel workers for release testing;
  serial roundtrip jobs are no longer the intended pre-push path
- local unit/smoke tests are still part of the release gate; clean cluster
  batches alone are not enough to push

## 3. Business Export

Use this when the request is equivalent to:

- "generate an Excel with these fields"
- "make the magnetic-site table"
- "export ferroelectric switching operations"
- "make a chirality/polar/tensor screening table"

Command:

```bash
python scripts/fsg_workflow.py export \
  --run-dir <existing-full-run-dir> \
  --preset standard-excel \
  --output artifacts/<name>.xlsx \
  --execute
```

Available presets:

```bash
python scripts/fsg_workflow.py export --run-dir <run-dir> --preset standard-excel --execute
python scripts/fsg_workflow.py export --run-dir <run-dir> --preset magnetic-site --execute
python scripts/fsg_workflow.py export --run-dir <run-dir> --preset ferroelectric-switching --execute
python scripts/fsg_workflow.py export --run-dir <run-dir> --preset symmetry-property --execute
```

This command reads existing `full_results.jsonl`.  It should not recompute
FindSpinGroup.  If a requested field is missing from the runtime result, rerun a
full batch with that feature enabled.

## 4. Quasi-2D Validation

Use this when the request is equivalent to:

- "test the 2D cases"
- "run the quasi-2D inputversion validation"
- "check 2D spin splitting after this change"

Command:

```bash
python scripts/fsg_workflow.py quasi2d-test \
  --profile fast-full-node \
  --tag <short-change-name> \
  --execute
```

For an explicit vacuum axis:

```bash
python scripts/fsg_workflow.py quasi2d-test \
  --profile fast-full-node \
  --vacuum-axis c \
  --tag <short-change-name> \
  --execute
```

Default intent:

- run the full route with `calculation_mode=quasi2d`
- use the configured quasi-2D inputversion dataset
- export a compact `selected.txt` containing index, phase, quasi-2D status,
  vacuum axis, magnetic phase, projected spin splitting, interpretation, and
  generic-point comparison fields
- skip business Excel export during compute

Acceptance:

- processed count matches the quasi-2D dataset
- runtime errors are classified by known database/data issues
- previous tracked 2D hard cases stay resolved
- `quasi_2d.generic_point_comparison` and `spin_splitting_changed` counts are
  summarized before making physics conclusions

## 5. User Install Smoke

Use this when the request is equivalent to:

- "test from a fresh Python environment"
- "can users pip install and run the CLI?"
- "check CLI entry points from scratch"

Command:

```bash
python scripts/fsg_workflow.py install-smoke \
  --python python3 \
  --package findspingroup \
  --execute
```

For a local wheel or source tree:

```bash
python scripts/fsg_workflow.py install-smoke \
  --python python3 \
  --package /path/to/findspingroup-<version>-py3-none-any.whl \
  --execute
```

Smoke checks:

- create a fresh virtual environment
- install the package
- import `findspingroup`
- print the package version
- run `fsg --help`
- run `findspingroup --help`

## Dry Run

All workflow commands are dry-run by default.  Omit `--execute` when you only
want to review the planned commands:

```bash
python scripts/fsg_workflow.py batch-test --profile fast-full-node
```
