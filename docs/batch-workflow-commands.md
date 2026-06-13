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
- SCIF roundtrip currently has one tracked non-fractional compact-output
  outlier: `0.37_U3Al2Si3`.  Its source structure still identifies as
  `79.5.1.2.P2`, but compact SCIF writes representative moments plus
  operations; re-expansion chooses a slightly different moment field and
  re-identifies as `5.5.1.1.P`.  This is tagged for a future
  structure-preserving SCIF mode.
- `2.21_TbOOH` is kept as a regression sample for oriented spin-frame identity
  formatting; it should no longer be counted as an accepted SCIF outlier after
  the current formatter fix.
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
- "check whether the 2D wave-config plane-preserving guard fires"

Command:

```bash
python scripts/fsg_workflow.py quasi2d-test \
  --profile fast-full-node \
  --tag <short-change-name> \
  --execute
```

For an explicit physical vacuum axis:

```bash
python scripts/fsg_workflow.py quasi2d-test \
  --profile fast-full-node \
  --vacuum-axis c \
  --tag <short-change-name> \
  --execute
```

For a quick diagnostic sample:

```bash
python scripts/fsg_workflow.py quasi2d-test \
  --profile fast-full-node \
  --vacuum-axis c \
  --limit 300 \
  --tag <short-change-name> \
  --execute
```

For a guard-only three-axis sweep:

```bash
python scripts/fsg_workflow.py quasi2d-test \
  --profile fast-full-node \
  --axis-sweep \
  --limit 300 \
  --tag <short-change-name> \
  --execute
```

`--axis-sweep` submits three independent explicit-axis runs for `a`, `b`, and
`c`.  Each axis gets its own output-root suffix and baseline suite.  Use it to
stress-test axis handling and the quasi-2D wave-config plane-preserving audit;
do not interpret forced `a` or `b` runs as physical unless those axes are the
actual vacuum axes for the inputs.

Default intent:

- run the full route with `calculation_mode=quasi2d`
- use the configured quasi-2D inputversion dataset
- export a compact `selected.txt` containing index, phase, quasi-2D status,
  vacuum axis, magnetic phase, projected spin splitting, interpretation,
  generic-point comparison fields, and 2D wave-config plane audit fields
- skip business Excel export during compute

Acceptance:

- processed count matches the quasi-2D dataset
- runtime errors are classified by known database/data issues
- previous tracked 2D hard cases stay resolved
- `quasi_2d.generic_point_comparison` and `spin_splitting_changed` counts are
  summarized before making physics conclusions
- for wave-config guard checks, successful records should have
  `quasi_2d.spin_texture_config_no_soc.operation_audit.non_plane_preserving_operation_count = 0`
  and the same SOC count equal to zero; forced wrong-axis errors should be
  classified separately from this audit

Operational notes:

- The configured quasi-2D dataset path must be visible from the target compute
  node, not only from the login node.  Full-node profiles may use a different
  filesystem from the standard cluster route.
- Keep the inputversion dataset cleaned of macOS `._*` resource files.  The
  batch resolver ignores those files, but a clean target dataset makes counts
  and diagnostics easier to read.
- If a target profile uses a shared virtual environment, verify the interpreter
  path on the compute node.  A symlinked Python that works on the login node can
  be broken on a different compute partition.

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

## Operational Checks

Before the first cluster-backed run in a new local checkout, verify the local
profile and remote runtime instead of rebuilding ad-hoc Slurm commands:

```bash
python scripts/fsg_workflow.py batch-test --profile fast-full-node --tag dry-run
```

The dry run should show a snapshot build, remote snapshot preparation, the
shared runtime symlinks, the selected dataset, the intended baseline suite, the
worker count, and the Slurm arguments from the profile.  If the profile file is
missing, create the gitignored `.fsg-batch-profiles.local.toml` from
`fsg-batch-profiles.example.toml`; do not hard-code private paths into tracked
docs or scripts.

Common failure modes and fixes:

- Missing Python packages usually means the Slurm job used system Python instead
  of the profile interpreter.  Fix the profile `python_bin`; do not tune FSG
  tolerances.
- A Python path that works on the login node may not be visible on every compute
  node.  Keep the profile tied to a verified runtime and node/partition
  combination.
- If `comparison.json` is missing or `comparison` is `null`, check whether the
  snapshot `batch_baselines` symlink points to the accepted baseline directory.
  Otherwise the first run may create a new baseline instead of comparing.
- Use the snapshot `PYTHONPATH` check to confirm the run is importing the
  current snapshot source.  The shared virtual environment's installed
  `findspingroup` version can be older and is not by itself evidence that the
  batch used stale code.
- For change validation, keep runtime Excel export off during compute.  Generate
  business Excel files later from an existing `full_results.jsonl`.

When the user asks for one of the standard scenarios, map it to the workflow
command instead of hand-writing `sbatch`:

- "batch test this change" -> `batch-test` on the 2241 full route
- "pre-push release batch test" -> `release-test`
- "generate an Excel/table" -> `export` from an existing full run
- "test 2D cases" -> `quasi2d-test`
- "test a fresh install / CLI" -> `install-smoke`
