# Advanced Notes

Advanced notes collect terminology and diagnostics that are important for
expert users, maintainers, and validation work, but should not be required for
the first successful run.

## Topics That Belong Here

SSG, OSSG, and MSG concepts
Conceptual relationship between nonrelativistic spin-space symmetry and
magnetic-space-group descriptions.

Cell settings
The difference between input, input magnetic primitive, database standard,
convention, ACC primitive, and ACC conventional settings.

Transform diagnostics
How transforms connect cells and operation payloads, and how route audits should
be read when a case fails.

Tolerance interpretation
How `space_tol`, `mtol`, `meigtol`, `matrix_tol`, and `parser_atol` affect
matching and classification.

SCIF generator details
Repo-local tags, symbolic numeric formatting, roundtrip behavior, and quasi-2D
metadata.

Batch validation
Regression suites, export columns, baseline comparison, and high-throughput
screening workflows.

## Placement Rule

If a page is needed to run the first example, choose an API, or read the main
result fields, it belongs in the guide.

If a page defines exact function signatures, returned fields, CLI flags, or
SCIF modes, it belongs in the reference.

If a page explains route internals, database settings, transform audits, or
development validation, it belongs here.
