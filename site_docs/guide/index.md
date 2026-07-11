# User Guide

The guide follows a researcher's workflow:

1. run one magnetic structure;
2. identify which symmetry description answers the question;
3. interpret the physics-facing outputs and their limitations;
4. test numerical reliability;
5. request operations or files only when needed.

## Recommended Path

| Step | Page | Outcome |
| ---: | --- | --- |
| 1 | [Quickstart](getting-started.md) | Run the CLI or Python API and understand the first six result lines. |
| 2 | [Interpret Your Result](understanding-results.md) | Distinguish OSSG, MSG, moment geometry, phase, spin splitting, AHC, and spin texture. |
| 3 | [Choose A Workflow](choosing-an-api.md) | Select quick analysis, full analysis, or input-cell operation export. |
| 4 | [Parameters And Reliability](reliability-and-tolerances.md) | Decide whether a tolerance change is justified and test sensitivity. |
| 5 | [Input Formats](input-formats.md) | Verify magnetic moments, coordinate conventions, and POSCAR/INCAR behavior. |
| 6 | [Examples](examples.md) | Adapt complete workflows to a research task. |

Use [Troubleshooting](troubleshooting.md) when a run fails or produces an
unexpected result.

## Where Detailed Information Lives

- Exact commands and exports: [CLI Reference](../reference/cli.md)
- Function signatures and returns: [Python API](../reference/python-api.md)
- Complete dictionary and nested field shapes:
  [Result Schemas](../reference/result-schemas/index.md)
- Terminology and settings glossary: [Concepts](concepts.md)
- SCIF modes and setting contracts: [SCIF Export](../reference/scif.md)

The complete field inventory is intentionally kept out of the learning path so
that diagnostic and legacy details do not obscure the main scientific result.
