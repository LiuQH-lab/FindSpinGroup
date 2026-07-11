# Reference

The reference documents exact public surfaces: Python functions, returned
objects, CLI flags, output fields, and SCIF export modes. It is deliberately
separate from the learning path so raw/legacy/diagnostic fields do not obscure
the main scientific conclusions.

Use the guide first if you are still deciding what to run. Use the reference
when you know the entry point and need precise parameter or output meaning.

For physical interpretation rather than schema lookup, use
[Interpret Your Result](../guide/understanding-results.md).

## Reference Pages

- [Python API](python-api.md) documents the main functions. Each main function
  has its own page with signature, parameters, returns, returned fields,
  examples, and notes.
- [MagSymmetryResult Object](magsymmetry-result.md) documents the full result
  object and recommended accessors.
- [Result Schemas](result-schemas/index.md) documents returned dictionaries by
  schema: `BasicResult`, `SummaryResult`, `StructuredResult`,
  `InputSSGResult`, and batch exports.
- [CLI](cli.md) documents command-line routes and flags.
- [SCIF Export](scif.md) documents `to_scif(...)` and SCIF output modes.

## Public, Diagnostic, And Raw Output

Documented summary fields and structured fields are the intended public output
surface for new users and integrations.

Diagnostic fields explain route decisions and are useful for debugging and
validation.

`to_dict()` exposes the raw compatibility surface. It may contain legacy names
and internal details, so new integrations should prefer documented summary or
structured outputs.
