# Reference

The reference documents the exact public surfaces: Python functions, returned
objects, CLI flags, output fields, and SCIF export modes.

Use the guide first if you are still deciding what to run. Use the reference
when you know the entry point and need precise parameter or output meaning.

## Reference Pages

- [Python API](python-api.md) documents the main functions.
- [MagSymmetryResult](magsymmetry-result.md) documents the full result object
  and recommended accessors.
- [CLI](cli.md) documents command-line routes and flags.
- [Output Fields](output-fields.md) documents the public fields by function and
  result layer.
- [SCIF Export](scif.md) documents `to_scif(...)` and SCIF output modes.

## Public, Diagnostic, And Raw Output

Documented summary fields and structured fields are the intended public output
surface for new users and integrations.

Diagnostic fields explain route decisions and are useful for debugging and
validation.

`to_dict()` exposes the raw compatibility surface. It may contain legacy names
and internal details, so new integrations should prefer documented summary or
structured outputs.
