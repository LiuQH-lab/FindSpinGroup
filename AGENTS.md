## FindSpinGroup Local Guardrails

- Do not add silent fallback candidates in the G0std/no-fraction or
  identify-index ACC-P standard-cell selection path. If the selected
  standard-cell transform cannot carry the paired `(cell, SSG)` through the
  required transform chain, raise with diagnostics instead of substituting a
  different basis.
- In particular, do not use `ssg_primitive.acc_primitive_trans` or the legacy
  ACC primitive transform to synthesize a replacement G0std candidate after the
  current integerized / identify-derived transform fails. That hides the
  standard-cell matrix bug.
- Keep cell and SSG transforms paired. The transform used to move the SSG must
  be the same setting transform used to move the cell, and failures in that
  paired transform are part of the diagnostic signal.
- Missing identify-index database records should remain visible errors unless
  the user explicitly asks for a temporary diagnostic shim.
