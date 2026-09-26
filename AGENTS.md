## FindSpinGroup Local Guardrails

For tasks that use or interpret FindSpinGroup rather than modify its internal
algorithms, read and follow [`FSG_AGENT_GUIDE.md`](FSG_AGENT_GUIDE.md) before
choosing a CLI route, Python API, output interpretation, or setting transform.
The guide is an AI execution protocol; the guardrails below remain authoritative
for code changes.

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

## Semantic And Mathematical Review

- Before changing an algorithm or shared numerical helper, identify each affected
  caller's purpose, mathematical object, coordinate/spin frame, translation
  equivalence, tolerance units, and downstream consumers. Similar code or names
  do not establish identical semantics.
- Derive the invariants and examine counterexamples before implementation. Tests
  are necessary evidence, not a substitute for this review: existing coverage
  may not protect symbols, generator selection, frame transport, or exports.
- Keep physical input equivalence, accepted-group representation, and display
  simplification separate. Never apply modulo-one reduction to lifted G0
  translations without the correct spin-identity translation lattice.
- Validate changes one semantic layer at a time. Explain every observed drift
  with concrete before/after evidence; do not accept a baseline just because
  calculation succeeded or the group index stayed unchanged.

## Collaboration and GitHub Workflow

- Do not merge pull requests automatically unless the user explicitly authorizes
  the merge in the current turn. By default, push the branch, open or prepare
  the PR page, share the link/content, and stop for the user to review and
  confirm the merge manually.
- For GitHub issues opened by others, prepare the technical reply and evidence,
  but do not close the issue automatically. The user prefers to reply personally
  first and close the issue only after that reply unless they explicitly
  delegate the reply/close action.
