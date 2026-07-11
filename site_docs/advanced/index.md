# Advanced Diagnostics And Validation

Use this section after the user-facing OSSG/MSG/physics result is understood.
Advanced fields explain **how** the route reached that result and how data move
between settings; they are not additional material properties.

## When Advanced Output Is Appropriate

- an OSSG label changes under small tolerance variations;
- an input, convention, database-standard, or ACC operation list disagrees;
- a transformed cell cannot be paired with its SSG operations;
- an input-cell operation export warns that the cell is not magnetic primitive;
- SCIF/POSCAR round trips change the identification;
- a batch regression differs from an accepted baseline.

For ordinary interpretation, stay with
[Interpret Your Result](../guide/understanding-results.md).

## Settings At A Glance

| Setting | Main purpose | Typical user product |
| --- | --- | --- |
| `input` | Preserve the user-supplied cell | Interoperability with an external code |
| `input_magnetic_primitive` | Remove magnetic-cell supercell redundancy | Primitive reference for input-cell warnings |
| `database_standard` | Match identify-index database conventions | Reproduction and route diagnostics |
| `convention` | Public OSSG/GSPG presentation | Symbols and human-facing operations |
| `acc_primitive` | ACC-aligned magnetic primitive cell | Matched VASP POSCAR/KPOINTS |
| `acc_conventional` | Conventional ACC presentation | Comparison/classification workflows |

A valid transformation chain must keep the cell and SSG paired. If a selected
setting transform cannot carry both through the required chain, that failure is
diagnostic evidence; silently substituting an unrelated basis would hide the
problem.

## Diagnostic Payloads

**`identify_index_details`.** Exact identification components, selected
database records, and route evidence.

**`acc_primitive_resolution_audit`.** How the ACC primitive transform/setting
was resolved.

**`transforms.audit` in `StructuredResult`.** Audits for standard-to-ACC and
convention transform chains.

**`operation_views`.** Named complete/generator/nontrivial/MSG operation views.
Each view carries a real-space and spin-frame setting.

**`tolerances`.** Numerical thresholds that belong to the result provenance.

## Validation Order

When a result is surprising:

1. verify input moments, occupancies, and coordinate conventions;
2. reproduce quick analysis with defaults;
3. vary one justified tolerance at a time;
4. compare `index`, MSG, `conf`, and phase evidence;
5. only then inspect identify-index, transforms, and operation settings;
6. validate generated files by parsing/rerunning them through the intended
   public route;
7. use the batch workflow for broad regression claims.

See [Parameters And Reliability](../guide/reliability-and-tolerances.md) before
interpreting a tolerance-dependent change.

## Batch Regression Evidence

The repository batch workflow records baseline, comparison, and error
artifacts. A clean single example is not evidence that a transform/output change
is safe across SSG types. Changes to standard-cell selection, operation
transforms, generated SCIF/POSCAR, or public schemas should use the relevant
focused tests and the agreed batch suite.

Operational commands are maintained in the repository batch workflow guide.
Accepted baselines should record code version and tolerance profile; do not
silently overwrite an accepted successful case with a different result.

## Scientific Boundary

Route diagnostics can establish reproducibility and internal consistency. They
cannot establish energetic stability, response magnitude, experimental domain
population, or transition kinetics. Those require the appropriate electronic-
structure, thermodynamic, or experimental analysis.
