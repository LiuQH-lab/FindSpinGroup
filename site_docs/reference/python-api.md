# Python API Reference

This section follows the style used by projects such as spglib and PyWebIO: each
public function has its own page, and each page is organized by signature,
parameters, returns, returned fields, examples, and notes.

For task-oriented explanations, see [Choosing an API](../guide/choosing-an-api.md).
For schema-first output documentation, see [Result Schemas](result-schemas/index.md).

## Main Functions

`find_spin_group_basic(...)`
Compact identification route. Use it for screening and simple integrations.

Read: [find_spin_group_basic](api/find-spin-group-basic.md)

`find_spin_group(...)`
Full analysis route. Use it when you need generated artifacts, operation
payloads, tensor outputs, quasi-2D diagnostics, or route audits.

Read: [find_spin_group](api/find-spin-group.md)

`find_spin_group_input_ssg(...)`
Input-cell operation route. Use it when another program needs SSG or MSG
operations in the input-cell setting.

Read: [find_spin_group_input_ssg](api/find-spin-group-input-ssg.md)

## Advanced Entry Points

These functions are exposed for specialized integrations and development
workflows. They are not the recommended starting points for new users.

`find_spin_group_acc_primitive(...)`
Specialized route for ACC primitive payloads.

`find_spin_group_poscar_ssg(...)`
Specialized POSCAR-facing SSG route.

`find_spin_group_from_data(...)`
Full route for integrations that already have parsed structure arrays.

`find_spin_group_basic_from_data(...)`
Basic route for integrations that already have parsed structure arrays.

`find_spin_group_acc_primitive_from_data(...)`
ACC primitive route for integrations that already have parsed structure arrays.
