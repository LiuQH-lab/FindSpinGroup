# Result Schemas

Result schema pages document returned dictionaries by object shape. This keeps
the API reference focused on function calls while giving every output field a
clear home.

## Which Schema Do I Need?

`BasicResult`
Returned by `find_spin_group_basic(...)` and the default `fsg file.mcif` route.
Use it for compact identification, screening, and simple integrations.

`SummaryResult`
Returned by `MagSymmetryResult.to_summary_dict()`. Use it when you ran the full
route but only need a display-friendly summary.

`StructuredResult`
Returned by `MagSymmetryResult.to_structured_dict()`. Use it for full
programmatic integrations that need groups, cells, transforms, properties, and
artifacts.

`InputSSGResult`
Returned by `find_spin_group_input_ssg(...)` and written by `fsg -w`. Use it
when a downstream tool needs input-cell SSG or MSG operations.

`Payload Definitions`
Reusable nested objects such as `SSGPayload`, `SSGOperation`, `MSGPayload`,
`MSGOperation`, `CellPayload`, and `TransformPayload`.

`BatchExport`
Export columns written by batch routes.

## Schema Style

Each schema page starts with a compact shape:

```python
{
    "field": type,
    "nested": {
        "field": type,
    },
}
```

Then it explains fields in prose blocks. Deep diagnostic payloads are described
by purpose and important keys instead of duplicating every internal audit field
in the first screen.
