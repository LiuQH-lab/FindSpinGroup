# FindSpinGroup Documentation

`findspingroup` is a Python package, command-line program, and
[web application](https://app.findspingroup.com) for identifying oriented spin
space group (OSSG) symmetry in magnetic crystal structures.

Given a structure with magnetic moments, FindSpinGroup identifies the OSSG,
reports the corresponding magnetic space group (MSG), classifies the magnetic
phase, and can generate symmetry data for downstream first-principles,
high-throughput, and web-application workflows.

## Where Things Belong

This documentation is split into three layers.

**Guide** explains how to complete user tasks. Start here if you are not sure
which command or Python function to use.

**Reference** documents the exact Python functions, return objects, CLI flags,
and output fields. Use it when you know the entry point and need precise
parameter or output meaning.

**Advanced Notes** explain internal settings, transform terminology, and
diagnostic routes that are useful for expert users and maintainers.

The repository `README.md` is intentionally shorter than this site. It is only
the project entry point: what the package does, how to install it, and where to
continue.

## Start Here

Use the quick path if you only need the identified group labels and magnetic
phase:

```python
from findspingroup import example_path, find_spin_group_basic

summary = find_spin_group_basic(example_path("0.800_MnTe.mcif"))

print(summary["index"])
print(summary["magnetic_phase"])
print(summary["msg_bns_number"], summary["msg_symbol"])
```

Expected output:

```text
194.164.1.1.L
AFM(Altermagnet)
63.457 Cmcm
```

## Main Routes

Use `find_spin_group_basic(...)` or `fsg file.mcif` for compact identification.

Use `find_spin_group(...)` or `fsg --all file.mcif` for the full result object,
generated SCIF, POSCAR, KPOINTS, tensor outputs, quasi-2D diagnostics, and audit
information.

Use `find_spin_group_input_ssg(...)` or `fsg -w file.mcif` when you need SSG and
MSG operations in the input-cell setting and optional helper files written to
disk.

## Next Steps

- [Getting Started](guide/getting-started.md) - install and run the first case.
- [Choosing an API](guide/choosing-an-api.md) - choose the right function or CLI
  route.
- [Understanding Results](guide/understanding-results.md) - read the OSSG, MSG,
  magnetic phase, settings, and artifacts.
- [Python API Reference](reference/python-api.md) - exact function signatures,
  parameters, returns, and examples.
- [Result Schemas](reference/result-schemas/index.md) - field-by-field output
  meaning by returned object.
