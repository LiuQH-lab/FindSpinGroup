# CLI Field Guide

`--show FIELD` reads a field from the payload produced by the selected analysis
level. Dot paths read nested dictionaries.

```bash
fsg structure.mcif --show magnetic_phase
fsg structure.mcif --show properties.ss_wo_soc
fsg --full structure.mcif --show msg_bns_number
```

Use `--json --show FIELD` for machine-readable output. An unknown/unavailable
field exits nonzero; if the field requires full analysis, add `--full`.

This page intentionally lists the high-value public fields rather than every
raw compatibility/debug key. Complete schemas are available under
[Result Schemas](result-schemas/index.md).

## Quick-Analysis Fields

Available from:

```bash
fsg structure.mcif --show FIELD
```

### Identification and magnetic order

| Field | Question answered | Type / setting |
| --- | --- | --- |
| `index` | Which OSSG was identified? | String identifier |
| `ossg_symbol_linear` | What is its public oriented symbol? | Convention-setting string |
| `msg_bns_number` | Which SOC-compatible MSG? | BNS number as string |
| `msg_symbol` | What is the MSG BNS symbol? | String |
| `msg_type` | Which MSG type? | Integer |
| `conf` | Collinear, coplanar, or noncoplanar moments? | String |
| `magnetic_phase` | How does the classifier label the supplied order? | String |
| `net_moment` | Magnitude of the magnetic-cell moment sum? | Scalar, μB |
| `zero_net_moment_tol` | Threshold used for a zero-net-moment decision? | Scalar, μB |
| `magnetic_phase_details` | Why was the phase label chosen? | Dictionary with classifier evidence/booleans |

Useful nested classifier booleans:

```text
magnetic_phase_details.is_altermagnet
magnetic_phase_details.is_spin_orbit_magnet
magnetic_phase_details.zero_net_moment
magnetic_phase_details.classification_rule
```

Use these booleans in scripts instead of interpreting display strings such as
`is_alter` or `is_som`.

### Symmetry-allowed physical responses

| Field | Meaning |
| --- | --- |
| `properties` | Compact spin-splitting, AHC, and phase-tag block |
| `properties.ss_wo_soc` | `No`, `k-dependent`, or `Zeeman` no-SOC splitting class |
| `properties.ss_w_soc` | `Yes`/`No` SOC splitting permission |
| `properties.ahc_wo_soc` | No-SOC AHC symmetry permission |
| `properties.ahc_w_soc` | SOC AHC symmetry permission |
| `spin_texture_config_no_soc` | Runtime OSSG spin-texture constraint |
| `spin_texture_config_soc` | Runtime MSG-compatible spin-texture constraint |
| `spin_texture_config_database` | Database reference for the identified SSG label |
| `vector_constraints_by_symmetry` | Allowed vector subspaces under SG, OSSG, and MSG |
| `ferroelectric_switching` | Symmetry-only domain/switching relations |

Spin-texture aliases:

```text
spin-texture-no-soc  -> spin_texture_config_no_soc
spin-texture-soc     -> spin_texture_config_soc
```

Common nested spin-texture fields:

```text
spin_texture_config_no_soc.spin_texture_type
spin_texture_config_no_soc.order
spin_texture_config_no_soc.nullity
spin_texture_config_no_soc.spin_rank
spin_texture_config_no_soc.momentum_space_spin_configuration
spin_texture_config_no_soc.basis
spin_texture_config_no_soc.basis_vectors
spin_texture_config_no_soc.basis_setting
```

### Symmetry labels needed for exact identity

These are valuable for database/reproduction work but are not usually the
first physical summary:

```text
g0_number
g0_symbol
l0_number
l0_symbol
it
ik
sspg
nsspg
spin_space_point_group_hm
spin_space_point_group_schoenflies
nontrivial_spin_space_point_group_hm
nontrivial_spin_space_point_group_schoenflies
acc_symbol
empg
space_group_number
space_group_symbol
```

### Polar, chiral, and numerical context

```text
sg_is_polar
sg_is_chiral
ssg_is_polar
ssg_is_chiral
msg_is_polar
msg_is_chiral
tolerances
tolerances.space_tol
tolerances.mtol
tolerances.meigtol
tolerances.matrix_tol
```

The quick payload does not serialize `parser_atol` inside `tolerances`; record a
non-default value from the command/call configuration separately.

## Full-Analysis Fields

Available from:

```bash
fsg --full structure.mcif --show FIELD
```

The full CLI currently exposes the raw compatibility surface. Prefer aliases
and well-documented direct products rather than discovering arbitrary fields.

### Main direct fields

```text
index
conf
magnetic_phase
msg_bns_number
msg_symbol
msg_type
spin_texture_config_no_soc
spin_texture_config_soc
vector_constraints_by_symmetry
magnetic_site_summary
operation_views
```

`quasi_2d` is populated only when quasi-2D analysis is explicitly requested:

```bash
fsg --full slab.mcif --calculation-mode quasi2d --vacuum-axis c --show quasi_2d
```

### Operation aliases

| Alias | Resolves to | Output |
| --- | --- | --- |
| `operation-views`, `ops` | `operation_views` | Summary of settings and named views |
| `gspg` | `gspg_text` | Compact GSPG text |
| `wp-chain`, `wyckoff-chain` | `wp_chain` | Wyckoff-chain rows |

One operation view can be selected by a full dot path:

```bash
fsg --full structure.mcif \
  --show operation_views.convention_oriented.views.msg
```

Every operation payload must be interpreted with its real-space `setting` and
`spin_frame_setting`.

### Artifact aliases

| Alias | Resolves to | Setting |
| --- | --- | --- |
| `kpoints`, `kpoints_text` | `KPOINTS` | ACC primitive reciprocal path paired with generated POSCAR |
| `poscar`, `primitive_poscar` | `acc_primitive_magnetic_cell_poscar` | ACC primitive magnetic cell |
| `scif_default`, `default_scif` | `scif` | Default SCIF mode |

For writing files, prefer the explicit artifact flags over shell redirection:

```bash
fsg structure.mcif --write-poscar-kpoints calc_inputs
fsg structure.mcif --write-scif output.scif
```

## Input-Cell Operation Export

`fsg -w FILE` writes a purpose-built JSON bundle and does not combine with
`--show`. Read the written/output summary first, especially:

```text
summary.is_input_magnetic_primitive
summary.input_ssg_may_be_incomplete
summary.warning
summary.input_ssg_index
summary.primitive_ssg_index
```

Then consume `ssg.ops` or `msg.ops` only with their accompanying settings.

## Interpretation Rules

- `Yes`/`allowed` means symmetry does not force a response to vanish; it does
  not predict a finite magnitude.
- `No`/`forbidden` means the response vanishes under the analyzed symmetry/model.
- Moment geometry (`conf`) is not the same as FM/AFM phase classification.
- A vector, operation, or basis expression without its setting/frame is
  incomplete information.
- Diagnostic/audit fields explain route decisions; they should not displace the
  OSSG/MSG/physics summary in a user interface.
