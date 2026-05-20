"""Public output schema constants for FindSpinGroup.

This module intentionally contains no heavy crystallographic imports.  It is a
small contract layer shared by scripts and tests so field names are not copied
by hand in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class OutputLayer:
    """Named result layer exposed by the public result contract."""

    name: str
    preferred_prefix: str
    legacy_prefixes: tuple[str, ...] = ()
    description: str = ""


LAYER_INPUT: Final = OutputLayer(
    name="input",
    preferred_prefix="input",
    description="Parsed input structure in its declared setting.",
)
LAYER_INPUT_MAGNETIC_PRIMITIVE: Final = OutputLayer(
    name="input_magnetic_primitive",
    preferred_prefix="input_magnetic_primitive",
    description="Magnetic primitive reduction of the input structure.",
)
LAYER_DATABASE_STANDARD: Final = OutputLayer(
    name="database_standard",
    preferred_prefix="database_standard",
    legacy_prefixes=("G0std", "L0std", "g0_standard", "l0_standard"),
    description="Selected G0/L0 database-standard setting used by identify-index.",
)
LAYER_CONVENTION: Final = OutputLayer(
    name="convention",
    preferred_prefix="convention",
    description="Public conventional setting used for OSSG/GSPG presentation.",
)
LAYER_ACC_PRIMITIVE: Final = OutputLayer(
    name="acc_primitive",
    preferred_prefix="acc_primitive",
    legacy_prefixes=("magnetic_primitive", "primitive_magnetic_cell"),
    description="ACC magnetic primitive output layer.",
)
LAYER_ACC_CONVENTIONAL: Final = OutputLayer(
    name="acc_conventional",
    preferred_prefix="acc_conventional",
    description="ACC conventional output layer.",
)


PUBLIC_OUTPUT_LAYERS: Final = (
    LAYER_INPUT,
    LAYER_INPUT_MAGNETIC_PRIMITIVE,
    LAYER_DATABASE_STANDARD,
    LAYER_CONVENTION,
    LAYER_ACC_PRIMITIVE,
    LAYER_ACC_CONVENTIONAL,
)


EXPORT_METADATA_COLUMNS: Final = (
    "source_fsg_version",
    "source_run_tag",
    "source_route",
)

EXPORT_CASE_COLUMNS: Final = (
    "case_id",
    "file_name",
    "status",
    "duration_seconds",
    "index",
    "conf",
    "phase",
    "acc",
    "msg_acc",
)

EXPORT_IDENTIFY_COLUMNS: Final = (
    "G0_id",
    "L0_id",
    "t_index",
    "k_index",
)

EXPORT_SPIN_POINT_GROUP_COLUMNS: Final = (
    "nsspg_hm",
    "nsspg_symbol",
    "sspg_hm",
    "sspg_symbol",
    "ssg_type",
    "spin_only_direction",
)

EXPORT_SYMBOL_COLUMNS: Final = (
    "ossg_symbol",
    "primitive_ssg_symbol",
    "sg_symbol",
    "sg_num",
    "sg_is_centrosymmetric",
    "sg_is_polar",
    "sg_is_chiral",
    "ossg_space_group_number",
    "ossg_is_centrosymmetric",
    "ossg_is_polar",
    "ossg_is_chiral",
    "msg_symbol",
    "msg_num",
    "msg_type",
    "msg_bns_number",
    "msg_og_number",
    "msg_parent_space_group_number",
    "msg_is_centrosymmetric",
    "msg_is_polar",
    "msg_is_chiral",
)

EXPORT_PROPERTY_COLUMNS: Final = (
    "spin_splitting_with_soc",
    "spin_splitting_without_soc",
    "ahc_with_soc",
    "ahc_without_soc",
    "is_altermagnet",
    "is_spin_orbit_magnet",
)

EXPORT_WYCKOFF_COLUMNS: Final = (
    "wyckoff_split",
    "acc_primitive_wyckoff_split",
)

EXPORT_MAGNETIC_SITE_COLUMNS: Final = (
    "magnetic_site_status",
    "magnetic_site_setting",
    "magnetic_site_sg_primitive_to_magnetic_primitive_cell_expansion",
    "magnetic_atom_count",
    "nonzero_moment_atom_count",
    "zero_moment_magnetic_atom_count",
    "magnetic_atom_selection_mode",
    "number_of_magnetic_orbits_sg",
    "number_of_magnetic_orbits_ssg",
    "number_of_magnetic_orbits_msg",
    "max_magnetic_site_dof_ssg",
    "max_magnetic_site_dof_msg",
    "total_magnetic_site_dof_ssg",
    "total_magnetic_site_dof_msg",
    "magnetic_wyckoff_dof_summary",
)

QUASI2D_EXPORT_COLUMNS: Final = (
    "case_id",
    "file_name",
    "index",
    "quasi2d_status",
    "quasi2d_source",
    "vacuum_axis_input",
    "spin_splitting_2d",
    "spin_splitting_2d_interpretation",
    "is_alter_2d",
    "quasi2d_magnetic_phase",
    "quasi2d_gp_label",
    "quasi2d_gp_symbol",
    "quasi2d_gp_k_input",
    "quasi2d_gp_k_acc",
    "quasi2d_gp_spin_splitting",
    "quasi2d_gp_spin_polarizations",
    "quasi2d_kpoint_projection_summary",
    "quasi2d_kpoints",
)

EXPORT_ERROR_COLUMNS: Final = (
    "error_type",
    "error_message",
)

EXPORT_ROW_COLUMNS: Final = (
    *EXPORT_METADATA_COLUMNS,
    *EXPORT_CASE_COLUMNS,
    *EXPORT_IDENTIFY_COLUMNS,
    *EXPORT_SPIN_POINT_GROUP_COLUMNS,
    *EXPORT_SYMBOL_COLUMNS,
    *EXPORT_PROPERTY_COLUMNS,
    *EXPORT_WYCKOFF_COLUMNS,
    *EXPORT_MAGNETIC_SITE_COLUMNS,
    *EXPORT_ERROR_COLUMNS,
)

MAGNETIC_ORBIT_EXPORT_COLUMNS: Final = (
    "case_id",
    "file_name",
    "index",
    "element",
    "site_count",
    "site_indices",
    "sg_wyckoff",
    "sg_wyckoff_index",
    "ssg_wyckoff_with_dof",
    "ssg_wyckoff",
    "ssg_wyckoff_index",
    "ssg_site_dof",
    "ssg_orbit_total_dof",
    "ssg_constraints",
    "ssg_representative_index",
    "msg_wyckoff_with_dof",
    "msg_wyckoff",
    "msg_wyckoff_index",
    "msg_site_dof",
    "msg_orbit_total_dof",
    "msg_constraints",
    "msg_representative_index",
)


def empty_export_row() -> dict[str, object | None]:
    """Return a row initialized with every public export column."""

    return {column: None for column in EXPORT_ROW_COLUMNS}


def complete_export_row(row: dict[str, object]) -> dict[str, object]:
    """Fill absent public export columns while preserving extra diagnostic keys."""

    completed = empty_export_row()
    completed.update(row)
    return completed
