import argparse
import json
import re
import sys
from pathlib import Path

from .find_spin_group import (
    NumpyEncoder,
    SCIF_CELL_MODE_DATABASE_STANDARD,
    SCIF_CELL_MODE_DATABASE_STANDARD_CARTESIAN,
    SCIF_CELL_MODE_DATABASE_STANDARD_ORIENTED,
    SCIF_CELL_MODE_INPUT_CARTESIAN,
    SCIF_CELL_MODE_INPUT_IDENTIFIED,
    SCIF_CELL_MODE_INPUT_ORIENTED,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_CARTESIAN,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_ORIENTED,
    SCIF_CELL_MODE_SSG_CONVENTION_CARTESIAN,
    SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED,
    find_spin_group,
    find_spin_group_acc_primitive,
    find_spin_group_basic,
    find_spin_group_input_ssg,
    find_spin_group_poscar_ssg,
    write_poscar_ssg_symmetry_dat,
    write_ssg_operation_matrices,
)
from .version import __version__


_AUTO_INPUT_EXTENSIONS = {".scif", ".mcif", ".cif", ".vasp", ".poscar"}
_AUTO_IGNORE_NAMES = {"ssg_symm.json", "input_poscar.vasp", "magnetic_primitive_poscar.vasp"}
_SCIF_CELL_MODE_CHOICES = (
    SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED,
    SCIF_CELL_MODE_SSG_CONVENTION_CARTESIAN,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_ORIENTED,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE_CARTESIAN,
    SCIF_CELL_MODE_DATABASE_STANDARD_ORIENTED,
    SCIF_CELL_MODE_DATABASE_STANDARD_CARTESIAN,
    SCIF_CELL_MODE_INPUT_ORIENTED,
    SCIF_CELL_MODE_INPUT_CARTESIAN,
    SCIF_CELL_MODE_MAGNETIC_PRIMITIVE,
    SCIF_CELL_MODE_DATABASE_STANDARD,
    SCIF_CELL_MODE_INPUT_IDENTIFIED,
)
_SHOW_FIELD_ALIASES = {
    "kpoints": "KPOINTS",
    "kpoints_text": "KPOINTS",
    "poscar": "acc_primitive_magnetic_cell_poscar",
    "primitive_poscar": "acc_primitive_magnetic_cell_poscar",
    "primitive-poscar": "acc_primitive_magnetic_cell_poscar",
    "acc_primitive_poscar": "acc_primitive_magnetic_cell_poscar",
    "acc-primitive-poscar": "acc_primitive_magnetic_cell_poscar",
    "scif_default": "scif",
    "default_scif": "scif",
    "gspg": "gspg_text",
    "operation-views": "operation_views",
    "ops": "operation_views",
    "wp-chain": "wp_chain",
    "wyckoff-chain": "wp_chain",
    "spin-texture-no-soc": "spin_texture_config_no_soc",
    "spin-texture-soc": "spin_texture_config_soc",
}
_POSCAR_MAGMOM_PATTERN = re.compile(r"^\s*#*\s*magmom\s*=", re.IGNORECASE | re.MULTILINE)
_INCAR_MAGMOM_PATTERN = re.compile(r"^\s*MAGMOM\s*=", re.IGNORECASE | re.MULTILINE)
_CIF_MOMENT_TAG_MARKERS = (
    "_atom_site_moment.",
    "_atom_site_moment_",
    "_atom_site_spin_moment.",
    "_atom_site_spin_moment_",
    "_atom_site_orbital_moment.",
    "_atom_site_orbital_moment_",
)


def _read_text_for_auto_detect(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _has_embedded_poscar_magmom(path: Path) -> bool:
    try:
        return _POSCAR_MAGMOM_PATTERN.search(_read_text_for_auto_detect(path)) is not None
    except OSError:
        return False


def _has_sibling_incar_magmom(path: Path) -> bool:
    incar_path = path.with_name("INCAR")
    if not incar_path.is_file():
        return False
    try:
        logical_text = []
        for raw_line in _read_text_for_auto_detect(incar_path).splitlines():
            line = re.split(r"[#!]", raw_line, maxsplit=1)[0].strip()
            if line:
                logical_text.append(line)
        return _INCAR_MAGMOM_PATTERN.search("\n".join(logical_text)) is not None
    except OSError:
        return False


def _has_cif_moment_tags(path: Path) -> bool:
    try:
        text = _read_text_for_auto_detect(path).lower()
    except OSError:
        return False
    return any(marker in text for marker in _CIF_MOMENT_TAG_MARKERS)


def _discover_structure_candidates(cwd: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(cwd.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower in _AUTO_IGNORE_NAMES:
            continue
        if name_lower in {"poscar", "contcar"} or path.suffix.lower() in _AUTO_INPUT_EXTENSIONS:
            if _candidate_priority(path) is None:
                continue
            candidates.append(path)
    return candidates


def _candidate_priority(path: Path) -> tuple[int, str] | None:
    name_lower = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".scif":
        return (0, name_lower)
    if suffix == ".mcif":
        return (1, name_lower)
    if suffix == ".cif" and _has_cif_moment_tags(path):
        return (2, name_lower)
    if name_lower == "poscar" and _has_sibling_incar_magmom(path):
        return (3, name_lower)
    if name_lower == "poscar" and _has_embedded_poscar_magmom(path):
        return (4, name_lower)
    if suffix in {".vasp", ".poscar"} and _has_embedded_poscar_magmom(path):
        return (5, name_lower)
    if name_lower == "contcar":
        return (6, name_lower)
    return None


def _select_structure_file(explicit_file: str | None) -> str:
    if explicit_file:
        return explicit_file

    cwd = Path.cwd()
    candidates = _discover_structure_candidates(cwd)
    if not candidates:
        raise ValueError("No readable structure file was found in the current directory.")

    ordered = sorted(candidates, key=_candidate_priority)
    selected = ordered[0]
    if len(ordered) == 1:
        print(f"[fsg] Auto-selected structure file: {selected.name}", file=sys.stderr)
    else:
        others = ", ".join(path.name for path in ordered[1:])
        print(
            f"[fsg] Multiple structure files found. Using {selected.name}. Other candidates: {others}",
            file=sys.stderr,
        )
    return str(selected)


def _to_serializable_payload(result):
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _resolve_show_path(value, path: str):
    path = _SHOW_FIELD_ALIASES.get(path, path)
    current = value
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(path)
            current = current[segment]
            continue
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        raise KeyError(path)
    return current


def _print_show_text(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


def _format_show_scalar(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_number(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        return f"{value:.8g}"
    return str(value)


def _format_vector(value) -> str:
    if not isinstance(value, (list, tuple)):
        return _format_show_scalar(value)
    return "[" + ", ".join(_format_number(component) for component in value) + "]"


def _format_matrix(value) -> str:
    if not isinstance(value, (list, tuple)):
        return _format_show_scalar(value)
    rows = []
    for row in value:
        if isinstance(row, (list, tuple)):
            rows.append("[" + ", ".join(_format_number(component) for component in row) + "]")
        else:
            rows.append(_format_number(row))
    return "[" + "; ".join(rows) + "]"


def _operation_row_from_payload(op: dict, index: int, seitz_latex: str | None = None) -> dict:
    return {
        "No.": op.get("index", index),
        "Spin Rotation": _format_matrix(op.get("spin_rotation")),
        "Space Rotation": _format_matrix(op.get("real_rotation")),
        "Space Translation": _format_vector(op.get("translation")),
        "Seitz Symbol": seitz_latex if seitz_latex is not None else op.get("seitz_latex", ""),
    }


def _format_table(rows: list[dict], *, max_rows: int | None = None) -> str:
    if not rows:
        return "(empty)"
    display_rows = rows if max_rows is None else rows[:max_rows]
    columns = list(display_rows[0].keys())
    widths = {
        column: max(len(str(column)), *(len(str(row.get(column, ""))) for row in display_rows))
        for column in columns
    }
    lines = [
        " | ".join(str(column).ljust(widths[column]) for column in columns),
        "-+-".join("-" * widths[column] for column in columns),
    ]
    for row in display_rows:
        lines.append(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def _looks_like_operation_view(value: dict) -> bool:
    return isinstance(value, dict) and isinstance(value.get("ops"), list)


def _format_operation_view(value: dict) -> str:
    lines = []
    label = value.get("label")
    if label:
        lines.append(str(label))
    if value.get("operation_count") is not None:
        lines.append(f"operation_count: {value.get('operation_count')}")
    if value.get("indices") is not None:
        indices = value.get("indices") or []
        lines.append(f"indices: {', '.join(str(index) for index in indices) if indices else '(empty)'}")
    note = value.get("note")
    if isinstance(note, dict):
        note_text = note.get("text")
        note_parts = [f"{key}: {val}" for key, val in note.items() if key != "text" and val not in (None, "")]
        if note_text:
            lines.append(f"note: {note_text}")
        if note_parts:
            lines.append("note_detail: " + "; ".join(note_parts))
    elif note:
        lines.append(f"note: {note}")

    seitz_latex = value.get("seitz_latex") or []
    rows = []
    for index, op in enumerate(value.get("ops") or [], start=1):
        if not isinstance(op, dict):
            continue
        seitz = seitz_latex[index - 1] if index - 1 < len(seitz_latex) else None
        rows.append(_operation_row_from_payload(op, index, seitz))
    if rows:
        if lines:
            lines.append("")
        lines.append(_format_table(rows))
    return "\n".join(lines) if lines else "(empty)"


def _format_operation_views_summary(value: dict) -> str:
    rows = []
    for setting_key, setting_payload in value.items():
        if not isinstance(setting_payload, dict):
            continue
        views = setting_payload.get("views") or {}
        for view_key, view_payload in views.items():
            if not isinstance(view_payload, dict):
                continue
            rows.append(
                {
                    "Setting": setting_key,
                    "Default": "yes" if setting_payload.get("default_view") == view_key else "",
                    "View": view_key,
                    "Ops": view_payload.get("operation_count", len(view_payload.get("ops") or [])),
                    "Label": view_payload.get("label", ""),
                }
            )
    return _format_table(rows) if rows else "(empty)"


def _format_spin_texture_config(value: dict) -> str:
    lines = []
    preferred_keys = (
        "spin_texture_type",
        "wave_type",
        "momentum_space_spin_configuration",
        "spin_rank",
        "nullity",
        "order",
        "source",
        "basis_setting",
    )
    for key in preferred_keys:
        if key in value:
            lines.append(f"{key}: {_format_show_scalar(value.get(key))}")
    basis = value.get("basis")
    if isinstance(basis, list):
        lines.append("basis:")
        if basis:
            lines.extend(f"  {index}. {item}" for index, item in enumerate(basis, start=1))
        else:
            lines.append("  (empty)")
    if "basis_latex" in value and isinstance(value.get("basis_latex"), list):
        lines.append("basis_latex:")
        basis_latex = value.get("basis_latex") or []
        if basis_latex:
            lines.extend(f"  {index}. {item}" for index, item in enumerate(basis_latex, start=1))
        else:
            lines.append("  (empty)")
    for key, val in value.items():
        if key in preferred_keys or key in {"basis", "basis_latex", "basis_by_order"}:
            continue
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            lines.append(f"{key}:")
            lines.append(_format_show_value(val, indent=2))
        else:
            lines.append(f"{key}: {_format_show_value(val, indent=2)}")
    if isinstance(value.get("basis_by_order"), list):
        lines.append("basis_by_order:")
        lines.append(_format_show_value(value["basis_by_order"], indent=2))
    return "\n".join(lines) if lines else "(empty)"


def _format_text_mapping_summary(value: dict) -> str:
    rows = []
    for key, text in value.items():
        if not isinstance(text, str):
            continue
        rows.append(
            {
                "Key": key,
                "Lines": len(text.splitlines()),
                "Characters": len(text),
            }
        )
    return _format_table(rows) if rows else "(empty)"


def _format_generic_dict(value: dict, *, indent: int = 0) -> str:
    lines = []
    prefix = " " * indent
    for key, val in value.items():
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            nested = _format_show_value(val, indent=indent + 2)
            lines.append(f"{prefix}{key}:")
            lines.append(nested)
        else:
            lines.append(f"{prefix}{key}: {_format_show_scalar(val)}")
    return "\n".join(lines) if lines else f"{prefix}(empty)"


def _format_generic_list(value: list, *, indent: int = 0) -> str:
    prefix = " " * indent
    if not value:
        return f"{prefix}(empty)"
    if all(isinstance(item, dict) for item in value):
        if all("basis" in item for item in value):
            lines = []
            for index, item in enumerate(value, start=1):
                label = item.get("order")
                heading = f"order {label}" if label is not None else f"entry {index}"
                lines.append(f"{prefix}{heading}:")
                lines.append(_format_show_value(item, indent=indent + 2))
            return "\n".join(lines)
        scalar_keys = []
        for key in value[0].keys():
            if all(not isinstance(item.get(key), (dict, list)) for item in value):
                scalar_keys.append(key)
        if scalar_keys:
            rows = [
                {key: _format_show_scalar(item.get(key)) for key in scalar_keys}
                for item in value
            ]
            return "\n".join(prefix + line if line else line for line in _format_table(rows, max_rows=80).splitlines())
    lines = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, (dict, list)):
            lines.append(f"{prefix}{index}.")
            lines.append(_format_show_value(item, indent=indent + 2))
        else:
            lines.append(f"{prefix}{index}. {_format_show_scalar(item)}")
    return "\n".join(lines)


def _format_show_value(value, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, str):
        return value if indent == 0 else "\n".join(prefix + line if line else line for line in value.splitlines())
    if isinstance(value, (int, float, bool)) or value is None:
        return prefix + _format_show_scalar(value)
    if isinstance(value, dict):
        if all(isinstance(item, str) and "\n" in item for item in value.values()):
            return _format_text_mapping_summary(value)
        if _looks_like_operation_view(value):
            rendered = _format_operation_view(value)
            return "\n".join(prefix + line if line else line for line in rendered.splitlines())
        if "views" in value and isinstance(value.get("views"), dict):
            return _format_operation_views_summary({"selected": value})
        if all(isinstance(item, dict) and "views" in item for item in value.values()):
            return _format_operation_views_summary(value)
        if "basis" in value and (
            "spin_texture_type" in value
            or "wave_type" in value
            or "momentum_space_spin_configuration" in value
        ):
            rendered = _format_spin_texture_config(value)
            return "\n".join(prefix + line if line else line for line in rendered.splitlines())
        return _format_generic_dict(value, indent=indent)
    if isinstance(value, list):
        if all(isinstance(item, dict) and "spin_rotation" in item and "real_rotation" in item for item in value):
            rows = [_operation_row_from_payload(item, index) for index, item in enumerate(value, start=1)]
            rendered = _format_table(rows)
            return "\n".join(prefix + line if line else line for line in rendered.splitlines())
        return _format_generic_list(value, indent=indent)
    return prefix + _format_show_scalar(value)


def _emit_payload(payload, show_paths: list[str] | None, *, output_json: bool = False):
    if not show_paths:
        print(json.dumps(payload, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        return

    resolved = {}
    missing = []
    for path in show_paths:
        resolved_path = _SHOW_FIELD_ALIASES.get(path, path)
        try:
            resolved[path] = _resolve_show_path(payload, resolved_path)
        except KeyError:
            resolved[path] = None
            missing.append(path)

    if missing:
        names = ", ".join(repr(path) for path in missing)
        raise ValueError(
            f"Unknown or unavailable --show field(s): {names}. "
            "The field may require --full; see `fsg --help` or the CLI field guide."
        )

    if output_json:
        if len(show_paths) == 1:
            print(json.dumps(resolved[show_paths[0]], indent=2, ensure_ascii=False, cls=NumpyEncoder))
        else:
            print(json.dumps(resolved, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        return

    if len(show_paths) == 1:
        _print_show_text(_format_show_value(resolved[show_paths[0]]))
        return

    sections = []
    for path in show_paths:
        sections.append(f"## {path}\n{_format_show_value(resolved[path])}")
    _print_show_text("\n\n".join(sections))


def _format_optional(value) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _format_bool(value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _format_group_number_symbol(number, symbol) -> str:
    parts = [_format_optional(number), _format_optional(symbol)]
    return " ".join(part for part in parts if part != "-") or "-"


def _index_detail(payload: dict, key: str, fallback_key: str):
    details = payload.get("identify_index_details")
    if isinstance(details, dict) and details.get(key) is not None:
        return details.get(key)
    return payload.get(fallback_key)


def _format_msg_summary(payload: dict) -> str:
    number = payload.get("msg_bns_number") or payload.get("msg_num")
    symbol = payload.get("msg_symbol")
    msg = _format_group_number_symbol(number, symbol)
    msg_type = payload.get("msg_type")
    if msg_type is not None:
        msg += f" (type {msg_type})"
    return msg


def _format_spin_texture_wave_basis(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return "-"
    wave = payload.get("spin_texture_type") or payload.get("wave_type")
    basis = payload.get("basis")
    if basis is None:
        basis_text = "-"
    elif isinstance(basis, list):
        basis_text = "; ".join(str(item) for item in basis) if basis else "none"
    else:
        basis_text = str(basis)
    parts = [f"wave={_format_optional(wave)}"]
    if payload.get("basis_setting"):
        parts.append(f"setting={payload['basis_setting']}")
    parts.append(f"basis={basis_text}")
    return "; ".join(parts)


def _format_symmetry_permission(value) -> str:
    if value == "Yes":
        return "allowed"
    if value == "No":
        return "forbidden"
    return _format_optional(value)


def _format_spin_texture_headline(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return "not available"
    wave = payload.get("spin_texture_type") or payload.get("wave_type")
    if wave is None:
        return "not available"
    details = []
    if payload.get("order") is not None:
        details.append(f"order {payload['order']}")
    configuration = payload.get("momentum_space_spin_configuration")
    if configuration and configuration != "zero":
        details.append(str(configuration))
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{wave}{suffix}"


def _format_net_moment(payload: dict) -> str:
    value = payload.get("net_moment")
    tol = payload.get("zero_net_moment_tol")
    if value is None and tol is None:
        return "-"
    text = f"{_format_optional(value)} μB"
    if tol is not None:
        text += f" (zero tol {tol} μB)"
    return text


def _flag_is_present(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"false", "no", "0", "none", "-"}


def _format_magnetic_phase_line(payload: dict) -> str:
    properties = payload.get("properties") or {}
    magnetic_phase = _format_optional(payload.get("magnetic_phase") or payload.get("phase"))
    return (
        f"{magnetic_phase}; "
        f"altermagnet={_format_bool(_flag_is_present(payload.get('is_alter') or properties.get('is_alter')))}; "
        f"spin-orbit magnet={_format_bool(_flag_is_present(payload.get('is_som') or properties.get('is_spin_orbit_magnet')))}"
    )


def _format_axes(axes) -> str:
    if axes is None:
        return "unknown"
    if not axes:
        return "none"
    labels = []
    for axis in axes:
        if isinstance(axis, dict):
            label = axis.get("label")
            if label:
                labels.append(str(label))
                continue
            components = axis.get("components")
            if components is not None:
                labels.append(str(components))
                continue
        labels.append(str(axis))
    return ", ".join(labels)


_VECTOR_CONSTRAINT_LABELS = {
    "real_space_t_even_p_odd": "T-even/P-odd real",
    "real_space_t_even_p_even": "T-even/P-even real",
    "real_space_t_odd_p_odd": "T-odd/P-odd real",
    "real_space_t_odd_p_even": "T-odd/P-even real",
    "spin_space_t_odd_p_even": "T-odd/P-even spin",
    "spin_space_t_even_p_even": "T-even/P-even spin",
}
_VECTOR_CONSTRAINT_ORDER = tuple(_VECTOR_CONSTRAINT_LABELS)


def _format_vector_constraint_entry(key: str, payload: dict) -> str:
    label = _VECTOR_CONSTRAINT_LABELS.get(key, key)
    free_dimension = payload.get("free_dimension")
    allowed_axes = payload.get("allowed_axes")
    axes = _format_axes(allowed_axes)
    setting = payload.get("allowed_axes_setting")
    setting_suffix = f" @ {setting}" if allowed_axes and setting else ""
    if free_dimension is None:
        return f"{label}={axes}{setting_suffix}"
    return f"{label}={free_dimension}D {axes}{setting_suffix}"


def _format_vector_constraint_scope(scope_payload: dict | None) -> str:
    if not isinstance(scope_payload, dict):
        return "-"
    constraints = scope_payload.get("constraints") or {}
    if not constraints:
        return "-"
    entries = []
    for key in _VECTOR_CONSTRAINT_ORDER:
        constraint_payload = constraints.get(key)
        if isinstance(constraint_payload, dict):
            entries.append(_format_vector_constraint_entry(key, constraint_payload))
    for key, constraint_payload in constraints.items():
        if key not in _VECTOR_CONSTRAINT_LABELS and isinstance(constraint_payload, dict):
            entries.append(_format_vector_constraint_entry(key, constraint_payload))
    return "; ".join(entries) if entries else "-"


def _format_vector_constraint_scope_with_flags(
    label: str,
    scope_payload: dict | None,
    *,
    is_polar,
    is_chiral,
) -> str:
    return (
        f"{label}: "
        f"polar={_format_bool(is_polar)}, chiral={_format_bool(is_chiral)}; "
        f"{_format_vector_constraint_scope(scope_payload)}"
    )


def _emit_basic_summary(payload: dict, *, source: str | None = None) -> None:
    properties = payload.get("properties") or {}
    phase = payload.get("magnetic_phase") or payload.get("phase")
    phase_text = _format_optional(phase)
    if phase_text != "-":
        phase_text = " ".join(phase_text.splitlines())
    lines = ["FindSpinGroup result"]
    if source:
        lines.append(f"Input: {source}")
    lines.extend(
        [
            f"OSSG: {_format_optional(payload.get('index'))}",
            f"MSG with SOC: {_format_msg_summary(payload)}",
            "Magnetic order: "
            f"{_format_optional(payload.get('conf'))}; {phase_text}",
            "Net moment: "
            f"{_format_number(payload.get('net_moment'))} μB "
            f"(zero threshold: {_format_number(payload.get('zero_net_moment_tol'))} μB)",
            "",
            "Symmetry-allowed responses:",
            "  Spin splitting: "
            f"without SOC {_format_symmetry_permission(properties.get('ss_wo_soc'))}; "
            f"with SOC {_format_symmetry_permission(properties.get('ss_w_soc'))}",
            "  AHC: "
            f"without SOC {_format_symmetry_permission(properties.get('ahc_wo_soc'))}; "
            f"with SOC {_format_symmetry_permission(properties.get('ahc_w_soc'))}",
            "  Leading spin texture: "
            f"without SOC {_format_spin_texture_headline(payload.get('spin_texture_config_no_soc'))}; "
            f"with SOC {_format_spin_texture_headline(payload.get('spin_texture_config_soc'))}",
            "",
            "Interpretation: allowed/forbidden are symmetry statements, not calculated magnitudes.",
            "More: `--details`, `--show FIELD`, `--json`, or `fsg --help`.",
        ]
    )
    print("\n".join(lines))


def _emit_detailed_basic_summary(payload: dict, *, source: str | None = None) -> None:
    properties = payload.get("properties") or {}
    vector_constraints = payload.get("vector_constraints_by_symmetry")
    lines = []
    if source:
        lines.append(f"Input: {source}")
    lines.extend(
        [
            "Identification:",
            f"OSSG symbol: {_format_optional(payload.get('ossg_symbol_linear') or payload.get('ossg_symbol'))}",
            f"Index: {_format_optional(payload.get('index'))}",
            "G0: "
            f"{_format_group_number_symbol(payload.get('g0_number'), payload.get('g0_symbol'))}; "
            f"L0: {_format_group_number_symbol(payload.get('l0_number'), payload.get('l0_symbol'))}; "
            f"t_index: {_format_optional(_index_detail(payload, 't_index', 'it'))}; "
            f"k_index: {_format_optional(_index_detail(payload, 'k_index', 'ik'))}",
            "Spin-space point group: "
            f"HM={_format_optional(payload.get('spin_space_point_group_hm') or payload.get('sspg'))}; "
            f"Schoenflies={_format_optional(payload.get('spin_space_point_group_schoenflies'))}",
            "Nontrivial spin-space point group: "
            f"HM={_format_optional(payload.get('nontrivial_spin_space_point_group_hm') or payload.get('nsspg'))}; "
            f"Schoenflies={_format_optional(payload.get('nontrivial_spin_space_point_group_schoenflies'))}",
            f"Space group: {_format_group_number_symbol(payload.get('space_group_number'), payload.get('space_group_symbol'))}",
            f"Magnetic space group: {_format_msg_summary(payload)}",
            f"Spin arithmetic crystal class: {_format_optional(payload.get('acc_symbol') or payload.get('acc'))}",
            f"EMPG: {_format_optional(payload.get('empg'))}",
            "",
            "Magnetic phase and properties:",
            f"Configuration: {_format_optional(payload.get('conf'))}",
            f"Magnetic phase: {_format_magnetic_phase_line(payload)}",
            f"Net moment: {_format_net_moment(payload)}",
            "Spin splitting: "
            f"w/o SOC={_format_optional(properties.get('ss_wo_soc'))}, "
            f"w/ SOC={_format_optional(properties.get('ss_w_soc'))}",
            "AHC: "
            f"w/o SOC={_format_optional(properties.get('ahc_wo_soc'))}, "
            f"w/ SOC={_format_optional(properties.get('ahc_w_soc'))}",
            "Spin texture w/o SOC: "
            f"{_format_spin_texture_wave_basis(payload.get('spin_texture_config_no_soc'))}",
            "Spin texture w/ SOC: "
            f"{_format_spin_texture_wave_basis(payload.get('spin_texture_config_soc'))}",
            "",
            "Vector constraints:",
            _format_vector_constraint_scope_with_flags(
                "SG",
                (vector_constraints or {}).get("sg"),
                is_polar=payload.get("sg_is_polar"),
                is_chiral=payload.get("sg_is_chiral"),
            ),
            _format_vector_constraint_scope_with_flags(
                "OSSG/G0",
                (vector_constraints or {}).get("ossg"),
                is_polar=payload.get("ssg_is_polar"),
                is_chiral=payload.get("ssg_is_chiral"),
            ),
            _format_vector_constraint_scope_with_flags(
                "MSG",
                (vector_constraints or {}).get("msg"),
                is_polar=payload.get("msg_is_polar"),
                is_chiral=payload.get("msg_is_chiral"),
            ),
        ]
    )
    print("\n".join(lines))


def _write_input_ssg_output_dir(directory: Path, payload: dict) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = [write_poscar_ssg_symmetry_dat(directory / "ssg_symm.json", payload)]
    input_poscar = payload.get("input_poscar")
    if input_poscar:
        path = directory / "input_poscar.vasp"
        path.write_text(input_poscar, encoding="utf-8")
        written.append(path)
    magnetic_primitive_poscar = payload.get("magnetic_primitive_poscar")
    if magnetic_primitive_poscar:
        path = directory / "magnetic_primitive_poscar.vasp"
        path.write_text(magnetic_primitive_poscar, encoding="utf-8")
        written.append(path)
    return written


def _full_route_result(args):
    return find_spin_group(
        args.structure_file,
        space_tol=args.space_tol,
        mtol=args.mtol,
        meigtol=args.meigtol,
        matrix_tol=args.matrix_tol,
        parser_atol=args.parser_atol,
        calculation_mode=args.calculation_mode,
        vacuum_axis=args.vacuum_axis,
        spin_texture_basis_max_order=args.spin_texture_basis_max_order,
        poscar_allow_incar_magmom=True,
        poscar_prefer_incar_magmom=True,
    )


def _write_scif_file(path: Path, result, *, cell_mode: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_scif(cell_mode=cell_mode), encoding="utf-8")
    return path


def _write_poscar_kpoints_dir(directory: Path, result) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    poscar = getattr(result, "acc_primitive_magnetic_cell_poscar", None)
    if not poscar:
        raise ValueError("Full-route result does not contain `acc_primitive_magnetic_cell_poscar`.")
    kpoints = getattr(result, "KPOINTS", None)
    if not kpoints:
        raise ValueError("Full-route result does not contain `KPOINTS`.")

    poscar_path = directory / "POSCAR"
    kpoints_path = directory / "KPOINTS"
    poscar_path.write_text(poscar, encoding="utf-8")
    kpoints_path.write_text(kpoints, encoding="utf-8")
    return [poscar_path, kpoints_path]


def _artifact_summary(result, written: list[Path]) -> dict:
    return {
        "written_files": [str(path) for path in written],
        "summary": {
            "index": getattr(result, "index", None),
            "conf": getattr(result, "conf", None),
            "phase": getattr(result, "phase", None),
            "magnetic_phase": getattr(result, "magnetic_phase", None),
            "KPOINTS_setting": getattr(result, "KPOINTS_setting", None),
            "KPOINTS_real_space_setting": getattr(result, "KPOINTS_real_space_setting", None),
        },
    }


def _uses_full_route(args) -> bool:
    return bool(
        args.all
        or args.mode == "full"
        or args.write_scif
        or args.write_poscar_kpoints
    )


def _validate_route_options(args) -> None:
    if args.spin_texture_basis_max_order is not None and args.spin_texture_basis_max_order < 0:
        raise ValueError("`--spin-texture-basis-max-order` must be non-negative.")

    artifact_writes = bool(args.write_scif or args.write_poscar_kpoints)
    if args.details and (
        args.mode is not None
        or args.all
        or args.write
        or artifact_writes
        or args.show
        or args.json
    ):
        raise ValueError(
            "`--details` expands the default quick summary and cannot be combined "
            "with another analysis/output selector."
        )
    if args.mode is not None:
        if args.all or args.write or args.show or artifact_writes:
            raise ValueError(
                "Use either legacy `--mode` or the new `--full/--show/-w/write-artifact` flags, not both."
            )
    elif args.all and (args.write or artifact_writes):
        raise ValueError("`--full/--all` cannot be combined with `-w/--write` or write-artifact flags.")
    elif args.write and artifact_writes:
        raise ValueError("`-w/--write` cannot be combined with write-artifact flags.")
    elif args.write and (args.show or args.json):
        raise ValueError("`-w/--write` cannot be combined with `--show` or `--json`.")
    elif artifact_writes and (args.show or args.json):
        raise ValueError("Write-artifact flags cannot be combined with `--show` or `--json`.")

    if args.all and not args.show:
        raise ValueError(
            "`--full/--all` requires at least one `--show FIELD`; the raw full result "
            "is too large and is not a stable JSON contract. Example: "
            "`fsg --full FILE --show operation-views`."
        )

    if args.write_ssg_matrices and args.mode != "acc-primitive":
        raise ValueError("`--write-ssg-matrices` is only valid with `--mode acc-primitive`.")
    if args.ssg_matrix_setting is not None and not args.write_ssg_matrices:
        raise ValueError("`--ssg-matrix-setting` requires `--write-ssg-matrices`.")
    if args.write_symmetry_dat and args.mode not in {"input-ssg", "poscar-ssg"}:
        raise ValueError("`--write-symmetry-dat` is only valid with `--mode input-ssg` or `--mode poscar-ssg`.")
    if args.scif_cell_mode is not None and not args.write_scif:
        raise ValueError("`--scif-cell-mode` requires `--write-scif`.")

    if args.calculation_mode != "3d" and not _uses_full_route(args):
        raise ValueError("`--calculation-mode` requires full analysis; use `--full`.")
    if args.vacuum_axis != "c" and not _uses_full_route(args):
        raise ValueError("`--vacuum-axis` requires full analysis; use `--full`.")


def _legacy_mode_payload(args):
    poscar_magmom_kwargs = {
        "poscar_allow_incar_magmom": True,
        "poscar_prefer_incar_magmom": True,
    }
    if args.mode == "basic":
        return find_spin_group_basic(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            parser_atol=args.parser_atol,
            spin_texture_basis_max_order=args.spin_texture_basis_max_order,
            **poscar_magmom_kwargs,
        )
    if args.mode == "acc-primitive":
        payload = find_spin_group_acc_primitive(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            parser_atol=args.parser_atol,
            spin_texture_basis_max_order=args.spin_texture_basis_max_order,
            **poscar_magmom_kwargs,
        )
        if args.write_ssg_matrices:
            matrix_setting = args.ssg_matrix_setting or "acc-primitive"
            key = (
                "acc_primitive_ssg_operation_matrices"
                if matrix_setting == "acc-primitive"
                else "acc_primitive_poscar_spin_frame_ssg_operation_matrices"
            )
            write_ssg_operation_matrices(args.write_ssg_matrices, payload[key])
        return payload
    if args.mode == "poscar-ssg":
        payload = find_spin_group_poscar_ssg(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            **poscar_magmom_kwargs,
        )
        if args.write_symmetry_dat:
            write_poscar_ssg_symmetry_dat(args.write_symmetry_dat, payload)
        return payload
    if args.mode == "input-ssg":
        payload = find_spin_group_input_ssg(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            **poscar_magmom_kwargs,
        )
        if args.write_symmetry_dat:
            write_poscar_ssg_symmetry_dat(args.write_symmetry_dat, payload)
        return payload
    result = find_spin_group(
        args.structure_file,
        space_tol=args.space_tol,
        mtol=args.mtol,
        meigtol=args.meigtol,
        matrix_tol=args.matrix_tol,
        parser_atol=args.parser_atol,
        calculation_mode=args.calculation_mode,
        vacuum_axis=args.vacuum_axis,
        spin_texture_basis_max_order=args.spin_texture_basis_max_order,
        **poscar_magmom_kwargs,
    )
    return _to_serializable_payload(result)


def main():
    parser = argparse.ArgumentParser(
        prog="fsg",
        usage="fsg [OPTIONS] [STRUCTURE]",
        description=(
            "Identify spin-space symmetry and symmetry-allowed physical responses\n"
            "from a structure containing magnetic moments."
        ),
        epilog=(
            "Examples:\n"
            "  fsg structure.mcif\n"
            "  fsg structure.mcif --show index --show magnetic_phase\n"
            "  fsg structure.mcif --show properties\n"
            "  fsg --full structure.mcif --show operation-views\n"
            "  fsg structure.mcif --write-poscar-kpoints calc_inputs\n\n"
            "Without STRUCTURE, fsg searches the current directory for a magnetic\n"
            "SCIF, mCIF, CIF, or POSCAR-like input. For POSCAR-like inputs the CLI\n"
            "prefers MAGMOM from a sibling INCAR when present.\n\n"
            "Results go to stdout; auto-selection messages and errors go to stderr."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "structure_file",
        nargs="?",
        metavar="STRUCTURE",
        help="Magnetic .mcif/.cif/.scif or POSCAR-like file; omit to auto-detect.",
    )

    output_group = parser.add_argument_group("analysis level and output")
    output_group.add_argument(
        "--full",
        "--all",
        dest="all",
        action="store_true",
        help="Run full analysis for selected --show fields; requires at least one --show FIELD.",
    )
    output_group.add_argument(
        "--details",
        action="store_true",
        help="Expand the default quick summary with group components, bases, and vector constraints.",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Print the selected analysis payload as machine-readable JSON.",
    )
    output_group.add_argument(
        "--show",
        action="append",
        default=[],
        metavar="FIELD",
        help="Print one field; repeat as needed. Supports dot paths such as properties.ss_wo_soc.",
    )

    export_group = parser.add_argument_group("file export")
    export_group.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Write input-cell operations and magnetic-primitive POSCAR files to the current directory.",
    )
    export_group.add_argument(
        "--write-scif",
        metavar="PATH",
        help="Run full analysis and write SCIF to PATH (replaces an existing file).",
    )
    export_group.add_argument(
        "--scif-cell-mode",
        choices=_SCIF_CELL_MODE_CHOICES,
        default=None,
        help="Cell/spin-frame setting for --write-scif (default: ssg_convention_oriented).",
    )
    export_group.add_argument(
        "--write-poscar-kpoints",
        metavar="DIR",
        help="Write matched ACC-primitive POSCAR and KPOINTS into DIR (replaces same-named files).",
    )

    interpretation_group = parser.add_argument_group("physical interpretation")
    interpretation_group.add_argument(
        "--calculation-mode",
        choices=["auto", "quasi2d", "2d", "3d", "bulk", "slab", "layer"],
        default="3d",
        help="Use quasi2d/2d/slab/layer for slab interpretation; default: 3d.",
    )
    interpretation_group.add_argument(
        "--vacuum-axis",
        choices=["a", "b", "c", "x", "y", "z", "0", "1", "2"],
        default="c",
        help="Input-cell axis normal to the slab for quasi-2D analysis; default: c.",
    )
    interpretation_group.add_argument(
        "--spin-texture-basis-max-order",
        type=int,
        default=None,
        metavar="N",
        help="Set the spin-texture search/output ceiling and emit basis_by_order through degree N.",
    )

    tolerance_group = parser.add_argument_group("advanced numerical tolerances")
    tolerance_group.add_argument(
        "--space-tol",
        "--space_tol",
        dest="space_tol",
        type=float,
        default=0.02,
        help="Shared spatial matching tolerance; default: 0.02.",
    )
    tolerance_group.add_argument(
        "--mtol",
        type=float,
        default=0.02,
        help="Magnetic-moment and zero-net-moment tolerance in μB; default: 0.02.",
    )
    tolerance_group.add_argument(
        "--meigtol",
        type=float,
        default=0.00002,
        help="Spin point-group eigenvalue tolerance; default: 2e-5.",
    )
    tolerance_group.add_argument(
        "--matrix-tol",
        "--matrix_tol",
        dest="matrix_tol",
        type=float,
        default=0.01,
        help="Matrix/standardization tolerance; default: 0.01.",
    )
    tolerance_group.add_argument(
        "--parser-atol",
        "--parser_atol",
        dest="parser_atol",
        type=float,
        default=0.02,
        help="Parser-side expanded-moment consistency tolerance; default: 0.02.",
    )

    legacy_group = parser.add_argument_group("legacy and specialized compatibility")
    legacy_group.add_argument(
        "--mode",
        choices=["full", "basic", "acc-primitive", "poscar-ssg", "input-ssg"],
        default=None,
        help="Legacy route selector; prefer quick analysis, --full, or -w.",
    )
    legacy_group.add_argument(
        "--write-ssg-matrices",
        help="When used with --mode acc-primitive, write the selected SSG operation matrices to a JSON file.",
    )
    legacy_group.add_argument(
        "--write-symmetry-dat",
        help="Legacy single-file writer for --mode input-ssg/poscar-ssg.",
    )
    legacy_group.add_argument(
        "--ssg-matrix-setting",
        choices=["acc-primitive", "poscar-spin-frame"],
        default=None,
        help="Which SSG setting to export when --write-ssg-matrices is used.",
    )

    args = parser.parse_args()

    try:
        args.structure_file = _select_structure_file(args.structure_file)
        _validate_route_options(args)

        if args.mode is not None:
            payload = _legacy_mode_payload(args)
            print(json.dumps(payload, indent=2, ensure_ascii=False, cls=NumpyEncoder))
            return

        if args.write:
            payload = find_spin_group_input_ssg(
                args.structure_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
                poscar_allow_incar_magmom=True,
                poscar_prefer_incar_magmom=True,
            )
            written = _write_input_ssg_output_dir(Path.cwd(), payload)
            print(
                json.dumps(
                    {
                        "written_files": [path.name for path in written],
                        "summary": payload["summary"],
                    },
                    indent=2,
                    ensure_ascii=False,
                    cls=NumpyEncoder,
                )
            )
            return

        if args.write_scif or args.write_poscar_kpoints:
            result = _full_route_result(args)
            written: list[Path] = []
            if args.write_scif:
                written.append(
                    _write_scif_file(
                        Path(args.write_scif),
                        result,
                        cell_mode=(
                            args.scif_cell_mode
                            or SCIF_CELL_MODE_SSG_CONVENTION_ORIENTED
                        ),
                    )
                )
            if args.write_poscar_kpoints:
                written.extend(_write_poscar_kpoints_dir(Path(args.write_poscar_kpoints), result))
            print(
                json.dumps(
                    _artifact_summary(result, written),
                    indent=2,
                    ensure_ascii=False,
                    cls=NumpyEncoder,
                )
            )
            return

        if args.all:
            payload = _to_serializable_payload(_full_route_result(args))
        else:
            payload = find_spin_group_basic(
                args.structure_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
                parser_atol=args.parser_atol,
                spin_texture_basis_max_order=args.spin_texture_basis_max_order,
                poscar_allow_incar_magmom=True,
                poscar_prefer_incar_magmom=True,
            )

        if args.show or args.json or args.all:
            _emit_payload(payload, args.show, output_json=args.json)
        elif args.details:
            _emit_detailed_basic_summary(payload, source=args.structure_file)
        else:
            _emit_basic_summary(payload, source=args.structure_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
