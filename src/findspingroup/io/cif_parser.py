import copy
import math
import numpy as np
import re
import shlex
import ast
from fractions import Fraction
from ..structure import AtomicSite,CrystalCell
from ..structure.cell import are_positions_equivalent
from ..utils import general_positions_to_matrix
from ..utils.matrix_utils import evaluate_numeric_expression
class CifParser:


    def __init__(self, filepath=None, source_text=None):
        self.filepath = filepath
        self.source_text = source_text
        self.data = {}

    def parse(self):
        if self.source_text is None:
            with open(self.filepath, 'rb') as f:
                raw = f.read()

            # 2. Try decoding as UTF-8, fallback to Latin-1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
        else:
            text = self.source_text

        # 3. Process the lines (split and remove empty ones)
        lines = [line for line in text.splitlines() if line.strip()]
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # ignore comments and empty lines
            if not line or line.startswith('#'):
                i += 1
                continue

            # loop block
            if line.lower() == 'loop_':
                try:
                    i = self._parse_loop(lines, i + 1)
                except Exception as e:
                    raise ValueError(f"Error parsing loop, check the format of the CIF file!")
                continue

            # single line
            if line.startswith('_'):
                i = self._parse_entry(lines, i)
                continue

            # skip
            i += 1

        return self.data

    def _parse_entry(self, lines, start_idx):
        line = lines[start_idx].strip()
        key, *rest = line.split(maxsplit=1)
        if rest:
            value = rest[0].strip()
        else:
            # next line
            start_idx += 1
            value = lines[start_idx].strip()
        self.data[key] =value
        return start_idx + 1

    def _parse_loop(self, lines, start_idx):
        keys = []
        values = []
        i = start_idx

        # get all keys
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('_'):
                keys.append(line)
                i += 1
            else:
                break

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('_') or line.startswith('#')or line.lower() == 'loop_':
                break

            try:
                parts = shlex.split(line, comments=False, posix=True)
            except ValueError:
                parts = re.split(r'\s+', line)

            values.append(parts)
            i += 1


        for idx, key in enumerate(keys):
            self.data[key] = [row[idx] for row in values]

        return i

    @staticmethod
    def _convert_value(value):
        try:
            if '.' in value or 'e' in value.lower() or 'E' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value



class ScifParser(CifParser):
    pass


def convert_string_to_float(s):
    match = re.search(r"(-?\d+(\.\d+)?)", s)
    if match:
        num = float(match.group(1))
        return num

    else:
        return ValueError('Error,check abc')


def _get_first_existing(data: dict, keys: list[str]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _get_first_existing_with_key(data: dict, keys: list[str]):
    for key in keys:
        if key in data:
            return key, data[key]
    return None, None


def _repo_local_scif_tag_candidates(*suffixes: str) -> list[str]:
    keys = []
    for suffix in suffixes:
        keys.extend(
            [
                f"_space_group_spin.fsg_{suffix}",
                f"_space_group_spin.fsg.{suffix}",
                f"_space_group_spin.{suffix}_fsg",
                f"_space_group_spin.{suffix}",
            ]
        )
    return keys


def _normalize_scif_scalar(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().strip('"').strip("'")
        if stripped in {"", ".", "?"}:
            return None
        return stripped
    return value


def _parse_scif_numeric_token(token: str):
    token = token.strip()
    try:
        return evaluate_numeric_expression(token)
    except Exception:
        return None


def _parse_scif_vector_string(value):
    normalized = _normalize_scif_scalar(value)
    if normalized is None:
        return None
    parts = [part.strip() for part in normalized.strip("[]").split(",")]
    numeric = [_parse_scif_numeric_token(part) for part in parts]
    return {
        "raw": normalized,
        "components": parts,
        "numeric_components": numeric if all(item is not None for item in numeric) else None,
    }


def _parse_scif_cartesian_list(value):
    normalized = _normalize_scif_scalar(value)
    if normalized is None:
        return None
    parts = normalized.strip("[]").split()
    numeric = [_parse_scif_numeric_token(part) for part in parts]
    return {
        "raw": normalized,
        "components": parts,
        "numeric_components": numeric if all(item is not None for item in numeric) else None,
    }


def _parse_scif_matrix_list(value):
    normalized = _normalize_scif_scalar(value)
    if normalized is None:
        return None
    try:
        parsed = ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        row_strings = re.findall(r"\[([^\[\]]+)\]", normalized)
        if len(row_strings) != 3:
            return {"raw": normalized, "numeric_components": None}
        rows = []
        try:
            for row in row_strings:
                parts = [part.strip() for part in row.split(",")]
                if len(parts) != 3:
                    return {"raw": normalized, "numeric_components": None}
                rows.append([evaluate_numeric_expression(part) for part in parts])
        except Exception:
            return {"raw": normalized, "numeric_components": None}
        return {
            "raw": normalized,
            "numeric_components": rows,
        }
    matrix = np.asarray(parsed, dtype=float)
    if matrix.shape != (3, 3):
        return {"raw": normalized, "numeric_components": None}
    return {
        "raw": normalized,
        "numeric_components": matrix.tolist(),
    }


def _parse_optional_numeric_scalar(value):
    normalized = _normalize_scif_scalar(value)
    if normalized is None:
        return None
    numeric = _parse_scif_numeric_token(normalized)
    return numeric if numeric is not None else normalized


def _extract_cif_metadata(data: dict):
    parent_name_key, parent_name_value = _get_first_existing_with_key(
        data,
        [
            '_parent_space_group.name_H-M_alt',
            '_parent_space_group.name_H-M',
        ],
    )
    parent_number_key, parent_number_value = _get_first_existing_with_key(
        data,
        [
            '_parent_space_group.IT_number',
        ],
    )
    parent_transform_key, parent_transform_value = _get_first_existing_with_key(
        data,
        [
            '_parent_space_group.transform_Pp_abc',
            '_parent_space_group.transform_to_standard_Pp_abc',
        ],
    )
    parent_child_transform_key, parent_child_transform_value = _get_first_existing_with_key(
        data,
        [
            '_parent_space_group.child_transform_Pp_abc',
        ],
    )

    cell_parameter_keys = [
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',
    ]
    cell_parameter_strings = {
        key: _normalize_scif_scalar(data.get(key))
        for key in cell_parameter_keys
    }

    return {
        "cell_parameter_strings": cell_parameter_strings,
        "parent_space_group": {
            "name_H_M_alt": _normalize_scif_scalar(parent_name_value),
            "IT_number": _parse_optional_numeric_scalar(parent_number_value),
            "transform_Pp_abc": _normalize_scif_scalar(parent_transform_value),
            "child_transform_Pp_abc": _normalize_scif_scalar(parent_child_transform_value),
            "source_tags": {
                "name_H_M_alt": parent_name_key,
                "IT_number": parent_number_key,
                "transform_Pp_abc": parent_transform_key,
                "child_transform_Pp_abc": parent_child_transform_key,
            },
        },
        "raw_cif_tags": copy.deepcopy(data),
    }


def _extract_scif_metadata(data: dict):
    parent_name_key, parent_name_value = _get_first_existing_with_key(
        data,
        [
            '_parent_space_group.name_H-M_alt',
            '_parent_space_group.name_H-M',
        ],
    )
    parent_number_key, parent_number_value = _get_first_existing_with_key(
        data,
        ['_parent_space_group.IT_number'],
    )
    parent_transform_key, parent_transform_value = _get_first_existing_with_key(
        data,
        [
            '_parent_space_group.transform_Pp_abc',
            '_parent_space_group.transform_to_standard_Pp_abc',
        ],
    )
    parent_child_transform_key, parent_child_transform_value = _get_first_existing_with_key(
        data,
        ['_parent_space_group.child_transform_Pp_abc'],
    )
    collinear_key, collinear_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.collinear_direction',
            '_space_group_spin.collinear_direction_xyz',
        ],
    )
    coplanar_key, coplanar_value = _get_first_existing_with_key(
        data,
        ['_space_group_spin.coplanar_perp_uvw'],
    )
    ssg_name_key, ssg_name_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.name_Chen_Liu',
            '_space_group_spin.name_Chen',
            '_space_group_spin.spin_space_group_name_Chen_Liu',
            '_space_group_spin.spin_space_group_name_Chen',
            '_space_group_spin.name_SpSG_Chen_Liu',
            '_space_group_spin.name_SpSG_Chen',
        ],
    )
    ssg_name_linear_key, ssg_name_linear_value = _get_first_existing_with_key(
        data,
        _repo_local_scif_tag_candidates(
            'oriented_spin_space_group_name_linear',
            'spin_space_group_name_linear',
        ),
    )
    ssg_number_key, ssg_number_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.number_Chen_Liu',
            '_space_group_spin.number_Chen',
            '_space_group_spin.spin_space_group_number_Chen_Liu',
            '_space_group_spin.spin_space_group_number_Chen',
            '_space_group_spin.number_SpSG_Chen_Liu',
            '_space_group_spin.number_SpSG_Chen',
        ],
    )
    transform_chen_key, transform_chen_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.transform_Chen_Liu_Pp_abcs',
            '_space_group_spin.transform_Chen_Pp_abcs',
        ],
    )
    rotation_axis_key, rotation_axis_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.rotation_axis_xyz',
            '_space_group_spin.rotation_axis',
        ],
    )
    rotation_axis_cartn_key, rotation_axis_cartn_value = _get_first_existing_with_key(
        data,
        ['_space_group_spin.rotation_axis_cartn'],
    )
    spin_space_point_group_key, spin_space_point_group_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.spin_space_point_group_name',
            *_repo_local_scif_tag_candidates('spin_space_point_group_name'),
        ],
    )
    transform_spinframe_abc_key, transform_spinframe_abc_value = _get_first_existing_with_key(
        data,
        ['_space_group_spin.transform_spinframe_P_abc'],
    )
    transform_spinframe_matrix_key, transform_spinframe_matrix_value = _get_first_existing_with_key(
        data,
        [
            '_space_group_spin.transform_spinframe_P_matrix',
            '_space_group_spin.tansform_spinframe_P_matrix',
        ],
    )

    operation_ids = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_operation.id',
            '_space_group_symop_spin_operation_id',
        ],
    )
    operation_xyzt = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_operation.xyzt',
            '_space_group_symop_spin_operation_xyzt',
        ],
    )
    operation_uvw = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_operation.uvw',
            '_space_group_symop_spin_operation_uvw',
        ],
    )
    operation_uvw_id = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_operation.uvw_id',
            '_space_group_symop_spin_operation_uvw_id',
        ],
    )

    lattice_ids = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_lattice.id',
            '_space_group_symop_spin_lattice_id',
        ],
    )
    lattice_xyzt = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_lattice.xyzt',
            '_space_group_symop_spin_lattice_xyzt',
        ],
    )
    lattice_uvw = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_lattice.uvw',
            '_space_group_symop_spin_lattice_uvw',
        ],
    )
    lattice_uvw_id = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_lattice.uvw_id',
            '_space_group_symop_spin_lattice_uvw_id',
        ],
    )

    spin_labels = _get_first_existing(
        data,
        [
            '_atom_site_spin_moment.label',
            '_atom_site_spin_moment_label',
        ],
    )
    spin_symmform = _get_first_existing(
        data,
        [
            '_atom_site_spin_moment.symmform_uvw',
            '_atom_site_spin_moment_symmform_uvw',
        ],
    )
    spin_symmform_rel = _get_first_existing(
        data,
        [
            '_atom_site_spin_moment.symmform_rel_uvw',
            '_atom_site_spin_moment_symmform_rel_uvw',
        ],
    )
    spin_magnitude = _get_first_existing(
        data,
        [
            '_atom_site_spin_moment.magnitude',
            '_atom_site_spin_moment_magnitude',
        ],
    )
    orbital_labels = _get_first_existing(
        data,
        [
            '_atom_site_orbital_moment.label',
            '_atom_site_orbital_moment_label',
        ],
    )
    orbital_x = _get_first_existing(
        data,
        [
            '_atom_site_orbital_moment.crystalaxis_x',
            '_atom_site_orbital_moment_crystalaxis_x',
            '_atom_site_orbital_moment.axis_x',
            '_atom_site_orbital_moment_axis_x',
        ],
    )
    orbital_y = _get_first_existing(
        data,
        [
            '_atom_site_orbital_moment.crystalaxis_y',
            '_atom_site_orbital_moment_crystalaxis_y',
            '_atom_site_orbital_moment.axis_y',
            '_atom_site_orbital_moment_axis_y',
        ],
    )
    orbital_z = _get_first_existing(
        data,
        [
            '_atom_site_orbital_moment.crystalaxis_z',
            '_atom_site_orbital_moment_crystalaxis_z',
            '_atom_site_orbital_moment.axis_z',
            '_atom_site_orbital_moment_axis_z',
        ],
    )
    orbital_symmform = _get_first_existing(
        data,
        [
            '_atom_site_orbital_moment.symmform_xyz',
            '_atom_site_orbital_moment_symmform_xyz',
        ],
    )
    orbital_magnitude = _get_first_existing(
        data,
        [
            '_atom_site_orbital_moment.magnitude',
            '_atom_site_orbital_moment_magnitude',
        ],
    )

    repo_local_extensions = {
        "spin_space_group_name_linear": _normalize_scif_scalar(ssg_name_linear_value),
        "spin_space_group_name_latex": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates(
                    'oriented_spin_space_group_name_latex',
                    'spin_space_group_name_latex',
                ),
            )
        ),
        "magnetic_phase": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('magnetic_phase'),
            )
        ),
        "parent_space_group_status": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('parent_space_group_status'),
            )
        ),
        "parent_space_group_matches_input": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('parent_space_group_matches_input'),
            )
        ),
        "input_parent_space_group_name_H_M_alt": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('input_parent_space_group_name_H-M_alt'),
            )
        ),
        "input_parent_space_group_IT_number": _parse_optional_numeric_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('input_parent_space_group_IT_number'),
            )
        ),
        "G0_number": _parse_optional_numeric_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('G0_number'),
            )
        ),
        "L0_number": _parse_optional_numeric_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('L0_number'),
            )
        ),
        "it": _parse_optional_numeric_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('it'),
            )
        ),
        "ik": _parse_optional_numeric_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('ik'),
            )
        ),
        "spin_space_point_group_name": _normalize_scif_scalar(spin_space_point_group_value),
        "spin_part_point_group": (
            _normalize_scif_scalar(spin_space_point_group_value)
            if _normalize_scif_scalar(spin_space_point_group_value) is not None
            else _normalize_scif_scalar(
                _get_first_existing(
                    data,
                    _repo_local_scif_tag_candidates('spin_part_point_group'),
                )
            )
        ),
        "transform_to_input_Pp": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('transform_to_input_Pp'),
            )
        ),
        "transform_to_magnetic_primitive_Pp": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('transform_to_magnetic_primitive_Pp'),
            )
        ),
        "transform_to_L0std_Pp": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('transform_to_L0std_Pp'),
            )
        ),
        "transform_to_G0std_Pp": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('transform_to_G0std_Pp'),
            )
        ),
        "transform_to_parent_space_group_Pp": _normalize_scif_scalar(
            _get_first_existing(
                data,
                _repo_local_scif_tag_candidates('transform_to_parent_space_group_Pp'),
            )
        ),
    }

    return {
        "cell_parameter_strings": {
            "_cell_length_a": _normalize_scif_scalar(data.get('_cell_length_a')),
            "_cell_length_b": _normalize_scif_scalar(data.get('_cell_length_b')),
            "_cell_length_c": _normalize_scif_scalar(data.get('_cell_length_c')),
            "_cell_angle_alpha": _normalize_scif_scalar(data.get('_cell_angle_alpha')),
            "_cell_angle_beta": _normalize_scif_scalar(data.get('_cell_angle_beta')),
            "_cell_angle_gamma": _normalize_scif_scalar(data.get('_cell_angle_gamma')),
        },
        "parent_space_group": {
            "name_H_M_alt": _normalize_scif_scalar(parent_name_value),
            "IT_number": _parse_optional_numeric_scalar(parent_number_value),
            "transform_Pp_abc": _normalize_scif_scalar(parent_transform_value),
            "child_transform_Pp_abc": _normalize_scif_scalar(parent_child_transform_value),
            "source_tags": {
                "name_H_M_alt": parent_name_key,
                "IT_number": parent_number_key,
                "transform_Pp_abc": parent_transform_key,
                "child_transform_Pp_abc": parent_child_transform_key,
            },
        },
        "space_group_spin": {
            "collinear_direction": _parse_scif_vector_string(collinear_value),
            "coplanar_perp_uvw": _parse_scif_vector_string(coplanar_value),
            "spin_space_group_name_chen": _normalize_scif_scalar(ssg_name_value),
            "spin_space_point_group_name": _normalize_scif_scalar(spin_space_point_group_value),
            "spin_space_group_name_linear": (
                repo_local_extensions["spin_space_group_name_linear"]
                if repo_local_extensions["spin_space_group_name_linear"] is not None
                else _normalize_scif_scalar(ssg_name_value)
            ),
            "spin_space_group_name_latex": repo_local_extensions["spin_space_group_name_latex"],
            "spin_space_group_number_chen": _normalize_scif_scalar(ssg_number_value),
            "transform_Chen_Pp_abcs": _normalize_scif_scalar(transform_chen_value),
            "rotation_angle": _parse_optional_numeric_scalar(data.get('_space_group_spin.rotation_angle')),
            "rotation_axis_xyz": _parse_scif_vector_string(rotation_axis_value),
            "rotation_axis_cartn": _parse_scif_cartesian_list(rotation_axis_cartn_value),
            "transform_spinframe_P_abc": _normalize_scif_scalar(transform_spinframe_abc_value),
            "transform_spinframe_P_matrix": _parse_scif_matrix_list(transform_spinframe_matrix_value),
            "G0_number": repo_local_extensions["G0_number"],
            "L0_number": repo_local_extensions["L0_number"],
            "it": repo_local_extensions["it"],
            "ik": repo_local_extensions["ik"],
            "spin_space_point_group_name": repo_local_extensions["spin_space_point_group_name"],
            "spin_part_point_group": repo_local_extensions["spin_part_point_group"],
            "transform_to_input_Pp": repo_local_extensions["transform_to_input_Pp"],
            "transform_to_magnetic_primitive_Pp": repo_local_extensions["transform_to_magnetic_primitive_Pp"],
            "transform_to_L0std_Pp": repo_local_extensions["transform_to_L0std_Pp"],
            "transform_to_G0std_Pp": repo_local_extensions["transform_to_G0std_Pp"],
            "transform_to_parent_space_group_Pp": repo_local_extensions["transform_to_parent_space_group_Pp"],
            "magnetic_phase": repo_local_extensions["magnetic_phase"],
            "parent_space_group_status": repo_local_extensions["parent_space_group_status"],
            "parent_space_group_matches_input": repo_local_extensions["parent_space_group_matches_input"],
            "input_parent_space_group_name_H_M_alt": repo_local_extensions["input_parent_space_group_name_H_M_alt"],
            "input_parent_space_group_IT_number": repo_local_extensions["input_parent_space_group_IT_number"],
            "repo_local_extensions": repo_local_extensions,
            "source_tags": {
                "collinear_direction": collinear_key,
                "coplanar_perp_uvw": coplanar_key,
                "spin_space_group_name_chen": ssg_name_key,
                "spin_space_group_name_linear": ssg_name_linear_key or ssg_name_key,
                "spin_space_group_name_latex": _get_first_existing_with_key(
                    data,
                    _repo_local_scif_tag_candidates(
                        'oriented_spin_space_group_name_latex',
                        'spin_space_group_name_latex',
                    ),
                )[0],
                "spin_space_group_number_chen": ssg_number_key,
                "transform_Chen_Pp_abcs": transform_chen_key,
                "rotation_axis_xyz": rotation_axis_key,
                "rotation_axis_cartn": rotation_axis_cartn_key,
                "spin_space_point_group_name": spin_space_point_group_key,
                "transform_spinframe_P_abc": transform_spinframe_abc_key,
                "transform_spinframe_P_matrix": transform_spinframe_matrix_key,
            },
        },
        "space_group_symop_spin_operation": {
            "id": operation_ids,
            "xyzt": operation_xyzt,
            "uvw": operation_uvw,
            "uvw_id": operation_uvw_id,
        },
        "space_group_symop_spin_lattice": {
            "id": lattice_ids,
            "xyzt": lattice_xyzt,
            "uvw": lattice_uvw,
            "uvw_id": lattice_uvw_id,
        },
        "atom_site_spin_moment": {
            "label": spin_labels,
            "axis_u": _get_first_existing(data, ['_atom_site_spin_moment.axis_u', '_atom_site_spin_moment_axis_u']),
            "axis_v": _get_first_existing(data, ['_atom_site_spin_moment.axis_v', '_atom_site_spin_moment_axis_v']),
            "axis_w": _get_first_existing(data, ['_atom_site_spin_moment.axis_w', '_atom_site_spin_moment_axis_w']),
            "symmform_uvw": spin_symmform,
            "symmform_rel_uvw": spin_symmform_rel,
            "magnitude": spin_magnitude,
        },
        "atom_site_orbital_moment": {
            "label": orbital_labels,
            "crystalaxis_x": orbital_x,
            "crystalaxis_y": orbital_y,
            "crystalaxis_z": orbital_z,
            "symmform_xyz": orbital_symmform,
            "magnitude": orbital_magnitude,
        },
        "raw_scif_tags": copy.deepcopy(data),
    }


def parse_scif_metadata(filename=None, *, source_text=None):
    return _extract_scif_metadata(ScifParser(filename, source_text=source_text).parse())

def parse_cif_metadata(filename=None, *, source_text=None):
    return _extract_cif_metadata(CifParser(filename, source_text=source_text).parse())


def parse_cif_file(filename, atol = 0.01, return_metadata=False):
    """

    Parameters:
        filename : byte
    Returns:
        Tuple containing:
        - latticefactors (np.ndarray): Array of lattice parameters [a, b, c, alpha, beta, gamma].
        - all_positions (list of np.ndarray): List of atomic positions in fractional coordinates.
        - all_elements (list of str): List of atomic species.
        - all_occupancies (list of float): List of atomic occupancies.
        - all_labels (list of str): List of atomic labels.
        - all_moments (list of np.ndarray): List of atomic magnetic moments. ( in lattice )
    """

    data = CifParser(filename).parse()
    metadata = _extract_cif_metadata(data)


    if all([i in data for i in ['_cell_length_a','_cell_length_b','_cell_length_c','_cell_angle_alpha','_cell_angle_beta','_cell_angle_gamma']]) :
        a = convert_string_to_float(data['_cell_length_a'])
        b = convert_string_to_float(data['_cell_length_b'])
        c = convert_string_to_float(data['_cell_length_c'])
        alpha = convert_string_to_float(data['_cell_angle_alpha'])
        beta = convert_string_to_float(data['_cell_angle_beta'])
        gamma = convert_string_to_float(data['_cell_angle_gamma'])
        latticefactors = np.array([a,b,c,alpha,beta,gamma])
    else:
        raise ValueError("CIF file missing cell parameters.")

    if all([i in data for i in ['_atom_site_fract_x','_atom_site_fract_y','_atom_site_fract_z']]):
        x = data['_atom_site_fract_x']
        y = data['_atom_site_fract_y']
        z = data['_atom_site_fract_z']
        if not (len(x) == len(y) == len(z)):
            raise ValueError("Inconsistent lengths for atomic positions.")
        initial_positions = np.array([[convert_string_to_float(xi), convert_string_to_float(yi), convert_string_to_float(zi)] for xi, yi, zi in zip(x, y, z)])
    else:
        raise ValueError("CIF file missing atomic position data.")

    if '_atom_site_occupancy' in data:
        initial_occupancy = [convert_string_to_float(occ) for occ in data['_atom_site_occupancy']]
        if len(initial_occupancy) != len(initial_positions):
            raise ValueError("Inconsistent lengths for occupancy and positions.")
    else:
        initial_occupancy = [1.0] * len(initial_positions)

    if '_atom_site_type_symbol' in data:
        initial_elements = data['_atom_site_type_symbol']
        if len(initial_elements) != len(initial_positions):
            raise ValueError("Inconsistent lengths for types and positions.")
    else:
        raise ValueError("CIF file missing atomic type data.")

    if '_atom_site_label' in data:
        initial_labels = data['_atom_site_label']
        if len(initial_labels) != len(initial_positions):
            raise ValueError("Inconsistent lengths for labels and positions.")
    else:
        initial_labels = [f"{initial_elements[i]}_{i+1}" for i in range(len(initial_positions))]


    label_keys = [
        '_atom_site_moment.label',
        '_atom_site_moment_label',
    ]

    mx_keys = [
        '_atom_site_moment.crystalaxis_x',
        '_atom_site_moment_crystalaxis_x',
    ]

    my_keys = [
        '_atom_site_moment.crystalaxis_y',
        '_atom_site_moment_crystalaxis_y',
    ]

    mz_keys = [
        '_atom_site_moment.crystalaxis_z',
        '_atom_site_moment_crystalaxis_z',
    ]


    moment_labels = _get_first_existing(data, label_keys)
    mx_list = _get_first_existing(data, mx_keys)
    my_list = _get_first_existing(data, my_keys)
    mz_list = _get_first_existing(data, mz_keys)

    # see if all data are available
    if all(v is not None for v in [moment_labels, mx_list, my_list, mz_list]):
        initial_moments = []

        for lbl in initial_labels:
            if lbl in moment_labels:
                idx = moment_labels.index(lbl)
                mx = convert_string_to_float(mx_list[idx])
                my = convert_string_to_float(my_list[idx])
                mz = convert_string_to_float(mz_list[idx])
                initial_moments.append(np.array([mx, my, mz]))
            else:
                initial_moments.append(np.array([0.0, 0.0, 0.0]))
    else:
        initial_moments = [np.array([0.0, 0.0, 0.0])] * len(initial_positions)

    mag_op_keys = [
        '_space_group_symop_magn_operation.xyz',
        '_space_group_symop_magn_operation_xyz',
        '_space_group_symop.magn_operation_xyz',
        '_space_group_symop_operation_xyz',
        '_space_group_symop.operation_xyz',
        '_space_group_symop_operation.xyz',
        '_symmetry_equiv_pos_as_xyz',
    ]
    symops = _get_first_existing(data, mag_op_keys)
    if symops is None:
        raise ValueError("CIF file missing symmetry operations.")

    mag_op_centering_keys = [
        '_space_group_symop_magn_centering.xyz',
        '_space_group_symop_magn_centering_xyz',
        '_space_group_symop.magn_centering_xyz',
    ]
    centering_ops = _get_first_existing(data, mag_op_centering_keys)
    if centering_ops is None:
        centering_ops = ["x, y, z, 1"]



    symops_matrices, time_reversal = general_positions_to_matrix(symops)
    certering_ops_matrices, centering_time_reversal = general_positions_to_matrix(centering_ops)

    # generate all atoms
    all_positions = []
    all_elements = []
    all_occupancies = []
    all_labels = []
    all_moments = []
    for pos, elem, occ, label, moment in sorted(zip(initial_positions, initial_elements, initial_occupancy, initial_labels,initial_moments),key=lambda x: [abs(i) for i in x[-1]],reverse=True):
        moment_inlattice = np.array([moment[0]/a, moment[1]/b, moment[2]/c])
        for op_index,op in enumerate(symops_matrices):
            for op_c_index,op_c in enumerate(certering_ops_matrices):
                new_pos = op[0] @ op_c[0]@ pos + op[1] + op_c[1]
                new_pos = new_pos % 1.0  # Ensure within [0,1)
                tr = time_reversal[op_index] * centering_time_reversal[op_c_index]

                same = False
                for old_index,old_pos in enumerate(all_positions):
                    if are_positions_equivalent(new_pos, old_pos) and all_elements[old_index] == elem and np.allclose(occ,all_occupancies[old_index],atol=0.000001): # deduplicate, same position & same element & same occupancy
                        same = True
                        break
                if same :
                    continue
                else:
                    all_positions.append(new_pos)
                    all_elements.append(elem)
                    all_occupancies.append(occ)
                    all_labels.append(label)
                    after_moment = round(np.linalg.det(op[0]))*op[0]*tr @ moment_inlattice
                    final_moment = np.array([after_moment[0]*a,after_moment[1]*b,after_moment[2]*c])
                    all_moments.append(final_moment)
    parsed = (latticefactors,all_positions, all_elements, all_occupancies, all_labels, all_moments)
    if return_metadata:
        return parsed, metadata
    return parsed


def parse_scif_file(filename=None, atol=0.02, return_metadata=False, *, source_text=None):
    data = ScifParser(filename, source_text=source_text).parse()
    metadata = _extract_scif_metadata(data)

    if all(
        key in data
        for key in [
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_cell_angle_alpha',
            '_cell_angle_beta',
            '_cell_angle_gamma',
        ]
    ):
        a = convert_string_to_float(data['_cell_length_a'])
        b = convert_string_to_float(data['_cell_length_b'])
        c = convert_string_to_float(data['_cell_length_c'])
        alpha = convert_string_to_float(data['_cell_angle_alpha'])
        beta = convert_string_to_float(data['_cell_angle_beta'])
        gamma = convert_string_to_float(data['_cell_angle_gamma'])
        latticefactors = np.array([a, b, c, alpha, beta, gamma])
    else:
        raise ValueError("SCIF file missing cell parameters.")

    atom_keys = [
        '_atom_site_label',
        '_atom_site_type_symbol',
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',
    ]
    if not all(key in data for key in atom_keys):
        raise ValueError("SCIF file missing atomic position data.")

    initial_labels = data['_atom_site_label']
    initial_elements = data['_atom_site_type_symbol']
    x = data['_atom_site_fract_x']
    y = data['_atom_site_fract_y']
    z = data['_atom_site_fract_z']
    initial_positions = np.array(
        [
            [convert_string_to_float(xi), convert_string_to_float(yi), convert_string_to_float(zi)]
            for xi, yi, zi in zip(x, y, z)
        ]
    )

    if '_atom_site_occupancy' in data:
        initial_occupancy = [convert_string_to_float(occ) for occ in data['_atom_site_occupancy']]
    else:
        initial_occupancy = [1.0] * len(initial_positions)

    spin_label_keys = [
        '_atom_site_spin_moment.label',
        '_atom_site_spin_moment_label',
    ]
    su_keys = [
        '_atom_site_spin_moment.axis_u',
        '_atom_site_spin_moment_axis_u',
    ]
    sv_keys = [
        '_atom_site_spin_moment.axis_v',
        '_atom_site_spin_moment_axis_v',
    ]
    sw_keys = [
        '_atom_site_spin_moment.axis_w',
        '_atom_site_spin_moment_axis_w',
    ]

    spin_labels = _get_first_existing(data, spin_label_keys)
    su_list = _get_first_existing(data, su_keys)
    sv_list = _get_first_existing(data, sv_keys)
    sw_list = _get_first_existing(data, sw_keys)

    moments_by_label = {}
    if all(v is not None for v in [spin_labels, su_list, sv_list, sw_list]):
        for lbl, su, sv, sw in zip(spin_labels, su_list, sv_list, sw_list):
            moments_by_label[lbl] = np.array(
                [
                    convert_string_to_float(su),
                    convert_string_to_float(sv),
                    convert_string_to_float(sw),
                ],
                dtype=float,
            )

    initial_moments = [moments_by_label.get(lbl, np.array([0.0, 0.0, 0.0])) for lbl in initial_labels]

    transform_spinframe_abc = _normalize_scif_scalar(
        _get_first_existing(data, ['_space_group_spin.transform_spinframe_P_abc'])
    )
    transform_spinframe_matrix = _parse_scif_matrix_list(
        _get_first_existing(
            data,
            [
                '_space_group_spin.transform_spinframe_P_matrix',
                '_space_group_spin.tansform_spinframe_P_matrix',
            ],
        )
    )
    spinframe_lengths_scale = None
    normalized_spinframe = None if transform_spinframe_abc is None else re.sub(r"\s+", "", transform_spinframe_abc)
    transform_spinframe_numeric = (
        None
        if transform_spinframe_matrix is None
        else transform_spinframe_matrix.get("numeric_components")
    )
    if normalized_spinframe == "a,b,c":
        spinframe_lengths_scale = np.diag(np.asarray(latticefactors[:3], dtype=float))
    elif (
        transform_spinframe_numeric is not None
        and np.allclose(np.asarray(transform_spinframe_numeric, dtype=float), np.eye(3), atol=1e-9)
    ):
        spinframe_lengths_scale = np.diag(np.asarray(latticefactors[:3], dtype=float))
    spinframe_lengths_scale_inv = (
        None if spinframe_lengths_scale is None else np.linalg.inv(spinframe_lengths_scale)
    )

    symop_xyzt = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_operation.xyzt',
            '_space_group_symop_spin_operation_xyzt',
        ],
    )
    symop_uvw = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_operation.uvw',
            '_space_group_symop_spin_operation_uvw',
        ],
    )
    if symop_xyzt is None or symop_uvw is None:
        raise ValueError("SCIF file missing spin-space-group operations.")

    spin_lattice_xyzt = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_lattice.xyzt',
            '_space_group_symop_spin_lattice_xyzt',
        ],
    )
    spin_lattice_uvw = _get_first_existing(
        data,
        [
            '_space_group_symop_spin_lattice.uvw',
            '_space_group_symop_spin_lattice_uvw',
        ],
    )
    if spin_lattice_xyzt is None:
        spin_lattice_xyzt = ['x, y, z, 1']
    if spin_lattice_uvw is None:
        spin_lattice_uvw = ['u, v, w']

    symop_real, _ = general_positions_to_matrix(symop_xyzt)
    symop_spin, _ = general_positions_to_matrix(symop_uvw, variables=('u', 'v', 'w'))
    lattice_real, _ = general_positions_to_matrix(spin_lattice_xyzt)
    lattice_spin, _ = general_positions_to_matrix(spin_lattice_uvw, variables=('u', 'v', 'w'))

    if not (len(symop_real) == len(symop_spin)):
        raise ValueError("Inconsistent SCIF spin-operation loop lengths.")
    if not (len(lattice_real) == len(lattice_spin)):
        raise ValueError("Inconsistent SCIF spin-lattice loop lengths.")

    all_positions = []
    all_elements = []
    all_occupancies = []
    all_labels = []
    all_moments = []

    for pos, elem, occ, label, moment in zip(
        initial_positions,
        initial_elements,
        initial_occupancy,
        initial_labels,
        initial_moments,
    ):
        for (real_op, real_shift), (spin_op, _) in zip(symop_real, symop_spin):
            for (lat_real_op, lat_real_shift), (lat_spin_op, _) in zip(lattice_real, lattice_spin):
                # Full SSG operations are reconstructed as non-centered NSSG op @ spin-lattice op.
                # The lattice translation must therefore be acted on by the real-space rotation.
                new_pos = real_op @ (lat_real_op @ pos + lat_real_shift) + real_shift
                new_pos = new_pos % 1.0
                if spinframe_lengths_scale is None:
                    new_moment = spin_op @ lat_spin_op @ np.asarray(moment, dtype=float)
                else:
                    # For the current oriented 'a,b,c' contract, uvw acts on
                    # relative components in the spin basis while atom moments
                    # are stored as absolute components along the corresponding
                    # basis directions.
                    new_moment = (
                        spinframe_lengths_scale
                        @ spin_op
                        @ lat_spin_op
                        @ spinframe_lengths_scale_inv
                        @ np.asarray(moment, dtype=float)
                    )

                same = False
                for old_index, old_pos in enumerate(all_positions):
                    if (
                        are_positions_equivalent(new_pos, old_pos)
                        and all_elements[old_index] == elem
                        and np.allclose(occ, all_occupancies[old_index], atol=1e-6)
                    ):
                        if not np.allclose(
                            new_moment,
                            all_moments[old_index],
                            atol=max(atol, 1e-6),
                        ):
                            raise ValueError(
                                "SCIF expansion produced inconsistent moments for the same atomic site. "
                                "Check spin-operation/spin-lattice composition order. "
                                "Suggested direction: first retry with "
                                "`find_spin_group(..., parser_atol=...)` or "
                                "`parse_scif_file(..., atol=...)` / `parse_scif_text(..., atol=...)`; "
                                "if you need the parsed-data flow, parse first and then call "
                                "`find_spin_group_from_data(...)`. "
                                "Adjust SCIF atom/moment precision before tuning point-group parameters."
                            )
                        same = True
                        break
                if same:
                    continue

                all_positions.append(new_pos)
                all_elements.append(elem)
                all_occupancies.append(occ)
                all_labels.append(label)
                all_moments.append(np.asarray(new_moment, dtype=float))

    parsed = (latticefactors, all_positions, all_elements, all_occupancies, all_labels, all_moments)
    if return_metadata:
        return parsed, metadata
    return parsed


def parse_scif_text(text, atol=0.02, return_metadata=False):
    return parse_scif_file(None, atol=atol, return_metadata=return_metadata, source_text=text)
