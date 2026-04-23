import json
import re
import warnings
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from spintensor import solve_ahe, solve_bcd, solve_imd, solve_qmd
from spglib import get_symmetry_dataset,get_magnetic_symmetry_dataset
from findspingroup.core.identify_symmetry_from_ops import (
    deduplicate_matrix_pairs,
    get_magnetic_space_group_from_operations,
    identify_point_group,
)
from findspingroup.core.identify_spin_space_group import (
    identify_spin_space_group_result,
)
from findspingroup.core.identify_index.contract_222 import (
    get_coplanar_222_lookup_entry,
    has_coplanar_222_lookup_group,
)
from findspingroup.core.tolerances import DEFAULT_TOL, Tolerances
from findspingroup.data import MSGMPG_DB
from findspingroup.io import parse_poscar_file, parse_structure_file
from findspingroup.io.scif_generator import (
    _format_scif_symbolic_scalar,
    affine_matrix_to_xyz_expression,
    generate_scif,
)
from findspingroup.structure import SpinSpaceGroup,SpinSpaceGroupOperation
from findspingroup.structure.group import integer_points_in_new_cell, op_key
from findspingroup.structure.cell import (
    CrystalCell,
    calculate_vector_coordinates_from_latticefactors,
    standardize_lattice,
)
from findspingroup.data.PG_SYMBOL import PG_IF_HEX_MAPPING, SG_HALL_MAPPING
from findspingroup.utils.matrix_utils import rref_with_tolerance, normalize_vector_to_zero
from findspingroup.utils.symbolic_format import (
    format_symbolic_scalar,
    symbolize_numeric_tokens_in_string,
)
from findspingroup.utils.space_group_flags import (
    msg_parent_space_group_info,
    space_group_has_real_space_inversion,
    space_group_is_chiral,
    space_group_is_polar,
)
from findspingroup.utils.seitz_symbol import canonicalize_group_seitz_descriptions
from findspingroup.version import __version__


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def _format_spin_only_direction(direction) -> str:
    if direction is None:
        return ""
    values = []
    for value in np.asarray(direction, dtype=float).reshape(-1):
        if abs(value) < 1e-4:
            value = 0.0
        values.append(_format_scif_symbolic_scalar(float(value), decimal_precision=6))
    return ",".join(values)


ACC_PRIMITIVE_SETTING = "acc_primitive"
ACC_CONVENTIONAL_SETTING = "acc_conventional"
INPUT_MAGNETIC_PRIMITIVE_SETTING = "input_magnetic_primitive"
INPUT_POSCAR_SETTING = "input_poscar"
ACC_PRIMITIVE_CARTESIAN_SETTING = "acc_primitive_cartesian"
ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING = "acc_primitive_poscar_spin_frame"
OSSG_ORIENTED_SPIN_FRAME_SETTING = "ossg_oriented_spin_frame"
G0_STANDARD_SETTING = "G0std"
L0_STANDARD_SETTING = "L0std"
SCIF_CELL_MODE_INPUT = "input"
SCIF_CELL_MODE_MAGNETIC_PRIMITIVE = "magnetic_primitive"
SCIF_CELL_MODE_G0STD_ORIENTED = "g0std_oriented"
def _should_degrade_identify_index_error(error: Exception) -> bool:
    message = str(error)
    return (
        message.startswith("No identify-index reduction record for ")
        or message.startswith("Cannot identify point-group map number for ")
    )


def _exact_translation_distance(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _ops_match_with_exact_translation(
    left: SpinSpaceGroupOperation,
    right: SpinSpaceGroupOperation,
    tol: float,
) -> bool:
    return (
        np.allclose(np.asarray(left[0], dtype=float), np.asarray(right[0], dtype=float), atol=tol)
        and np.allclose(np.asarray(left[1], dtype=float), np.asarray(right[1], dtype=float), atol=tol)
        and _exact_translation_distance(left[2], right[2]) < tol
    )


def _deduplicate_ops_with_exact_translation(
    ops: list[SpinSpaceGroupOperation],
    tol: float,
) -> list[SpinSpaceGroupOperation]:
    ordered_ops = sorted(ops, key=op_key)
    unique_ops: list[SpinSpaceGroupOperation] = []
    for op in ordered_ops:
        if any(_ops_match_with_exact_translation(op, existing, tol) for existing in unique_ops):
            continue
        unique_ops.append(op)
    return unique_ops


def _deduplicate_translation_vectors_exact(translations, tol: float) -> list[np.ndarray]:
    unique_translations: list[np.ndarray] = []
    for translation in translations:
        vector = np.asarray(translation, dtype=float)
        if any(_exact_translation_distance(vector, existing) < tol for existing in unique_translations):
            continue
        unique_translations.append(vector)
    return unique_translations


def _translations_equivalent_mod_pure_translations(
    left,
    right,
    pure_translations,
    tol: float,
) -> bool:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    if _exact_translation_distance(left_vector, right_vector) < tol:
        return True

    difference = left_vector - right_vector
    for pure_translation in pure_translations:
        if getNormInf(difference, np.asarray(pure_translation, dtype=float), mode=True) < tol:
            return True
    return False


def _translation_equivalent_mod_integer(left, right, tol: float) -> bool:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    if _exact_translation_distance(left_vector, right_vector) < tol:
        return True
    difference = left_vector - right_vector
    nearest_integer = np.rint(difference)
    return bool(np.max(np.abs(difference - nearest_integer)) < tol)


def _real_op_bucket_decimals(tol: float) -> int:
    tol = float(max(tol, 1e-12))
    return max(0, int(np.ceil(-np.log10(tol))) - 1)


def _real_rotation_bucket_key(rotation, tol: float):
    decimals = _real_op_bucket_decimals(tol)
    arr = np.asarray(rotation, dtype=float).reshape(-1)
    return tuple(np.round(arr, decimals))


def _fractional_translation_bucket_params(tol: float):
    tol = float(max(tol, 1e-12))
    bins = max(1, int(np.ceil(1.0 / tol)))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(tol / bucket_width)))
    return bins, neighbor_radius


def _fractional_translation_bucket_key(translation, bins: int):
    wrapped = np.mod(np.asarray(translation, dtype=float), 1.0)
    indices = np.floor(wrapped * bins).astype(int) % bins
    return tuple(int(value) for value in indices)


def _fractional_translation_neighbor_keys(bucket_key, bins: int, neighbor_radius: int):
    for dx in range(-neighbor_radius, neighbor_radius + 1):
        for dy in range(-neighbor_radius, neighbor_radius + 1):
            for dz in range(-neighbor_radius, neighbor_radius + 1):
                yield (
                    (bucket_key[0] + dx) % bins,
                    (bucket_key[1] + dy) % bins,
                    (bucket_key[2] + dz) % bins,
                )


def _collect_unique_real_ops_with_spin_sets(
    ops: list[SpinSpaceGroupOperation],
    *,
    tol: float,
) -> list[dict]:
    records: list[dict] = []
    bins, neighbor_radius = _fractional_translation_bucket_params(tol)
    exact_buckets: dict[tuple, list[dict]] = {}
    for op in ops:
        spin_rotation = np.asarray(op[0], dtype=float)
        real_rotation = np.asarray(op[1], dtype=float)
        translation = np.asarray(op[2], dtype=float)

        matched = None
        rotation_key = _real_rotation_bucket_key(real_rotation, tol)
        translation_key = _fractional_translation_bucket_key(translation, bins)
        for neighbor_key in _fractional_translation_neighbor_keys(translation_key, bins, neighbor_radius):
            for record in exact_buckets.get((rotation_key, neighbor_key), ()):
                if not np.allclose(real_rotation, record["rotation"], atol=tol, rtol=0):
                    continue
                if _exact_translation_distance(translation, record["translation"]) >= tol:
                    continue
                matched = record
                break
            if matched is not None:
                break

        if matched is None:
            matched = {
                "rotation": real_rotation,
                "translation": translation,
                "spin_rotations": [],
            }
            records.append(matched)
            exact_buckets.setdefault((rotation_key, translation_key), []).append(matched)

        if not any(np.allclose(spin_rotation, existing, atol=tol, rtol=0) for existing in matched["spin_rotations"]):
            matched["spin_rotations"].append(spin_rotation)

    for record in records:
        record["spin_rotations"] = sorted(
            record["spin_rotations"],
            key=lambda matrix: tuple(np.round(np.asarray(matrix, dtype=float).flatten(), 6)),
        )
    return records


def _match_real_op_record(
    candidate: dict,
    records: list[dict],
    *,
    tol: float,
    pure_translation_vectors: list[np.ndarray],
) -> tuple[dict | None, str]:
    rotation_key = _real_rotation_bucket_key(candidate["rotation"], tol)
    rotation_candidates = [
        record
        for record in records
        if _real_rotation_bucket_key(record["rotation"], tol) == rotation_key
    ]

    for record in rotation_candidates:
        if np.allclose(candidate["rotation"], record["rotation"], atol=tol, rtol=0) and _exact_translation_distance(
            candidate["translation"], record["translation"]
        ) < tol:
            return record, "exact"

    for record in rotation_candidates:
        if np.allclose(candidate["rotation"], record["rotation"], atol=tol, rtol=0) and _translation_equivalent_mod_integer(
            candidate["translation"], record["translation"], tol
        ):
            return record, "mod_integer"

    for record in rotation_candidates:
        if not np.allclose(candidate["rotation"], record["rotation"], atol=tol, rtol=0):
            continue
        if _translations_equivalent_mod_pure_translations(
            candidate["translation"],
            record["translation"],
            pure_translation_vectors,
            tol,
        ):
            return record, "mod_pure_translation"

    return None, "none"


def audit_spatial_transform_effect(
    ssg: SpinSpaceGroup,
    transformation_matrix: np.ndarray,
    origin_shift: np.ndarray,
    *,
    tol: float = 1e-6,
    det_tol: float = 1e-2,
    use_nssg: bool = True,
) -> dict:
    """
    Audit how a spatial setting transform affects the real-space part of an SSG.

    The helper distinguishes:
    - exact real-op preservation
    - preservation mod integer lattice vectors
    - preservation mod the group's pure real-space translations
    - whether the associated spin-op set attached to a matched real op changes
    """
    source_ops = list(ssg.nssg if use_nssg else ssg.ops)
    transformation_matrix = np.asarray(transformation_matrix, dtype=float)
    origin_shift = np.asarray(origin_shift, dtype=float)
    determinant = float(np.linalg.det(transformation_matrix))
    volume_preserving = abs(abs(determinant) - 1.0) <= det_tol

    if not volume_preserving:
        return {
            "tol": float(tol),
            "det_tol": float(det_tol),
            "use_nssg": bool(use_nssg),
            "transform_matrix": transformation_matrix.tolist(),
            "origin_shift": origin_shift.tolist(),
            "determinant": determinant,
            "volume_preserving": False,
            "can_be_affine_normalizer_equivalent": False,
            "source_real_op_count": None,
            "transformed_real_op_count": None,
            "real_ops_exact_same": False,
            "real_ops_same_mod_integer": False,
            "real_ops_same_mod_pure_translations": False,
            "paired_spin_changed_count": None,
            "unmatched_source_indices": None,
            "transformed_to_source": [],
        }

    transformed_ssg = ssg.transform(transformation_matrix, origin_shift)
    transformed_ops = list(transformed_ssg.nssg if use_nssg else transformed_ssg.ops)

    source_records = _collect_unique_real_ops_with_spin_sets(source_ops, tol=tol)
    transformed_records = _collect_unique_real_ops_with_spin_sets(transformed_ops, tol=tol)
    pure_translation_vectors = [np.asarray(item[1], dtype=float) for item in ssg.pure_t_group]

    transformed_to_source = []
    exact_preserved = True
    mod_integer_preserved = True
    mod_pure_preserved = True
    paired_spin_changed_count = 0

    for transformed_index, transformed_record in enumerate(transformed_records):
        matched_record, match_kind = _match_real_op_record(
            transformed_record,
            source_records,
            tol=tol,
            pure_translation_vectors=pure_translation_vectors,
        )
        if match_kind != "exact":
            exact_preserved = False
        if match_kind not in {"exact", "mod_integer"}:
            mod_integer_preserved = False
        if match_kind not in {"exact", "mod_integer", "mod_pure_translation"}:
            mod_pure_preserved = False

        spin_set_same = None
        matched_index = None
        if matched_record is not None:
            matched_index = next(
                (idx for idx, record in enumerate(source_records) if record is matched_record),
                None,
            )
            source_spin_rotations = matched_record["spin_rotations"]
            transformed_spin_rotations = transformed_record["spin_rotations"]
            spin_set_same = (
                len(source_spin_rotations) == len(transformed_spin_rotations)
                and all(
                    np.allclose(left, right, atol=tol, rtol=0)
                    for left, right in zip(source_spin_rotations, transformed_spin_rotations)
                )
            )
            if not spin_set_same:
                paired_spin_changed_count += 1

        transformed_to_source.append(
            {
                "transformed_index": transformed_index,
                "source_index": matched_index,
                "match_kind": match_kind,
                "rotation": np.asarray(transformed_record["rotation"], dtype=float).tolist(),
                "translation": np.asarray(transformed_record["translation"], dtype=float).tolist(),
                "spin_set_same": spin_set_same,
                "source_spin_count": None if matched_record is None else len(matched_record["spin_rotations"]),
                "transformed_spin_count": len(transformed_record["spin_rotations"]),
            }
        )

    unmatched_source_indices = []
    for source_index, source_record in enumerate(source_records):
        matched = False
        for transformed_record in transformed_records:
            _, match_kind = _match_real_op_record(
                source_record,
                [transformed_record],
                tol=tol,
                pure_translation_vectors=pure_translation_vectors,
            )
            if match_kind != "none":
                matched = True
                break
        if not matched:
            unmatched_source_indices.append(source_index)

    return {
        "tol": float(tol),
        "det_tol": float(det_tol),
        "use_nssg": bool(use_nssg),
        "transform_matrix": transformation_matrix.tolist(),
        "origin_shift": origin_shift.tolist(),
        "determinant": determinant,
        "volume_preserving": True,
        "can_be_affine_normalizer_equivalent": mod_pure_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "source_real_op_count": len(source_records),
        "transformed_real_op_count": len(transformed_records),
        "real_ops_exact_same": exact_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "real_ops_same_mod_integer": mod_integer_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "real_ops_same_mod_pure_translations": mod_pure_preserved
        and len(source_records) == len(transformed_records)
        and not unmatched_source_indices,
        "paired_spin_changed_count": paired_spin_changed_count,
        "unmatched_source_indices": unmatched_source_indices,
        "transformed_to_source": transformed_to_source,
    }


def _identify_nssg_ops(conf: str, ops: list[SpinSpaceGroupOperation], tol: float) -> list[SpinSpaceGroupOperation]:
    if conf == 'Collinear':
        return [
            op for op in ops
            if np.allclose(op[0], -np.eye(3), atol=tol) or np.allclose(op[0], np.eye(3), atol=tol)
        ]
    if conf == 'Coplanar':
        return [op for op in ops if np.linalg.det(op[0]) > 0]
    return list(ops)


@dataclass
class IdentifyNoFracGroup:
    input_ops: list[SpinSpaceGroupOperation]
    conf: str
    tol: float
    ops: list[SpinSpaceGroupOperation] = field(init=False)
    spin_translation_group: list[SpinSpaceGroupOperation] = field(init=False)
    pure_translations: list[np.ndarray] = field(init=False)
    sog: list[SpinSpaceGroupOperation] = field(init=False)
    nssg: list[SpinSpaceGroupOperation] = field(init=False)
    n_spin_part_point_ops: list[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.ops = _deduplicate_ops_with_exact_translation(list(self.input_ops), self.tol)
        self.spin_translation_group = [
            op for op in self.ops if np.allclose(op[1], np.eye(3), atol=self.tol)
        ]
        self.pure_translations = _deduplicate_translation_vectors_exact(
            [
                np.asarray(op[2], dtype=float)
                for op in self.spin_translation_group
                if np.allclose(op[0], np.eye(3), atol=self.tol)
            ],
            self.tol,
        )
        if not self.pure_translations:
            self.pure_translations = [np.zeros(3)]
        self.sog = _deduplicate_ops_with_exact_translation(
            [
                op
                for op in self.spin_translation_group
                if _exact_translation_distance(op[2], np.zeros(3)) < self.tol
            ],
            self.tol,
        )
        self.nssg = _identify_nssg_ops(self.conf, self.ops, self.tol)
        self.n_spin_part_point_ops = deduplicate_matrix_pairs([op[0] for op in self.nssg], tol=0.1)


class MagSymmetryResult:
    def __init__(self, cell, symmetry, properties):
        self.input_ssg_ops = symmetry.get('input_ssg_ops', None)
        self.spin_only = symmetry.get('spin_only', None)
        self.ssg_std_cell = symmetry.get('ssg_std_cell', None)
        self.T_input_to_ssg_std = symmetry.get('T_input_to_ssg_std', None)
        self.T_input_to_mag_primitive = symmetry.get('T_input_to_mag_primitive', None)
        self.T_input_to_input_magnetic_primitive = symmetry.get(
            'T_input_to_input_magnetic_primitive',
            self.T_input_to_mag_primitive,
        )
        self.T_input_to_acc_primitive = symmetry.get('T_input_to_acc_primitive', None)
        self.G0_symbol = symmetry.get('G0_symbol', None)
        self.G0_num = symmetry.get('G0_num', None)
        self.L0_symbol = symmetry.get('L0_symbol', None)
        self.L0_num = symmetry.get('L0_num', None)
        self.it = symmetry.get('it', None)
        self.ik = symmetry.get('ik', None)
        self.SSPG_symbol_hm = symmetry.get('SSPG_symbol_hm', None)
        self.SSPG_symbol_s = symmetry.get('SSPG_symbol_s', None)
        self.input_space_group_number = symmetry.get('input_space_group_number', None)
        self.input_space_group_symbol = symmetry.get('input_space_group_symbol', None)
        self.sg_has_real_space_inversion = symmetry.get('sg_has_real_space_inversion', None)
        self.sg_is_polar = symmetry.get('sg_is_polar', None)
        self.sg_is_chiral = symmetry.get('sg_is_chiral', None)
        self.input_space_group_basis_or_setting = symmetry.get(
            'input_space_group_basis_or_setting',
            None,
        )
        self.source_structure_metadata = symmetry.get('source_structure_metadata', None)
        self.source_parent_space_group = symmetry.get('source_parent_space_group', None)
        self.source_cell_parameter_strings = symmetry.get('source_cell_parameter_strings', None)

        self.input_magnetic_primitive_cell = cell.get('input_magnetic_primitive_cell', None)
        self.input_magnetic_primitive_cell_setting = cell.get(
            'input_magnetic_primitive_cell_setting',
            INPUT_MAGNETIC_PRIMITIVE_SETTING,
        )
        self.input_magnetic_primitive_cell_poscar = cell.get(
            'input_magnetic_primitive_cell_poscar',
            None,
        )
        self.input_magnetic_primitive_cell_detail = cell.get(
            'input_magnetic_primitive_cell_detail',
            None,
        )

        self.magnetic_primitive_cell = cell.get(
            'magnetic_primitive_cell',
            cell['primitive_magnetic_cell'],
        )
        self.magnetic_primitive_cell_setting = cell.get(
            'magnetic_primitive_cell_setting',
            ACC_PRIMITIVE_SETTING,
        )
        self.magnetic_primitive_cell_poscar = cell.get(
            'magnetic_primitive_cell_poscar',
            cell['primitive_magnetic_cell_poscar'],
        )
        self.magnetic_primitive_cell_detail = cell.get(
            'magnetic_primitive_cell_detail',
            cell.get('primitive_magnetic_cell_detail', None),
        )

        self.primitive_magnetic_cell = cell['primitive_magnetic_cell']
        self.primitive_magnetic_cell_setting = cell.get(
            'primitive_magnetic_cell_setting',
            self.magnetic_primitive_cell_setting,
        )
        self.primitive_magnetic_cell_poscar = cell['primitive_magnetic_cell_poscar']
        self.scif = cell['scif']
        self.scif_outputs = cell.get(
            'scif_outputs',
            {
                SCIF_CELL_MODE_G0STD_ORIENTED: self.scif,
            },
        )
        self.scif_cell_modes = cell.get(
            'scif_cell_modes',
            [SCIF_CELL_MODE_G0STD_ORIENTED],
        )
        self.primitive_magnetic_cell_detail = cell.get(
            'primitive_magnetic_cell_detail',
            self.magnetic_primitive_cell_detail,
        )
        self.acc_primitive_magnetic_cell = cell.get(
            'acc_primitive_magnetic_cell',
            self.magnetic_primitive_cell,
        )
        self.acc_primitive_magnetic_cell_setting = cell.get(
            'acc_primitive_magnetic_cell_setting',
            self.primitive_magnetic_cell_setting,
        )
        self.acc_primitive_magnetic_cell_poscar = cell.get(
            'acc_primitive_magnetic_cell_poscar',
            self.primitive_magnetic_cell_poscar,
        )
        self.acc_primitive_magnetic_cell_detail = cell.get(
            'acc_primitive_magnetic_cell_detail',
            self.magnetic_primitive_cell_detail,
        )
        self.acc_conventional_cell = cell.get('acc_conventional_cell', None)
        self.acc_conventional_cell_setting = cell.get('acc_conventional_cell_setting', None)
        self.acc_conventional_cell_detail = cell.get('acc_conventional_cell_detail', None)
        self.g0_standard_cell = cell.get('g0_standard_cell', None)
        self.l0_standard_cell = cell.get('l0_standard_cell', None)
        self.convention_cell = cell.get('convention_cell', None)
        self.convention_cell_setting = cell.get('convention_cell_setting', None)
        self.convention_cell_detail = cell.get('convention_cell_detail', None)
        self.wp_chain = cell.get('wp_chain', None)


        self.index = symmetry['index']
        self.conf = symmetry['configuration']
        self.magnetic_phase = symmetry['magnetic_phase']
        self.magnetic_phase_base = symmetry.get('magnetic_phase_base', self.magnetic_phase)
        self.magnetic_phase_modifier = symmetry.get('magnetic_phase_modifier', '')
        self.magnetic_phase_spin_orbit_magnet = symmetry.get('magnetic_phase_spin_orbit_magnet', '')
        self.magnetic_phase_details = symmetry.get('magnetic_phase_details', None)
        self.acc = symmetry['acc']
        self.msg_acc = symmetry.get('msg_acc', None)
        self.KPOINTS = symmetry['KPOINTS']
        self.KPOINTS_setting = symmetry.get('KPOINTS_setting', ACC_PRIMITIVE_SETTING)
        self.KPOINTS_real_space_setting = symmetry.get(
            'KPOINTS_real_space_setting',
            self.KPOINTS_setting,
        )
        self.spin_polarizations = symmetry['spin_polarizations']
        self.spin_polarizations_setting = symmetry.get(
            'spin_polarizations_setting',
            ACC_PRIMITIVE_CARTESIAN_SETTING,
        )
        self.spin_polarizations_real_space_setting = symmetry.get(
            'spin_polarizations_real_space_setting',
            self.KPOINTS_real_space_setting,
        )
        self.spin_polarizations_spin_frame = symmetry.get(
            'spin_polarizations_spin_frame',
            self.spin_polarizations_setting,
        )
        self.spin_polarizations_acc_cartesian = symmetry.get(
            'spin_polarizations_acc_cartesian',
            self.spin_polarizations,
        )
        self.spin_polarizations_acc_cartesian_setting = symmetry.get(
            'spin_polarizations_acc_cartesian_setting',
            self.spin_polarizations_setting,
        )
        self.acc_primitive_real_cartesian_to_poscar_spin_frame = symmetry.get(
            'acc_primitive_real_cartesian_to_poscar_spin_frame',
            None,
        )
        self.poscar_spin_frame_to_acc_primitive_real_cartesian = symmetry.get(
            'poscar_spin_frame_to_acc_primitive_real_cartesian',
            None,
        )
        self.real_cartesian_to_spin_frame = symmetry.get(
            'real_cartesian_to_spin_frame',
            self.acc_primitive_real_cartesian_to_poscar_spin_frame,
        )
        self.spin_frame_to_real_cartesian = symmetry.get(
            'spin_frame_to_real_cartesian',
            self.poscar_spin_frame_to_acc_primitive_real_cartesian,
        )
        self.spin_polarizations_acc_poscar_spin_frame = symmetry.get(
            'spin_polarizations_acc_poscar_spin_frame',
            None,
        )
        self.spin_polarizations_acc_poscar_spin_frame_setting = symmetry.get(
            'spin_polarizations_acc_poscar_spin_frame_setting',
            ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
        )


        self.input_magnetic_primitive_ssg_ops = symmetry.get('input_magnetic_primitive_ssg_ops', None)
        self.input_magnetic_primitive_ssg_setting = symmetry.get(
            'input_magnetic_primitive_ssg_setting',
            INPUT_MAGNETIC_PRIMITIVE_SETTING,
        )
        self.input_magnetic_primitive_ssg_seitz = symmetry.get(
            'input_magnetic_primitive_ssg_seitz',
            None,
        )
        self.input_magnetic_primitive_ssg_seitz_latex = symmetry.get(
            'input_magnetic_primitive_ssg_seitz_latex',
            None,
        )
        self.input_magnetic_primitive_ssg_seitz_descriptions = symmetry.get(
            'input_magnetic_primitive_ssg_seitz_descriptions',
            None,
        )
        self.input_magnetic_primitive_ssg_international_linear = symmetry.get(
            'input_magnetic_primitive_ssg_international_linear',
            None,
        )
        self.input_magnetic_primitive_ssg_international_latex = symmetry.get(
            'input_magnetic_primitive_ssg_international_latex',
            None,
        )
        self.input_magnetic_primitive_ssg_symbol_calibration_tol = symmetry.get(
            'input_magnetic_primitive_ssg_symbol_calibration_tol',
            None,
        )
        self.input_magnetic_primitive_ssg_type = symmetry.get(
            'input_magnetic_primitive_ssg_type',
            None,
        )

        self.magnetic_primitive_ssg_ops = symmetry.get(
            'magnetic_primitive_ssg_ops',
            symmetry['primitive_magnetic_cell_ssg_ops'],
        )
        self.magnetic_primitive_ssg_setting = symmetry.get(
            'magnetic_primitive_ssg_setting',
            ACC_PRIMITIVE_SETTING,
        )
        self.magnetic_primitive_ssg_seitz = symmetry.get(
            'magnetic_primitive_ssg_seitz',
            symmetry.get('primitive_magnetic_cell_ssg_seitz', None),
        )
        self.magnetic_primitive_ssg_seitz_latex = symmetry.get(
            'magnetic_primitive_ssg_seitz_latex',
            symmetry.get('primitive_magnetic_cell_ssg_seitz_latex', None),
        )
        self.magnetic_primitive_ssg_seitz_descriptions = symmetry.get(
            'magnetic_primitive_ssg_seitz_descriptions',
            symmetry.get('primitive_magnetic_cell_ssg_seitz_descriptions', None),
        )
        self.magnetic_primitive_ssg_international_linear = symmetry.get(
            'magnetic_primitive_ssg_international_linear',
            symmetry.get('primitive_magnetic_cell_ssg_international_linear', None),
        )
        self.magnetic_primitive_ssg_international_latex = symmetry.get(
            'magnetic_primitive_ssg_international_latex',
            symmetry.get('primitive_magnetic_cell_ssg_international_latex', None),
        )
        self.magnetic_primitive_ssg_symbol_calibration_tol = symmetry.get(
            'magnetic_primitive_ssg_symbol_calibration_tol',
            symmetry.get('primitive_magnetic_cell_ssg_symbol_calibration_tol', None),
        )
        self.magnetic_primitive_ssg_type = symmetry.get(
            'magnetic_primitive_ssg_type',
            symmetry.get('primitive_magnetic_cell_ssg_type', None),
        )

        self.primitive_magnetic_cell_ssg_ops = symmetry['primitive_magnetic_cell_ssg_ops']
        self.primitive_magnetic_cell_ssg_setting = symmetry.get(
            'primitive_magnetic_cell_ssg_setting',
            self.magnetic_primitive_ssg_setting,
        )
        self.primitive_magnetic_cell_ssg_seitz = symmetry.get('primitive_magnetic_cell_ssg_seitz', None)
        self.primitive_magnetic_cell_ssg_seitz_latex = symmetry.get(
            'primitive_magnetic_cell_ssg_seitz_latex',
            None,
        )
        self.primitive_magnetic_cell_ssg_seitz_descriptions = symmetry.get(
            'primitive_magnetic_cell_ssg_seitz_descriptions',
            None,
        )
        self.primitive_magnetic_cell_ssg_international_linear = symmetry.get(
            'primitive_magnetic_cell_ssg_international_linear', None
        )
        self.primitive_magnetic_cell_ssg_international_latex = symmetry.get(
            'primitive_magnetic_cell_ssg_international_latex', None
        )
        self.primitive_magnetic_cell_ssg_symbol_calibration_tol = symmetry.get(
            'primitive_magnetic_cell_ssg_symbol_calibration_tol',
            self.magnetic_primitive_ssg_symbol_calibration_tol,
        )
        self.acc_primitive_ssg_ops = symmetry.get(
            'acc_primitive_ssg_ops',
            self.primitive_magnetic_cell_ssg_ops,
        )
        self.acc_primitive_ssg_setting = symmetry.get(
            'acc_primitive_ssg_setting',
            self.primitive_magnetic_cell_ssg_setting,
        )
        self.acc_primitive_ssg_seitz = symmetry.get(
            'acc_primitive_ssg_seitz',
            self.primitive_magnetic_cell_ssg_seitz,
        )
        self.acc_primitive_ssg_seitz_latex = symmetry.get(
            'acc_primitive_ssg_seitz_latex',
            self.primitive_magnetic_cell_ssg_seitz_latex,
        )
        self.acc_primitive_ssg_seitz_descriptions = symmetry.get(
            'acc_primitive_ssg_seitz_descriptions',
            self.primitive_magnetic_cell_ssg_seitz_descriptions,
        )
        self.acc_primitive_ssg_international_linear = symmetry.get(
            'acc_primitive_ssg_international_linear',
            self.primitive_magnetic_cell_ssg_international_linear,
        )
        self.acc_primitive_ssg_international_latex = symmetry.get(
            'acc_primitive_ssg_international_latex',
            self.primitive_magnetic_cell_ssg_international_latex,
        )
        self.acc_primitive_ssg_symbol_calibration_tol = symmetry.get(
            'acc_primitive_ssg_symbol_calibration_tol',
            self.magnetic_primitive_ssg_symbol_calibration_tol,
        )
        self.acc_conventional_ssg_ops = symmetry.get('acc_conventional_ssg_ops', None)
        self.acc_conventional_ssg_setting = symmetry.get('acc_conventional_ssg_setting', None)
        self.acc_conventional_ssg_seitz = symmetry.get('acc_conventional_ssg_seitz', None)
        self.acc_conventional_ssg_seitz_latex = symmetry.get('acc_conventional_ssg_seitz_latex', None)
        self.acc_conventional_ssg_seitz_descriptions = symmetry.get(
            'acc_conventional_ssg_seitz_descriptions',
            None,
        )
        self.acc_conventional_ssg_international_linear = symmetry.get(
            'acc_conventional_ssg_international_linear',
            None,
        )
        self.acc_conventional_ssg_international_latex = symmetry.get(
            'acc_conventional_ssg_international_latex',
            None,
        )
        self.acc_conventional_ssg_symbol_calibration_tol = symmetry.get(
            'acc_conventional_ssg_symbol_calibration_tol',
            None,
        )
        self.primitive_magnetic_cell_ssg_type = symmetry.get('primitive_magnetic_cell_ssg_type', None)
        self.spin_part_point_group = symmetry['full_spin_part_point_group']
        self.identify_index_details = symmetry.get('identify_index_details', None)
        self.msg_num = symmetry.get('msg_num', None)
        self.msg_type = symmetry.get('msg_type', None)
        self.msg_symbol = symmetry.get('msg_symbol', None)
        self.msg_bns_number = symmetry.get('msg_bns_number', None)
        self.msg_og_number = symmetry.get('msg_og_number', None)
        self.msg_parent_space_group_number = symmetry.get('msg_parent_space_group_number', None)
        self.msg_has_real_space_inversion = symmetry.get('msg_has_real_space_inversion', None)
        self.msg_is_polar = symmetry.get('msg_is_polar', None)
        self.msg_is_chiral = symmetry.get('msg_is_chiral', None)
        self.tolerances = symmetry.get('tolerances', None)
        self.symbol_calibration_tol = symmetry.get(
            'symbol_calibration_tol',
            self.acc_primitive_ssg_symbol_calibration_tol,
        )
        self.gspg_ops = symmetry.get('gspg_ops', None)
        self.gspg_raw_ops = symmetry.get('gspg_raw_ops', None)
        self.gspg_ops_xyz_uvw = symmetry.get('gspg_ops_xyz_uvw', None)
        self.gspg_spin_only_ops = symmetry.get('gspg_spin_only_ops', None)
        self.gspg_spin_only_ops_xyz_uvw = symmetry.get('gspg_spin_only_ops_xyz_uvw', None)
        self.gspg_collinear_axis = symmetry.get('gspg_collinear_axis', None)
        self.gspg_symbol_linear = symmetry.get('gspg_symbol_linear', None)
        self.gspg_symbol_latex = symmetry.get('gspg_symbol_latex', None)
        self.gspg_effective_mpg_symbol = symmetry.get('gspg_effective_mpg_symbol', None)
        self.g0_standard_ssg_ops = symmetry.get('g0_standard_ssg_ops', None)
        self.g0_standard_ssg_seitz = symmetry.get('g0_standard_ssg_seitz', None)
        self.g0_standard_ssg_seitz_latex = symmetry.get('g0_standard_ssg_seitz_latex', None)
        self.g0_standard_ssg_seitz_descriptions = symmetry.get(
            'g0_standard_ssg_seitz_descriptions',
            None,
        )
        self.l0_standard_ssg_ops = symmetry.get('l0_standard_ssg_ops', None)
        self.l0_standard_ssg_seitz = symmetry.get('l0_standard_ssg_seitz', None)
        self.l0_standard_ssg_seitz_latex = symmetry.get('l0_standard_ssg_seitz_latex', None)
        self.l0_standard_ssg_seitz_descriptions = symmetry.get(
            'l0_standard_ssg_seitz_descriptions',
            None,
        )
        self.convention_ssg_ops = symmetry.get('convention_ssg_ops', None)
        self.convention_ssg_setting = symmetry.get('convention_ssg_setting', None)
        self.convention_ssg_spin_frame_setting = symmetry.get(
            'convention_ssg_spin_frame_setting',
            None,
        )
        self.ossg_space_group_number = symmetry.get('ossg_space_group_number', None)
        self.ossg_has_real_space_inversion = symmetry.get('ossg_has_real_space_inversion', None)
        self.ossg_is_polar = symmetry.get('ossg_is_polar', None)
        self.ossg_is_chiral = symmetry.get('ossg_is_chiral', None)
        self.convention_spin_only_direction = symmetry.get('convention_spin_only_direction', "")
        self.convention_ssg_seitz = symmetry.get('convention_ssg_seitz', None)
        self.convention_ssg_seitz_latex = symmetry.get('convention_ssg_seitz_latex', None)
        self.convention_ssg_seitz_descriptions = symmetry.get(
            'convention_ssg_seitz_descriptions',
            None,
        )
        self.convention_nssg_ops = symmetry.get('convention_nssg_ops', None)
        self.convention_nssg_seitz = symmetry.get('convention_nssg_seitz', None)
        self.convention_nssg_seitz_latex = symmetry.get('convention_nssg_seitz_latex', None)
        self.convention_ssg_international_linear = symmetry.get(
            'convention_ssg_international_linear',
            None,
        )
        self.convention_ssg_international_latex = symmetry.get(
            'convention_ssg_international_latex',
            None,
        )
        self.convention_ssg_symbol_calibration_tol = symmetry.get(
            'convention_ssg_symbol_calibration_tol',
            None,
        )
        self.raw_T_input_to_G0std = symmetry.get('raw_T_input_to_G0std', None)
        self.raw_T_input_to_L0std = symmetry.get('raw_T_input_to_L0std', None)
        self.magnetic_primitive_msg_ops = symmetry.get(
            'magnetic_primitive_msg_ops',
            symmetry.get('primitive_msg_ops', None),
        )
        self.magnetic_primitive_msg_ops_setting = symmetry.get(
            'magnetic_primitive_msg_ops_setting',
            ACC_PRIMITIVE_SETTING,
        )
        self.magnetic_primitive_msg_ops_spin_frame_setting = symmetry.get(
            'magnetic_primitive_msg_ops_spin_frame_setting',
            None,
        )
        self.primitive_msg_ops = symmetry.get('primitive_msg_ops', self.magnetic_primitive_msg_ops)
        self.primitive_msg_ops_setting = symmetry.get(
            'primitive_msg_ops_setting',
            self.magnetic_primitive_msg_ops_setting,
        )
        self.acc_primitive_msg_ops = symmetry.get(
            'acc_primitive_msg_ops',
            self.primitive_msg_ops,
        )
        self.acc_primitive_msg_ops_setting = symmetry.get(
            'acc_primitive_msg_ops_setting',
            self.primitive_msg_ops_setting,
        )
        self.primitive_msg_ops_spin_frame_setting = symmetry.get(
            'primitive_msg_ops_spin_frame_setting',
            self.magnetic_primitive_msg_ops_spin_frame_setting,
        )
        self.acc_primitive_msg_ops_spin_frame_setting = symmetry.get(
            'acc_primitive_msg_ops_spin_frame_setting',
            self.primitive_msg_ops_spin_frame_setting,
        )
        self.msg_little_group_symbols = symmetry.get('msg_little_group_symbols', None)
        self.msg_spin_polarizations = symmetry.get('msg_spin_polarizations', None)
        self.msg_spin_polarizations_setting = symmetry.get(
            'msg_spin_polarizations_setting',
            ACC_PRIMITIVE_CARTESIAN_SETTING,
        )
        self.msg_spin_polarizations_real_space_setting = symmetry.get(
            'msg_spin_polarizations_real_space_setting',
            self.KPOINTS_real_space_setting,
        )
        self.msg_spin_polarizations_spin_frame = symmetry.get(
            'msg_spin_polarizations_spin_frame',
            self.msg_spin_polarizations_setting,
        )
        self.msg_spin_polarizations_acc_cartesian = symmetry.get(
            'msg_spin_polarizations_acc_cartesian',
            self.msg_spin_polarizations,
        )
        self.msg_spin_polarizations_acc_cartesian_setting = symmetry.get(
            'msg_spin_polarizations_acc_cartesian_setting',
            self.msg_spin_polarizations_setting,
        )
        self.msg_spin_polarizations_acc_poscar_spin_frame = symmetry.get(
            'msg_spin_polarizations_acc_poscar_spin_frame',
            None,
        )
        self.msg_spin_polarizations_acc_poscar_spin_frame_setting = symmetry.get(
            'msg_spin_polarizations_acc_poscar_spin_frame_setting',
            ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
        )
        self.T_input_to_G0std = symmetry.get('T_input_to_G0std', None)
        self.T_G0std_to_primitive = symmetry.get('T_G0std_to_primitive', None)
        self.T_G0std_to_acc_primitive = symmetry.get(
            'T_G0std_to_acc_primitive',
            self.T_G0std_to_primitive,
        )
        self.T_input_to_L0std = symmetry.get('T_input_to_L0std', None)
        self.T_L0std_to_primitive = symmetry.get('T_L0std_to_primitive', None)
        self.T_L0std_to_acc_primitive = symmetry.get(
            'T_L0std_to_acc_primitive',
            self.T_L0std_to_primitive,
        )
        self.T_input_to_convention = symmetry.get('T_input_to_convention', None)
        self.T_convention_to_acc_primitive = symmetry.get(
            'T_convention_to_acc_primitive',
            None,
        )
        self.T_convention_to_acc_conventional = symmetry.get(
            'T_convention_to_acc_conventional',
            None,
        )
        self.T_convention_to_acc_conventional_is_convention_self_automorphism = symmetry.get(
            'T_convention_to_acc_conventional_is_convention_self_automorphism',
            None,
        )
        self.T_convention_to_acc_conventional_label = symmetry.get(
            'T_convention_to_acc_conventional_label',
            None,
        )
        self.T_convention_to_acc_conventional_audit = symmetry.get(
            'T_convention_to_acc_conventional_audit',
            None,
        )
        self.selected_standard_setting = symmetry.get('selected_standard_setting', None)
        self.T_selected_standard_to_acc_conventional = symmetry.get(
            'T_selected_standard_to_acc_conventional',
            None,
        )
        self.T_selected_standard_to_acc_conventional_is_self_automorphism = symmetry.get(
            'T_selected_standard_to_acc_conventional_is_self_automorphism',
            None,
        )
        self.T_selected_standard_to_acc_conventional_label = symmetry.get(
            'T_selected_standard_to_acc_conventional_label',
            None,
        )
        self.T_selected_standard_to_acc_conventional_audit = symmetry.get(
            'T_selected_standard_to_acc_conventional_audit',
            None,
        )


        self.spinsplitting_w_soc = properties['ss_w_soc']
        self.spinsplitting_wo_soc = properties['ss_wo_soc']
        self.ahc_w_soc = properties['ahc_w_soc']
        self.ahc_wo_soc = properties['ahc_wo_soc']
        self.is_alter = properties['is_alter']
        self.is_spin_orbit_magnet = properties.get('is_spin_orbit_magnet', '')
        self.tensor_outputs = properties.get('tensor_outputs', {})
        self.AHE_woSOC = self.tensor_outputs.get('AHE_woSOC')
        self.AHE_wSOC = self.tensor_outputs.get('AHE_wSOC')
        self.BCDTensor = self.tensor_outputs.get('BCDTensor')
        self.MSGBCDTensor = self.tensor_outputs.get('MSGBCDTensor')
        self.QMDTensor = self.tensor_outputs.get('QMDTensor')
        self.MSGQMDTensor = self.tensor_outputs.get('MSGQMDTensor')
        self.IMDTensor = self.tensor_outputs.get('IMDTensor')
        self.MSGIMDTensor = self.tensor_outputs.get('MSGIMDTensor')

    def __repr__(self):
        props = self.properties_summary()
        display_symbol = (
            self.primitive_magnetic_cell_ssg_international_linear
            or self.primitive_magnetic_cell_ssg_international_latex
            or self.primitive_magnetic_cell_ssg_seitz
            or "Unknown"
        )
        return (f"<{display_symbol}>\n"
                f"  index: {self.index}\n"
                f"  conf : {self.conf}\n"
                f"  phase: {self.magnetic_phase}\n"
                f"  acc  : {self.acc}\n"
                f"  properties: {{\n"
                f"      ss_w_soc : {props['ss_w_soc']},\n"
                f"      ss_wo_soc: {props['ss_wo_soc']},\n"
                f"      ahc_w_soc: {props['ahc_w_soc']},\n"
                f"      ahc_wo_soc: {props['ahc_wo_soc']},\n"
                f"      is_alter : {props['is_alter']},\n"
                f"      is_spin_orbit_magnet : {props['is_spin_orbit_magnet']}\n"
                f"  }}")

    def properties_summary(self):
        return {
            'ss_w_soc': self.spinsplitting_w_soc,
            'ss_wo_soc': self.spinsplitting_wo_soc,
            'ahc_w_soc': self.ahc_w_soc,
            'ahc_wo_soc': self.ahc_wo_soc,
            'is_alter': self.is_alter,
            'is_spin_orbit_magnet': self.is_spin_orbit_magnet,
        }

    def gspg_summary(self):
        return {
            'symbol_linear': self.gspg_symbol_linear,
        }

    def to_summary_dict(self):
        return {
            'index': self.index,
            'conf': self.conf,
            'phase': self.magnetic_phase,
            'acc': self.acc,
            'msg_acc': self.msg_acc,
            'properties': self.properties_summary(),
            'gspg': self.gspg_summary(),
        }

    def to_dict(self):
        return self.__dict__

    def save_json(self):
        return json.dumps(self.__dict__, indent=4,cls=NumpyEncoder)

    def to_scif(
        self,
        *,
        cell_mode: str = SCIF_CELL_MODE_G0STD_ORIENTED,
    ) -> str:
        try:
            return self.scif_outputs[cell_mode]
        except KeyError as exc:
            available = sorted(self.scif_outputs.keys())
            raise ValueError(
                f"Unsupported scif output cell_mode: {cell_mode}. "
                f"Available: {available}"
            ) from exc






AFM_LIKE_BASE_PHASES = {"AFM"}
FM_LIKE_BASE_PHASES = {"FM/FiM", "Compensated FiM"}


def is_alter(condition, magnetic_phase_base, spinsplitting):
    if (
        condition == 'Collinear'
        and magnetic_phase_base in AFM_LIKE_BASE_PHASES
        and spinsplitting == 'k-dependent'
    ):
        return '(Altermagnet)'
    return ''


def spin_splitting_wo_soc(magnetic_phase_base, is_ss_gp):
    if magnetic_phase_base in AFM_LIKE_BASE_PHASES:
        if is_ss_gp == "no spin splitting":
            return 'No'
        return 'k-dependent'
    return 'Zeeman'

def spin_splitting_w_soc(ssg:SpinSpaceGroup):
    if ssg.is_PT:
        return 'No'
    else:
        return 'Yes'


def is_ahc(mpg):
    if mpg == None:
        return 'Error, cannot determine MSG.'
    if mpg in MSGMPG_DB.FMMPG_INTlist:
        wSOC = 'Yes'
    else:
        wSOC = 'No'
    return wSOC


def _serialize_tensor_solution(solution, operations_count):
    constraint_matrix, nullspace_basis, relations, components = solution
    free_parameters = int(nullspace_basis.shape[1]) if nullspace_basis.ndim == 2 else 0

    def _symbolize_display(value):
        if isinstance(value, str):
            return symbolize_numeric_tokens_in_string(value, sqrt_tol=1e-4)
        if isinstance(value, list):
            return [_symbolize_display(item) for item in value]
        if isinstance(value, tuple):
            return [_symbolize_display(item) for item in value]
        if isinstance(value, dict):
            return {key: _symbolize_display(item) for key, item in value.items()}
        return value

    return {
        'operations_count': operations_count,
        'constraint_shape': list(constraint_matrix.shape),
        'nullspace_shape': list(nullspace_basis.shape),
        'free_parameters': free_parameters,
        'is_zero': free_parameters == 0,
        'relations': _symbolize_display(relations),
        'components': _symbolize_display(components),
        'nullspace_basis': nullspace_basis.tolist(),
    }


def _lattice_column_matrix(cell: CrystalCell) -> np.ndarray:
    return np.asarray(cell.lattice_matrix, dtype=float).T


def _cartesianize_similarity(matrix: np.ndarray, lattice_col: np.ndarray) -> np.ndarray:
    return lattice_col @ np.asarray(matrix, dtype=float) @ np.linalg.inv(lattice_col)


def _poscar_spin_frame_rotation(cell: CrystalCell) -> np.ndarray:
    _, rotation = standardize_lattice(np.asarray(cell.lattice_matrix, dtype=float))
    return np.asarray(rotation, dtype=float)


def _ossg_oriented_spin_frame_ssg(ssg: SpinSpaceGroup, cell: CrystalCell) -> SpinSpaceGroup:
    lattice_col = _lattice_column_matrix(cell)
    oriented_ssg = ssg.transform_spin(np.linalg.inv(lattice_col))
    if oriented_ssg.real_space_metric is not None:
        return oriented_ssg
    return SpinSpaceGroup(
        oriented_ssg.ops,
        tol=ssg.tol,
        real_space_metric=np.asarray(cell.lattice_matrix, dtype=float) @ np.asarray(cell.lattice_matrix, dtype=float).T,
    )


def _tensor_ops_wo_soc(ssg: SpinSpaceGroup, cell: CrystalCell):
    lattice_col = _lattice_column_matrix(cell)
    oriented_ssg = _ossg_oriented_spin_frame_ssg(ssg, cell)
    return [
        [
            _cartesianize_similarity(Rs, lattice_col),
            _cartesianize_similarity(Rr, lattice_col),
        ]
        for Rs, Rr in oriented_ssg.gspg_ops_raw
    ]


def _tensor_ops_w_soc(ssg: SpinSpaceGroup, cell: CrystalCell, tol: float):
    lattice_col = _lattice_column_matrix(cell)
    oriented_ssg = _ossg_oriented_spin_frame_ssg(ssg, cell)
    magnetic_point_group = oriented_ssg.msg_ops
    return [
        [
            _cartesianize_similarity(Rs, lattice_col),
            _cartesianize_similarity(Rr, lattice_col),
        ]
        for Rs, Rr, _ in magnetic_point_group
    ]


def _compute_tensor_outputs(ssg: SpinSpaceGroup, cell: CrystalCell, tol: float):
    ops_wo_soc = _tensor_ops_wo_soc(ssg, cell)
    ops_w_soc = _tensor_ops_w_soc(ssg, cell, tol=tol)
    tensor_specs = {
        'AHE_woSOC': (solve_ahe, ops_wo_soc, {'symbol': r'\sigma', 'use_antisymmetry': True}),
        'AHE_wSOC': (solve_ahe, ops_w_soc, {'symbol': r'\sigma', 'use_antisymmetry': True}),
        'BCDTensor': (solve_bcd, ops_wo_soc, {'symbol': 'D'}),
        'MSGBCDTensor': (solve_bcd, ops_w_soc, {'symbol': 'D'}),
        'QMDTensor': (solve_qmd, ops_wo_soc, {'symbol': 'Q'}),
        'MSGQMDTensor': (solve_qmd, ops_w_soc, {'symbol': 'Q'}),
        'IMDTensor': (solve_imd, ops_wo_soc, {'symbol': 'I'}),
        'MSGIMDTensor': (solve_imd, ops_w_soc, {'symbol': 'I'}),
    }
    tensor_outputs = {}
    for key, (solver, operations, kwargs) in tensor_specs.items():
        try:
            tensor_outputs[key] = _serialize_tensor_solution(
                solver(operations, **kwargs),
                operations_count=len(operations),
            )
        except Exception as error:
            tensor_outputs[key] = {'error': str(error)}
    return tensor_outputs


def _serialize_cell_snapshot(cell: CrystalCell) -> dict:
    lattice, positions, type_ids, moments = cell.to_spglib(mag=True)
    return {
        'lattice': np.asarray(lattice, dtype=float).tolist(),
        'positions': np.asarray(positions, dtype=float).tolist(),
        'type_ids': list(type_ids),
        'moments': np.asarray(moments, dtype=float).tolist(),
        'elements': [cell.atom_types_to_symbol[type_id] for type_id in type_ids],
        'occupancies': [float(cell.atom_types_to_occupancies[type_id]) for type_id in type_ids],
    }


def _serialize_gspg_ops(ops) -> list[list[list[list[float]]]]:
    return [
        [
            np.asarray(spin_rotation, dtype=float).tolist(),
            np.asarray(space_rotation, dtype=float).tolist(),
        ]
        for spin_rotation, space_rotation in ops
    ]


def _gspg_time_reversal_from_spin_rotation(spin_rotation: np.ndarray, *, tol: float = 1e-6) -> int | None:
    det = float(np.linalg.det(np.asarray(spin_rotation, dtype=float)))
    if abs(det - 1.0) < tol:
        return 1
    if abs(det + 1.0) < tol:
        return -1
    return None


def _serialize_gspg_xyz_uvw_ops(
    ops,
    *,
    tol: float = 1e-6,
    translation: np.ndarray | None = None,
) -> list[dict]:
    zero_translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    payload = []
    for idx, (spin_rotation, real_rotation) in enumerate(ops):
        spin_rotation = np.asarray(spin_rotation, dtype=float)
        real_rotation = np.asarray(real_rotation, dtype=float)
        time_reversal = _gspg_time_reversal_from_spin_rotation(spin_rotation, tol=tol)
        xyzt = affine_matrix_to_xyz_expression(real_rotation, zero_translation)
        if time_reversal is not None:
            xyzt = f"{xyzt},{time_reversal:+d}"
        payload.append(
            {
                "index": idx + 1,
                "xyzt": xyzt,
                "uvw": affine_matrix_to_xyz_expression(spin_rotation),
                "time_reversal": time_reversal,
                "spin_rotation": spin_rotation.tolist(),
                "real_rotation": real_rotation.tolist(),
                "translation": zero_translation.tolist(),
            }
        )
    return payload


def _serialize_ssg_operation_matrices(
    ops: list[SpinSpaceGroupOperation],
) -> list[dict]:
    return [
        {
            "index": idx + 1,
            "spin_rotation": np.asarray(op.spin_rotation, dtype=float).tolist(),
            "real_rotation": np.asarray(op.rotation, dtype=float).tolist(),
            "translation": np.asarray(op.translation, dtype=float).tolist(),
        }
        for idx, op in enumerate(ops)
    ]


def _serialize_msg_operation_matrices(
    ops: list[SpinSpaceGroupOperation],
    *,
    tol: float,
) -> list[dict]:
    return [
        {
            "index": idx + 1,
            "time_reversal": int(op.magnetic_time_reversal(atol=tol)),
            "real_rotation": np.asarray(op[1], dtype=float).tolist(),
            "translation": np.asarray(op[2], dtype=float).tolist(),
        }
        for idx, op in enumerate(ops)
    ]


def _serialize_effective_mpg_ops(ops) -> list[list]:
    return [
        [
            int(time_reversal),
            np.asarray(rotation, dtype=float).tolist(),
        ]
        for time_reversal, rotation in ops
    ]


def _serialize_rotation_ops(ops) -> list[list[list[float]]]:
    return [np.asarray(rotation, dtype=float).tolist() for rotation in ops]


def _serialize_seitz_descriptions(descriptions) -> list[dict]:
    return json.loads(json.dumps(descriptions, cls=NumpyEncoder))


def _serialize_op_list_seitz_symbols(
    ops: list[SpinSpaceGroupOperation],
    *,
    tol: float,
) -> tuple[list[str], list[str]]:
    descriptions = [
        op.seitz_description(tol=tol, max_order=120, max_axis_denom=12)
        for op in ops
    ]
    canonicalized = canonicalize_group_seitz_descriptions(
        descriptions,
        tol=tol,
        max_axis_denom=12,
    )
    return (
        [item["symbol"] for item in canonicalized],
        [item["symbol_latex"] for item in canonicalized],
    )


def _build_gspg_payload(
    ssg: SpinSpaceGroup,
    *,
    real_space_setting: str,
    spin_frame_setting: str,
) -> dict:
    presented_ops = ssg.gspg.ops
    raw_ops = ssg.gspg.raw_ops
    output_mode = (
        "reduced_point_part_with_spin_only_annotation"
        if ssg.gspg.public_ops_are_reduced
        else "explicit_ops"
    )

    spin_only_ops = [
        [np.asarray(rotation, dtype=float), np.eye(3)]
        for rotation in ssg.gspg_spin_only_ops
    ]

    return {
        "gspg_ops": _serialize_gspg_ops(presented_ops),
        "gspg_raw_ops": _serialize_gspg_ops(raw_ops),
        "gspg_ops_xyz_uvw": _serialize_gspg_xyz_uvw_ops(presented_ops, tol=ssg.tol),
        "gspg_spin_only_ops": _serialize_gspg_ops(spin_only_ops),
        "gspg_spin_only_ops_xyz_uvw": _serialize_gspg_xyz_uvw_ops(spin_only_ops, tol=ssg.tol),
        "gspg_collinear_axis": (
            None if ssg.gspg.collinear_axis is None else np.asarray(ssg.gspg.collinear_axis, dtype=float).tolist()
        ),
        "gspg_symbol_linear": ssg.gspg.symbol_linear,
        "gspg_symbol_latex": ssg.gspg.symbol_latex,
        "gspg_effective_mpg_symbol": ssg.gspg.empg_symbol,
    }


def _spin_only_component_symbols(ssg: SpinSpaceGroup) -> tuple[str, str]:
    if ssg.conf == "Collinear":
        if len(ssg.sog) == 4:
            return "∞m", "C∞v"
        if len(ssg.sog) == 8:
            return "∞/mm", "D∞h"
        raise ValueError("Collinear spin-only symbol identification error")

    info = identify_point_group([np.asarray(op[0], dtype=float) for op in ssg.sog], _id=True)
    return info[0], info[4]


def _compose_setting_transform(
    source_matrix: np.ndarray,
    source_shift: np.ndarray,
    target_matrix: np.ndarray,
    target_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = target_matrix @ np.linalg.inv(source_matrix)
    shift = normalize_vector_to_zero(
        target_shift - target_matrix @ np.linalg.inv(source_matrix) @ source_shift,
        atol=1e-10,
    )
    return transform, shift


def _chain_setting_transform(
    first_matrix: np.ndarray,
    first_shift: np.ndarray,
    second_matrix: np.ndarray,
    second_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(second_matrix, dtype=float) @ np.asarray(first_matrix, dtype=float)
    shift = normalize_vector_to_zero(
        np.asarray(second_matrix, dtype=float) @ np.asarray(first_shift, dtype=float)
        + np.asarray(second_shift, dtype=float),
        atol=1e-10,
    )
    return transform, shift


def _invert_setting_transform(
    transform: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform_inv = np.linalg.inv(transform)
    shift_inv = normalize_vector_to_zero(-transform_inv @ shift, atol=1e-10)
    return transform_inv, shift_inv


def _identity_setting_transform() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3), np.zeros(3)


def _identify_affine_4x4_to_setting_transform(
    matrix4x4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the identify-index internal 4x4 affine layout into the public
    direct-transform pair `(A, b)` with:

        x_target = A x_source + b

    Internal identify-index matrices use the nonstandard layout generated by
    `make_4d_matrix(...)`, where the affine block is stored as:

        [[1, 0, 0, 0],
         [b0, A00, A01, A02],
         [b1, A10, A11, A12],
         [b2, A20, A21, A22]]
    """
    matrix4x4 = np.asarray(matrix4x4, dtype=float)
    if matrix4x4.shape != (4, 4):
        raise ValueError(f"Expected identify affine 4x4 matrix, got shape {matrix4x4.shape}.")
    return matrix4x4[1:, 1:], matrix4x4[1:, 0]


def _acc_setting_allows_input_collapse(acc_symbol: str | None) -> bool:
    if not acc_symbol:
        return False
    return acc_symbol.rstrip().endswith("P")


def _cartesianized_input_cell(cell: CrystalCell) -> CrystalCell:
    moments_cartesian = cell.moments_cartesian
    return CrystalCell(
        lattice=np.asarray(cell.lattice_matrix, dtype=float),
        positions=np.asarray(cell.positions, dtype=float),
        occupancies=list(cell.occupancies),
        elements=list(cell.elements),
        moments=None if moments_cartesian is None else np.asarray(moments_cartesian, dtype=float),
        spin_setting=None if moments_cartesian is None else "cartesian",
        tol=cell.tol,
    )


def _cell_to_poscar_preserving_lattice(cell: CrystalCell, filename: str) -> str:
    lattice, positions, types, moments = cell.to_spglib(mag=True)
    rows = sorted(
        zip(positions, types, moments),
        key=lambda item: (
            item[1],
            item[2][0],
            item[2][1],
            item[2][2],
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    positions_sorted, types_sorted, moments_sorted = zip(*rows)

    atom_name = ["initial"]
    count = ["initial"]
    for atom_type in types_sorted:
        symbol = cell.atom_types_to_symbol[int(atom_type)]
        if symbol != atom_name[-1]:
            atom_name.append(symbol)
            count.append(1)
        else:
            count[-1] += 1

    return "\n".join(
        [
            filename + f"#FINDSPINGROUP(version{__version__})",
            "1",
            *(" ".join(f"{value:.10f}" for value in row) for row in np.asarray(lattice, dtype=float)),
            " ".join(atom_name[1:]),
            " ".join(map(str, count[1:])),
            "direct",
            *(" ".join(f"{value:.10f}" for value in position) for position in positions_sorted),
            "# MAGMOM="
            + " ".join(
                " ".join(f"{value:.10f}" for value in moment)
                for moment in moments_sorted
            ),
        ]
    )


def _canonicalize_standard_setting_transform(
    source_cell: CrystalCell,
    source_ssg: SpinSpaceGroup,
    transform: np.ndarray,
    shift: np.ndarray,
    *,
    tol: float = 1e-5,
) -> tuple[CrystalCell, SpinSpaceGroup, tuple[np.ndarray, np.ndarray], dict]:
    transform = np.asarray(transform, dtype=float)
    shift = np.asarray(shift, dtype=float)
    audit = audit_spatial_transform_effect(
        source_ssg,
        transform,
        shift,
        tol=tol,
        use_nssg=False,
    )
    if audit["real_ops_exact_same"]:
        return source_cell, source_ssg, _identity_setting_transform(), audit

    target_cell = source_cell.transform(transform, shift)
    target_ssg = source_ssg.transform(transform, shift)
    normalized = (
        transform,
        normalize_vector_to_zero(shift, atol=1e-10),
    )
    return target_cell, target_ssg, normalized, audit


def _canonicalize_input_to_standard_setting(
    input_cell_cartesian: CrystalCell,
    target_cell: CrystalCell,
    target_ssg: SpinSpaceGroup,
    transform_input_to_target: tuple[np.ndarray, np.ndarray],
    *,
    allow_identity_collapse: bool = True,
    tol: float = 1e-5,
) -> tuple[CrystalCell, SpinSpaceGroup, tuple[np.ndarray, np.ndarray], dict]:
    transform = np.asarray(transform_input_to_target[0], dtype=float)
    shift = np.asarray(transform_input_to_target[1], dtype=float)
    audit = audit_spatial_transform_effect(
        target_ssg,
        transform,
        shift,
        tol=tol,
        use_nssg=False,
    )
    if allow_identity_collapse and audit["real_ops_exact_same"]:
        target_to_input = _invert_setting_transform(transform, shift)
        input_basis_ssg = target_ssg.transform(*target_to_input)
        return input_cell_cartesian, input_basis_ssg, _identity_setting_transform(), audit
    return (
        target_cell,
        target_ssg,
        (transform, normalize_vector_to_zero(shift, atol=1e-10)),
        audit,
    )


def _format_basis_transform_component(value: float, symbol: str, *, tol: float = 1e-10) -> str:
    numeric = float(value)
    if abs(numeric) <= tol:
        return ""
    fraction = Fraction(numeric).limit_denominator(12)
    if abs(float(fraction) - numeric) > 1e-9:
        coeff = f"{numeric:.6f}".rstrip("0").rstrip(".")
    elif fraction.denominator == 1:
        coeff = str(fraction.numerator)
    else:
        coeff = f"{fraction.numerator}/{fraction.denominator}"
    if coeff == "1":
        return symbol
    if coeff == "-1":
        return f"-{symbol}"
    return f"{coeff}{symbol}"


def _format_basis_transform_rows(matrix: np.ndarray, symbols: tuple[str, str, str], *, tol: float = 1e-10) -> str:
    matrix = np.asarray(matrix, dtype=float)
    rows = []
    for row in matrix:
        pieces = [
            _format_basis_transform_component(value, symbol, tol=tol)
            for value, symbol in zip(row, symbols)
        ]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            rows.append("0")
            continue
        rendered = pieces[0]
        for piece in pieces[1:]:
            rendered += piece if piece.startswith("-") else f"+{piece}"
        rows.append(rendered)
    return ",".join(rows)


def _build_candidate_transform_chen_pp_abcs_hex_spatial_cubic_spin_from_identify(
    *,
    current_space_to_input_basis: np.ndarray,
    identify_point_group_transformation: np.ndarray,
) -> dict:
    """
    Provisional current-file -> Chen transform for the 3.24 audit slice.

    Source side:
    - current emitted `.scif` spatial basis = current `G0std_oriented` hex basis `(a,b,c)`
    - current emitted spin basis = file-declared `(as,bs,cs)` with `transform_spinframe_P_abc = 'a,b,c'`

    Target side for this slice:
    - Chen spatial basis = same hex basis `(A,B,C) = (a,b,c)`
    - Chen spin basis = cubic basis aligned to the input cubic setting and
      ordered/oriented according to the identify-index point-group map

    `current_space_to_input_basis` is the row-matrix whose rows are the input
    cubic basis vectors expressed in the current emitted spatial basis.
    `identify_point_group_transformation` is the identify-index 3x3 point-group
    transformation returned for the Chen equivalent-map resolution.
    """
    current_space_to_input_basis = np.asarray(current_space_to_input_basis, dtype=float)
    identify_point_group_transformation = np.asarray(
        identify_point_group_transformation,
        dtype=float,
    )
    spin_basis_current_to_chen = (
        identify_point_group_transformation @ np.linalg.inv(current_space_to_input_basis)
    )
    space_basis_current_to_chen = np.eye(3)
    origin_current_to_chen = np.zeros(3)
    return {
        "from_spatial_setting": "current_scif_g0std_oriented_hex",
        "from_spin_frame": "current_file_spinframe_(as,bs,cs)",
        "to_spatial_setting": "chen_hex_spatial",
        "to_spin_frame": "chen_cubic_spin_basis",
        "space_basis_rows_abc": space_basis_current_to_chen.tolist(),
        "origin_shift_p": origin_current_to_chen.tolist(),
        "spin_basis_rows_abcs": spin_basis_current_to_chen.tolist(),
        "transform_Chen_Pp_abcs": (
            f"{_format_basis_transform_rows(space_basis_current_to_chen, ('a', 'b', 'c'))};"
            f"0,0,0;"
            f"{_format_basis_transform_rows(spin_basis_current_to_chen, ('as', 'bs', 'cs'))}"
        ),
    }


def _spin_transform_to_in_lattice(cell: CrystalCell) -> np.ndarray:
    actual_basis = np.array(
        [vector / np.linalg.norm(vector) for vector in np.asarray(cell.lattice_matrix, dtype=float)],
        dtype=float,
    ).T
    return np.linalg.inv(actual_basis)


def _spin_transform_to_oriented_abc(cell: CrystalCell) -> np.ndarray:
    lattice_col = np.asarray(cell.lattice_matrix, dtype=float).T
    return np.linalg.inv(lattice_col)


def _build_scif_export_targets(
    *,
    input_cell: CrystalCell,
    magnetic_primitive_cell: CrystalCell,
    ssg_primitive: SpinSpaceGroup,
    acc_magnetic_primitive_cell: CrystalCell,
    acc_magnetic_primitive_ssg: SpinSpaceGroup,
    G0std_cell: CrystalCell,
    G0std_ssg: SpinSpaceGroup,
    L0std_cell: CrystalCell,
    L0std_ssg: SpinSpaceGroup,
    transformation_input_to_primitive: tuple[np.ndarray, np.ndarray],
    transformation_input_to_acc_primitive: tuple[np.ndarray, np.ndarray],
    transformation_input_to_G0std: tuple[np.ndarray, np.ndarray],
    transformation_input_to_L0std: tuple[np.ndarray, np.ndarray],
):
    primitive_to_input = _invert_setting_transform(*transformation_input_to_primitive)
    input_ssg = ssg_primitive.transform(*primitive_to_input)

    primitive_in_lattice = _spin_transform_to_in_lattice(magnetic_primitive_cell)
    acc_primitive_in_lattice = _spin_transform_to_in_lattice(acc_magnetic_primitive_cell)
    g0std_in_lattice = _spin_transform_to_in_lattice(G0std_cell)
    l0std_in_lattice = _spin_transform_to_in_lattice(L0std_cell)
    input_in_lattice = _spin_transform_to_in_lattice(input_cell)
    primitive_true_abc = _spin_transform_to_oriented_abc(magnetic_primitive_cell)
    acc_primitive_true_abc = _spin_transform_to_oriented_abc(acc_magnetic_primitive_cell)
    g0std_true_abc = _spin_transform_to_oriented_abc(G0std_cell)
    l0std_true_abc = _spin_transform_to_oriented_abc(L0std_cell)
    input_true_abc = _spin_transform_to_oriented_abc(input_cell)

    def _basis_tag_transforms_for_export(
        transformation_input_to_export: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            "input": transformation_input_to_export,
            "magnetic_primitive": _compose_setting_transform(
                transformation_input_to_acc_primitive[0],
                transformation_input_to_acc_primitive[1],
                transformation_input_to_export[0],
                transformation_input_to_export[1],
            ),
            "G0std": _compose_setting_transform(
                transformation_input_to_G0std[0],
                transformation_input_to_G0std[1],
                transformation_input_to_export[0],
                transformation_input_to_export[1],
            ),
            "L0std": _compose_setting_transform(
                transformation_input_to_L0std[0],
                transformation_input_to_L0std[1],
                transformation_input_to_export[0],
                transformation_input_to_export[1],
            ),
        }

    return {
        SCIF_CELL_MODE_MAGNETIC_PRIMITIVE: {
            "export_cell": acc_magnetic_primitive_cell.transform_spin(acc_primitive_in_lattice, "in_lattice"),
            "export_ssg": acc_magnetic_primitive_ssg.transform_spin(acc_primitive_true_abc),
            "transform_input_to_export": transformation_input_to_acc_primitive,
            "basis_tag_transforms": _basis_tag_transforms_for_export(
                transformation_input_to_acc_primitive,
            ),
            "setting_name": ACC_PRIMITIVE_SETTING,
        },
        SCIF_CELL_MODE_G0STD_ORIENTED: {
            "export_cell": G0std_cell.transform_spin(g0std_in_lattice, "in_lattice"),
            "export_ssg": G0std_ssg.transform_spin(g0std_true_abc),
            "transform_input_to_export": transformation_input_to_G0std,
            "basis_tag_transforms": _basis_tag_transforms_for_export(
                transformation_input_to_G0std,
            ),
            "setting_name": G0_STANDARD_SETTING,
        },
        SCIF_CELL_MODE_INPUT: {
            "export_cell": input_cell,
            "export_ssg": input_ssg.transform_spin(input_true_abc),
            "transform_input_to_export": (np.eye(3), np.zeros(3)),
            "basis_tag_transforms": _basis_tag_transforms_for_export(
                (np.eye(3), np.zeros(3)),
            ),
            "setting_name": "input",
        },
    }


def _identify_parent_space_group_for_export_cell(
    export_cell: CrystalCell,
    *,
    symprec: float,
    source_parent_space_group: dict | None = None,
    reuse_source_transforms: bool = False,
):
    cell = export_cell.to_spglib(mag=False)
    dataset = get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return None, {
            "status": "generation_failed",
            "matches_input": None,
            "input_name_H_M_alt": (
                None if source_parent_space_group is None else source_parent_space_group.get("name_H_M_alt")
            ),
            "input_IT_number": (
                None if source_parent_space_group is None else source_parent_space_group.get("IT_number")
            ),
        }
    if dataset.number in SG_HALL_MAPPING:
        dataset = get_symmetry_dataset(cell, symprec=symprec, hall_number=SG_HALL_MAPPING[dataset.number])

    generated = {
        "name_H_M_alt": str(dataset.international),
        "IT_number": int(dataset.number),
        "transform_to_parent_space_group_Pp": "",
    }
    try:
        generated["transform_to_parent_space_group_Pp"] = affine_matrix_to_xyz_expression(
            np.asarray(dataset.transformation_matrix, dtype=float).T,
            normalize_vector_to_zero(np.asarray(dataset.origin_shift, dtype=float), atol=1e-9),
            ('a', 'b', 'c'),
            separate_translation=True,
            coeff_precision=6,
        )
    except Exception as exc:
        warnings.warn(
            f"Unable to serialize parent-space-group transform for export cell: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        generated["child_transform_Pp_abc"] = affine_matrix_to_xyz_expression(
            np.asarray(dataset.transformation_matrix, dtype=float).T,
            normalize_vector_to_zero(np.asarray(dataset.origin_shift, dtype=float), atol=1e-9),
            ('a', 'b', 'c'),
            separate_translation=True,
            coeff_precision=6,
        )
    except Exception as exc:
        warnings.warn(
            f"Unable to serialize parent-space-group child transform for export cell: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    matches_input = None
    if source_parent_space_group is not None:
        checks = []
        input_number = source_parent_space_group.get("IT_number")
        input_name = source_parent_space_group.get("name_H_M_alt")
        if input_number is not None:
            checks.append(int(round(float(input_number))) == generated["IT_number"])
        if input_name is not None:
            checks.append(
                re.sub(r"\s+", "", str(input_name).strip())
                == re.sub(r"\s+", "", generated["name_H_M_alt"])
            )
        if checks:
            matches_input = all(checks)
        if matches_input and reuse_source_transforms:
            parent_transform = source_parent_space_group.get("transform_Pp_abc")
            child_transform = source_parent_space_group.get("child_transform_Pp_abc")
            if input_name is not None:
                generated["name_H_M_alt"] = str(input_name).strip()
            if input_number is not None:
                generated["IT_number"] = int(round(float(input_number)))
            if parent_transform is not None:
                generated["transform_Pp_abc"] = parent_transform
            if child_transform is not None and "child_transform_Pp_abc" not in generated:
                generated["child_transform_Pp_abc"] = child_transform
        elif matches_input:
            if input_name is not None:
                generated["name_H_M_alt"] = str(input_name).strip()
            if input_number is not None:
                generated["IT_number"] = int(round(float(input_number)))

    if matches_input is True:
        status = "matches_input_metadata"
    elif matches_input is False:
        status = "generated_differs_from_input_metadata"
    else:
        status = "generated_without_input_metadata"

    return generated, {
        "status": status,
        "matches_input": matches_input,
        "input_name_H_M_alt": (
            None if source_parent_space_group is None else source_parent_space_group.get("name_H_M_alt")
        ),
        "input_IT_number": (
            None if source_parent_space_group is None else source_parent_space_group.get("IT_number")
        ),
    }


def _primitive_msg_ops_from_ssg(ssg_ops, tol: float, time_reversal_resolver=None) -> list[list]:
    primitive_msg_ops = []
    for op in ssg_ops:
        if time_reversal_resolver is None:
            time_reversal = op.magnetic_time_reversal(atol=tol)
        else:
            time_reversal = time_reversal_resolver(op)
        if time_reversal is None:
            continue
        primitive_msg_ops.append(
            [
                int(time_reversal),
                np.asarray(op[1], dtype=float),
                np.asarray(op[2], dtype=float),
            ]
        )
    return primitive_msg_ops


def _get_magnetic_little_group(kpoint, primitive_msg_operations, tol: float) -> list[list]:
    magnetic_little_group = []
    primitive_kpoint = np.asarray(kpoint, dtype=float)
    for time_reversal, rotation, translation in primitive_msg_operations:
        transformed_kpoint = time_reversal * np.asarray(rotation, dtype=float) @ primitive_kpoint
        if getNormInf(transformed_kpoint % 1, primitive_kpoint) < tol:
            magnetic_little_group.append([time_reversal, rotation, translation])
    return magnetic_little_group


def _get_spin_constraint_for_msg_little_groups(
    little_groups: list[list[list]],
    cell: CrystalCell,
    tol: float,
    spin_frame_rotation: np.ndarray | None = None,
) -> list[list[str]]:
    lattice_col = _lattice_column_matrix(cell)
    target_rotation = None if spin_frame_rotation is None else np.asarray(spin_frame_rotation, dtype=float)
    target_rotation_inv = None if target_rotation is None else np.linalg.inv(target_rotation)
    constraints = []
    for little_group in little_groups:
        spin_matrices = []
        for time_reversal, rotation, _ in little_group:
            rotation_cartesian = _cartesianize_similarity(rotation, lattice_col)
            if target_rotation is not None:
                rotation_cartesian = target_rotation @ rotation_cartesian @ target_rotation_inv
            spin_matrices.append(time_reversal * np.linalg.det(rotation_cartesian) * rotation_cartesian - np.eye(3))
        spinmatrices = np.vstack(deduplicate_matrix_pairs(spin_matrices, tol=tol))
        constraints.append(combine_parametric_solutions(rref_with_tolerance(spinmatrices)))
    return constraints


def _build_msg_little_group_payload(
    ssg: SpinSpaceGroup,
    cell: CrystalCell,
    tol: float,
    spin_frame_rotation: np.ndarray | None = None,
) -> tuple[list[list], list[str], list[list[str]]]:
    primitive_msg_ops, little_groups, little_group_symbols = _build_msg_little_group_core(
        ssg,
        tol=tol,
    )
    spin_constraints = _get_spin_constraint_for_msg_little_groups(
        little_groups,
        cell=cell,
        tol=tol,
        spin_frame_rotation=spin_frame_rotation,
    )
    return primitive_msg_ops, little_group_symbols, spin_constraints


def _build_msg_little_group_core(
    ssg: SpinSpaceGroup,
    *,
    tol: float,
) -> tuple[list[list], list[list[list]], list[str]]:
    primitive_msg_ops = _primitive_msg_ops_from_ssg(
        ssg.msg_ops,
        tol=tol,
        time_reversal_resolver=ssg.classify_magnetic_operation,
    )
    little_groups = [
        _get_magnetic_little_group(kpoint, primitive_msg_ops, tol=tol)
        for kpoint in ssg.kpoints_primitive
    ]
    little_group_symbols = []
    for group in little_groups:
        if not group:
            little_group_symbols.append("1")
            continue
        msg_info = get_magnetic_space_group_from_operations(group)
        little_group_symbols.append(msg_info["mpg_symbol"] if msg_info else "Unknown")
    return primitive_msg_ops, little_groups, little_group_symbols


def _make_wp_chain(wp_sg, wp_ssg, wp_msg, cell, atom_types_dict):
    chain = tuple(
        (
            atom_types_dict[int(cell[2][i])],
            wp_sg[i][0],
            wp_sg[i][1],
            wp_ssg[i][0],
            wp_ssg[i][1],
            wp_msg[i][0],
            wp_msg[i][1],
        )
        for i in range(min(len(wp_sg), len(wp_ssg), len(wp_msg)))
    )
    return sorted(set(chain), key=lambda item: (str(item[0]), item[1:]))


def _build_wp_chain_payload(g0_cell: CrystalCell, g0_ssg: SpinSpaceGroup, tol_cfg: Tolerances):
    sg_dataset = get_symmetry_dataset(g0_cell.to_spglib(), symprec=tol_cfg.space)
    msg_dataset_magnetic = get_magnetic_symmetry_dataset(
        g0_cell.to_spglib(mag=True),
        symprec=tol_cfg.space,
        mag_symprec=tol_cfg.moment,
    )
    if msg_dataset_magnetic is None:
        return []
    msg_ops = [list(item) for item in zip(msg_dataset_magnetic.rotations, msg_dataset_magnetic.translations)]
    msg_dataset = get_G0_dataset_for_cell(msg_ops, g0_cell.to_spglib(mag=True), tol_cfg.space)
    ssg_dataset = get_G0_dataset_for_cell(g0_ssg.G0_ops, g0_cell.to_spglib(mag=True), tol_cfg.space)
    wp_extended_sg = get_wp_from_dataset(sg_dataset, max=False)
    wp_extended_ssg = get_wp_from_dataset(ssg_dataset, max=True)
    wp_extended_msg = get_wp_from_dataset(msg_dataset, max=True)
    return _make_wp_chain(
        wp_extended_sg,
        wp_extended_ssg,
        wp_extended_msg,
        g0_cell.to_spglib(mag=True),
        g0_cell.atom_types_to_symbol,
    )


def _is_fm_fim_spin_point_group_symbol(symbol: str) -> bool:
    normalized = str(symbol).strip()
    return (
        bool(re.match(r"^C\d+(?!h)", normalized))
        or bool(re.match(r"^Cs", normalized))
        or normalized == "C∞v"
        or normalized == "C∞ v"
        or bool(re.match(r"^C_\{\\infty} v", normalized))
        or bool(re.match(r"^C\*v", normalized))
        or normalized == "∞m"
    )


def _is_fm_fim_spin_point_group(*symbols: str) -> bool:
    return any(_is_fm_fim_spin_point_group_symbol(symbol) for symbol in symbols if symbol is not None)


def classify_magnetic_phase(
    *,
    conf,
    full_spin_part_point_group_hm,
    full_spin_part_point_group_s,
    net_moment,
    mpg_identifier,
    is_ss_gp,
):
    net_moment_value = float(net_moment)
    zero_net_moment = abs(net_moment_value) < 1e-4
    fm_like_by_spin_point_group = _is_fm_fim_spin_point_group(
        full_spin_part_point_group_hm,
        full_spin_part_point_group_s,
    )
    som_by_mpg = mpg_identifier in MSGMPG_DB.FMMPG_INTlist if mpg_identifier is not None else False

    if fm_like_by_spin_point_group:
        base_phase = 'Compensated FiM' if zero_net_moment else 'FM/FiM'
        classification_rule = 'fm_like_spin_point_group'
    else:
        base_phase = 'AFM'
        classification_rule = (
            'afm_with_spin_orbit_magnet'
            if som_by_mpg
            else 'default_antiferromagnetic'
        )

    ss_wo_soc = spin_splitting_wo_soc(base_phase, is_ss_gp)
    alter_tag = is_alter(conf, base_phase, ss_wo_soc)
    som_tag = '(SOM)' if base_phase == 'AFM' and som_by_mpg else ''
    phase = base_phase + alter_tag
    if som_tag:
        phase += '\n' + som_tag

    return {
        'phase': phase,
        'base_phase': base_phase,
        'modifier': alter_tag,
        'spin_orbit_magnet_tag': som_tag,
        'details': {
            'conf': conf,
            'full_spin_part_point_group_hm': full_spin_part_point_group_hm,
            'full_spin_part_point_group_s': full_spin_part_point_group_s,
            'mpg_identifier': mpg_identifier,
            'net_moment': net_moment_value,
            'zero_net_moment': zero_net_moment,
            'fm_like_by_spin_point_group': fm_like_by_spin_point_group,
            'som_by_mpg': som_by_mpg,
            'classification_rule': classification_rule,
            'base_phase': base_phase,
            'modifier': alter_tag,
            'spin_orbit_magnet_tag': som_tag,
            'spin_splitting_without_soc': ss_wo_soc,
            'is_altermagnet': bool(alter_tag),
            'is_spin_orbit_magnet': bool(som_tag),
        },
        'spin_splitting_without_soc': ss_wo_soc,
        'is_alter': alter_tag,
        'is_spin_orbit_magnet': som_tag,
    }


def get_magnetic_phase(full_spin_part_point_group_hm, full_spin_part_point_group_s, net_moment, mpg, conf=None, is_ss_gp=None):
    if conf is None or is_ss_gp is None:
        hm_symbol = full_spin_part_point_group_hm
        s_symbol = full_spin_part_point_group_s
        if full_spin_part_point_group_s is None:
            hm_symbol = None
            s_symbol = full_spin_part_point_group_hm
        return classify_magnetic_phase(
            conf='Unknown',
            full_spin_part_point_group_hm=hm_symbol,
            full_spin_part_point_group_s=s_symbol,
            net_moment=net_moment,
            mpg_identifier=mpg,
            is_ss_gp='spin splitting',
        )['base_phase']



def getNormInf(matrix1, matrix2, mode=True):
    if mode == True:
        a = np.array(matrix1) % 1
        b = np.array(matrix2) % 1
        c = [1, 2, 3]
        for i in range(3):
            if a[i] > b[i]:
                c[i] = min(a[i] - b[i], 1 + b[i] - a[i])
            if a[i] < b[i]:
                c[i] = min(b[i] - a[i], 1 + a[i] - b[i])
            if a[i] == b[i]:
                c[i] = 0
        max_value = max(c)
    else:
        diff = np.abs(matrix1 - matrix2)
        max_value = np.max(diff)
    return max_value

def combine_parametric_solutions(rref_matrix, tol=1e-3):
    import numpy as np

    A = np.array(rref_matrix, dtype=float)
    rows, cols = A.shape
    pivot_cols = []


    for i in range(rows):
        for j in range(cols):
            if abs(A[i, j]) > tol:
                pivot_cols.append(j)
                break

    pivot_cols = set(pivot_cols)
    free_vars = [j for j in range(cols) if j not in pivot_cols]


    symbols = ['Sx', 'Sy', 'Sz']
    vector_expr = ['0'] * cols

    for free_idx, var_col in enumerate(free_vars):
        coeffs = [0] * cols
        coeffs[var_col] = 1
        for row_idx in range(rows):
            row = A[row_idx]
            pivot_col = next((j for j in range(cols) if abs(row[j]) > tol), None)
            if pivot_col is not None and abs(row[var_col]) > tol:
                coeffs[pivot_col] = -row[var_col]

        if len(free_vars) == 1:
            first_nonzero_component = next(i for i, value in enumerate(coeffs) if abs(value) > tol)
            var_name = symbols[first_nonzero_component]
        else:
            var_name = symbols[free_idx]


        for i in range(cols):
            c = coeffs[i]
            if abs(c) < tol:
                continue
            if vector_expr[i] == '0':
                if abs(c - 1) < tol:
                    vector_expr[i] = var_name
                elif abs(c + 1) < tol:
                    vector_expr[i] = f"-{var_name}"
                else:
                    vector_expr[i] = f"{format_symbolic_scalar(c)}*{var_name}"
            else:
                if abs(c - 1) < tol:
                    vector_expr[i] += f" + {var_name}"
                elif abs(c + 1) < tol:
                    vector_expr[i] += f" - {var_name}"
                elif c > 0:
                    vector_expr[i] += f" + {format_symbolic_scalar(c)}*{var_name}"
                else:
                    vector_expr[i] += f" - {format_symbolic_scalar(abs(c))}*{var_name}"

    return vector_expr

def calculate_freedom_degree(matrices : list[np.ndarray],tol=0.01):
    """
        calculate freedom degree from matrices
    """
    stack_matrices = np.vstack(matrices-np.eye(3)).astype(np.float64)

    # rref(stack_matrices, tol=0.01)
    # pending for (mx,my,mz) representation
    constraints = combine_parametric_solutions(rref_with_tolerance(stack_matrices))
    return 3 - np.linalg.matrix_rank(stack_matrices,tol=tol), constraints

def get_spin_wyckoff(ssg_cell : CrystalCell, ssg_ops , atol =  0.001) -> (list, list):
    """
    Calculate spin Wyckoff positions information.

    Parameters:
        ssg_cell_spglib (list): A list containing cell information.
                         - ssg_cell[1]: Atomic positions (numpy array).
                         - ssg_cell[3]: Magnetic moments (numpy array).
        ssg_ops (list): A list of symmetry operations, where each operation is a np list (Rs ||Rr | t).

    Returns:
        Tuple[dict, dict]:
            - magnetic_index: A dictionary mapping magnetic atom indices to their multiplicities.
            - magnetic_index_site_symmetry: A dictionary mapping magnetic atom indices to their site symmetry operations.
    """

    if not ssg_cell or not ssg_ops:
        raise ValueError("Input ssg_cell and ssg_ops cannot be empty.")
    ssg_cell_spglib = ssg_cell.to_spglib(mag=True)

    coords = np.array(ssg_cell_spglib[1])
    atom_types = list(ssg_cell.atom_types)

    bins = max(1, int(np.ceil(1.0 / max(atol, 1e-12))))
    bucket_width = 1.0 / bins
    neighbor_radius = max(1, int(np.ceil(atol / bucket_width)))

    def _bucket_key(position):
        wrapped = np.mod(np.asarray(position, dtype=float), 1.0)
        indices = np.floor(wrapped * bins).astype(int) % bins
        return tuple(int(value) for value in indices)

    def _neighbor_keys(bucket_key):
        for dx in range(-neighbor_radius, neighbor_radius + 1):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                for dz in range(-neighbor_radius, neighbor_radius + 1):
                    yield (
                        (bucket_key[0] + dx) % bins,
                        (bucket_key[1] + dy) % bins,
                        (bucket_key[2] + dz) % bins,
                    )

    typed_position_buckets: dict[tuple, list[int]] = {}
    for index, coord in enumerate(coords):
        typed_position_buckets.setdefault((atom_types[index], _bucket_key(coord)), []).append(index)

    # Get indices of magnetic atoms and initialization

    magnetic_index = ssg_cell.magnetic_atom_indices

    num_atoms = len(coords)
    assigned = [False] * num_atoms
    equivalence_classes = []

    equivalence_classes_spin = []

    for i in range(num_atoms):
        if assigned[i]:
            continue
        class_i = []
        site_symmetry_ops = []
        for op in ssg_ops:
            Rr = np.array(op[1])
            t = np.array(op[2])
            trans = normalize_vector_to_zero(Rr @ coords[i] + t)
            candidate_indices = []
            seen_candidates = set()
            for neighbor_key in _neighbor_keys(_bucket_key(trans)):
                for candidate in typed_position_buckets.get((atom_types[i], neighbor_key), ()):
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    candidate_indices.append(candidate)
            candidate_indices.sort()
            for j in candidate_indices:
                dist = getNormInf(trans, coords[j])
                if dist < atol:
                    if j not in class_i:
                        class_i.append(j)
                        assigned[j] = True
                    # Collect every operation that stabilizes the
                    # representative site, not only the first one that also
                    # claims the class member.
                    if i == j:
                        site_symmetry_ops.append(np.array(op[0]))
                    break
        equivalence_classes.append({
            "representative_index": i,
            "class_indices": class_i,
            "site_symmetry_ops": site_symmetry_ops
        })
        if i in magnetic_index:
            equivalence_classes_spin.append({
                "representative_index": i,
                "class_indices": class_i,
                "site_symmetry_ops": site_symmetry_ops
            })
        # print(class_i)

    # Calculate site symmetry of representative magnetic atoms

    # get degree of freedom of moment
    magnetic_representative_dof = {}
    constraints = []
    for info in equivalence_classes_spin:
        dof, constraint = calculate_freedom_degree(info['site_symmetry_ops'], tol=atol)
        magnetic_representative_dof[info['representative_index']] = int(dof)
        constraints.append(constraint)

    return magnetic_index, equivalence_classes, magnetic_representative_dof,equivalence_classes_spin,constraints


def _identify_ssg_index_details(file_name,ssg_primitive:SpinSpaceGroup,tol = 0.001):
    """
    only for G0std_nofrac
    """
    from findspingroup.data.SG_SYMBOL import SGgeneratorDict
    from findspingroup.data.PG_SYMBOL import PG_SCH_TO_ID_INDEX
    from findspingroup.core.identify_index.functions.get_stand_trans import get_stand_trans

    def _normalized_direction(vector):
        direction = np.asarray(vector, dtype=float).reshape(-1)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Cannot normalize a zero-length identify-index direction.")
        return direction / norm

    def _eigen_direction(matrix, eigenvalue):
        eigenvalues, eigenvectors = np.linalg.eig(np.asarray(matrix, dtype=float))
        for idx, value in enumerate(eigenvalues):
            if np.isclose(value, eigenvalue, atol=tol):
                return _normalized_direction(eigenvectors[:, idx].real)
        raise ValueError(
            f"Cannot find eigen-direction with eigenvalue {eigenvalue} for "
            f"identify-index matrix {np.asarray(matrix, dtype=float).tolist()}."
        )

    def _resolve_order_two_coplanar_suffix(ssg_std_nofrac):
        if ssg_std_nofrac.conf != 'Coplanar' or len(ssg_std_nofrac.n_spin_part_point_ops) != 2:
            return None
        spin_only_mirror = next(
            op[0]
            for op in ssg_std_nofrac.sog
            if not np.allclose(op[0], np.eye(3), atol=tol)
        )
        spin_twofold = next(
            op[0]
            for op in ssg_std_nofrac.nssg
            if not np.allclose(op[0], np.eye(3), atol=tol)
        )
        mirror_normal = _eigen_direction(spin_only_mirror, -1.0)
        twofold_axis = _eigen_direction(spin_twofold, 1.0)
        alignment = abs(float(np.dot(mirror_normal, twofold_axis)))
        axis_tol = max(tol, 1e-3)
        if alignment >= 1.0 - axis_tol:
            return 'P1'
        if alignment <= axis_tol:
            return 'P2'
        raise ValueError(
            "Ambiguous P1/P2 classification for "
            f"{file_name}: |dot(mirror_normal, twofold_axis)|={alignment:.6f}."
        )

    def _canonicalize_axis_sign(direction):
        normalized = _normalized_direction(direction)
        for value in normalized:
            if abs(value) < max(tol, 1e-6):
                continue
            return normalized if value > 0 else -normalized
        return normalized

    def _classify_axis_aligned_mirror(matrix):
        matrix = np.asarray(matrix, dtype=float)
        if np.allclose(matrix, np.diag([1.0, 1.0, -1.0]), atol=tol):
            return "Mz"
        if np.allclose(matrix, np.diag([1.0, -1.0, 1.0]), atol=tol):
            return "My"
        if np.allclose(matrix, np.diag([-1.0, 1.0, 1.0]), atol=tol):
            return "Mx"
        raise ValueError(
            f"Unsupported Coplanar+D2 target spin-only matrix {matrix.tolist()} for {file_name}."
        )

    def _canonical_branch_spin_transform(target_spin_only_matrix):
        target_label = _classify_axis_aligned_mirror(target_spin_only_matrix)
        if target_label == "Mz":
            return np.eye(3), target_label
        if target_label == "My":
            return np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=float,
            ), target_label
        if target_label == "Mx":
            return np.array(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=float,
            ), target_label
        raise ValueError(
            f"Unsupported Coplanar+D2 target spin-only label {target_label!r} for {file_name}."
        )

    def _build_coplanar_d2_spin_normalization(current_ops):
        temp_group = IdentifyNoFracGroup(current_ops, conf='Coplanar', tol=tol)
        spin_only_mirror = _select_preferred_candidate(
            [
                op
                for op in temp_group.sog
                if not np.allclose(op[0], np.eye(3), atol=tol)
            ],
            [],
        )
        if spin_only_mirror is None:
            raise ValueError(
                f"Cannot find nontrivial spin-only mirror for Coplanar+D2 identify branch in {file_name}."
            )
        mirror_normal = _canonicalize_axis_sign(_eigen_direction(spin_only_mirror[0], -1.0))

        in_plane_axis = None
        for op in sorted(
            [
                op
                for op in temp_group.nssg
                if not np.allclose(op[0], np.eye(3), atol=tol)
            ],
            key=_operation_candidate_sort_key,
        ):
            candidate_axis = _eigen_direction(op[0], 1.0)
            projected = candidate_axis - float(np.dot(candidate_axis, mirror_normal)) * mirror_normal
            if np.linalg.norm(projected) > max(tol, 1e-4):
                in_plane_axis = _canonicalize_axis_sign(projected)
                break

        if in_plane_axis is None:
            for basis_vector in np.eye(3):
                projected = basis_vector - float(np.dot(basis_vector, mirror_normal)) * mirror_normal
                if np.linalg.norm(projected) > max(tol, 1e-4):
                    in_plane_axis = _canonicalize_axis_sign(projected)
                    break
        if in_plane_axis is None:
            raise ValueError(
                f"Cannot determine in-plane reference axis for Coplanar+D2 identify branch in {file_name}."
            )

        z_axis = mirror_normal
        x_axis = _normalized_direction(in_plane_axis)
        y_axis = _normalized_direction(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < max(tol, 1e-4):
            raise ValueError(
                f"Degenerate Coplanar+D2 spin normalization frame in {file_name}."
            )
        y_axis = _canonicalize_axis_sign(y_axis)
        frame = np.vstack([x_axis, y_axis, z_axis])
        if np.linalg.det(frame) < 0:
            y_axis = -y_axis
            frame = np.vstack([x_axis, y_axis, z_axis])
        return frame, mirror_normal

    def _transform_ops_preserving_integer_shifts(ops, transformation_matrix, origin_shift, frac=True, all_trans=True):
        transformation_matrix = np.asarray(transformation_matrix, dtype=float)
        origin_shift = np.asarray(origin_shift, dtype=float)
        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        if frac:
            lattice_shifts = integer_points_in_new_cell(transformation_matrix_inv.T)
        else:
            lattice_shifts = [np.zeros(3)]
        if not all_trans:
            lattice_shifts = [np.zeros(3)]

        transformed_ops = []
        for op in ops:
            spin_rotation = np.asarray(op[0], dtype=float)
            real_rotation = np.asarray(op[1], dtype=float)
            translation = np.asarray(op[2], dtype=float)
            for lattice_shift in lattice_shifts:
                lifted_translation = translation + np.asarray(lattice_shift, dtype=float)
                new_rotation = transformation_matrix @ real_rotation @ transformation_matrix_inv
                if frac:
                    new_translation = normalize_vector_to_zero(
                        ((np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ lifted_translation),
                        atol=1e-4,
                    )
                else:
                    new_translation = (
                        (np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ lifted_translation
                    )
                transformed_ops.append(
                    SpinSpaceGroupOperation(
                        spin_rotation,
                        new_rotation,
                        new_translation,
                    )
                )
        return transformed_ops

    def _transform_spin_ops_preserving_order(ops, spin_transformation_matrix):
        spin_transformation_matrix = np.asarray(spin_transformation_matrix, dtype=float)
        spin_transformation_matrix_inv = np.linalg.inv(spin_transformation_matrix)
        transformed_ops = []
        for op in ops:
            transformed_ops.append(
                SpinSpaceGroupOperation(
                    spin_transformation_matrix @ np.asarray(op[0], dtype=float) @ spin_transformation_matrix_inv,
                    np.asarray(op[1], dtype=float),
                    np.asarray(op[2], dtype=float),
                )
            )
        return transformed_ops

    def _operation_candidate_sort_key(op):
        spin_rotation = np.asarray(op[0], dtype=float)
        nontrivial_spin = not np.allclose(spin_rotation, np.eye(3), atol=tol)
        spin_distance = float(np.linalg.norm(spin_rotation - np.eye(3)))
        spin_signature = tuple(np.round(spin_rotation, 6).flatten())
        return (
            0 if nontrivial_spin else 1,
            -round(spin_distance, 6),
            spin_signature,
        )

    def _select_preferred_candidate(exact_candidates, equivalent_candidates):
        candidates = exact_candidates or equivalent_candidates
        if not candidates:
            return None
        return sorted(candidates, key=_operation_candidate_sort_key)[0]

    def _match_name_generators(sg_num: int, canonical_nssg_ops, pure_translation_lattice):
        """
        identify name_maps follow the canonical point-group generator
        convention used by the identify database / point-group table.
        """
        sg_info = SGgeneratorDict[sg_num]
        generators = []
        for ind in range((len(sg_info) - 1) // 2):
            gen_rot, gen_t = eval(sg_info[2 * ind + 2])
            gen_t = np.array(gen_t, dtype=float)
            gen_rot = np.array(gen_rot).reshape((3, 3))
            generators.append([gen_rot, gen_t])

        matched_generators = []
        for gen_rot, gen_t in generators:
            exact_candidates = []
            equivalent_candidates = []
            for op in canonical_nssg_ops:
                if not np.allclose(gen_rot, op[1], atol=tol):
                    continue
                if _exact_translation_distance(gen_t, op[2]) < tol:
                    exact_candidates.append(op)
                    continue
                if _translations_equivalent_mod_pure_translations(
                    gen_t,
                    op[2],
                    pure_translation_lattice,
                    tol,
                ):
                    equivalent_candidates.append(op)
            preferred = _select_preferred_candidate(exact_candidates, equivalent_candidates)
            if preferred is None:
                raise ValueError(
                    f"Cannot find canonical identify generator {(gen_rot.tolist(), gen_t.tolist())} "
                    "in G0std_nofrac.nssg."
                )
            matched_generators.append(SpinSpaceGroupOperation(preferred[0], preferred[1], preferred[2]))
        return matched_generators

    def _match_translation_generators(translation_source_ops, lattice_translations):
        """
        identify translation_maps encode the SSG operations corresponding to the
        three nofrac lattice translations [1,0,0], [0,1,0], [0,0,1].
        """
        generators_trans = [
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        ]
        matched_translations = []
        for gen_t in generators_trans:
            exact_candidates = []
            equivalent_candidates = []
            for op in translation_source_ops:
                if not np.allclose(op[1], np.eye(3), atol=tol):
                    continue
                translation = np.asarray(op[2], dtype=float)
                if _exact_translation_distance(gen_t, translation) < tol:
                    exact_candidates.append(op)
                    continue
                for lattice_t in lattice_translations:
                    if _exact_translation_distance(gen_t, translation - lattice_t) < tol:
                        equivalent_candidates.append(op)
                        break
            preferred = _select_preferred_candidate(exact_candidates, equivalent_candidates)
            if preferred is None:
                matched_translations.append(SpinSpaceGroupOperation(np.eye(3), np.eye(3), gen_t))
            else:
                matched_translations.append(SpinSpaceGroupOperation(preferred[0], preferred[1], gen_t))
        return matched_translations
    spin_T = np.eye(3)
    identify_ops_nofrac = _transform_ops_preserving_integer_shifts(
        ssg_primitive._input_ops,
        ssg_primitive.transformation_to_G0std,
        ssg_primitive.origin_shift_to_G0std,
    )
    identify_ops_nofrac = _transform_ops_preserving_integer_shifts(
        identify_ops_nofrac,
        ssg_primitive.transformation_to_G0std_id @ np.linalg.inv(ssg_primitive.transformation_to_G0std),
        np.array([0, 0, 0]),
        frac=False,
    )
    identify_ops_nofrac = _transform_spin_ops_preserving_order(
        identify_ops_nofrac,
        np.linalg.inv(ssg_primitive.n_spin_part_std_transformation),
    )
    G0_num = ssg_primitive.G0_num
    L0_num = ssg_primitive.L0_num
    it = ssg_primitive.it
    ik = ssg_primitive.ik
    nsspg_order = len(ssg_primitive.n_spin_part_point_ops)
    use_222_contract = (
        ssg_primitive.conf == 'Coplanar'
        and ssg_primitive.n_spin_part_point_group_symbol_s == 'D2'
        and has_coplanar_222_lookup_group((L0_num, G0_num), (nsspg_order, it, ik))
    )
    coplanar_d2_spin_normalization = np.eye(3)
    if use_222_contract:
        coplanar_d2_spin_normalization, _ = _build_coplanar_d2_spin_normalization(identify_ops_nofrac)
        identify_ops_nofrac = _transform_spin_ops_preserving_order(
            identify_ops_nofrac,
            coplanar_d2_spin_normalization,
        )
    identify_nofrac_group = IdentifyNoFracGroup(
        identify_ops_nofrac,
        conf=ssg_primitive.conf,
        tol=tol,
    )
    identify_generator_source_ops = list(identify_nofrac_group.nssg)
    identify_translation_lattice = identify_nofrac_group.pure_translations
    if it * ik != nsspg_order:
        raise ValueError(
            "Inconsistent NSSPG invariants for "
            f"{file_name}: it*ik={it * ik}, |nsspg|={nsspg_order}, "
            f"it={it}, ik={ik}, spin_pg={ssg_primitive.n_spin_part_point_group_symbol_s}, "
            f"G0={G0_num}, L0={L0_num}."
        )
    coplanar_suffix = _resolve_order_two_coplanar_suffix(identify_nofrac_group)
    pg = PG_SCH_TO_ID_INDEX[ssg_primitive.n_spin_part_point_group_symbol_s] # map to identify-pg list
    name_generators = _match_name_generators(
        G0_num,
        identify_generator_source_ops,
        identify_translation_lattice,
    )
    translation_generators = _match_translation_generators(
        identify_generator_source_ops,
        identify_translation_lattice,
    )
    generators_hm = [[(spin_T@ i[0]@np.linalg.inv(spin_T)).round(5).tolist(),[i[1].round(5).tolist(),i[2].round(5).tolist()]] for i in name_generators]
    generators_lattice =[[(spin_T@ i[0]@np.linalg.inv(spin_T)).round(5).tolist(),[i[1].round(5).tolist(),i[2].round(5).tolist()]] for i in  translation_generators]
    transformation_G0std_to_L0std = [ssg_primitive.G0std_L0std_transformation.round(5),ssg_primitive.G0std_L0std_origin_shift.round(5)]

    transformation_L0std_to_G0std = np.linalg.inv(np.block([[transformation_G0std_to_L0std[0],transformation_G0std_to_L0std[1].reshape(3,1)],[np.zeros((1,3)), np.ones((1,1))]]))
    transformation_L0std_to_G0std = [transformation_L0std_to_G0std[:3,:3].tolist(),transformation_L0std_to_G0std[:3,3:4].reshape(3).tolist()]

    data = {
        'filename':file_name,
        'L0_id':L0_num,
        'G0_id':G0_num,
        't_index':it,
        'k_index':ik,
        'point_group_id':pg,
        'name_maps':generators_hm,
        'translation_maps':generators_lattice,
        'transformation_matrix':transformation_L0std_to_G0std
    }
    L0_id, G0_id, it, ik, iso,T = data['L0_id'], data['G0_id'], data['t_index'], data['k_index'], data['point_group_id'],  data['transformation_matrix']
    name_maps, translation_maps = data['name_maps'], data['translation_maps']

    if ssg_primitive.conf == 'Collinear':
        last_index = '.L'
    elif ssg_primitive.conf == 'Coplanar':
        last_index = f'.{coplanar_suffix}' if coplanar_suffix is not None else '.P'
    else:
        last_index = ''
    map_result = get_stand_trans(
        L0_id,
        G0_id,
        it,
        ik,
        iso,
        T,
        name_maps,
        translation_maps,
        tol=tol,
        use_222_contract=use_222_contract,
        return_map_info=use_222_contract,
    )
    if use_222_contract:
        map_num, trans1, trans2, identify_map_info = map_result
        lookup_entry = get_coplanar_222_lookup_entry(
            (L0_id, G0_id),
            (nsspg_order, it, ik),
            map_num,
        )
        if lookup_entry is None:
            raise ValueError(
                "Missing Coplanar+D2 Excel-backed lookup entry for "
                f"(L0,G0)=({L0_id},{G0_id}), (total,t,k)=({nsspg_order},{it},{ik}), map_num={map_num}."
            )
        last_index = f".{lookup_entry['configuration_suffix']}"
        final_index = str(lookup_entry["final_index"])
        final_index_parts = final_index.split(".")
        if len(final_index_parts) != 5:
            raise ValueError(
                f"Unexpected Coplanar+D2 lookup index format {final_index!r} for {file_name}."
            )
        map_num = int(final_index_parts[3])
        trans2_raw = np.asarray(trans2, dtype=float)
        q_transform = np.asarray(coplanar_d2_spin_normalization, dtype=float)
        branch_spin_transform, branch_spin_target = _canonical_branch_spin_transform(
            lookup_entry["spin_only_matrix"]
        )
        trans2 = branch_spin_transform @ trans2_raw @ q_transform
        final_index_string = final_index
    else:
        map_num, trans1, trans2 = map_result
        final_index_string = f'{G0_id}.{L0_id}.{ik}.{map_num}{last_index}'

    public_space_transform, public_space_shift = _identify_affine_4x4_to_setting_transform(trans1)

    return {
        'filename': file_name,
        'index': final_index_string,
        'configuration': ssg_primitive.conf,
        'G0_id': G0_id,
        'L0_id': L0_id,
        't_index': it,
        'k_index': ik,
        'point_group_id': pg,
        'equivalent_map_index': map_num,
        'configuration_suffix': last_index.lstrip('.'),
        'name_maps': name_maps,
        'translation_maps': generators_lattice,
        'transformation_matrix': transformation_L0std_to_G0std,
        'space_group_transformation': [
            np.asarray(public_space_transform, dtype=float).tolist(),
            np.asarray(public_space_shift, dtype=float).tolist(),
        ],
        'space_group_transformation_raw_4x4': np.asarray(trans1, dtype=float).tolist(),
        'point_group_transformation': np.asarray(trans2, dtype=float).tolist(),
        'point_group_transformation_raw': (
            np.asarray(trans2_raw, dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_q_transform': (
            np.asarray(q_transform, dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_b_transform': (
            np.asarray(branch_spin_transform, dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_target_spin_only_matrix': (
            np.asarray(lookup_entry["spin_only_matrix"], dtype=float).tolist() if use_222_contract else None
        ),
        'coplanar_222_target_spin_only_label': (
            branch_spin_target if use_222_contract else None
        ),
        'equivalent_map_resolution': 'database',
        'canonical_transformations_available': True,
        'special_record': None,
    }


def _identify_ssg_index(file_name,ssg_primitive:SpinSpaceGroup,tol = 0.001):
    return _identify_ssg_index_details(file_name, ssg_primitive, tol=tol)['index']





def get_G0_dataset_for_cell(space_group_operations, cell, symprec):
    # weirdSite = np.array([0.4275710, 0.591580, 0.233338700])
    weirdSite = np.array([0.1715870, 0.27754210, 0.737388700])
    # weirdSite = np.array([0.1, 0.2, 0.7])
    # weirdSite = np.array([0,0,0])
    defaultpos = [i for i in cell[1]]
    defaulttypes = [i for i in cell[2]]
    # print(defaulttypes)
    typesForGerator = [max(defaulttypes) + 1]
    # print(typesForGerator)
    generatePosition = [weirdSite]
    for i in space_group_operations:
        # print(i)
        temp = normalize_vector_to_zero(i[0]@weirdSite+i[1] ,atol=1e-8)
        # print(temp)
        if not any(np.allclose(temp, j, atol=1e-4) for j in generatePosition):
            generatePosition.append(temp)
            typesForGerator.append(max(defaulttypes) + 1)
    cells = (cell[0], defaultpos + generatePosition, defaulttypes + typesForGerator)
    space_group_dataset =get_symmetry_dataset(cells, symprec=symprec)
    if space_group_dataset.number in SG_HALL_MAPPING:
        space_group_dataset =get_symmetry_dataset(cells, symprec=symprec, hall_number=SG_HALL_MAPPING[space_group_dataset.number])

    return space_group_dataset

#------------------
# Wyckoff
def get_wp_from_dataset(dataset,max=True):
    temp_eq = {}
    first_index = {}
    last_index = 0
    wp_temp=[]
    for ind, eq_label in enumerate(dataset.equivalent_atoms):
        if eq_label not in temp_eq:
            temp_eq[eq_label] = 1
            first_index[eq_label] = ind
            last_index = ind
        else:
            temp_eq[eq_label] += 1
    di = {key:str(value)+ dataset.wyckoffs[first_index[key]]for key,value in temp_eq.items()}

    if max:
        wp = [(di[i],i) for i in dataset.equivalent_atoms[:last_index]]
    else:
        wp = [(di[i],i) for i in dataset.equivalent_atoms]
    return wp


def wyckoff_analysis(ssg_cell: CrystalCell, ssg: SpinSpaceGroup, rtol=0.02):
    from spglib import get_symmetry_dataset,get_magnetic_symmetry_dataset
    sg_dataset = get_symmetry_dataset(ssg_cell.to_spglib())
    msg_dataset_magnetic = get_magnetic_symmetry_dataset(ssg_cell.to_spglib(mag=True),symprec=rtol)
    if msg_dataset_magnetic is None:
        raise ValueError("Magnetic symmetry dataset could not be determined during wyckoff analysis.")
    msg_dataset = get_G0_dataset_for_cell(ssg_cell.to_spglib(),[i for i in zip(msg_dataset_magnetic.rotations,msg_dataset_magnetic.translations)])
    ssg_dataset = get_G0_dataset_for_cell(ssg.G0_ops,ssg_cell.to_spglib(mag=True),rtol)
    if ssg_dataset.number != ssg.G0_num:
        raise ValueError(f"Warning: Wyckoff analysis found different space group number!From cell: {ssg_dataset.number}, From SSG: {ssg.G0_num}")
    wp_extended_sg = get_wp_from_dataset(sg_dataset,max=False)
    wp_extended_ssg =get_wp_from_dataset(ssg_dataset,max=True)
    wp_extended_msg = get_wp_from_dataset(msg_dataset,max=True)
#--------------------

def get_msg_from_ossg(ossg_ops,tol=0.01):
    """
    Get magnetic space group operations from oriented spin space group operations.

    Parameters:
    ossg_ops (list): A list of oriented spin space group operations.
    tol (float): Tolerance for numerical comparisons.

    Returns:
    list: A list of operations satisfying the MSG condition Rs = +/- Rr.
    """
    msg_ops = []
    for op in ossg_ops:
        if op.magnetic_time_reversal(atol=tol) is not None:
            msg_ops.append(op)
    return msg_ops


def _find_spin_group_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    source_metadata: dict | None = None,
    parser_atol: float | None = None,
    input_spin_setting: str = "in_lattice",
) -> MagSymmetryResult:
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    magnetic_primitive_cell: CrystalCell
    magnetic_primitive_cell,Tmatrix_Tp_input__p_primitive = input_cell.get_primitive_structure(magnetic=True)
    identify_result = identify_spin_space_group_result(
        magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    ssg_primitive: SpinSpaceGroup = identify_result.ssg
    input_space_group = identify_result.input_space_group
    input_space_group_number = None if input_space_group is None else input_space_group.number
    input_space_group_symbol = None if input_space_group is None else input_space_group.symbol
    input_space_group_basis_or_setting = (
        None if input_space_group is None else input_space_group.basis_or_setting
    )

    try:
        msg_dataset_primitive = get_magnetic_symmetry_dataset(
            magnetic_primitive_cell.to_spglib(mag=True),
            symprec=tol_cfg.space,
            mag_symprec=tol_cfg.moment,
        )
        spglib_msg_num = msg_dataset_primitive.uni_number
        mpg_symbol = MSGMPG_DB.OG_NUM_TO_MPG[
            MSGMPG_DB.BNS_TO_OG_NUM[MSGMPG_DB.MSG_INT_TO_BNS[spglib_msg_num][0]]
        ]["pointgroup_no"]
    except Exception:
        mpg_symbol = None

    magnetic_phase_payload = classify_magnetic_phase(
        conf=ssg_primitive.conf,
        full_spin_part_point_group_hm=ssg_primitive.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=ssg_primitive.spin_part_point_group_symbol_s,
        net_moment=magnetic_primitive_cell.net_moment,
        mpg_identifier=mpg_symbol,
        is_ss_gp=ssg_primitive.is_spinsplitting[-1],
    )
    magnetic_phase = magnetic_phase_payload['phase']
    magnetic_phase_base = magnetic_phase_payload['base_phase']
    magnetic_phase_modifier = magnetic_phase_payload['modifier']
    magnetic_phase_details = magnetic_phase_payload['details']
    ss_w_soc = spin_splitting_w_soc(ssg_primitive)
    ahc_w_soc = is_ahc(mpg_symbol)
    ss_wo_soc = magnetic_phase_payload['spin_splitting_without_soc']
    ahc_wo_soc = is_ahc(ssg_primitive.gspg.empg_symbol)
    alter = magnetic_phase_payload['is_alter']


    transformation_input_to_primitive = (
        Tmatrix_Tp_input__p_primitive,
        np.zeros(3),
    )
    input_cell_cartesian = _cartesianized_input_cell(input_cell)
    identify_index_details = None
    identify_info = None
    try:
        identify_index_details = _identify_ssg_index_details(
            source_name,
            ssg_primitive,
            tol=tol_cfg.m_matrix_tol,
        )
        identify_info = identify_index_details['index']
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        warnings.warn(
            f"Identify-index output unavailable for {source_name}: {exc}. "
            "Continuing with identify-index-derived outputs set to None.",
            RuntimeWarning,
            stacklevel=2,
        )
    input_magnetic_primitive_poscar = magnetic_primitive_cell.to_poscar(source_name)
    raw_transformation_primitive_to_G0std = (
        np.asarray(ssg_primitive.transformation_to_G0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_G0std, dtype=float),
    )
    raw_transformation_primitive_to_L0std = (
        np.asarray(ssg_primitive.transformation_to_L0std, dtype=float),
        np.asarray(ssg_primitive.origin_shift_to_L0std, dtype=float),
    )
    raw_G0std_cell = magnetic_primitive_cell.transform(*raw_transformation_primitive_to_G0std)
    raw_L0std_cell = magnetic_primitive_cell.transform(*raw_transformation_primitive_to_L0std)
    raw_G0std_ssg = ssg_primitive.transform(*raw_transformation_primitive_to_G0std)
    raw_L0std_ssg = ssg_primitive.transform(*raw_transformation_primitive_to_L0std)

    raw_transformation_input_to_G0std = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        raw_transformation_primitive_to_G0std[0],
        raw_transformation_primitive_to_G0std[1],
    )
    raw_transformation_input_to_L0std = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        raw_transformation_primitive_to_L0std[0],
        raw_transformation_primitive_to_L0std[1],
    )

    allow_input_collapse = _acc_setting_allows_input_collapse(ssg_primitive.acc)

    G0std_cell, G0std_ssg, transformation_input_to_G0std, _ = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        raw_G0std_cell,
        raw_G0std_ssg,
        raw_transformation_input_to_G0std,
        allow_identity_collapse=allow_input_collapse,
    )
    L0std_cell, L0std_ssg, transformation_input_to_L0std, _ = _canonicalize_input_to_standard_setting(
        input_cell_cartesian,
        raw_L0std_cell,
        raw_L0std_ssg,
        raw_transformation_input_to_L0std,
        allow_identity_collapse=allow_input_collapse,
    )

    transformation_primitive_to_acc_primitive = (
        np.asarray(ssg_primitive.acc_primitive_trans, dtype=float),
        np.asarray(ssg_primitive.acc_primitive_origin_shift, dtype=float),
    )
    acc_magnetic_primitive_cell = magnetic_primitive_cell.transform(*transformation_primitive_to_acc_primitive)
    acc_magnetic_primitive_ssg = ssg_primitive.transform(*transformation_primitive_to_acc_primitive)
    transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        transformation_primitive_to_acc_primitive[0],
        transformation_primitive_to_acc_primitive[1],
    )
    transformation_G0std_to_primitive = _compose_setting_transform(
        transformation_input_to_G0std[0],
        transformation_input_to_G0std[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    transformation_L0std_to_primitive = _compose_setting_transform(
        transformation_input_to_L0std[0],
        transformation_input_to_L0std[1],
        transformation_input_to_acc_primitive[0],
        transformation_input_to_acc_primitive[1],
    )
    acc_p_c_poscar = acc_magnetic_primitive_cell.to_poscar(source_name)
    acc_real_cartesian_to_poscar_spin_frame = _poscar_spin_frame_rotation(acc_magnetic_primitive_cell)
    poscar_spin_frame_to_acc_real_cartesian = np.linalg.inv(acc_real_cartesian_to_poscar_spin_frame)
    acc_magnetic_primitive_ssg_in_poscar_spin_frame = acc_magnetic_primitive_ssg.transform_spin(
        acc_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(
        acc_magnetic_primitive_ssg,
        acc_magnetic_primitive_cell,
    )
    internal_msg_info = acc_primitive_ossg.msg_info
    msg_num = None if internal_msg_info is None else internal_msg_info.get("msg_int_num")
    msg_type = None if internal_msg_info is None else internal_msg_info.get("msg_type")
    msg_symbol = None if internal_msg_info is None else internal_msg_info.get("msg_bns_symbol")
    selected_standard_setting = (
        L0_STANDARD_SETTING
        if acc_magnetic_primitive_ssg.international_symbol_type == "k"
        else G0_STANDARD_SETTING
    )
    if selected_standard_setting == G0_STANDARD_SETTING:
        selected_standard_cell = G0std_cell
        selected_standard_ssg = G0std_ssg
        transformation_input_to_selected_standard = transformation_input_to_G0std
        transformation_selected_standard_to_acc_primitive = transformation_G0std_to_primitive
    else:
        selected_standard_cell = L0std_cell
        selected_standard_ssg = L0std_ssg
        transformation_input_to_selected_standard = transformation_input_to_L0std
        transformation_selected_standard_to_acc_primitive = transformation_L0std_to_primitive

    actual_transformation_acc_primitive_to_acc_conventional = (
        np.asarray(acc_magnetic_primitive_ssg.cptrans, dtype=float),
        np.zeros(3),
    )
    actual_acc_conventional_cell = acc_magnetic_primitive_cell.transform(
        *actual_transformation_acc_primitive_to_acc_conventional
    )
    actual_acc_conventional_ssg = acc_magnetic_primitive_ssg.transform(
        *actual_transformation_acc_primitive_to_acc_conventional
    )
    actual_transformation_selected_standard_to_acc_conventional = _chain_setting_transform(
        transformation_selected_standard_to_acc_primitive[0],
        transformation_selected_standard_to_acc_primitive[1],
        actual_transformation_acc_primitive_to_acc_conventional[0],
        actual_transformation_acc_primitive_to_acc_conventional[1],
    )
    actual_selected_standard_to_acc_conventional_audit = audit_spatial_transform_effect(
        selected_standard_ssg,
        actual_transformation_selected_standard_to_acc_conventional[0],
        actual_transformation_selected_standard_to_acc_conventional[1],
        tol=tol_cfg.m_matrix_tol,
        use_nssg=False,
    )
    actual_selected_standard_to_acc_conventional_label = (
        "self_automorphism"
        if actual_selected_standard_to_acc_conventional_audit["real_ops_exact_same"]
        else "setting_change"
    )
    acc_primitive_output_cell = acc_magnetic_primitive_cell
    acc_primitive_output_ssg = acc_magnetic_primitive_ssg
    transformation_input_to_acc_primitive_output = transformation_input_to_acc_primitive
    transformation_G0std_to_acc_primitive_output = transformation_G0std_to_primitive
    transformation_L0std_to_acc_primitive_output = transformation_L0std_to_primitive
    acc_conventional_cell = actual_acc_conventional_cell
    acc_conventional_ssg = actual_acc_conventional_ssg
    transformation_selected_standard_to_acc_conventional = actual_transformation_selected_standard_to_acc_conventional
    selected_standard_to_acc_conventional_audit = actual_selected_standard_to_acc_conventional_audit
    selected_standard_to_acc_conventional_label = actual_selected_standard_to_acc_conventional_label
    convention_setting = selected_standard_setting
    convention_cell = selected_standard_cell
    convention_cell_snapshot = _serialize_cell_snapshot(selected_standard_cell)
    convention_ssg = selected_standard_ssg
    transformation_input_to_convention = transformation_input_to_selected_standard
    transformation_convention_to_primitive = transformation_selected_standard_to_acc_primitive
    transformation_convention_to_acc_conventional = transformation_selected_standard_to_acc_conventional
    convention_to_acc_conventional_audit = selected_standard_to_acc_conventional_audit
    convention_to_acc_conventional_label = selected_standard_to_acc_conventional_label

    public_ossg_ssg = _ossg_oriented_spin_frame_ssg(convention_ssg, convention_cell)
    try:
        msg_acc = SpinSpaceGroup(public_ossg_ssg.msg_ops).acc
    except Exception:
        msg_acc = None
    gspg_payload = _build_gspg_payload(
        public_ossg_ssg,
        real_space_setting=convention_setting,
        spin_frame_setting=OSSG_ORIENTED_SPIN_FRAME_SETTING,
    )
    msg_parent_info = msg_parent_space_group_info(msg_num)
    ossg_space_group_number = None if identify_index_details is None else identify_index_details.get("G0_id")
    acc_primitive_output_poscar = acc_primitive_output_cell.to_poscar(source_name)
    acc_output_real_cartesian_to_poscar_spin_frame = _poscar_spin_frame_rotation(acc_primitive_output_cell)
    poscar_spin_frame_to_acc_output_real_cartesian = np.linalg.inv(
        acc_output_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_output_ssg_in_poscar_spin_frame = acc_primitive_output_ssg.transform_spin(
        acc_output_real_cartesian_to_poscar_spin_frame
    )

    KPOINTS = acc_primitive_output_ssg.KPOINTS
    SS =  acc_primitive_output_ssg.spin_polarizations
    SS_poscar = acc_primitive_output_ssg_in_poscar_spin_frame.spin_polarizations
    primitive_msg_ops, msg_little_groups, msg_little_group_symbols = _build_msg_little_group_core(
        acc_primitive_ossg,
        tol=tol_cfg.m_matrix_tol,
    )
    msg_spin_polarizations = _get_spin_constraint_for_msg_little_groups(
        msg_little_groups,
        acc_magnetic_primitive_cell,
        tol=tol_cfg.m_matrix_tol,
    )
    msg_spin_polarizations_poscar = _get_spin_constraint_for_msg_little_groups(
        msg_little_groups,
        acc_magnetic_primitive_cell,
        tol=tol_cfg.m_matrix_tol,
        spin_frame_rotation=acc_real_cartesian_to_poscar_spin_frame,
    )
    tensor_outputs = _compute_tensor_outputs(
        acc_magnetic_primitive_ssg,
        acc_magnetic_primitive_cell,
        tol=tol_cfg.m_matrix_tol,
    )

    convention_nssg_ops = public_ossg_ssg.nssg
    convention_nssg_seitz, convention_nssg_seitz_latex = _serialize_op_list_seitz_symbols(
        convention_nssg_ops,
        tol=public_ossg_ssg.symbol_calibration_tol,
    )

    scif_export_targets = _build_scif_export_targets(
        input_cell=input_cell,
        magnetic_primitive_cell=magnetic_primitive_cell,
        ssg_primitive=ssg_primitive,
        acc_magnetic_primitive_cell=acc_magnetic_primitive_cell,
        acc_magnetic_primitive_ssg=acc_magnetic_primitive_ssg,
        G0std_cell=G0std_cell,
        G0std_ssg=G0std_ssg,
        L0std_cell=L0std_cell,
        L0std_ssg=L0std_ssg,
        transformation_input_to_primitive=transformation_input_to_primitive,
        transformation_input_to_acc_primitive=transformation_input_to_acc_primitive,
        transformation_input_to_G0std=transformation_input_to_G0std,
        transformation_input_to_L0std=transformation_input_to_L0std,
    )
    wp_chain = _build_wp_chain_payload(G0std_cell, G0std_ssg, tol_cfg)

    scif_outputs = {}
    for cell_mode, export_target in scif_export_targets.items():
        export_cell = export_target["export_cell"]
        export_ssg = export_target["export_ssg"]
        export_wyckoff = get_spin_wyckoff(export_cell, export_ssg.ops)
        source_parent_space_group = (
            None if source_metadata is None else source_metadata.get("parent_space_group")
        )
        generated_parent_space_group, parent_space_group_comparison = (
            _identify_parent_space_group_for_export_cell(
                export_cell,
                symprec=tol_cfg.space,
                source_parent_space_group=source_parent_space_group,
                reuse_source_transforms=(cell_mode == SCIF_CELL_MODE_INPUT),
            )
        )
        source_cell_parameter_strings = (
            None
            if source_metadata is None or cell_mode != SCIF_CELL_MODE_INPUT
            else source_metadata.get("cell_parameter_strings")
        )
        scif_outputs[cell_mode] = generate_scif(
            source_name,
            export_cell,
            export_ssg,
            export_wyckoff,
            export_target["basis_tag_transforms"],
            ssg_primitive,
            spin_space_group_index=identify_info,
            spin_space_group_name=public_ossg_ssg.international_symbol_linear_current_frame,
            spin_space_group_name_linear=public_ossg_ssg.international_symbol_linear_current_frame,
            spin_space_group_name_latex=public_ossg_ssg.international_symbol_latex_current_frame,
            magnetic_phase=magnetic_phase,
            identify_index_details=identify_index_details,
            source_cell_parameter_strings=source_cell_parameter_strings,
            parent_space_group=generated_parent_space_group,
            source_parent_space_group=source_parent_space_group,
            parent_space_group_comparison=parent_space_group_comparison,
        )

    scif = scif_outputs[SCIF_CELL_MODE_G0STD_ORIENTED]


    result = {
        'index':ssg_primitive.index,
        'spin_part_pg':ssg_primitive.spin_part_point_group_symbol_hm,
        'conf':ssg_primitive.conf,
        'id_index_info':identify_info,
        'scif':scif,
        'scif_outputs': scif_outputs,
        'scif_cell_modes': sorted(scif_export_targets.keys()),
        'poscar_mp':acc_primitive_output_poscar,
        'acc':ssg_primitive.acc,
        'msg_acc': msg_acc,
        'KPOINTS':KPOINTS
    }

    cell = {
        'input_magnetic_primitive_cell': magnetic_primitive_cell.to_spglib(mag=True),
        'input_magnetic_primitive_cell_setting': INPUT_MAGNETIC_PRIMITIVE_SETTING,
        'input_magnetic_primitive_cell_poscar': input_magnetic_primitive_poscar,
        'input_magnetic_primitive_cell_detail': _serialize_cell_snapshot(magnetic_primitive_cell),
        'acc_conventional_cell': acc_conventional_cell.to_spglib(mag=True),
        'acc_conventional_cell_setting': ACC_CONVENTIONAL_SETTING,
        'acc_conventional_cell_detail': _serialize_cell_snapshot(acc_conventional_cell),
        'magnetic_primitive_cell': acc_magnetic_primitive_cell.to_spglib(mag=True),
        'magnetic_primitive_cell_setting': ACC_PRIMITIVE_SETTING,
        'magnetic_primitive_cell_poscar': acc_p_c_poscar,
        'magnetic_primitive_cell_detail': _serialize_cell_snapshot(acc_magnetic_primitive_cell),
        'primitive_magnetic_cell':acc_magnetic_primitive_cell.to_spglib(mag=True),
        'primitive_magnetic_cell_setting': ACC_PRIMITIVE_SETTING,
        'primitive_magnetic_cell_poscar':acc_p_c_poscar,
        'scif': scif,
        'scif_outputs': scif_outputs,
        'scif_cell_modes': sorted(scif_export_targets.keys()),
        'primitive_magnetic_cell_detail': _serialize_cell_snapshot(acc_magnetic_primitive_cell),
        'acc_primitive_magnetic_cell': acc_primitive_output_cell.to_spglib(mag=True),
        'acc_primitive_magnetic_cell_setting': ACC_PRIMITIVE_SETTING,
        'acc_primitive_magnetic_cell_poscar': acc_primitive_output_poscar,
        'acc_primitive_magnetic_cell_detail': _serialize_cell_snapshot(acc_primitive_output_cell),
        'g0_standard_cell': _serialize_cell_snapshot(G0std_cell),
        'l0_standard_cell': _serialize_cell_snapshot(L0std_cell),
        'convention_cell': convention_cell.to_spglib(mag=True),
        'convention_cell_setting': convention_setting,
        'convention_cell_detail': convention_cell_snapshot,
        'wp_chain': wp_chain,
        'scif':scif,
    }
    symmetry = {'index':identify_info,
                'configuration':ssg_primitive.conf,
                'magnetic_phase':magnetic_phase,
                'magnetic_phase_base': magnetic_phase_base,
                'magnetic_phase_modifier': magnetic_phase_modifier,
                'magnetic_phase_spin_orbit_magnet': magnetic_phase_payload['spin_orbit_magnet_tag'],
                'magnetic_phase_details': magnetic_phase_details,
                'acc':ssg_primitive.acc,
                'msg_acc': msg_acc,
                'G0_symbol': ssg_primitive.G0_symbol,
                'G0_num': int(ssg_primitive.G0_num),
                'L0_symbol': ssg_primitive.L0_symbol,
                'L0_num': int(ssg_primitive.L0_num),
                'it': int(ssg_primitive.it),
                'ik': int(ssg_primitive.ik),
                'SSPG_symbol_hm': ssg_primitive.spin_part_point_group_symbol_hm,
                'SSPG_symbol_s': ssg_primitive.spin_part_point_group_symbol_s,
                'input_space_group_number': input_space_group_number,
                'input_space_group_symbol': input_space_group_symbol,
                'sg_has_real_space_inversion': space_group_has_real_space_inversion(input_space_group_number),
                'sg_is_polar': space_group_is_polar(input_space_group_number),
                'sg_is_chiral': space_group_is_chiral(input_space_group_number),
                'input_space_group_basis_or_setting': input_space_group_basis_or_setting,
                'source_structure_metadata': source_metadata,
                'source_parent_space_group': (
                    None if source_metadata is None else source_metadata.get('parent_space_group')
                ),
                'source_cell_parameter_strings': (
                    None if source_metadata is None else source_metadata.get('cell_parameter_strings')
                ),
                'KPOINTS':KPOINTS,
                'KPOINTS_setting': ACC_PRIMITIVE_SETTING,
                'KPOINTS_real_space_setting': ACC_PRIMITIVE_SETTING,
                'input_magnetic_primitive_ssg_ops': ssg_primitive.ops,
                'input_magnetic_primitive_ssg_setting': INPUT_MAGNETIC_PRIMITIVE_SETTING,
                'input_magnetic_primitive_ssg_seitz': ssg_primitive.seitz_symbols,
                'input_magnetic_primitive_ssg_seitz_latex': ssg_primitive.seitz_symbols_latex,
                'input_magnetic_primitive_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    ssg_primitive.seitz_descriptions
                ),
                'input_magnetic_primitive_ssg_international_linear': ssg_primitive.international_symbol_linear,
                'input_magnetic_primitive_ssg_international_latex': ssg_primitive.international_symbol_latex,
                'input_magnetic_primitive_ssg_symbol_calibration_tol': ssg_primitive.symbol_calibration_tol,
                'input_magnetic_primitive_ssg_type': ssg_primitive.international_symbol_type,
                'magnetic_primitive_ssg_ops': acc_magnetic_primitive_ssg.ops,
                'magnetic_primitive_ssg_setting': ACC_PRIMITIVE_SETTING,
                'magnetic_primitive_ssg_seitz': acc_magnetic_primitive_ssg.seitz_symbols,
                'magnetic_primitive_ssg_seitz_latex': acc_magnetic_primitive_ssg.seitz_symbols_latex,
                'magnetic_primitive_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_magnetic_primitive_ssg.seitz_descriptions
                ),
                'magnetic_primitive_ssg_international_linear': acc_magnetic_primitive_ssg.international_symbol_linear,
                'magnetic_primitive_ssg_international_latex': acc_magnetic_primitive_ssg.international_symbol_latex,
                'magnetic_primitive_ssg_symbol_calibration_tol': acc_magnetic_primitive_ssg.symbol_calibration_tol,
                'magnetic_primitive_ssg_type': acc_magnetic_primitive_ssg.international_symbol_type,
                'primitive_magnetic_cell_ssg_ops':acc_magnetic_primitive_ssg.ops,
                'primitive_magnetic_cell_ssg_setting': ACC_PRIMITIVE_SETTING,
                'primitive_magnetic_cell_ssg_seitz':acc_magnetic_primitive_ssg.seitz_symbols,
                'primitive_magnetic_cell_ssg_seitz_latex':acc_magnetic_primitive_ssg.seitz_symbols_latex,
                'primitive_magnetic_cell_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_magnetic_primitive_ssg.seitz_descriptions
                ),
                'primitive_magnetic_cell_ssg_international_linear':acc_magnetic_primitive_ssg.international_symbol_linear,
                'primitive_magnetic_cell_ssg_international_latex':acc_magnetic_primitive_ssg.international_symbol_latex,
                'primitive_magnetic_cell_ssg_symbol_calibration_tol': acc_magnetic_primitive_ssg.symbol_calibration_tol,
                'acc_primitive_ssg_ops': acc_primitive_output_ssg.ops,
                'acc_primitive_ssg_setting': ACC_PRIMITIVE_SETTING,
                'acc_primitive_ssg_seitz': acc_primitive_output_ssg.seitz_symbols,
                'acc_primitive_ssg_seitz_latex': acc_primitive_output_ssg.seitz_symbols_latex,
                'acc_primitive_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_primitive_output_ssg.seitz_descriptions
                ),
                'acc_primitive_ssg_international_linear': acc_primitive_output_ssg.international_symbol_linear,
                'acc_primitive_ssg_international_latex': acc_primitive_output_ssg.international_symbol_latex,
                'acc_primitive_ssg_symbol_calibration_tol': acc_primitive_output_ssg.symbol_calibration_tol,
                'symbol_calibration_tol': acc_magnetic_primitive_ssg.symbol_calibration_tol,
                'primitive_magnetic_cell_ssg_type':acc_magnetic_primitive_ssg.international_symbol_type,
                'full_spin_part_point_group':ssg_primitive.spin_part_point_group_symbol_hm,
                'identify_index_details':identify_index_details,
                'g0_standard_ssg_ops': G0std_ssg.ops,
                'g0_standard_ssg_seitz': G0std_ssg.seitz_symbols,
                'g0_standard_ssg_seitz_latex': G0std_ssg.seitz_symbols_latex,
                'g0_standard_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    G0std_ssg.seitz_descriptions
                ),
                'l0_standard_ssg_ops': L0std_ssg.ops,
                'l0_standard_ssg_seitz': L0std_ssg.seitz_symbols,
                'l0_standard_ssg_seitz_latex': L0std_ssg.seitz_symbols_latex,
                'l0_standard_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    L0std_ssg.seitz_descriptions
                ),
                'acc_conventional_ssg_ops': acc_conventional_ssg.ops,
                'acc_conventional_ssg_setting': ACC_CONVENTIONAL_SETTING,
                'acc_conventional_ssg_seitz': acc_conventional_ssg.seitz_symbols,
                'acc_conventional_ssg_seitz_latex': acc_conventional_ssg.seitz_symbols_latex,
                'acc_conventional_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    acc_conventional_ssg.seitz_descriptions
                ),
                'acc_conventional_ssg_international_linear': acc_conventional_ssg.international_symbol_linear,
                'acc_conventional_ssg_international_latex': acc_conventional_ssg.international_symbol_latex,
                'acc_conventional_ssg_symbol_calibration_tol': acc_conventional_ssg.symbol_calibration_tol,
                'convention_ssg_ops': public_ossg_ssg.ops,
                'convention_ssg_setting': convention_setting,
                'convention_ssg_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'ossg_space_group_number': ossg_space_group_number,
                'ossg_has_real_space_inversion': space_group_has_real_space_inversion(ossg_space_group_number),
                'ossg_is_polar': space_group_is_polar(ossg_space_group_number),
                'ossg_is_chiral': space_group_is_chiral(ossg_space_group_number),
                'convention_spin_only_direction': _format_spin_only_direction(public_ossg_ssg.sog_direction),
                'convention_ssg_seitz': public_ossg_ssg.seitz_symbols,
                'convention_ssg_seitz_latex': public_ossg_ssg.seitz_symbols_latex,
                'convention_ssg_seitz_descriptions': _serialize_seitz_descriptions(
                    public_ossg_ssg.seitz_descriptions
                ),
                'convention_nssg_ops': convention_nssg_ops,
                'convention_nssg_seitz': convention_nssg_seitz,
                'convention_nssg_seitz_latex': convention_nssg_seitz_latex,
                'convention_ssg_international_linear': public_ossg_ssg.international_symbol_linear_current_frame,
                'convention_ssg_international_latex': public_ossg_ssg.international_symbol_latex_current_frame,
                'convention_ssg_symbol_calibration_tol': public_ossg_ssg.symbol_calibration_tol,
                'primitive_msg_ops': [
                    [int(item[0]), np.asarray(item[1], dtype=float).tolist(), np.asarray(item[2], dtype=float).tolist()]
                    for item in primitive_msg_ops
                ],
                'primitive_msg_ops_setting': ACC_PRIMITIVE_SETTING,
                'primitive_msg_ops_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'magnetic_primitive_msg_ops': [
                    [int(item[0]), np.asarray(item[1], dtype=float).tolist(), np.asarray(item[2], dtype=float).tolist()]
                    for item in primitive_msg_ops
                ],
                'magnetic_primitive_msg_ops_setting': ACC_PRIMITIVE_SETTING,
                'magnetic_primitive_msg_ops_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'acc_primitive_msg_ops': [
                    [int(item[0]), np.asarray(item[1], dtype=float).tolist(), np.asarray(item[2], dtype=float).tolist()]
                    for item in primitive_msg_ops
                ],
                'acc_primitive_msg_ops_setting': ACC_PRIMITIVE_SETTING,
                'acc_primitive_msg_ops_spin_frame_setting': OSSG_ORIENTED_SPIN_FRAME_SETTING,
                'msg_little_group_symbols': msg_little_group_symbols,
                'msg_spin_polarizations': msg_spin_polarizations_poscar,
                'msg_spin_polarizations_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'msg_spin_polarizations_real_space_setting': ACC_PRIMITIVE_SETTING,
                'msg_spin_polarizations_spin_frame': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'msg_spin_polarizations_acc_cartesian': msg_spin_polarizations,
                'msg_spin_polarizations_acc_cartesian_setting': ACC_PRIMITIVE_CARTESIAN_SETTING,
                'msg_spin_polarizations_acc_poscar_spin_frame': msg_spin_polarizations_poscar,
                'msg_spin_polarizations_acc_poscar_spin_frame_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'T_input_to_G0std': (
                    np.asarray(transformation_input_to_G0std[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_G0std[1], dtype=float).tolist(),
                ),
                'raw_T_input_to_G0std': (
                    np.asarray(raw_transformation_input_to_G0std[0], dtype=float).tolist(),
                    np.asarray(raw_transformation_input_to_G0std[1], dtype=float).tolist(),
                ),
                'T_G0std_to_primitive': (
                    np.asarray(transformation_G0std_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_G0std_to_primitive[1], dtype=float).tolist(),
                ),
                'T_G0std_to_acc_primitive': (
                    np.asarray(transformation_G0std_to_acc_primitive_output[0], dtype=float).tolist(),
                    np.asarray(transformation_G0std_to_acc_primitive_output[1], dtype=float).tolist(),
                ),
                'T_input_to_L0std': (
                    np.asarray(transformation_input_to_L0std[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_L0std[1], dtype=float).tolist(),
                ),
                'raw_T_input_to_L0std': (
                    np.asarray(raw_transformation_input_to_L0std[0], dtype=float).tolist(),
                    np.asarray(raw_transformation_input_to_L0std[1], dtype=float).tolist(),
                ),
                'T_L0std_to_primitive': (
                    np.asarray(transformation_L0std_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_L0std_to_primitive[1], dtype=float).tolist(),
                ),
                'T_L0std_to_acc_primitive': (
                    np.asarray(transformation_L0std_to_acc_primitive_output[0], dtype=float).tolist(),
                    np.asarray(transformation_L0std_to_acc_primitive_output[1], dtype=float).tolist(),
                ),
                'T_input_to_convention': (
                    np.asarray(transformation_input_to_convention[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_convention[1], dtype=float).tolist(),
                ),
                'T_convention_to_acc_primitive': (
                    np.asarray(transformation_convention_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_convention_to_primitive[1], dtype=float).tolist(),
                ),
                'T_convention_to_acc_conventional': (
                    np.asarray(transformation_convention_to_acc_conventional[0], dtype=float).tolist(),
                    np.asarray(transformation_convention_to_acc_conventional[1], dtype=float).tolist(),
                ),
                'T_convention_to_acc_conventional_is_convention_self_automorphism': (
                    convention_to_acc_conventional_audit["real_ops_exact_same"]
                ),
                'T_convention_to_acc_conventional_label': convention_to_acc_conventional_label,
                'T_convention_to_acc_conventional_audit': {
                    'real_ops_exact_same': convention_to_acc_conventional_audit['real_ops_exact_same'],
                    'real_ops_same_mod_integer': convention_to_acc_conventional_audit['real_ops_same_mod_integer'],
                    'real_ops_same_mod_pure_translations': convention_to_acc_conventional_audit['real_ops_same_mod_pure_translations'],
                    'paired_spin_changed_count': convention_to_acc_conventional_audit['paired_spin_changed_count'],
                    'determinant': convention_to_acc_conventional_audit['determinant'],
                    'volume_preserving': convention_to_acc_conventional_audit['volume_preserving'],
                },
                'selected_standard_setting': selected_standard_setting,
                'T_selected_standard_to_acc_conventional': (
                    np.asarray(transformation_selected_standard_to_acc_conventional[0], dtype=float).tolist(),
                    np.asarray(transformation_selected_standard_to_acc_conventional[1], dtype=float).tolist(),
                ),
                'T_selected_standard_to_acc_conventional_is_self_automorphism': (
                    selected_standard_to_acc_conventional_audit['real_ops_exact_same']
                ),
                'T_selected_standard_to_acc_conventional_label': (
                    selected_standard_to_acc_conventional_label
                ),
                'T_selected_standard_to_acc_conventional_audit': {
                    'real_ops_exact_same': selected_standard_to_acc_conventional_audit['real_ops_exact_same'],
                    'real_ops_same_mod_integer': selected_standard_to_acc_conventional_audit['real_ops_same_mod_integer'],
                    'real_ops_same_mod_pure_translations': selected_standard_to_acc_conventional_audit['real_ops_same_mod_pure_translations'],
                    'paired_spin_changed_count': selected_standard_to_acc_conventional_audit['paired_spin_changed_count'],
                    'determinant': selected_standard_to_acc_conventional_audit['determinant'],
                    'volume_preserving': selected_standard_to_acc_conventional_audit['volume_preserving'],
                },
                'T_input_to_mag_primitive': (
                    np.asarray(transformation_input_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_primitive[1], dtype=float).tolist(),
                ),
                'T_input_to_input_magnetic_primitive': (
                    np.asarray(transformation_input_to_primitive[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_primitive[1], dtype=float).tolist(),
                ),
                'T_input_to_acc_primitive': (
                    np.asarray(transformation_input_to_acc_primitive_output[0], dtype=float).tolist(),
                    np.asarray(transformation_input_to_acc_primitive_output[1], dtype=float).tolist(),
                ),
                'msg_num': msg_num,
                'msg_type': msg_type,
                'msg_bns_number': msg_parent_info['bns_number'],
                'msg_og_number': msg_parent_info['og_number'],
                'msg_parent_space_group_number': msg_parent_info['bns_parent_space_group_number'],
                'msg_has_real_space_inversion': msg_parent_info['has_real_space_inversion'],
                'msg_is_polar': msg_parent_info['is_polar'],
                'msg_is_chiral': msg_parent_info['is_chiral'],
                'tolerances': {
                    'space_tol': float(tol_cfg.space),
                    'mtol': float(tol_cfg.moment),
                    'meigtol': float(tol_cfg.m_eig),
                    'matrix_tol': float(tol_cfg.m_matrix_tol),
                    'parser_atol': None if parser_atol is None else float(parser_atol),
                },
                'spin_polarizations':SS_poscar,
                'spin_polarizations_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'spin_polarizations_real_space_setting': ACC_PRIMITIVE_SETTING,
                'spin_polarizations_spin_frame': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'spin_polarizations_acc_cartesian': SS,
                'spin_polarizations_acc_cartesian_setting': ACC_PRIMITIVE_CARTESIAN_SETTING,
                'acc_primitive_real_cartesian_to_poscar_spin_frame': np.asarray(
                    acc_output_real_cartesian_to_poscar_spin_frame, dtype=float
                ).tolist(),
                'poscar_spin_frame_to_acc_primitive_real_cartesian': np.asarray(
                    poscar_spin_frame_to_acc_output_real_cartesian, dtype=float
                ).tolist(),
                'real_cartesian_to_spin_frame': np.asarray(
                    acc_output_real_cartesian_to_poscar_spin_frame, dtype=float
                ).tolist(),
                'spin_frame_to_real_cartesian': np.asarray(
                    poscar_spin_frame_to_acc_output_real_cartesian, dtype=float
                ).tolist(),
                'spin_polarizations_acc_poscar_spin_frame': SS_poscar,
                'spin_polarizations_acc_poscar_spin_frame_setting': ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
                'msg_symbol':msg_symbol,
                **gspg_payload}
    properties = {
        'ss_w_soc':ss_w_soc,
        'ss_wo_soc':ss_wo_soc,
        'ahc_w_soc':ahc_w_soc,
        'ahc_wo_soc':ahc_wo_soc,
        'is_alter':alter,
        'is_spin_orbit_magnet': magnetic_phase_payload['is_spin_orbit_magnet'],
        'magnetic_phase_base': magnetic_phase_base,
        'magnetic_phase_modifier': magnetic_phase_modifier,
        'tensor_outputs': tensor_outputs,
    }

    return MagSymmetryResult(cell,symmetry,properties)


def _find_spin_group_basic_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    input_spin_setting: str = "in_lattice",
) -> dict:
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    magnetic_primitive_cell, transformation_input_to_primitive = input_cell.get_primitive_structure(
        magnetic=True
    )
    identify_result = identify_spin_space_group_result(
        magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    ssg_primitive: SpinSpaceGroup = identify_result.ssg
    input_space_group = identify_result.input_space_group
    input_space_group_number = None if input_space_group is None else input_space_group.number
    input_space_group_symbol = None if input_space_group is None else input_space_group.symbol

    identify_index_details = None
    identify_info = None
    try:
        identify_index_details = _identify_ssg_index_details(
            source_name,
            ssg_primitive,
            tol=tol_cfg.m_matrix_tol,
        )
        identify_info = identify_index_details["index"]
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        warnings.warn(
            f"Identify-index output unavailable for {source_name}: {exc}. "
            "Continuing with identify-index-derived outputs set to None.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        msg_dataset_primitive = get_magnetic_symmetry_dataset(
            magnetic_primitive_cell.to_spglib(mag=True),
            symprec=tol_cfg.space,
            mag_symprec=tol_cfg.moment,
        )
        spglib_msg_num = msg_dataset_primitive.uni_number
        mpg_symbol = MSGMPG_DB.OG_NUM_TO_MPG[
            MSGMPG_DB.BNS_TO_OG_NUM[MSGMPG_DB.MSG_INT_TO_BNS[spglib_msg_num][0]]
        ]["pointgroup_no"]
    except Exception:
        mpg_symbol = None

    magnetic_phase_payload = classify_magnetic_phase(
        conf=ssg_primitive.conf,
        full_spin_part_point_group_hm=ssg_primitive.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=ssg_primitive.spin_part_point_group_symbol_s,
        net_moment=magnetic_primitive_cell.net_moment,
        mpg_identifier=mpg_symbol,
        is_ss_gp=ssg_primitive.is_spinsplitting[-1],
    )

    transformation_primitive_to_acc_primitive = (
        np.asarray(ssg_primitive.acc_primitive_trans, dtype=float),
        np.asarray(ssg_primitive.acc_primitive_origin_shift, dtype=float),
    )
    acc_magnetic_primitive_cell = magnetic_primitive_cell.transform(*transformation_primitive_to_acc_primitive)
    acc_magnetic_primitive_ssg = ssg_primitive.transform(*transformation_primitive_to_acc_primitive)
    acc_primitive_ossg = _ossg_oriented_spin_frame_ssg(
        acc_magnetic_primitive_ssg,
        acc_magnetic_primitive_cell,
    )
    internal_msg_info = acc_primitive_ossg.msg_info
    msg_num = None if internal_msg_info is None else internal_msg_info.get("msg_int_num")
    msg_symbol = None if internal_msg_info is None else internal_msg_info.get("msg_bns_symbol")
    msg_parent_info = msg_parent_space_group_info(msg_num)

    ssg_space_group_number = int(ssg_primitive.G0_num)

    return {
        "index": identify_info,
        "g0_symbol": ssg_primitive.G0_symbol,
        "g0_number": int(ssg_primitive.G0_num),
        "l0_symbol": ssg_primitive.L0_symbol,
        "l0_number": int(ssg_primitive.L0_num),
        "it": int(ssg_primitive.it),
        "ik": int(ssg_primitive.ik),
        "nsspg": ssg_primitive.n_spin_part_point_group_symbol_hm,
        "sspg": ssg_primitive.spin_part_point_group_symbol_hm,
        "acc_symbol": ssg_primitive.acc,
        "space_group_symbol": input_space_group_symbol,
        "space_group_number": input_space_group_number,
        "msg_symbol": msg_symbol,
        "msg_bns_number": msg_parent_info["bns_number"],
        "msg_og_number": msg_parent_info["og_number"],
        "empg": ssg_primitive.gspg.empg_symbol,
        "conf": ssg_primitive.conf,
        "magnetic_phase": magnetic_phase_payload["phase"],
        "is_alter": magnetic_phase_payload["is_alter"],
        "is_som": magnetic_phase_payload["is_spin_orbit_magnet"],
        "sg_is_polar": space_group_is_polar(input_space_group_number),
        "sg_is_chiral": space_group_is_chiral(input_space_group_number),
        "ssg_is_polar": space_group_is_polar(ssg_space_group_number),
        "ssg_is_chiral": space_group_is_chiral(ssg_space_group_number),
        "msg_is_polar": msg_parent_info["is_polar"],
        "msg_is_chiral": msg_parent_info["is_chiral"],
    }


def _find_spin_group_acc_primitive_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    input_spin_setting: str = "in_lattice",
) -> dict:
    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    magnetic_primitive_cell, transformation_input_to_primitive = input_cell.get_primitive_structure(
        magnetic=True
    )
    identify_result = identify_spin_space_group_result(
        magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    ssg_primitive: SpinSpaceGroup = identify_result.ssg

    identify_index_details = None
    identify_info = None
    try:
        identify_index_details = ssg_primitive.identify_index_details(
            source_name,
            tol=tol_cfg.m_matrix_tol,
        )
        identify_info = identify_index_details["index"]
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        warnings.warn(
            f"Identify-index output unavailable for {source_name}: {exc}. "
            "Continuing with identify-index-derived outputs set to None.",
            RuntimeWarning,
            stacklevel=2,
        )

    transformation_primitive_to_acc_primitive = (
        np.asarray(ssg_primitive.acc_primitive_trans, dtype=float),
        np.asarray(ssg_primitive.acc_primitive_origin_shift, dtype=float),
    )
    acc_primitive_cell = magnetic_primitive_cell.transform(*transformation_primitive_to_acc_primitive)
    acc_primitive_ssg = ssg_primitive.transform(*transformation_primitive_to_acc_primitive)
    transformation_input_to_acc_primitive = _chain_setting_transform(
        transformation_input_to_primitive[0],
        transformation_input_to_primitive[1],
        transformation_primitive_to_acc_primitive[0],
        transformation_primitive_to_acc_primitive[1],
    )
    acc_primitive_poscar = acc_primitive_cell.to_poscar(source_name)
    acc_real_cartesian_to_poscar_spin_frame = _poscar_spin_frame_rotation(acc_primitive_cell)
    poscar_spin_frame_to_acc_real_cartesian = np.linalg.inv(
        acc_real_cartesian_to_poscar_spin_frame
    )
    acc_primitive_ssg_in_poscar_spin_frame = acc_primitive_ssg.transform_spin(
        acc_real_cartesian_to_poscar_spin_frame
    )

    return {
        "index": identify_info,
        "acc_symbol": ssg_primitive.acc,
        "conf": ssg_primitive.conf,
        "acc_primitive_cell_setting": ACC_PRIMITIVE_SETTING,
        "acc_primitive_cell_detail": _serialize_cell_snapshot(acc_primitive_cell),
        "acc_primitive_poscar": acc_primitive_poscar,
        "acc_primitive_ssg_setting": ACC_PRIMITIVE_SETTING,
        "acc_primitive_ssg_international_linear": acc_primitive_ssg.international_symbol_linear,
        "acc_primitive_ssg_operation_matrices": _serialize_ssg_operation_matrices(
            list(acc_primitive_ssg.ops)
        ),
        "acc_primitive_poscar_spin_frame_setting": ACC_PRIMITIVE_POSCAR_SPIN_FRAME_SETTING,
        "acc_primitive_poscar_spin_frame_ssg_operation_matrices": _serialize_ssg_operation_matrices(
            list(acc_primitive_ssg_in_poscar_spin_frame.ops)
        ),
        "acc_primitive_real_cartesian_to_poscar_spin_frame": np.asarray(
            acc_real_cartesian_to_poscar_spin_frame, dtype=float
        ).tolist(),
        "poscar_spin_frame_to_acc_primitive_real_cartesian": np.asarray(
            poscar_spin_frame_to_acc_real_cartesian, dtype=float
        ).tolist(),
        "T_input_to_acc_primitive": (
            np.asarray(transformation_input_to_acc_primitive[0], dtype=float).tolist(),
            np.asarray(transformation_input_to_acc_primitive[1], dtype=float).tolist(),
        ),
    }


def find_spin_group_from_data(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    source_metadata: dict | None = None,
    input_spin_setting: str = "in_lattice",
    space_tol = 0.02,
    mtol = 0.02,
    meigtol = 0.00002,
    matrix_tol = 0.01,
) -> MagSymmetryResult:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    return _find_spin_group_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        source_metadata=source_metadata,
        parser_atol=None,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_basic_from_data(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    input_spin_setting="in_lattice",
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    return _find_spin_group_basic_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_acc_primitive_from_data(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    input_spin_setting="in_lattice",
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    return _find_spin_group_acc_primitive_from_parsed(
        source_name,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def _has_explicit_magnetic_moments(moments, *, tol: float = 1e-8) -> bool:
    if moments is None:
        return False
    array = np.asarray(moments, dtype=float)
    if array.size == 0:
        return False
    if array.ndim == 1:
        return bool(np.linalg.norm(array) > tol)
    return bool(np.any(np.linalg.norm(array, axis=1) > tol))


def _find_spin_group_input_ssg_from_parsed(
    source_name: str,
    lattice_factors,
    positions,
    elements,
    occupancies,
    moments,
    tol_cfg: Tolerances,
    *,
    input_spin_setting: str,
    source_format: str,
) -> dict:
    if not _has_explicit_magnetic_moments(moments):
        raise ValueError(
            f"Input magnetic-SSG route requires explicit magnetic moments; none were found in {source_name}."
        )

    input_cell = CrystalCell(
        lattice_factors,
        positions,
        occupancies,
        elements,
        moments,
        spin_setting=input_spin_setting,
        tol=tol_cfg,
    )
    identify_cell = (
        input_cell
        if input_cell.spin_setting == "cartesian"
        else _cartesianized_input_cell(input_cell)
    )
    input_magnetic_primitive_cell, transformation_input_to_input_magnetic_primitive = (
        identify_cell.get_primitive_structure(magnetic=True)
    )
    primitive_transform = np.asarray(
        transformation_input_to_input_magnetic_primitive,
        dtype=float,
    )
    primitive_det = float(np.linalg.det(primitive_transform))
    is_input_magnetic_primitive = bool(np.isclose(abs(primitive_det), 1.0, atol=1e-6))

    primitive_identify_result = identify_spin_space_group_result(
        input_magnetic_primitive_cell,
        find_primitive=False,
        tol=tol_cfg,
    )
    primitive_ssg: SpinSpaceGroup = primitive_identify_result.ssg

    primitive_identify_info = None
    try:
        primitive_identify_info = primitive_ssg.identify_index(
            source_name,
            tol=tol_cfg.m_matrix_tol,
        )
    except ValueError as exc:
        if not _should_degrade_identify_index_error(exc):
            raise
        warnings.warn(
            f"Primitive identify-index output unavailable for {source_name}: {exc}. "
            "Continuing with primitive identify-index-derived outputs set to None.",
            RuntimeWarning,
            stacklevel=2,
        )
        primitive_identify_info = primitive_ssg.index
    primitive_identify_info = primitive_identify_info or primitive_ssg.index

    primitive_ossg = _ossg_oriented_spin_frame_ssg(primitive_ssg, input_magnetic_primitive_cell)

    if is_input_magnetic_primitive:
        primitive_to_input = _invert_setting_transform(
            primitive_transform,
            np.zeros(3),
        )
        input_ssg = primitive_ssg.transform(*primitive_to_input)
        identify_info = primitive_identify_info
    else:
        input_identify_result = identify_spin_space_group_result(
            identify_cell,
            find_primitive=False,
            tol=tol_cfg,
        )
        input_ssg = input_identify_result.ssg
        identify_info = None
        try:
            identify_info = input_ssg.identify_index(
                source_name,
                tol=tol_cfg.m_matrix_tol,
            )
        except ValueError as exc:
            if not _should_degrade_identify_index_error(exc):
                raise
            warnings.warn(
                f"Identify-index output unavailable for {source_name}: {exc}. "
                "Continuing with identify-index-derived outputs set to None.",
                RuntimeWarning,
                stacklevel=2,
            )
        identify_info = identify_info or input_ssg.index

    input_ossg = _ossg_oriented_spin_frame_ssg(input_ssg, identify_cell)
    magnetic_phase_payload = classify_magnetic_phase(
        conf=input_ssg.conf,
        full_spin_part_point_group_hm=input_ssg.spin_part_point_group_symbol_hm,
        full_spin_part_point_group_s=input_ssg.spin_part_point_group_symbol_s,
        net_moment=identify_cell.net_moment,
        mpg_identifier=input_ossg.mpg_num,
        is_ss_gp=input_ssg.is_spinsplitting[-1],
    )
    warning = None
    if not is_input_magnetic_primitive:
        warning = (
            "Input cell is not a magnetic primitive cell; the input-cell SSG may be missing "
            "symmetry operations relative to the magnetic primitive setting."
        )

    input_poscar = None
    if source_format != "poscar":
        input_poscar = _cell_to_poscar_preserving_lattice(
            identify_cell,
            Path(source_name).name,
        )

    magnetic_primitive_poscar = None
    if not is_input_magnetic_primitive:
        magnetic_primitive_poscar = _cell_to_poscar_preserving_lattice(
            input_magnetic_primitive_cell,
            f"{Path(source_name).name}_magnetic_primitive"
        )

    return {
        "summary": {
            "input_ssg_index": identify_info or input_ssg.index,
            "primitive_ssg_index": primitive_identify_info,
            "input_conf": input_ssg.conf,
            "input_spin_only_direction": _format_spin_only_direction(input_ossg.sog_direction),
            "input_magnetic_phase": magnetic_phase_payload["phase"],
            "input_ssg_database_symbol": input_ssg.international_symbol_linear,
            "input_msg_num": input_ossg.msg_int_num,
            "primitive_msg_num": primitive_ossg.msg_int_num,
            "input_msg_bns_number": input_ossg.msg_bns_num,
            "primitive_msg_bns_number": primitive_ossg.msg_bns_num,
            "input_msg_symbol": input_ossg.msg_bns_symbol,
            "is_input_magnetic_primitive": is_input_magnetic_primitive,
            "input_ssg_may_be_incomplete": not is_input_magnetic_primitive,
            "warning": warning,
        },
        "ssg": {
            "setting": INPUT_POSCAR_SETTING,
            "spin_frame_setting": identify_cell.spin_setting,
            "ops": _serialize_ssg_operation_matrices(list(input_ssg.ops)),
        },
        "msg": {
            "setting": INPUT_POSCAR_SETTING,
            "spin_frame_setting": OSSG_ORIENTED_SPIN_FRAME_SETTING,
            "ops": _serialize_msg_operation_matrices(list(input_ossg.msg_ops), tol=input_ossg.tol),
        },
        "primitive_relation": {
            "T_input_to_input_magnetic_primitive": primitive_transform.tolist(),
            "determinant": primitive_det,
        },
        "input_poscar": input_poscar,
        "magnetic_primitive_poscar": magnetic_primitive_poscar,
    }


def find_spin_group(
    cif: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    parser_atol=0.02,
) -> MagSymmetryResult:
    """
    Find the spin space group of a crystal structure given in a CIF file.

    Parameters:
    cif (str): Path to the CIF file.
    space_tol (float): Tolerance for space group determination.
    mtol (float): Tolerance for magnetic moment determination.
    meigtol (float): Tolerance for eigenvalue determination.
    matrix_tol (float): Tolerance for point-group standardization matrices.
    parser_atol (float): Parsing tolerance for CIF / SCIF structure expansion.

    Returns:
    dict: A dictionary containing the spin space group information and related data.
    """

    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    parsed, source_metadata = parse_structure_file(cif, atol=parser_atol, return_metadata=True)
    lattice_factors,positions, elements, occupancies, labels, moments = parsed
    input_spin_setting = (
        "in_lattice" if source_metadata is None else source_metadata.get("spin_setting", "in_lattice")
    )
    return _find_spin_group_from_parsed(
        cif,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        source_metadata=source_metadata,
        parser_atol=parser_atol,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_basic(
    cif: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    parser_atol=0.02,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    parsed, _source_metadata = parse_structure_file(cif, atol=parser_atol, return_metadata=True)
    lattice_factors, positions, elements, occupancies, labels, moments = parsed
    input_spin_setting = (
        "in_lattice" if _source_metadata is None else _source_metadata.get("spin_setting", "in_lattice")
    )
    return _find_spin_group_basic_from_parsed(
        cif,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_acc_primitive(
    cif: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
    parser_atol=0.02,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    parsed, _source_metadata = parse_structure_file(cif, atol=parser_atol, return_metadata=True)
    lattice_factors, positions, elements, occupancies, labels, moments = parsed
    input_spin_setting = (
        "in_lattice" if _source_metadata is None else _source_metadata.get("spin_setting", "in_lattice")
    )
    return _find_spin_group_acc_primitive_from_parsed(
        cif,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
    )


def find_spin_group_input_ssg(
    structure_file: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
) -> dict:
    tol_cfg = Tolerances(space_tol, mtol, meigtol, m_matrix_tol=matrix_tol)
    path = Path(structure_file)
    suffix = path.suffix.lower()
    basename = path.name.lower()
    if suffix in {".vasp", ".poscar"} or basename in {"poscar", "contcar"}:
        lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(
            structure_file,
            allow_incar_magmom=False,
            require_embedded_magmom=True,
        )
        source_format = "poscar"
        input_spin_setting = "cartesian"
    else:
        parsed, source_metadata = parse_structure_file(
            structure_file,
            return_metadata=True,
        )
        lattice_factors, positions, elements, occupancies, labels, moments = parsed
        source_format = "unknown" if source_metadata is None else source_metadata.get("source_format", "unknown")
        input_spin_setting = (
            "in_lattice" if source_metadata is None else source_metadata.get("spin_setting", "in_lattice")
        )
    return _find_spin_group_input_ssg_from_parsed(
        structure_file,
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
        tol_cfg,
        input_spin_setting=input_spin_setting,
        source_format=source_format,
    )


def find_spin_group_poscar_ssg(
    poscar: str,
    space_tol=0.02,
    mtol=0.02,
    meigtol=0.00002,
    matrix_tol=0.01,
) -> dict:
    return find_spin_group_input_ssg(
        poscar,
        space_tol=space_tol,
        mtol=mtol,
        meigtol=meigtol,
        matrix_tol=matrix_tol,
    )


def write_ssg_operation_matrices(path: str | Path, operations: list[dict]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(operations, indent=2, ensure_ascii=False, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    return output_path


def _is_json_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_structured_json(value, indent: int = 0) -> str:
    current_indent = "  " * indent
    next_indent = "  " * (indent + 1)

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        items = list(value.items())
        for idx, (key, item) in enumerate(items):
            comma = "," if idx < len(items) - 1 else ""
            rendered = _format_structured_json(item, indent + 1)
            if "\n" not in rendered:
                lines.append(f"{next_indent}{json.dumps(key, ensure_ascii=False)}: {rendered}{comma}")
                continue
            rendered_lines = rendered.splitlines()
            lines.append(f"{next_indent}{json.dumps(key, ensure_ascii=False)}: {rendered_lines[0]}")
            lines.extend(rendered_lines[1:-1])
            lines.append(f"{rendered_lines[-1]}{comma}")
        lines.append(f"{current_indent}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(_is_json_scalar(item) for item in value):
            return json.dumps(value, ensure_ascii=False)
        if all(isinstance(item, list) and all(_is_json_scalar(entry) for entry in item) for item in value):
            lines = ["["]
            for idx, row in enumerate(value):
                comma = "," if idx < len(value) - 1 else ""
                lines.append(f"{next_indent}{json.dumps(row, ensure_ascii=False)}{comma}")
            lines.append(f"{current_indent}]")
            return "\n".join(lines)
        lines = ["["]
        for idx, item in enumerate(value):
            comma = "," if idx < len(value) - 1 else ""
            rendered = _format_structured_json(item, indent + 1)
            if "\n" not in rendered:
                lines.append(f"{next_indent}{rendered}{comma}")
                continue
            rendered_lines = rendered.splitlines()
            lines.append(f"{next_indent}{rendered_lines[0]}")
            lines.extend(rendered_lines[1:-1])
            lines.append(f"{rendered_lines[-1]}{comma}")
        lines.append(f"{current_indent}]")
        return "\n".join(lines)

    return json.dumps(value, ensure_ascii=False)


def write_poscar_ssg_symmetry_dat(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "summary": payload.get("summary") or {},
        "ssg": payload.get("ssg") or {},
        "msg": payload.get("msg") or {},
        "primitive_relation": payload.get("primitive_relation") or {},
        "input_poscar": payload.get("input_poscar"),
        "magnetic_primitive_poscar": payload.get("magnetic_primitive_poscar"),
        "format": "findspingroup.poscar_ssg.v1",
    }

    normalized_document = json.loads(json.dumps(document, ensure_ascii=False, cls=NumpyEncoder))

    output_path.write_text(
        _format_structured_json(normalized_document) + "\n",
        encoding="utf-8",
    )
    return output_path
