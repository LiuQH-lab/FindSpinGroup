import re
from copy import deepcopy
from findspingroup.version import __version__
import numpy as np

from functools import cached_property, lru_cache

from findspingroup.core.tolerances import DEFAULT_TOL, Tolerances
from findspingroup.structure.cell import AtomicSite
from findspingroup.core.identify_symmetry_from_ops import deduplicate_matrix_pairs, get_space_group_from_operations, \
    get_arithmetic_crystal_class_from_ops, identify_point_group, get_magnetic_space_group_from_operations
from findspingroup.utils.matrix_utils import getNormInf, integerize_matrix, rref_with_tolerance, in_space_group, \
    normalize_vector_to_zero
from findspingroup.utils.seitz_symbol import (
    _axis_parameter_subscript,
    calibrated_symbol_tol,
    canonicalize_group_seitz_descriptions,
    describe_point_operation,
    describe_spin_space_operation,
)
from findspingroup.utils.international_symbol import build_international_symbol
from findspingroup.utils.symbolic_format import format_symbolic_scalar


def parse_label_and_value(text):
    label, value = text.split(':', 1)
    return label, value


def combine_parametric_solutions(rref_matrix, tol=1e-3):

    A = np.array(rref_matrix, dtype=float)
    rows, cols = A.shape
    pivot_cols = []
    free_vars = []

    # Find pivot columns.
    for i in range(rows):
        for j in range(cols):
            if abs(A[i, j]) > tol:
                pivot_cols.append(j)
                break

    pivot_cols = set(pivot_cols)
    free_vars = [j for j in range(cols) if j not in pivot_cols]

    # Build the solution vector for each free variable.
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

        # Accumulate the symbolic vector expression.
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


def _to_latex_point_token(token: str) -> str:
    token = token.replace("alpha", r"\alpha")
    token = token.replace("beta", r"\beta")
    token = token.replace("gamma", r"\gamma")
    return re.sub(r"-(\d+)", r"\\bar{\1}", token)


def _axis_subscript_from_point_info(info: dict, *, latex: bool = False) -> str:
    axis_kind = info.get("axis_kind")
    axis_direction = info.get("axis_direction")
    if axis_kind == "direction" and axis_direction is not None:
        direction = tuple(int(v) for v in axis_direction)
        if any(abs(v) > 9 for v in direction):
            return f"{direction[0]},{direction[1]},{direction[2]}"
        return f"{direction[0]}{direction[1]}{direction[2]}"
    return _axis_parameter_subscript(info.get("axis_parameter_values"), latex=latex)


def _gspg_spin_only_symbol_from_rotations(rotations, conf: str, tol: float) -> dict[str, str]:
    symbol_tol = calibrated_symbol_tol(tol)
    unique_rotations = deduplicate_matrix_pairs([np.asarray(op, dtype=float) for op in rotations], tol=tol)
    non_identity = [op for op in unique_rotations if not np.allclose(op, np.eye(3), atol=tol)]

    if not non_identity:
        return {
            "hm": "1",
            "s": "C1",
            "linear": "",
            "latex": "",
        }

    if conf == "Collinear":
        candidate = next(
            (op for op in non_identity if np.linalg.det(op) > 0),
            non_identity[0],
        )
        info = describe_point_operation(candidate, tol=symbol_tol, max_order=120, max_axis_denom=12)
        sub_linear = _axis_subscript_from_point_info(info, latex=False)
        sub_latex = _axis_subscript_from_point_info(info, latex=True)

        if len(unique_rotations) == 8:
            hm_symbol = "∞/mm"
            s_symbol = "D∞h"
            linear = f"∞_{{{sub_linear}}}/mm|1"
            latex = rf"^{{\infty_{{{sub_latex}}}/mm}}1"
        else:
            hm_symbol = "∞m"
            s_symbol = "C∞v"
            linear = f"∞_{{{sub_linear}}}m|1"
            latex = rf"^{{\infty_{{{sub_latex}}}m}}1"

        return {
            "hm": hm_symbol,
            "s": s_symbol,
            "linear": linear,
            "latex": latex,
        }

    info = identify_point_group(unique_rotations, _id=True)
    hm_symbol = info[0]
    s_symbol = info[4]

    if hm_symbol == "1":
        return {
            "hm": "1",
            "s": "C1",
            "linear": "",
            "latex": "",
        }

    if len(info[3]) == 1:
        token = info[1][info[3][0]][2]
        linear = f"{token}|1"
        latex = rf"^{{{_to_latex_point_token(token)}}}1"
    else:
        linear = f"{hm_symbol}|1"
        latex = rf"^{{{_to_latex_point_token(hm_symbol)}}}1"

    return {
        "hm": hm_symbol,
        "s": s_symbol,
        "linear": linear,
        "latex": latex,
    }




def _normalize_group_tol(tol: float | Tolerances) -> float:
    if isinstance(tol, Tolerances):
        return tol.m_matrix_tol
    return float(tol)


def _matrix_bytes_key(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype=np.float64).reshape(3, 3).tobytes()


def _vector_bytes_key(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float64).reshape(3).tobytes()


def _restore_matrix_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float64).reshape(3, 3).copy()


def _restore_vector_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float64).reshape(3).copy()


def _space_group_ops_cache_key(ops) -> tuple:
    return tuple(
        (_matrix_bytes_key(rotation), _vector_bytes_key(translation))
        for rotation, translation in ops
    )


def _magnetic_space_group_ops_cache_key(ops) -> tuple:
    return tuple(
        (int(time_reversal), _matrix_bytes_key(rotation), _vector_bytes_key(translation))
        for time_reversal, rotation, translation in ops
    )


def _pair_matrix_ops_cache_key(ops) -> tuple:
    return tuple(
        (_matrix_bytes_key(left), _matrix_bytes_key(right))
        for left, right in ops
    )


def _matrix_vector_ops_cache_key(ops) -> tuple:
    return tuple(
        (_matrix_bytes_key(rotation), _vector_bytes_key(translation))
        for rotation, translation in ops
    )


@lru_cache(maxsize=512)
def _cached_space_group_dataset_by_key(ops_key: tuple):
    ops = [
        [_restore_matrix_from_bytes(rotation_bytes), _restore_vector_from_bytes(translation_bytes)]
        for rotation_bytes, translation_bytes in ops_key
    ]
    dataset = get_space_group_from_operations(ops)
    return (
        dataset.international,
        int(dataset.number),
        _matrix_bytes_key(dataset.transformation_matrix),
        _vector_bytes_key(dataset.origin_shift),
    )


@lru_cache(maxsize=512)
def _cached_acc_symbol_info_by_key(ops_key: tuple):
    ops = [
        [_restore_matrix_from_bytes(rotation_bytes), _restore_vector_from_bytes(translation_bytes)]
        for rotation_bytes, translation_bytes in ops_key
    ]
    acc, primitive_trans, primitive_origin_shift, _ = get_arithmetic_crystal_class_from_ops(
        ops,
        include_kpath=False,
    )
    return (
        acc,
        _matrix_bytes_key(primitive_trans),
        _vector_bytes_key(primitive_origin_shift),
    )


@lru_cache(maxsize=256)
def _cached_acc_kpath_info_by_key(ops_key: tuple):
    ops = [
        [_restore_matrix_from_bytes(rotation_bytes), _restore_vector_from_bytes(translation_bytes)]
        for rotation_bytes, translation_bytes in ops_key
    ]
    _, _, _, kpath_info = get_arithmetic_crystal_class_from_ops(
        ops,
        include_kpath=True,
    )
    return deepcopy(kpath_info)


@lru_cache(maxsize=512)
def _cached_magnetic_space_group_info_by_key(ops_key: tuple):
    ops = [
        [time_reversal, _restore_matrix_from_bytes(rotation_bytes), _restore_vector_from_bytes(translation_bytes)]
        for time_reversal, rotation_bytes, translation_bytes in ops_key
    ]
    info = get_magnetic_space_group_from_operations(ops)
    return None if info is None else deepcopy(info)


def _matrix_group_closure(generators, *, tol: float, limit: int = 256):
    ops = deduplicate_matrix_pairs([np.asarray(op, dtype=float) for op in generators], tol=tol)
    changed = True
    while changed:
        changed = False
        current = list(ops)
        for left in current:
            for right in current:
                product = np.asarray(left, dtype=float) @ np.asarray(right, dtype=float)
                if any(np.allclose(product, existing, atol=tol) for existing in ops):
                    continue
                ops.append(product)
                changed = True
                if len(ops) > limit:
                    raise RuntimeError("Matrix-group closure exceeded the configured limit.")
        ops = deduplicate_matrix_pairs(ops, tol=tol)
    return ops


def _matrix_sets_match(left, right, *, tol: float) -> bool:
    left = deduplicate_matrix_pairs([np.asarray(op, dtype=float) for op in left], tol=tol)
    right = deduplicate_matrix_pairs([np.asarray(op, dtype=float) for op in right], tol=tol)
    if len(left) != len(right):
        return False
    return all(any(np.allclose(a, b, atol=tol) for b in right) for a in left)


def _effective_proper_rotation(rotation, *, tol: float) -> np.ndarray | None:
    rotation = np.asarray(rotation, dtype=float)
    det_rotation = float(np.linalg.det(rotation))
    if abs(det_rotation - 1.0) < tol:
        return rotation
    if abs(det_rotation + 1.0) < tol:
        return -rotation
    return None


def _canonicalize_axis_direction(axis, *, tol: float) -> np.ndarray | None:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < tol:
        return None
    axis = axis / norm
    for value in axis:
        if abs(value) > tol:
            if value < 0:
                axis = -axis
            break
    return axis


def _normalize_metric(metric) -> np.ndarray | None:
    if metric is None:
        return None
    metric = np.asarray(metric, dtype=float)
    if metric.shape != (3, 3):
        raise ValueError("Real-space metric must be a 3x3 matrix.")
    return metric


def _metric_cosine(left, right, *, metric=None, tol: float) -> float | None:
    left_axis = _canonicalize_axis_direction(left, tol=tol)
    right_axis = _canonicalize_axis_direction(right, tol=tol)
    if left_axis is None or right_axis is None:
        return None

    metric = _normalize_metric(metric)
    if metric is None:
        return float(np.dot(left_axis, right_axis))

    left_norm_sq = float(left_axis @ metric @ left_axis)
    right_norm_sq = float(right_axis @ metric @ right_axis)
    if left_norm_sq < tol or right_norm_sq < tol:
        return None
    return float((left_axis @ metric @ right_axis) / np.sqrt(left_norm_sq * right_norm_sq))


def _effective_rotation_axis(rotation, *, tol: float) -> np.ndarray | None:
    effective_rotation = _effective_proper_rotation(rotation, tol=tol)
    if effective_rotation is None:
        return None
    eigenvalues, eigenvectors = np.linalg.eig(effective_rotation)
    matches = np.isclose(eigenvalues, 1.0, atol=tol)
    if not np.any(matches):
        return None
    axis = eigenvectors[:, matches][:, 0].real
    return _canonicalize_axis_direction(axis, tol=tol)


def _axes_parallel(left, right, *, metric=None, tol: float) -> bool:
    cosine = _metric_cosine(left, right, metric=metric, tol=tol)
    if cosine is None:
        return False
    return abs(abs(cosine) - 1.0) < tol


def _axes_perpendicular(left, right, *, metric=None, tol: float) -> bool:
    cosine = _metric_cosine(left, right, metric=metric, tol=tol)
    if cosine is None:
        return False
    return abs(cosine) < tol

class BrillouinZoneMatcher:
    def __init__(self, rules):
        self.parsed_rules = []
        for label, pattern, splitting in rules:
            parsed = self._parse_pattern(pattern)
            score = self._calculate_specificity_score(parsed)
            self.parsed_rules.append({
                'label': label,
                'pattern': parsed,
                'splitting': splitting,
                'score': score
            })

        self.parsed_rules.sort(key=lambda x: x['score'], reverse=True)

    def _parse_pattern(self, pattern_str):
        content = pattern_str.strip("()").split(",")
        parsed = []
        for item in content:
            item = item.strip()

            try:
                val = float(eval(item, {"__builtins__": None}, {}))
                parsed.append({'type': 'fixed', 'val': val})
            except:
                parsed.append({'type': 'var', 'name': item})
        return parsed

    def _calculate_specificity_score(self, parsed_pattern):
        score = 0
        vars_seen = set()

        for p in parsed_pattern:
            if p['type'] == 'fixed':
                score += 10
            elif p['type'] == 'var':
                if p['name'] in vars_seen:
                    score += 5
                vars_seen.add(p['name'])
        return score

    def check(self, u, v, w, tol=1e-5):
        input_k = np.mod([u, v, w],1)

        for rule in self.parsed_rules:
            match = True
            var_map = {}

            for i in range(3):
                rule_comp = rule['pattern'][i]
                input_val = input_k[i]

                if rule_comp['type'] == 'fixed':
                    if abs(input_val - rule_comp['val']) > tol:
                        match = False
                        break

                elif rule_comp['type'] == 'var':
                    var_name = rule_comp['name']
                    if var_name in var_map:
                        if abs(input_val - var_map[var_name]) > tol:
                            match = False
                            break
                    else:
                        var_map[var_name] = input_val

            if match:
                return {
                    "matched_label": rule['label'],
                    "has_splitting": rule['splitting'],
                    "k_point": (u, v, w)
                }

        raise ValueError(f"No matching rule found for k-point ({u}, {v}, {w})")


def write_kpoints(seekpath_out, matcher: BrillouinZoneMatcher, num_points=40, extra_kpoints=None):
    """
    Write k-point path string with SOC splitting info for Endpoints AND Path.
    """
    kpts = seekpath_out['point_coords']
    path = seekpath_out['path']

    def append_low_sym_points_simple_chain(extra_points):
        """
        Connect all extra points into a simple chain in input order.
        Path logic: GAMMA -> P1 -> P2 -> ... -> Pn -> GAMMA.

        Parameters:
        extra_points: list of tuples, e.g. [([0.1, 0, 0], "MyP1"), ([0.2, 0, 0], "MyP2")]
        seekpath_output: dict, output from `seekpath.get_path()`
        """

        # Copy the base data.
        point_coords = {}
        path_list = []

        # Return immediately if no extra points are provided.
        if not extra_points:
            return {'point_coords': point_coords, 'path': path_list}

        # Register all new coordinates and preserve their input order.
        new_labels_ordered = []
        for coords, label in extra_points:
            point_coords[label] = np.array(coords)
            new_labels_ordered.append(label)

        # Build the chain path. Seekpath usually labels Gamma as `GAMMA`.
        gamma_label = 'GAMMA'

        # A. Start segment: GAMMA -> first extra point.
        first_point = new_labels_ordered[0]
        path_list.append((gamma_label, first_point))

        # B. Middle segments in the provided order.
        for i in range(len(new_labels_ordered) - 1):
            current_p = new_labels_ordered[i]
            next_p = new_labels_ordered[i + 1]
            path_list.append((current_p, next_p))

        # C. Final segment: last extra point -> GAMMA.
        last_point = new_labels_ordered[-1]
        path_list.append((last_point, gamma_label))

        return {'point_coords': point_coords, 'path': path_list}

    def fmt(label):
        # Render a cleaner display label.
        return 'Γ' if label == 'GAMMA' else label.replace('_', '')

    # Helper: determine the splitting state for one k-point.
    def get_split_status(u, v, w):
        """
        Return `(matched_label, is_splitting)` for a k-point coordinate.
        """
        result = matcher.check(u, v, w)
        if result:
            return result['matched_label'], result['has_splitting']
        else:
            # Use a conservative fallback if no match is found.
            return "Unknown", True

    # Helper: format the display tag.
    def make_tag(label, is_splitting, is_path=False):
        """
        Format the display string.
        `is_path=True` is used for path information and `False` for endpoints.
        """

        highlight = "***" if is_splitting else ""

        if is_path:
            return f"| {label} {highlight}"
        else:
            return f"{highlight}"

    def _write_kpoints(s_head,path_list,kpts_list):
        path_label = []
        for start_label, end_label in path_list:
            k1 = np.array(kpts_list[start_label])
            k2 = np.array(kpts_list[end_label])

            # Use the midpoint to represent the path segment.
            mid_k = (k1 + k2) / 2.0

            lbl_start, split_start = get_split_status(k1[0], k1[1], k1[2])
            lbl_end, split_end = get_split_status(k2[0], k2[1], k2[2])
            lbl_mid, split_mid = get_split_status(mid_k[0], mid_k[1], mid_k[2])

            path_label.append(lbl_mid)

            tag_start_pt = make_tag(lbl_start, split_start, is_path=False)
            tag_end_pt = make_tag(lbl_end, split_end, is_path=False)
            tag_path = make_tag(lbl_mid, split_mid, is_path=True)

            s_head.append(
                f"{k1[0]:10.6f} {k1[1]:10.6f} {k1[2]:10.6f} ! "
                f"{fmt(start_label) + ' ' + tag_start_pt:<9}\n"
            )

            s_head.append(
                f"{k2[0]:10.6f} {k2[1]:10.6f} {k2[2]:10.6f} ! "
                f"{fmt(end_label) + ' ' + tag_end_pt:<9} {tag_path}\n"
            )

            s_head.append("\n")
        return path_label,s_head

    s = []
    # Write the file header.
    s.append(f"Generated by seekpath and findspingroup v{__version__} (*** for spin splitting)\n ")
    s.append(f"{num_points}\nLine-mode\nReciprocal\n")
    path_label,s = _write_kpoints(s,path,kpts)

    if extra_kpoints:
        add_kpoints = []
        for i in extra_kpoints:
            if matcher.check(*i[0])['matched_label'] in path_label:
                continue
            else:
                add_kpoints.append(i)
        extra_seekpath = append_low_sym_points_simple_chain(add_kpoints)
        s = _write_kpoints(s,extra_seekpath['path'],extra_seekpath['point_coords']|kpts)[1]
    return ''.join(s)

def find_uvw_whole_string(data_list):
    """
    Return the indices of strings containing at least two of `u`, `v`, `w`.
    """
    target_chars = {'u', 'v', 'w'}
    indices = []
    for index, text in enumerate(data_list):
        common_chars = set(text) & target_chars

        if len(common_chars) >= 2:
            indices.append(index)


    return indices

def op_key(op):
    rot1, rot2, t = op
    rot1 = np.asarray(rot1)
    rot2 = np.asarray(rot2)
    t    = np.asarray(t)

    # Rank operations by proximity to the identity.
    d_rot2 = np.linalg.norm(rot2 - np.identity(3), ord='fro')
    d_t    = np.linalg.norm(t)
    d_rot1 = np.linalg.norm(rot1 - np.identity(3), ord='fro')

    return (d_rot2, d_t, d_rot1)


def _dedup_bucket_decimals(atol: float) -> int:
    """
    Return a coarse rounding precision for dedup bucketing.

    The bucket key is intentionally coarser than the final equality tolerance so
    that operations equal within ``atol`` always fall into the same bucket, while
    still separating obviously different operations. Bucket membership never
    decides equality on its own; it only reduces the candidate set for the exact
    tolerant comparison.
    """
    atol = float(max(atol, 1e-12))
    return max(0, int(np.ceil(-np.log10(atol))) - 1)


def _op_bucket_key(op, atol: float):
    decimals = _dedup_bucket_decimals(atol)
    spin_rotation = tuple(np.round(np.asarray(op[0], dtype=float).reshape(-1), decimals))
    real_rotation = tuple(np.round(np.asarray(op[1], dtype=float).reshape(-1), decimals))
    translation = tuple(np.round(np.asarray(op[2], dtype=float).reshape(-1), decimals))
    return spin_rotation, real_rotation, translation

def check_divisible(a, b):
    if b == 0:
        raise ValueError("cannot divide by zero")
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")
    return a // b


def _validate_array_format(array, expected_shape):
    """Unified validation function for arrays with expected shapes."""
    array = np.array(array, dtype=np.float64)

    # Handle translation vector special case (accept both (3,) and (3,1))
    if array.shape == (3,1):
        array = array.reshape(3, )

    if array.shape != expected_shape:
        raise ValueError(f"must have shape {expected_shape}, got shape {array.shape}")

    return array

def find_group_generators(ops: list) -> list:
    """Find a minimal set of generators for the group defined by ops."""
    generators = []
    current_group = set()
    # todo: this is not finished yet.
    for op in ops:
        if op not in current_group:
            generators.append(op)
            # Generate new elements by combining with existing group elements
            new_elements = set()
            for g in current_group:
                new_elements.add(g @ op)
                new_elements.add(op @ g)
            new_elements.add(op)
            current_group.update(new_elements)

    return generators

def integer_points_in_new_cell(T, tol=1e-5):
    """
    Return all integer points inside the unit cell spanned by the new basis.

    `T` is a 3x3 matrix whose row vectors express the new basis in the old basis.
    """
    T = np.asarray(T, dtype=float)

    # Compute the 8 unit-cell vertices for u in {0,1}^3.
    corners = np.array([[i, j, k]
                        for i in [0, 1]
                        for j in [0, 1]
                        for k in [0, 1]], dtype=float)
    vertices = corners @ T              # shape: (8, 3)
    # print(vertices)
    # Axis-aligned bounding box.
    mins = np.floor(vertices.min(axis=0)).astype(int)
    maxs = np.ceil(vertices.max(axis=0)).astype(int)

    # Precompute the inverse to map integer points back to fractional u.
    invT = np.linalg.inv(T)

    points = []
    for i in range(mins[0], maxs[0] + 1):
        for j in range(mins[1], maxs[1] + 1):
            for k in range(mins[2], maxs[2] + 1):
                n = np.array([i, j, k], dtype=float)
                u = n @ invT
                # print('n',n,'u',u)
                # Keep only points whose fractional coordinates lie in [0,1).
                if np.all(u >= -tol) and np.all(u < 1 - tol):
                    points.append((i, j, k))

    return points


class SpinSpaceGroupOperation:
    """
    Represents a spin space group operation consisting of a rotation, translation (mod 1), and spin rotation.

    Attributes:
        rotation (np.ndarray): A 3x3 rotation matrix.
        translation (np.ndarray): A 3x1 translation vector.
        spin_rotation (np.ndarray): A 3x3 spin rotation matrix.

    Methods:
        __matmul__(other): Composes two symmetry operations or acts on an atomic site or vector.
        inv(): Returns the inverse of the spin space group operation.
        to_spg_op(): Converts to SpinPointGroupOperation by dropping the translation part.

        tolist(): Returns the operation as a list [spin_rotation, rotation, translation].


    """
    def __init__(self, spin_rotation, rotation, translation ):
        self.rotation = _validate_array_format(rotation,(3,3))  # 3x3 matrix
        self.translation = _validate_array_format(translation,(3,))  # 3x1 vector
        self.spin_rotation = _validate_array_format(spin_rotation,(3,3))  # 3x3 matrix
        self._data = [self.spin_rotation, self.rotation, self.translation ]


    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"SpinOperation(spin_rotation={self.spin_rotation}|| rotation={self.rotation}| translation={self.translation})"

    def __matmul__(self, other):
        if isinstance(other, SpinSpaceGroupOperation):
            # compose two symmetry operations
            new_rotation = self.rotation @ other.rotation
            new_translation = normalize_vector_to_zero( self.rotation @ other.translation + self.translation,atol=1e-4)
            new_spin_rotation = self.spin_rotation @ other.spin_rotation
            # constructor expects (spin_rotation, rotation, translation)
            return SpinSpaceGroupOperation(new_spin_rotation, new_rotation, new_translation)

        elif isinstance(other, AtomicSite):
            # act on an atomic site
            new_position = normalize_vector_to_zero(self.rotation @ other.position + self.translation,atol=1e-9)
            new_magnetic_moment = self.spin_rotation @ other.magnetic_moment
            return AtomicSite(new_position, new_magnetic_moment,other.occupancy, other.element_symbol)

        elif isinstance(other, np.ndarray):
            # act on normal vector or [spin, position] vector
            if other.shape == (3,):
                new_vector = self.rotation @ other  # only rotation
                return new_vector
            elif other.shape == (6,):
                position = self.rotation @ other [3:6] + self.translation.flatten()
                magnetic_moment = self.spin_rotation @ other[0:3]
                return position + magnetic_moment
            else:
                raise ValueError("Unsupported ndarray shape for SpinOperation @ vector")

        else:
            raise TypeError("SpinOperation @ unsupported type")

    def inv(self) -> 'SpinSpaceGroupOperation':
        """Inverse of the spin space group operation. {Rs||R|t} -> {Rs^{-1}||R^{-1}|-R{-1}*t}"""
        inv_rotation = np.linalg.inv(self.rotation)
        inv_translation = normalize_vector_to_zero(-inv_rotation @ self.translation ,atol=1e-4)
        inv_spin_rotation = np.linalg.inv(self.spin_rotation)
        return SpinSpaceGroupOperation(inv_spin_rotation, inv_rotation, inv_translation)

    def to_spg_op(self) -> 'SpinPointGroupOperation':
        """Convert to SpinPointGroupOperation by dropping the translation part."""
        return SpinPointGroupOperation(deepcopy(self.spin_rotation), deepcopy(self.rotation))

    def tolist(self):
        return [self.spin_rotation.round(6).tolist(), self.rotation.round(6).tolist(), self.translation.round(6).tolist()]

    def seitz_description(self, tol=1e-6, max_order=120, max_axis_denom=12):
        """
        Return a structured Seitz description for this operation.

        Output keys include:
            spin: point-operation info for Rs
            real: point-operation info for Rr
            translation_symbol: formatted tau_{...}
            symbol: combined label like { A || B | tau_{...} }
        """
        return describe_spin_space_operation(
            self.spin_rotation,
            self.rotation,
            self.translation,
            tol=tol,
            max_order=max_order,
            max_axis_denom=max_axis_denom,
        )

    def to_seitz_symbol(self, tol=1e-6, max_order=120, max_axis_denom=12):
        return self.seitz_description(
            tol=tol,
            max_order=max_order,
            max_axis_denom=max_axis_denom,
        )["symbol"]

    def to_seitz_symbol_latex(self, tol=1e-6, max_order=120, max_axis_denom=12):
        return self.seitz_description(
            tol=tol,
            max_order=max_order,
            max_axis_denom=max_axis_denom,
        )["symbol_latex"]

    @classmethod
    def identity(cls) -> 'SpinSpaceGroupOperation':
        """Returns the identity operation."""
        return cls(np.eye(3), np.eye(3), np.zeros(3))

    def is_same_with(self,other, atol=1e-3):
        A1, B1, C1 = self._data
        A2, B2, C2 = other._data
        if np.allclose(A1, A2, atol=atol) and np.allclose(B1, B2, atol=atol) and getNormInf(C1, C2) < atol:
            return True
        else:
            return False

    def magnetic_time_reversal(self, atol=1e-3):
        det_rotation = float(np.linalg.det(self.rotation))
        if abs(det_rotation - 1.0) < atol:
            effective_rotation = self.rotation
        elif abs(det_rotation + 1.0) < atol:
            # Magnetic moments are axial vectors, so improper real-space
            # operations contribute an extra det(Rr) factor in the spin action.
            effective_rotation = -self.rotation
        else:
            return None

        if np.allclose(self.spin_rotation, effective_rotation, atol=atol):
            return 1
        if np.allclose(self.spin_rotation, -effective_rotation, atol=atol):
            return -1
        return None

    def is_magnetic_space_group_operation(self, atol=1e-3):
        return self.magnetic_time_reversal(atol=atol) is not None

class SpinPointGroupOperation:
    """
    Represents a spin point group operation consisting of a rotation and spin rotation.

    Attributes:
        rotation (np.ndarray): A 3x3 rotation matrix.
        spin_rotation (np.ndarray): A 3x3 spin rotation matrix.

    Methods:
        __matmul__(other): Composes two symmetry operations or acts on an atomic site or vector.
        inv(): Returns the inverse of the spin point group operation.
        act_on_kpoint(k_point): Acts on a k-point in reciprocal space.
        effective_k_operation(): Returns the effective k-point operation.

        tolist(): Returns the operation as a list [spin_rotation, rotation].
    """
    def __init__(self,spin_rotation, rotation):
        self.rotation = _validate_array_format(rotation,(3,3))  # 3x3 matrix
        self.spin_rotation = _validate_array_format(spin_rotation,(3,3))  # 3x3 matrix


        self._data = [ self.spin_rotation, self.rotation]
        self._spin_det_sign = np.sign(np.linalg.det(self.spin_rotation))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"SpinPointGroupOperation(spin_rotation={self.spin_rotation}|| rotation={self.rotation})"


    def __matmul__(self, other):
        if isinstance(other, SpinPointGroupOperation):
            # compose two symmetry operations
            new_rotation = self.rotation @ other.rotation
            new_spin_rotation = self.spin_rotation @ other.spin_rotation
            # constructor expects (spin_rotation, rotation)
            return SpinPointGroupOperation(new_spin_rotation, new_rotation)

        elif isinstance(other, AtomicSite):
            # act on an atomic site
            new_position = normalize_vector_to_zero(self.rotation @ other.position )
            new_magnetic_moment = self.spin_rotation @ other.magnetic_moment
            return AtomicSite(new_position, new_magnetic_moment,other.occupancy, other.element_symbol)

        elif isinstance(other, np.ndarray):
            # act on normal vector or [spin, position] vector
            if other.shape == (3,):
                new_vector = self.rotation @ other  # only rotation
                return new_vector
            elif other.shape == (6,):
                position = self.rotation @ other [3:6]
                magnetic_moment = self.spin_rotation @ other[0:3]
                return position + magnetic_moment
            else:
                raise ValueError("Unsupported ndarray shape for SpinPointGroupOperation @ vector")

        else:
            raise TypeError("SpinPointGroupOperation @ unsupported type")

    def inv(self):
        """Inverse of the spin point group operation. {Rs||R} -> {Rs^{-1}||R^{-1}}"""
        inv_rotation = np.linalg.inv(self.rotation)
        inv_spin_rotation = np.linalg.inv(self.spin_rotation)
        return SpinPointGroupOperation(inv_spin_rotation, inv_rotation)

    def act_on_kpoint(self, k_point):
        """Acts on a k-point in reciprocal space."""
        k_point = _validate_array_format(k_point, (3,))
        new_k_point = self.rotation @ k_point * self._spin_det_sign % 1
        return new_k_point

    def effective_k_operation(self):
        """Returns the effective k-point operation."""
        return self.rotation * self._spin_det_sign

    def tolist(self):
        return [self.spin_rotation, self.rotation]


def fetch_ssg_by_index(index:str):
    """Fetch the spin space group operations from database by its index.
    This is a placeholder function. In a real implementation, this would query a database or a predefined dictionary.
    """



    # Example placeholder data
    SSG_DATA = {}
    example_generators = SSG_DATA.get(index, False)

    if not example_generators:
        raise ValueError(f"Spin space group with index {index} not found in the database.")
    else:
        return example_generators





class SpinSpaceGroup:
    """
    Represents a spin space group defined by its symmetry operations.
    Refactored for lazy evaluation using cached_property.
    """

    def __init__(
        self,
        input_data: str | list,
        tol: float | Tolerances = DEFAULT_TOL,
        *,
        real_space_metric=None,
    ):
        """
        Initializes a SpinSpaceGroup instance.
        """
        self.tol = _normalize_group_tol(tol)
        self.real_space_metric = _normalize_metric(real_space_metric)
        self._input_index = None
        self._input_ops = []

        if isinstance(input_data, str):
            try:
                self._input_ops = fetch_ssg_by_index(input_data)
            except ValueError:
                raise ValueError(f"Spin space group with index {input_data} not found in the database.")

            self._input_index = input_data
            self.lattice_settings = 'G0_standard'
            self.spin_settings = 'cartesian'
            self.relative_settings = 'Arbitrary'

        elif isinstance(input_data, list):
            if all(isinstance(op, SpinSpaceGroupOperation) for op in input_data):
                self._input_ops = input_data
            elif all(isinstance(op, list) and len(op) == 3 for op in input_data):
                self._input_ops = [SpinSpaceGroupOperation(op[0], op[1], op[2]) for op in input_data]
            else:
                raise ValueError(
                    "List must contain either SpinOperation instances or lists of [rotation, translation, spin_rotation].")

            self.settings = 'primitive'
            self.spin_settings = 'lattice'
            self.relative_settings = 'OSSG'

        else:
            raise TypeError("Input must be either a string index or a list of SpinOperation instances or lists.")

    @cached_property
    def ops(self):
        """Return the sorted operation list."""
        # Group operations should be unique. Some spglib-derived paths can emit
        # duplicated operations, which corrupts downstream group-order
        # invariants such as it * ik.
        ordered_ops = sorted(self._input_ops, key=op_key)
        unique_ops = []
        dedup_tol = min(self.tol, 1e-6)
        bucketed_ops: dict[tuple, list[SpinSpaceGroupOperation]] = {}
        for op in ordered_ops:
            bucket_key = _op_bucket_key(op, dedup_tol)
            candidates = bucketed_ops.get(bucket_key, [])
            if any(op.is_same_with(existing, atol=dedup_tol) for existing in candidates):
                continue
            unique_ops.append(op)
            bucketed_ops.setdefault(bucket_key, []).append(op)
        return unique_ops

    @cached_property
    def spin_translation_group(self):
        return self.get_spin_translation_group()

    @cached_property
    def pure_t_group(self):
        return self.get_pure_translations()

    @cached_property
    def is_primitive(self):
        return self._is_primitive()

    @cached_property
    def sog(self):
        return self.get_spin_only()

    @cached_property
    def collinear_axis(self):
        if self.conf != "Collinear":
            return None
        direction = np.asarray(self.sog_direction, dtype=float)
        if direction.ndim == 2:
            direction = direction[:, 0]
        direction = direction.reshape(3)
        norm = np.linalg.norm(direction)
        if norm < self.tol:
            return None
        return direction / norm

    @cached_property
    def collinear_spin_promotion_order(self):
        if self.conf != "Collinear":
            return None

        axis = self.collinear_axis
        if axis is None:
            return None

        promoted_order = 2
        for op in self.ops:
            rotation = np.asarray(op[1], dtype=float)
            effective_rotation = _effective_proper_rotation(rotation, tol=self.tol)
            if effective_rotation is None:
                continue
            if not _axes_parallel(
                _effective_rotation_axis(rotation, tol=self.tol),
                axis,
                tol=self.tol,
            ):
                continue
            info = describe_point_operation(effective_rotation, tol=self.tol, max_order=120, max_axis_denom=12)
            symbol = info.get("hm_symbol")
            if symbol is None:
                continue
            symbol = symbol.lstrip("-")
            if symbol in {"3", "4", "6"}:
                promoted_order = max(promoted_order, int(symbol))
        return promoted_order

    @cached_property
    def gspg_spin_only_ops(self):
        return deduplicate_matrix_pairs(
            [np.array(op[0]) for op in self.ops if np.allclose(op[1], np.eye(3), atol=self.tol)],
            tol=self.tol,
        )

    @cached_property
    def gspg_spin_only_symbol(self):
        return _gspg_spin_only_symbol_from_rotations(self.gspg_spin_only_ops, self.conf, self.tol)

    @cached_property
    def gspg_ops_raw(self):
        return deduplicate_matrix_pairs([[i[0], i[1]] for i in self.ops], tol=self.tol)

    @cached_property
    def _configuration_data(self):
        """Intermediate tuple used to expose `conf` and `sog_direction`."""
        return self.get_configuration()

    @property
    def conf(self):
        return self._configuration_data[0]

    @property
    def sog_direction(self):
        return self._configuration_data[1]

    @cached_property
    def nssg(self):
        return self.get_nssg()

    @cached_property
    def identity_real_nssg_ops(self):
        identity = np.eye(3)
        return [
            op for op in self.nssg
            if np.allclose(np.asarray(op[1], dtype=float), identity, atol=self.tol, rtol=0)
        ]

    @cached_property
    def n_spin_translation_group(self):
        return self.get_nontrivial_spin_translation_group()

    @cached_property
    def msg_ops(self):
        return self.get_magnetic_space_group_operations()

    @cached_property
    def magnetic_space_group_ops(self):
        return self.msg_ops

    @cached_property
    def msg_info(self):
        return self.get_magnetic_space_group_info()

    @property
    def magnetic_space_group_info(self):
        return self.msg_info

    @property
    def msg_int_num(self):
        return None if self.msg_info is None else self.msg_info.get("msg_int_num")

    @property
    def msg_bns_num(self):
        return None if self.msg_info is None else self.msg_info.get("msg_bns_num")

    @property
    def msg_bns_symbol(self):
        return None if self.msg_info is None else self.msg_info.get("msg_bns_symbol")

    @property
    def msg_og_num(self):
        return None if self.msg_info is None else self.msg_info.get("msg_og_num")

    @property
    def msg_og_symbol(self):
        return None if self.msg_info is None else self.msg_info.get("msg_og_symbol")

    @property
    def msg_type(self):
        return None if self.msg_info is None else self.msg_info.get("msg_type")

    @property
    def mpg_num(self):
        return None if self.msg_info is None else self.msg_info.get("mpg_num")

    @property
    def mpg_symbol(self):
        return None if self.msg_info is None else self.msg_info.get("mpg_symbol")

    # =========================================================================
    # G0 / L0 / group relationship helpers
    # =========================================================================

    @cached_property
    def G0_ops(self):
        return [[i[1], i[2]] for i in self.nssg]

    @cached_property
    def L0_ops(self):
        return [[i[1], i[2]] for i in self.nssg if np.allclose(np.eye(3), i[0], atol=0.1)]

    @cached_property
    def itik(self):
        return check_divisible(len(self.G0_ops), len(self.L0_ops))

    @cached_property
    def ik(self):
        return check_divisible(len(self.n_spin_translation_group), len(self.pure_t_group))

    @cached_property
    def it(self):
        return check_divisible(self.itik, self.ik)

    @cached_property
    def spin_part_point_ops(self):
        return deduplicate_matrix_pairs([i[0] for i in self.ops], tol=0.03)

    @cached_property
    def n_spin_part_point_ops(self):
        return deduplicate_matrix_pairs([i[0] for i in self.nssg], tol=0.03)

    # --- G0 Info ---
    @cached_property
    def _G0_info_data(self):
        """Compute all G0-related data and return it as a pure dictionary."""
        (
            G0_symbol,
            G0_num,
            transformation_matrix_bytes,
            origin_shift_bytes,
        ) = _cached_space_group_dataset_by_key(_space_group_ops_cache_key(self.G0_ops))

        if 74 < G0_num < 195:
            constraint = 'a=b'
        elif G0_num >= 195:
            constraint = 'a=b=c'
        else:
            constraint = None

        transformation_to_G0std_id = _restore_matrix_from_bytes(transformation_matrix_bytes)
        origin_shift_to_G0std_id = _restore_vector_from_bytes(origin_shift_bytes)

        transformation_to_G0std = np.linalg.inv(integerize_matrix(
            np.linalg.inv(transformation_to_G0std_id), mod='col', constraint=constraint))

        origin_shift_to_G0std = transformation_to_G0std @ np.linalg.inv(
            transformation_to_G0std_id) @ origin_shift_to_G0std_id

        if in_space_group([transformation_to_G0std, origin_shift_to_G0std], self.G0_ops, tol=1e-4):
            transformation_to_G0std = np.eye(3)
            origin_shift_to_G0std = np.array([0, 0, 0])

        return {
            'symbol': G0_symbol,
            'num': G0_num,
            'trans_to_std_id': transformation_to_G0std_id,
            'origin_shift_to_std_id': origin_shift_to_G0std_id,
            'trans_to_std': transformation_to_G0std,
            'origin_shift_to_std': origin_shift_to_G0std
        }

    # Expose G0-derived data as properties for API consistency.
    @property
    def G0_symbol(self):
        return self._G0_info_data['symbol']

    @property
    def G0_num(self):
        return self._G0_info_data['num']

    @property
    def transformation_to_G0std_id(self):
        return self._G0_info_data['trans_to_std_id']

    @property
    def origin_shift_to_G0std_id(self):
        return self._G0_info_data['origin_shift_to_std_id']

    @property
    def transformation_to_G0std(self):
        return self._G0_info_data['trans_to_std']

    @property
    def origin_shift_to_G0std(self):
        return self._G0_info_data['origin_shift_to_std']

    # --- L0 Info ---
    @cached_property
    def _L0_info_data(self):
        (
            international,
            number,
            transformation_matrix_bytes,
            origin_shift_bytes,
        ) = _cached_space_group_dataset_by_key(_space_group_ops_cache_key(self.L0_ops))
        return {
            'symbol': international,
            'num': number,
            'trans_to_std': _restore_matrix_from_bytes(transformation_matrix_bytes),
            'origin_shift_to_std': _restore_vector_from_bytes(origin_shift_bytes),
        }

    @property
    def L0_symbol(self):
        return self._L0_info_data['symbol']

    @property
    def L0_num(self):
        return self._L0_info_data['num']

    @property
    def transformation_to_L0std(self):
        return self._L0_info_data['trans_to_std']

    @property
    def origin_shift_to_L0std(self):
        return self._L0_info_data['origin_shift_to_std']

    # --- Index ---
    @cached_property
    def index(self):
        if self._input_index:
            return self._input_index
        return self._get_ssg_index_from_ops()

    # --- Transformations ---
    @cached_property
    def G0std_L0std_transformation(self):
        return self.transformation_to_L0std @ np.linalg.inv(self.transformation_to_G0std_id)

    @cached_property
    def G0std_L0std_origin_shift(self):
        return normalize_vector_to_zero(
            -self.transformation_to_L0std @ np.linalg.inv(
                self.transformation_to_G0std_id) @ self.origin_shift_to_G0std_id + self.origin_shift_to_L0std,
            atol=1e-10
        )

    # =========================================================================
    # Point Group & Configuration Identification
    # =========================================================================

    @cached_property
    def _n_spin_part_pg_info(self):
        return identify_point_group(self.n_spin_part_point_ops, _id = True)

    @property
    def n_spin_part_point_group_symbol_hm(self):
        return self._n_spin_part_pg_info[0]

    @property
    def n_spin_part_std_transformation(self):
        return self._n_spin_part_pg_info[2]

    @property
    def n_spin_part_point_group_symbol_s(self):
        return self._n_spin_part_pg_info[4]

    @cached_property
    def _spin_part_pg_info(self):
        if self.conf != 'Collinear':
            return identify_point_group(self.spin_part_point_ops, _id = True)
        elif self.conf == 'Collinear':
            # Return a dummy record with the same shape as identify_point_group.
            spin_part_order = len(self.spin_part_point_ops)
            if spin_part_order == 4:
                return ('∞m', [], np.eye(3), [], 'C∞v')
            elif spin_part_order == 8:
                return ('∞/mm', [], np.eye(3), [], 'D∞h')
            else:
                raise ValueError('Collinear spin point group identification error')
        else:
            raise ValueError('Configuration identification error')

    @property
    def spin_part_point_group_symbol_hm(self):
        return self._spin_part_pg_info[0]

    @property
    def sppg_ops_info(self):
        return self._spin_part_pg_info[1]

    @property
    def spin_part_std_transformation(self):
        return self._spin_part_pg_info[2]

    @property
    def sppg_generators_index(self):
        return self._spin_part_pg_info[3]

    @property
    def spin_part_point_group_symbol_s(self):
        return self._spin_part_pg_info[4]

    @cached_property
    def spin_part_std_cartesian_transformation(self):
        return np.array([[1, -1 / 2, 0], [0, np.sqrt(3) / 2, 0], [0, 0, 1]]) @ np.linalg.inv(
            self.spin_part_std_transformation)

    # =========================================================================
    # K-Path & Arithmetic Class Info
    # =========================================================================

    @cached_property
    def ncnssg(self):
        return self.get_non_centered_nssg_ops()

    @cached_property
    def gspg(self):
        return self.get_general_spin_point_group_operations()

    @cached_property
    def _effective_PG_data(self):
        return self.get_effective_PG_operations()

    @property
    def eMPG(self):
        return self._effective_PG_data[0]

    @property
    def ekPG(self):
        return self._effective_PG_data[1]

    @cached_property
    def _acc_info_data(self):
        acc, primitive_trans_bytes, primitive_origin_shift_bytes = _cached_acc_symbol_info_by_key(
            _matrix_vector_ops_cache_key([[i, j[1]] for j in self.pure_t_group for i in self.ekPG])
        )
        return (
            acc,
            _restore_matrix_from_bytes(primitive_trans_bytes),
            _restore_vector_from_bytes(primitive_origin_shift_bytes),
        )

    @property
    def acc(self):
        return self._acc_info_data[0]

    @property
    def acc_primitive_trans(self):
        return self._acc_info_data[1]

    @property
    def acc_primitive_origin_shift(self):
        return self._acc_info_data[2]

    @cached_property
    def kpath_info(self):
        return _cached_acc_kpath_info_by_key(
            _matrix_vector_ops_cache_key([[i, j[1]] for j in self.pure_t_group for i in self.ekPG])
        )

    @cached_property
    def acc_num(self):
        from findspingroup.data import ARITHMETIC_CRYSTAL_CLASS
        mapper = {value: key for key, value in ARITHMETIC_CRYSTAL_CLASS.SYMMORPHIC_SPACE_GROUPNUM__ACCSYMBOL.items()}
        if self.acc in mapper:
            return mapper[self.acc]
        else:
            raise ValueError('arithmetic_crystal_class error')

    @cached_property
    def cptrans(self):
        from findspingroup.data import ARITHMETIC_CRYSTAL_CLASS
        if self.acc_num in ARITHMETIC_CRYSTAL_CLASS.COMPLEXACC:
            return np.array(ARITHMETIC_CRYSTAL_CLASS.CONVENTIONAL_PRIMITIVE_TRANSFORMATIONS[
                                ARITHMETIC_CRYSTAL_CLASS.COMPLEXACC[self.acc_num]])
        else:
            return np.eye(3)

    @cached_property
    def _kpoints_data(self):
        from findspingroup.data import ARITHMETIC_CRYSTAL_CLASS
        k_conv = ARITHMETIC_CRYSTAL_CLASS.ACC_K_POINTS_CONVENTIONAL[self.acc_num]
        k_prim = ARITHMETIC_CRYSTAL_CLASS.ACC_K_POINTS_PRIMITIVE[self.acc_num]

        k_sym_c, k_val_c = zip(*k_conv)
        k_sym_p, k_val_p = zip(*k_prim)

        k_label, k_prim_str = zip(*(parse_label_and_value(i) for i in k_sym_p))
        return {
            'conv_sym': k_sym_c, 'conv_val': k_val_c,
            'prim_sym': k_sym_p, 'prim_val': k_val_p,
            'label': k_label, 'prim_str': k_prim_str
        }

    @property
    def kpoints_symbol_conventional(self):
        return self._kpoints_data['conv_sym']

    @property
    def kpoints_conventional(self):
        return self._kpoints_data['conv_val']

    @property
    def kpoints_symbol_primitive(self):
        return self._kpoints_data['prim_sym']

    @property
    def kpoints_primitive(self):
        return self._kpoints_data['prim_val']

    @property
    def kpoints_label(self):
        return self._kpoints_data['label']

    @property
    def kpoints_primitive_string(self):
        return self._kpoints_data['prim_str']

    @cached_property
    def little_groups(self):
        return self.get_little_groups()

    @cached_property
    def _little_group_spin_analysis(self):
        analysis = []
        for little_group in self.little_groups:
            spin_matrices = deduplicate_matrix_pairs(
                [op[0] - np.eye(3) for op in little_group],
                tol=self.tol,
            )
            stacked = np.vstack(spin_matrices)
            singular_values = np.linalg.svd(stacked.astype(np.float32))[1]
            spin_splitting = (
                'no spin splitting'
                if all(abs(value) > 1e-3 for value in singular_values)
                else 'spin splitting'
            )
            polarizations = combine_parametric_solutions(rref_with_tolerance(stacked))
            analysis.append(
                {
                    "spin_splitting": spin_splitting,
                    "spin_polarizations": polarizations,
                }
            )
        return analysis

    @cached_property
    def little_groups_symbols(self):
        return self.get_little_groups_symbols()

    @cached_property
    def is_spinsplitting(self):
        return self.is_spin_splitting()

    @cached_property
    def spin_polarizations(self):
        return self.get_spin_polarizations()

    @cached_property
    def KPOINTS(self):
        return self.get_KPOINTS()

    @cached_property
    def is_PT(self):
        return self._is_PT()

    @cached_property
    def seitz_descriptions(self):
        return self.get_seitz_descriptions(tol=self.symbol_calibration_tol)

    @cached_property
    def seitz_symbols(self):
        return [item["symbol"] for item in self.seitz_descriptions]

    @cached_property
    def seitz_symbols_latex(self):
        return [item["symbol_latex"] for item in self.seitz_descriptions]

    @cached_property
    def international_symbol(self):
        return self.get_international_symbol(tol=self.symbol_calibration_tol)

    @cached_property
    def international_symbol_current_frame(self):
        return self.get_international_symbol(
            tol=self.symbol_calibration_tol,
            basis_mode="current",
        )

    @cached_property
    def international_symbol_linear(self):
        return self.international_symbol["linear"]

    @cached_property
    def international_symbol_latex(self):
        return self.international_symbol["latex"]

    @cached_property
    def international_symbol_linear_current_frame(self):
        return self.international_symbol_current_frame["linear"]

    @cached_property
    def international_symbol_latex_current_frame(self):
        return self.international_symbol_current_frame["latex"]

    @cached_property
    def international_symbol_type(self):
        return self.international_symbol["type"]

    @cached_property
    def symbol_calibration_tol(self):
        return calibrated_symbol_tol(self.tol)

    # =========================================================================
    # Original logic methods (static and instance behavior unchanged)
    # =========================================================================

    @staticmethod
    def is_close_matrix_pair(pair1, pair2, tol=1e-5):
        if len(pair1) != len(pair2):
            raise ValueError("Compare two vectors of different lengths.")
        for i, j in enumerate(pair1):
            if not np.allclose(pair1[i], pair2[i], atol=tol):
                return False
        return True

    @staticmethod
    def has_op(target_op, operations, tol=1e-5):
        for op in operations:
            if SpinSpaceGroup.is_close_matrix_pair(op, target_op, tol):
                return True
        return False

    def _is_PT(self):
        for op in self.ops:
            if np.allclose(-np.eye(3), op[1], atol=self.tol) and np.allclose(-np.eye(3), op[0], atol=self.tol):
                return True
        return False

    def get_seitz_descriptions(self, tol=1e-6, max_order=120, max_axis_denom=12):
        symbol_tol = calibrated_symbol_tol(tol)
        descriptions = [
            op.seitz_description(tol=symbol_tol, max_order=max_order, max_axis_denom=max_axis_denom)
            for op in self.ops
        ]
        return canonicalize_group_seitz_descriptions(
            descriptions,
            tol=symbol_tol,
            max_axis_denom=max_axis_denom,
        )

    def get_international_symbol(self, tol=1e-4, *, basis_mode: str = "standard"):
        return build_international_symbol(self, tol=tol, basis_mode=basis_mode)

    def get_KPOINTS(self):
        spin_splitting_info = [(self.kpoints_label[i], self.kpoints_primitive_string[i], True)
                               if j == 'spin splitting' else (self.kpoints_label[i], self.kpoints_primitive_string[i],
                                                              False)
                               for i, j in enumerate(self.is_spinsplitting)]
        matcher = BrillouinZoneMatcher(spin_splitting_info)
        low_symm_indices = find_uvw_whole_string(self.kpoints_symbol_primitive)
        extra_point_info = [(self.kpoints_primitive[ind], self.kpoints_label[ind]) for ind in low_symm_indices]
        return write_kpoints(self.kpath_info, matcher, extra_kpoints=extra_point_info)

    def get_little_groups_symbols(self):
        latex_symbols = []
        for index, little_group in enumerate(self.little_groups):
            try:
                spin_part = deduplicate_matrix_pairs([np.array(op[0]) for op in little_group])
                real_part = deduplicate_matrix_pairs([np.array(op[1]) for op in little_group])
                spin_info = identify_point_group(spin_part)
                real_info = identify_point_group(real_part)

                if self.conf == 'Collinear':
                    t_count = 0
                    for op in little_group:
                        if np.allclose(np.array(op[1]), np.eye(3), self.tol):
                            t_count += 1
                    if t_count == 2:
                        spin_only_symbol = '^{\\infty }1'
                    elif t_count == 4:
                        spin_only_symbol = '^{\\infty m}1'
                    elif t_count == 8:
                        spin_only_symbol = '^{\\infty /mm}1'
                    else:
                        raise ValueError(
                            f'Wrong spin translation group of k little group {self.kpoints_symbol_primitive[index]}')  # Fixed symbol reference
                else:
                    general_spin_only = []
                    for op in little_group:
                        if np.allclose(np.array(op[1]), np.eye(3), self.tol):
                            general_spin_only.append(np.array(op[0]))
                    pg_symbol = identify_point_group(general_spin_only)[0]
                    if pg_symbol != '1':
                        spin_only_symbol = f"^{{{pg_symbol}}}1"
                    else:
                        spin_only_symbol = ''

                # match spin op
                spin_generators = []
                for index_g in real_info[3]:  # fixed loop variable name clash
                    for op in little_group:
                        if np.allclose(np.array(op[1]), real_info[1][index_g][0], self.tol):
                            spin_generators.append(op[0])
                            break

                spin_generators_symbols = []
                for spin_op in spin_generators:
                    for op in spin_info[1]:
                        if np.allclose(np.array(op[0]), spin_op, self.tol):
                            spin_generators_symbols.append(op[2])

                latex = ''
                if bool(re.search(r'/', real_info[0])):
                    count = 0
                    for i, index_g in enumerate(real_info[3]):
                        if count == 1:
                            latex = latex + '/' + '^{' + spin_generators_symbols[i] + '}' + real_info[1][index_g][2]
                        else:
                            latex = latex + '^{' + spin_generators_symbols[i] + '}' + real_info[1][index_g][2]
                        count += 1
                    latex = latex + spin_only_symbol
                else:
                    for i, index_g in enumerate(real_info[3]):
                        latex = latex + '^{' + spin_generators_symbols[i] + '}' + real_info[1][index_g][2]
                    latex = latex + spin_only_symbol

                latex_symbols.append(latex)
            except ValueError:
                # Little-group symbol generation is display metadata. If PG
                # standardization fails on one little group, degrade to a
                # placeholder instead of failing the whole result/UI payload.
                latex_symbols.append('?')
        return latex_symbols

    def get_little_groups(self):
        k_little_groups = []
        if self.is_primitive:
            kpoints = self.kpoints_primitive
        else:
            kpoints = self.kpoints_conventional

        effective_ops = [
            np.linalg.det(op[0]) * np.array(np.linalg.inv(op[1]).T)
            for op in self.gspg_ops_raw
        ]

        if self.cptrans is None or np.allclose(self.cptrans, np.eye(3)):
            # Simplified logic check: if P center lattice not complex lattice
            for k_point in kpoints:
                primitive_kpoint = np.asarray(k_point, dtype=float)
                little_group = []
                for op, eop in zip(self.gspg_ops_raw, effective_ops):
                    target_kpoint = eop @ primitive_kpoint % 1
                    diff = getNormInf(primitive_kpoint % 1, target_kpoint)
                    if diff < self.tol:
                        little_group.append(op)
                k_little_groups.append(little_group)
        else:
            # if complex center lattice
            cptrans = np.asarray(self.cptrans, dtype=float)
            cptrans_inv = np.linalg.inv(cptrans)
            conjugated_effective_ops = [cptrans_inv @ eop @ cptrans for eop in effective_ops]
            for k_point in kpoints:
                kpoint_array = np.asarray(k_point, dtype=float)
                little_group = []
                for op, eop, conjugated_eop in zip(self.gspg_ops_raw, effective_ops, conjugated_effective_ops):
                    if self.is_primitive:
                        target_kpoint = eop @ kpoint_array % 1
                        diff = getNormInf(kpoint_array % 1, target_kpoint)
                        if diff < self.tol:
                            little_group.append(op)
                    else:
                        # Check complex lattice condition
                        primitive_kpoint = cptrans.T @ kpoint_array % 1
                        transformed_primitive = conjugated_eop @ primitive_kpoint % 1
                        if getNormInf(primitive_kpoint, transformed_primitive) < self.tol:
                            little_group.append(op)
                k_little_groups.append(little_group)
        return k_little_groups

    def _is_primitive(self):
        if len(self.pure_t_group) > 1:
            return False
        else:
            return True

    def get_effective_PG_operations(self):
        effective_magnetic_point_group = []
        effective_k_point_group = []
        for i in self.gspg_ops_raw:
            if abs(np.linalg.det(i[0]) - 1) < self.tol:
                effective_magnetic_point_group.append([np.eye(3), i[1]])
                effective_k_point_group.append(np.array(i[1]))
            elif abs(np.linalg.det(i[0]) + 1) < self.tol:
                effective_magnetic_point_group.append([-np.eye(3), i[0]])
                effective_k_point_group.append(-1 * np.array(i[1]))
            else:
                raise ValueError('tolerance error when getting general spin point group')
        return deduplicate_matrix_pairs(effective_magnetic_point_group, tol=self.tol), deduplicate_matrix_pairs(
            effective_k_point_group, tol=self.tol)

    def get_general_spin_point_group_operations(self)->'GeneralizedSpinPointGroup':
        if self.conf == "Collinear":
            ops = deduplicate_matrix_pairs(
                [[np.asarray(op[0], dtype=float), np.asarray(op[1], dtype=float)] for op in self.nssg],
                tol=self.tol,
            )
            raw_ops = self.gspg_ops_raw
        else:
            ops = self.gspg_ops_raw
            raw_ops = ops
        point_part_linear = self.international_symbol.get("point_part_linear", "")
        point_part_latex = self.international_symbol.get("point_part_latex", "")
        spin_only_linear = self.gspg_spin_only_symbol["linear"]
        spin_only_latex = self.gspg_spin_only_symbol["latex"]

        symbol_linear = f"{point_part_linear} {spin_only_linear}".strip() if point_part_linear else spin_only_linear
        symbol_latex = point_part_latex + spin_only_latex if point_part_latex else spin_only_latex

        return GeneralizedSpinPointGroup(
            ops,
            raw_ops=raw_ops,
            configuration=self.conf,
            collinear_axis=self.collinear_axis,
            symbol_linear=symbol_linear,
            symbol_latex=symbol_latex,
            point_part_linear=point_part_linear,
            point_part_latex=point_part_latex,
            spin_only_linear=spin_only_linear,
            spin_only_latex=spin_only_latex,
            spin_only_symbol_hm=self.gspg_spin_only_symbol["hm"],
            spin_only_symbol_s=self.gspg_spin_only_symbol["s"],
            tol=self.tol,
        )

    def get_non_centered_nssg_ops(self):
        eq_class = []
        num_ops = len(self.nssg)
        assigned = [False] * num_ops
        for i in range(num_ops):
            if assigned[i]: continue
            class_i = []
            for j in self.n_spin_translation_group:
                check_op = self.nssg[i] @ j
                for k in range(num_ops):
                    if assigned[k]: continue
                    if self.nssg[k].is_same_with(check_op, self.tol):
                        class_i.append(k)
                        assigned[k] = True
                        break
            eq_class.append(class_i)
        if len(set([len(i) for i in eq_class])) != 1:
            raise ValueError('Wrong number of co-set. Check tolerance!')
        non_centered_nssg_ops = []
        for i in eq_class:
            non_centered_nssg_ops.append(self.nssg[i[0]])
        return non_centered_nssg_ops

    def get_nontrivial_spin_translation_group(self):
        nontrivial_spin_translation_group = []
        for i in self.nssg:
            if np.allclose(i[1], np.eye(3), self.tol):
                nontrivial_spin_translation_group.append(i)
        return nontrivial_spin_translation_group

    def get_nssg(self):
        nssg = []
        if self.conf == 'Collinear':
            for i in self.ops:
                if np.allclose(i[0], -np.eye(3), self.tol) or np.allclose(i[0], np.eye(3), self.tol):
                    nssg.append(i)
        elif self.conf == 'Coplanar':
            for i in self.ops:
                if np.linalg.det(i[0]) > 0:
                    nssg.append(i)
        else:
            nssg = self.ops
        return nssg

    def get_configuration(self):
        if len(self.sog) == 8 or len(self.sog) == 4:
            for operations in self.sog:
                if not np.allclose(operations[0], np.eye(3), atol=0.1) and abs(np.linalg.det(operations[0]) - 1) < 1e-2:
                    eigvals, eigvecs = np.linalg.eig(operations[0])
                    direction = eigvecs[:, np.isclose(eigvals, 1.0, atol=0.1)].real
                    return 'Collinear', direction
        if len(self.sog) == 2:
            for operations in self.sog:
                if not np.allclose(operations[0], np.eye(3), atol=0.1):
                    eigvals, eigvecs = np.linalg.eig(operations[0])
                    direction = eigvecs[:, np.isclose(eigvals, -1.0, atol=0.1)].real
                    return 'Coplanar', direction
        if len(self.sog) == 1:
            return 'Noncoplanar', None
        raise ValueError('Wrong spin only groups. Check tolerance!')

    def get_spin_translation_group(self):
        spin_translation_group = []
        for op in self.ops:
            if np.allclose(op[1], np.eye(3), self.tol):
                spin_translation_group.append(op)
        return deduplicate_matrix_pairs(spin_translation_group)

    def get_magnetic_space_group_operations(self):
        candidate_ops = self.ops
        if self.conf == "Collinear":
            candidate_ops = self._build_collinear_msg_candidate_ops()
        return [
            op
            for op in candidate_ops
            if self.classify_magnetic_operation(op) is not None
        ]

    def get_magnetic_space_group_info(self):
        magnetic_space_group_operations = []
        for op in self.msg_ops:
            time_reversal = self.classify_magnetic_operation(op)
            if time_reversal is None:
                continue
            magnetic_space_group_operations.append(
                [
                    int(time_reversal),
                    np.asarray(op[1], dtype=float),
                    np.asarray(op[2], dtype=float),
                ]
            )
        if not magnetic_space_group_operations:
            return None
        return _cached_magnetic_space_group_info_by_key(
            _magnetic_space_group_ops_cache_key(magnetic_space_group_operations)
        )

    def classify_magnetic_operation(self, op):
        return op.magnetic_time_reversal(atol=self.tol)

    def _collinear_spin_only_promotion_rotations(self):
        axis = self.collinear_axis
        reduced_spin_only = deduplicate_matrix_pairs(
            [np.asarray(op[0], dtype=float) for op in self.sog],
            tol=self.tol,
        )
        if axis is None:
            return reduced_spin_only

        axis_preserving_real_rotations = []
        for op in self.ops:
            rotation = np.asarray(op[1], dtype=float)
            effective_rotation = _effective_proper_rotation(rotation, tol=self.tol)
            if effective_rotation is None:
                continue
            if not _axes_parallel(
                _effective_rotation_axis(rotation, tol=self.tol),
                axis,
                tol=self.tol,
            ):
                continue
            info = describe_point_operation(effective_rotation, tol=self.tol, max_order=120, max_axis_denom=12)
            if info.get("hm_symbol", "").lstrip("-") not in {"3", "4", "6"}:
                continue
            axis_preserving_real_rotations.append(effective_rotation)

        if not axis_preserving_real_rotations:
            return reduced_spin_only

        unique_axial_rotations = deduplicate_matrix_pairs(axis_preserving_real_rotations, tol=self.tol)
        rotations_with_order = []
        for rotation in unique_axial_rotations:
            info = describe_point_operation(rotation, tol=self.tol, max_order=120, max_axis_denom=12)
            order = info.get("order")
            if order is None:
                continue
            rotations_with_order.append((int(order), rotation))

        if not rotations_with_order:
            return reduced_spin_only

        highest_order = max(order for order, _ in rotations_with_order)
        highest_order_generators = [rotation for order, rotation in rotations_with_order if order == highest_order]

        return _matrix_group_closure(
            list(reduced_spin_only) + highest_order_generators[:1],
            tol=self.tol,
        )

    def _build_collinear_perpendicular_direct_candidates(self):
        axis = self.collinear_axis
        if axis is None:
            return []

        direct_candidates = []
        for op in self.nssg:
            rotation = np.asarray(op[1], dtype=float)
            effective_rotation = _effective_proper_rotation(rotation, tol=self.tol)
            if effective_rotation is None:
                continue

            info = describe_point_operation(
                effective_rotation,
                tol=self.tol,
                max_order=120,
                max_axis_denom=12,
            )
            if info.get("hm_symbol", "").lstrip("-") != "2":
                continue

            effective_axis = _effective_rotation_axis(rotation, tol=self.tol)
            if not _axes_perpendicular(
                effective_axis,
                axis,
                metric=self.real_space_metric,
                tol=self.tol,
            ):
                continue

            # Perpendicular m / 2 candidates correspond to the primed branch,
            # so the promoted spin-only action uses -R_eff rather than +R_eff.
            spin_only_op = SpinSpaceGroupOperation(-effective_rotation, np.eye(3), np.zeros(3))
            direct_candidates.append(spin_only_op @ op)

        unique_ops = []
        for op in sorted(direct_candidates, key=op_key):
            if any(op.is_same_with(existing, atol=self.tol) for existing in unique_ops):
                continue
            unique_ops.append(op)
        return unique_ops

    def _build_collinear_msg_candidate_ops(self):
        promotion_rotations = self._collinear_spin_only_promotion_rotations()
        direct_candidates = self._build_collinear_perpendicular_direct_candidates()
        reduced_spin_only = deduplicate_matrix_pairs(
            [np.asarray(op[0], dtype=float) for op in self.sog],
            tol=self.tol,
        )
        if _matrix_sets_match(promotion_rotations, reduced_spin_only, tol=self.tol) and not direct_candidates:
            return self.ops

        spin_only_ops = [
            SpinSpaceGroupOperation(spin_rotation, np.eye(3), np.zeros(3))
            for spin_rotation in promotion_rotations
        ]
        candidate_ops = []
        for spin_only_op in spin_only_ops:
            for op in self.nssg:
                candidate_ops.append(spin_only_op @ op)
        candidate_ops.extend(direct_candidates)

        unique_ops = []
        for op in sorted(candidate_ops, key=op_key):
            if any(op.is_same_with(existing, atol=self.tol) for existing in unique_ops):
                continue
            unique_ops.append(op)
        return unique_ops

    def get_pure_translations(self):
        pure_translations = []
        for op in self.spin_translation_group:
            if np.allclose(op[0], np.eye(3), self.tol):
                pure_translations.append([op[1], op[2]])
        return pure_translations

    def get_spin_only(self):
        spin_only_group = []
        for i in self.spin_translation_group:
            if np.allclose(i[2], np.zeros(3), atol=1e-5):
                spin_only_group.append(i)
        return spin_only_group

    def build_multiplication_table(self):
        n = len(self.ops)
        table = np.empty((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                product = self.ops[i] @ self.ops[j]
                for k in range(n):
                    if product == self.ops[k]:
                        table[i, j] = k
                        break
                else:
                    raise ValueError("Product not found in group operations.")
        return table

    def _get_ssg_index_from_ops(self) -> str:
        G0 = self.G0_num
        L0 = self.L0_num
        it = self.it
        ik = self.ik
        return f"{G0}.{L0}.{it}.{ik}"

    def validate_nsspg_invariants(self):
        expected_order = len(self.n_spin_part_point_ops)
        actual_order = self.it * self.ik
        if actual_order != expected_order:
            raise ValueError(
                "Inconsistent NSSPG invariants: "
                f"it*ik={actual_order}, |nsspg|={expected_order}, "
                f"it={self.it}, ik={self.ik}, spin_pg={self.n_spin_part_point_group_symbol_s}, "
                f"G0={self.G0_num}, L0={self.L0_num}."
            )
        return True

    def get_index(self):
        return self.index

    def __hash__(self):
        return hash(self.index)

    def __len__(self):
        return len(self.ops)

    def __repr__(self):
        lines = [f"<SpinSpaceGroup #{self.index} '>"]
        for i, group in enumerate(self.ops):
            lines.append(f"Group {i+1}:")
            mats = [np.atleast_2d(np.array(m)).reshape(3, -1) for m in group]
            mat_strs = [np.array2string(m,
                                        formatter={'float_kind': lambda x: f"{x:5.3f}"},
                                        separator=' ', suppress_small=True).splitlines()
                        for m in mats]
            widths = [max(len(line) for line in s) for s in mat_strs]
            n_rows = max(len(s) for s in mat_strs)
            for r in range(n_rows):
                row_parts = []
                for j, s in enumerate(mat_strs):
                    content = s[r] if r < len(s) else " " * widths[j]
                    row_parts.append(content.ljust(widths[j]))
                    if j == len(mat_strs) - 2:
                        row_parts.append("   |   ")
                    else:
                        row_parts.append("   ")
                lines.append("".join(row_parts))
            lines.append("")
        return "\n".join(lines)

    def get_generators(self):
        return find_group_generators(self.ops)

    def to_spin_point_group(self):
        spg_ops = [op.to_spg_op() for op in self.ops]
        return GeneralizedSpinPointGroup(spg_ops)

    def transform(self, transformation_matrix, origin_shift, frac=True, all_trans=True):
        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        if frac:
            translations = integer_points_in_new_cell(transformation_matrix_inv.T)
        else:
            translations = [np.zeros(3)]
        if not all_trans:
            translations = [np.zeros(3)]

        new_ops = []
        for op in [[i[0], i[1], i[2] + np.array(j)] for i in self.ops for j in translations]:
            new_rotation = transformation_matrix @ op[1] @ transformation_matrix_inv
            if frac:
                new_translation = normalize_vector_to_zero(
                    ((np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ op[2]), atol=1e-4)
            else:
                new_translation = ((np.eye(3) - new_rotation) @ origin_shift + transformation_matrix @ op[2])
            new_op = SpinSpaceGroupOperation(op[0], new_rotation, new_translation)
            new_ops.append(new_op)

        # Return a new instance that lazily recomputes its derived properties.
        new_metric = self.real_space_metric
        if new_metric is not None:
            transformation_matrix = np.asarray(transformation_matrix, dtype=float)
            transformation_matrix_inv = np.linalg.inv(transformation_matrix)
            new_metric = transformation_matrix_inv.T @ new_metric @ transformation_matrix_inv
        return SpinSpaceGroup(new_ops, tol=self.tol, real_space_metric=new_metric)

    def transform_spin(self, spin_transformation_matrix):
        spin_transformation_matrix_inv = np.linalg.inv(spin_transformation_matrix)
        new_ops = []
        for op in self.ops:
            new_spin_rotation = spin_transformation_matrix @ op[0] @ spin_transformation_matrix_inv
            new_op = SpinSpaceGroupOperation(new_spin_rotation, op[1], op[2])
            new_ops.append(new_op)
        return SpinSpaceGroup(new_ops, tol=self.tol, real_space_metric=self.real_space_metric)

    def get_attributes_from_database(self):
        attributes = {
            "crystal_system": "cubic",
            "point_group": "m-3m",
            "lattice_type": "P",
        }
        return attributes

    def is_spin_splitting(self):
        return [item["spin_splitting"] for item in self._little_group_spin_analysis]

    def get_spin_polarizations(self):
        return [item["spin_polarizations"] for item in self._little_group_spin_analysis]

class GeneralizedSpinPointGroup:
    def __init__(
        self,
        ops,
        *,
        raw_ops=None,
        configuration: str | None = None,
        collinear_axis=None,
        symbol_linear: str | None = None,
        symbol_latex: str | None = None,
        point_part_linear: str | None = None,
        point_part_latex: str | None = None,
        spin_only_linear: str | None = None,
        spin_only_latex: str | None = None,
        spin_only_symbol_hm: str | None = None,
        spin_only_symbol_s: str | None = None,
        tol: float = 1e-3,
    ):
        self.ops = ops
        self.raw_ops = ops if raw_ops is None else raw_ops
        self.configuration = configuration
        self.collinear_axis = (
            None if collinear_axis is None else np.asarray(collinear_axis, dtype=float)
        )
        self.public_ops_are_reduced = self.raw_ops is not self.ops and len(self.raw_ops) != len(self.ops)
        self.symbol_linear = symbol_linear
        self.symbol_latex = symbol_latex
        self.point_part_linear = point_part_linear
        self.point_part_latex = point_part_latex
        self.spin_only_linear = spin_only_linear
        self.spin_only_latex = spin_only_latex
        self.spin_only_symbol_hm = spin_only_symbol_hm
        self.spin_only_symbol_s = spin_only_symbol_s
        self.tol = float(tol)

    def __getitem__(self, index):
        return self.ops[index]

    def format_operations(self):
        lines = [f"<GeneralizedSpinPointGroup with {len(self.ops)} operations>"]
        for i, group in enumerate(self.ops):
            lines.append(f"Group {i}:")
            mats = [np.atleast_2d(np.array(m)).reshape(3, -1) for m in group]
            mat_strs = [np.array2string(m,
                                        formatter={'float_kind': lambda x: f"{x:5.3f}"},
                                        separator=' ', suppress_small=True).splitlines()
                        for m in mats]
            widths = [max(len(line) for line in s) for s in mat_strs]
            n_rows = max(len(s) for s in mat_strs)
            for r in range(n_rows):
                row_parts = []
                for j, s in enumerate(mat_strs):
                    content = s[r] if r < len(s) else " " * widths[j]
                    row_parts.append(content.ljust(widths[j]))
                    if j == len(mat_strs) - 2:
                        row_parts.append("   |   ")
                    else:
                        row_parts.append("   ")
                lines.append("".join(row_parts))
            lines.append("")
        return "\n".join(lines)

    def __repr__(self):
        return self.symbol_linear or self.format_operations()

    def to_dict(self):
        return {
            "ops": [
                [
                    np.asarray(spin_rotation, dtype=float).tolist(),
                    np.asarray(space_rotation, dtype=float).tolist(),
                ]
                for spin_rotation, space_rotation in self.ops
            ],
            "raw_ops": [
                [
                    np.asarray(spin_rotation, dtype=float).tolist(),
                    np.asarray(space_rotation, dtype=float).tolist(),
                ]
                for spin_rotation, space_rotation in self.raw_ops
            ],
            "configuration": self.configuration,
            "collinear_axis": None if self.collinear_axis is None else self.collinear_axis.tolist(),
            "public_ops_are_reduced": self.public_ops_are_reduced,
            "symbol_linear": self.symbol_linear,
            "symbol_latex": self.symbol_latex,
            "point_part_linear": self.point_part_linear,
            "point_part_latex": self.point_part_latex,
            "spin_only_linear": self.spin_only_linear,
            "spin_only_latex": self.spin_only_latex,
            "spin_only_symbol_hm": self.spin_only_symbol_hm,
            "spin_only_symbol_s": self.spin_only_symbol_s,
            "effective_magnetic_point_group": [
                [int(time_reversal), np.asarray(rotation, dtype=float).tolist()]
                for time_reversal, rotation in self.effective_magnetic_point_group
            ],
            "effective_magnetic_point_group_symbol": self.empg_symbol,
            "symbol_calibration_tol": calibrated_symbol_tol(self.tol),
        }

    @cached_property
    def effective_magnetic_point_group(self):
        return self.get_effective_magnetic_point_group()

    @cached_property
    def empg_symbol(self):
        return self.get_empg_symbol()

    def get_effective_magnetic_point_group(self):
        effective_magnetic_point_group = []
        for i in self.raw_ops:
            if abs(np.linalg.det(i[0]) - 1) < self.tol:
                effective_magnetic_point_group.append([1, i[1]])
            elif abs(np.linalg.det(i[0]) + 1) < self.tol:
                effective_magnetic_point_group.append([-1, i[1]])
            else:
                raise ValueError('tolerance error when getting general spin point group')

        return deduplicate_matrix_pairs(effective_magnetic_point_group, tol=self.tol)


    def get_empg_symbol(self):
        empg_ops = self.effective_magnetic_point_group
        for i in empg_ops:
            if i[0]==-1 and np.allclose(i[1],np.eye(3),atol=1e-5):
                try:
                    effective_space_rotation = deduplicate_matrix_pairs([_[1] for _ in empg_ops], tol=1e-5)
                    empg_symbol=get_space_group_from_operations([[j,np.array([0,0,0])] for j in effective_space_rotation]).pointgroup+"1'"
                except:
                    empg_symbol=None
                return empg_symbol
        empg_info = get_magnetic_space_group_from_operations([[i[0],i[1],np.array([0,0,0])] for i in empg_ops])
        return empg_info['mpg_symbol'] if empg_info else None
