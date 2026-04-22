import math
from fractions import Fraction
import re
import warnings

import numpy as np
from findspingroup.version import __version__
from findspingroup.structure import CrystalCell, SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.utils.matrix_utils import normalize_vector_to_zero
from findspingroup.utils.symbolic_format import format_symbolic_scalar


SCIF_OPERATION_FULL_PRECISION = 15


def getangletwovector(v1,v2):
    dot_product = np.dot(v1, v2)

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    cos_angle = dot_product / (v1_norm * v2_norm)

    radian = np.arccos(np.clip(cos_angle, -1, 1))

    degree = np.degrees(radian)
    return degree

def getprimitivelattice(lattice):
    a= np.linalg.norm(lattice[0])
    b= np.linalg.norm(lattice[1])
    c= np.linalg.norm(lattice[2])
    alpha = getangletwovector(lattice[2],lattice[1])
    beta = getangletwovector(lattice[0],lattice[2])
    gamma = getangletwovector(lattice[0],lattice[1])
    return a,b,c,alpha,beta,gamma

def _scif_spin_tag_names() -> dict[str, str]:
    return {
        "collinear_direction": "_space_group_spin.collinear_direction_xyz",
        "coplanar_perp_uvw": "_space_group_spin.coplanar_perp_uvw",
        "rotation_axis": "_space_group_spin.rotation_axis",
        "rotation_axis_cartn": None,
        "rotation_angle": "_space_group_spin.rotation_angle",
        "ssg_number": "_space_group_spin.number_Chen",
        "ssg_name": "_space_group_spin.name_Chen",
        "ssg_name_linear": "_space_group_spin.name_Chen",
    }


def _scif_repo_local_extension_tag_names() -> dict[str, str]:
    return {
        "ssg_name_linear": "_space_group_spin.fsg_oriented_spin_space_group_name_linear",
        "ssg_name_latex": "_space_group_spin.fsg_oriented_spin_space_group_name_latex",
        "magnetic_phase": "_space_group_spin.fsg_magnetic_phase",
        "spin_arithmetic_crystal_class_symbol": "_space_group_spin.fsg_spin_arithmetic_crystal_class_symbol",
        "magnetic_arithmetic_crystal_class_symbol": "_space_group_spin.fsg_magnetic_arithmetic_crystal_class_symbol",
        "parent_space_group_status": "_space_group_spin.fsg_parent_space_group_status",
        "parent_space_group_matches_input": "_space_group_spin.fsg_parent_space_group_matches_input",
        "input_parent_space_group_name": "_space_group_spin.fsg_input_parent_space_group_name_H-M_alt",
        "input_parent_space_group_number": "_space_group_spin.fsg_input_parent_space_group_IT_number",
        "G0_number": "_space_group_spin.fsg_G0_number",
        "L0_number": "_space_group_spin.fsg_L0_number",
        "it": "_space_group_spin.fsg_it",
        "ik": "_space_group_spin.fsg_ik",
        "spin_space_point_group_name": "_space_group_spin.fsg_spin_space_point_group_name",
        "transform_to_input_Pp": "_space_group_spin.fsg_transform_to_input_Pp",
        "transform_to_magnetic_primitive_Pp": "_space_group_spin.fsg_transform_to_magnetic_primitive_Pp",
        "transform_to_L0std_Pp": "_space_group_spin.fsg_transform_to_L0std_Pp",
        "transform_to_G0std_Pp": "_space_group_spin.fsg_transform_to_G0std_Pp",
    }


def _quote_scif_string(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f"\"{escaped}\""


def _format_scif_float(value: float, precision: int = 6, zero_tol: float = 1e-12) -> str:
    numeric = float(value)
    if abs(numeric) <= zero_tol:
        return "0"
    return f"{numeric:.{precision}f}".rstrip("0").rstrip(".")


def _stabilize_fractional_boundary_value(value: float, *, boundary_tol: float = 1e-5) -> float:
    numeric = float(value)
    wrapped = numeric % 1.0
    if abs(wrapped) < boundary_tol or abs(wrapped - 1.0) < boundary_tol:
        return 0.0
    return numeric


def _format_scif_symbolic_scalar(
    value: float,
    *,
    decimal_precision: int = SCIF_OPERATION_FULL_PRECISION,
    zero_tol: float = 1e-12,
    rational_tol: float = 1e-9,
    sqrt_tol: float = 5e-6,
    max_denominator: int = 12,
    sqrt_values: tuple[int, ...] = (2, 3, 5, 6),
) -> str:
    return format_symbolic_scalar(
        value,
        decimal_precision=decimal_precision,
        zero_tol=zero_tol,
        rational_tol=rational_tol,
        sqrt_tol=sqrt_tol,
        max_denominator=max_denominator,
        sqrt_values=sqrt_values,
    )


def write_scif_spin_only(conf, spin_only_direction):
    if spin_only_direction is not None:
        direction = []
        for i in spin_only_direction:
            if abs(i) < 1e-4:
                direction.append(0)
            else:
                direction.append(i)
    def _format_collinear_direction_for_scif(direction_values):
        numeric = np.asarray(
            [
                value.item() if hasattr(value, "item") else value
                for value in direction_values
            ],
            dtype=float,
        ).reshape(-1)
        if np.linalg.norm(numeric) < 1e-10:
            return ",".join(
                _format_scif_symbolic_scalar(
                    i.item() if hasattr(i, "item") else i,
                    decimal_precision=6,
                )
                for i in direction_values
            )

        rounded_int = np.rint(numeric).astype(int)
        if np.allclose(numeric, rounded_int, atol=1e-4):
            ints = rounded_int.tolist()
        else:
            nonzero = [abs(v) for v in numeric if abs(v) > 1e-6]
            if not nonzero:
                ints = [0, 0, 0]
            else:
                scale = min(nonzero)
                scaled = numeric / scale
                rounded_scaled = np.rint(scaled).astype(int)
                if np.allclose(scaled, rounded_scaled, atol=1e-3):
                    ints = rounded_scaled.tolist()
                else:
                    return ",".join(
                        _format_scif_symbolic_scalar(
                            i.item() if hasattr(i, "item") else i,
                            decimal_precision=6,
                        )
                        for i in direction_values
                    )

        nonzero_ints = [abs(v) for v in ints if v != 0]
        if nonzero_ints:
            divisor = nonzero_ints[0]
            for value in nonzero_ints[1:]:
                divisor = math.gcd(divisor, value)
            if divisor > 1:
                ints = [int(v / divisor) for v in ints]
        return ",".join(str(v) for v in ints)
    tags = _scif_spin_tag_names()
    rotation_axis_cartn_line = (
        f"\n{tags['rotation_axis_cartn']}  ."
        if tags["rotation_axis_cartn"] is not None
        else ""
    )
    if conf == 'Collinear':
        spin_only: str = (
            f"""{tags['collinear_direction']} '{_format_collinear_direction_for_scif(direction)}'\n"""
            + f"{tags['coplanar_perp_uvw']}   . \n{tags['rotation_axis']}  .{rotation_axis_cartn_line} \n{tags['rotation_angle']} ."
        )
    elif conf == 'Coplanar':
        spin_only :str = (
            f"{tags['collinear_direction']} .\n"
            + f"""{tags['coplanar_perp_uvw']}   '{','.join([_format_scif_symbolic_scalar(i.item() if hasattr(i, "item") else i, decimal_precision=6) for i in direction])}' """
            + f"\n{tags['rotation_axis']}  .{rotation_axis_cartn_line} \n{tags['rotation_angle']} ."
        )
    else:
        spin_only :str = (
            f"{tags['collinear_direction']} .\n"
            + f"{tags['coplanar_perp_uvw']}   . \n{tags['rotation_axis']}  .{rotation_axis_cartn_line} \n{tags['rotation_angle']} ."
        )
    return spin_only

def write_scif_lattice(
    lattice: tuple | list,
    *,
    cell_parameter_strings: dict | None = None,
    computed_precision: int = 6,
) -> str:
    if cell_parameter_strings is not None and all(
        cell_parameter_strings.get(key) is not None
        for key in [
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_cell_angle_alpha',
            '_cell_angle_beta',
            '_cell_angle_gamma',
        ]
    ):
        a = f"{'_cell_length_a':<20} {cell_parameter_strings['_cell_length_a']}"
        b = f"{'_cell_length_b':<20} {cell_parameter_strings['_cell_length_b']}"
        c = f"{'_cell_length_c':<20} {cell_parameter_strings['_cell_length_c']}"
        alpha = f"{'_cell_angle_alpha':<20} {cell_parameter_strings['_cell_angle_alpha']}"
        beta = f"{'_cell_angle_beta':<20} {cell_parameter_strings['_cell_angle_beta']}"
        gamma = f"{'_cell_angle_gamma':<20} {cell_parameter_strings['_cell_angle_gamma']}"
    else:
        a = f"{'_cell_length_a':<20} {lattice[0]:>18.{computed_precision}f}"
        b = f"{'_cell_length_b':<20} {lattice[1]:>18.{computed_precision}f}"
        c = f"{'_cell_length_c':<20} {lattice[2]:>18.{computed_precision}f}"
        alpha = f"{'_cell_angle_alpha':<20} {lattice[3]:>18.{computed_precision}f}"
        beta = f"{'_cell_angle_beta':<20} {lattice[4]:>18.{computed_precision}f}"
        gamma = f"{'_cell_angle_gamma':<20} {lattice[5]:>18.{computed_precision}f}"
    lattice_text = '\n'.join([a,b,c,alpha,beta,gamma])
    return lattice_text+'\n'


def _format_scif_matrix(matrix3x3, *, precision: int = 6) -> str:
    matrix = np.asarray(matrix3x3, dtype=float)
    rows = []
    for row in matrix:
        items = [
            _format_scif_symbolic_scalar(value, decimal_precision=precision) for value in row
        ]
        rows.append(f"[{','.join(items)}]")
    return f"[{','.join(rows)}]"

def affine_matrix_to_xyz_expression(
    matrix3x3,
    translation3x1=None,
    variables=('x', 'y', 'z'),
    *,
    separate_translation=False,
    coeff_precision: int = 6,
) -> str:
    """
    Convert affine matrix (3x3 + translation) to string like:
      - "x,y,z"                            (no translation)
      - "x+1/2,y+1/2,z"                    (embedded translation)
      - "x,y,z;1/2,1/2,0"                  (separate_translation=True)
    """

    # If no translation is given, use (u,v,w) and zero translation (your original logic)
    if translation3x1 is None:
        variables = ('u', 'v', 'w')
        translation3x1 = [0, 0, 0]

    result = []

    for row, t in zip(matrix3x3, translation3x1):
        terms = []
        for coeff, var in zip(row, variables):
            if abs(coeff) < 0.001:
                continue
            elif abs(coeff - 1) < 0.001:
                terms.append(f"{var}")
            elif abs(coeff + 1) < 0.001:
                terms.append(f"-{var}")
            else:
                coeff_str = _format_scif_symbolic_scalar(
                    coeff, decimal_precision=coeff_precision
                )
                terms.append(f"{coeff_str}{var}")

        # Only add translation into the expression if we are NOT separating it
        if (not separate_translation) and abs(t) > 1e-3:
            terms.append(str(Fraction(t).limit_denominator(100)))

        result.append('+'.join(terms).replace('+-', '-'))

    expr_part = ",".join(result)

    # If we don't want separate translation, keep original behaviour
    if not separate_translation:
        return expr_part

    # Build the ";a,b,c" translation part
    trans_terms = []
    for t in translation3x1:
        if abs(t) < 1e-3:
            trans_terms.append("0")
        else:
            trans_terms.append(str(Fraction(t).limit_denominator(100)))

    trans_part = ",".join(trans_terms)
    return f"{expr_part};{trans_part}"


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


def _format_basis_transform_rows(
    matrix: np.ndarray,
    symbols: tuple[str, str, str],
    *,
    tol: float = 1e-10,
) -> str:
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


def _format_repo_local_magnetic_phase(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(part.strip() for part in str(value).splitlines())


def _direct_transform_to_pp_string(
    matrix: np.ndarray,
    shift: np.ndarray,
    symbols: tuple[str, str, str],
    *,
    coeff_precision: int = 6,
) -> str:
    """
    Convert a direct current->target affine transform into crystallographic
    `Pp` basis-change notation.

    If the direct transform is:
        x_target = A x_current + o

    then the exported `Pp` notation should encode:
        x_current = P x_target + p

    with:
        P = A^{-1}
        p = -A^{-1} o
    """
    matrix = np.asarray(matrix, dtype=float)
    shift = np.asarray(shift, dtype=float)
    basis_rows = np.linalg.inv(matrix)
    origin_shift = normalize_vector_to_zero(-basis_rows @ shift, atol=1e-9)
    return affine_matrix_to_xyz_expression(
        basis_rows,
        origin_shift,
        symbols,
        separate_translation=True,
        coeff_precision=coeff_precision,
    )


def _chain_direct_setting_transform(
    first_matrix: np.ndarray,
    first_shift: np.ndarray,
    second_matrix: np.ndarray,
    second_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first_matrix = np.asarray(first_matrix, dtype=float)
    first_shift = np.asarray(first_shift, dtype=float)
    second_matrix = np.asarray(second_matrix, dtype=float)
    second_shift = np.asarray(second_shift, dtype=float)
    transform = second_matrix @ first_matrix
    shift = normalize_vector_to_zero(second_matrix @ first_shift + second_shift, atol=1e-10)
    return transform, shift


def _normalize_identify_space_transform(transform_value) -> tuple[np.ndarray, np.ndarray] | None:
    if transform_value is None:
        return None
    try:
        transform_array = np.asarray(transform_value, dtype=float)
    except (TypeError, ValueError):
        transform_array = None
    if transform_array is not None and transform_array.shape == (4, 4):
        return transform_array[1:, 1:], transform_array[1:, 0]
    if isinstance(transform_value, (list, tuple)) and len(transform_value) == 2:
        matrix = np.asarray(transform_value[0], dtype=float)
        shift = np.asarray(transform_value[1], dtype=float)
        if matrix.shape == (3, 3) and shift.shape == (3,):
            return matrix, shift
    return None


def _express_identify_space_transform_in_export_frame(
    identify_transform: tuple[np.ndarray, np.ndarray],
    *,
    basis_tag_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ssg_primitive: SpinSpaceGroup,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify-index space transforms are solved in the hidden `G0std_nofrac`
    operation frame. The emitted SCIF, however, is written in the current export
    frame. Re-express the identify transform in the emitted frame by conjugating
    through the export -> G0std -> nofrac coordinate transform.
    """
    identify_matrix = np.asarray(identify_transform[0], dtype=float)
    identify_shift = np.asarray(identify_transform[1], dtype=float)

    export_to_g0std = basis_tag_transforms.get("G0std")
    if export_to_g0std is None:
        export_to_g0std = (np.eye(3), np.zeros(3))
    export_to_g0std_matrix = np.asarray(export_to_g0std[0], dtype=float)
    export_to_g0std_shift = np.asarray(export_to_g0std[1], dtype=float)

    g0std_to_nofrac_matrix = (
        np.asarray(ssg_primitive.transformation_to_G0std_id, dtype=float)
        @ np.linalg.inv(np.asarray(ssg_primitive.transformation_to_G0std, dtype=float))
    )
    export_to_nofrac_matrix = g0std_to_nofrac_matrix @ export_to_g0std_matrix
    export_to_nofrac_shift = g0std_to_nofrac_matrix @ export_to_g0std_shift
    nofrac_to_export_matrix = np.linalg.inv(export_to_nofrac_matrix)

    export_matrix = nofrac_to_export_matrix @ identify_matrix @ export_to_nofrac_matrix
    export_shift = nofrac_to_export_matrix @ (
        identify_matrix @ export_to_nofrac_shift
        + identify_shift
        - export_to_nofrac_shift
    )
    return export_matrix, normalize_vector_to_zero(export_shift, atol=1e-10)


def _source_to_target_basis_pp_string(
    matrix: np.ndarray,
    shift: np.ndarray,
    symbols: tuple[str, str, str],
    *,
    coeff_precision: int = 6,
) -> str:
    """
    Convert a direct source->target fractional-coordinate transform into the
    external basis-relation form used by repo-local SCIF tags.

    If the direct transform is:
        p_target = A p_source + o

    then the basis relation to emit is:
        (a_source, b_source, c_source) = (a_target, b_target, c_target) A

    Since `affine_matrix_to_xyz_expression` formats row-wise linear forms, we
    pass `A.T` so that each emitted row corresponds to one source basis vector
    written in the target basis. The translation part is the source origin
    expressed in target fractional coordinates, i.e. `o`.
    """
    matrix = np.asarray(matrix, dtype=float)
    shift = normalize_vector_to_zero(np.asarray(shift, dtype=float), atol=1e-9)
    return affine_matrix_to_xyz_expression(
        matrix.T,
        shift,
        symbols,
        separate_translation=True,
        coeff_precision=coeff_precision,
    )


def _build_transform_chen_pp_abcs(
    source_name: str,
    cell_G0: CrystalCell,
    ssg: SpinSpaceGroup,
    basis_tag_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ssg_primitive: SpinSpaceGroup,
    identify_index_details: dict | None,
):
    if identify_index_details is None:
        return None

    transform_parts = _resolve_transform_chen_parts(
        cell_G0=cell_G0,
        ssg=ssg,
        basis_tag_transforms=basis_tag_transforms,
        ssg_primitive=ssg_primitive,
        identify_index_details=identify_index_details,
    )
    if transform_parts is None:
        return None
    space_matrix = transform_parts["space_matrix"]
    space_shift = transform_parts["space_shift"]
    spin_basis_rows = transform_parts["spin_basis_rows"]

    return (
        f"_space_group_spin.transform_Chen_Pp_abcs  "
        f"'{_direct_transform_to_pp_string(space_matrix, space_shift, ('a', 'b', 'c'), coeff_precision=6)};"
        f"{_format_basis_transform_rows(spin_basis_rows, ('as', 'bs', 'cs'))}'"
    )


def _resolve_transform_chen_parts(
    *,
    cell_G0: CrystalCell,
    ssg: SpinSpaceGroup,
    basis_tag_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ssg_primitive: SpinSpaceGroup,
    identify_index_details: dict | None,
):
    if identify_index_details is None:
        return None

    point_group_transformation = identify_index_details.get("point_group_transformation")
    space_group_transformation = _normalize_identify_space_transform(
        identify_index_details.get("space_group_transformation")
    )
    if point_group_transformation is None or space_group_transformation is None:
        return None

    current_to_chen_space = _express_identify_space_transform_in_export_frame(
        space_group_transformation,
        basis_tag_transforms=basis_tag_transforms,
        ssg_primitive=ssg_primitive,
    )

    space_matrix = np.asarray(current_to_chen_space[0], dtype=float)
    space_shift = np.asarray(current_to_chen_space[1], dtype=float)
    if abs(np.linalg.det(space_matrix)) < 1e-8:
        return None

    lattice_col = np.asarray(cell_G0.lattice_matrix, dtype=float).T
    if ssg.conf == "Collinear":
        spin_basis_rows = np.eye(3)
    else:
        standard_spin_ssg = ssg.transform_spin(lattice_col)
        spin_standardization = np.linalg.inv(
            np.asarray(standard_spin_ssg.n_spin_part_std_transformation, dtype=float)
        )
        identify_spin_map = np.asarray(point_group_transformation, dtype=float)
        spin_basis_rows = identify_spin_map @ spin_standardization @ lattice_col

    return {
        "space_matrix": space_matrix,
        "space_shift": space_shift,
        "spin_basis_rows": spin_basis_rows,
    }


def _build_collinear_chen_spin_basis_rows(
    *,
    lattice_col: np.ndarray,
    collinear_axis: np.ndarray,
) -> np.ndarray:
    axis = np.asarray(collinear_axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("Collinear Chen spin basis requires a non-zero collinear axis.")
    axis = axis / norm

    axis_cart = np.asarray(lattice_col, dtype=float) @ axis
    axis_cart_norm = np.linalg.norm(axis_cart)
    if axis_cart_norm < 1e-12:
        raise ValueError("Collinear Chen spin basis requires a non-zero cartesian axis.")
    e3 = axis_cart / axis_cart_norm

    reference_candidates = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    e1 = None
    for ref in reference_candidates:
        candidate = ref - np.dot(ref, e3) * e3
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm > 1e-8:
            e1 = candidate / candidate_norm
            break
    if e1 is None:
        raise ValueError("Failed to build deterministic perpendicular basis for collinear Chen frame.")
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)

    target_basis_cart = np.column_stack([e1, e2, e3])
    return np.linalg.inv(target_basis_cart) @ np.asarray(lattice_col, dtype=float)


def _transform_ssg_ops_to_chen_frame(
    ssg: SpinSpaceGroup,
    transform_parts: dict,
) -> SpinSpaceGroup:
    space_matrix = np.asarray(transform_parts["space_matrix"], dtype=float)
    space_shift = np.asarray(transform_parts["space_shift"], dtype=float)
    spin_basis_rows = np.asarray(transform_parts["spin_basis_rows"], dtype=float)
    space_matrix_inv = np.linalg.inv(space_matrix)
    spin_basis_rows_inv = np.linalg.inv(spin_basis_rows)

    transformed_ops = []
    for op in ssg.ops:
        spin_rotation = np.asarray(op[0], dtype=float)
        real_rotation = np.asarray(op[1], dtype=float)
        translation = np.asarray(op[2], dtype=float)
        new_real_rotation = space_matrix @ real_rotation @ space_matrix_inv
        new_spin_rotation = spin_basis_rows @ spin_rotation @ spin_basis_rows_inv
        new_translation = space_matrix @ translation + (np.eye(3) - new_real_rotation) @ space_shift
        transformed_ops.append(
            SpinSpaceGroupOperation(
                new_spin_rotation,
                new_real_rotation,
                new_translation,
            )
        )
    return SpinSpaceGroup(transformed_ops, tol=ssg.tol)


def _build_chen_linear_name(
    source_name: str,
    cell_G0: CrystalCell,
    ssg: SpinSpaceGroup,
    basis_tag_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ssg_primitive: SpinSpaceGroup,
    identify_index_details: dict | None,
) -> str | None:
    transform_parts = _resolve_transform_chen_parts(
        cell_G0=cell_G0,
        ssg=ssg,
        basis_tag_transforms=basis_tag_transforms,
        ssg_primitive=ssg_primitive,
        identify_index_details=identify_index_details,
    )
    if transform_parts is None:
        return None
    try:
        chen_ssg = _transform_ssg_ops_to_chen_frame(ssg, transform_parts)
        return chen_ssg.international_symbol_linear_current_frame
    except Exception as exc:  # pragma: no cover - defensive degrade path
        warnings.warn(
            f"Unable to build Chen/database-facing linear name for {source_name}: {exc}",
            RuntimeWarning,
        )
        return None


def _parse_solver_numeric_token(token: str) -> float:
    token = token.strip()
    if token in {"", "+"}:
        return 1.0
    if token == "-":
        return -1.0
    if "/" in token and token.count("/") == 1 and re.fullmatch(r"[+-]?\d+/\d+", token):
        return float(Fraction(token))
    return float(token)


def _parse_solver_component_expression(
    expr: str,
    *,
    parameter_symbols: tuple[str, ...] = ("Sx", "Sy", "Sz"),
    zero_tol: float = 1e-10,
) -> dict[str, float]:
    coefficients = {symbol: 0.0 for symbol in parameter_symbols}
    normalized = expr.replace(" ", "")
    if normalized in {"", "0"}:
        return coefficients

    for term in re.findall(r"[+-]?[^+-]+", normalized):
        matched_symbol = next((symbol for symbol in parameter_symbols if term.endswith(symbol)), None)
        if matched_symbol is None:
            continue
        coeff_token = term[: -len(matched_symbol)]
        if coeff_token.endswith("*"):
            coeff_token = coeff_token[:-1]
        coefficients[matched_symbol] += _parse_solver_numeric_token(coeff_token)

    for symbol, value in list(coefficients.items()):
        if abs(value) <= zero_tol:
            coefficients[symbol] = 0.0
    return coefficients


def _render_relative_solver_row(
    coeffs: np.ndarray,
    parameter_names: list[str],
    *,
    zero_tol: float = 1e-10,
) -> str:
    def _snap_solver_coeff(value: float) -> float:
        numeric = float(value)
        if abs(numeric) <= max(zero_tol, 1e-10):
            return 0.0
        if abs(numeric - 1.0) <= 1e-4:
            return 1.0
        if abs(numeric + 1.0) <= 1e-4:
            return -1.0
        return numeric

    tokens = []
    for coeff, parameter in zip(coeffs, parameter_names):
        coeff = _snap_solver_coeff(coeff)
        if abs(coeff) <= zero_tol:
            continue
        coeff_str = _format_scif_symbolic_scalar(coeff, decimal_precision=6, zero_tol=zero_tol)
        if coeff_str == "1":
            tokens.append(parameter)
        elif coeff_str == "-1":
            tokens.append(f"-{parameter}")
        else:
            tokens.append(f"{coeff_str}{parameter}")

    if not tokens:
        return "0"

    rendered = tokens[0]
    for token in tokens[1:]:
        rendered += token if token.startswith("-") else f"+{token}"
    return rendered


def _solver_constraints_to_matrix(
    constraints: list[str],
    *,
    zero_tol: float = 1e-10,
) -> np.ndarray:
    parameter_symbols = ("Sx", "Sy", "Sz")
    return np.array(
        [
            [_parse_solver_component_expression(expr, parameter_symbols=parameter_symbols, zero_tol=zero_tol)[symbol]
             for symbol in parameter_symbols]
            for expr in constraints
        ],
        dtype=float,
    )


def _solver_matrix_to_symmform(
    coefficient_matrix: np.ndarray,
    *,
    zero_tol: float = 1e-10,
) -> str:
    used_columns = [
        index for index in range(coefficient_matrix.shape[1])
        if np.any(np.abs(coefficient_matrix[:, index]) > zero_tol)
    ]
    if not used_columns:
        return "0,0,0"

    normalized_columns = []
    pivot_rows = []
    for column_index in used_columns:
        column = coefficient_matrix[:, column_index].copy()
        pivot_row = next(i for i, value in enumerate(column) if abs(value) > zero_tol)
        pivot_value = column[pivot_row]
        if pivot_value < 0:
            column *= -1
            pivot_value *= -1
        column /= pivot_value
        normalized_columns.append(column)
        pivot_rows.append(pivot_row)

    parameter_pool = ["u", "v", "w"]
    parameter_names = []
    used_parameter_names = set()
    for pivot_row in pivot_rows:
        candidate = parameter_pool[pivot_row]
        if candidate in used_parameter_names:
            candidate = next(name for name in parameter_pool if name not in used_parameter_names)
        parameter_names.append(candidate)
        used_parameter_names.add(candidate)

    normalized_matrix = np.column_stack(normalized_columns)
    rendered_rows = [
        _render_relative_solver_row(normalized_matrix[row_index], parameter_names, zero_tol=zero_tol)
        for row_index in range(3)
    ]
    return ",".join(rendered_rows)


def _solver_constraints_to_relative_symmform(
    constraints: list[str],
    *,
    zero_tol: float = 1e-10,
) -> str:
    """
    Convert solver-derived site-symmetry constraints into the raw lattice-basis
    `a,b,c` coefficient relation used by `symmform_rel_uvw`.
    """
    return _solver_matrix_to_symmform(
        _solver_constraints_to_matrix(constraints, zero_tol=zero_tol),
        zero_tol=zero_tol,
    )


def _solver_constraints_to_absolute_symmform(
    constraints: list[str],
    lattice_rows,
    *,
    zero_tol: float = 1e-10,
) -> str:
    """
    Convert solver-derived site-symmetry constraints into the normalized-basis
    relation used by `symmform_uvw`, i.e. in the
    `a/|a|, b/|b|, c/|c|` component frame.
    """
    coefficient_matrix = _solver_constraints_to_matrix(constraints, zero_tol=zero_tol)
    basis_lengths = np.linalg.norm(np.asarray(lattice_rows, dtype=float), axis=1)
    return _solver_matrix_to_symmform(
        np.diag(basis_lengths) @ coefficient_matrix,
        zero_tol=zero_tol,
    )

def trans_matrix_ssg_to_text(op:list, *, coeff_precision: int = 6)->str:
    if np.linalg.det(op[0]) > 0:
        eop = '+1'
    else:
        eop = '-1'

    left = f"{affine_matrix_to_xyz_expression(op[1], op[2], coeff_precision=coeff_precision)},{eop}"
    right = affine_matrix_to_xyz_expression(op[0], coeff_precision=coeff_precision)

    min_align = 30
    if len(left) >= min_align:
        align_width = len(left) + 5
    else:
        align_width = min_align

    text : str = f'{left:<{align_width}}{right:>{align_width}}'
    return text

def write_scif_nssg_no_center(non_centered_nssg_ops, *, coeff_precision: int = 6):
    nssg_text = "loop_\n_space_group_symop_spin_operation.id\n_space_group_symop_spin_operation.xyzt\n_space_group_symop_spin_operation.uvw\n"
    for index,op in enumerate(non_centered_nssg_ops):
        nssg_text = nssg_text + f'{index+1} '+trans_matrix_ssg_to_text(op, coeff_precision=coeff_precision) +'\n'
    return nssg_text

def write_scif_spin_translation(spin_translation_ops, *, coeff_precision: int = 6):
    nssg_text = "loop_\n_space_group_symop_spin_lattice.id\n_space_group_symop_spin_lattice.xyzt\n_space_group_symop_spin_lattice.uvw\n"
    for index,op in enumerate(spin_translation_ops):
        nssg_text = nssg_text + f'{index+1} '+trans_matrix_ssg_to_text(op, coeff_precision=coeff_precision) +'\n'
    return nssg_text


def _resolve_scif_operation_loops(ssg: SpinSpaceGroup):
    """
    Preferred export shape is:
    - main loop: one representative per NSTG coset (`ncnssg`)
    - spin-lattice loop: `n_spin_translation_group`

    In non-primitive settings this decomposition can fail because the current
    `mod Z^3` Seitz representation is no longer faithful enough for equal-size
    coset partitioning. In that case fall back to exporting the full `nssg`
    in the main loop and only the identity in the spin-lattice loop.
    """
    try:
        return list(ssg.ncnssg), list(ssg.n_spin_translation_group)
    except ValueError as exc:
        if "Wrong number of co-set" not in str(exc):
            raise
        return list(ssg.nssg), [SpinSpaceGroupOperation.identity()]

def write_scif_atoms(
    ssg_cell,
    occup_dict,
    atom_dict,
    eq_classes,
    eq_classes_spin,
    symmetry_constraints=None,
    *,
    position_precision: int = 8,
    moment_precision: int = SCIF_OPERATION_FULL_PRECISION,
    magnitude_precision: int = 3,
):
    def _absolute_symmform_from_moment(moment_components, *, zero_tol: float = 1e-10) -> str:
        """
        Express the representative moment in the normalized
        `a/|a|, b/|b|, c/|c|` component frame used by `axis_u/v/w`.
        """
        moment = np.asarray(moment_components, dtype=float)
        if np.linalg.norm(moment) <= zero_tol:
            return "0,0,0"

        parameter_names = ("u", "v", "w")
        pivot = next(index for index, value in enumerate(moment) if abs(value) > zero_tol)
        parameter = parameter_names[pivot]
        scale = moment[pivot]

        tokens = []
        for value in moment:
            coeff = value / scale
            if abs(coeff) <= zero_tol:
                tokens.append("0")
                continue

            coeff_str = _format_scif_symbolic_scalar(coeff, decimal_precision=6, zero_tol=zero_tol)
            if coeff_str == "1":
                tokens.append(parameter)
            elif coeff_str == "-1":
                tokens.append(f"-{parameter}")
            else:
                tokens.append(f"{coeff_str}{parameter}")

        return ",".join(tokens)

    def _relative_symmform_from_moment(moment_components, lattice_rows, *, zero_tol: float = 1e-10) -> str:
        """
        Express the representative moment as a relative ``u,v,w`` relation.

        Current formal `.scif` direction follows the Manu-style interpretation:
        ``axis_u/v/w`` are absolute components along the chosen spin basis,
        while ``symmform_uvw`` should describe the same moment in relative
        ``u,v,w`` coordinates. For the current mainline generator, the spin
        basis coincides with the lattice basis declared in the file.
        """
        moment = np.asarray(moment_components, dtype=float)
        basis_lengths = np.linalg.norm(np.asarray(lattice_rows, dtype=float), axis=1)
        relative = np.zeros(3, dtype=float)
        nonzero_basis = basis_lengths > zero_tol
        relative[nonzero_basis] = moment[nonzero_basis] / basis_lengths[nonzero_basis]

        if np.linalg.norm(relative) <= zero_tol:
            return "0,0,0"

        parameter_names = ("u", "v", "w")
        pivot = next(index for index, value in enumerate(relative) if abs(value) > zero_tol)
        parameter = parameter_names[pivot]
        scale = relative[pivot]

        tokens = []
        for value in relative:
            coeff = value / scale
            if abs(coeff) <= zero_tol:
                tokens.append("0")
                continue

            coeff_str = _format_scif_symbolic_scalar(coeff, decimal_precision=6, zero_tol=zero_tol)
            if coeff_str == "1":
                tokens.append(parameter)
            elif coeff_str == "-1":
                tokens.append(f"-{parameter}")
            else:
                tokens.append(f"{coeff_str}{parameter}")

        return ",".join(tokens)

    coords = np.array(ssg_cell[1])
    spins = np.array(ssg_cell[3])
    element_symbols = [atom_dict[i] for i in ssg_cell[2]]
    element_occupancies = [occup_dict[i] for i in ssg_cell[2]]
    all_site_symbols = []
    for eq in eq_classes:
        rep_idx = eq["representative_index"]
        symbol = element_symbols[rep_idx]
        if symbol not in all_site_symbols:
            all_site_symbols.append(symbol)
    output_lines = [
        "loop_",
        "_atom_type_symbol",
    ]
    output_lines.extend(all_site_symbols)
    output_lines.extend([
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_symmetry_multiplicity"
    ])
    element_counts = {}

    for eq in eq_classes:
        rep_idx = eq["representative_index"]
        symbol = element_symbols[rep_idx]
        element_counts[symbol] = element_counts.get(symbol, 0) + 1
        label = f"{symbol}{element_counts[symbol]}"
        x, y, z = [
            _stabilize_fractional_boundary_value(value)
            for value in coords[rep_idx]
        ]
        occupancy = element_occupancies[rep_idx]
        mult = len(eq["class_indices"])
        output_lines.append(
            f"{label}\t{symbol}\t{_format_scif_float(x, precision=position_precision)}\t"
            f"{_format_scif_float(y, precision=position_precision)}\t"
            f"{_format_scif_float(z, precision=position_precision)}\t{occupancy}\t{mult:.0f}"
        )



    output_lines.extend(['\n',
        "loop_",
        "_atom_site_spin_moment.label",
        "_atom_site_spin_moment.axis_u",
        "_atom_site_spin_moment.axis_v",
        "_atom_site_spin_moment.axis_w",
        "_atom_site_spin_moment.symmform_uvw",
        "_atom_site_spin_moment.symmform_rel_uvw",
        "_atom_site_spin_moment.magnitude"
    ])
    element_counts = {}
    constraint_map = {}
    if symmetry_constraints is not None:
        constraint_map = {
            eq["representative_index"]: constraint
            for eq, constraint in zip(eq_classes_spin, symmetry_constraints)
        }
    for eq in eq_classes_spin:
        rep_idx = eq["representative_index"]
        symbol = element_symbols[rep_idx]
        element_counts[symbol] = element_counts.get(symbol, 0) + 1
        label = f"{symbol}{element_counts[symbol]}"
        x, y, z = spins[rep_idx]
        symmform_uvw = (
            _solver_constraints_to_absolute_symmform(constraint_map[rep_idx], ssg_cell[0])
            if rep_idx in constraint_map
            else _absolute_symmform_from_moment(spins[rep_idx])
        )
        symmform_rel_uvw = (
            _solver_constraints_to_relative_symmform(constraint_map[rep_idx])
            if rep_idx in constraint_map
            else _relative_symmform_from_moment(spins[rep_idx], ssg_cell[0])
        )
        magnitude = np.linalg.norm(np.array([v/np.linalg.norm(v) for v in ssg_cell[0]]).T @ spins[rep_idx])
        output_lines.append(
            f"{label}\t{_format_scif_float(x, precision=moment_precision)}\t"
            f"{_format_scif_float(y, precision=moment_precision)}\t"
            f"{_format_scif_float(z, precision=moment_precision)}\t{symmform_uvw}\t"
            f"{symmform_rel_uvw}\t"
            f"{magnitude:.{magnitude_precision}f}"
        )

    return "\n".join(output_lines)



def generate_scif(
    filename,
    cell_G0:CrystalCell,
    ssg:SpinSpaceGroup,
    spin_wyckoff_positions,
    basis_tag_transforms,
    ssg_primitive:SpinSpaceGroup,
    *,
    spin_space_group_index: str | None = None,
    spin_space_group_name: str | None = None,
    spin_space_group_name_linear: str | None = None,
    spin_space_group_name_latex: str | None = None,
    magnetic_phase: str | None = None,
    identify_index_details: dict | None = None,
    coeff_precision: int = SCIF_OPERATION_FULL_PRECISION,
    position_precision: int = 8,
    moment_precision: int = SCIF_OPERATION_FULL_PRECISION,
    matrix_precision: int = SCIF_OPERATION_FULL_PRECISION,
    magnitude_precision: int = 3,
    source_cell_parameter_strings: dict | None = None,
    parent_space_group: dict | None = None,
    source_parent_space_group: dict | None = None,
    parent_space_group_comparison: dict | None = None,
):
    """
    input:
    1.relation between real space and spin space
    2.spin only part
    3.ssg number and symbol todo: wait for algorithm
    4.lattice
    5.ssg operations with spin operations in lattice
    6.spin translation operations
    7.atoms
    8.spins
    """
    index_to_occup = cell_G0.atom_types_to_occupancies
    index_to_element = cell_G0.atom_types_to_symbol
    cell = cell_G0.to_spglib(mag=True)
    configuration = ssg.conf
    norm_direction = ssg.sog_direction
    non_centered_nssg_ops, nontrivial_spin_translation_ops = _resolve_scif_operation_loops(ssg)



    head = "#\\#CIF_2.0\n#"+str(filename)    +     " \n# Created by FINDSPINGROUP " +f' version - {__version__}'+"\ndata_5yOhtAoR"

    tags = _scif_spin_tag_names()
    repo_tags = _scif_repo_local_extension_tag_names()
    transform_spinframe = "_space_group_spin.transform_spinframe_P_abc  'a,b,c'"
    # oriented

    spin_only = write_scif_spin_only(configuration, norm_direction)

    chen_number = spin_space_group_index
    chen_name = _build_chen_linear_name(
        filename,
        cell_G0,
        ssg,
        basis_tag_transforms,
        ssg_primitive,
        identify_index_details,
    )
    latex_name = spin_space_group_name_latex
    oriented_linear_name = spin_space_group_name_linear
    if chen_number is None:
        ssg_num = f"\n{tags['ssg_number']}  ."
        ssg_name = f"{tags['ssg_name']}     .\n"
    else:
        ssg_num = f"\n{tags['ssg_number']}  {_quote_scif_string(chen_number)}"
        if chen_name is None:
            ssg_name = f"{tags['ssg_name']}     .\n"
        else:
            ssg_name = f"{tags['ssg_name']}     {_quote_scif_string(chen_name)}\n"
    if oriented_linear_name is None:
        ssg_name_linear = f"{repo_tags['ssg_name_linear']}     ?"
    else:
        ssg_name_linear = f"{repo_tags['ssg_name_linear']}     {_quote_scif_string(oriented_linear_name)}"
    if latex_name is None:
        ssg_name_latex = f"{repo_tags['ssg_name_latex']}     ?"
    else:
        ssg_name_latex = f"{repo_tags['ssg_name_latex']}     {_quote_scif_string(latex_name)}"
    transform_chen_pp_abcs = _build_transform_chen_pp_abcs(
        filename,
        cell_G0,
        ssg,
        basis_tag_transforms,
        ssg_primitive,
        identify_index_details,
    )

    transform_to_input_Pp = (
        f"{repo_tags['transform_to_input_Pp']}  "
        f"'{_source_to_target_basis_pp_string(basis_tag_transforms['input'][0], basis_tag_transforms['input'][1], ('a','b','c'), coeff_precision=coeff_precision)}'"
    )
    transform_to_magnetic_primitive_Pp = (
        f"{repo_tags['transform_to_magnetic_primitive_Pp']}  "
        f"'{_source_to_target_basis_pp_string(basis_tag_transforms['magnetic_primitive'][0], basis_tag_transforms['magnetic_primitive'][1], ('a','b','c'), coeff_precision=coeff_precision)}'"
    )
    transform_to_magnetic_L0_Pp = (
        f"{repo_tags['transform_to_L0std_Pp']}  "
        f"'{_source_to_target_basis_pp_string(basis_tag_transforms['L0std'][0], basis_tag_transforms['L0std'][1], ('a','b','c'), coeff_precision=coeff_precision)}'"
    )
    transform_to_magnetic_G0_Pp = (
        f"{repo_tags['transform_to_G0std_Pp']}  "
        f"'{_source_to_target_basis_pp_string(basis_tag_transforms['G0std'][0], basis_tag_transforms['G0std'][1], ('a','b','c'), coeff_precision=coeff_precision)}'"
    )
    magnetic_phase_value = _format_repo_local_magnetic_phase(magnetic_phase)
    magnetic_phase_line = (
        f"{repo_tags['magnetic_phase']}  {_quote_scif_string(magnetic_phase_value)}"
        if magnetic_phase_value != ""
        else f"{repo_tags['magnetic_phase']}  \"\""
    )
    spin_acc_value = getattr(ssg, "acc", None)
    spin_acc_line = (
        f"{repo_tags['spin_arithmetic_crystal_class_symbol']}  {_quote_scif_string(spin_acc_value)}"
        if spin_acc_value is not None
        else f"{repo_tags['spin_arithmetic_crystal_class_symbol']}  ."
    )
    magnetic_acc_value = None
    try:
        magnetic_acc_value = SpinSpaceGroup(ssg.msg_ops).acc if ssg.msg_ops else None
    except Exception:
        magnetic_acc_value = None
    magnetic_acc_line = (
        f"{repo_tags['magnetic_arithmetic_crystal_class_symbol']}  {_quote_scif_string(magnetic_acc_value)}"
        if magnetic_acc_value is not None
        else f"{repo_tags['magnetic_arithmetic_crystal_class_symbol']}  ."
    )
    scif_core_spin_metadata = "\n".join(
        [
            ssg_num,
            ssg_name.rstrip(),
            transform_chen_pp_abcs
            if transform_chen_pp_abcs is not None
            else "_space_group_spin.transform_Chen_Pp_abcs  .",
        ]
    )
    scif_repo_local_extensions = "\n".join(
        [
            "# repo-local FINDSPINGROUP extensions",
            ssg_name_linear,
            ssg_name_latex,
            f"{repo_tags['G0_number']}  {int(ssg_primitive.G0_num)}",
            f"{repo_tags['L0_number']}  {int(ssg_primitive.L0_num)}",
            f"{repo_tags['it']}  {int(ssg_primitive.it)}",
            f"{repo_tags['ik']}  {int(ssg_primitive.ik)}",
            f"{repo_tags['spin_space_point_group_name']}  {_quote_scif_string(ssg_primitive.spin_part_point_group_symbol_hm)}",
            magnetic_phase_line,
            spin_acc_line,
            magnetic_acc_line,
            "",
            transform_to_input_Pp,
            transform_to_magnetic_primitive_Pp,
            transform_to_magnetic_L0_Pp,
            transform_to_magnetic_G0_Pp,
            "",
            (
                f"{repo_tags['input_parent_space_group_name']}  "
                f"{_quote_scif_string(parent_space_group_comparison['input_name_H_M_alt'])}"
                if parent_space_group_comparison is not None
                and parent_space_group_comparison.get('input_name_H_M_alt') is not None
                else f"{repo_tags['input_parent_space_group_name']}  ."
            ),
            (
                f"{repo_tags['input_parent_space_group_number']}  "
                f"{int(round(float(parent_space_group_comparison['input_IT_number'])))}"
                if parent_space_group_comparison is not None
                and parent_space_group_comparison.get('input_IT_number') is not None
                else f"{repo_tags['input_parent_space_group_number']}  ."
            ),
        ]
    )
    parent_space_group_lines = []
    if parent_space_group is not None:
        parent_name = parent_space_group.get("name_H_M_alt")
        parent_it = parent_space_group.get("IT_number")
        parent_transform = parent_space_group.get("transform_Pp_abc")
        child_transform = parent_space_group.get("child_transform_Pp_abc")
        if parent_name is not None:
            parent_space_group_lines.append(
                f"_parent_space_group.name_H-M_alt  {_quote_scif_string(parent_name)}"
            )
        if parent_it is not None:
            parent_space_group_lines.append(
                f"_parent_space_group.IT_number  {int(parent_it)}"
            )
        if parent_transform is not None:
            parent_space_group_lines.append(
                f"_parent_space_group.transform_Pp_abc  {_quote_scif_string(parent_transform)}"
            )
        if child_transform is not None:
            parent_space_group_lines.append(
                f"_parent_space_group.child_transform_Pp_abc  {_quote_scif_string(child_transform)}"
            )
    parent_space_group_block = "\n".join(parent_space_group_lines)

    lattice = write_scif_lattice(
        getprimitivelattice(cell[0]),
        cell_parameter_strings=source_cell_parameter_strings,
    )

    nssg_operations = write_scif_nssg_no_center(non_centered_nssg_ops, coeff_precision=coeff_precision)

    spin_translation = write_scif_spin_translation(nontrivial_spin_translation_ops, coeff_precision=coeff_precision)

    atoms_spins = write_scif_atoms(
        cell,
        index_to_occup,
        index_to_element,
        spin_wyckoff_positions[1],
        spin_wyckoff_positions[3],
        symmetry_constraints=spin_wyckoff_positions[4],
        position_precision=position_precision,
        moment_precision=moment_precision,
        magnitude_precision=magnitude_precision,
    )

    sections = [
        head,
        transform_spinframe,
        spin_only,
        scif_core_spin_metadata,
        scif_repo_local_extensions,
    ]
    if parent_space_group_block:
        sections.extend([parent_space_group_block, ""])
    sections.extend([
        lattice,
        nssg_operations,
        spin_translation,
        atoms_spins,
    ])

    scif: str = '\n'.join(sections)
    return scif
