from fractions import Fraction

import numpy as np
from findspingroup.version import __version__
from findspingroup.structure import CrystalCell, SpinSpaceGroup
from findspingroup.utils.matrix_utils import normalize_vector_to_zero


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

def write_scif_spin_only(conf, spin_only_direction):
    if spin_only_direction is not None:
        direction = []
        for i in spin_only_direction:
            if abs(i) < 1e-4:
                direction.append(0)
            else:
                direction.append(i)
    if conf == 'Collinear':
        spin_only: str = f"""_space_group_spin.collinear_direction_xyz '{','.join([f'{i.item() if hasattr(i, "item") else i:.3f}'.rstrip('0').rstrip('.') for i in direction])}'\n""" + \
                         "_space_group_spin.coplanar_perp_uvw   . \n_space_group_spin.rotation_axis  ? \n_space_group_spin.rotation_angle ?"
    elif conf == 'Coplanar':
        spin_only :str = "_space_group_spin.collinear_direction_xyz .\n" + \
                    f"""_space_group_spin.coplanar_perp_uvw   '{','.join([f'{i.item() if hasattr(i, "item") else i:.3f}'.rstrip('0').rstrip('.') for i in direction])}' """+"\n_space_group_spin.rotation_axis  ? \n_space_group_spin.rotation_angle ?"
    else:
        spin_only :str = "_space_group_spin.collinear_direction_xyz .\n" + \
                    "_space_group_spin.coplanar_perp_uvw   . \n_space_group_spin.rotation_axis  ? \n_space_group_spin.rotation_angle ?"
    return spin_only

def write_scif_lattice(lattice: tuple|list) -> str:
    a = f"{'_cell_length_a':<20} {lattice[0]:>10.3f}"
    b = f"{'_cell_length_b':<20} {lattice[1]:>10.3f}"
    c = f"{'_cell_length_c':<20} {lattice[2]:>10.3f}"
    alpha = f"{'_cell_angle_alpha':<20} {lattice[3]:>10.3f}"
    beta = f"{'_cell_angle_beta':<20} {lattice[4]:>10.3f}"
    gamma = f"{'_cell_angle_gamma':<20} {lattice[5]:>10.3f}"
    lattice_text = '\n'.join([a,b,c,alpha,beta,gamma])
    return lattice_text+'\n'

def affine_matrix_to_xyz_expression(
    matrix3x3,
    translation3x1=None,
    variables=('x', 'y', 'z'),
    *,
    separate_translation=False,
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
                # format coefficient, e.g. 0.500000 -> "0.5"
                coeff_str = f"{coeff:.6f}".rstrip('0').rstrip('.')
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

def trans_matrix_ssg_to_text(op:list)->str:
    if np.linalg.det(op[0]) > 0:
        eop = '+1'
    else:
        eop = '-1'

    left = f'{affine_matrix_to_xyz_expression(op[1], op[2])},{eop}'
    right = affine_matrix_to_xyz_expression(op[0])

    min_align = 30
    if len(left) >= min_align:
        align_width = len(left) + 5
    else:
        align_width = min_align

    text : str = f'{left:<{align_width}}{right:>{align_width}}'
    return text

def write_scif_nssg_no_center(non_centered_nssg_ops):
    nssg_text = "loop_\n_space_group_symop_spin_operation.id\n_space_group_symop_spin_operation.xyzt\n_space_group_symop_spin_operation.uvw\n"
    for index,op in enumerate(non_centered_nssg_ops):
        nssg_text = nssg_text + f'{index+1} '+trans_matrix_ssg_to_text(op) +'\n'
    return nssg_text

def write_scif_spin_translation(spin_translation_ops):
    nssg_text = "loop_\n_space_group_symop_spin_lattice.id\n_space_group_symop_spin_lattice.xyzt\n_space_group_symop_spin_lattice.uvw\n"
    for index,op in enumerate(spin_translation_ops):
        nssg_text = nssg_text + f'{index+1} '+trans_matrix_ssg_to_text(op) +'\n'
    return nssg_text

def write_scif_atoms(ssg_cell, occup_dict,atom_dict, eq_classes, eq_classes_spin, constraints):
    coords = np.array(ssg_cell[1])
    spins = np.array(ssg_cell[3])
    element_symbols = [atom_dict[i] for i in ssg_cell[2]]
    element_occupancies = [occup_dict[i] for i in ssg_cell[2]]
    output_lines = [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_symmetry_multiplicity"
    ]
    element_counts = {}

    for eq in eq_classes:
        rep_idx = eq["representative_index"]
        symbol = element_symbols[rep_idx]
        element_counts[symbol] = element_counts.get(symbol, 0) + 1
        label = f"{symbol}{element_counts[symbol]}"
        x, y, z = coords[rep_idx]
        occupancy = element_occupancies[rep_idx]
        mult = len(eq["class_indices"])
        output_lines.append(f"{label}\t{symbol}\t{x:.6f}\t{y:.6f}\t{z:.6f}\t{occupancy}\t{mult:.0f}")



    output_lines.extend(['\n',
        "loop_",
        "_atom_site_spin_moment.label",
        "_atom_site_spin_moment.axis_u",
        "_atom_site_spin_moment.axis_v",
        "_atom_site_spin_moment.axis_w",
        "_atom_site_spin_moment.symmform_uvw",
        "_atom_site_spin_moment.magnitude"
    ])
    element_counts = {}
    for i,eq in enumerate(eq_classes_spin):
        rep_idx = eq["representative_index"]
        symbol = element_symbols[rep_idx]
        element_counts[symbol] = element_counts.get(symbol, 0) + 1
        label = f"{symbol}{element_counts[symbol]}"
        x, y, z = spins[rep_idx]
        symmform = ','.join(constraints[i])
        magnitude = np.linalg.norm(np.array([v/np.linalg.norm(v) for v in ssg_cell[0]]).T @ spins[rep_idx])
        output_lines.append(f"{label}\t{x:.6f}\t{y:.6f}\t{z:.6f}\t{symmform}\t{magnitude:.3f}")





    return "\n".join(output_lines)



def generate_scif(filename,cell_G0:CrystalCell,ssg:SpinSpaceGroup,spin_wyckoff_positions,transformation,ssg_primitive:SpinSpaceGroup):
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
    non_centered_nssg_ops = ssg.ncnssg
    nontrivial_spin_translation_ops = ssg.n_spin_translation_group



    head = "#\\#CIF_2.0\n#"+str(filename)    +     " \n# Created by FINDSPINGROUP " +f' version - {__version__}'+"\ndata_5yOhtAoR"

    transform_spinframe_P_abc = "_space_group_spin.transform_spinframe_P_abc  'a,b,c'"
    # oriented

    spin_only = write_scif_spin_only(configuration, norm_direction)

    ssg_num = "\n_space_group_spin.number_SpSG_Chen  \"\""
    ssg_name = "_space_group_spin.name_SpSG_Chen     ?\n"

    symmetry_info = (f"_space_group_spin.G0_number  '{ssg_primitive.G0_num}'\n"+
                     f"_space_group_spin.L0_number  '{ssg_primitive.L0_num}'\n" +
                     f"_space_group_spin.it  '{ssg_primitive.it}'\n" +
                     f"_space_group_spin.ik  '{ssg_primitive.ik}'\n" +
                     f"_space_group_spin.spin_part_point_group  '{ssg_primitive.spin_part_point_group_symbol_hm}'\n"
                     )



    transform_to_input_Pp = f"_space_group_spin.transform_to_input_Pp  '{affine_matrix_to_xyz_expression(transformation[0].T,normalize_vector_to_zero(transformation[1],atol=1e-9),('a','b','c'),separate_translation=True)}'"
    transform_to_magnetic_primitive_Pp = f"_space_group_spin.transform_to_magnetic_primitive_Pp  '{affine_matrix_to_xyz_expression(np.linalg.inv(ssg.acc_primitive_trans).T,normalize_vector_to_zero(-np.linalg.inv(ssg.acc_primitive_trans)@ssg.acc_primitive_origin_shift,atol=1e-9),('a','b','c'),separate_translation=True)}'"
    transform_to_magnetic_L0_Pp = f"_space_group_spin.transform_to_L0std_Pp  '{affine_matrix_to_xyz_expression(np.linalg.inv(ssg.transformation_to_L0std).T,normalize_vector_to_zero(-np.linalg.inv(ssg.transformation_to_L0std)@ssg.origin_shift_to_L0std,atol=1e-9),('a','b','c'),separate_translation=True)}'"
    transform_to_magnetic_G0_Pp = f"_space_group_spin.transform_to_G0std_Pp  '{affine_matrix_to_xyz_expression(np.linalg.inv(ssg.transformation_to_G0std).T,normalize_vector_to_zero(-np.linalg.inv(ssg.transformation_to_G0std)@ssg.origin_shift_to_G0std,atol=1e-9),('a','b','c'),separate_translation=True)}'\n"
    # transform_to_magnetic_G0_Pp = f"_space_group_spin.transform_to_G0std_Pp  'a,b,c;0,0,0'\n"
    lattice = write_scif_lattice(getprimitivelattice(cell[0]))

    nssg_operations = write_scif_nssg_no_center(non_centered_nssg_ops)

    spin_translation = write_scif_spin_translation(nontrivial_spin_translation_ops)

    atoms_spins = write_scif_atoms(cell,index_to_occup,index_to_element,spin_wyckoff_positions[1],spin_wyckoff_positions[3],spin_wyckoff_positions[4])



    scif: str ='\n'.join([head,
                          transform_spinframe_P_abc,
                          spin_only,
                          ssg_num,
                          ssg_name,
                          symmetry_info,
                          transform_to_input_Pp,
                          transform_to_magnetic_primitive_Pp,
                          transform_to_magnetic_L0_Pp,
                          transform_to_magnetic_G0_Pp,
                          lattice,
                          nssg_operations,
                          spin_translation,
                          atoms_spins])
    return scif

