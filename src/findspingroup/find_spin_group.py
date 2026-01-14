import json
import re

import numpy as np
from spglib import get_symmetry_dataset,get_magnetic_symmetry_dataset

from findspingroup.core.identify_spin_space_group import identify_spin_space_group
from findspingroup.core.tolerances import DEFAULT_TOL, Tolerances
from findspingroup.data import MSGMPG_DB
from findspingroup.io import parse_cif_file
from findspingroup.io.scif_generator import generate_scif
from findspingroup.structure import SpinSpaceGroup,SpinSpaceGroupOperation
from findspingroup.structure.cell import CrystalCell
from findspingroup.data.PG_SYMBOL import PG_IF_HEX_MAPPING, SG_HALL_MAPPING
from findspingroup.utils.matrix_utils import rref_with_tolerance, normalize_vector_to_zero


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class MagSymmetryResult:
    def __init__(self, cell, symmetry, properties):

        self.primitive_magnetic_cell = cell['primitive_magnetic_cell']
        self.primitive_magnetic_cell_poscar = cell['primitive_magnetic_cell_poscar']
        self.scif = cell['scif']


        self.index = symmetry['index']
        self.conf = symmetry['configuration']
        self.magnetic_phase = symmetry['magnetic_phase']
        self.acc = symmetry['acc']
        self.KPOINTS = symmetry['KPOINTS']
        self.primitive_magnetic_cell_ssg_ops = symmetry['primitive_magnetic_cell_ssg_ops']
        self.spin_part_point_group = symmetry['full_spin_part_point_group']


        self.spinsplitting_w_soc = properties['ss_w_soc']
        self.spinsplitting_wo_soc = properties['ss_wo_soc']
        self.ahc_w_soc = properties['ahc_w_soc']
        self.ahc_wo_soc = properties['ahc_wo_soc']
        self.is_alter = properties['is_alter']

    def __repr__(self):
        return (f"<{self.__class__.__name__}>\n"
                f"  index: {self.index}\n"
                f"  conf : {self.conf}\n"
                f"  phase: {self.magnetic_phase}\n"
                f"  acc  : {self.acc}\n"
                f"  properties: {{\n"
                f"      ss_w_soc : {self.spinsplitting_w_soc},\n"
                f"      ss_wo_soc: {self.spinsplitting_wo_soc},\n"
                f"      ahc_w_soc: {self.ahc_w_soc},\n"
                f"      ahc_wo_soc: {self.ahc_wo_soc},\n"
                f"      is_alter : {self.is_alter}\n"
                f"  }}")


    def to_dict(self):
        return self.__dict__

    def save_json(self):
        return json.dumps(self.__dict__, indent=4,cls=NumpyEncoder)






def is_alter(condition, magphase, spinsplitting):
    if condition == 'Collinear' and (magphase == 'AFM' or magphase == 'Canting\\ AFM') and spinsplitting == 'k-dependent':
        return '(Altermagnet)'
    else:
        return ''

def spin_splitting_wo_soc(magnetic_phase, is_ss_gp):
    if magnetic_phase == 'AFM' or magnetic_phase == 'Canting\\ AFM':
        if is_ss_gp=="no spin splitting":
            return 'No'
        else:
            return 'k-dependent'
    else:
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

def get_magnetic_phase(full_spin_part_point_group, net_moment, mpg):
    if bool(re.match(r'^C\d+(?!h)', full_spin_part_point_group)) or bool(re.match(r'^Cs', full_spin_part_point_group)) or bool(re.match(r'^C_\{\\infty} v', full_spin_part_point_group)) or bool(re.match(r'^C\*v', full_spin_part_point_group)):
        if abs(net_moment) < 1e-4:
            return 'Compensated FiM'
        else:
            return 'FM/FiM'
    if mpg in MSGMPG_DB.FMMPG_INTlist:
        return 'Canting\\ AFM'
    else:
        return 'AFM'



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
        var_name = symbols[free_idx]
        coeffs = [0] * cols
        coeffs[var_col] = 1
        for row_idx in range(rows):
            row = A[row_idx]
            pivot_col = next((j for j in range(cols) if abs(row[j]) > tol), None)
            if pivot_col is not None and abs(row[var_col]) > tol:
                coeffs[pivot_col] = -row[var_col]


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
                    vector_expr[i] = f"{round(c, 3)}*{var_name}"
            else:
                if abs(c - 1) < tol:
                    vector_expr[i] += f" + {var_name}"
                elif abs(c + 1) < tol:
                    vector_expr[i] += f" - {var_name}"
                elif c > 0:
                    vector_expr[i] += f" + {round(c, 3)}*{var_name}"
                else:
                    vector_expr[i] += f" - {abs(round(c, 3))}*{var_name}"

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
            for j in range(num_atoms):
                dist = getNormInf(trans, coords[j])
                if dist < atol:
                    if j not in class_i and ssg_cell.atom_types[i] == ssg_cell.atom_types[j]:
                        class_i.append(j)
                        assigned[j] = True
                        # 判断是否是代表元自身的不动操作
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


def _identify_ssg_index(file_name,ssg_primitive:SpinSpaceGroup):
    """
    only for G0std_nofrac
    """
    from findspingroup.data.SG_SYMBOL import SGgeneratorDict
    def _match_ssg_generator(sg_num: int, ssg_std_ops_nofrac, tol=1e-3):
        """
        only for G0std_nofrac
        """
        sg_info = SGgeneratorDict[sg_num]

        generators = []
        for ind in range((len(sg_info) - 1) // 2):
            gen_rot, gen_t = eval(sg_info[2 * ind + 2])
            gen_t = np.array(gen_t, dtype=float)
            gen_rot = np.array(gen_rot).reshape((3, 3))
            generators.append([gen_rot, gen_t])
        generators_trans = [[np.eye(3), np.array([1, 0, 0])], [np.eye(3), np.array([0, 1, 0])],
                           [np.eye(3), np.array([0, 0, 1])]]  # add translations
        ssg_generators = []
        for gen_op in generators:
            found = False
            for op in ssg_std_ops_nofrac:
                if np.allclose(gen_op[0], op[1], atol=tol) and getNormInf(gen_op[1], op[2],mode=False) < tol:
                    found = True
                    ssg_generators.append(SpinSpaceGroupOperation(op[0], op[1], op[2]))
                    break
            if not found:
                raise ValueError(f"Cannot find generator {gen_op} in the provided operations.")
        for gen_op in generators_trans:
            found = False
            for op in ssg_std_ops_nofrac:
                if np.allclose(gen_op[0], op[1], atol=tol) and getNormInf(gen_op[1], op[2],mode=False) < tol:
                    found = True
                    ssg_generators.append(SpinSpaceGroupOperation(op[0], op[1], op[2]))
                    break
            if not found:
                ssg_generators.append(SpinSpaceGroupOperation(np.eye(3), np.eye(3), gen_op[1]))

        return ssg_generators


    if PG_IF_HEX_MAPPING.get(ssg_primitive.n_spin_part_point_group_symbol_s,0) == 1:
        spin_T  = ssg_primitive.spin_part_std_cartesian_transformation
    else:
        spin_T = np.eye(3)


    ssg_G0_std_nofrac = ssg_primitive.transform(ssg_primitive.transformation_to_G0std, ssg_primitive.origin_shift_to_G0std ).transform(
        ssg_primitive.transformation_to_G0std_id @ np.linalg.inv(ssg_primitive.transformation_to_G0std),
        np.array([0, 0, 0]), frac=False)
    ssg_G0_std_nofrac = ssg_G0_std_nofrac.transform_spin(np.linalg.inv(ssg_primitive.n_spin_part_std_transformation))
    G0_num = ssg_primitive.G0_num
    L0_num = ssg_primitive.L0_num
    it = ssg_primitive.it
    ik = ssg_primitive.ik
    pg = ssg_primitive.n_spin_part_point_group_symbol_hm
    generators = _match_ssg_generator(G0_num,ssg_G0_std_nofrac.nssg)
    generators_hm = [[(spin_T@ i[0]@np.linalg.inv(spin_T)).round(6).tolist(),i[1].round(6).tolist(),i[2].round(6).tolist()] for i in generators[:-3]]
    generators_lattice =[[(spin_T@ i[0]@np.linalg.inv(spin_T)).round(6).tolist(),i[1].round(6).tolist(),i[2].round(6).tolist()] for i in  generators[-3:]]
    transformation_G0std_to_L0std = [ssg_primitive.G0std_L0std_transformation.round(6).tolist(),ssg_primitive.G0std_L0std_origin_shift.round(6).tolist()]

    return f"{file_name}\t{G0_num}\t{L0_num}\t{it}\t{ik}\t{pg}\t{generators_hm}\t{generators_lattice}\t{transformation_G0std_to_L0std}"





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
            generatePosition.append(temp[0][1])
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



def find_spin_groups(cif: str, space_tol = 0.02, mtol = 0.02, meigtol = 0.00002) -> MagSymmetryResult:
    """
    Find the spin space group of a crystal structure given in a CIF file.

    Parameters:
    cif (str): Path to the CIF file.
    space_tol (float): Tolerance for space group determination.
    mtol (float): Tolerance for magnetic moment determination.
    meigtol (float): Tolerance for eigenvalue determination.

    Returns:
    dict: A dictionary containing the spin space group information and related data.
    """

    DEFAULT_TOL=Tolerances(space_tol, mtol, meigtol)


    lattice_factors,positions, elements, occupancies, labels, moments = parse_cif_file(cif)
    magnetic_primitive_cell: CrystalCell
    magnetic_primitive_cell,Tmatrix_Tp_input__p_primitive = CrystalCell(lattice_factors, positions,occupancies, elements,  moments,spin_setting="in_lattice").get_primitive_structure(magnetic=True)
    ssg_primitive :SpinSpaceGroup = identify_spin_space_group(magnetic_primitive_cell,find_primitive=False)

    try:
        msg_dataset_primitive = get_magnetic_symmetry_dataset(magnetic_primitive_cell.to_spglib(mag=True), symprec=space_tol,mag_symprec=DEFAULT_TOL.moment)
        p_to_msg_transformation = msg_dataset_primitive.transformation_matrix
        p_to_msg_originshift = msg_dataset_primitive.origin_shift
        msg_num = msg_dataset_primitive.uni_number
        msg_type = msg_dataset_primitive.msg_type
        mpg_symbol = MSGMPG_DB.OG_NUM_TO_MPG[MSGMPG_DB.BNS_TO_OG_NUM[MSGMPG_DB.MSG_INT_TO_BNS[msg_num][0]]]["pointgroup_no"]
    except Exception as e:
        mpg_symbol = None
        msg_num = None
        msg_type = None
        mpg_symbol =None
        p_to_msg_transformation = None
        p_to_msg_originshift = None


    magnetic_phase = get_magnetic_phase(ssg_primitive.spin_part_point_group_symbol_s,magnetic_primitive_cell.net_moment,mpg_symbol)
    ss_w_soc = spin_splitting_w_soc(ssg_primitive)
    ahc_w_soc = is_ahc(mpg_symbol)
    ss_wo_soc = spin_splitting_wo_soc(magnetic_phase,ssg_primitive.is_spinsplitting[-1])
    ahc_wo_soc = is_ahc(ssg_primitive.gspg.empg_symbol)
    alter = is_alter(ssg_primitive.conf, magnetic_phase, ss_wo_soc)
    magnetic_phase = magnetic_phase+alter


    transformation_G0_to_input = (ssg_primitive.transformation_to_G0std @ Tmatrix_Tp_input__p_primitive,ssg_primitive.origin_shift_to_G0std)


    identify_info =_identify_ssg_index(cif,ssg_primitive)



    G0std_cell:CrystalCell = magnetic_primitive_cell.transform(ssg_primitive.transformation_to_G0std,ssg_primitive.origin_shift_to_G0std)
    acc_magnetic_primitive_cell:CrystalCell = magnetic_primitive_cell.transform(ssg_primitive.acc_primitive_trans,ssg_primitive.acc_primitive_origin_shift)
    acc_p_c_poscar = acc_magnetic_primitive_cell.to_poscar(cif)
    G0std_ssg:SpinSpaceGroup = ssg_primitive.transform(ssg_primitive.transformation_to_G0std,ssg_primitive.origin_shift_to_G0std)
    acc_magnetic_primitive_ssg:SpinSpaceGroup = ssg_primitive.transform(ssg_primitive.acc_primitive_trans,ssg_primitive.acc_primitive_origin_shift)
    KPOINTS = acc_magnetic_primitive_ssg.KPOINTS

    G0std_cell_in_lattice :CrystalCell = G0std_cell.transform_spin(np.linalg.inv(np.array([v / np.linalg.norm(v) for v in G0std_cell.lattice_matrix]).T),'in_lattice')
    G0std_ssg_in_lattice :SpinSpaceGroup = G0std_ssg.transform_spin( np.linalg.inv(np.array([v / np.linalg.norm(v) for v in G0std_cell.lattice_matrix]).T))

    G0_wyckoff = get_spin_wyckoff(G0std_cell_in_lattice,G0std_ssg_in_lattice.ops)


    scif = generate_scif(cif,G0std_cell_in_lattice,G0std_ssg_in_lattice,G0_wyckoff,transformation_G0_to_input,ssg_primitive)


    result = {
        'index':ssg_primitive.index,
        'spin_part_pg':ssg_primitive.spin_part_point_group_symbol_hm,
        'conf':ssg_primitive.conf,
        'id_index_info':identify_info,
        'scif':scif,
        'poscar_mp':acc_p_c_poscar,
        'acc':ssg_primitive.acc,
        'KPOINTS':KPOINTS
    }

    cell = {'primitive_magnetic_cell':acc_magnetic_primitive_cell.to_spglib(mag=True),'primitive_magnetic_cell_poscar':acc_p_c_poscar,'scif':scif}
    symmetry = {'index':ssg_primitive.index,
                'configuration':ssg_primitive.conf,
                'magnetic_phase':magnetic_phase,
                'acc':ssg_primitive.acc,
                'KPOINTS':KPOINTS,
                'primitive_magnetic_cell_ssg_ops':acc_magnetic_primitive_ssg.ops,
                'full_spin_part_point_group':ssg_primitive.spin_part_point_group_symbol_hm,

                'msg_symbol':mpg_symbol}
    properties = {'ss_w_soc':ss_w_soc,'ss_wo_soc':ss_wo_soc,'ahc_w_soc':ahc_w_soc,'ahc_wo_soc':ahc_wo_soc,'is_alter':alter}

    return MagSymmetryResult(cell,symmetry,properties)