import copy
import math
import numpy as np
import re
from ..structure import AtomicSite,CrystalCell
from ..structure.cell import are_positions_equivalent
from ..utils import general_positions_to_matrix
class CifParser:


    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}

    def parse(self):

        with open(self.filepath, 'rb') as f:
            raw = f.read()

        # 2. Try decoding as UTF-8, fallback to Latin-1
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

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

def parse_cif_file(filename, atol = 0.01):
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

    def get_first_existing(data: dict, keys: list[str]):
        for k in keys:
            if k in data:
                return data[k]
        return None

    data = CifParser(filename).parse()


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


    moment_labels = get_first_existing(data, label_keys)
    mx_list = get_first_existing(data, mx_keys)
    my_list = get_first_existing(data, my_keys)
    mz_list = get_first_existing(data, mz_keys)

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
        '_space_group_symop_operation.xyz'
    ]
    symops = get_first_existing(data, mag_op_keys)
    if symops is None:
        raise ValueError("CIF file missing symmetry operations.")

    mag_op_centering_keys = [
        '_space_group_symop_magn_centering.xyz',
        '_space_group_symop_magn_centering_xyz',
        '_space_group_symop.magn_centering_xyz',
    ]
    centering_ops = get_first_existing(data, mag_op_centering_keys)
    if centering_ops is None:
        raise ValueError("CIF file missing symmetry operations.")



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
    return latticefactors,all_positions, all_elements, all_occupancies, all_labels, all_moments

