
import numpy as np
import re
from collections import deque

def rref_with_tolerance(A, tol=1e-3):
    A = A.astype(float)
    m, n = A.shape
    i = 0  # row
    for j in range(n):  # column
        # Find pivot
        max_row = np.argmax(np.abs(A[i:m, j])) + i
        if np.abs(A[max_row, j]) < tol:
            continue  # skip near-zero pivot
        # Swap rows
        A[[i, max_row]] = A[[max_row, i]]
        # Normalize pivot row
        A[i] = A[i] / A[i, j]
        # Eliminate other rows
        for k in range(m):
            if k != i:
                A[k] -= A[k, j] * A[i]
        i += 1
        if i == m:
            break
    return A

def normalize_vector_to_zero(v,atol=1e-10):
    new_v = []
    for i in v:
        if i %1 < atol or (-i) %1 < atol:
            new_v.append(0.0)
        else:
            new_v.append(i %1)
    return np.array(new_v)

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


def check_3x3_numeric_matrix(mat):
    # numpy array
    if isinstance(mat, np.ndarray):
        return mat.shape == (3, 3) and np.issubdtype(mat.dtype, np.number)

    # list or tuple
    if isinstance(mat, (list, tuple)):
        return (
                len(mat) == 3 and
                all(isinstance(row, (list, tuple)) and len(row) == 3 for row in mat) and
                all(isinstance(elem, (int, float)) for row in mat for elem in row)
        )

    return False






def parse_single_coords(coords_str: str):
    """
    parse a single coordinate expression like "x+1/2,-y+1/2,z,1"
    matrix: (3,3)
    shift:  (3,1)
    """
    # split by commas
    coords = [c.strip() for c in coords_str.replace("'",'').split(',')]
    if len(coords) not in (3, 4):
        raise ValueError(f"Invalid coordinate expression: {coords_str}")

    # time reversal flag
    t_flag = 1
    if len(coords) == 4:
        try:
            t_flag = int(coords[3])
            if t_flag not in (1, -1):
                t_flag = 1
        except ValueError:
            raise ValueError(f"Invalid coordinate expression: {coords_str}")

    # initialize matrix and shift
    matrix = np.zeros((3, 3), dtype=float)
    shift = np.zeros(3, dtype=float)

    vars = ['x', 'y', 'z']

    for i, expr in enumerate(coords[:3]):
        expr = expr.replace('−', '-')  # fix minus sign issue unicode
        # get coefficients for x, y, z
        for j, var in enumerate(vars):
            # look for patterns like 'x', '-x', '1/2x', '-1/2x', '0.5x', etc.
            pattern = rf'([+\-]?\d*(?:/\d+)?)\s*{var}'
            matches = re.findall(pattern, expr)
            coeff_sum = 0.0
            for m in matches:
                if m in ('', '+', '-'):
                    coeff_sum += 1.0 if m in ('', '+') else -1.0
                elif '/' in m:
                    num, den = m.split('/')
                    coeff_sum += float(num) / float(den)
                else:
                    coeff_sum += float(m)
            matrix[i, j] = coeff_sum

        # get constant term (shift)
        expr_no_vars = re.sub(r'[+\-]?\s*\d*(?:/\d+)?\s*[xyz]', '', expr)
        expr_no_vars = expr_no_vars.strip()
        if expr_no_vars:
            try:
                shift[i] = eval(expr_no_vars)
            except Exception:
                shift[i] = 0.0

    return matrix, shift, t_flag


def general_positions_to_matrix(general_positions: str|list):
    """
    analyse general positions
    format: '1 x,y,z,+1'
    :return:
        gp_matrices: [[matrix, shift], ...]
        timereversal: [±1, ...]
    """
    if isinstance(general_positions, str):
        lines = general_positions.strip().split('\n')
        lines = [re.split(r'\s+', line)[1] for line in lines]
    else:
        lines = general_positions
    gp_matrices = []
    timereversal = []

    for line in lines:
        # input should be like "x+1/2,-y+1/2,z,1"
        matrix, shift, t_flag = parse_single_coords(line)
        gp_matrices.append([matrix, shift])
        timereversal.append(t_flag)

    return gp_matrices, timereversal


def integerize_matrix(mat, max_mult=50, mod = 'row',constraint = None):
    """
    integerize each row of a 3x3 matrix
    mat: 2D list or np.array, shape=(3,3)
    max_mult: maximum multiplier to try
    """
    from fractions import Fraction
    import math
    def cal_lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    def lcm_of_three(a, b, c):
        lcm_ab = cal_lcm(a, b)
        return cal_lcm(lcm_ab, c)
    if mod == 'row':
        mat = np.array(mat, dtype=float)
    elif mod == 'col':
        mat = np.array(mat, dtype=float).T
    else:
        raise ValueError("mod should be 'row' or 'col'")

    result = np.zeros_like(mat, dtype=int)

    lcms = []
    for index,row in enumerate(mat):
        col1 = Fraction(row[0]).limit_denominator(max_mult)
        col2 = Fraction(row[1]).limit_denominator(max_mult)
        col3 = Fraction(row[2]).limit_denominator(max_mult)
        lcm = lcm_of_three(col1.denominator, col2.denominator, col3.denominator)
        lcms.append(lcm)
        result[index][0] = int(col1 * lcm)
        result[index][1] = int(col2 * lcm)
        result[index][2] = int(col3 * lcm)

    if constraint is None:
        pass
    elif constraint == 'a=b=c':
        final_lcm = lcm_of_three(lcms[0], lcms[1], lcms[2])
        for i in range(3):
            factor = final_lcm // lcms[i]
            result[i] = result[i] * factor
    elif constraint == 'a=b':
        final_lcm = lcm_of_three(lcms[0], lcms[1], 1)
        for i in range(2):
            factor = final_lcm // lcms[i]
            result[i] = result[i] * factor
    else:
        raise ValueError("Unsupported constraint type")

    if mod == 'row':
        return result
    elif mod == 'col':
        return result.T
    else:
        raise ValueError("mod should be 'row' or 'col'")



def mat_close(A, B, tol=1e-6):
    """Check if two matrices are equal within a tolerance."""
    return np.allclose(A, B, atol=tol)

def in_group(matrix, group):
    """Check if a matrix is already in the group (up to numerical tolerance)."""
    for g in group:
        if mat_close(matrix, g):
            return True
    return False

def in_space_group(op,group,tol = 1e-5):
    """Check if a matrix is already in the group (up to numerical tolerance)."""
    R1, t1 = op
    for g in group:
        R2, t2 = g
        if np.allclose(R1, R2,tol) and getNormInf(t1, t2)<tol:
            return True
    return False

def generate_point_group(generators):
    group = []
    queue = deque(generators)

    while queue:
        current = queue.popleft()
        if not in_group(current, group):
            group.append(current)
            for g in group:
                # 两种乘法顺序都尝试，构造闭包
                prod1 = np.dot(current, g)
                prod2 = np.dot(g, current)
                queue.append(prod1)
                queue.append(prod2)
    return group
