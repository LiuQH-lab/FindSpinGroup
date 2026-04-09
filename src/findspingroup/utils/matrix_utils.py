
import numpy as np
import re
import ast
import math
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
    if mode:
        a = np.mod(np.asarray(matrix1, dtype=float), 1.0)
        b = np.mod(np.asarray(matrix2, dtype=float), 1.0)
        diff = np.abs(a - b)
        wrapped = np.minimum(diff, 1.0 - diff)
        return float(np.max(wrapped))
    diff = np.abs(np.asarray(matrix1, dtype=float) - np.asarray(matrix2, dtype=float))
    return float(np.max(diff))


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






def evaluate_numeric_expression(expr: str) -> float:
    expr = expr.strip().replace('−', '-')
    if not expr:
        raise ValueError("Empty numeric expression")
    expr = re.sub(r'(\d|\))(?=sqrt\()', r'\1*', expr)

    node = ast.parse(expr, mode='eval')

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return math.sqrt(_eval(node.args[0]))
        raise ValueError(f"Unsupported numeric expression: {expr}")

    return float(_eval(node))


def _split_linear_terms(expr: str):
    return [term for term in re.findall(r'[+\-]?[^+\-]+', expr.replace(' ', '')) if term]


def parse_single_coords(coords_str: str, variables=('x', 'y', 'z')):
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

    for i, expr in enumerate(coords[:3]):
        expr = expr.replace('−', '-')  # fix minus sign issue unicode
        for term in _split_linear_terms(expr):
            matched_var = False
            for j, var in enumerate(variables):
                if term.endswith(var):
                    coeff_expr = term[: -len(var)]
                    if coeff_expr in ('', '+'):
                        coeff = 1.0
                    elif coeff_expr == '-':
                        coeff = -1.0
                    else:
                        coeff = evaluate_numeric_expression(coeff_expr)
                    matrix[i, j] += coeff
                    matched_var = True
                    break
            if not matched_var:
                shift[i] += evaluate_numeric_expression(term)

    return matrix, shift, t_flag


def general_positions_to_matrix(general_positions: str|list, variables=('x', 'y', 'z')):
    """
    analyse general positions
    format: '1 x,y,z,+1'
    :return:
        gp_matrices: [[matrix, shift], ...]
        timereversal: [±1, ...]
    """
    if isinstance(general_positions, str):
        lines = general_positions.strip().split('\n')
        lines = [re.split(r'\s+', line, maxsplit=1)[1] for line in lines]
    else:
        lines = general_positions
    gp_matrices = []
    timereversal = []

    for line in lines:
        # input should be like "x+1/2,-y+1/2,z,1"
        matrix, shift, t_flag = parse_single_coords(line, variables=variables)
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
                # Try both multiplication orders while building the closure.
                prod1 = np.dot(current, g)
                prod2 = np.dot(g, current)
                queue.append(prod1)
                queue.append(prod2)
    return group
