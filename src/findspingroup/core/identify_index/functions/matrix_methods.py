import numpy as np

# Build a 4x4 affine matrix from the `[rotation, translation]` format.
def make_4d_matrix(matrix_list): 
    a = matrix_list[0]
    b = matrix_list[1]
    M = np.array([
        [1.0,   0.0,   0.0,   0.0],
        [b[0], a[0][0], a[0][1], a[0][2]],
        [b[1], a[1][0], a[1][1], a[1][2]],
        [b[2], a[2][0], a[2][1], a[2][2]]
    ], dtype=np.float32)
    return M

# Convert a map list `[A, B]` into matrix form.
def make_map_list(map_list): 
    map_matrix_list = []
    for item in map_list:
        A_matrix = np.asarray(item[0], dtype=np.float32)
        B_matrix = make_4d_matrix(item[1])
        
        map_matrix_list.append([A_matrix, B_matrix])
    return map_matrix_list

# Compare matrices with a default tolerance of 0.001.
def is_matrix_equal(A, B, tol=0.001):
    if A.shape != B.shape:
        return False
    if A.shape == B.shape:
        return np.max(np.abs(A - B)) <= tol

# Normalize the precision of a point-group matrix.
def adjust_point_matrix(A):
    arr = np.asarray(A, dtype=np.float32).copy()
    rounded = np.round(arr * 10000) / 10000
    return np.round(rounded, 8)

# Wrap the translation component modulo `m`.
def adjust_space_matrix(A, m, tol = 0.001):
    B = A.copy()
    for i in range(1, 4):
        B[i,0] = B[i,0] % m
        if abs(B[i,0]) < tol or abs(B[i,0] - m)< tol:
            B[i,0] = 0.0
    return B

# Normalize both the point and space parts of a map pair.
def adjust_map(pair, m):
    A = pair[0].copy()
    B = pair[1].copy()
    A_adj = adjust_point_matrix(A)
    B_adj = adjust_space_matrix(B, m)

    return [A_adj, B_adj]


def is_matrix_in(M,L,tol =0.001):
    is_in = False
    for B in L:
        if is_matrix_equal(M, B, tol=tol):
            is_in = True
            break
    return is_in

def is_matrices_in(Gen,L,tol = 0.001):
    is_in = True
    for A in Gen:
        if not is_matrix_in(A,L,tol = tol):
            is_in = False
            break
    return is_in

def map_transformation(map_txt_list,transformation_matrix):
    map_list = make_map_list(map_txt_list)
    T = np.asarray(transformation_matrix, dtype=np.float32)
    T_inv = np.linalg.inv(T)
    A = []
    for i in range(len(map_list)):
        map = map_list[i]
        new_map = [map[0], T @ map[1] @ T_inv]
        A.append(new_map)
    return A

def map_point_trans(map_txt_list,transformation_matrix,point_norm):
    map_list = make_map_list(map_txt_list)
    T = np.asarray(transformation_matrix, dtype=np.float32)
    T_inv = np.linalg.inv(T)
    T2 = np.asarray(point_norm, dtype=np.float32)
    T2_inv = np.linalg.inv(T2)
    A = []
    for i in range(len(map_list)):
        map = map_list[i]
        new_map = [T2 @ map[0] @ T2_inv, T @ map[1] @ T_inv]
        A.append(new_map)
    return A
