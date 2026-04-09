import numpy as np

# Build a 4x4 affine matrix from a rotation-translation pair.
def make_4d_matrix(matrix_list): 
    a = matrix_list[0]
    b = matrix_list[1]
    M = np.matrix([
        [1.0,   0.0,   0.0,   0.0],
        [b[0], a[0][0], a[0][1], a[0][2]],
        [b[1], a[1][0], a[1][1], a[1][2]],
        [b[2], a[2][0], a[2][1], a[2][2]]
    ], dtype=np.float32)
    return M

# Convert a list of map payloads into matrix pairs [A, B].
def make_map_list(map_list): 
    map_matrix_list = []
    for item in map_list:
        A_matrix = np.matrix(item[0])
        B_matrix = make_4d_matrix(item[1])
        
        map_matrix_list.append([A_matrix, B_matrix])
    return map_matrix_list

# Compare two matrices under the default tolerance.
def is_matrix_equal(A, B, tol=0.001):
    if A.shape != B.shape:
        return False
    if A.shape == B.shape:
        return np.max(np.abs(A - B)) <= tol

# Round a point-group matrix to the expected precision.
def adjust_point_matrix(A):
    A_adj = []
    for row in A:
        new_row = []
        for i in range(0,3):
            rounded_val = round(row[0,i] * 10000) / 10000
            new_row.append(round(rounded_val, 8))
        A_adj.append(new_row)
    return np.matrix(A)
    
# Reduce space-group translations modulo m.
def adjust_space_matrix(A, m):
    B = A.copy()
    for i in range(1, 4):
        B[i,0] = B[i,0] % m
        if abs(B[i,0]) < 1e-4 or abs(B[i,0] - m)< 1e-4:
            B[i,0] = 0.0
    return B

# Normalize both point and space parts of a map pair.
def adjust_map(pair, m):
    A = pair[0].copy()
    B = pair[1].copy()
    A_adj = adjust_point_matrix(A)
    B_adj = adjust_space_matrix(B, m)
    
    return [A_adj, B_adj]


def is_matrix_in(M,L):
    is_in = False 
    for B in L:
        if is_matrix_equal(M, B, tol=0.001):
            is_in = True
            break
    return is_in
    
def is_matrices_in(Gen,L):
    is_in = True
    for A in Gen:
        if not is_matrix_in(A,L):
            is_in = False
            break
    return is_in

def map_transformation(map_txt_list,transformation_matrix):
    map_list = make_map_list(map_txt_list)
    T = transformation_matrix
    A = []
    for i in range(len(map_list)):
        map = map_list[i]
        new_map = [map[0],(T*map[1]) *np.matrix.getI(T) ]
        A.append(new_map)
    return A

def map_point_trans(map_txt_list,transformation_matrix,point_norm):
    map_list = make_map_list(map_txt_list)
    T = transformation_matrix
    T2 = point_norm
    A = []
    for i in range(len(map_list)):
        map = map_list[i]
        new_map = [(T2*map[0])*np.matrix.getI(T2),(T*map[1]) *np.matrix.getI(T) ]
        A.append(new_map)
    return A
