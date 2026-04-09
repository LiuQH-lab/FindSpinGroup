import numpy as np
import os
import sys
# Resolve the current file path.
current_path = os.path.dirname(os.path.abspath(__file__))
# Resolve the parent directory.
parent_path = os.path.dirname(current_path)
# Add the parent directory to sys.path.
sys.path.insert(0, parent_path)

from functions import is_matrix_equal
from databases import get_point_group



def get_norm_matrices(point_group_num):
    point_group = get_point_group(point_group_num)
    point_group_normal_matrices = [np.matrix(point_group['all_norm_matrices'][i]) for i in range(len(point_group['all_norm_matrices']))]
    return point_group_normal_matrices

def find_map_num(map_list,point_group_num):
    def find_ind(A,list):
        for i in range(len(list)):
            if is_matrix_equal(A,list[i],tol = 0.001):
                return i
                break
    point_group = get_point_group(point_group_num)
    point_group_matrices = [np.matrix(point_group['all_matrices'][i]) for i in range(len(point_group['all_matrices']))]
    
    gen_num = []
    for i in range(len(map_list)):
        num = find_ind(map_list[i][0],point_group_matrices)
        gen_num.append(num +1)
    all_gen_num = point_group['generator_numbers']
    for i in range(len(all_gen_num)):
        if all_gen_num[i] == gen_num:
            map_num = i+1
            break
    
    point_group_mapsets = point_group['map_sets']
    flag = False
    for i in range(len(point_group_mapsets)):
        maps = point_group_mapsets[i]
        for j in range(len(maps)):
            if maps[j] == map_num:
                head = maps[0]
                in_num = j+1
                flag = True
                break
        if flag:
            break

    return {"mapnum": map_num,
            "head_map_num": head,
            "in_map_set_num": in_num
            }

# Manual smoke test.
if __name__ == "__main__":
    try:
        point_group_num = 14
        map_list = [[np.matrix([[-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]]),np.matrix([[ 1.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  1.]])], 
        [np.matrix([[ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]]),
        np.matrix([[ 1. ,  0. ,  0. ,  0. ],
        [ 0. ,  1. ,  0. ,  0. ],
        [ 0.5,  0. ,  1. ,  0. ],
        [ 0.5,  0. ,  0. , -1. ]])]]
        norm = get_norm_matrices(point_group_num)
        map_num = find_map_num(map_list,point_group_num)

        print(map_num)
        print(norm) 

    except Exception as ex:
        print(f"Error: {str(ex)}")


