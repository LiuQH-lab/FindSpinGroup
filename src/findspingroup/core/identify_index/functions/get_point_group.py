import numpy as np
import os
import sys
# Resolve the absolute path to the current file.
current_path = os.path.dirname(os.path.abspath(__file__))
# Resolve the parent directory.
parent_path = os.path.dirname(current_path)
# Add the parent directory to the import path.
sys.path.insert(0, parent_path)

from . import is_matrix_equal
from ..databases import get_point_group



def get_norm_matrices(point_group_num, *, use_222_contract=False):
    point_group = get_point_group(point_group_num, use_222_contract=use_222_contract)
    point_group_normal_matrices = [
        np.asarray(point_group['all_norm_matrices'][i], dtype=np.float32)
        for i in range(len(point_group['all_norm_matrices']))
    ]
    return point_group_normal_matrices

def find_map_num(map_list,point_group_num,tol = 0.001, *, use_222_contract=False):
    def find_ind(A,list):
        for i in range(len(list)):
            if is_matrix_equal(A,list[i],tol = tol):
                return i
                break
    point_group = get_point_group(point_group_num, use_222_contract=use_222_contract)
    point_group_matrices = [
        np.asarray(point_group['all_matrices'][i], dtype=np.float32)
        for i in range(len(point_group['all_matrices']))
    ]
    
    gen_num = []
    for i in range(len(map_list)):
        num = find_ind(map_list[i][0],point_group_matrices)
        gen_num.append(None if num is None else num + 1)
    all_gen_num = point_group['generator_numbers']
    map_num = None
    for i in range(len(all_gen_num)):
        if all_gen_num[i] == gen_num:
            map_num = i+1
            break

    if map_num is None:
        best_candidate = None
        for i, candidate_gen_num in enumerate(all_gen_num, start=1):
            overlap = sum(
                1
                for actual_num, candidate_num in zip(gen_num, candidate_gen_num)
                if actual_num is not None and actual_num == candidate_num
            )
            if overlap == 0:
                continue
            distance = 0.0
            for j, candidate_num in enumerate(candidate_gen_num):
                distance += float(
                    np.max(np.abs(map_list[j][0] - point_group_matrices[candidate_num - 1]))
                )
            score = (overlap, -distance, -i)
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, i)
        if best_candidate is None:
            raise ValueError(
                f"Cannot identify point-group map number for point_group={point_group_num}, "
                f"generator_numbers={gen_num}."
            )
        map_num = best_candidate[1]

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
    if not flag:
        raise ValueError(
            f"Cannot locate map-set head for point_group={point_group_num}, map_num={map_num}."
        )

    return {"mapnum": map_num,
            "head_map_num": head,
            "in_map_set_num": in_num
            }

# Manual smoke test.
if __name__ == "__main__":
    try:
        point_group_num = 14
        map_list = [[np.array([[-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]]),np.array([[ 1.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  1.]])], 
        [np.array([[ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]]),
        np.array([[ 1. ,  0. ,  0. ,  0. ],
        [ 0. ,  1. ,  0. ,  0. ],
        [ 0.5,  0. ,  1. ,  0. ],
        [ 0.5,  0. ,  0. , -1. ]])]]
        norm = get_norm_matrices(point_group_num)
        map_num = find_map_num(map_list,point_group_num)

        print(map_num)
        print(norm) 

    except Exception as ex:
        print(f"Error: {str(ex)}")


