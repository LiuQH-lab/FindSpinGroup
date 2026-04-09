import numpy as np
import os
import sys
import itertools
# Resolve the current file path.
current_path = os.path.dirname(os.path.abspath(__file__))
# Resolve the parent directory.
parent_path = os.path.dirname(current_path)
# Add the parent directory to sys.path.
sys.path.insert(0, parent_path)

from functions import *

import copy
from collections import deque

# Given existing generator and translation maps, recover the maps for the
# standard generators listed in stand_gen_txt_list.
def find_stand_gen_maps(gen_maps_list,translation_maps_list,stand_gen_txt_list, m):
    gen_num = len(stand_gen_txt_list)
    stand_gen_matrices_list = [make_4d_matrix(p) for p in stand_gen_txt_list] 
    stand_gen_mod1_list = [adjust_space_matrix(p,1) for p in stand_gen_matrices_list]
    stand_map_mod1_list = [[] for i in range(0,gen_num)]
    found_mod1_num = 0
    # First find the size-1 generator correspondence.
    initial_gen_maps = copy.deepcopy(gen_maps_list)
    elements = [adjust_map(p, m) for p in initial_gen_maps]
    queue = deque(elements)
    adjusted_gen_elements = [adjust_space_matrix(p[1],1) for p in initial_gen_maps]
    while queue:
        cur_A, cur_B = queue.popleft()
        for i in range(0,gen_num):
            if is_matrix_equal(adjust_space_matrix(cur_B,1), stand_gen_mod1_list[i], tol=0.001):
                stand_map_mod1_list[i] = [cur_A, cur_B]
                found_mod1_num += 1
        if found_mod1_num == gen_num:
            break
        # Expand the current pair through the generator set.
        for elem_A, elem_B in elements:
            # Multiply the matrix pair.
            new_A = cur_A * elem_A
            new_B = cur_B * elem_B
            # Normalize precision before storing.
            new_pair = adjust_map([new_A,new_B],m)
            new_pair_space = adjust_space_matrix(new_pair[1],1)
            is_new = not is_matrix_in(new_pair_space,adjusted_gen_elements)
            if is_new:
                adjusted_gen_elements.append(new_pair_space)
                queue.append(new_pair)

    # Next expand from size 1 to size m using the translation-map basis.
    stand_gen_translation_list = [adjust_space_matrix(stand_gen_matrices_list[i] * np.matrix.getI(stand_map_mod1_list[i][1]),m) for i in range(0,gen_num)]
    stand_map_translation_list = [[] for p in stand_gen_translation_list]
    
    A = adjust_map(translation_maps_list[0],m)
    B = adjust_map(translation_maps_list[1],m)
    C = adjust_map(translation_maps_list[2],m)
    found_translation_num = 0
    for i,j,k in itertools.product(range(m), range(m), range(m)):
        if found_translation_num == gen_num:
            break
        space_translation_matrix = adjust_space_matrix(np.linalg.matrix_power(A[1],i)*np.linalg.matrix_power(B[1],j)*np.linalg.matrix_power(C[1],k),m)
        for z in range(0,gen_num):
            if is_matrix_equal(space_translation_matrix,stand_gen_translation_list[z], tol=0.001):
                point_translation_matrix = np.linalg.matrix_power(A[0],i)*np.linalg.matrix_power(B[0],j)*np.linalg.matrix_power(C[0],k)
                stand_map_translation_list[z] = [point_translation_matrix, space_translation_matrix]
                found_translation_num += 1
    # Combine the mod-1 and translation pieces into the final maps.
    
    stand_maps_list = []
    
    for i in range(0,gen_num):
        adjusted_space_group = adjust_space_matrix(stand_map_translation_list[i][1] * stand_map_mod1_list[i][1],m)
        adjusted_point_group = adjust_point_matrix(stand_map_translation_list[i][0] * stand_map_mod1_list[i][0])
        new_map = [adjusted_point_group,adjusted_space_group]
        stand_maps_list.append(new_map)
    
    return stand_maps_list

# Manual smoke test.
if __name__ == "__main__":
    try:
        # Prepare a sample query.
        L0_id, G0_id, ik, it, iso = 13,64,2,2,14
        
        gen_maps_list = [
            [[[1,0,0],[0,-1,0],[0,0,-1]],[[[1,-1,0],[1,0,0],[0,0,1]],[0,0,0.5]]],
            [[[-1,0,0],[0,-1,0],[0,0,-1]],[[[1,0,0],[0,1,0],[0,0,-1]],[0,0,0.5]]],
            [[[-1,0,0],[0,-1,0],[0,0,-1]],[[[-1,1,0],[0,1,0],[0,0,1]],[0,0,0]]],
            [[[-1,0,0],[0,-1,0],[0,0,-1]],[[[-1,0,0],[-1,1,0],[0,0,1]],[0,0,0.5]]]
        ]
        gen_maps_list = make_map_list(gen_maps_list)
        
        translation_maps_list = [
            [[[-0.5,-0.86621,0],[0.86621,-0.5,0],[0,0,1]],[[[1,0,0],[0,1,0],[0,0,1]],[1,0,0]]],
            [[[-0.5,-0.86621,0],[0.86621,-0.5,0],[0,0,1]],[[[1,0,0],[0,1,0],[0,0,1]],[0,1,0]]],
            [[[1,0,0],[0,1,0],[0,0,1]],[[[1,0,0],[0,1,0],[0,0,1]],[0,0,1]]]
        ]
        translation_maps_list = make_map_list(translation_maps_list)
        
        stand_gen_list = [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], [2.0, 0.0, 0.5]], [[[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.5]]]

        m = 3
        result = find_stand_gen_maps(gen_maps_list,translation_maps_list,stand_gen_list, m)
        print(f"\nFound: {result}")
        
    except Exception as ex:
        print(f"Error: {str(ex)}")
