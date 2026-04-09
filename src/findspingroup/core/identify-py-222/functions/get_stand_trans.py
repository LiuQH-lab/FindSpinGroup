import numpy as np
from functions import *
from databases import *

def get_stand_trans(L0_id, G0_id, it, ik, iso,T,name_maps,translation_maps):
    TM = make_4d_matrix(T)
    ssg_ttm = find_ssg_transformation(L0_id, G0_id, it, ik, iso, TM)
    transformation_matrix = make_4d_matrix(ssg_ttm['transformation_matrix'])
    ssg_map_list = find_ssg_map(L0_id, G0_id, it, ik, iso)
    
    number = 1
    while number <= len(ssg_map_list):
        map = ssg_map_list[number-1]
        if not is_matrix_equal(transformation_matrix, make_4d_matrix(map['transformation_matrix']), tol=0.001):
            ssg_map_list.remove(map)
        else:
            number += 1
    cell_size = ssg_ttm['cell_size']
    
    TTM = np.matrix.getI(make_4d_matrix(ssg_ttm['TTM']))
    
    new_name_maps_matrices = map_transformation(name_maps,TTM)
    new_translation_maps_matrices = map_transformation(translation_maps,TTM)
    stand_map = find_stand_gen_maps(new_name_maps_matrices,new_translation_maps_matrices,ssg_ttm['gen_matrices'], cell_size)
    map_num = find_map_num(stand_map,iso)
    norm = get_norm_matrices(iso)
    
    has_found = False
    for map in ssg_map_list:
        for i in range(len(map['all_maps'])):
            if map['all_maps'][i][0] == map_num['head_map_num']:
                final_map = map
                tran_num = final_map['all_maps'][i][1]
                trans_map_matrix = make_4d_matrix(final_map['transformation_maps'][i])
                # print(final_map['old_trans_1'],final_map['old_trans_2'])
                trans_old_space_matrix = make_4d_matrix(final_map['old_trans_1'])
                trans_old_point_matrix = np.matrix(final_map['old_trans_2'])
                old_map_num = final_map['old_num']
                has_found = True
                break
        if has_found:
            break
    final_transformation_matrix = np.matrix.getI(trans_old_space_matrix)*np.matrix.getI(trans_map_matrix)*TTM
    normal_matrix_1 = np.matrix(norm[map_num['in_map_set_num']-1])
    normal_matrix_2 = np.matrix(norm[tran_num-1])
    final_normal_matrix = np.matrix.getI(trans_old_point_matrix)*np.matrix.getI(normal_matrix_2)*(normal_matrix_1)
    return old_map_num,final_transformation_matrix,final_normal_matrix