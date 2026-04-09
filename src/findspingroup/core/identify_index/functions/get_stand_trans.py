import numpy as np
from . import *
from ..databases import *
from .find_ssg_reduce import find_ssg_transformation

def get_stand_trans(
    L0_id,
    G0_id,
    it,
    ik,
    iso,
    T,
    name_maps,
    translation_maps,
    tol=0.001,
    *,
    use_222_contract=False,
    return_map_info=False,
):
    if iso == 0:
        # The caller assembles the full identify index. For the trivial
        # point-group branch there is only one equivalent map.
        return 1, np.identity(4), np.identity(3)
    TM = make_4d_matrix(T)

    ssg_ttm = find_ssg_transformation(
        L0_id,
        G0_id,
        it,
        ik,
        iso,
        TM,
        tol=tol,
        use_222_contract=use_222_contract,
    )
    if "transformation_matrix" not in ssg_ttm:
        raise ValueError(
            "No identify-index reduction record for "
            f"L0={L0_id}, G0={G0_id}, it={it}, ik={ik}, iso={iso}. "
            "Suggested direction: treat this as identify-index database/special-case coverage first; "
            "do not tune `space_tol`, `mtol`, `meigtol`, or `matrix_tol` first."
        )

    transformation_matrix = make_4d_matrix(ssg_ttm['transformation_matrix'])
    ssg_map_list = find_ssg_map(
        L0_id,
        G0_id,
        it,
        ik,
        iso,
        use_222_contract=use_222_contract,
    )
    
    number = 1
    while number <= len(ssg_map_list):
        map = ssg_map_list[number-1]
        if not is_matrix_equal(transformation_matrix, make_4d_matrix(map['transformation_matrix']), tol=tol):
            ssg_map_list.remove(map)
        else:
            number += 1
    cell_size = ssg_ttm['cell_size']
    
    TTM = np.linalg.inv(make_4d_matrix(ssg_ttm['TTM']))
    
    new_name_maps_matrices = map_transformation(name_maps,TTM)
    new_translation_maps_matrices = map_transformation(translation_maps,TTM)
    stand_map = find_stand_gen_maps(new_name_maps_matrices,new_translation_maps_matrices,ssg_ttm['gen_matrices'], cell_size)
    map_num = find_map_num(stand_map,iso,tol = tol, use_222_contract=use_222_contract)
    norm = get_norm_matrices(iso, use_222_contract=use_222_contract)
    
    has_found = False
    for map in ssg_map_list:
        for i in range(len(map['all_maps'])):
            if map['all_maps'][i][0] == map_num['head_map_num']:
                final_map = map
                tran_num = final_map['all_maps'][i][1]
                trans_map_matrix = make_4d_matrix(final_map['transformation_maps'][i])
                # print(final_map['old_trans_1'],final_map['old_trans_2'])
                trans_old_space_matrix = make_4d_matrix(final_map['old_trans_1'])
                trans_old_point_matrix = np.asarray(final_map['old_trans_2'], dtype=np.float32)
                old_map_num = final_map['old_num']
                has_found = True
                break
        if has_found:
            break
    final_transformation_matrix = np.linalg.inv(trans_old_space_matrix) @ np.linalg.inv(trans_map_matrix) @ TTM
    normal_matrix_1 = np.asarray(norm[map_num['in_map_set_num']-1], dtype=np.float32)
    normal_matrix_2 = np.asarray(norm[tran_num-1], dtype=np.float32)
    final_normal_matrix = np.linalg.inv(trans_old_point_matrix) @ np.linalg.inv(normal_matrix_2) @ normal_matrix_1
    if return_map_info:
        return old_map_num, final_transformation_matrix, final_normal_matrix, map_num
    return old_map_num,final_transformation_matrix,final_normal_matrix
