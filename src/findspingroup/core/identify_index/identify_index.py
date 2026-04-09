import os
# Path setup for local identify-index debugging.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(ROOT_DIR, 'test_material')
JSON_PATH = os.path.join(TEST_PATH, 'conbs.json')
# from utils import load_test_data
# from findspingroup.core.identify_index.functions import *
# from findspingroup.core.identify_index.databases import *



# def process_group_data():
#     #
#     group_data = load_test_data(JSON_PATH)
#
#
#     if group_data:
#         data_entry = group_data
#
#         return data_entry
#     else:
#         print("json is empty")
#         return None
    
# if __name__ == "__main__":
#     data = process_group_data()
#     L0_id, G0_id, it, ik, iso,T = data['L0_id'], data['G0_id'], data['t_index'], data['k_index'], data['point_group_id'],  data['transformation_matrix']
#     name_maps, translation_maps = data['name_maps'], data['translation_maps']
#     map_num,trans1,trans2 = get_stand_trans(L0_id, G0_id, it, ik, iso,T,name_maps,translation_maps)
#     print(f'L0 = {L0_id}, G0 = {G0_id}, it = {it}, ik = {ik}, equivalent_map = {map_num}, point_group = {iso}')
#     print(f'space_group_transformation = {trans1}')
#     print(f'point_group_transformation = {trans2}')
