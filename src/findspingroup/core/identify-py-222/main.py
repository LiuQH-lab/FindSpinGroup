import os
# Path configuration.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(ROOT_DIR, 'test_material')
JSON_PATH = os.path.join(TEST_PATH, 'all_222.json')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'output.txt')
from utils import load_test_data
from functions import *
from databases import *



def process_group_data():
    # Load the JSON payload.
    group_data = load_test_data(JSON_PATH)
    
    if group_data:
        return group_data
    else:
        print("Warning: JSON payload is empty")
        return None
    
if __name__ == "__main__":
    all_data = process_group_data()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in all_data:
            L0_id, G0_id, it, ik, iso,T = data['L0_id'], data['G0_id'], data['t_index'], data['k_index'], data['point_group_id'],  data['transformation_matrix']
            name_maps, translation_maps = data['name_maps'], data['translation_maps']
            map_num,trans1,trans2 = get_stand_trans(L0_id, G0_id, it, ik, iso,T,name_maps,translation_maps)
            line = f'[{L0_id},{G0_id}];[4,{it},{ik}];{data["map_id"]};{T};{map_num};{trans1};{trans2}****'
            f.write(line + '\n')
