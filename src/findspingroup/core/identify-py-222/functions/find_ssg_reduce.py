import os
import sys
import numpy as np
# Resolve the current file path.
current_path = os.path.dirname(os.path.abspath(__file__))
# Resolve the parent directory.
parent_path = os.path.dirname(current_path)
# Add the parent directory to sys.path.
sys.path.insert(0, parent_path)

from databases import find_ssg_reduce,get_space_group
from functions import *

def find_ssg_transformation(L0_id, G0_id, it, ik, iso,TM):
    result = {}
    results = find_ssg_reduce(L0_id, G0_id, it, ik, iso)
    L0 = get_space_group(L0_id)

    # Build the full L0 matrix set.
    E0 = [[[[1,0,0],[0,1,0],[0,0,1]],[1,0,0]],[[[1,0,0],[0,1,0],[0,0,1]],[0,1,0]],[[[1,0,0],[0,1,0],[0,0,1]],[0,0,1]],[[[1,0,0],[0,1,0],[0,0,1]],[1,1,0]],[[[1,0,0],[0,1,0],[0,0,1]],[1,0,1]],[[[1,0,0],[0,1,0],[0,0,1]],[0,1,1]],[[[1,0,0],[0,1,0],[0,0,1]],[1,1,1]]]
    All = [make_4d_matrix(p) for p in E0]
    for ma in L0['all_matrices']:
        All.append(make_4d_matrix(ma))
    # Build the L0 generator set.
    E = [[[[1,0,0],[0,1,0],[0,0,1]],[1,0,0]],[[[1,0,0],[0,1,0],[0,0,1]],[0,1,0]],[[[1,0,0],[0,1,0],[0,0,1]],[0,0,1]]]
    Gen = []
    for gen in L0['generators'] + E:
        Gen.append(make_4d_matrix(gen))
    
    # Each item is a candidate SSG record.
    for item in results:
        flag = False
        if flag:
            break
        # Validate whether the item belongs to the same L0 setting.
        for ttm in item['TTM']:
            if flag:
                break
            newTM = np.matrix.getI(make_4d_matrix(ttm[0]))*(TM)
            newgen = []
            for gen in Gen:
                newgen.append(adjust_space_matrix(newTM*gen*np.matrix.getI(newTM),1))
            if is_matrices_in(newgen,All):
                result = {"indices": [L0_id,G0_id,it,ik],
                "cell_size": item['cell_size'],
                "gen_matrices": item['gen_matrix'],
                "transformation_matrix": item['transformation_matrix'],
                "TTM": ttm[1]}
                flag = True
                break 
    return result

# Manual smoke test.
if __name__ == "__main__":
    # print("=== SSG relation query tool ===")
    
    try:
        # Prepare a sample query.
        L0_id, G0_id, it, ik, iso = 5,23,2,1,1

        T = [[[-1,0,0],[-1,0,1],[0,1,0]],[0,0,0]]
        TM = make_4d_matrix(T)
        # Run the query.
        result = find_ssg_transformation(L0_id, G0_id, it, ik, iso,TM)
        print(f"\nFound: {result}")
        
    except Exception as ex:
        print(f"Error: {str(ex)}")
