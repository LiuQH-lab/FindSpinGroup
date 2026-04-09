import sqlite3
import json
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Database path under the parent `databases` directory.
db_path = os.path.join(script_dir, '..', 'databases', 'point_groups.db')
db_path_222 = os.path.join(
    script_dir,
    '..',
    '..',
    'identify-py-222',
    'databases',
    'point_groups.db',
)

def _query_point_group(group_id, path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT all_matrices, all_norm_matrices,generator_numbers,map_sets '
        'FROM point_groups WHERE id = ?',
        (group_id,),
    )
    result = cursor.fetchone()
    conn.close()
    if not result:
        return None
    return {
        'id': group_id,
        'all_matrices': json.loads(result[0]),
        'all_norm_matrices': json.loads(result[1]),
        'generator_numbers': json.loads(result[2]),
        'map_sets': json.loads(result[3]),
    }

def get_point_group(group_id, *, use_222_contract=False):
    """Query a point-group record by id."""
    try:
        if use_222_contract and group_id == 14:
            result = _query_point_group(group_id, db_path_222)
            if result is not None:
                return result
        return _query_point_group(group_id, db_path)
    except sqlite3.OperationalError:
        print(f"Database file not found: {os.path.abspath(db_path)}")
        return None
    except Exception as e:
        print(f"Failed to query point group: {str(e)}")
        return None

# Manual smoke test.
if __name__ == "__main__":
    
    try:
        num = 14
        
        result = get_point_group(num)
        print(result)
        
    except Exception as ex:
        print(f"Error: {str(ex)}")

