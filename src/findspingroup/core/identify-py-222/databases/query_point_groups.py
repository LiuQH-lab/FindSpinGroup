import sqlite3
import json
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Database path in the parent databases directory.
db_path = os.path.join(script_dir, '..', 'databases', 'point_groups.db')

def get_point_group(group_id):
    """Query a point-group record by id."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT all_matrices, all_norm_matrices,generator_numbers,map_sets FROM point_groups WHERE id = ?', (group_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        # Return parsed JSON payloads.
        return {
            'id': group_id,
            'all_matrices': json.loads(result[0]),
            'all_norm_matrices': json.loads(result[1]),
            'generator_numbers': json.loads(result[2]),
            'map_sets': json.loads(result[3])
        }
        
    except sqlite3.OperationalError:
        print(f"Error: database file not found: '{os.path.abspath(db_path)}'")
        return None
    except Exception as e:
        print(f"Point-group query failed: {str(e)}")
        return None
    
# Manual smoke test.
if __name__ == "__main__":
    
    try:
        num = 14
        
        result = get_point_group(num)
        print(result)
        
    except Exception as ex:
        print(f"Error: {str(ex)}")

