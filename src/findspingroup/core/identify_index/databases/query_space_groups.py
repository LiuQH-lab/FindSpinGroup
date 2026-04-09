import sqlite3
import json
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Database path under the parent `databases` directory.
db_path = os.path.join(script_dir, '..', 'databases', 'space_groups.db')

def get_space_group(group_id):
    """Query a space-group record by id."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT generators, all_matrices FROM space_groups WHERE id = ?', (group_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return {
            'generators': json.loads(result[0]),
            'all_matrices': json.loads(result[1])
        }
        
    except sqlite3.OperationalError:
        print(f"Database file not found: {os.path.abspath(db_path)}")
        return None
    except Exception as e:
        print(f"Failed to query space group: {str(e)}")
        return None
