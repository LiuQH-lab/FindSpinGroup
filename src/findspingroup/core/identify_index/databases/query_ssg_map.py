import sqlite3
import json
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Database path.
db_path = os.path.join(script_dir, '..', 'databases', 'ssg_map.db')
db_path_222 = os.path.join(
    script_dir,
    '..',
    '..',
    'identify-py-222',
    'databases',
    'ssg_map.db',
)

def _query_ssg_map(db_path, L0_id, G0_id, it, ik, isonum):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id, L0_id, G0_id, it, ik, num, isonum, 
               transformation_matrix, all_maps, transformation_maps, old_num,old_trans_1,old_trans_2 
        FROM ssg_map
        WHERE L0_id = ? AND G0_id = ? AND it = ? AND ik = ? AND isonum = ?
        ''',
        (L0_id, G0_id, it, ik, isonum),
    )
    results = []
    for row in cursor.fetchall():
        all_maps = row[8]
        if isinstance(all_maps, str):
            try:
                map_list = json.loads(all_maps)
            except json.JSONDecodeError:
                map_list = eval(all_maps) if isinstance(all_maps, str) else all_maps
        else:
            map_list = all_maps
        results.append({
            'id': row[0],
            'L0_id': row[1],
            'G0_id': row[2],
            'it': row[3],
            'ik': row[4],
            'num': row[5],
            'isonum': row[6],
            'transformation_matrix': json.loads(row[7]),
            'all_maps': map_list,
            'transformation_maps': json.loads(row[9]),
            'old_num': json.loads(row[10]),
            'old_trans_1': json.loads(row[11]),
            'old_trans_2': json.loads(row[12]),
        })
    conn.close()
    return results

def find_ssg_map(L0_id, G0_id, it, ik, isonum, *, use_222_contract=False):
    """Find all matching SSG map records."""
    try:
        if use_222_contract and isonum == 14:
            results = _query_ssg_map(db_path_222, L0_id, G0_id, it, ik, isonum)
            if results:
                return results
        return _query_ssg_map(db_path, L0_id, G0_id, it, ik, isonum)
    except sqlite3.OperationalError:
        print(f"Database file not found: {os.path.abspath(db_path)}")
        return None
    except Exception as e:
        print(f"Failed to query SSG map: {str(e)}")
        return None


# Manual smoke test.
if __name__ == "__main__":

    print("=== SSG_map query tool ===")
    
    try:
        L0_id, G0_id, it, ik, iso = 1,16,4,1,16
        print(f"\nQuerying record: L0_id={L0_id}, G0_id={G0_id}, it={it}, ik={ik}, isonum contains {iso}")
        
        results = find_ssg_map(L0_id, G0_id, it, ik, iso)

        print(f"\nFound {len(results)} matching records:")
        print(results)
        
    except ValueError:
        print("Error: all parameters must be integers")
    except Exception as ex:
        print(f"Error: {str(ex)}")
