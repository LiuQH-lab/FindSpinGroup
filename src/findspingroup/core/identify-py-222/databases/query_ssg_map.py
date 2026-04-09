import sqlite3
import json
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Database path.
db_path = os.path.join(script_dir, '..', 'databases', 'ssg_map.db')

def find_ssg_map(L0_id, G0_id, it, ik, isonum):
    """Return all matching SSG map records."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query matching records.
        cursor.execute('''
        SELECT id, L0_id, G0_id, it, ik, num, isonum, 
               transformation_matrix, all_maps, transformation_maps, old_num,old_trans_1,old_trans_2 
        FROM ssg_map
        WHERE L0_id = ? AND G0_id = ? AND it = ? AND ik = ? AND isonum = ?
        ''', (L0_id, G0_id, it, ik,isonum))
        
        results = []
        for row in cursor.fetchall():
            # Parse the all_maps payload into a list.
            all_maps = row[8]
            if isinstance(all_maps, str):
                try:
                    map_list = json.loads(all_maps)
                except json.JSONDecodeError:
                    # Fall back to a Python-style list string.
                    map_list = eval(all_maps) if isinstance(all_maps, str) else all_maps
            else:
                map_list = all_maps

            
            # Return structured data.
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
                'old_num':json.loads(row[10]),
                'old_trans_1':json.loads(row[11]),
                'old_trans_2':json.loads(row[12])
            })
                
        return results
    except sqlite3.OperationalError:
        print(f"Error: database file not found: '{os.path.abspath(db_path)}'")
        return None
    except Exception as e:
        print(f"SSG map query failed: {str(e)}")
        return None



# Manual smoke test.
if __name__ == "__main__":

    print("=== SSG_map query tool ===")
    
    try:
        L0_id, G0_id, it, ik, iso = 1,16,4,1,16
        print(f"\nQuerying: L0_id={L0_id}, G0_id={G0_id}, it={it}, ik={ik}, isonum contains {iso}")
        
        results = find_ssg_map(L0_id, G0_id, it, ik, iso)

        print(f"\nFound {len(results)} matching records:")
        print(results)
        
    except ValueError:
        print("Error: all parameters must be integers")
    except Exception as ex:
        print(f"Error: {str(ex)}")
