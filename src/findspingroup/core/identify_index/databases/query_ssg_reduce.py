import sqlite3
import json
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Database path.
db_path = os.path.join(script_dir, '..', 'databases', 'ssg_reduce.db')
db_path_222 = os.path.join(
    script_dir,
    '..',
    '..',
    'identify-py-222',
    'databases',
    'ssg_reduce.db',
)

def _query_ssg_reduce(db_path, L0_id, G0_id, it, ik, iso):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id, L0_id, G0_id, it, ik, cell_size, isonum, 
               transformation_matrix, gen_matrix, TTM 
        FROM ssg_reduce
        WHERE L0_id = ? AND G0_id = ? AND it = ? AND ik = ?
        ''',
        (L0_id, G0_id, it, ik),
    )
    results = []
    for row in cursor.fetchall():
        isonum_field = row[6]
        if isinstance(isonum_field, str):
            try:
                isonum_list = json.loads(isonum_field)
            except json.JSONDecodeError:
                isonum_list = eval(isonum_field) if isinstance(isonum_field, str) else isonum_field
        else:
            isonum_list = isonum_field
        if not isinstance(isonum_list, list):
            isonum_list = [isonum_list]
        if iso in isonum_list:
            results.append({
                'id': row[0],
                'L0_id': row[1],
                'G0_id': row[2],
                'it': row[3],
                'ik': row[4],
                'cell_size': row[5],
                'isonum': isonum_list,
                'transformation_matrix': json.loads(row[7]),
                'gen_matrix': json.loads(row[8]),
                'TTM': json.loads(row[9])
            })
    conn.close()
    return results

def find_ssg_reduce(L0_id, G0_id, it, ik, iso, *, use_222_contract=False):
    """Find all matching SSG reduction records."""
    try:
        if use_222_contract and iso == 14:
            return _query_ssg_reduce(db_path_222, L0_id, G0_id, it, ik, iso)
        return _query_ssg_reduce(db_path, L0_id, G0_id, it, ik, iso)
    except sqlite3.OperationalError:
        failed_db_path = db_path_222 if use_222_contract and iso == 14 else db_path
        print(f"Database file not found: {os.path.abspath(failed_db_path)}")
        return []
    except Exception as ex:
        print(f"Failed to query SSG reduction records: {str(ex)}")
        return []

# Manual smoke test.
if __name__ == "__main__":
    def format_results(results):
        """Format query results for manual inspection."""
        if not results:
            return "No matching records found"
        
        output = []
        for i, item in enumerate(results, 1):
            output.append(f"\nRecord #{i} (ID: {item['id']}):")
            output.append(f"L0_id: {item['L0_id']}, G0_id: {item['G0_id']}")
            output.append(f"it: {item['it']}, ik: {item['ik']}, cell_size: {item['cell_size']}")
            output.append(f"isonum: {item['isonum']}")
            output.append(f"transformation_matrix: {item['transformation_matrix']}")
            output.append(f"gen_matrix: {item['gen_matrix']}")
            output.append(f"\nTTM: {item['TTM']}")
            
        return "\n".join(output)

    print("=== SSG_reduce query tool ===")
    
    try:
        L0_id, G0_id, it, ik, iso = 47,221,6,8,21
        print(f"\nQuerying record: L0_id={L0_id}, G0_id={G0_id}, it={it}, ik={ik}, isonum contains {iso}")
        
        results = find_ssg_reduce(L0_id, G0_id, it, ik, iso)
        print(f"\nFound {len(results)} matching records:")
        print(format_results(results))
        
    except ValueError:
        print("Error: all parameters must be integers")
    except Exception as ex:
        print(f"Error: {str(ex)}")
