import json
import sqlite3
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))



def create_database():
    # Create the database and output directory.
    db_path = os.path.join(script_dir, '../..', 'databases', 'ssg_map.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the table schema.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ssg_map (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        L0_id INTEGER NOT NULL,
        G0_id INTEGER NOT NULL,
        it INTEGER NOT NULL,
        ik INTEGER NOT NULL,
        num INTEGER NOT NULL,
        isonum INTEGER NOT NULL,
        transformation_matrix TEXT NOT NULL,
        all_maps TEXT NOT NULL,
        transformation_maps TEXT NOT NULL,
        old_num TEXT NOT NULL,
        old_trans_1 TEXT NOT NULL,
        old_trans_2 TEXT NOT NULL
    )
    ''')
    # Reset existing rows before import.
    cursor.execute('DELETE FROM ssg_map')

    # Create an index for faster lookups.
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ids ON ssg_map (L0_id, G0_id, it, ik, isonum)')
    
    return conn, cursor, db_path

def import_files(cursor):
    # Import all configured source files.
    data_count = 0
    files_path = [os.path.join(script_dir,'222_map.json')
                  ]
    for file_name in files_path:
        if not os.path.exists(file_name):
            print(f"Warning: file not found, skipping '{file_name}'")
            continue
            
        with open(file_name, 'r') as f:
            try:
                # First try the standard JSON-array format.
                data = json.load(f)
                process_data(cursor, data)
                data_count += len(data)
                print(f"Imported {len(data)} records from {file_name}")
            except json.JSONDecodeError:
                print(f"Falling back to line-by-line parsing for {file_name}...")
                f.seek(0)
                data_count += process_line_by_line(f, cursor)

    return data_count

def process_data(cursor, data):
    for item in data:

        all_maps = json.dumps(item["all_maps"])
        # Serialize matrix fields.
        trans_matrix = json.dumps(item["transformation_matrix"])
        transformation_maps = json.dumps(item["transformation_maps"])
        old_num = json.dumps(item["old_num"])
        old_trans_1 = json.dumps(item["old_trans_1"])
        old_trans_2 = json.dumps(item["old_trans_2"])

        cursor.execute('''
        INSERT INTO ssg_map (L0_id, G0_id, it, ik, num, isonum, transformation_matrix, all_maps,transformation_maps,old_num,old_trans_1,old_trans_2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)
        ''', (
            item["L0_id"],
            item["G0_id"],
            item["it"],
            item["ik"],
            item["num"],
            item["isonum"],
            trans_matrix,
            all_maps,
            transformation_maps,
            old_num,
            old_trans_1,
            old_trans_2
        ))

def process_line_by_line(f, cursor):
    count = 0
    current_obj = {}
    for line in f:
        stripped = line.strip()
        # Skip blank lines and comments.
        if not stripped or stripped.startswith('//') or stripped.startswith('#'):
            continue
            
        # Detect the start of an object.
        if '{' in stripped:
            current_obj = {}
            
        # Parse key-value pairs.
        elif ':' in stripped and current_obj is not None:
            key, value = stripped.split(':', 1)
            key = key.strip().strip('"')
            value = value.strip().rstrip(',')
            
            # Parse JSON values when possible.
            try:
                current_obj[key] = json.loads(value)
            except json.JSONDecodeError:
                # Fall back to a plain string.
                current_obj[key] = value.strip('"')
        
        # Detect the end of an object.
        if '}' in stripped and current_obj:
            # Insert the current object.
            all_maps = json.dumps(current_obj["all_maps"])
            trans_matrix = json.dumps(current_obj["transformation_matrix"])
            transformation_maps = json.dumps(current_obj["transformation_maps"])
            
            cursor.execute('''
            INSERT INTO ssg_map (L0_id, G0_id, it, ik, num, isonum, transformation_matrix,all_maps, transformation_maps)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                current_obj["L0_id"],
                current_obj["G0_id"],
                current_obj["it"],
                current_obj["ik"],
                current_obj["num"],
                current_obj["isonum"],
                trans_matrix,
                all_maps,
                transformation_maps
            ))
            
            count += 1
            current_obj = {}
            
    return count

if __name__ == "__main__":
    conn, cursor, db_path = create_database()
    
    print("Starting import...")
    total_count = import_files(cursor)
    
    conn.commit()
    conn.close()
    
    if total_count > 0:
        print(f"\nDatabase created successfully at: {os.path.abspath(db_path)}")
        print(f"Imported {total_count} records in total")
    else:
        print("No data was imported; please check the file format and file paths")
