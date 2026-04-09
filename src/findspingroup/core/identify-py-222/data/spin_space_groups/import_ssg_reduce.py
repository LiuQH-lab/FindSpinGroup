import json
import sqlite3
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))


def create_database():
    # Create the database and output directory.
    db_path = os.path.join(script_dir, '../..', 'databases', 'ssg_reduce.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the table schema.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ssg_reduce (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        L0_id INTEGER NOT NULL,
        G0_id INTEGER NOT NULL,
        it INTEGER NOT NULL,
        ik INTEGER NOT NULL,
        cell_size INTEGER NOT NULL,
        isonum TEXT NOT NULL,  -- stored as a JSON array
        transformation_matrix TEXT NOT NULL,
        gen_matrix TEXT NOT NULL,
        TTM TEXT NOT NULL
    )
    ''')
    # Reset existing rows before import.
    cursor.execute('DELETE FROM ssg_reduce')

    # Create an index for faster lookups.
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ids ON ssg_reduce (L0_id, G0_id, it, ik)')
    
    return conn, cursor, db_path

def import_files(cursor):
    # Import all configured source files.
    data_count = 0
    files_path = [os.path.join(script_dir,'222_reduce.json')]
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
        # Serialize the isonum array.
        isonum_str = json.dumps(item.get("isonum", []))
        
        # Serialize matrix fields.
        trans_matrix = json.dumps(item["transformation_matrix"])
        gen_matrix = json.dumps(item["gen_matrix"])
        ttm = json.dumps(item["TTM"])
        
        cursor.execute('''
        INSERT INTO ssg_reduce (L0_id, G0_id, it, ik, cell_size, isonum, transformation_matrix, gen_matrix, TTM)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item["L0_id"],
            item["G0_id"],
            item["it"],
            item["ik"],
            item["cell_size"],
            isonum_str,
            trans_matrix,
            gen_matrix,
            ttm
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
            isonum_str = json.dumps(current_obj.get("isonum") or [])
            trans_matrix = json.dumps(current_obj["transformation_matrix"])
            gen_matrix = json.dumps(current_obj["gen_matrix"])
            ttm = json.dumps(current_obj["TTM"])
            
            cursor.execute('''
            INSERT INTO ssg_reduce (L0_id, G0_id, it, ik, cell_size, isonum, transformation_matrix,gen_matrix, TTM)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)
            ''', (
                current_obj["L0_id"],
                current_obj["G0_id"],
                current_obj["it"],
                current_obj["ik"],
                current_obj["cell_size"],
                isonum_str,
                trans_matrix,
                gen_matrix,
                ttm
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
