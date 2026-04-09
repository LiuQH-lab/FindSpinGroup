import json
import sqlite3
import os

# Resolve the current script directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# JSON file path, stored next to this script.
json_path = os.path.join(script_dir, 'space_groups.json')
# Database path in the parent databases directory.
db_path = os.path.join(script_dir, '../..', 'databases', 'space_groups.db')

try:
    # Load the JSON payload.
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create the target directory if needed.
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Open the database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the table.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS space_groups (
        id INTEGER PRIMARY KEY,
        generators TEXT NOT NULL,
        all_matrices TEXT NOT NULL
    )
    ''')
    
    # Reset existing rows before import.
    cursor.execute('DELETE FROM space_groups')
    
    # Insert all rows.
    for group in data:
        cursor.execute('''
        INSERT INTO space_groups (id, generators, all_matrices)
        VALUES (?, ?, ?)
        ''', (
            group['space_group_id'],
            json.dumps(group['generators']),
            json.dumps(group['all_matrices'])
        ))
    
    conn.commit()
    conn.close()
    print(f"Database created successfully at {os.path.abspath(db_path)}")
    print(f"Imported {len(data)} space-group records")

except FileNotFoundError:
    print(f"Error: JSON file not found: '{os.path.abspath(json_path)}'")
except Exception as e:
    print(f"Import failed: {str(e)}")
