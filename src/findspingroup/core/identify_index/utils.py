import json
import os
def load_test_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"JSON file does not exist: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {str(e)} at line {e.lineno}")
