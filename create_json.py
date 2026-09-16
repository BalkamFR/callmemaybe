from pathlib import Path
import json

def create_json(raw_list, path="data/output/function_calling_results.json"):
    parsed_data = [item if isinstance(item, dict) else json.loads(item) for item in raw_list]
    json_str = json.dumps(parsed_data, indent=4)

    file_path = Path(path)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(json_str)