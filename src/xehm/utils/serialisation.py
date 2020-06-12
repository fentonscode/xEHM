import json as js
from typing import List, Any


# Select out (clean) a set of keys from an existing dictionary
def select_dictionary_keys(old: dict, select_keys: List[Any]) -> dict:
    return {sel: old[sel] for sel in select_keys}


# Query if a dictionary has a list of required keys
def contains_required_keys(data: dict, required_keys: List[Any]) -> bool:
    return all(req in data for req in required_keys)


# Output a dictionary to a json file and manage some common exceptions
def export_dictionary_to_json(data: dict, file_name: str):
    if not data:
        raise ValueError("Trying to export empty or invalid dictionary")
    if not file_name:
        raise ValueError("No file name supplied")
    try:
        with open(file_name, "w") as file:
            js.dump(data, file, indent=4, sort_keys=True)
    except OSError as error:
        print("Error saving: " + error.filename)
        raise ValueError("Invalid file name or couldn't write to file")
    except js.JSONDecodeError as error:
        print("JSON Error: " + error.msg)
        raise ValueError("Invalid JSON format")
