import json
from typing import Any, Dict, List, Union

# Mapping from target field -> list of candidate source paths (in priority order)
FIELD_CANDIDATES: Dict[str, List[str]] = {
    "username": ["profile_id"],
    "full_name": ["full_name"],
    "headline": ["headline"],
    "about": ["about"],
    "location": ["location"],
    "profile_photo": ["profile_photo"],
    "projects": ["projects"],
}

def get_by_path(obj: Any, path: str) -> Any:
    """
    Safely get a nested value from obj using a dot-separated path.
    Returns None if any part is missing.
    If the traversal hits a list and the path is not fully consumed,
    returns the list (common for fields like "data.projects").
    """
    if obj is None:
        return None

    parts = path.split('.')
    cur = obj
    for i, part in enumerate(parts):
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
            else:
                return None
        elif isinstance(cur, list):
            # If current is a list and there are no more parts, return the list.
            # If there are remaining parts, it's ambiguous to map -> return the list.
            return cur
        else:
            # current is a primitive (str/int/...), but path still has parts -> missing
            return None
    return cur

def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a normalized dict with the requested target keys.
    For each target key, check candidate source paths in order and use first non-None value.
    If no candidate yields a value, set None.
    """
    normalized: Dict[str, Any] = {}
    for target_key, candidates in FIELD_CANDIDATES.items():
        value = None
        for candidate in candidates:
            value = get_by_path(record, candidate)
            if value is not None:
                break
        # Ensure field exists even if None
        normalized[target_key] = value
    return normalized

def normalize_profiles(input_data: Union[List[Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Accepts either a list of records or a single record (dict).
    Returns a list of normalized records.
    """
    records = []
    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, dict):
                records.append(normalize_record(item))
            else:
                # if item is not a dict (unlikely), keep an empty normalized template
                records.append(normalize_record({}))
    elif isinstance(input_data, dict):
        records.append(normalize_record(input_data))
    else:
        raise ValueError("Input JSON must be a list or an object (dict).")
    return records

if __name__ == "__main__":
    INPUT_FILE = "filtered.json"
    OUTPUT_FILE = "linkedin_profiles_normalized.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = normalize_profiles(data)

    # If you prefer a single object when input was a single object, you can adjust.
    # Here we always write a list for simplicity.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"✅ Normalized {len(normalized)} record(s) -> {OUTPUT_FILE}")
