import json

def filter_json_fields(data, fields_to_keep):
    """
    Recursively keep only the fields listed in `fields_to_keep`.
    Fields should be written as dot-separated paths (e.g. "data.name", "response.headline").
    """
    def keep_fields(obj, prefix=''):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                # Keep if exact match or if deeper nested fields exist
                if any(f == full_key or f.startswith(full_key + ".") for f in fields_to_keep):
                    new_dict[key] = keep_fields(value, full_key)
            return new_dict
        elif isinstance(obj, list):
            return [keep_fields(item, prefix) for item in obj]
        else:
            return obj

    return keep_fields(data)


# Example usage:
if __name__ == "__main__":
    with open("unified_linkedin_profiles_all.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🧩 Fields you want to keep
    fields_to_keep = [
        "profile_id",
        "full_name",
        "headline",
        "about",
        "description",
        "location",
        "profile_photo",
        "projects.name",
        "projects.description"
    ]

    filtered_data = filter_json_fields(data, fields_to_keep)

    with open("filtered.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print("✅ Filtered JSON saved to filtered.json")
