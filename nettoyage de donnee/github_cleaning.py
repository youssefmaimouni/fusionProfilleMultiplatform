#!/usr/bin/env python3
import json
import sys

def convert(input_path, output_path, drop_fields=None):
    drop_fields = set(drop_fields or [])
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If top-level is already an array, keep it (optional safety)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Convert {username: {..}} => [{username: ..., ...}, ...]
        records = []
        for uname, profile in data.items():
            rec = dict(profile) if isinstance(profile, dict) else {"value": profile}
            rec["username"] = uname
            # drop unwanted fields
            for field in list(drop_fields):
                if field in rec:
                    rec.pop(field, None)
            records.append(rec)
    else:
        raise ValueError("Unexpected JSON structure: expected object or array at top-level")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_profiles.py all_data_github.json github_cleaned.json")
        sys.exit(1)
    # Example drop list: edit if you want to remove fields
    drop = ["profile_readme", "followers","following", "website", "avatar_local_path","platform","public_repos_count","updated_at","created_at","company","skills","network_metrics","activity_metrics","account_created","text_embedding","visual_embedding"]
    convert(sys.argv[1], sys.argv[2], drop_fields=drop)
