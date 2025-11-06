#!/usr/bin/env python3
"""
clean_twitter_json.py

Usage examples:

# JSON array input -> cleaned JSON array output
python twitter_cleaning.py --input all_profiles_combined_twitter.json --output twitter_data_cleaned.json

"""

import json
import argparse
from pathlib import Path
from typing import Any, List


def load_input(path: Path, ndjson: bool):
    if ndjson:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]


def write_output(path: Path, records: List[dict], ndjson: bool):
    if ndjson:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def get_path(obj: Any, dotpath: str):
    """Return value at dotpath, or None if missing."""
    parts = dotpath.split(".") if dotpath else []
    cur = obj
    for p in parts:
        if not isinstance(cur, dict):
            return None
        if p not in cur:
            return None
        cur = cur[p]
    return cur


def del_path(obj: dict, dotpath: str):
    """Delete the key at dotpath if present (no error if missing)."""
    parts = dotpath.split(".")
    cur = obj
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    # remove final key if exists
    last = parts[-1]
    if isinstance(cur, dict) and last in cur:
        del cur[last]


def is_empty_value(v: Any) -> bool:
    """Consider None, empty string, empty list, empty dict as empty."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple, set)) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False


def get_username_for_dedupe(rec: dict):
    """Try profile_data.user_profile.username then 'Twitter Username' then None."""
    u = get_path(rec, "profile_data.user_profile.username")
    if u:
        return u
    # also try top-level user_profile.username (some records might have it at top)
    u_top = get_path(rec, "user_profile.username")
    if u_top:
        return u_top
    u2 = rec.get("Twitter Username")
    return u2


def transform_record(rec: dict):
    """
    Flatten user_profile and profile image into top-level fields as requested.

    - Prefer profile_data.user_profile over user_profile if present.
    - Create top-level 'username' and 'bio' keys from the nested user_profile.
    - If top-level image_url is empty and a profile_image_url exists (either top-level
      or profile_data.profile_image_url), set image_url to that value.
    - Remove the original user_profile / profile_data.user_profile and profile_image_url
      fields so the final output matches the desired shape.
    - Finally, remove profile_data if it is an empty dict or if all its contents are empty.
    """

    def dict_or_list_all_empty(x):
        """Return True if x (dict/list/other) is empty by our `is_empty_value` rules,
        or if all nested values are empty. Works recursively for dicts and lists."""
        if is_empty_value(x):
            return True
        if isinstance(x, dict):
            if len(x) == 0:
                return True
            for v in x.values():
                if not dict_or_list_all_empty(v):
                    return False
            return True
        if isinstance(x, (list, tuple, set)):
            if len(x) == 0:
                return True
            for item in x:
                if not dict_or_list_all_empty(item):
                    return False
            return True
        # primitive non-empty (str with content, number, etc.)
        return False

    # 1) Flatten user_profile -> username, bio
    user_profile = get_path(rec, "profile_data.user_profile")
    if not isinstance(user_profile, dict):
        user_profile = rec.get("user_profile")

    if isinstance(user_profile, dict):
        if "username" not in rec and user_profile.get("username") is not None:
            rec["username"] = user_profile.get("username")
        if "bio" not in rec and user_profile.get("bio") is not None:
            rec["bio"] = user_profile.get("bio")

    # 2) If image_url is empty/null, try to fill from profile_image_url variants
    cur_image = rec.get("image_url")
    if is_empty_value(cur_image):
        profile_img = get_path(rec, "profile_data.profile_image_url")
        if is_empty_value(profile_img):
            profile_img = rec.get("profile_image_url")
        if not is_empty_value(profile_img):
            rec["image_url"] = profile_img

    # 3) Remove the original nested containers to avoid duplication
    if "user_profile" in rec and isinstance(rec["user_profile"], dict):
        del rec["user_profile"]
    del_path(rec, "profile_data.user_profile")
    if "profile_image_url" in rec:
        del rec["profile_image_url"]
    del_path(rec, "profile_data.profile_image_url")

    # 4) If profile_data exists but is empty (or only contains empty values), remove it entirely
    if "profile_data" in rec:
        try:
            if dict_or_list_all_empty(rec["profile_data"]):
                del rec["profile_data"]
        except Exception:
            # be conservative: if something unexpected occurs, leave it as-is
            pass

def clean_records(records: List[dict],
                  drop_fields: List[str],
                  important_fields: List[str],
                  dedupe: bool = True,
                  keep: str = "first"):
    """
    drop_fields: list of dot paths to delete from each record
    important_fields: list of dot paths which are considered non-empty indicators;
                      if ALL these fields are empty -> drop the record
    dedupe: whether to remove duplicate usernames
    keep: 'first' or 'last' duplicate to keep
    """
    seen = {}  # username -> index (if keep == 'first') or replaced index
    out = []

    for rec in records:
        # FIRST: transform / flatten the record so subsequent checks use top-level fields
        transform_record(rec)

        # 1) Remove requested fields
        for f in drop_fields:
            del_path(rec, f)

        # 2) Check if all important fields are empty
        all_empty = True
        for pf in important_fields:
            val = get_path(rec, pf)
            # support checking top-level keys too (if pf has no dot)
            if val is None and "." not in pf:
                val = rec.get(pf)
            if not is_empty_value(val):
                all_empty = False
                break

        if all_empty:
            # skip this record
            continue

        # 3) Deduplicate by username
        if dedupe:
            username = get_username_for_dedupe(rec)
            # normalize username if it's not a string
            if isinstance(username, (int, float)):
                username = str(username)
            if username is None:
                # treat None as a normal record (can't dedupe)
                out.append(rec)
            else:
                if username in seen:
                    # already seen
                    if keep == "first":
                        # skip this duplicate
                        continue
                    else:
                        # keep == 'last' -> replace previous
                        prev_idx = seen[username]
                        out[prev_idx] = rec
                        # seen mapping still points to same index
                else:
                    seen[username] = len(out)
                    out.append(rec)
        else:
            out.append(rec)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSON (array) or NDJSON file")
    parser.add_argument("--output", "-o", required=True, help="Output file path (JSON array or NDJSON)")
    parser.add_argument("--ndjson", action="store_true", help="Set if input/output should be NDJSON (line-delimited)")
    parser.add_argument("--keep", choices=("first", "last"), default="first", help="When deduping, keep first or last occurrence")
    args = parser.parse_args()

    IN = Path(args.input)
    OUT = Path(args.output)
    ndjson = args.ndjson

    # -------------- CONFIGURE these two lists --------------
    # Fields to remove from each record (dot notation supported)
    drop_fields = [
        # add fields you want to delete globally
        # example: "profile_data.some_unwanted_field", "some_top_level_field"
        # I'll include the fields you mentioned as "unnecessary columns" if you want them removed:
        "profile_data.followers",
        "profile_data.followers[]",            # harmless if not present
        "profile_data.followers[].follower_bio",
        "profile_data.followers[].follower_name",
        "profile_data.following",
        "profile_data.following[]",
        "profile_data.following[].following_bio",
        "profile_data.following[].following_name",
        "profile_data.tweets",
        "profile_data.tweets[]",
        "profile_data.tweets[].tweet_content",
        "profile_data.tweets[].tweet_date",
        # if you want to remove the top-level image_url as well:
        # "image_url"
    ]

    # Fields which must not all be empty; if ALL listed fields are empty -> delete the record.
    # Use dot notation for nested fields.
    important_fields = [
        "image_url",
        "profile_data.followers",
        "profile_data.following",
        "profile_data.profile_image_url",
        "profile_data.tweets",
        "profile_data.user_profile",
        "profile_data.user_profile.bio",
        # also consider top-level user_profile fields and top-level username/bio:
        "user_profile",
        "user_profile.bio",
        "username",
        "bio",
    ]
    # -------------------------------------------------------

    print("Loading input...")
    records = load_input(IN, ndjson=ndjson)
    print(f"Loaded {len(records)} records.")

    cleaned = clean_records(records,
                            drop_fields=drop_fields,
                            important_fields=important_fields,
                            dedupe=True,
                            keep=args.keep)

    print(f"After cleaning: {len(cleaned)} records remain.")
    write_output(OUT, cleaned, ndjson=ndjson)
    print(f"Wrote cleaned output to: {OUT}")


if __name__ == "__main__":
    main()
