import json
import re

def extract_twitter_username(url):
    """Extract the Twitter username from a URL."""
    if not url:
        return None
    match = re.search(r"(?:https?://)?(?:www\.)?twitter\.com/([A-Za-z0-9_]+)", url)
    return match.group(1) if match else None

def extract_twitter_usernames(input_file, output_file):
    # Load the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    usernames = []

    for record in data:
        twitter_url = None

        # Try to find the Twitter URL
        if isinstance(record.get("external_links"), dict):
            twitter_url = record["external_links"].get("twitter")

        # Extract username
        username = extract_twitter_username(twitter_url)
        if username:
            usernames.append({
                "user_id": record.get("user_id"),
                "username": username
            })

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(usernames, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(usernames)} usernames → {output_file}")


# Example usage
if __name__ == "__main__":
    extract_twitter_usernames("github_cleaned.json", "twitter_usernames.json")
