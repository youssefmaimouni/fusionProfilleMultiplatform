import json
import re

def extract_twitter_username(url):
    """Extract Twitter username from a URL."""
    if not url:
        return None
    match = re.search(r"(?:https?://)?(?:www\.)?twitter\.com/([A-Za-z0-9_]+)", url)
    return match.group(1) if match else None

def extract_usernames(input_file, output_file):
    # Load JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []

    for record in data:
        github_user = record.get("GitHub Username")
        link = record.get("Link")
        twitter_username = extract_twitter_username(link)

        if twitter_username:
            output.append({
                "GitHub Username": github_user,
                "Twitter Username": twitter_username
            })

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(output)} Twitter usernames → {output_file}")


# Example usage
if __name__ == "__main__":
    extract_usernames("twitter.json", "github_twitter_usernames.json")
