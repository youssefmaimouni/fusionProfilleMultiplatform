import json
import os

def get_all_profiles(directories):
    """Get all unique profiles from the specified directories"""
    profiles_dict = {}  # Use a dict to automatically remove duplicates

    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    file_path = os.path.join(directory, filename)
                    username = os.path.splitext(filename)[0]
                    if username in profiles_dict:
                        continue  # Skip duplicate usernames
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            profile_data = json.load(f)
                            profiles_dict[username] = {
                                "Twitter Username": username,
                                "profile_data": profile_data
                            }
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    return list(profiles_dict.values())


# List of directories containing scraped profiles
profile_directories = [
    "scraped_profiles",
    "VM1_Data/scraped_profiles",
    "VM2_Data/scraped_profiles",
]

# Get all unique profiles
all_profiles = get_all_profiles(profile_directories)

# Try to load profile images if available
try:
    with open("profile_images_links.json", "r", encoding="utf-8") as f:
        images_data = json.load(f)
        images_dict = {
            entry["username"]: entry["image_url"]
            for entry in images_data if "username" in entry
        }

        # Add image URLs to profiles
        for profile in all_profiles:
            if profile["Twitter Username"] :
                username = profile["Twitter Username"] 
            else:
                username = profile["username"]
            profile["image_url"] = images_dict.get(username)
except Exception as e:
    print(f"Could not load profile images: {e}")

# Save all unique profiles to new file
output_file = "all_profiles_combined1.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_profiles, f, ensure_ascii=False, indent=2)

print(f"✅ Found {len(all_profiles)} unique profiles")
print(f"💾 Saved all profiles to {output_file}")
