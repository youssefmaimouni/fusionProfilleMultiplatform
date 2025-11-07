import json
from pathlib import Path

COOKIES_FILE = Path(__file__).parent / "app" / "twitter_cookies.json"

cookies = json.loads(COOKIES_FILE.read_text(encoding="utf-8"))

for c in cookies:
    # Ensure sameSite is one of the allowed values
    ss = c.get("sameSite", "")
    if isinstance(ss, str):
        ss_lower = ss.lower()
        if "strict" in ss_lower:
            c["sameSite"] = "Strict"
        elif "lax" in ss_lower:
            c["sameSite"] = "Lax"
        elif "none" in ss_lower:
            c["sameSite"] = "None"
        else:
            c.pop("sameSite", None)  # remove invalid sameSite
    else:
        c.pop("sameSite", None)  # remove if not a string

# Save corrected cookies back (optional)
Path(COOKIES_FILE).write_text(json.dumps(cookies, indent=2))
print("Cookies fixed!")
