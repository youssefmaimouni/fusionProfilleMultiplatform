import asyncio
import json
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

async def get_photo_image_url(username):
    url = f"https://x.com/{username}/photo"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_selector("img", timeout=10000)
            imgs = await page.query_selector_all("img")
            for img in imgs:
                src = await img.get_attribute("src")
                if src and "profile_images" in src:
                    await browser.close()
                    return src
        except PlaywrightTimeoutError:
            print(f"Timeout for {username}, skipping.")
        except Exception as e:
            print(f"Error for {username}: {e}")
        await browser.close()
        return None

async def main():
    with open("twitter_location_only.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    usernames = [entry["Twitter Username"] for entry in data]
    results = []
    output_file = "profile_images_links.json"

    # Try to load existing results if the script was interrupted before
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_usernames = {entry["username"] for entry in results}
    except (FileNotFoundError, json.JSONDecodeError):
        done_usernames = set()

    for username in usernames:
        if username in done_usernames:
            print(f"Already fetched for {username}, skipping.")
            continue
        print(f"Fetching for {username}...")
        image_url = await get_photo_image_url(username)
        results.append({"username": username, "image_url": image_url})
        # Save after each fetch
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("Results saved to profile_images_links.json")

if __name__ == "__main__":
    asyncio.run(main())