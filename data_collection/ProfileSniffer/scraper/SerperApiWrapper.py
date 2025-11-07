# Load environment variables
import json
import os
import time

import requests
from dotenv import load_dotenv


class NoTokenAvailable(Exception):
    """Raised when no token with remaining uses is available."""
    pass


class SerperAPIWrapper:
    BASE_URL = "https://google.serper.dev/search"

    def __init__(self, tokens: dict[str, int]):
        self.tokens = tokens
        self.api_token = None

    # --- Internal helpers ---
    def _fetch_token(self) -> str | None:
        """Return a token with remaining uses > 0, otherwise None."""
        for token, remaining in self.tokens.items():
            if remaining > 0:
                return token
        return None

    def _update_token_usage(self, token: str):
        """Decrement token use count or remove it if it reaches 0."""
        if token not in self.tokens:
            return
        if self.tokens[token] > 1:
            self.tokens[token] -= 1
        else:
            del self.tokens[token]

    # --- Main search ---
    def search(self, query, location=None, num_results=10, start=0):
        """Perform one search request using Serper API."""
        token = self._fetch_token()
        if not token:
            raise NoTokenAvailable("No valid Serper API token available.")
        self.api_token = token

        headers = {
            "X-API-KEY": self.api_token,
            "Content-Type": "application/json",
        }

        payload = {
            "q": f"{query} site:ma.linkedin.com/in/",
            "num": num_results,
            "start": start,
            "location": "Morocco",
            "gl": "ma"
        }

        if location:
            payload["location"] = location

        response = requests.post(self.BASE_URL, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}

        self._update_token_usage(self.api_token)
        return response.json()

    # --- Paginated fetching ---
    def fetch_all_results(self, query, location=None, results_per_page=10, max_pages=10, delay=1):
        """Fetch multiple pages of search results."""
        all_results = []
        page = 0

        while True:
            try:
                page_results = self.search(query, location, results_per_page, page * results_per_page)
            except NoTokenAvailable:
                print("❌ No available API tokens remaining.")
                break

            organic_results = page_results.get("organic", [])
            if not organic_results:
                print("No more results found or reached limit.")
                break

            for item in organic_results:
                all_results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })

            print(
                f"Fetched page {page + 1} [max:{max_pages}] with {len(organic_results)} results, total {len(all_results)}")

            page += 1
            if page >= max_pages:
                break

            time.sleep(delay)

        return all_results

    # --- Save results ---
    @classmethod
    def save_results(cls, results: list[dict], output_filename: str):
        existing = []
        if os.path.exists(output_filename):
            with open(output_filename, "r", encoding="utf-8") as f:
                existing = json.load(f)
        combined = existing + results
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=4)
        print(f"💾 Results appended to {output_filename}")


# --- Main execution ---
if __name__ == "__main__":
    load_dotenv()

    SERPER_MAX_SEARCH_PAGES = os.getenv("SERPER_MAX_SEARCH_PAGES")
    tokens_json = os.getenv("SERPER_TOKENS", "{}")
    serper_tokens = json.loads(tokens_json)

    search_engine = SerperAPIWrapper(serper_tokens)

    search_queries = [
        "Frontend Developer",
        "Backend Developer",
        "AI Engineer",
        "JavaScript Developer",
        "PHP Developer",
        "Python Developer",
        "C# Developer",
        "IT Technician",
        "Développeur front-end",
        "Développeur back-end",
        "Ingénieur IA",
        "Développeur JavaScript",
        "Développeur PHP",
        "Développeur Python",
        "Développeur C#",
        "Technicien informatique"
    ]
    search_location = "Morocco"

    all_data = []
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = search_engine.fetch_all_results(query, search_location, max_pages=SERPER_MAX_SEARCH_PAGES)
        all_data.extend(results)

    search_engine.save_results(all_data, "serper_search_results.json")
    print(f"\n✅ Done. Total results: {len(all_data)}")
