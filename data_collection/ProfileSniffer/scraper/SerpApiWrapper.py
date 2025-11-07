import json
import os
import time

from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables from .env file
load_dotenv()


class NoTokenAvailable(Exception):
    """Raised when no token with remaining uses is available."""
    pass


class SerpApiWrapper:
    def __init__(self, tokens: dict[str, int], search_engine: str):
        self.tokens: dict = tokens
        self.api_token: str | None = None
        self.search_engine: str = search_engine

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

    def search(self, search_query, location, results_per_request=10, start_index=0):
        if not (1 <= results_per_request <= 10):
            raise ValueError("results_per_request must be between 1 and 10 (API limit).")

        # ✅ Fetch a valid token
        token = self._fetch_token()
        if not token:
            raise NoTokenAvailable("No valid SerpAPI token available.")
        self.api_token = token

        params = {
            "api_key": self.api_token,
            "engine": self.search_engine,
            "q": f"{search_query} site:ma.linkedin.com/in/",
            "location": location,
            "num": results_per_request,
            "start": start_index,
        }

        search = GoogleSearch(params)
        page_results = search.get_dict()

        # ✅ Update or remove token after use
        self._update_token_usage(self.api_token)

        return page_results

    def fetch_all_results(self, search_query, location, results_per_page=10, max_pages=10, delay=1):

        all_results = []
        page = 0

        while True:
            try:
                page_results = self.search(search_query, location, results_per_page, page * results_per_page)
            except NoTokenAvailable:
                print("❌ No available API tokens remaining.")
                break

            # Check for search results
            organic_results = page_results.get("organic_results", [])
            if not organic_results:
                print("No more results found or reached limit.")
                break

            # Extract desired fields
            for item in organic_results:
                all_results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })

            print(
                f"Fetched page {page + 1} [max:{max_pages}] with {len(organic_results)} results, total is {len(all_results)}")

            # Pagination limit
            page += 1
            if page >= max_pages or "next" not in page_results.get("serpapi_pagination", {}):
                break

            time.sleep(delay)

        return all_results

    # --- Save results ---
    @classmethod
    def save_results(cls, results: list[dict], output_filename: str):
        """Save results as a JSON file."""
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"💾 Results saved to {output_filename}")


if __name__ == "__main__":
    SERP_API_ENGINE_ID = os.environ['SERP_API_ENGINE_ID']
    tokens_json = os.getenv("SERP_API_TOKENS", "{}")
    serp_api_tokens = json.loads(tokens_json)

    search_engine = SerpApiWrapper(serp_api_tokens, SERP_API_ENGINE_ID)

    search_queries = ["Machine Learning Engineer", "Data Analyst", "Software Engineer", "AI Researcher"]
    search_location = "Morocco"

    all_data = []
    for search_query in search_queries:
        print(f"\n🔍 Starting search for: '{search_query}'")

        results = search_engine.fetch_all_results(search_query, search_location, max_pages=40)

        all_data.extend(results)

    search_engine.save_results(all_data, "search_results.json")

    print(all_data)
