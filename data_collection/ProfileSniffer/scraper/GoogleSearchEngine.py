import csv
import os
import time

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_CLOUD_API_KEY = os.environ['GOOGLE_CLOUD_API_KEY']
GOOGLE_SEARCH_ENGINE_ID = os.environ['GOOGLE_SEARCH_ENGINE_ID']


class GoogleSearchEngine:
    def __init__(self, cloud_api, search_engine_id):
        self.cloud_api = cloud_api
        self.search_engine_id = search_engine_id
        self.base_search_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query, results_per_request: int = 10, start_index: int = 0):
        if not (1 <= results_per_request <= 10):
            raise ValueError("results_per_request must be between 1 and 10 (API limit).")

        if not (1 <= start_index <= 91):
            raise ValueError("start_index must be between 1 and 91 (API limit).")

        params = {
            "key": self.cloud_api,
            "cx": self.search_engine_id,
            "q": query,
            "num": results_per_request,
            "start": start_index,
            "format": "json",
            "cr": "countryMA",  # Restrict to Morocco
            "gl": "ma",  # Region = Morocco
        }

        response = requests.get(self.base_search_url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_n_results(self, search_query, results_count=20, delay=1):
        if not 1 <= results_count <= 100:
            raise ValueError("results_count must be between 1 and 100 (API limit).")

        results = []
        results_per_request = 10
        pages_needed = (results_count + results_per_request - 1) // results_per_request

        for page in range(pages_needed):
            print(f"Fetching page {page + 1}/{pages_needed}")

            start = page * results_per_request + 1

            data = self.search(search_query, start_index=start, results_per_request=results_per_request)

            items = data.get("items", [])
            results.extend(items)

            time.sleep(delay)

        return results


def save_all_results_to_csv(all_results, filename="google_search_results.csv"):
    """Save all search results (from multiple queries) into one CSV file."""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Search Query", "Title", "Snippet", "Link"])

        for entry in all_results:
            writer.writerow([
                entry.get("query", ""),
                entry.get("title", ""),
                entry.get("snippet", ""),
                entry.get("link", "")
            ])

    print(f"\n✅ All results saved to {filename}")


if __name__ == "__main__":
    search_engine = GoogleSearchEngine(GOOGLE_CLOUD_API_KEY, GOOGLE_SEARCH_ENGINE_ID)

    search_terms = [
        "Information Technology",
        "Technologies de l'information",
        "IT services",
        "Services informatiques",
        "Digital transformation",
        "Transformation numérique",
        "Computer science",
        "Informatique",
        "Software development",
        "Développement logiciel",
        "Cloud computing",
        "Informatique en nuage"
    ]

    all_results = []
    for term in search_terms:
        results = search_engine.fetch_n_results(term, results_count=100)
        print(term, len(results))
        for item in results:
            all_results.append({
                "query": term,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })

    save_all_results_to_csv(all_results)
