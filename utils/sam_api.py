from datetime import datetime, timedelta
import requests

class SamAPIClient:
    BASE_URL = "https://api.sam.gov/opportunities/v2/search"

    def __init__(self, api_key):
        self.api_key = api_key

    def search_opportunities(self, title=None, ptype=None, ncode=None, posted_from=None, posted_to=None, limit=100, max_pages=1000):
        if not posted_from:
            posted_from = (datetime.now() - timedelta(days=30)).strftime("%m/%d/%Y")
        if not posted_to:
            posted_to = datetime.now().strftime("%m/%d/%Y")

        all_results = []
        for page in range(max_pages):
            offset = page * limit

            params = {
                "api_key": self.api_key,
                "postedFrom": posted_from,
                "postedTo": posted_to,
                "limit": limit,
                "offset": offset
            }

            if title:
                params["title"] = title
            if ptype:
                params["ptype"] = ptype
            if ncode:
                params["ncode"] = ncode

            print(f"🔎 Fetching page {page + 1} with offset {offset}")
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json().get("opportunitiesData", [])

            if not data:
                break

            all_results.extend(data)

        return all_results
