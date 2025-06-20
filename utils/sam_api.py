from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class SamAPIClient:
    BASE_URL = "https://api.sam.gov/opportunities/v2/search"

    def __init__(self, api_key):
        self.api_key = api_key

    def _fetch_page(self, page, limit, title, ptype, ncode, posted_from, posted_to):
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

        print(f"🔎 Fetching page {page + 1} (offset {offset})...")
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json().get("opportunitiesData", [])

    def _fetch_total_records(self, title=None, ptype=None, ncode=None, posted_from=None, posted_to=None):
        """Fetch just the first page to see how many total records there are."""
        params = {
            "api_key": self.api_key,
            "postedFrom": posted_from,
            "postedTo": posted_to,
            "limit": 1,
            "offset": 0
        }

        if title:
            params["title"] = title
        if ptype:
            params["ptype"] = ptype
        if ncode:
            params["ncode"] = ncode

        print("🔎 Fetching total record count...")
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        total_records = data.get("totalRecords", 0)
        print(f"📈 Total records available: {total_records}")
        return total_records

    def search_opportunities(self, title=None, ptype=None, ncode=None, posted_from=None, posted_to=None, limit=100, max_workers=8):
        if not posted_from:
            posted_from = (datetime.now() - timedelta(days=1)).strftime("%m/%d/%Y")
        if not posted_to:
            posted_to = datetime.now().strftime("%m/%d/%Y")

        # Step 1: Fetch total number of records
        total_records = self._fetch_total_records(title, ptype, ncode, posted_from, posted_to)

        # Step 2: Calculate number of pages
        total_pages = (total_records + limit - 1) // limit  # Ceiling division
        print(f"🗂️ Fetching {total_pages} pages (limit {limit} records per page)")

        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_page, page, limit, title, ptype, ncode, posted_from, posted_to): page
                for page in range(total_pages)
            }

            for future in as_completed(futures):
                page = futures[future]
                try:
                    page_data = future.result()
                    if not page_data:
                        continue
                    all_results.extend(page_data)
                except Exception as e:
                    print(f"⚠️ Error fetching page {page + 1}: {e}")

        print(f"✅ Successfully fetched {len(all_results)} total solicitations.")
        return all_results
