from __future__ import annotations

import os
from typing import Any

import httpx

from agents.search_providers.base import QuotaExhaustedError, RateLimitError, infer_company_from_result, is_quota_message, organic_to_results, profile_location_hint

APIFY_ACTOR = os.getenv("APIFY_GOOGLE_SEARCH_ACTOR", "apify~google-search-scraper")


class ApifyProvider:
    name = "apify"

    def __init__(self, api_token: str | None = None) -> None:
        self.api_token = api_token or os.getenv("APIFY_API_TOKEN", "")

    def is_configured(self) -> bool:
        return bool(self.api_token) and self.api_token != "your_apify_api_token_here"

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]:
        if not self.is_configured():
            return []

        if engine == "google_jobs":
            query = f"{query} jobs {location}".strip()

        url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/run-sync-get-dataset-items"
        response = await client.post(
            url,
            params={"token": self.api_token, "timeout": 45},
            json={
                "queries": query,
                "maxPagesPerQuery": 1,
                "countryCode": "it",
                "languageCode": "it",
                "mobileResults": False,
            },
        )

        if response.status_code == 429:
            raise RateLimitError(f"Apify rate limit ({response.status_code})")
        if response.status_code == 402:
            message = response.text
            if is_quota_message(message):
                raise QuotaExhaustedError(f"Apify quota ({response.status_code})")
            raise RateLimitError(f"Apify temporary limit ({response.status_code})")

        if response.status_code >= 400:
            message = response.text
            if is_quota_message(message):
                raise QuotaExhaustedError(message)
            raise RuntimeError(f"Apify error {response.status_code}: {message[:200]}")

        data = response.json()
        if not isinstance(data, list):
            return []

        organic_items: list[dict[str, str]] = []
        for page in data:
            if not isinstance(page, dict):
                continue
            for item in page.get("organicResults", page.get("results", [])):
                if not isinstance(item, dict):
                    continue
                link = str(item.get("url") or item.get("link") or "")
                title = str(item.get("title", ""))
                snippet = str(item.get("description") or item.get("snippet") or "")
                organic_items.append(
                    {
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "company_name": infer_company_from_result(title, link),
                        "location": profile_location_hint(title, snippet),
                    },
                )

        return organic_to_results(organic_items, source_engine="apify")
