from __future__ import annotations

import os
from typing import Any

import httpx

from agents.search_providers.base import QuotaExhaustedError, RateLimitError, infer_company_from_result, is_quota_message, organic_to_results, profile_location_hint

SERPER_SEARCH_URL = "https://google.serper.dev/search"


class SerperProvider:
    name = "serper"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_serper_api_key_here"

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]:
        if not self.is_configured():
            return []

        search_query = query
        if engine == "google_jobs":
            search_query = f"{query} jobs {location}".strip()

        payload: dict[str, Any] = {"q": search_query, "gl": "it", "hl": "it", "num": 10}
        if location.strip():
            payload["location"] = location

        response = await client.post(
            SERPER_SEARCH_URL,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )

        if response.status_code == 429:
            raise RateLimitError(f"Serper rate limit ({response.status_code})")
        if response.status_code == 402:
            raise QuotaExhaustedError(f"Serper quota ({response.status_code})")

        if response.status_code >= 400:
            message = response.text
            if is_quota_message(message):
                raise QuotaExhaustedError(message)
            raise RuntimeError(f"Serper error {response.status_code}: {message[:200]}")

        data = response.json()

        if engine == "google_jobs":
            jobs = self._parse_jobs(data)
            if jobs:
                return jobs

        organic_items: list[dict[str, str]] = []
        for item in data.get("organic", []):
            if not isinstance(item, dict):
                continue
            link = str(item.get("link", ""))
            title = str(item.get("title", ""))
            snippet = str(item.get("snippet", ""))
            organic_items.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "company_name": infer_company_from_result(title, link),
                    "location": profile_location_hint(title, snippet) or location,
                },
            )
        return organic_to_results(organic_items, source_engine="serper")

    def _parse_jobs(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in data.get("jobs", []):
            if not isinstance(item, dict):
                continue
            link = str(item.get("link") or item.get("applyLink") or "")
            if not link:
                continue
            title = str(item.get("title", ""))
            company = str(item.get("companyName") or item.get("company", ""))
            results.append(
                {
                    "source_engine": "serper_jobs",
                    "title": title,
                    "company_name": company,
                    "location": str(item.get("location", "")),
                    "description": str(item.get("description", "")),
                    "apply_options": [{"link": link}],
                    "job_id": link,
                    "link": link,
                },
            )
        return results
