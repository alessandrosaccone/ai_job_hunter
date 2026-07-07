from __future__ import annotations

import os
from typing import Any

import httpx

from agents.search_providers.base import QuotaExhaustedError, infer_company_from_result, is_quota_message, profile_location_hint

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


class SerpApiProvider:
    name = "serpapi"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_serpapi_key_here"

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]:
        if not self.is_configured():
            return []

        params = {
            "engine": engine,
            "q": query,
            "api_key": self.api_key,
            "num": 10,
        }
        if location.strip():
            params["location"] = location
        response = await client.get(SERPAPI_ENDPOINT, params=params)

        if response.status_code == 429:
            raise QuotaExhaustedError("SerpApi rate limit (429)")

        payload = response.json()
        error = payload.get("error")
        if error:
            message = str(error)
            if is_quota_message(message):
                raise QuotaExhaustedError(message)
            raise RuntimeError(message)

        response.raise_for_status()

        if engine == "google_jobs":
            jobs = payload.get("jobs_results", [])
            return [{"source_engine": "google_jobs", **item} for item in jobs if isinstance(item, dict)]

        organic = payload.get("organic_results", [])
        parsed: list[dict[str, Any]] = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            link = item.get("link", "")
            title = str(item.get("title", ""))
            snippet = str(item.get("snippet", ""))
            parsed.append(
                {
                    "source_engine": "google",
                    "title": title,
                    "company_name": infer_company_from_result(title, str(link)),
                    "location": profile_location_hint(title, snippet),
                    "description": snippet,
                    "apply_options": [{"link": link}],
                    "job_id": link,
                    "link": link,
                },
            )
        return parsed
