from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from agents.search_providers.base import QuotaExhaustedError, RateLimitError, infer_company_from_result, is_quota_message, organic_to_results, profile_location_hint

DATAFORSEO_ORGANIC_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"
DATAFORSEO_JOBS_URL = "https://api.dataforseo.com/v3/serp/google/jobs/live/regular"


class DataForSeoProvider:
    name = "dataforseo"

    def __init__(
        self,
        login: str | None = None,
        password: str | None = None,
    ) -> None:
        self.login = login or os.getenv("DATAFORSEO_LOGIN", "")
        self.password = password or os.getenv("DATAFORSEO_PASSWORD", "")

    def is_configured(self) -> bool:
        return bool(self.login and self.password)

    def _auth_header(self) -> dict[str, str]:
        token = base64.b64encode(f"{self.login}:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]:
        if not self.is_configured():
            return []

        endpoint = DATAFORSEO_JOBS_URL if engine == "google_jobs" else DATAFORSEO_ORGANIC_URL
        payload = [
            {
                "keyword": query,
                "location_name": location or "Italy",
                "language_code": "it",
                "depth": 10,
            },
        ]

        response = await client.post(endpoint, headers=self._auth_header(), json=payload)
        if response.status_code == 429:
            raise RateLimitError(f"DataForSEO rate limit ({response.status_code})")
        if response.status_code == 402:
            raise QuotaExhaustedError(f"DataForSEO quota ({response.status_code})")

        data = response.json()
        if data.get("status_code") in {40200, 40501}:
            raise QuotaExhaustedError(str(data.get("status_message", "DataForSEO quota")))

        message = str(data.get("status_message", ""))
        if is_quota_message(message):
            raise QuotaExhaustedError(message)

        tasks = data.get("tasks", [])
        if not tasks:
            return []

        task = tasks[0]
        if task.get("status_code") != 20000:
            task_message = str(task.get("status_message", ""))
            if is_quota_message(task_message):
                raise QuotaExhaustedError(task_message)
            return []

        result_blocks = task.get("result", [])
        if not result_blocks:
            return []

        items = result_blocks[0].get("items", [])
        if engine == "google_jobs":
            return self._parse_jobs(items)

        organic_items: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in {None, "organic"}:
                continue
            link = str(item.get("url", ""))
            title = str(item.get("title", ""))
            snippet = str(item.get("description", ""))
            organic_items.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "company_name": infer_company_from_result(title, link),
                    "location": profile_location_hint(title, snippet),
                },
            )
        return organic_to_results(organic_items, source_engine="dataforseo")

    def _parse_jobs(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            link = str(item.get("url") or item.get("job_link") or "")
            if not link:
                continue
            title = str(item.get("title", ""))
            results.append(
                {
                    "source_engine": "dataforseo_jobs",
                    "title": title,
                    "company_name": str(item.get("company_name", "")),
                    "location": str(item.get("location", "")),
                    "description": str(item.get("description", "")),
                    "apply_options": [{"link": link}],
                    "job_id": link,
                    "link": link,
                },
            )
        return results
