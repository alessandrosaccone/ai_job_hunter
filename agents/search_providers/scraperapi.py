from __future__ import annotations

import os
import re
from html import unescape
from typing import Any
from urllib.parse import quote, unquote

import httpx

from agents.search_providers.base import QuotaExhaustedError, RateLimitError, infer_company_from_result, is_quota_message, organic_to_results, profile_location_hint

SCRAPERAPI_URL = "https://api.scraperapi.com"


class ScraperApiProvider:
    name = "scraperapi"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("SCRAPERAPI_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_scraperapi_api_key_here"

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

        google_url = f"https://www.google.com/search?q={quote(query)}&hl=it&gl=it&num=10"
        response = await client.get(
            SCRAPERAPI_URL,
            params={
                "api_key": self.api_key,
                "url": google_url,
                "country_code": "it",
            },
        )

        if response.status_code == 429:
            raise RateLimitError(f"ScraperAPI rate limit ({response.status_code})")
        if response.status_code == 403:
            raise RateLimitError(f"ScraperAPI temporary block ({response.status_code})")
        if response.status_code == 402:
            raise QuotaExhaustedError(f"ScraperAPI quota ({response.status_code})")

        if response.status_code >= 400:
            message = response.text
            if is_quota_message(message):
                raise QuotaExhaustedError(message)
            raise RuntimeError(f"ScraperAPI error {response.status_code}")

        return organic_to_results(
            self._parse_google_html(response.text),
            source_engine="scraperapi",
        )

    def _parse_google_html(self, html: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        seen: set[str] = set()

        for match in re.finditer(r'<a[^>]+href="(/url\?q=|)(https?://[^"&]+)[^"]*"[^>]*>(.*?)</a>', html, re.I | re.S):
            link = unquote(match.group(2))
            if link in seen or "google.com" in link:
                continue
            title = re.sub(r"<[^>]+>", "", match.group(3))
            title = unescape(title).strip()
            if len(title) < 4:
                continue
            seen.add(link)
            items.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": "",
                    "company_name": infer_company_from_result(title, link),
                    "location": profile_location_hint(title, ""),
                },
            )
            if len(items) >= 10:
                break

        return items
