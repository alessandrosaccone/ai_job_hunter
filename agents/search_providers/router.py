from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import httpx

from agents.job_search_fallback import search_jobs_deepseek, search_jobs_duckduckgo
from agents.search_providers.apify import ApifyProvider
from agents.search_providers.dataforseo import DataForSeoProvider
from agents.search_providers.serpapi_provider import SerpApiProvider
from agents.search_providers.scraperapi import ScraperApiProvider
from agents.search_providers.serper import SerperProvider
from agents.search_providers.base import QuotaExhaustedError
from storage.search_quota import is_provider_exhausted, mark_provider_exhausted

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER_ORDER = (
    "serpapi",
    "serper",
    "dataforseo",
    "apify",
    "scraperapi",
    "duckduckgo",
    "deepseek",
)

_PROVIDER_LABELS = {
    "serpapi": "SerpApi",
    "serper": "Serper",
    "dataforseo": "DataForSEO",
    "apify": "Apify",
    "scraperapi": "ScraperAPI",
    "duckduckgo": "DuckDuckGo",
    "deepseek": "DeepSeek web",
}


class SearchProvider(Protocol):
    name: str

    def is_configured(self) -> bool: ...

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]: ...


def _empty_stats() -> dict[str, int]:
    return {"ok": 0, "empty": 0, "fail": 0, "results": 0}


class DuckDuckGoProvider:
    name = "duckduckgo"

    def is_configured(self) -> bool:
        return True

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]:
        if engine == "google_jobs":
            return []
        return await search_jobs_duckduckgo(query)


class DeepSeekProvider:
    name = "deepseek"

    def is_configured(self) -> bool:
        key = os.getenv("DEEPSEEK_API_KEY", "")
        return bool(key) and key != "your_deepseek_api_key_here"

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
    ) -> list[dict[str, Any]]:
        if engine == "google_jobs" or not self.is_configured():
            return []
        return await search_jobs_deepseek(query, location)


class JobSearchRouter:
    def __init__(self) -> None:
        self._providers: dict[str, SearchProvider] = {
            "serpapi": SerpApiProvider(),
            "serper": SerperProvider(),
            "dataforseo": DataForSeoProvider(),
            "apify": ApifyProvider(),
            "scraperapi": ScraperApiProvider(),
            "duckduckgo": DuckDuckGoProvider(),
            "deepseek": DeepSeekProvider(),
        }
        self._order = self._load_order()
        self._session_disabled: set[str] = set()
        self._usage_stats: dict[str, dict[str, int]] = {}

    def _load_order(self) -> list[str]:
        raw = os.getenv("SEARCH_PROVIDER_ORDER", "")
        if raw.strip():
            order = [name.strip().lower() for name in raw.split(",") if name.strip()]
        else:
            order = list(DEFAULT_PROVIDER_ORDER)
        return [name for name in order if name in self._providers]

    def reset_usage_stats(self) -> None:
        self._usage_stats = {}

    def get_usage_stats(self) -> dict[str, dict[str, int]]:
        return {
            name: dict(stats)
            for name, stats in sorted(self._usage_stats.items())
            if any(stats.values())
        }

    def _stats_for(self, provider_name: str) -> dict[str, int]:
        return self._usage_stats.setdefault(provider_name, _empty_stats())

    def _short_query(self, query: str, limit: int = 72) -> str:
        cleaned = " ".join(query.split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: limit - 1]}…"

    async def search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
        location: str,
        *,
        job_board_filter: Any | None = None,
        exclude_providers: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        excluded = exclude_providers or set()
        short_query = self._short_query(query)
        tried: list[str] = []

        for provider_name in self._order:
            if provider_name in excluded:
                continue
            if provider_name in self._session_disabled:
                continue
            if is_provider_exhausted(provider_name):
                continue

            provider = self._providers[provider_name]
            if not provider.is_configured():
                continue

            tried.append(provider_name)
            stats = self._stats_for(provider_name)
            label = _PROVIDER_LABELS.get(provider_name, provider_name)

            try:
                results = await provider.search(client, engine, query, location)
            except QuotaExhaustedError as exc:
                stats["fail"] += 1
                mark_provider_exhausted(provider_name)
                self._session_disabled.add(provider_name)
                logger.warning("[%s] Quota esaurita: %s", label, exc)
                continue
            except Exception as exc:
                stats["fail"] += 1
                logger.warning("[%s] Ricerca fallita per '%s': %s", label, short_query, exc)
                continue

            if job_board_filter and provider_name in {"duckduckgo", "scraperapi", "apify", "serper", "dataforseo"}:
                if engine != "google_jobs":
                    results = [
                        item
                        for item in results
                        if job_board_filter(item.get("link") or item.get("apply_options", [{}])[0].get("link", ""))
                    ]

            if results:
                stats["ok"] += 1
                stats["results"] += len(results)
                logger.info(
                    "[%s] OK — %s risultati | engine=%s | '%s'",
                    label,
                    len(results),
                    engine,
                    short_query,
                )
                return results, provider_name

            stats["empty"] += 1
            logger.info("[%s] 0 risultati | engine=%s | '%s' → prossimo provider", label, engine, short_query)

        if tried:
            tried_labels = ", ".join(_PROVIDER_LABELS.get(name, name) for name in tried)
            logger.info(
                "Nessun risultato dopo %s provider (%s) | engine=%s | '%s'",
                len(tried),
                tried_labels,
                engine,
                short_query,
            )
        return [], None

    def configured_providers(self) -> list[str]:
        return [name for name in self._order if self._providers[name].is_configured()]
