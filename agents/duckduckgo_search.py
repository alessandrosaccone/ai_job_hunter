from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def search_web(query: str, max_results: int = 10) -> list[dict[str, str]]:
    try:
        return await asyncio.to_thread(_search_sync, query, max_results)
    except Exception as exc:
        logger.warning("[DuckDuckGo] Search failed for '%s': %s", query, exc)
        return []


def _search_sync(query: str, max_results: int) -> list[dict[str, str]]:
    ddgs_cls = _load_ddgs_class()
    if ddgs_cls is None:
        return []

    results: list[dict[str, str]] = []
    with ddgs_cls() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "link": str(item.get("href", item.get("link", ""))),
                    "snippet": str(item.get("body", item.get("snippet", ""))),
                },
            )
    return results


def _load_ddgs_class() -> Any | None:
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        pass

    try:
        from duckduckgo_search import DDGS

        logger.debug("[DuckDuckGo] Using legacy duckduckgo-search package.")
        return DDGS
    except ImportError as exc:
        logger.warning("[DuckDuckGo] Neither ddgs nor duckduckgo-search is installed: %s", exc)
        return None
