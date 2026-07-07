from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

QUOTA_TOKENS = ("quota", "limit", "run out", "exhausted", "billing", "credit", "plan", "insufficient")


class QuotaExhaustedError(Exception):
    """Monthly/billing quota exhausted — skip provider until next month."""


class RateLimitError(Exception):
    """Temporary rate limit (HTTP 429 etc.) — cooldown, not permanent."""


def is_quota_message(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in QUOTA_TOKENS)


def organic_to_results(
    items: list[dict[str, str]],
    *,
    source_engine: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        link = item.get("link", "")
        if not link:
            continue
        results.append(
            {
                "source_engine": source_engine,
                "title": item.get("title", ""),
                "company_name": item.get("company_name", ""),
                "location": item.get("location", ""),
                "description": item.get("snippet", ""),
                "apply_options": [{"link": link}],
                "job_id": link,
                "link": link,
            },
        )
    return results


def profile_location_hint(title: str, snippet: str) -> str:
    match = re.search(r"\b(?:in|a|@)\s+([A-Za-zÀ-ÿ\s,-]{3,40})", f"{title} {snippet}")
    return match.group(1).strip() if match else ""


def infer_company_from_result(title: str, link: str) -> str:
    if " - " in title:
        return title.split(" - ")[-1].strip()
    if " | " in title:
        return title.split(" | ")[-1].strip()
    if not link:
        return ""
    host = urlparse(link).netloc.replace("www.", "")
    return host.split(".")[0].replace("-", " ").title()
