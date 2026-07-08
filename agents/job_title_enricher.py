from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from html import unescape
from typing import Any
from urllib.parse import urlparse

import httpx

from agents.job_prefilter import (
    extract_posting_salary_hint,
    format_salary_range_eur,
    job_posting_salary_display,
    linkedin_applications_closed,
)
from models.job import JobPosting

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SCRAPERAPI_URL = "https://api.scraperapi.com"
MAX_CONCURRENT_ENRICHMENTS = 4

BULK_TITLE_RE = re.compile(
    r"\b(\d{2,})\s+(?:jobs?|offerte?|positions?|annunci|lavori|openings?|empleos?)\b",
    re.I,
)
AGGREGATE_TITLE_RE = re.compile(
    r"(?:empleos?\s+de|jobs?\s+in\b|offers?\s+in\b|lavori\s+a\b|vacancies\s+in\b|\+)\s*",
    re.I,
)
GENERIC_TITLES = {
    "software engineer",
    "software developer",
    "developer",
    "engineer",
    "offerta di lavoro",
    "job offer",
}
BAD_COMPANY_NAMES = {
    "es",
    "it",
    "de",
    "uk",
    "fr",
    "jobs",
    "empleos",
    "www",
    "linkedin",
    "indeed",
    "glassdoor",
}
JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
META_TAG_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?P<key>[^"\']+)["\'][^>]+content=["\'](?P<content>[^"\']+)["\']',
    re.I,
)
META_TAG_RE_ALT = re.compile(
    r'<meta[^>]+content=["\'](?P<content>[^"\']+)["\'][^>]+(?:property|name)=["\'](?P<key>[^"\']+)["\']',
    re.I,
)
TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
LINKEDIN_HIRING_RE = re.compile(
    r"^(?P<company>.+?)\s+hiring\s+(?P<title>.+?)\s+in\s+.+\s+\|\s*LinkedIn\s*$",
    re.I,
)
LINKEDIN_HIRING_IT_RE = re.compile(
    r"^(?P<company>.+?)\s+sta assumendo\s+(?P<title>.+?)\s+in\s+.+$",
    re.I,
)
LINKEDIN_PIPE_RE = re.compile(
    r"^(?P<title>.+?)\s+\|\s+(?P<company>.+?)\s+\|\s*LinkedIn\s*$",
    re.I,
)
REASONING_TITLE_RE = re.compile(
    r"(?:posizione|ruolo|titolo|role|position)\s+(?:di\s+|del\s+|della\s+)?([^\.]+?)\s+presso\s+",
    re.I,
)
SALARY_PATTERN_RE = re.compile(
    r"(?:€\s*\d{2,3}(?:[.,]\d{3})?|\d{2,3}\s*[kK]\s*€?|\d{2,3}(?:[.,]\d{3})?\s*EUR)",
    re.I,
)
SALARY_HINT_WORD_RE = re.compile(
    r"\b(?:salary|compensation|stipendio|retribuzione|ral|range|pay)\b",
    re.I,
)


def looks_like_bulk_listing_title(title: str) -> bool:
    cleaned = (title or "").strip()
    if not cleaned:
        return True
    if BULK_TITLE_RE.search(cleaned):
        return True
    if AGGREGATE_TITLE_RE.search(cleaned):
        return True
    if cleaned.startswith("+") and re.search(r"\d", cleaned):
        return True
    return False


def needs_title_enrichment(title: str) -> bool:
    cleaned = (title or "").strip()
    if not cleaned:
        return True
    if looks_like_bulk_listing_title(cleaned):
        return True
    if cleaned.lower() in GENERIC_TITLES:
        return True
    if cleaned.isdigit():
        return True
    if " sta assumendo " in cleaned.lower() or " hiring " in cleaned.lower():
        return True
    return False


def needs_company_enrichment(company: str) -> bool:
    cleaned = (company or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in BAD_COMPANY_NAMES:
        return True
    if len(cleaned) <= 2:
        return True
    return False


DIRECT_JOB_POSTING_MARKERS = (
    "linkedin.com/jobs/view",
    "jobs.lever.co",
    "boards.greenhouse.io",
)


def _is_direct_job_posting_url(url: str) -> bool:
    lowered = (url or "").lower()
    if "myworkdayjobs.com" in lowered or "myworkdaysite.com" in lowered:
        return "/job/" in lowered
    return any(marker in lowered for marker in DIRECT_JOB_POSTING_MARKERS)


DESCRIPTION_SHORT_THRESHOLD = 800


def needs_job_enrichment(job: JobPosting) -> bool:
    if needs_title_enrichment(job.title) or needs_company_enrichment(job.company):
        return True
    if job_posting_salary_display(job):
        return False
    if job.source == "serpapi":
        return True
    if _is_direct_job_posting_url(job.url):
        return True
    return False


def needs_accurate_match(job: JobPosting) -> bool:
    """True when the posting likely has only a SERP snippet, not the full job text."""
    if job.source == "serpapi":
        return True
    if len((job.description or "").strip()) < DESCRIPTION_SHORT_THRESHOLD:
        return True
    return False


def parse_social_title(raw_title: str) -> tuple[str | None, str | None]:
    title = unescape(raw_title or "").strip()
    if not title:
        return None, None

    hiring_match = LINKEDIN_HIRING_RE.match(title)
    if hiring_match:
        return hiring_match.group("title").strip(), hiring_match.group("company").strip()

    hiring_it_match = LINKEDIN_HIRING_IT_RE.match(title)
    if hiring_it_match:
        return hiring_it_match.group("title").strip(), hiring_it_match.group("company").strip()

    pipe_match = LINKEDIN_PIPE_RE.match(title)
    if pipe_match:
        return pipe_match.group("title").strip(), pipe_match.group("company").strip()

    title = re.sub(r"\s+\|\s+LinkedIn\s*$", "", title, flags=re.I).strip()

    if " hiring " in title.lower():
        parts = re.split(r"\s+hiring\s+", title, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            company = parts[0].strip()
            rest = re.split(r"\s+in\s+", parts[1], maxsplit=1, flags=re.I)[0].strip()
            if company and rest:
                return rest, company

    if " | " in title:
        chunks = [chunk.strip() for chunk in title.split("|") if chunk.strip()]
        if len(chunks) >= 2:
            return chunks[0], chunks[1]

    if " - " in title:
        left, right = [part.strip() for part in title.split(" - ", 1)]
        if left and right and not looks_like_bulk_listing_title(left):
            return left, right

    if looks_like_bulk_listing_title(title):
        return None, None
    return title, None


def extract_from_html(html: str, url: str) -> dict[str, str]:
    decoded = unescape(html or "")
    meta: dict[str, str] = {}

    for pattern in (META_TAG_RE, META_TAG_RE_ALT):
        for match in pattern.finditer(decoded):
            key = match.group("key").lower()
            content = unescape(match.group("content")).strip()
            if content:
                meta.setdefault(key, content)

    for key in ("og:title", "twitter:title", "title"):
        if key in meta:
            parsed_title, parsed_company = parse_social_title(meta[key])
            if parsed_title:
                meta.setdefault("job_title", parsed_title)
            if parsed_company:
                meta.setdefault("company", parsed_company)
    for key in ("og:description", "twitter:description", "description"):
        if key in meta and "description" not in meta:
            meta["description"] = _clean_description(meta[key])

    title_match = TITLE_TAG_RE.search(decoded)
    if title_match and "job_title" not in meta:
        parsed_title, parsed_company = parse_social_title(title_match.group(1))
        if parsed_title:
            meta["job_title"] = parsed_title
        if parsed_company:
            meta.setdefault("company", parsed_company)

    for script_match in JSON_LD_RE.finditer(decoded):
        try:
            payload = json.loads(script_match.group(1))
        except json.JSONDecodeError:
            continue
        for node in _iter_json_nodes(payload):
            if not isinstance(node, dict):
                continue
            node_type = node.get("@type", "")
            types = node_type if isinstance(node_type, list) else [node_type]
            if not any(str(item).lower() == "jobposting" for item in types):
                continue
            if node.get("title") and "job_title" not in meta:
                meta["job_title"] = str(node["title"]).strip()
            org = node.get("hiringOrganization") or node.get("employer")
            if isinstance(org, dict) and org.get("name") and "company" not in meta:
                meta["company"] = str(org["name"]).strip()
            if node.get("description") and "description" not in meta:
                meta["description"] = _clean_description(str(node["description"]))
            if node.get("jobLocation") and "location" not in meta:
                meta["location"] = _location_from_json(node["jobLocation"])
            salary_hint = _salary_from_json_ld(node.get("baseSalary"))
            if salary_hint and "salary_hint" not in meta:
                meta["salary_hint"] = salary_hint

    if "job_title" not in meta:
        slug_title = _title_from_url_slug(url)
        if slug_title:
            meta["job_title"] = slug_title

    salary_context = _extract_salary_context(decoded)
    if salary_context:
        existing = meta.get("description", "")
        if existing:
            if salary_context.lower() not in existing.lower():
                meta["description"] = f"{existing} {salary_context}".strip()
        else:
            meta["description"] = salary_context

    if linkedin_applications_closed(decoded):
        meta["applications_closed"] = "true"

    if "salary_hint" not in meta:
        salary_hint = extract_posting_salary_hint(decoded)
        if salary_hint:
            meta["salary_hint"] = salary_hint

    return meta


def title_from_description(description: str) -> str | None:
    cleaned = unescape(description or "").strip()
    if not cleaned or looks_like_bulk_listing_title(cleaned):
        return None

    parts = re.split(r"\s*[·•|]\s*", cleaned)
    candidates: list[str] = []
    for part in parts:
        candidate = part.strip(" -–—")
        if not candidate:
            continue
        if looks_like_bulk_listing_title(candidate):
            continue
        if re.search(r"\d{2,}\+?", candidate) and "engineer" not in candidate.lower():
            continue
        if " at " in candidate.lower():
            candidate = re.split(r"\s+at\s+", candidate, maxsplit=1, flags=re.I)[0].strip()
        if len(candidate) >= 8:
            candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    return candidates[0]


def title_from_reasoning(reasoning: str) -> str | None:
    match = REASONING_TITLE_RE.search(reasoning or "")
    if not match:
        return None
    candidate = match.group(1).strip(" '\"")
    if candidate and not looks_like_bulk_listing_title(candidate):
        return candidate
    return None


def refine_job_metadata(job: JobPosting) -> JobPosting:
    title = (job.title or "").strip()
    company = (job.company or "").strip()
    parsed_title, parsed_company = parse_social_title(title)
    updates: dict[str, Any] = {}
    if parsed_title and parsed_title != title:
        updates["title"] = parsed_title
    if parsed_company and needs_company_enrichment(company):
        updates["company"] = parsed_company
    if updates:
        return job.model_copy(update=updates)
    return job


class JobTitleEnricher:
    def __init__(self, scraperapi_key: str | None = None) -> None:
        key = scraperapi_key if scraperapi_key is not None else os.getenv("SCRAPERAPI_API_KEY", "")
        self._scraperapi_key = key if key and "your_" not in key else ""

    async def enrich(self, client: httpx.AsyncClient, job: JobPosting) -> JobPosting:
        if not needs_job_enrichment(job):
            return job

        html = await self._fetch_html(client, job.url)
        extracted = extract_from_html(html, job.url) if html else {}

        title = extracted.get("job_title")
        company = extracted.get("company")
        description = extracted.get("description")
        location = extracted.get("location")
        salary_hint = extracted.get("salary_hint")

        if not title:
            title = title_from_description(job.description)
        if not title:
            title = _title_from_url_slug(job.url)

        if needs_company_enrichment(job.company) and company:
            new_company = company
        else:
            new_company = job.company

        updates: dict[str, Any] = {}
        if title and not needs_title_enrichment(title):
            updates["title"] = title
        if new_company and new_company != job.company:
            updates["company"] = new_company
        if description and _should_replace_description(job.description, description):
            updates["description"] = description[:8000]
        if not salary_hint and description:
            salary_hint = extract_posting_salary_hint(description)
        if not salary_hint and html:
            salary_hint = extract_posting_salary_hint(html)
        if salary_hint and not job.salary_hint:
            updates["salary_hint"] = salary_hint
        if location and (not job.location or job.location.lower() in {"spain", "italy", "italia"}):
            updates["location"] = location
        if extracted.get("applications_closed") == "true":
            updates["raw_metadata"] = {
                **job.raw_metadata,
                "applications_closed": True,
            }

        if not updates:
            return refine_job_metadata(job)

        return refine_job_metadata(job.model_copy(update=updates))

    async def enrich_for_accurate_match(
        self,
        client: httpx.AsyncClient,
        job: JobPosting,
    ) -> tuple[JobPosting, bool, bool]:
        """Fetch the job page and merge a fuller description (and related metadata).

        Returns (enriched_job, description_enriched, page_fetched).
        """
        previous_len = len((job.description or "").strip())
        html = await self._fetch_html(client, job.url)
        if not html:
            return refine_job_metadata(job), False, False

        extracted = extract_from_html(html, job.url)
        title = extracted.get("job_title")
        company = extracted.get("company")
        description = extracted.get("description")
        location = extracted.get("location")
        salary_hint = extracted.get("salary_hint")

        if not title:
            title = title_from_description(job.description)
        if not title:
            title = _title_from_url_slug(job.url)

        if needs_company_enrichment(job.company) and company:
            new_company = company
        else:
            new_company = job.company

        updates: dict[str, Any] = {}
        description_enriched = False
        if description and _should_replace_description_accurate(job.description, description):
            updates["description"] = description[:8000]
            description_enriched = True
        if title and not needs_title_enrichment(title):
            updates["title"] = title
        if new_company and new_company != job.company:
            updates["company"] = new_company
        if not salary_hint and description:
            salary_hint = extract_posting_salary_hint(description)
        if not salary_hint and html:
            salary_hint = extract_posting_salary_hint(html)
        if salary_hint and not job.salary_hint:
            updates["salary_hint"] = salary_hint
        if location and (not job.location or job.location.lower() in {"spain", "italy", "italia"}):
            updates["location"] = location
        if extracted.get("applications_closed") == "true":
            updates["raw_metadata"] = {
                **job.raw_metadata,
                "applications_closed": True,
            }

        if not updates:
            enriched = refine_job_metadata(job)
            new_len = len((enriched.description or "").strip())
            enriched_flag = new_len > previous_len and new_len >= DESCRIPTION_SHORT_THRESHOLD
            return enriched, enriched_flag, True

        enriched = refine_job_metadata(job.model_copy(update=updates))
        new_len = len((enriched.description or "").strip())
        if new_len > previous_len:
            description_enriched = True
        return enriched, description_enriched, True

    async def enrich_many(
        self,
        client: httpx.AsyncClient,
        jobs: list[JobPosting],
    ) -> list[JobPosting]:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_ENRICHMENTS)
        targets = [index for index, job in enumerate(jobs) if needs_job_enrichment(job)]
        if not targets:
            return jobs

        enriched = list(jobs)

        async def enrich_index(index: int) -> None:
            async with semaphore:
                try:
                    enriched[index] = await self.enrich(client, jobs[index])
                except Exception as exc:
                    logger.warning(
                        "[JobTitleEnricher] Failed to enrich %s: %s",
                        jobs[index].url,
                        exc,
                    )

        await asyncio.gather(*(enrich_index(index) for index in targets))
        return enriched

    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "it-IT,it;q=0.9,en;q=0.8"},
            )
            if response.status_code == 200 and len(response.text) > 500:
                return response.text
        except Exception as exc:
            logger.debug("[JobTitleEnricher] Direct fetch failed for %s: %s", url, exc)

        if not self._scraperapi_key:
            return None

        try:
            response = await client.get(
                SCRAPERAPI_URL,
                params={"api_key": self._scraperapi_key, "url": url, "country_code": "it"},
                timeout=60.0,
            )
            if response.status_code == 200 and len(response.text) > 500:
                return response.text
        except Exception as exc:
            logger.warning("[JobTitleEnricher] ScraperAPI fetch failed for %s: %s", url, exc)
        return None


def _iter_json_nodes(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        nodes: list[Any] = []
        for item in payload:
            nodes.extend(_iter_json_nodes(item))
        return nodes
    if isinstance(payload, dict):
        nodes = [payload]
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                nodes.extend(_iter_json_nodes(item))
        return nodes
    return []


def _location_from_json(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        address = value.get("address")
        if isinstance(address, dict):
            city = address.get("addressLocality") or address.get("addressRegion")
            country = address.get("addressCountry")
            bits = [str(part).strip() for part in (city, country) if part]
            if bits:
                return ", ".join(bits)
        if value.get("name"):
            return str(value["name"]).strip()
    if isinstance(value, list) and value:
        return _location_from_json(value[0])
    return ""


def _clean_description(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _has_salary_signal(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if SALARY_PATTERN_RE.search(cleaned):
        return True
    if re.search(r"(?:USD|\$)\s*\d", cleaned, flags=re.I):
        return True
    return bool(SALARY_HINT_WORD_RE.search(cleaned) and re.search(r"\d", cleaned))


def _extract_salary_context(text: str) -> str:
    cleaned = _clean_description(text)
    if not cleaned:
        return ""
    chunks = re.split(r"(?<=[\.;])\s+", cleaned)
    salary_chunks: list[str] = []
    for chunk in chunks:
        if _has_salary_signal(chunk):
            salary_chunks.append(chunk.strip())
    if salary_chunks:
        return " ".join(salary_chunks[:4])[:1500]
    match = SALARY_PATTERN_RE.search(cleaned)
    if not match:
        return ""
    start = max(match.start() - 220, 0)
    end = min(match.end() + 240, len(cleaned))
    return cleaned[start:end].strip()


def _should_replace_description(current: str, candidate: str) -> bool:
    current_clean = (current or "").strip()
    candidate_clean = (candidate or "").strip()
    if not candidate_clean:
        return False
    if not current_clean:
        return True
    if len(current_clean) < 300:
        return True
    if _has_salary_signal(candidate_clean) and not _has_salary_signal(current_clean):
        return True
    if looks_like_bulk_listing_title(current_clean):
        return True
    return False


def _should_replace_description_accurate(current: str, candidate: str) -> bool:
    current_clean = (current or "").strip()
    candidate_clean = (candidate or "").strip()
    if not candidate_clean:
        return False
    if not current_clean:
        return True
    if len(candidate_clean) > len(current_clean) * 1.15:
        return True
    if len(current_clean) < DESCRIPTION_SHORT_THRESHOLD and len(candidate_clean) > len(current_clean):
        return True
    if _has_salary_signal(candidate_clean) and not _has_salary_signal(current_clean):
        return True
    return False


def _title_from_url_slug(url: str) -> str | None:
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1]
    slug = re.sub(r"^jv_", "", slug, flags=re.I)
    slug = re.sub(r"-\d{6,}$", "", slug)
    slug = slug.replace("-", " ").strip()
    if slug and len(slug) > 3 and not slug.isdigit() and not looks_like_bulk_listing_title(slug):
        return slug.title()
    return None


def _salary_from_json_ld(value: Any) -> str | None:
    if isinstance(value, dict):
        currency = str(value.get("currency") or "EUR").upper()
        raw_value = value.get("value")
        if isinstance(raw_value, dict):
            low = raw_value.get("minValue") or raw_value.get("value")
            high = raw_value.get("maxValue") or raw_value.get("value")
            if low is not None and high is not None:
                low_int = int(float(low))
                high_int = int(float(high))
                if currency == "EUR":
                    return format_salary_range_eur(min(low_int, high_int), max(low_int, high_int))
                amount = max(low_int, high_int)
                return f"{amount:,}".replace(",", ".") + f" {currency}"
        if value.get("value") is not None:
            amount = int(float(value["value"]))
            if currency == "EUR":
                return format_salary_range_eur(amount, amount)
            return f"{amount:,}".replace(",", ".") + f" {currency}"
    return None
