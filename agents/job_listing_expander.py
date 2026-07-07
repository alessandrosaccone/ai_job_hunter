from __future__ import annotations

import logging
import os
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from agents.job_prefilter import _extract_salary_range
from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

MAX_JOBS_PER_LISTING = 40
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

LINKEDIN_JOB_RE = re.compile(
    r"https?://(?:[\w.-]+\.)?linkedin\.com/jobs/view/(?:[^\"'?\s<>]*?)?(\d{6,})",
    re.I,
)
GLASSDOOR_JOB_RE = re.compile(
    r"https?://(?:[\w.-]+\.)?glassdoor\.[a-z.]+/job-listing/[^\"'\s<>]+",
    re.I,
)
INDEED_JOB_RE = re.compile(
    r"https?://(?:[\w.-]+\.)?indeed\.[a-z.]+/(?:viewjob\?[^\"'\s<>]*jk=[^\"'\s&<>]+|rc/clk\?[^\"'\s<>]*jk=[^\"'\s&<>]+)",
    re.I,
)
BULK_TITLE_RE = re.compile(
    r"\b(\d{2,})\s+(?:jobs?|offerte?|positions?|annunci|lavori|openings?)\b",
    re.I,
)
SCRAPERAPI_URL = "https://api.scraperapi.com"


def is_individual_job_url(url: str) -> bool:
    lowered = url.lower()
    if LINKEDIN_JOB_RE.search(url):
        return True
    if GLASSDOOR_JOB_RE.search(url):
        return True
    if INDEED_JOB_RE.search(url):
        return True
    if "jobs.lever.co" in lowered or "boards.greenhouse.io" in lowered:
        return True
    if "myworkdayjobs.com" in lowered and "/job/" in lowered:
        return True
    return False


def is_aggregate_job_listing_url(url: str) -> bool:
    if not url or is_individual_job_url(url):
        return False

    lowered = url.lower()
    parsed = urlparse(lowered)
    host = parsed.netloc
    path = parsed.path

    if "linkedin.com" in host:
        if "/jobs/search" in path or "/jobs/collections" in path:
            return True
        if "/jobs/" in path and "/jobs/view/" not in path:
            return True

    if "glassdoor." in host:
        if "/job-listing/" in path:
            return False
        if "/job/" in path or "/lavoro/" in path or "jobs.htm" in path or "job-listings" in path:
            return True
        if re.search(r"/job/[^/]+-jobs-srch", path):
            return True

    if "indeed." in host:
        if "/viewjob" in path or "jk=" in lowered:
            return False
        if "/jobs" in path or "/q-" in path or "q=" in parsed.query:
            return True

    return False


def looks_like_bulk_listing_title(title: str) -> bool:
    return bool(BULK_TITLE_RE.search(title))


class JobListingExpander:
    def __init__(self, search_router: Any | None = None) -> None:
        self.search_router = search_router
        self._scraperapi_key = os.getenv("SCRAPERAPI_API_KEY", "")

    async def expand(
        self,
        client: httpx.AsyncClient,
        job: JobPosting,
        profile: UserProfile,
    ) -> list[JobPosting]:
        if not self.should_expand(job):
            return [job]

        html = await self._fetch_html(client, job.url)
        urls = self._extract_job_urls(html or "", job.url)
        if not urls and self.search_router is not None:
            urls = await self._search_individual_jobs(client, job, profile)

        if not urls:
            logger.info("[JobListingExpander] No individual jobs extracted from %s", job.url)
            return []

        expanded: list[JobPosting] = []
        seen: set[str] = set()
        for index, url in enumerate(urls[:MAX_JOBS_PER_LISTING]):
            key = url.lower().rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                JobPosting(
                    id=f"{job.id}-expanded-{index}",
                    title=self._title_from_url(url, job.title),
                    company=job.company,
                    url=url,
                    source=job.source,
                    location=job.location or profile.location,
                    description=job.description[:1500] if job.description else "",
                    salary_hint=job.salary_hint,
                    work_mode_hint=job.work_mode_hint,
                    raw_metadata={"expanded_from": job.url, **job.raw_metadata},
                ),
            )

        logger.info(
            "[JobListingExpander] Expanded %s into %s individual jobs.",
            job.url,
            len(expanded),
        )
        return expanded

    def should_expand(self, job: JobPosting) -> bool:
        if is_aggregate_job_listing_url(job.url):
            return True
        if looks_like_bulk_listing_title(job.title):
            return True
        return False

    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "it-IT,it;q=0.9,en;q=0.8"},
            )
            if response.status_code == 200 and len(response.text) > 800:
                return response.text
        except Exception as exc:
            logger.debug("[JobListingExpander] Direct fetch failed for %s: %s", url, exc)

        if not self._scraperapi_key or self._scraperapi_key == "your_scraperapi_api_key_here":
            return None

        try:
            response = await client.get(
                SCRAPERAPI_URL,
                params={"api_key": self._scraperapi_key, "url": url, "country_code": "it"},
                timeout=60.0,
            )
            if response.status_code == 200 and len(response.text) > 800:
                return response.text
        except Exception as exc:
            logger.warning("[JobListingExpander] ScraperAPI fetch failed for %s: %s", url, exc)
        return None

    def _extract_job_urls(self, html: str, source_url: str) -> list[str]:
        decoded = unescape(html)
        candidates: list[str] = []

        for pattern in (LINKEDIN_JOB_RE, GLASSDOOR_JOB_RE, INDEED_JOB_RE):
            for match in pattern.finditer(decoded):
                link = match.group(0).split('"')[0].split("'")[0].rstrip("\\")
                candidates.append(self._normalize_extracted_url(link, pattern is LINKEDIN_JOB_RE))

        if not candidates:
            for href in re.findall(r'href=["\']([^"\']+)["\']', decoded, re.I):
                absolute = self._resolve_href(href, source_url)
                if absolute and is_individual_job_url(absolute):
                    candidates.append(absolute)

        deduped: list[str] = []
        seen: set[str] = set()
        for url in candidates:
            normalized = url.rstrip("/")
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _normalize_extracted_url(self, url: str, linkedin: bool) -> str:
        cleaned = unquote(url).strip()
        if linkedin:
            job_id_match = re.search(r"(\d{6,})", cleaned)
            if job_id_match:
                return f"https://www.linkedin.com/jobs/view/{job_id_match.group(1)}"
        return cleaned

    def _resolve_href(self, href: str, base_url: str) -> str | None:
        href = unquote(href.strip())
        if href.startswith("//"):
            return f"https:{href}"
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{href}"
        return None

    async def _search_individual_jobs(
        self,
        client: httpx.AsyncClient,
        job: JobPosting,
        profile: UserProfile,
    ) -> list[str]:
        if self.search_router is None:
            return []

        query_terms = self._query_terms_from_job(job)
        if not query_terms:
            return []

        location = job.location or profile.location or profile.search_location_targets()[0]
        site_filters = self._site_filters_for_url(job.url)
        urls: list[str] = []
        for site_filter in site_filters:
            query = f"{query_terms} {site_filter}".strip()
            results, _provider = await self.search_router.search(
                client,
                "google",
                query,
                location,
                job_board_filter=is_individual_job_url,
            )
            for item in results:
                link = str(item.get("link") or "")
                if is_individual_job_url(link):
                    urls.append(link)
            if urls:
                break
        return urls

    def _query_terms_from_job(self, job: JobPosting) -> str:
        parsed = urlparse(job.url)
        params = parse_qs(parsed.query)
        for key in ("keywords", "q", "keyword", "search"):
            if key in params and params[key]:
                return params[key][0].replace("+", " ").strip()

        title = job.title
        bulk_match = BULK_TITLE_RE.search(title)
        if bulk_match:
            title = BULK_TITLE_RE.sub("", title).strip(" -|,")
        if title and not looks_like_bulk_listing_title(title):
            return title[:120]
        return ""

    def _site_filters_for_url(self, url: str) -> tuple[str, ...]:
        lowered = url.lower()
        if "linkedin.com" in lowered:
            return ("site:linkedin.com/jobs/view",)
        if "glassdoor." in lowered:
            return ("site:glassdoor.com/job-listing", "site:glassdoor.it/job-listing")
        if "indeed." in lowered:
            return ("site:indeed.com/viewjob", "site:it.indeed.com/viewjob")
        return (
            "site:linkedin.com/jobs/view",
            "site:glassdoor.com/job-listing",
            "site:indeed.com/viewjob",
        )

    def _title_from_url(self, url: str, fallback: str) -> str:
        parsed = urlparse(url)
        slug = parsed.path.rstrip("/").split("/")[-1]
        slug = re.sub(r"^jv_", "", slug, flags=re.I)
        slug = re.sub(r"-\d{6,}$", "", slug)
        slug = slug.replace("-", " ").strip()
        if slug and len(slug) > 3 and not slug.isdigit():
            return slug.title()
        if looks_like_bulk_listing_title(fallback):
            return "Offerta di lavoro"
        return fallback or "Offerta di lavoro"


def match_salary_sort_key(result: Any) -> tuple[int, int]:
    """Higher salaries first; entries without RAL last."""
    amount = _match_salary_midpoint(result)
    if amount is None:
        return (1, 0)
    return (0, -amount)


def _match_salary_midpoint(result: Any) -> int | None:
    estimated = getattr(result, "estimated_salary_eur", None)
    if estimated:
        parsed = _extract_salary_range(str(estimated))
        if parsed:
            return (parsed[0] + parsed[1]) // 2

    salary_indicated = getattr(result, "salary_indicated", False)
    job = getattr(result, "job", None)
    if salary_indicated and job is not None and job.salary_hint:
        parsed = _extract_salary_range(str(job.salary_hint))
        if parsed:
            return (parsed[0] + parsed[1]) // 2

    if job is not None:
        parsed = _extract_salary_range(job.description[:3000])
        if parsed:
            return (parsed[0] + parsed[1]) // 2
    return None
