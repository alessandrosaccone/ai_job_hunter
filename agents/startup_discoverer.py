from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.base import BaseAgent
from agents.job_prefilter import _experience_level_matches
from agents.keyword_expander import KeywordExpander
from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES_PATH = Path("config/target_companies.json")
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
MAX_KEYWORDS_PER_SCAN = 5
MAX_QUERIES_PER_KEYWORD = 3

JOB_BOARD_DOMAINS = (
    "jobs.lever.co",
    "boards.greenhouse.io",
    "myworkdayjobs.com",
    "myworkdaysite.com",
    "linkedin.com/jobs",
    "indeed.com",
    "stepstone.it",
    "stepstone.de",
    "infojobs.it",
)

QUERY_TEMPLATES = (
    "{keyword} {location}",
    "{keyword} {location} jobs",
    "{keyword} site:linkedin.com/jobs {location}",
    "{keyword} site:indeed.com {location}",
    "{keyword} site:stepstone.it {location}",
    "{keyword} (site:myworkdayjobs.com OR site:jobs.lever.co OR site:boards.greenhouse.io) {location}",
)


class StartupDiscoverer(BaseAgent):
    name = "StartupDiscoverer"

    def __init__(
        self,
        api_key: str | None = None,
        companies_path: Path | str = DEFAULT_COMPANIES_PATH,
        timeout_seconds: float = 30.0,
        keyword_expander: KeywordExpander | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")
        self.companies_path = Path(companies_path)
        self.timeout_seconds = timeout_seconds
        self.keyword_expander = keyword_expander or KeywordExpander()

    async def run(self, profile: UserProfile) -> list[JobPosting]:
        if not self.api_key or self.api_key == "your_serpapi_key_here":
            logger.warning("[StartupDiscoverer] SERPAPI_API_KEY missing. Skipping discovery.")
            return []

        keywords = await self.keyword_expander.expand(profile)
        keywords = keywords[:MAX_KEYWORDS_PER_SCAN]
        excluded_companies = self._load_excluded_companies(profile)
        jobs_by_url: dict[str, JobPosting] = {}

        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            search_tasks = []
            for keyword in keywords:
                for template in QUERY_TEMPLATES[:MAX_QUERIES_PER_KEYWORD]:
                    query = template.format(keyword=keyword, location=profile.location).strip()
                    search_tasks.append(self._search_all_engines(client, query))

            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for batch in results:
                if isinstance(batch, Exception):
                    logger.warning("[StartupDiscoverer] Search batch failed: %s", batch)
                    continue
                for item in batch:
                    job = self._normalize_result(item, profile)
                    if not job:
                        continue
                    if job.company.lower() in excluded_companies:
                        continue
                    if not _experience_level_matches(job, profile):
                        continue
                    jobs_by_url.setdefault(job.dedup_key, job)

        logger.info("[StartupDiscoverer] Collected %s unique jobs from SerpApi.", len(jobs_by_url))
        return list(jobs_by_url.values())

    def _load_excluded_companies(self, profile: UserProfile) -> set[str]:
        if not self.companies_path.exists():
            return set()
        with self.companies_path.open(encoding="utf-8") as handle:
            companies = json.load(handle)
        return {
            company.get("name", "").lower()
            for company in companies
            if company.get("name")
            and profile.career_field in company.get("fields", ["tech"])
        }

    async def _search_all_engines(
        self,
        client: httpx.AsyncClient,
        query: str,
    ) -> list[dict[str, Any]]:
        google_jobs = await self._safe_search(client, "google_jobs", query)
        google_web = await self._safe_search(client, "google", f"{query} jobs")
        return [*google_jobs, *google_web]

    async def _safe_search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
    ) -> list[dict[str, Any]]:
        try:
            return await self._search(client, engine, query)
        except Exception as exc:
            logger.warning("[StartupDiscoverer] SerpApi %s failed for '%s': %s", engine, query, exc)
            return []

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _search(
        self,
        client: httpx.AsyncClient,
        engine: str,
        query: str,
    ) -> list[dict[str, Any]]:
        params = {
            "engine": engine,
            "q": query,
            "api_key": self.api_key,
            "num": 10,
        }
        response = await client.get(SERPAPI_ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()

        if engine == "google_jobs":
            jobs = payload.get("jobs_results", [])
            return [{"source_engine": "google_jobs", **item} for item in jobs if isinstance(item, dict)]

        organic = payload.get("organic_results", [])
        parsed: list[dict[str, Any]] = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            link = item.get("link", "")
            if not self._is_job_board_url(link):
                continue
            parsed.append(
                {
                    "source_engine": "google",
                    "title": item.get("title", ""),
                    "company_name": self._infer_company_from_result(item),
                    "location": profile_location_hint(item),
                    "description": item.get("snippet", ""),
                    "apply_options": [{"link": link}],
                    "job_id": link,
                }
            )
        return parsed

    def _normalize_result(self, item: dict[str, Any], profile: UserProfile) -> JobPosting | None:
        title = (item.get("title") or "").strip()
        company = (item.get("company_name") or self._infer_company_from_result(item) or "").strip()
        if not title:
            return None

        url = self._extract_apply_url(item)
        if not url:
            return None

        if "lever.co" in url:
            source = "lever"
        elif "greenhouse.io" in url:
            source = "greenhouse"
        elif "myworkdayjobs.com" in url or "myworkdaysite.com" in url:
            source = "workday"
        else:
            source = "serpapi"

        description = item.get("description", "")
        detected_extensions = item.get("detected_extensions", {}) or {}
        salary_hint = detected_extensions.get("salary")
        work_mode_hint = detected_extensions.get("schedule") or detected_extensions.get("work_type")

        if not company:
            company = urlparse(url).netloc.replace("www.", "")

        return JobPosting(
            id=str(item.get("job_id") or url),
            title=title,
            company=company,
            url=url,
            source=source,  # type: ignore[arg-type]
            location=item.get("location", "") or profile.location,
            description=description,
            salary_hint=str(salary_hint) if salary_hint else None,
            work_mode_hint=str(work_mode_hint) if work_mode_hint else None,
            raw_metadata=item,
        )

    def _extract_apply_url(self, item: dict[str, Any]) -> str | None:
        apply_options = item.get("apply_options", []) or []
        for option in apply_options:
            link = option.get("link")
            if link and self._is_job_board_url(link):
                return link

        related_links = item.get("related_links", []) or []
        for link_item in related_links:
            link = link_item.get("link")
            if link and self._is_job_board_url(link):
                return link

        share_link = item.get("share_link")
        if share_link:
            return share_link

        link = item.get("link")
        if link and self._is_job_board_url(link):
            return link

        return None

    def _is_job_board_url(self, url: str) -> bool:
        lowered = url.lower()
        if any(domain in lowered for domain in JOB_BOARD_DOMAINS):
            return True
        return any(token in lowered for token in ("/jobs/", "/job/", "careers", "vacancy"))

    def _infer_company_from_result(self, item: dict[str, Any]) -> str:
        title = item.get("title", "")
        if " - " in title:
            return title.split(" - ")[-1].strip()
        if " | " in title:
            return title.split(" | ")[-1].strip()
        link = item.get("link", "")
        if not link:
            return ""
        host = urlparse(link).netloc.replace("www.", "")
        return host.split(".")[0].replace("-", " ").title()


def profile_location_hint(item: dict[str, Any]) -> str:
    snippet = item.get("snippet", "")
    title = item.get("title", "")
    match = re.search(r"\b(?:in|a|@)\s+([A-Za-zÀ-ÿ\s,-]{3,40})", f"{title} {snippet}")
    return match.group(1).strip() if match else ""
