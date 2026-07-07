from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from agents.base import BaseAgent
from agents.job_prefilter import _experience_level_matches
from agents.job_listing_expander import JobListingExpander
from agents.keyword_expander import KeywordExpander
from agents.search_providers.router import JobSearchRouter
from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES_PATH = Path("config/target_companies.json")
MAX_KEYWORDS_PER_SCAN = 5
MAX_QUERIES_PER_KEYWORD = 3

JOB_BOARD_DOMAINS = (
    "jobs.lever.co",
    "boards.greenhouse.io",
    "myworkdayjobs.com",
    "myworkdaysite.com",
    "linkedin.com/jobs",
    "glassdoor.com",
    "glassdoor.it",
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
        companies_path: Path | str = DEFAULT_COMPANIES_PATH,
        timeout_seconds: float = 45.0,
        keyword_expander: KeywordExpander | None = None,
        search_router: JobSearchRouter | None = None,
    ) -> None:
        self.companies_path = Path(companies_path)
        self.timeout_seconds = timeout_seconds
        self.keyword_expander = keyword_expander or KeywordExpander()
        self.search_router = search_router or JobSearchRouter()
        self.listing_expander = JobListingExpander(search_router=self.search_router)
        self._provider_hits: dict[str, int] = {}

    async def run(self, profile: UserProfile) -> list[JobPosting]:
        keywords = await self.keyword_expander.expand(profile)
        keywords = keywords[:MAX_KEYWORDS_PER_SCAN]
        excluded_companies = self._load_excluded_companies(profile)
        jobs_by_url: dict[str, JobPosting] = {}
        self._provider_hits = {}

        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            location_targets = profile.search_location_targets()
            search_tasks = []
            for keyword in keywords:
                for location_target in location_targets:
                    for template in QUERY_TEMPLATES[:MAX_QUERIES_PER_KEYWORD]:
                        query = template.format(keyword=keyword, location=location_target).strip()
                        search_tasks.append(self._search_all_engines(client, query, location_target))

            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for batch in results:
                if isinstance(batch, Exception):
                    logger.warning("[StartupDiscoverer] Search batch failed: %s", batch)
                    continue
                for item in batch:
                    jobs = await self._collect_jobs_from_result(client, item, profile)
                    for job in jobs:
                        if job.company.lower() in excluded_companies:
                            continue
                        if not _experience_level_matches(job, profile):
                            continue
                        jobs_by_url.setdefault(job.dedup_key, job)

        if self._provider_hits:
            logger.info("[StartupDiscoverer] Provider usage: %s", self._provider_hits)
        logger.info("[StartupDiscoverer] Collected %s unique jobs.", len(jobs_by_url))
        return list(jobs_by_url.values())

    async def _collect_jobs_from_result(
        self,
        client: httpx.AsyncClient,
        item: dict[str, Any],
        profile: UserProfile,
    ) -> list[JobPosting]:
        job = self._normalize_result(item, profile)
        if not job:
            return []

        if self.listing_expander.should_expand(job):
            expanded = await self.listing_expander.expand(client, job, profile)
            if expanded:
                return expanded
            return []
        return [job]

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
        location: str,
    ) -> list[dict[str, Any]]:
        google_jobs, _provider = await self.search_router.search(
            client,
            "google_jobs",
            query,
            location,
            job_board_filter=self._is_job_board_url,
        )
        self._count_provider(_provider, len(google_jobs))

        google_web, provider = await self.search_router.search(
            client,
            "google",
            f"{query} jobs",
            location,
            job_board_filter=self._is_job_board_url,
        )
        self._count_provider(provider, len(google_web))

        filtered_web = [item for item in google_web if self._is_job_board_url(self._item_link(item))]
        return [*google_jobs, *filtered_web]

    def _count_provider(self, provider: str | None, count: int) -> None:
        if provider and count:
            self._provider_hits[provider] = self._provider_hits.get(provider, 0) + count

    def _item_link(self, item: dict[str, Any]) -> str:
        link = item.get("link")
        if link:
            return str(link)
        apply_options = item.get("apply_options", [])
        if apply_options and isinstance(apply_options[0], dict):
            return str(apply_options[0].get("link", ""))
        return ""

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

