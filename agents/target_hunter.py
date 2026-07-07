from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.base import BaseAgent
from agents.job_prefilter import _experience_level_matches
from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES_PATH = Path("config/target_companies.json")
LEVER_US_BASE = "https://api.lever.co/v0/postings"
LEVER_EU_BASE = "https://api.eu.lever.co/v0/postings"
GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards"


class TargetHunter(BaseAgent):
    name = "TargetHunter"

    def __init__(
        self,
        companies_path: Path | str = DEFAULT_COMPANIES_PATH,
        max_concurrency: int = 5,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.companies_path = Path(companies_path)
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds

    def _load_companies(self, profile: UserProfile) -> list[dict[str, Any]]:
        if not self.companies_path.exists():
            logger.warning("[TargetHunter] Companies file not found: %s", self.companies_path)
            return []
        with self.companies_path.open(encoding="utf-8") as handle:
            companies = json.load(handle)
        return [
            company
            for company in companies
            if profile.career_field in company.get("fields", ["tech"])
        ]

    async def run(self, profile: UserProfile) -> list[JobPosting]:
        companies = self._load_companies(profile)
        if not companies:
            logger.warning("[TargetHunter] No companies configured for field '%s'.", profile.career_field)
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            tasks = [
                self._fetch_company_jobs(client, semaphore, company, profile)
                for company in companies
            ]
            nested_results = await asyncio.gather(*tasks)
        return [job for batch in nested_results for job in batch]

    async def _fetch_company_jobs(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        company: dict[str, Any],
        profile: UserProfile,
    ) -> list[JobPosting]:
        async with semaphore:
            ats = company.get("ats", "").lower()
            slug = company.get("slug", "")
            name = company.get("name", slug)
            region = company.get("region", "us").lower()

            if not slug:
                return []

            try:
                if ats == "lever":
                    raw_jobs = await self._fetch_lever_jobs(client, slug, region)
                    jobs = [self._normalize_lever_job(item, name) for item in raw_jobs]
                elif ats == "greenhouse":
                    raw_jobs = await self._fetch_greenhouse_jobs(client, slug)
                    jobs = [self._normalize_greenhouse_job(item, name) for item in raw_jobs]
                else:
                    logger.warning("[TargetHunter] Unsupported ATS '%s' for %s", ats, name)
                    return []
            except Exception as exc:
                logger.warning("[TargetHunter] Failed to fetch jobs for %s: %s", name, exc)
                return []

            filtered = [
                job
                for job in jobs
                if self._matches_keywords(job, profile.target_roles)
                and _experience_level_matches(job, profile)
            ]
            return filtered

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _get_json(self, client: httpx.AsyncClient, url: str) -> Any:
        response = await client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def _fetch_lever_jobs(
        self,
        client: httpx.AsyncClient,
        slug: str,
        region: str,
    ) -> list[dict[str, Any]]:
        bases = [LEVER_EU_BASE, LEVER_US_BASE] if region == "eu" else [LEVER_US_BASE, LEVER_EU_BASE]
        for base in bases:
            url = f"{base}/{slug}?mode=json"
            payload = await self._get_json(client, url)
            if payload is not None:
                return payload if isinstance(payload, list) else []
        return []

    async def _fetch_greenhouse_jobs(self, client: httpx.AsyncClient, slug: str) -> list[dict[str, Any]]:
        url = f"{GREENHOUSE_BASE}/{slug}/jobs?content=true"
        payload = await self._get_json(client, url)
        if not payload:
            return []
        jobs = payload.get("jobs", [])
        return jobs if isinstance(jobs, list) else []

    def _matches_keywords(self, job: JobPosting, keywords: list[str]) -> bool:
        haystack = f"{job.title} {job.description}".lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    def _normalize_lever_job(self, item: dict[str, Any], company_name: str) -> JobPosting:
        categories = item.get("categories", {}) or {}
        location = categories.get("location") or categories.get("allLocations", "")
        if isinstance(location, list):
            location = ", ".join(location)

        hosted_url = item.get("hostedUrl") or item.get("applyUrl") or ""
        posting_id = str(item.get("id", hosted_url))

        return JobPosting(
            id=posting_id,
            title=item.get("text", "Unknown role"),
            company=company_name,
            url=hosted_url or f"https://jobs.lever.co/{company_name.lower()}/{posting_id}",
            source="lever",
            location=str(location),
            description=self._strip_html(item.get("descriptionPlain") or item.get("description", "")),
            salary_hint=self._extract_salary_hint(item),
            work_mode_hint=categories.get("commitment"),
            raw_metadata=item,
        )

    def _normalize_greenhouse_job(self, item: dict[str, Any], company_name: str) -> JobPosting:
        location = item.get("location", {}).get("name", "")
        absolute_url = item.get("absolute_url", "")

        return JobPosting(
            id=str(item.get("id", absolute_url)),
            title=item.get("title", "Unknown role"),
            company=company_name,
            url=absolute_url,
            source="greenhouse",
            location=location,
            description=self._strip_html(item.get("content", "")),
            salary_hint=None,
            work_mode_hint=None,
            raw_metadata=item,
        )

    def _strip_html(self, value: str) -> str:
        return (
            value.replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("<br />", "\n")
            .replace("<li>", "- ")
            .replace("</li>", "\n")
            .replace("<p>", "")
            .replace("</p>", "\n")
        )

    def _extract_salary_hint(self, item: dict[str, Any]) -> str | None:
        salary_range = item.get("salaryRange") or item.get("salary_range")
        if isinstance(salary_range, dict):
            minimum = salary_range.get("min")
            maximum = salary_range.get("max")
            currency = salary_range.get("currency", "EUR")
            if minimum or maximum:
                return f"{minimum}-{maximum} {currency}"
        return None
