from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.base import BaseAgent
from agents.company_config import company_storage_key, load_target_companies
from agents.job_prefilter import _experience_level_matches
from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES_PATH = Path("config/target_companies.json")
LEVER_US_BASE = "https://api.lever.co/v0/postings"
LEVER_EU_BASE = "https://api.eu.lever.co/v0/postings"
GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards"
WORKDAY_DEFAULT_LOCALE = "en-US"
WORKDAY_PAGE_SIZE = 20
WORKDAY_MAX_JOBS = 200


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

    def _load_companies(
        self,
        profile: UserProfile,
        *,
        discovered_companies: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        static_companies = load_target_companies(
            self.companies_path,
            career_field=profile.career_field,
        )

        merged: dict[str, dict[str, Any]] = {}
        for company in static_companies:
            if profile.career_field in company.get("fields", ["tech"]):
                merged[company_storage_key(company)] = company

        for company in discovered_companies or []:
            if profile.career_field not in company.get("fields", ["tech"]):
                continue
            merged.setdefault(company_storage_key(company), company)

        return list(merged.values())

    async def run(
        self,
        profile: UserProfile,
        *,
        discovered_companies: list[dict[str, Any]] | None = None,
        extra_companies: list[dict[str, Any]] | None = None,
    ) -> list[JobPosting]:
        companies = self._load_companies(profile, discovered_companies=discovered_companies)
        if extra_companies:
            known = {company_storage_key(company) for company in companies}
            for company in extra_companies:
                if profile.career_field not in company.get("fields", ["tech"]):
                    continue
                key = company_storage_key(company)
                if key not in known:
                    companies.append(company)
                    known.add(key)

        if not companies:
            logger.warning("[TargetHunter] No companies configured for field '%s'.", profile.career_field)
            return []

        return await self._fetch_for_companies(profile, companies)

    async def fetch_companies(
        self,
        profile: UserProfile,
        companies: list[dict[str, Any]],
    ) -> list[JobPosting]:
        if not companies:
            return []
        return await self._fetch_for_companies(profile, companies)

    async def _fetch_for_companies(
        self,
        profile: UserProfile,
        companies: list[dict[str, Any]],
    ) -> list[JobPosting]:
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

            if not slug and ats != "workday":
                return []

            try:
                if ats == "lever":
                    raw_jobs = await self._fetch_lever_jobs(client, slug, region)
                    jobs = [self._normalize_lever_job(item, name) for item in raw_jobs]
                elif ats == "greenhouse":
                    raw_jobs = await self._fetch_greenhouse_jobs(client, slug)
                    jobs = [self._normalize_greenhouse_job(item, name) for item in raw_jobs]
                elif ats == "workday":
                    raw_jobs = await self._fetch_workday_jobs(client, company)
                    jobs = [self._normalize_workday_job(item, name, company) for item in raw_jobs]
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

    async def probe_company_access(
        self,
        client: httpx.AsyncClient,
        company: dict[str, Any],
    ) -> bool:
        ats = str(company.get("ats", "")).lower()
        slug = str(company.get("slug", ""))
        region = str(company.get("region", "us")).lower()
        if not slug:
            return False

        if ats == "lever":
            bases = [LEVER_EU_BASE, LEVER_US_BASE] if region == "eu" else [LEVER_US_BASE, LEVER_EU_BASE]
            for base in bases:
                response = await client.get(f"{base}/{slug}?mode=json")
                if response.status_code == 200:
                    return True
            return False

        if ats == "greenhouse":
            response = await client.get(f"{GREENHOUSE_BASE}/{slug}/jobs?content=true")
            return response.status_code == 200

        if ats == "workday":
            jobs = await self._fetch_workday_jobs(client, company, limit=1)
            return bool(jobs)

        return False

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

    def _workday_config(self, company: dict[str, Any]) -> dict[str, str]:
        return {
            "host": str(company.get("host", "")).strip(),
            "tenant": str(company.get("tenant", "")).strip(),
            "site": str(company.get("slug", "")).strip(),
            "locale": str(company.get("locale", WORKDAY_DEFAULT_LOCALE)).strip() or WORKDAY_DEFAULT_LOCALE,
        }

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _fetch_workday_jobs(
        self,
        client: httpx.AsyncClient,
        company: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        cfg = self._workday_config(company)
        if not cfg["host"] or not cfg["tenant"] or not cfg["site"]:
            return []

        url = f"https://{cfg['host']}/wday/cxs/{cfg['tenant']}/{cfg['site']}/jobs"
        page_size = limit or WORKDAY_PAGE_SIZE
        max_jobs = limit or WORKDAY_MAX_JOBS
        collected: list[dict[str, Any]] = []
        offset = 0

        while offset < max_jobs:
            response = await client.post(
                url,
                json={
                    "appliedFacets": {},
                    "limit": min(page_size, max_jobs - offset),
                    "offset": offset,
                    "searchText": "",
                },
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code == 404:
                break
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("jobPostings", [])
            if not isinstance(batch, list) or not batch:
                break
            collected.extend(batch)
            total = int(payload.get("total", len(collected)))
            offset += len(batch)
            if limit is not None or offset >= total or len(batch) < page_size:
                break

        return collected

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

    def _normalize_workday_job(
        self,
        item: dict[str, Any],
        company_name: str,
        company: dict[str, Any],
    ) -> JobPosting:
        cfg = self._workday_config(company)
        external_path = str(item.get("externalPath", ""))
        absolute_url = f"https://{cfg['host']}/{cfg['locale']}/{cfg['site']}{external_path}"
        bullet_fields = item.get("bulletFields", [])
        posting_id = bullet_fields[0] if isinstance(bullet_fields, list) and bullet_fields else external_path
        location = str(item.get("locationsText", ""))
        description = "\n".join(
            part
            for part in (
                item.get("title", ""),
                location,
                item.get("postedOn", ""),
            )
            if part
        )

        return JobPosting(
            id=str(posting_id or absolute_url),
            title=item.get("title", "Unknown role"),
            company=company_name,
            url=absolute_url,
            source="workday",
            location=location,
            description=description,
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
