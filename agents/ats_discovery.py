from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from agents.company_config import company_storage_key
from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

SUPPORTED_ATS = frozenset({"lever", "greenhouse", "workday"})
WORKDAY_HOST_RE = re.compile(r"^([a-z0-9_-]+)\.wd\d+\.myworkdayjobs\.com$", re.I)
WORKDAY_SITE_HOST_RE = re.compile(r"^([a-z0-9_-]+)\.myworkdaysite\.com$", re.I)
LOCALE_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.I)


def company_key(ats: str, slug: str, *, tenant: str = "") -> str:
    if ats.lower() == "workday":
        return f"workday:{tenant.lower()}:{slug.lower()}"
    return f"{ats.lower()}:{slug.lower()}"


def parse_ats_from_url(url: str) -> dict[str, str] | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    parts = [part for part in parsed.path.split("/") if part]

    if "lever.co" in host and parts:
        return {
            "ats": "lever",
            "slug": parts[0],
            "region": "eu" if "eu.lever.co" in host else "us",
        }

    if "greenhouse.io" in host and parts:
        slug = parts[0]
        if slug in {"embed", "jobs"}:
            return None
        return {
            "ats": "greenhouse",
            "slug": slug,
            "region": "eu" if "eu.greenhouse.io" in host else "us",
        }

    tenant_match = WORKDAY_HOST_RE.match(host) or WORKDAY_SITE_HOST_RE.match(host)
    if tenant_match and parts:
        locale = "en-US"
        site_index = 0
        if LOCALE_RE.match(parts[0]):
            locale = parts[0]
            site_index = 1
        if site_index >= len(parts):
            return None
        site = parts[site_index]
        if site.lower() in {"job", "jobs", "apply"}:
            return None
        return {
            "ats": "workday",
            "slug": site,
            "tenant": tenant_match.group(1),
            "host": host,
            "locale": locale,
            "region": "eu",
        }

    return None


def _slug_to_name(slug: str) -> str:
    cleaned = slug.replace("-", " ").replace("_", " ").strip()
    return cleaned.title() if cleaned else slug


def extract_ats_candidates(jobs: list[JobPosting]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    for job in jobs:
        parsed = parse_ats_from_url(job.url)
        if not parsed:
            continue

        ats = parsed["ats"]
        slug = parsed["slug"]
        key = company_key(ats, slug, tenant=parsed.get("tenant", ""))
        if key in candidates:
            continue

        name = job.company.strip() or _slug_to_name(slug)
        candidate = {
            "name": name,
            "ats": ats,
            "slug": slug,
            "region": parsed.get("region", "us"),
            "fields": ["tech"],
            "source_url": job.url,
        }
        if ats == "workday":
            candidate["tenant"] = parsed["tenant"]
            candidate["host"] = parsed["host"]
            candidate["locale"] = parsed.get("locale", "en-US")
        candidates[key] = candidate

    return list(candidates.values())


def known_company_keys(companies: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for company in companies:
        ats = str(company.get("ats", "")).lower()
        slug = str(company.get("slug", "")).lower()
        if ats in SUPPORTED_ATS and slug:
            keys.add(company_storage_key(company))
    return keys


async def probe_ats_board(
    client: httpx.AsyncClient,
    company: dict[str, Any],
    *,
    probe_company_access,
) -> bool:
    try:
        return await probe_company_access(client, company)
    except Exception as exc:
        logger.warning(
            "[ATSDiscovery] Probe failed for %s (%s/%s): %s",
            company.get("name", company.get("slug")),
            company.get("ats"),
            company.get("slug"),
            exc,
        )
        return False


async def discover_and_verify_companies(
    jobs: list[JobPosting],
    profile: UserProfile,
    known_companies: list[dict[str, Any]],
    target_hunter: Any,
) -> list[dict[str, Any]]:
    known_keys = known_company_keys(known_companies)
    candidates = [
        candidate
        for candidate in extract_ats_candidates(jobs)
        if company_key(
            candidate["ats"],
            candidate["slug"],
            tenant=candidate.get("tenant", ""),
        )
        not in known_keys
    ]

    if not candidates:
        return []

    verified: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=target_hunter.timeout_seconds) as client:
        for candidate in candidates:
            if profile.career_field not in candidate.get("fields", ["tech"]):
                candidate["fields"] = [profile.career_field]

            accessible = await probe_ats_board(
                client,
                candidate,
                probe_company_access=target_hunter.probe_company_access,
            )
            if accessible:
                verified.append(candidate)
                logger.info(
                    "[ATSDiscovery] ATS accessibile: %s (%s/%s)",
                    candidate["name"],
                    candidate["ats"],
                    candidate["slug"],
                )
            else:
                logger.info(
                    "[ATSDiscovery] ATS non accessibile, skip: %s (%s/%s)",
                    candidate["name"],
                    candidate["ats"],
                    candidate["slug"],
                )

    return verified
