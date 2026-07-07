from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from agents.location_matcher import LocationMatcher
from agents.role_matcher import RoleMatcher
from models.job import JobPosting
from models.user_profile import ExperienceLevel, UserProfile

ProgressCallback = Callable[[str, dict[str, Any]], None]

LEVEL_KEYWORDS: dict[ExperienceLevel, tuple[str, ...]] = {
    "graduate": ("graduate", "new grad", "recent graduate", "laureat"),
    "internship": ("intern", "internship", "stage", "stagista", "tirocinio", "trainee"),
    "entry": ("entry level", "entry-level", "junior", "associate", "graduate"),
    "mid": ("mid level", "mid-level", "intermediate", " ii ", "engineer ii"),
    "senior": ("senior", "sr.", " sr ", "staff", "principal"),
    "manager": ("manager", "head of", "director", "lead", "team lead", "vp "),
}

REMOTE_HINTS = ("remote", "remoto", "da remoto", "work from home", "wfh", "anywhere")
ONSITE_HINTS = ("on-site", "onsite", "in office", "in sede", "hybrid")
SALARY_TOLERANCE_EUR = 4000


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _work_mode_matches(job: JobPosting, profile: UserProfile) -> bool:
    job_text = _normalize(f"{job.location} {job.title} {job.description} {job.work_mode_hint or ''}")

    if profile.work_mode == "Remote":
        if any(hint in job_text for hint in ONSITE_HINTS) and not any(
            hint in job_text for hint in REMOTE_HINTS
        ):
            return False
        return True

    if profile.work_mode == "Full-time in office":
        if any(hint in job_text for hint in REMOTE_HINTS) and not any(
            hint in job_text for hint in ONSITE_HINTS
        ):
            return False
        return True

    return True


def _detect_levels(text: str) -> set[ExperienceLevel]:
    normalized = _normalize(text)
    detected: set[ExperienceLevel] = set()
    for level, keywords in LEVEL_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            detected.add(level)
    return detected


def _experience_level_matches(job: JobPosting, profile: UserProfile) -> bool:
    job_text = _normalize(f"{job.title} {job.description}")
    detected = _detect_levels(job_text)
    if not detected:
        return True

    desired = profile.experience_level
    if desired in detected:
        return True

    compatible: dict[ExperienceLevel, set[ExperienceLevel]] = {
        "graduate": {"graduate", "entry", "internship"},
        "internship": {"internship", "entry", "graduate"},
        "entry": {"entry", "graduate", "internship", "mid"},
        "mid": {"mid", "entry", "senior"},
        "senior": {"senior", "mid", "manager"},
        "manager": {"manager", "senior"},
    }
    return bool(detected.intersection(compatible.get(desired, {desired})))


def _parse_amount(raw: str) -> int | None:
    cleaned = raw.lower().replace(",", "").replace(".", "").strip()
    if not cleaned:
        return None
    if cleaned.endswith("k"):
        digits = cleaned[:-1]
        if digits.isdigit():
            return int(digits) * 1000
    if cleaned.isdigit():
        value = int(cleaned)
        if value < 200:  # probabile "k" scritto male o valore mensile ambiguo
            return None
        return value
    return None


def _extract_salary_range(text: str) -> tuple[int, int] | None:
    if not text.strip():
        return None

    normalized = text.replace("€", " EUR ").replace("eur", " EUR ")
    range_match = re.search(
        r"(\d{1,3}[kK]|\d{2,3}[.,]?\d{3})\s*(?:-|to|–|—)\s*(\d{1,3}[kK]|\d{2,3}[.,]?\d{3})",
        normalized,
    )
    if range_match:
        low = _parse_amount(range_match.group(1))
        high = _parse_amount(range_match.group(2))
        if low and high:
            return min(low, high), max(low, high)

    amounts: list[int] = []
    for token in re.findall(r"\d{1,3}[kK]|\d{2,3}[.,]?\d{3}", normalized):
        parsed = _parse_amount(token)
        if parsed and parsed >= 15000:
            amounts.append(parsed)

    if not amounts:
        return None
    if len(amounts) == 1:
        return amounts[0], amounts[0]
    return min(amounts), max(amounts)


def _job_salary_range(job: JobPosting) -> tuple[int, int] | None:
    if job.salary_hint:
        parsed = _extract_salary_range(job.salary_hint)
        if parsed:
            return parsed
    return _extract_salary_range(job.description[:3000])


def _salary_matches(job: JobPosting, profile: UserProfile) -> bool:
    if not profile.fundamental_criteria.salary:
        return True
    if profile.desired_salary_eur is None:
        return True

    salary_range = _job_salary_range(job)
    if salary_range is None:
        return True

    _, max_salary = salary_range
    minimum_acceptable = profile.desired_salary_eur - SALARY_TOLERANCE_EUR
    return max_salary >= minimum_acceptable


def passes_non_location_criteria(job: JobPosting, profile: UserProfile) -> bool:
    criteria = profile.fundamental_criteria
    if criteria.salary and not _salary_matches(job, profile):
        return False
    if criteria.work_mode and not _work_mode_matches(job, profile):
        return False
    if criteria.experience_level and not _experience_level_matches(job, profile):
        return False
    return True

async def filter_jobs_for_ai(
    jobs: list[JobPosting],
    profile: UserProfile,
    location_matcher: LocationMatcher | None = None,
    role_matcher: RoleMatcher | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[JobPosting], int]:
    if not jobs:
        return [], 0

    loc_matcher = location_matcher or LocationMatcher()
    rol_matcher = role_matcher or RoleMatcher()
    candidates = jobs
    skipped = 0

    if profile.fundamental_criteria.location and profile.location_places():
        if on_progress:
            on_progress(
                "status",
                {"message": f"Verifica località AI su {len(candidates)} annunci..."},
            )
        candidates, location_skipped = await loc_matcher.filter_jobs(candidates, profile)
        skipped += location_skipped

    if profile.fundamental_criteria.target_role and profile.target_roles:
        if on_progress:
            on_progress(
                "status",
                {"message": f"Verifica ruolo AI su {len(candidates)} annunci..."},
            )
        candidates, role_skipped = await rol_matcher.filter_jobs(candidates, profile)
        skipped += role_skipped

    passed: list[JobPosting] = []
    for job in candidates:
        if passes_non_location_criteria(job, profile):
            passed.append(job)
        else:
            skipped += 1

    return passed, skipped
