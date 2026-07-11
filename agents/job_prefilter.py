from __future__ import annotations

import re
from collections.abc import Callable
from html import unescape
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
LINKEDIN_CLOSED_PATTERNS = (
    "non accetta piu candidature",
    "non accetta ulteriori candidature",
    "questa offerta di lavoro non accetta",
    "offerta di lavoro non accetta",
    "no longer accepting applications",
    "no longer accept applications",
    "not accepting applications",
    "applications are no longer being accepted",
    "this job is no longer accepting applications",
    "job is no longer accepting applications",
    "applications closed",
)
SALARY_TOLERANCE_EUR = 4000
SALARY_KEYWORD_RE = re.compile(
    r"\b(?:salary|compensation|stipendio|retribuzione|ral|pay range|gross annual)\b",
    re.I,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _normalize_closed_text(text: str) -> str:
    normalized = _normalize(text)
    for src, dst in (
        ("più", "piu"),
        ("à", "a"),
        ("è", "e"),
        ("é", "e"),
        ("ò", "o"),
        ("ù", "u"),
    ):
        normalized = normalized.replace(src, dst)
    return normalized


def _is_linkedin_job_url(url: str) -> bool:
    return "linkedin.com" in (url or "").lower()


def linkedin_applications_closed(text: str) -> bool:
    normalized = _normalize_closed_text(text)
    return any(pattern in normalized for pattern in LINKEDIN_CLOSED_PATTERNS)


def job_posting_closed(job: JobPosting) -> bool:
    if job.raw_metadata.get("applications_closed"):
        return True
    if not _is_linkedin_job_url(job.url):
        return False
    return linkedin_applications_closed(f"{job.title} {job.description}")


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

    allowed = profile.allowed_experience_levels()
    if profile.experience_level_rule.mode == "exact":
        return detected.issubset(allowed)

    return bool(detected.intersection(allowed))


def _parse_amount(raw: str) -> int | None:
    cleaned = raw.lower().replace(",", "").replace(".", "").replace(" ", "").strip()
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
    currency = r"(?:EUR|USD|\$|€)?"
    amount = r"(\d{1,3}\s*[kK]|\d{2,3}(?:[.,]\d{3})?)"
    range_match = re.search(
        rf"{currency}\s*{amount}\s*(?:-|to|–|—)\s*{currency}\s*{amount}",
        normalized,
        flags=re.I,
    )
    if range_match:
        low = _parse_amount(range_match.group(1))
        high = _parse_amount(range_match.group(2))
        if low and high and _is_plausible_annual_salary(low, high, text=normalized):
            return min(low, high), max(low, high)

    usd_match = re.search(
        r"(?:USD|\$)\s*(\d{1,3}[kK]|\d{2,3}(?:[.,]\d{3})?)",
        normalized,
        flags=re.I,
    )
    if usd_match:
        parsed = _parse_amount(usd_match.group(1))
        if parsed:
            return parsed, parsed

    up_to_match = re.search(
        r"(?:up to|fino a|jusqu'à)\s*(?:USD|\$|€|EUR)?\s*(\d{1,3}\s*[kK]|\d{2,3}(?:[.,]\d{3})?)",
        normalized,
        flags=re.I,
    )
    if up_to_match:
        parsed = _parse_amount(up_to_match.group(1))
        if parsed:
            return parsed, parsed

    amounts: list[int] = []
    for token in re.findall(r"\d{1,3}\s*[kK]|\d{2,3}[.,]\d{3}", normalized):
        parsed = _parse_amount(token)
        if parsed and 10_000 <= parsed <= 350_000:
            amounts.append(parsed)

    if not amounts:
        return None
    if len(amounts) == 1:
        return amounts[0], amounts[0]
    low, high = min(amounts), max(amounts)
    if _is_plausible_annual_salary(low, high, text=normalized):
        return low, high
    return None


def _is_plausible_annual_salary(low: int, high: int, *, text: str = "", is_usd: bool = False) -> bool:
    max_allowed = 600_000 if is_usd else 350_000
    if low < 10_000 or high < 10_000:
        return False
    if low > max_allowed or high > max_allowed:
        return False
    if high > low * 4:
        return False
    return True


def extract_posting_salary_hint(text: str) -> str | None:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = re.sub(r"\s+", " ", unescape(cleaned)).strip()
    if not cleaned:
        return None

    patterns = (
        r"(?:RAL|salary|stipendio|retribuzione|compensation|gross annual|gross salary)[^\.]{0,180}?(?:EUR|USD|\$|€)?\s*(\d{1,3}[.,]?\d{3}|\d{2,3}\s*[kK])\s*(?:-|to|–|—)\s*(?:EUR|USD|\$|€)?\s*(\d{1,3}[.,]?\d{3}|\d{2,3}\s*[kK])",
        r"(?:EUR|USD|\$|€)?\s*(\d{1,3}[.,]?\d{3}|\d{2,3}\s*[kK])\s*(?:-|to|–|—)\s*(?:EUR|USD|\$|€)?\s*(\d{1,3}[.,]?\d{3}|\d{2,3}\s*[kK])[^\.]{0,80}(?:EUR|€|RAL|annual|annuo|lordo|gross)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.I)
        if not match:
            continue
        low = _parse_amount(match.group(1))
        high = _parse_amount(match.group(2))
        is_usd = bool(re.search(r"(?:USD|\$)", match.group(0), flags=re.I))
        if low and high and _is_plausible_annual_salary(low, high, is_usd=is_usd):
            if is_usd:
                return f"{max(low, high):,}".replace(",", ".") + " USD"
            return format_salary_range_eur(min(low, high), max(low, high))

    for chunk in re.split(r"(?<=[\.;])\s+", cleaned):
        if not SALARY_KEYWORD_RE.search(chunk):
            continue
        parsed = _extract_salary_range(chunk)
        if not parsed:
            continue
        is_usd = bool(re.search(r"(?:USD|\$)", chunk, flags=re.I))
        if _is_plausible_annual_salary(parsed[0], parsed[1], is_usd=is_usd):
            if is_usd:
                return f"{parsed[1]:,}".replace(",", ".") + " USD"
            return format_salary_range_eur(parsed[0], parsed[1])
    return None


def format_salary_range_eur(low: int, high: int) -> str:
    def fmt(value: int) -> str:
        return f"{value:,}".replace(",", ".")

    if low == high:
        return f"{fmt(low)} EUR"
    return f"{fmt(low)}-{fmt(high)} EUR"


def job_posting_salary_display(job: JobPosting) -> str | None:
    if job.salary_hint and job.salary_hint.strip():
        return job.salary_hint.strip()
    hint = extract_posting_salary_hint(job.description[:8000])
    if hint:
        return hint
    parsed = _job_salary_range(job)
    if not parsed:
        return None
    text = f"{job.description[:4000]} {job.salary_hint or ''}"
    if re.search(r"(?:USD|\$)", text, flags=re.I):
        amount = parsed[1]
        return f"{amount:,}".replace(",", ".") + " USD"
    return format_salary_range_eur(parsed[0], parsed[1])


def _job_salary_range(job: JobPosting) -> tuple[int, int] | None:
    if job.salary_hint:
        parsed = _extract_salary_range(job.salary_hint)
        if parsed:
            return parsed
    hint = extract_posting_salary_hint(job.description[:8000])
    if hint:
        parsed = _extract_salary_range(hint)
        if parsed:
            return parsed
    return None


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
    skipped = 0
    candidates: list[JobPosting] = []
    for job in jobs:
        if job_posting_closed(job):
            skipped += 1
            continue
        candidates.append(job)
    if not candidates:
        return [], skipped

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
