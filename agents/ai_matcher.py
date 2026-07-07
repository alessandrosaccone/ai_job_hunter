from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from agents.base import BaseAgent
from agents.job_prefilter import _job_salary_range
from agents.salary_researcher import SalaryResearcher, SalaryResearchResult
from models.job import JobPosting, MatchResult
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert career matching assistant.
Compare one job description against the full user profile and return ONLY valid json.

Hard rules:
1. If the user wants Remote and the job is clearly on-site only, set approved=false.
2. If the user wants Full-time in office and the job is remote-only, set approved=false.
3. If salary IS explicitly mentioned and the maximum is clearly below desired_salary_eur (minus ~4000 EUR tolerance), heavily penalize or reject.
4. If salary is NOT indicated in the job (salary_indicated_in_posting=false):
   - ALWAYS state clearly in reasoning (Italian) that RAL was NOT written in the posting.
   - Use salary_web_research from the prompt when provided: it comes from real web search (Glassdoor, Levels.fyi, etc.).
   - Mention the researched range in reasoning if salary_web_research contains estimated_salary_eur.
   - If salary_web_research is missing or has no estimate, say web research found no reliable figures.
   - Apply a modest penalty to match_score (about 0.5-1.0 points) because missing salary transparency is a negative signal.
   - Do NOT reject the job solely for missing salary.
   - NEVER invent salary numbers yourself. Always leave estimated_salary_eur as null in your json.
5. If estimated range (when salary missing) seems well below desired_salary_eur, penalize score but still you may approve if otherwise strong.
6. Obey free_text_preferences exactly (e.g. reject old-school consulting, prefer fintech startups).
7. Respect career_field and experience_level from the profile.
8. match_score must be between 0 and 10.
9. approved=true only for strong fits aligned with all constraints (missing salary alone is never a reason to set approved=false).
10. Assess the likely recruitment/application channel and CV strategy:
   - application_channel: one of "human_recruiter", "ats", "mixed", "unknown"
   - human_recruiter: CV likely read by people → recommend human-friendly CV (clear narrative, readable layout, achievements, cover letter if appropriate)
   - ats: CV likely screened by ATS or rigid portal (Lever/Greenhouse/Workday, keyword forms) → recommend textual keyword-optimized CV matching the job description
   - mixed: ATS screening first then human review → recommend preparing both versions or a hybrid
   - unknown: insufficient signals
   Use signals from: job.source, apply URL/domain, company type/size cues, named recruiter contacts, email vs portal apply, tone of posting, keyword-heavy requirements lists.
   - cv_strategy: 1-2 concise sentences in Italian explaining which CV type to send and practical tips.

Respond with this json schema:
{
  "match_score": 8,
  "approved": true,
  "reasoning": "Detailed explanation in Italian.",
  "salary_indicated": true,
  "application_channel": "ats",
  "cv_strategy": "Suggerimento pratico in italiano su quale CV usare."
}
When salary is stated in the posting, salary_indicated=true.
When salary is missing, salary_indicated=false.
Never include estimated_salary_eur in your response — salary estimates are handled separately via web search.
"""


class AIMatchResponse(BaseModel):
    match_score: float = Field(ge=0, le=10)
    approved: bool
    reasoning: str
    salary_indicated: bool = True
    application_channel: str = "unknown"
    cv_strategy: str | None = None

    @field_validator("application_channel", mode="before")
    @classmethod
    def normalize_application_channel(cls, value: object) -> str:
        allowed = {"human_recruiter", "ats", "mixed", "unknown"}
        if isinstance(value, str) and value in allowed:
            return value
        return "unknown"


class AIMatcher(BaseAgent):
    name = "AIMatcher"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_concurrency: int = 3,
        salary_researcher: SalaryResearcher | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_concurrency = max_concurrency
        self.salary_researcher = salary_researcher or SalaryResearcher()
        self._client: AsyncOpenAI | None = None

    async def run(self, profile: UserProfile) -> list[JobPosting]:
        raise NotImplementedError("Use match(job, profile) for AI matching.")

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def match_jobs(
        self,
        jobs: list[JobPosting],
        profile: UserProfile,
    ) -> list[MatchResult]:
        if not jobs:
            return []

        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            logger.warning("[AIMatcher] DEEPSEEK_API_KEY missing. Skipping AI matching.")
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._match_with_semaphore(semaphore, job, profile) for job in jobs]
        return await asyncio.gather(*tasks)

    async def match(self, job: JobPosting, profile: UserProfile) -> MatchResult:
        try:
            salary_research: SalaryResearchResult | None = None
            if _job_salary_range(job) is None:
                salary_research = await self.salary_researcher.research(job, profile)

            response_payload = await self._call_model(job, profile, salary_research)
            parsed = AIMatchResponse.model_validate(response_payload)
            posting_has_salary = _job_salary_range(job) is not None
            result = MatchResult(
                job=job,
                match_score=parsed.match_score,
                approved=parsed.approved,
                reasoning=parsed.reasoning,
                salary_indicated=posting_has_salary,
                application_channel=parsed.application_channel,
                cv_strategy=parsed.cv_strategy,
                estimated_salary_eur=None,
            )
            if salary_research:
                result.salary_research_summary = salary_research.research_summary
                result.estimated_salary_eur = salary_research.estimated_salary_eur
                if not posting_has_salary:
                    result.salary_indicated = False
            return result
        except Exception as exc:
            logger.warning("[AIMatcher] Matching failed for %s: %s", job.title, exc)
            return MatchResult(
                job=job,
                match_score=0,
                approved=False,
                reasoning=f"AI matching failed: {exc}",
            )

    async def _match_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        job: JobPosting,
        profile: UserProfile,
    ) -> MatchResult:
        async with semaphore:
            return await self.match(job, profile)

    async def _call_model(
        self,
        job: JobPosting,
        profile: UserProfile,
        salary_research: SalaryResearchResult | None = None,
    ) -> dict[str, Any]:
        user_prompt = self._build_user_prompt(job, profile, salary_research)
        for attempt in range(2):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = completion.choices[0].message.content or "{}"
                return json.loads(content)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 1:
                    raise
                logger.warning("[AIMatcher] Invalid JSON on attempt %s: %s", attempt + 1, exc)
        return {}

    def _build_user_prompt(
        self,
        job: JobPosting,
        profile: UserProfile,
        salary_research: SalaryResearchResult | None = None,
    ) -> str:
        profile_payload = profile.model_dump()
        salary_range = _job_salary_range(job)
        job_payload = {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "url": job.url,
            "source": job.source,
            "salary_hint": job.salary_hint,
            "salary_indicated_in_posting": salary_range is not None,
            "parsed_salary_range_eur": (
                {"min": salary_range[0], "max": salary_range[1]} if salary_range else None
            ),
            "salary_web_research": (
                salary_research.model_dump() if salary_research else None
            ),
            "work_mode_hint": job.work_mode_hint,
            "description": job.description[:6000],
        }
        return (
            "Evaluate this job against the user profile and return json only.\n\n"
            f"USER_PROFILE_JSON:\n{json.dumps(profile_payload, ensure_ascii=False, indent=2)}\n\n"
            f"JOB_JSON:\n{json.dumps(job_payload, ensure_ascii=False, indent=2)}"
        )
