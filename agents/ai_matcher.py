from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from agents.base import BaseAgent
from models.job import JobPosting, MatchResult
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert career matching assistant.
Compare one job description against the full user profile and return ONLY valid json.

Hard rules:
1. If the user wants Remote and the job is clearly on-site only, set approved=false.
2. If the user wants Full-time in office and the job is remote-only, set approved=false.
3. If salary is mentioned and the maximum is below desired_salary_eur, heavily penalize or reject.
4. Obey free_text_preferences exactly (e.g. reject old-school consulting, prefer fintech startups).
5. Respect career_field and experience_level from the profile.
6. match_score must be between 0 and 10.
7. approved=true only for strong fits aligned with all constraints.

Respond with this json schema:
{
  "match_score": 8,
  "approved": true,
  "reasoning": "Detailed explanation in Italian or English."
}
"""


class AIMatchResponse(BaseModel):
    match_score: float = Field(ge=0, le=10)
    approved: bool
    reasoning: str


class AIMatcher(BaseAgent):
    name = "AIMatcher"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_concurrency: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_concurrency = max_concurrency
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
            response_payload = await self._call_model(job, profile)
            parsed = AIMatchResponse.model_validate(response_payload)
            return MatchResult(job=job, **parsed.model_dump())
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

    async def _call_model(self, job: JobPosting, profile: UserProfile) -> dict[str, Any]:
        user_prompt = self._build_user_prompt(job, profile)
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

    def _build_user_prompt(self, job: JobPosting, profile: UserProfile) -> str:
        profile_payload = profile.model_dump()
        job_payload = {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "url": job.url,
            "salary_hint": job.salary_hint,
            "work_mode_hint": job.work_mode_hint,
            "description": job.description[:6000],
        }
        return (
            "Evaluate this job against the user profile and return json only.\n\n"
            f"USER_PROFILE_JSON:\n{json.dumps(profile_payload, ensure_ascii=False, indent=2)}\n\n"
            f"JOB_JSON:\n{json.dumps(job_payload, ensure_ascii=False, indent=2)}"
        )
