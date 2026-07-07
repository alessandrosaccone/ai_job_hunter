from __future__ import annotations

import json
import logging
import os

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fast job-role filter.
Given user target roles and job snippets, decide if the job role fits.

Rules:
- Match synonyms and translations (software engineer = sviluppatore software, account executive = sales).
- Match if the job aligns with ANY user target role.
- Consider career_field context.
- If the role is vague or unclear in the posting, set matches=true (let downstream AI decide).
- Reject only when the role is clearly unrelated to all target roles.

Return ONLY json:
{
  "results": [
    {"id": "job-id", "matches": true, "reason": "short"}
  ]
}
"""

BATCH_SIZE = 8


class RoleMatchItem(BaseModel):
    id: str
    matches: bool
    reason: str = ""


class RoleMatchBatch(BaseModel):
    results: list[RoleMatchItem]


class RoleMatcher:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.batch_size = batch_size
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_deepseek_api_key_here"

    async def filter_jobs(
        self,
        jobs: list[JobPosting],
        profile: UserProfile,
    ) -> tuple[list[JobPosting], int]:
        if not profile.target_roles or not jobs:
            return jobs, 0

        if not self.is_configured:
            logger.warning("[RoleMatcher] DEEPSEEK_API_KEY missing. Skipping role AI filter.")
            return jobs, 0

        matched_ids: set[str] = set()
        skipped = 0

        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start : start + self.batch_size]
            batch_results = await self._evaluate_batch(batch, profile)
            for job in batch:
                if batch_results.get(job.id, False):
                    matched_ids.add(job.id)
                else:
                    skipped += 1

        passed = [job for job in jobs if job.id in matched_ids]
        return passed, skipped

    async def _evaluate_batch(
        self,
        jobs: list[JobPosting],
        profile: UserProfile,
    ) -> dict[str, bool]:
        payload = {
            "career_field": profile.career_field,
            "target_roles": profile.target_roles,
            "experience_level": profile.experience_level,
            "jobs": [
                {
                    "id": job.id,
                    "title": job.title,
                    "description": job.description[:800],
                }
                for job in jobs
            ],
        }

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Evaluate role fit and return json only:\n"
                            f"{json.dumps(payload, ensure_ascii=False)}"
                        ),
                    },
                ],
            )
            content = completion.choices[0].message.content or "{}"
            parsed = RoleMatchBatch.model_validate(json.loads(content))
            results = {item.id: item.matches for item in parsed.results}
            if len(results) < len(jobs):
                for job, item in zip(jobs, parsed.results, strict=False):
                    results.setdefault(job.id, item.matches)
            return results
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            logger.warning("[RoleMatcher] Batch failed, keeping jobs in batch: %s", exc)
            return {job.id: True for job in jobs}
