from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fast geographic job filter.
Given user acceptable places (cities and/or countries) and job snippets, decide location fit.

Rules:
- Accept city name translations (Milano = Milan, München = Munich).
- If user lists only a country (e.g. Italy, Spain), match jobs located in that country.
- If user lists a city, match jobs in that city or its metro area.
- Fully remote jobs match if compatible with user work_mode Remote.
- When uncertain, prefer matches=false to save downstream cost.

Return ONLY json:
{
  "results": [
    {"id": "job-id", "matches": true, "reason": "short"}
  ]
}
"""

BATCH_SIZE = 8


class LocationMatchItem(BaseModel):
    id: str
    matches: bool
    reason: str = ""


class LocationMatchBatch(BaseModel):
    results: list[LocationMatchItem]


class LocationMatcher:
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
        places = profile.location_places()
        if not places or not jobs:
            return jobs, 0

        if not self.is_configured:
            logger.warning("[LocationMatcher] DEEPSEEK_API_KEY missing. Skipping location AI filter.")
            return jobs, 0

        matched_ids: set[str] = set()
        skipped = 0

        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start : start + self.batch_size]
            batch_results = await self._evaluate_batch(batch, profile, places)
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
        places: list[str],
    ) -> dict[str, bool]:
        payload = {
            "user_places": places,
            "user_work_mode": profile.work_mode,
            "jobs": [
                {
                    "id": job.id,
                    "title": job.title,
                    "location": job.location,
                    "work_mode_hint": job.work_mode_hint,
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
                        "content": f"Evaluate location fit and return json only:\n{json.dumps(payload, ensure_ascii=False)}",
                    },
                ],
            )
            content = completion.choices[0].message.content or "{}"
            parsed = LocationMatchBatch.model_validate(json.loads(content))
            results = {item.id: item.matches for item in parsed.results}
            if len(results) < len(jobs):
                for job, item in zip(jobs, parsed.results, strict=False):
                    results.setdefault(job.id, item.matches)
            return results
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            logger.warning("[LocationMatcher] Batch failed, keeping jobs in batch: %s", exc)
            return {job.id: True for job in jobs}
